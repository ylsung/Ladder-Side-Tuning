# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv
import os
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch
from tqdm import tqdm
from .vision_helpers import to_image_list
import json

from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, ColorJitter

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000


class ImageReader():
    def __init__(self, args):
        if args.vqa_style_transform:
            from src.tasks.vision_helpers import PadToGivenSize

            from torchvision.transforms import Resize

            min_size = args.image_size_min
            max_size = args.image_size_max
            flip_horizontal_prob = 0.0
            flip_vertical_prob = 0.0
            brightness = 0.0
            contrast = 0.0
            saturation = 0.0
            hue = 0.0
            color_jitter = ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
            self.transform = Compose(
                [
                    color_jitter,
                    Resize((min_size, max_size)),
                    lambda image: image.convert("RGB"),
                    PadToGivenSize(min_size, max_size) if args.add_zero_padding else lambda image:image,
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
    def __getitem__(self, img_id):
        image_file_name = "data/nlvr/images/{}.png".format(img_id)
        image = Image.open(image_file_name)
        feats = self.transform(image)  # Raw image as a tensor: 3 x 224 x 224
        return feats


class NLVRDataset:
    def __init__(self, splits='train'):
        super().__init__()

        self.splits = splits.split(',')

        self.data = []
        for split in self.splits:
            self.data.extend(
                json.load(open(f'data/nlvr/{split}.json')))

        # List to dict (for evaluation and others)
        self.id2datum = {}
        self.identifier2uid = {}
        for datum in self.data:
            self.id2datum[datum['uid']] = datum

            self.identifier2uid[datum['identifier']] = datum['uid']

        # Answers
        self.ans2label = {"false": 0, "true": 1}
        self.label2ans = {v: k for k, v in self.ans2label.items()}
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)


    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class NLVRTorchDataset(Dataset):
    def __init__(self, dataset: NLVRDataset):
        super().__init__()

        ### Control options
        self.input_raw_images = args.input_raw_images
        self.vqa_style_transform = args.vqa_style_transform
        self.use_h5_file = args.use_h5_file
        self.image_size_min = args.image_size_min
        self.image_size_max = args.image_size_max
        self.dynamic_padding = args.dynamic_padding
        self.add_zero_padding= args.add_zero_padding
        ###

        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # calculate image size
        if not os.path.exists("data/nlvr/width_heigths.json"):
            w_h_records = {}
            root_name = "data/nlvr/images/"
            failed_counter = 0
            for root, dirs, files in os.walk(root_name, topdown=False):
                for file in tqdm(files):
                    try:
                        image = np.asarray(Image.open(os.path.join(root_name, file)))
                        w = image.shape[0]
                        h = image.shape[1]
                        w_h_records[os.path.join(root_name, file)] = (w, h)
                    except:
                        failed_counter += 1
            print("Skipped {} files".format(failed_counter))

            with open("data/nlvr/width_heigths.json", "w") as f:
                json.dump(w_h_records, f)
            assert(0)
        else:
            with open("data/nlvr/width_heigths.json") as f:
                self.w_h_records = json.load(f)

        if self.input_raw_images:
            from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, ColorJitter
            #device = "cuda" if torch.cuda.is_available() else "cpu"
            #model, preprocess = clip.load("ViT-B/32", device=device)        
            self.image_reader = ImageReader(args)
            self.data = self.raw_dataset.data

            if topk is not None:
                self.data = self.data[:topk]

        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def get_height_and_width(self, index):
        datum = self.data[index]
        img_id = datum["img0"]
        
        image_file_name = "data/nlvr/images/{}.png".format(img_id)
        w, h = self.w_h_records[image_file_name]
        return h, w

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        if self.input_raw_images:
            return self.getitem_clip(item)
        else:
            return self.getitem_butd(item)
    
    def getitem_clip(self, item):
        datum = self.data[item]

        img_id = datum['uid']
        ques_id = img_id
        ques = datum['sent']
        # img_id : COCO_val2014_000000393267
        # actual image file name: data/mscoco/val2014/COCO_val2014_000000393267.jpg
        
        images = []
        for key in ['img0', 'img1']:
            img_id = datum[key]

            image = self.image_reader[img_id]

            images.append(image)

        feats = torch.cat(images, dim=-1)
        
        #feats = self.transform(Image.open(image_file_name))  # Raw image as a tensor: 3 x 224 x 224
        #if feats.size()[1] > feats.size()[2]:
        #    feats = to_image_list([feats], max_size=(3, 1000, 600))[0]
        #else:
        #    feats = to_image_list([feats], max_size=(3, 600, 1000))[0]

        boxes = torch.Tensor([0.0]) # Just being lazy

        # Provide label (target)
        
        if 'label' in datum:
            label = datum['label']
            return ques_id, feats, boxes, ques, torch.LongTensor([label])
        else:
            return ques_id, feats, boxes, ques
        
    def collate_fn(self, batch):
        if len(batch[0]) == 5:
            ques_id, feats, boxes, ques, target = zip(*batch)
        else:
            ques_id, feats, boxes, ques = zip(*batch)
        if self.input_raw_images and self.vqa_style_transform:
            if self.dynamic_padding:
                feats = to_image_list(feats)
            else:
                if feats[0].size(1) <= feats[0].size(2):
                    feats = to_image_list(feats, max_size=(3, self.image_size_min, self.image_size_max))
                else:
                    feats = to_image_list(feats, max_size=(3, self.image_size_max, self.image_size_min))
        else:
            feats = torch.stack(feats, dim=0)
        boxes = torch.stack(boxes, dim=0)
        #ques_id = torch.LongTensor(ques_id)
        if len(batch[0]) == 5:
            target = torch.stack(target, dim=0)
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques

    def getitem_butd(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        if self.use_h5_file:
            image_index, obj_num, feats, boxes, img_h, img_w, obj_labels, obj_confs, attr_labels, attr_confs = self.image_feature_dataset[img_id]
        else:
            # Get image info
            img_info = self.imgid2img[img_id]
            obj_num = img_info['num_boxes']
            feats = img_info['features'].copy()
            boxes = img_info['boxes'].copy()
            assert obj_num == len(boxes) == len(feats)

            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        boxes = torch.from_numpy(boxes)
        feats = torch.from_numpy(feats)
        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class NLVREvaluator:
    def __init__(self, dataset: NLVRDataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans == label:
                score += 1
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


