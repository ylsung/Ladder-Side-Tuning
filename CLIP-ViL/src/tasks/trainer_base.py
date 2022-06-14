import torch.nn as nn

from lxrt.adapters import AdapterController
from lxrt.modeling import VisualFeatEncoder
from clip.model import VisualAdapter

class TrainerBase:
    def initialize_side_network(self, pruned_state_dict):
        # Intialize side transformer
        backbone_state_dict = self.model.state_dict()
        for n, p in self.model.named_parameters():
            if "side_visn_fc" in n:
                infer_n = n.split(".")
                infer_n[4] = "visn_fc"
                infer_n = ".".join(infer_n)
                print(n, infer_n)
                state = backbone_state_dict[infer_n]
                p.data.copy_(state)

            if "side_block_l" in n:
                infer_n = n.split(".")
                infer_n[4] = "layer"
                infer_n = ".".join(infer_n)
                print(n, infer_n)
                state = pruned_state_dict[infer_n]
                p.data.copy_(state)

            if "side_block_r" in n:
                infer_n = n.split(".")
                infer_n[4] = "r_layers"
                infer_n = ".".join(infer_n)
                print(n, infer_n)
                state = pruned_state_dict[infer_n]
                p.data.copy_(state)

            if "side_block_x" in n:
                infer_n = n.split(".")
                infer_n[4] = "x_layers"
                infer_n = ".".join(infer_n)
                print(n, infer_n)
                state = pruned_state_dict[infer_n]
                p.data.copy_(state)

    def print_trainable_params_percentage(self, model):
        # if "bart-base" in self.args.backbone:
        #     orig_param_size = 139420416
        # elif "t5-base" in self.args.backbone:
        #     orig_param_size = 222903552
        # else:
        #     print(f"Don't know the parameters number of this {self.args.backbone}")
        #     orig_param_size = -1

        orig_param_size = sum(p.numel() for p in model.parameters())

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        trainable_size = count_parameters(model)

        percentage = trainable_size / orig_param_size * 100

        print(f"Trainable param percentage: {percentage:.2f}%")

        print(trainable_size)

        return percentage

    def freeze_whole_model(self):
        for n, p in self.model.named_parameters():
            p.requires_grad = False

    def unfreeze_parameters(self):    
        targets = ["logit_fc"]
        # unfreeze the parameters in targets anyway
        for n, p in self.model.named_parameters():
            if any(t in n for t in targets):
                p.requires_grad = True
                print(f"{n} is trainable...")

        if self.args.use_lora:
            for n, p in self.model.named_parameters():
                if "lora" in n:
                    p.requires_grad = True
                    print(f"{n} is trainable...")

                if "bias" in n and "visual_model" not in n:
                    p.requires_grad = True
                    print(f"{n} is trainable...")
   
        for name, sub_module in self.model.named_modules():
            # if self.args.unfreeze_vis_encoder:
            #     if isinstance(sub_module, (CLIPResNetEncoder)):
            #         print(f"{name} is trainable...")
            #         # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
            #         for param_name, param in sub_module.named_parameters():
            #             param.requires_grad = True


            # if self.args.use_vis_adapter:
            #     if isinstance(sub_module, (VisualAdapter)):
            #         print(f"{name} is trainable...")
            #         # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
            #         for param_name, param in sub_module.named_parameters():
            #             param.requires_grad = True

            # train the visual projection layer if not using side transformers
            if not self.args.freeze_visual_projection and not self.args.use_side_transformers:
                if isinstance(sub_module, VisualFeatEncoder):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_adapter:
                if isinstance(sub_module, nn.LayerNorm):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

                if isinstance(sub_module, (AdapterController)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_vis_adapter:
                if isinstance(sub_module, nn.BatchNorm2d):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

                if isinstance(sub_module, (VisualAdapter)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_side_transformers:
                if "side" in name:
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_bn:
                if isinstance(sub_module, nn.BatchNorm2d):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True