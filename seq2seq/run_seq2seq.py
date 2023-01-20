# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import functools
import logging
import torch 
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ["WANDB_DISABLED"] = "true"
import sys
import subprocess
from typing import Optional, List

from datasets import load_dataset, load_metric, concatenate_datasets
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from seq2seq.utils import get_adapter_config
from seq2seq.data import AutoTask
from seq2seq.data import TaskDataCollatorForSeq2Seq, DataCollatorForT5MLM
from seq2seq.third_party.trainers import Seq2SeqTrainer
from training_args import AdapterTrainingArguments
from seq2seq.utils import modify_model_after_init, save_training_config 
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments 
from seq2seq.third_party.models import T5Config, T5ForConditionalGeneration
from seq2seq.data import AutoPostProcessor

logger = logging.getLogger(__name__)

def run_command(command):
    output = subprocess.getoutput(command)
    return output


TASK_TO_METRICS = {"mrpc": ["accuracy", "f1"],
                  "cola": ['matthews_correlation'],
                  "stsb": ['pearson', 'spearmanr'],
                  'sst2': ['accuracy'],
                  "mnli": ["accuracy"],
                  "mnli_mismatched": ["accuracy"],
                  "mnli_matched": ["accuracy"],
                  "qnli": ["accuracy"],
                  "rte": ["accuracy"],
                  "wnli": ["accuracy"],
                  "qqp": ["accuracy", "f1"],
                  "superglue-boolq": ["accuracy"],
                  "superglue-rte": ["accuracy"],
                  "superglue-cb": ["f1_multiclass", "accuracy"],
                  "superglue-copa": ["accuracy"],
                  "superglue-multirc": ["f1", "em"],
                  "superglue-wic": ["accuracy"],
                  "superglue-wsc.fixed": ["accuracy"],
                  "superglue-record": ["f1", "em"]
         }

# run_seq2seq parameters.
@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    print_num_parameters: Optional[bool] = field(default=False, metadata={"help": "If set, print the parameters of "
                                                                                 "the model."})
    do_test: Optional[bool] = field(default=False, metadata={"help": "If set, evaluates the test performance."})
    split_validation_test: Optional[bool] = field(default=False,
                                                  metadata={"help": "If set, for the datasets which do not"
                                                                    "have the test set, we use validation set as their"
                                                                    "test set and make a validation set from either"
                                                                    "splitting the validation set into half (for smaller"
                                                                    "than 10K samples datasets), or by using 1K examples"
                                                                    "from training set as validation set (for larger"
                                                                    " datasets)."})
    compute_time: Optional[bool] = field(default=False, metadata={"help": "If set measures the time."})
    compute_memory: Optional[bool] = field(default=False, metadata={"help": "if set, measures the memory"})
    prefix_length: Optional[int] = field(default=100, metadata={"help": "Defines the length for prefix tuning."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    eval_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the evaluation dataset to use (via the datasets library)."}
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the evaluation dataset to use (via the datasets library)."}
    )
    test_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the test dataset to use (via the datasets library)."}
    )
    test_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the test dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."}
    )
    num_beams: Optional[int] = field(default=None, metadata={"help": "Number of beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    task_adapters: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from task adapters to the tasks."}
    )
    task_embeddings: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from tasks to the tasks embeddings."}
    )
    data_seed: Optional[int] = field(default=42, metadata={"help": "seed used to shuffle the data."})

    def __post_init__(self):
        if self.task_name is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
        if self.test_max_target_length is None:
            self.test_max_target_length = self.max_target_length


def resize_token_embeddings(adapter_args, model, tokenizer):
    if adapter_args.create_side_lm:
        tmp_get_input_embeddings = model.get_input_embeddings
        tmp_set_input_embeddings = model.set_input_embeddings
        tmp_set_output_embeddings = model.set_output_embeddings
        tmp_get_output_embeddings = model.get_output_embeddings

        # resize side network's modules
        model.get_input_embeddings = model.get_side_input_embeddings
        model.set_input_embeddings = model.set_side_input_embeddings
        model.set_output_embeddings = model.set_side_output_embeddings
        model.get_output_embeddings = model.get_side_output_embeddings

        model.resize_token_embeddings(len(tokenizer))

        model.get_input_embeddings = tmp_get_input_embeddings
        model.set_input_embeddings = tmp_set_input_embeddings
        model.set_output_embeddings = tmp_set_output_embeddings
        model.get_output_embeddings = tmp_get_output_embeddings

        model.resize_token_embeddings(len(tokenizer))
    else:
        model.resize_token_embeddings(len(tokenizer))


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               AdapterTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("#### last_checkpoint ", last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            '''
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            '''
            pass 
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `summary_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = T5Config.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.train_task_adapters = adapter_args.train_task_adapters
    config.train_side_ladder = adapter_args.train_side_ladder
    config.prefix_tuning = adapter_args.prefix_tuning
    config.lambda_distill = adapter_args.lambda_distill
    config.lambda_label = adapter_args.lambda_label
    config.lambda_kd_ir = adapter_args.lambda_kd_ir
    config.gate_T = adapter_args.gate_T
    config.use_gate = adapter_args.use_gate
    config.gate_alpha = adapter_args.gate_alpha
    training_args.use_gate = adapter_args.use_gate
    config.add_residual_after = adapter_args.add_residual_after
    config.encoder_side_layers = adapter_args.encoder_side_layers
    config.decoder_side_layers = adapter_args.decoder_side_layers
    config.side_downsample_pool = adapter_args.side_downsample_pool
    config.add_bias_sampling = adapter_args.add_bias_sampling
    config.merge_last = adapter_args.merge_last


    if adapter_args.train_lora:
        from seq2seq.lora import LoraConfig
        lora_config = LoraConfig()
        lora_config.lora_dim = adapter_args.lora_dim
        lora_config.tasks = [data_args.task_name]
    else:
        lora_config = None

    adapter_config = get_adapter_config(adapter_args, data_args, training_args, config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    data_args.dataset_name = [data_args.task_name]
    data_args.eval_dataset_name = [data_args.eval_dataset_name]
    data_args.test_dataset_name = [data_args.test_dataset_name]
    data_args.dataset_config_name = [data_args.dataset_config_name]
    data_args.eval_dataset_config_name = [data_args.eval_dataset_config_name]
    data_args.test_dataset_config_name = [data_args.test_dataset_config_name]
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(data_args.eval_dataset_config_name)
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(data_args.test_dataset_config_name)

    # Temporarily set max_target_length for training.
    #max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    
    def preprocess_function(examples, max_target_length):

        model_inputs = tokenizer(examples['source'], max_length=data_args.max_source_length,
                                padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target'], max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["extra_fields"] = examples['extra_fields']
        return model_inputs

    def preprocess_function_t5_mlm(examples, *args, **kwargs):
        # just extract the source, and the data collator will do the rest of work
        return {"source": examples["source"], "extra_fields": examples['extra_fields']}

    preprocess_function_chosen = preprocess_function_t5_mlm if adapter_args.train_t5_mlm else preprocess_function

    column_names = ['source', 'target', 'extra_fields']
    performance_metrics = {}
    if training_args.do_train:
        train_datasets = [AutoTask.get(dataset_name,
                                       dataset_config_name,
                                       seed=data_args.data_seed).get(
            split="train", 
            split_validation_test=training_args.split_validation_test,
            add_prefix=False if adapter_args.train_task_adapters else True,
            n_obs=data_args.max_train_samples)
            for dataset_name, dataset_config_name\
            in zip(data_args.dataset_name, data_args.dataset_config_name)]
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(\
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)\
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)]
        for i, train_dataset in enumerate(train_datasets):
            train_datasets[i] = train_datasets[i].map(
                functools.partial(preprocess_function_chosen, max_target_length=max_target_lengths[i]),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names, # if train_dataset != "superglue-record" else column_names+["answers"],
                load_from_cache_file=not data_args.overwrite_cache,
            )
        train_dataset = concatenate_datasets(train_datasets)
   
    if training_args.do_eval:
        eval_datasets = {eval_dataset: AutoTask.get(eval_dataset, eval_dataset_config,
            seed=data_args.data_seed).get(
            split="validation", 
            split_validation_test=training_args.split_validation_test,
            add_prefix=False if adapter_args.train_task_adapters else True,
            n_obs=data_args.max_val_samples)
            for eval_dataset, eval_dataset_config in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)}
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length( \
            tokenizer=tokenizer, default_max_length=data_args.max_target_length) \
            for dataset_name, dataset_config_name in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)]
        for k, name in enumerate(eval_datasets):
            eval_datasets[name] = eval_datasets[name].map(
                    functools.partial(preprocess_function_chosen, max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names, # if name != "superglue-record" else column_names+["answers"],
                    load_from_cache_file=not data_args.overwrite_cache,
            )

    if training_args.do_test:
        test_datasets = {test_dataset: AutoTask.get(test_dataset, test_dataset_config,
            seed=data_args.data_seed).get(
            split="test", 
            split_validation_test=training_args.split_validation_test,
            add_prefix=False if adapter_args.train_task_adapters else True,
            n_obs=data_args.max_test_samples)
            for test_dataset, test_dataset_config in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)}
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length( \
            tokenizer=tokenizer, default_max_length=data_args.max_target_length) \
            for dataset_name, dataset_config_name in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)]
        for k, name in enumerate(test_datasets):
            test_datasets[name] = test_datasets[name].map(
                    functools.partial(preprocess_function_chosen, max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if adapter_args.train_t5_mlm:
        training_args.remove_unused_columns = False # avoid removing example["source"] when feeding examples to data collator
        data_collator = DataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=adapter_args.mlm_ratio,
            mean_noise_span_length=3,
            input_length=data_args.max_source_length,
            target_length=data_args.max_target_length,
            pad_token_id=config.pad_token_id,
            decoder_start_token_id=config.decoder_start_token_id,
        )
    elif data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = TaskDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    if adapter_args.train_t5_mlm:
        from seq2seq.metrics import metrics
        # just to avoid error happens, whatever metric is used doesn't effect the selected models in distillation
        eval_metrics = [AutoTask.get(dataset_name, dataset_config_name).metric if dataset_name in ["cola", "stsb"] else [metrics.accuracy] \
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)][0]
    else:
        # Metric, we assume we have only one training task.
        eval_metrics = [AutoTask.get(dataset_name, dataset_config_name).metric\
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)][0]

    # Extracts the extra information needed to evaluate on each dataset.
    # These information are only used in the compute_metrics.
    # We will assume that the test/eval dataloader does not change the order of 
    # the data.

    if training_args.do_train:
        data_info = {"eval": eval_datasets[data_args.eval_dataset_name[0]]['extra_fields'],
                    "test": test_datasets[data_args.test_dataset_name[0]]['extra_fields'], 
                    "train": train_dataset['extra_fields']}
    else:
        data_info = {"eval": eval_datasets[data_args.eval_dataset_name[0]]['extra_fields'],
                    "test": test_datasets[data_args.test_dataset_name[0]]['extra_fields']}

    def compute_metrics(eval_preds):
        preds, labels, data_info = eval_preds
        post_processor = AutoPostProcessor.get(data_args.dataset_name[0], tokenizer,
                                               data_args.ignore_pad_token_for_loss)
        decoded_preds, decoded_labels = post_processor.process(preds, labels, data_info)
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result

    # we create a stand-alone model (without a side transformer) for computing fisher
    # so we call it before creating the model with side transformer to avoid holding two models 
    # in the save time (to save memory).
    if "fisher" in adapter_args.load_side_pretrained_weights:
        from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))

        from seq2seq.pruning.fisher import compute_fisher
        from transformers import DataCollatorForSeq2Seq

        d_collator_for_fish = DataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
        importance_measure = compute_fisher(model, train_dataset, d_collator_for_fish, num_samples=adapter_args.samples_for_fisher)

        if adapter_args.use_bottleneck:
            from seq2seq.pruning.pruning_methods import pruning_bottleneck
            method = pruning_bottleneck
        if adapter_args.create_side_lm:
            from seq2seq.pruning.pruning_methods import pruning_with_residual
            method = pruning_with_residual
        else:
            from seq2seq.pruning.pruning_methods import pruning_v2, pruning_v3, pruning_v4, pruning_v5
            pruning_version = adapter_args.load_side_pretrained_weights.split("-")[-1]
            method = eval(f"pruning_{pruning_version}")

        print(method.__name__)

        pruned_state_dict = method(model, adapter_args.task_reduction_factor, importance_measure=importance_measure)

    if "t5-base" in adapter_args.load_side_pretrained_weights:
        from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))

        if adapter_args.use_bottleneck:
            from seq2seq.pruning.pruning_methods import pruning_bottleneck
            method = pruning_bottleneck
        if adapter_args.create_side_lm:
            from seq2seq.pruning.pruning_methods import pruning_with_residual
            method = pruning_with_residual
        else:
            from seq2seq.pruning.pruning_methods import pruning_v2, pruning_v3, pruning_v4, pruning_v5
            pruning_version = adapter_args.load_side_pretrained_weights.split("-")[-1]
            method = eval(f"pruning_{pruning_version}")

        print(method.__name__)

        pruned_state_dict = method(model, adapter_args.task_reduction_factor)

    # Initialize the model 
    if adapter_args.train_side_transformer:
        if adapter_args.use_bottleneck:
            from seq2seq.third_party.models.t5.modeling_bottleneck_t5 import T5ForConditionalGeneration
        elif adapter_args.use_updown:
            from seq2seq.third_party.models.t5.modeling_side_updown_t5 import T5ForConditionalGeneration
        elif adapter_args.create_side_lm:
            from seq2seq.third_party.models.t5.modeling_side_logit_t5 import T5ForConditionalGeneration
        else:
            from seq2seq.third_party.models.t5.modeling_side_t5 import T5ForConditionalGeneration
    elif adapter_args.lit_distillation:
        from seq2seq.third_party.models.t5.modeling_lit import T5ForConditionalGeneration
    elif adapter_args.train_side_cross_transformer:
        from seq2seq.third_party.models.t5.modeling_cross_side_t5 import T5ForConditionalGeneration
    elif adapter_args.train_deepsidenet_transformer:
        from seq2seq.third_party.models.t5.modeling_deepsidenet_t5 import T5ForConditionalGeneration
    else:
        from seq2seq.third_party.models.t5.modeling_t5 import T5ForConditionalGeneration

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        adapter_config=adapter_config,
        lora_config=lora_config
    )

    # model.resize_token_embeddings(len(tokenizer))

    resize_token_embeddings(adapter_args, model, tokenizer)
    
    # Intialize side transformer
    if adapter_args.load_side_pretrained_weights == "t5-base":
        state_dict = model.state_dict()
        for n, p in model.named_parameters():
            if "side_block" in n:
                infer_n = n.split(".")
                infer_n[1] = "block"
                infer_n = ".".join(infer_n)

                print(n, infer_n)

                state = state_dict[infer_n]

                p.data.copy_(state.data)

            if "final_side_layer_norm" in n:
                infer_n = n.split("_")
                infer_n.pop(1)
                infer_n = "_".join(infer_n)

                print(n, infer_n)

                state = state_dict[infer_n]

                p.data.copy_(state.data)

            if "side_lm_head" in n:
                infer_n = n.split(".")
                infer_n[1] = "lm_head"
                infer_n = ".".join(infer_n)

                print(n, infer_n)

                state = pruned_state_dict[infer_n]

                p.data.copy_(state)

            if "side_shared" in n:
                infer_n = n.split(".")
                infer_n[0] = "shared"
                infer_n = ".".join(infer_n)

                print(n, infer_n)

                state = pruned_state_dict[infer_n]

                p.data.copy_(state)

    elif adapter_args.load_side_pretrained_weights == "arbitrary":
        state_dict = model.state_dict()
        for n, p in model.named_parameters():
            if "side_block" in n:
                infer_n = n.split(".")
                infer_n[1] = "block"
                infer_n = ".".join(infer_n)

                print(n, infer_n)

                state = state_dict[infer_n]

                if len(state.shape) == 2:
                    load_weights = state[:p.shape[0], :p.shape[1]]
                elif len(state.shape) == 1:
                    load_weights = state[:p.shape[0]]

                p.data.copy_(load_weights)

            if "final_side_layer_norm" in n:
                infer_n = n.split("_")
                infer_n.pop(1)
                infer_n = "_".join(infer_n)

                print(n, infer_n)

                state = state_dict[infer_n]

                load_weights = state[:p.shape[0]]

                p.data.copy_(load_weights)

            if "side_lm_head" in n:
                infer_n = n.split(".")
                infer_n[1] = "lm_head"
                infer_n = ".".join(infer_n)

                print(n, infer_n)

                state = pruned_state_dict[infer_n]

                p.data.copy_(state)

            if "side_shared" in n:
                infer_n = n.split(".")
                infer_n[0] = "shared"
                infer_n = ".".join(infer_n)

                print(n, infer_n)

                state = pruned_state_dict[infer_n]

                p.data.copy_(state)
    # elif "t5-base" in adapter_args.load_side_pretrained_weights:
    #     print(f"Load weights from {adapter_args.load_side_pretrained_weights}")
    #     state_dict = torch.load(adapter_args.load_side_pretrained_weights)
    #     self_model_state_dict = model.state_dict()
    #     for n, p in model.named_parameters():
    #         if "side_block" in n:
    #             infer_n = n.split(".")
    #             infer_n[1] = "block"
    #             infer_n = ".".join(infer_n)

    #             print(n, infer_n)

    #             if "relative_attention_bias" in infer_n:
    #                 # the size is wrong in pre-trained weights, so load from self model
    #                 state = self_model_state_dict[infer_n]
    #             else:
    #                 state = state_dict[infer_n]

    #             p.data.copy_(state)

    #         if "final_side_layer_norm" in n:
    #             infer_n = n.split("_")
    #             infer_n.pop(1)
    #             infer_n = "_".join(infer_n)

    #             print(n, infer_n)

    #             if adapter_args.use_updown:
    #                 # the size is wrong in pre-trained weights, so load from self model
    #                 state = self_model_state_dict[infer_n]
    #             else:
    #                 state = state_dict[infer_n]

    #             p.data.copy_(state)
    elif "fisher" in adapter_args.load_side_pretrained_weights or "t5-base" in adapter_args.load_side_pretrained_weights:
        self_model_state_dict = model.state_dict()
        for n, p in model.named_parameters():
            if "side_block" in n:
                if "relative_attention_bias" in n:
                    # only in the first layer of the pre-trained model
                    infer_n = n.split(".")
                    infer_n[1] = "block"
                    infer_n[2] = "0"
                    infer_n = ".".join(infer_n)

                    # the size is wrong in pre-trained weights, so load from self model
                    state = self_model_state_dict[infer_n]
                    p.data.copy_(state)

                elif adapter_args.train_side_cross_transformer and "encoder" in n and any([_t in n for _t in ["1.EncDecAttention", "1.layer_norm", "2.layer_norm", "2.DenseReluDense"]]):
                    infer_n = n.split(".")
                    if "1.EncDecAttention" in n:
                        infer_n[1] = "block"
                        infer_n[4] = "0"
                        infer_n[5] = "SelfAttention"

                    if "1.layer_norm" in n:
                        infer_n[1] = "block"
                        infer_n[4] = "0"

                    if "2.DenseReluDense" in n:
                        infer_n[1] = "block"
                        infer_n[4] = "1"

                    if "2.layer_norm" in n:
                        infer_n[1] = "block"
                        infer_n[4] = "1"

                    infer_n = ".".join(infer_n)
                    print(n, infer_n)

                    state = pruned_state_dict[infer_n]

                    p.data.copy_(state.data)

                elif adapter_args.train_side_cross_transformer and "decoder" in n and any([_t in n for _t in ["1.EncDecAttention", "1.layer_norm", "2.EncDecAttention", "2.layer_norm", "3.layer_norm", "3.DenseReluDense"]]):
                    infer_n = n.split(".")
                    if "1.EncDecAttention" in n:
                        # side cross attn
                        infer_n[1] = "block"
                        infer_n[4] = "0"
                        infer_n[5] = "SelfAttention"

                    if "1.layer_norm" in n:
                        infer_n[1] = "block"
                        infer_n[4] = "0"

                    if "2.EncDecAttention" in n:
                        # cross attn
                        infer_n[1] = "block"
                        infer_n[4] = "1"

                    if "2.layer_norm" in n:
                        infer_n[1] = "block"
                        infer_n[4] = "1"

                    if "3.DenseReluDense" in n:
                        infer_n[1] = "block"
                        infer_n[4] = "2"

                    if "3.layer_norm" in n:
                        infer_n[1] = "block"
                        infer_n[4] = "2"

                    infer_n = ".".join(infer_n)
                    print(n, infer_n)

                    state = pruned_state_dict[infer_n]

                    p.data.copy_(state.data)
        
                else:
                    infer_n = n.split(".")
                    infer_n[1] = "block"
                    infer_n = ".".join(infer_n)

                    state = pruned_state_dict[infer_n]

                    p.data.copy_(state)

                print(n, infer_n)
                
            if "final_side_layer_norm" in n:
                infer_n = n.split("_")
                infer_n.pop(1)
                infer_n = "_".join(infer_n)

                print(n, infer_n)

                if adapter_args.use_updown:
                    # the size is wrong in pre-trained weights, so load from self model
                    state = self_model_state_dict[infer_n]
                else:
                    state = pruned_state_dict[infer_n]

                p.data.copy_(state)

            if "side_lm_head" in n:
                infer_n = n.split(".")
                infer_n[1] = "lm_head"
                infer_n = ".".join(infer_n)

                print(n, infer_n)

                state = pruned_state_dict[infer_n]

                p.data.copy_(state)

            if "side_shared" in n:
                infer_n = n.split(".")
                infer_n[0] = "shared"
                infer_n = ".".join(infer_n)

                print(n, infer_n)

                state = pruned_state_dict[infer_n]

                p.data.copy_(state)

    elif adapter_args.load_side_pretrained_weights == "":
        # don't load pretrained weights for the side network
        pass
    else:
        raise NotImplementedError

    model, total_trainable_params_percent = modify_model_after_init(model, training_args, adapter_args)

    if adapter_args.init_side_downsampling:
        for n, p in model.named_parameters():
            if "side_downsamples" in n or "side_first_downsample" in n:
                p.requires_grad = False
                if "bias" in n:
                    p.data.zero_()
                    continue

                identity_matrix = torch.zeros(p.shape)
                perm = torch.randperm(p.shape[1])
                idx = perm[:p.shape[0]]
                idx = torch.sort(idx)[0]
                identity_matrix[torch.arange(p.shape[0]), idx] = 1
                p.data.copy_(identity_matrix)

            if "side_final_upsample" in n:
                p.requires_grad = False
                if "bias" in n:
                    p.data.zero_()
                    continue

                identity_matrix = torch.zeros(p.shape)
                perm = torch.randperm(p.shape[0])
                idx = perm[:p.shape[1]]
                idx = torch.sort(idx)[0]
                identity_matrix[idx, torch.arange(p.shape[1])] = 1
                p.data.copy_(identity_matrix)

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=list(eval_datasets.values())[0] if training_args.do_eval else None,
        data_info = data_info,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        evaluation_metrics = TASK_TO_METRICS[data_args.dataset_name[0]]
    )
    # Saves training config. 
    if trainer.is_world_process_zero():
       os.makedirs(training_args.output_dir, exist_ok=True)
       save_training_config(sys.argv[1], training_args.output_dir)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"total_time in minutes ": total_time})
        
        trainer.save_model()  # Saves the tokenizer too for easy upload
        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        train_metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
        performance_metrics.update({"peak_memory": peak_memory})
    if training_args.compute_memory or training_args.compute_time:
        print(performance_metrics)
        trainer.save_metrics("performance", performance_metrics)

    trainer.save_metrics("updated_params", {"updated_params": total_trainable_params_percent})
    
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        for task, eval_dataset in eval_datasets.items():
            metrics = trainer.evaluate(eval_dataset=eval_dataset,
               max_length=data_args.val_max_target_length, num_beams=data_args.num_beams,
            )
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # only useful when computing inference memory
        if torch.cuda.is_available() and training_args.compute_memory:
            peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
            print(
                "Memory utilization",
                peak_memory,
                "GB"
            )

    # Test
    if training_args.do_test:
        logger.info("*** Test ***")
        for task, test_dataset in test_datasets.items():
            metrics = trainer.evaluate(eval_dataset=test_dataset,
              max_length=data_args.test_max_target_length, num_beams=data_args.num_beams,
              metric_key_prefix="test"
            )
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
