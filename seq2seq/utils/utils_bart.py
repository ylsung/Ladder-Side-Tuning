import os 
import regex as re
import logging
from dataclasses import fields
import torch.nn as nn
import json
from collections import OrderedDict

from seq2seq.adapters import (AutoAdapterConfig, AdapterController, Adapter, HyperComplexAdapter)
from projections.intrinsic import intrinsic_dimension, intrinsic_dimension_said
from seq2seq.third_party.models.t5 import T5LayerNorm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_dir(output_dir):
    """
    Checks whether to the output_dir already exists and creates it if not.
    Args:
      output_dir: path to the output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_adapter_config(adapter_args, data_args, training_args, config):
    if adapter_args.train_task_adapters or adapter_args.prefix_tuning or adapter_args.bitfit or adapter_args.train_side_ladder or adapter_args.train_side_transformer or adapter_args.train_side_cross_transformer or adapter_args.train_deepsidenet_transformer:
        adapter_config = AutoAdapterConfig.get(adapter_args.adapter_config_name)
        adapter_config.input_dim = config.d_model

        adapter_config.train_side_transformer = adapter_args.train_side_transformer

        if adapter_args.train_task_adapters or adapter_args.train_side_ladder or adapter_args.train_side_transformer or adapter_args.train_side_cross_transformer or adapter_args.train_deepsidenet_transformer:
            data_args.tasks = [data_args.task_name] 
            adapter_config.tasks = data_args.tasks
        adapter_params = [field.name for field in fields(adapter_args)]
        for p in adapter_params:
            if hasattr(adapter_args, p) and hasattr(adapter_config, p) and\
                    getattr(adapter_args, p) is not None:
                setattr(adapter_config, p, getattr(adapter_args, p))
            else:
                logger.warning(f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute")
        adapter_config.device = training_args.device
        adapter_config.output_dir = training_args.output_dir
    else:
        adapter_config = None
    return adapter_config



def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def freeze_model_params(model, adapter_args):
    """
    Freezes the model parameters based on the given setting in the arguments.
    Args:
      model: the given model.
      adapter_args: defines the adapters arguments.
    """
    # If we are training adapters, we freeze all parameters except the
    # adapter parameters like adapter controllers.
    if adapter_args.train_task_adapters or adapter_args.train_side_ladder or adapter_args.train_side_transformer or adapter_args.train_side_cross_transformer or adapter_args.train_deepsidenet_transformer:
        freeze_params(model)
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (AdapterController, Adapter)):
                if isinstance(sub_module, (AdapterController, HyperComplexAdapter)) and adapter_args.hypercomplex_adapters:
                    for param_name, param in sub_module.named_parameters():
                        if any(x in param_name for x in ["phm_rule", "phm_rule_left", "phm_rule_right"]) and not adapter_args.learn_phm:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                else:
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
        
        if adapter_args.train_side_ladder:
            for n, p in model.named_parameters():
                if "side" in n:
                    p.requires_grad = True

        if adapter_args.train_side_transformer or adapter_args.train_deepsidenet_transformer:
            for n, p in model.named_parameters():
                if "side" in n:
                    p.requires_grad = True

        if adapter_args.train_side_cross_transformer:
            for n, p in model.named_parameters():
                if "side" in n:
                    p.requires_grad = True

        if adapter_args.freeze_side_lm:
            for param in model.side_lm_head.parameters():
                param.requires_grad = False

        if adapter_args.hypercomplex_adapters and adapter_args.shared_phm_rule:
            if adapter_args.factorized_phm_rule:
               model.phm_rule_left.requires_grad = True
               model.phm_rule_right.requires_grad = True
            else:
               model.phm_rule.requires_grad = True
                 
        if adapter_args.hypercomplex_adapters and adapter_args.shared_W_phm:
            if adapter_args.factorized_phm:
               model.W_down_left.requires_grad = True
               model.W_down_right.requires_grad = True
               model.W_up_left.requires_grad = True
               model.W_up_right.requires_grad = True
            else:
               model.W_down.requires_grad = True
               model.W_up.requires_grad = True

    # Unfreezes last linear layer of decoder.
    if adapter_args.unfreeze_lm_head:
        for param in model.classification_head.parameters():
              param.requires_grad = True

    # Unfreezes layer norms.
    if adapter_args.unfreeze_layer_norms:
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (T5LayerNorm, nn.LayerNorm)):
                if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

    if adapter_args.prefix_tuning:           
        freeze_params(model)
        for n, m in model.named_parameters():
            if "prefix_shared" == n:
               m.requires_grad = True 

    # For bitfit we freeze the whole model except for the biases and the final classifier layer.
    if adapter_args.bitfit: 
        freeze_params(model)
        # unfreeze bias terms.
        for n,p in model.named_parameters():
          if ".bias" in n:
            p.requires_grad = True

        # unfreeze the classifier.
        for param in model.lm_head.parameters():
            param.requires_grad = True
        if adapter_args.freeze_bitfit_lm_head:
           for n, param in model.lm_head.named_parameters():
                if "bias" in n:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if adapter_args.freeze_bitfit_lm_head_all:
           for n, param in model.lm_head.named_parameters():
                param.requires_grad = False

def get_adapter_params_names(model):
    """
    Returns adapter related parameters names.
    Args:
      model: the given model.
    """
    params_names = []
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, (AdapterController, Adapter)):
           for param_name, param in sub_module.named_parameters():
               params_names.append(name+"."+param_name)
    return params_names      


def get_layer_norm_params_names(model):
    """Returns the layer norms parameters.
    Args:
        model: the given model.
    """
    params_names = []
    for name, sub_module in model.named_modules():
        if isinstance(sub_module,  (T5LayerNorm, nn.LayerNorm)):
           for param_name, param in sub_module.named_parameters():
               params_names.append(name+"."+param_name)
    return params_names


def get_last_checkpoint(output_dir):
    if os.path.exists(os.path.join(output_dir, 'pytorch_model.bin')):
        return output_dir
    return None


def pad_punctuation(text):
   """Re-implementation of _pad_punctuation in t5. This function adds spaces
   around punctuation. While this pads punctuation as expected, it has the 
   unexpected effected of padding certain unicode characters with accents, with
   spaces as well. For instance: "François" becomes "Fran ç ois"""
   # Pad everything except for: underscores (_), whitespace (\s),
   # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
   text = re.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', text)
   # Collapse consecutive whitespace into one space.
   text = re.sub(r'\s+', ' ', text)
   return text


def modify_model_after_init(model, training_args, adapter_args):
    # Freezes model parameters.
    freeze_model_params(model, adapter_args)
    if adapter_args.intrinsic_model:
        if adapter_args.intrinsic_said:
           model = intrinsic_dimension_said(model, adapter_args.intrinsic_dim,\
               training_args.output_dir, set(), adapter_args.intrinsic_projection, "cpu")
        else:
           model = intrinsic_dimension(model, adapter_args.intrinsic_dim,\
               training_args.output_dir, set(), adapter_args.intrinsic_projection, "cpu")


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Model Trainable Parameters {} *****".format(trainable_params))
    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         print("inside n ", n)
    if training_args.print_num_parameters:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info("##### Parameter name %s", name)
        total_lm_head_params = sum(p.numel() for p in model.classification_head.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_trainable_bias_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.endswith(".b"))
        total_trainable_layernorm_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ".layer_norm.weight" in n)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total trainable parameters %s", total_trainable_params)
        logger.info("Total traianable bias parameters %s", total_trainable_bias_params)
        logger.info("Total trainable layernorm parameters %s", total_trainable_layernorm_params)
        logger.info("Total parameters %s", total_params)
        t5_base_params=204730370
        # total params since we have 8 task, it is Y = 1*BERT + 8*ADAPTERS, and final number is Y/BERT ("1.3x")
        total_params_ratio = ((total_params-t5_base_params)*8+t5_base_params)/t5_base_params
        total_trainable_params_percent =(total_trainable_params/t5_base_params)*100
        total_trainable_bias_params_percent =(total_trainable_bias_params/total_trainable_params)*100
        total_trainable_layernorm_params_percent =(total_trainable_layernorm_params/total_trainable_params)*100
        total_trainable_lm_head_params_percent =(total_lm_head_params/t5_base_params)*100
        logger.info("For adapters/prompt-tuning, total params %s", total_params_ratio)
        logger.info("For intrinsic, total params %s", total_params/t5_base_params)
        logger.info("Total trainable params %s", total_trainable_params_percent)
        logger.info("Total trainable bias params %s", total_trainable_bias_params_percent)
        logger.info("Total trainable layernorm params %s", total_trainable_layernorm_params_percent)
        logger.info("Total lm_head params %s", total_trainable_lm_head_params_percent)

    return model, total_trainable_params_percent

def save_json(filepath, dictionary):
   with open(filepath, "w") as outfile:
      json.dump(dictionary, outfile)


def read_json(filepath):
   f = open(filepath,)
   return json.load(f)


def save_training_config(config_file, output_dir):
   json_data = read_json(config_file)
   save_json(os.path.join(output_dir, "training_config.json"), json_data)

