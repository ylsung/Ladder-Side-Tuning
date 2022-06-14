from dataclasses import dataclass


@dataclass
class LoraConfig(object):
    lora_dim = 16
    lora_alpha = 32
    lora_dropout = 0.1
    tasks = ["dummy_task"]
    use_single_lora = False
