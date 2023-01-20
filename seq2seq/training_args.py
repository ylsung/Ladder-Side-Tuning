from seq2seq.adapters import ADAPTER_CONFIG_MAPPING
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AdapterTrainingArguments:
    """Defines the adapters parameters."""
    train_task_adapters: Optional[bool] = field(default=False,
                                                metadata={"help": "If set, adds task adapters in the model."})
    train_side_ladder: Optional[bool] = field(default=False,
                                                metadata={"help": "If set, adds side ladder in the model."})
    train_side_transformer: Optional[bool] = field(default=False,
                                                metadata={"help": "If set, adds side transformer in the model."})
    train_deepsidenet_transformer: Optional[bool] = field(default=False,
                                                metadata={"help": "If set, adds side deep transformer in the model."})
    train_side_cross_transformer: Optional[bool] = field(default=False,
                                                metadata={"help": "If set, adds side cross transformer in the model."})
    train_lora: Optional[bool] = field(default=False,
                                                metadata={"help": "If set, adds lora in the model."})
    adapter_config_name: Optional[str] = field(
        default="adapter", metadata={"help": "config name for the adapter layers, should be selected "
        f"in {sorted(ADAPTER_CONFIG_MAPPING.keys())}."}
    )
    add_layer_norm_before_adapter: Optional[bool] = field(default=False, metadata={
        "help": "whether to have layer-norm before adapter."})
    add_layer_norm_after_adapter: Optional[bool] = field(default=True,
        metadata={"help": "whether to have layer-norm after adapter."})
    hidden_dim: Optional[int] = field(default=128, metadata={"help": "defines the default hidden dimension for "
        "adapter layers."})
    task_reduction_factor: Optional[int] = field(default=16, metadata={"help": "defines the default reduction factor for "
        "adapter layers."})
    non_linearity: Optional[str] = field(default="swish", metadata={"help": "Defines nonlinearity for adapter layers."})
    unfreeze_lm_head: bool = field(default=False, metadata={"help": "If set unfreeze the last linear layer."})
    freeze_lm_head: bool = field(default=False, metadata={"help": "If set, freeze the last linear layer."})
    unfreeze_layer_norms: bool = field(default=False, metadata={"help": "If set, unfreezes the layer norms."})
    task_adapter_layers_encoder: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which task adapters is"
                                                                                      "added in the encoder."})
    task_adapter_layers_decoder: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which task adapters is"
                                                                                      "added in the decoder."})
    task_adapter_in_decoder: Optional[bool] = field(default=True, metadata={"help": "If set to false, do not include"
                                                                                    "task adapters in the decoder."})
    hypercomplex_adapters: Optional[bool] = field(default=False, metadata={"help": "If set, uses the hypercomplex layers"
                                                                                "for adapters."})
    hypercomplex_division: Optional[int] = field(default=8, metadata={"help": "Defines the number to divide the dimensions"
                                                                              "of the linear layer by it."})
    intrinsic_model: Optional[bool] = field(default=False, metadata={"help": "If set, computes all parameters of the "
                                                                             "model with an intrinsic vector."})
    intrinsic_said: Optional[bool] = field(default=False, metadata={"help": "If set, computes the SAID version of the"
                                                                            "model with intrinsic vector."})
    intrinsic_dim: Optional[int] = field(default=100, metadata={"help": "Defines the intrinsic dimensionality."})
    normalize_intrinsic_projections: Optional[bool] = field(default=False, metadata={"help": "If set, normalizes "
        "the intrinsic projection matrices."})
    intrinsic_projection: Optional[str] = field(default="fastfood", metadata={"help": "Defines the type of projection"
        "for intrinsic adapters, it can be random or fastfood."})
    learn_phm: Optional[bool] = field(default=True, metadata={"help": "If set, learns the phm rules in Hypercomplex adapters."})
    normalize_phm_weight: Optional[bool] = field(default=False, metadata={"help": "Weather to normalize the weights of"
                                                                                  "the PHM layer."})
    intrinsic_layer_norms: Optional[bool] = field(default=False, metadata={"help": "If selected, then in case of unfreezing"
        " layernorms for intrinsic_adapters case, it also adds the layernorms parameters inside the parameters given for"
        " the intrinsic projection, and if this is not set, those parameters are not projected with intrinsic vector."})
    hypercomplex_nonlinearity: Optional[str] = field(default="glorot-uniform", metadata={"help": "Defines the nonlinearity for the"
        " hypercomplex adapter layers."})
    shared_phm_rule: Optional[bool] = field(default=False, metadata={"help": "If set, uses a shared phm rules for all"
        " hypercomplex adapter layers."})
    factorized_phm: Optional[bool] = field(default=False, metadata={"help": "If set, it factorizes the weights for the W in"
        " hypercomplex adapters."})
    shared_W_phm: Optional[bool] = field(default=False, metadata={"help": "If set, shares the W in phm adapter layers between all adapters."})
    factorized_phm_rule: Optional[bool] = field(default=False, metadata={"help": "If set, it factorizes the shared weights for the W in"
        " hypercomplex adapters."})
    phm_c_init: Optional[str] = field(default="normal", metadata={"help": "Initialization for the phm rules."})
    phm_rank: Optional[int] = field(default=1, metadata={"help":"sets the rank for the phm decomposition."})
    phm_init_range: Optional[float] = field(default=0.01, metadata={"help": "defines the phm init range."})
    add_adapter_in_feed_forward: Optional[bool] = field(default=True, metadata={"help": "If set, adds adapters in the feedforward."})
    add_adapter_in_self_attention: Optional[bool] = field(default=True, metadata={"help": "If set, adds adapters in the selfattention"})
    prefix_tuning: Optional[bool] = field(default=False, metadata={"help": "If set, uses prefix tuning."})
    prefix_dim: Optional[int] = field(default=100, metadata={"help": "Specifies the prefix embedding dimension."})
    init_prefix_from_vocab: Optional[bool] = field(default=False, metadata={"help": "Initialize prefix from the tokens of pretrained t5-base model."})
    kronecker_prod: Optional[bool] = field(default=False, metadata={"help": "If set, compute the kronecker using another version."})
    bitfit: Optional[bool] = field(default=False, metadata={"help": "If set, we train the bitfit model."})
    freeze_bitfit_lm_head: Optional[bool] = field(default=False, metadata={"help": "If set, freezes the classifier in bitfit."})
    freeze_bitfit_lm_head_all: Optional[bool] = field(default=False, metadata={"help": "If set, freezes the classifier in bitfit."})
    # Low-rank adapters.
    low_rank_adapters: Optional[bool] = field(default=False, metadata={"help": "If set, uses the low-rank adapters."})
    low_rank_w_init: Optional[str] = field(default="glorot-uniform", metadata={"help": "Defines the initialization for low-rank adapters."})
    low_rank_rank: Optional[int] = field(default=1, metadata={"help": "Defines the rank of low-rank adapters."})
    lambda_distill: float = field(default=0, metadata={"help": "The weight for distill loss."})
    lambda_label: float = field(default=1, metadata={"help": "The weight for label loss."})
    lambda_kd_ir: float = field(default=1, metadata={"help": "The weight for balancing kd and ir loss (only used in LIT distillation)."})
    lit_distillation: bool = field(default=False, metadata={"help": "Whether to use LIT distillation."})
    gate_T: float = field(default=0.1, metadata={"help": "The temperature for gates."})
    gate_alpha: float = field(default=0, metadata={"help": "The initial parameter for gating"})
    use_gate: str = field(default="none", metadata={"help": "Whether to use gates"})
    load_side_pretrained_weights: str = field(default="", metadata={"help": "The intial weights for side network."})
    use_bottleneck: bool = field(default=False, metadata={"help": "If set, uses the transformers with bottleneck."})
    use_updown: bool = field(default=False, metadata={"help": "If set, uses the upsampling and downsampling in every side transformers layers."})
    add_residual_after: bool = field(default=False, metadata={"help": "If set, adds backbone's residual after side module"})
    encoder_side_layers: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which side transformer is"
                                                                                      "added in the encoder."})
    decoder_side_layers: Optional[List[int]] = field(default=None, metadata={"help": "Defines the layers id"
                                                                                      "in which side transformer is"
                                                                                      "added in the decoder."})

    side_downsample_pool: bool = field(default=False, metadata={"help": "If set, use pooling in side downsampling."})
    init_side_downsampling: bool = field(default=False, metadata={"help": "If set, initialize side downsampling with something like identity matrix."})
    create_side_lm: bool = field(default=False, metadata={"help": "If set, create lm_head and embedding layers for the side network."})
    freeze_side_lm: bool = field(default=False, metadata={"help": "If set, freeze the lm_head and embedding layers of the side network."})

    samples_for_fisher: Optional[int] = field(default=1024, metadata={"help": "How many samples are used to compute fisher information?"})
    weight_scale: bool = field(default=False, metadata={"help": "If set, scale the side network weights before training."})
    add_bias_sampling: bool = field(default=False, metadata={"help": "If set, add biases in up/down sampling layers."})

    trainable_encoder_layers: Optional[List[int]] = field(default=None, metadata={"help": "Defines the encoder layers id"
                                                                                      "in which parameters are trainable"})
    trainable_decoder_layers: Optional[List[int]] = field(default=None, metadata={"help": "Defines the decoder layers id"
                                                                                      "in which parameters are trainable"})

    lora_dim: int = field(default=16, metadata={"help": "The dimension of the lora linear layers."})
    merge_last: bool = field(default=False, metadata={"help": "If set, merge the information after the last layer of the backbone and side network."})

    train_t5_mlm: bool = field(default=False, metadata={"help": "If set, use t5 mlm data collator and use t5 mlm objective to train the model"})
    mlm_ratio: float = field(default=0.15, metadata={"help": "The masking ratio for MLM objective"}) 