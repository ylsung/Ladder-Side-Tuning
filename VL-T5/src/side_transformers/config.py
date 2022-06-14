

class SideConfig:
    decoder_side_layers = None
    encoder_side_layers = None
    add_residual_after = False
    side_downsample_pool = False
    add_bias_sampling = True
    use_gate = "learnable"
    gate_T = 0.1
    gate_alpha = 0