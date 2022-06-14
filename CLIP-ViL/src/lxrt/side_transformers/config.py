class SideConfig:
    task_reduction_factor = 8
    encoder_side_layers = None
    add_residual_after = False
    use_gate = "learnable"
    gate_alpha = 0
    gate_T = 0.1
    side_downsample_pool = False