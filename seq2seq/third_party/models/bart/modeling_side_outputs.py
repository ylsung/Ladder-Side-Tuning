from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
)


@dataclass
class SideBaseModelOutput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    last_side_hidden_state: torch.FloatTensor = None
    side_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    side_attentions: Optional[Tuple[torch.FloatTensor]] = None
