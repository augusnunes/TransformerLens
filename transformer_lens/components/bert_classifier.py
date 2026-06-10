"""Hooked Encoder Bert Classifier Component.

This module contains all the component :class:`BertClassifier`.
"""

from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.config.hooked_transformer_config import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint


class BertClassifier(nn.Module):
    """
    O que ele faz.
    O propósito dele.
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.n_class, dtype=self.cfg.dtype))
        self.b = nn.Parameter(torch.zeros(self.cfg.n_class, dtype=self.cfg.dtype))
        self.hook_classifier_out = HookPoint()

    def forward(
        self, resid: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch d_model"]:
        logits_output = torch.matmul(resid, self.W) + self.b
        logits_output = self.hook_classifier_out(logits_output)
        return logits_output
