"""WESAD 1D-CNN 用 Pseudo-Label test-time adaptation."""

import torch
import torch.nn as nn

from TTA.adapt_algorithm.common import (
    collect_bn1d_params,
    configure_bn1d_for_adaptation,
    copy_model_and_optimizer,
    load_model_and_optimizer,
    pseudo_label_loss,
)


class PL(nn.Module):
    """高信頼な予測を pseudo label として BN1d affine を更新する。"""

    def __init__(self, args, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.threshold = getattr(args, "pseudo_threshold", 0.9)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer, self.threshold)
        return outputs

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, threshold):
    """1 バッチで pseudo-label loss を最小化する。"""
    model.train()
    logits = model(x)
    loss = pseudo_label_loss(logits, confidence_threshold=threshold)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        logits = model(x)
    return logits


def collect_params(model):
    return collect_bn1d_params(model)


def configure_model(model):
    return configure_bn1d_for_adaptation(model, train=True, update_params=True, use_batch_stats=True)
