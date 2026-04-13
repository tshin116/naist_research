"""WESAD 1D-CNN 用 SHOT-style test-time adaptation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from TTA.adapt_algorithm.common import (
    binary_logits_to_two_class,
    collect_bn1d_params,
    configure_bn1d_for_adaptation,
    copy_model_and_optimizer,
    load_model_and_optimizer,
    pseudo_label_loss,
    softmax_entropy,
)


class SHOT(nn.Module):
    """Entropy、diversity、pseudo-label loss で BN1d affine を更新する。"""

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


def loss_shot(logits, threshold):
    """SHOT の entropy + diversity + pseudo-label loss。"""
    two_class_logits = binary_logits_to_two_class(logits)
    ent_loss = softmax_entropy(two_class_logits).mean()
    probs = torch.softmax(two_class_logits, dim=1)
    mean_probs = probs.mean(dim=0)
    diversity = torch.sum(mean_probs * torch.log(mean_probs.clamp_min(1e-12)))
    pl_loss = pseudo_label_loss(two_class_logits, confidence_threshold=threshold)
    return ent_loss + diversity + 0.1 * pl_loss


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, threshold):
    model.train()
    logits = model(x)
    loss = loss_shot(logits, threshold)
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
