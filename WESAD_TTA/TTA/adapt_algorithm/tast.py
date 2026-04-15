"""WESAD 1D-CNN 用 TAST.

WESAD 版では追加 MLP ensemble ではなく、TAST_BN と同じ support-neighbor
prototype 予測を、モデル重みを更新しない classifier adjustment として使う。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from TTA.adapt_algorithm.common import (
    forward_with_features,
    logits_to_class_logits,
    probs_to_model_logits,
    softmax_entropy,
)
from TTA.adapt_algorithm.tast_bn import compute_logits, select_supports, target_generation


class TAST(nn.Module):
    """BN 更新を行わない TAST。support memory だけで予測を調整する。"""

    def __init__(self, args, model, optimizer=None, steps=1, episodic=False):
        super().__init__()
        self.args = args
        self.model = model
        self.steps = steps
        self.episodic = episodic
        self.num_classes = getattr(args, "num_classes", 2)
        self.label_mode = getattr(args, "label_mode", "binary")
        self.filter_K = getattr(args, "filter_K", 16)
        self.tau = getattr(args, "tast_tau", 10.0)
        self.k = getattr(args, "tast_k", 1)
        self.supports = None
        self.labels = None
        self.ent = None

    def forward(self, x):
        logits = None
        for _ in range(self.steps):
            logits = forward_and_adapt(self, x, self.model)
        return logits


@torch.no_grad()
def forward_and_adapt(self, x, model):
    model.eval()
    logits, feature = forward_with_features(model, x)
    class_logits = logits_to_class_logits(logits, self.label_mode)
    yhat = F.one_hot(class_logits.argmax(1), self.num_classes).float()
    ent = softmax_entropy(class_logits)

    if self.supports is None:
        self.supports = feature.detach()
        self.labels = yhat.detach()
        self.ent = ent.detach()
    else:
        self.supports = torch.cat([self.supports.to(feature.device), feature.detach()])
        self.labels = torch.cat([self.labels.to(feature.device), yhat.detach()])
        self.ent = torch.cat([self.ent.to(feature.device), ent.detach()])

    supports, labels = select_supports(self)
    _, outputs = target_generation(self, feature, supports, labels)
    return probs_to_model_logits(outputs, self.label_mode)


def collect_params(model):
    return [], []


def configure_model(model):
    model.eval()
    model.requires_grad_(False)
    return model
