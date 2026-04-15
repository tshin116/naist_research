"""WESAD 1D-CNN 用 T3A."""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from TTA.adapt_algorithm.common import (
    classifier_weights,
    forward_with_features,
    softmax_entropy,
)


class t3a(nn.Module):
    """Test-Time Classifier Adjustment を WESAD の feature 空間で行う。"""

    def __init__(self, args, model, optimizer=None, steps=1, episodic=False):
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.num_classes = getattr(args, "num_classes", 3)
        self.filter_K = getattr(args, "filter_K", 16)
        self.model.eval()
        self.model.requires_grad_(False)

        weights, bias = classifier_weights(model)
        self.register_buffer("warmup_supports", weights.detach().clone())
        warmup_logits = weights @ weights.T
        if bias is not None:
            warmup_logits = warmup_logits + bias.detach().view(1, -1)
        self.register_buffer("warmup_labels", F.one_hot(warmup_logits.argmax(1), self.num_classes).float())
        self.register_buffer("warmup_ent", softmax_entropy(warmup_logits).detach())

        self.supports = self.warmup_supports.detach().clone()
        self.labels = self.warmup_labels.detach().clone()
        self.ent = self.warmup_ent.detach().clone()
        self.model_state = deepcopy(self.model.state_dict())

    def forward(self, x):
        if self.episodic:
            self.reset()
        logits = None
        for _ in range(self.steps):
            logits = forward_and_adapt(self, x, self.model)
        return logits

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.supports = self.warmup_supports.detach().clone()
        self.labels = self.warmup_labels.detach().clone()
        self.ent = self.warmup_ent.detach().clone()


@torch.no_grad()
def forward_and_adapt(self, x, model):
    logits, feature = forward_with_features(model, x)
    yhat = F.one_hot(logits.argmax(1), self.num_classes).float()
    ent = softmax_entropy(logits)

    self.supports = self.supports.to(feature.device)
    self.labels = self.labels.to(feature.device)
    self.ent = self.ent.to(feature.device)
    self.supports = torch.cat([self.supports, feature.detach()])
    self.labels = torch.cat([self.labels, yhat.detach()])
    self.ent = torch.cat([self.ent, ent.detach()])

    supports, labels = select_supports(self)
    supports = F.normalize(supports, dim=1)
    weights = supports.T @ labels
    adjusted_logits = feature @ F.normalize(weights, dim=0)
    return adjusted_logits


def select_supports(self):
    ent_s = self.ent
    y_hat = self.labels.argmax(dim=1).long()
    if self.filter_K == -1:
        indices = torch.arange(len(ent_s), device=ent_s.device)
    else:
        indices = []
        all_indices = torch.arange(len(ent_s), device=ent_s.device)
        for class_idx in range(self.num_classes):
            class_indices = all_indices[y_hat == class_idx]
            if class_indices.numel() == 0:
                continue
            _, order = torch.sort(ent_s[class_indices])
            indices.append(class_indices[order][: self.filter_K])
        indices = torch.cat(indices) if indices else all_indices

    self.supports = self.supports[indices]
    self.labels = self.labels[indices]
    self.ent = self.ent[indices]
    return self.supports, self.labels


def collect_params(model):
    return [], []


def configure_model(model):
    model.eval()
    model.requires_grad_(False)
    return model
