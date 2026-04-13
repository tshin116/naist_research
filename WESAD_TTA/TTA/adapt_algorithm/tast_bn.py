"""WESAD 1D-CNN 用 TAST_BN."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from TTA.adapt_algorithm.common import (
    binary_logits_to_two_class,
    collect_bn1d_params,
    configure_bn1d_for_adaptation,
    copy_model_and_optimizer,
    forward_with_features,
    load_model_and_optimizer,
    softmax_entropy,
    two_class_probs_to_binary_logits,
)


class TAST_BN(nn.Module):
    """BN1d を更新しながら、低 entropy support の近傍プロトタイプで予測する。"""

    def __init__(self, args, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.num_classes = getattr(args, "num_classes", 2)
        self.filter_K = getattr(args, "filter_K", 16)
        self.tau = getattr(args, "tast_tau", 10.0)
        self.k = getattr(args, "tast_k", 1)
        self.supports = None
        self.labels = None
        self.ent = None
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(self, x, self.model, self.optimizer)
        return outputs

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.supports = None
        self.labels = None
        self.ent = None


@torch.enable_grad()
def forward_and_adapt(self, x, model, optimizer):
    model.train()
    logits, feature = forward_with_features(model, x)
    two_class_logits = binary_logits_to_two_class(logits)
    yhat = F.one_hot(two_class_logits.argmax(1), self.num_classes).float()
    ent = softmax_entropy(two_class_logits)

    if self.supports is None:
        self.supports = feature.detach()
        self.labels = yhat.detach()
        self.ent = ent.detach()
    else:
        self.supports = torch.cat([self.supports.to(feature.device), feature.detach()])
        self.labels = torch.cat([self.labels.to(feature.device), yhat.detach()])
        self.ent = torch.cat([self.ent.to(feature.device), ent.detach()])

    supports, labels = select_supports(self)
    targets, outputs = target_generation(self, feature, supports, labels)
    proto_logits = compute_logits(self, feature, supports, labels)
    loss = F.kl_div(proto_logits.log_softmax(-1), targets, reduction="batchmean")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return two_class_probs_to_binary_logits(outputs)


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


def compute_logits(self, feature, supports, labels):
    centroids = (labels / (labels.sum(dim=0, keepdim=True) + 1e-12)).T @ supports
    feature = F.normalize(feature, dim=1)
    centroids = F.normalize(centroids, dim=1)
    return self.tau * feature @ centroids.T


def target_generation(self, feature, supports, labels):
    weights = torch.exp(-cosine_distance(feature, supports))
    k = min(self.k, max(1, supports.size(0)))
    _, indices = torch.topk(weights, k, sorted=False)
    topk = torch.zeros_like(weights).scatter_(1, indices, 1)
    support_logits = compute_logits(self, supports, supports, labels)
    support_targets = F.one_hot(support_logits.argmax(-1), num_classes=self.num_classes).float()
    support_outputs = torch.softmax(support_logits, dim=1)
    targets = topk @ support_targets
    outputs = topk @ support_outputs
    targets = targets / (targets.sum(dim=1, keepdim=True) + 1e-12)
    outputs = outputs / (outputs.sum(dim=1, keepdim=True) + 1e-12)
    return targets, outputs


def cosine_distance(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 1.0 - x @ y.T


def collect_params(model):
    return collect_bn1d_params(model)


def configure_model(model):
    return configure_bn1d_for_adaptation(model, train=True, update_params=True, use_batch_stats=True)
