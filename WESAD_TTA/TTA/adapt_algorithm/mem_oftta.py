"""Memory-balanced OFTTA for WESAD 1D-CNN.

This variant keeps source classifier prototypes as permanent anchors, adds
class-balanced target supports from confident test samples, weakens test-batch
BN statistics and support updates on class-biased batches, and resets short-term
supports when a block transition is detected.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from TTA.adapt_algorithm.common import (
    class_logits_to_model_logits,
    classifier_weights,
    forward_with_features,
    logits_to_class_logits,
    softmax_entropy,
)


class GatedWeightedBN1d(nn.Module):
    """BatchNorm1d using a gated mix of source running stats and test stats."""

    def __init__(self, layer, prior):
        super().__init__()
        if not 0.0 <= prior <= 1.0:
            raise ValueError("prior must be in [0, 1]")
        self.layer = layer
        self.layer.eval()
        self.base_prior = prior
        self.gate = 1.0
        self.running_mean = deepcopy(layer.running_mean.detach().clone())
        self.running_std = deepcopy(torch.sqrt(layer.running_var.detach().clone()) + layer.eps)

    def set_gate(self, gate):
        self.gate = float(max(0.0, min(1.0, gate)))

    def forward(self, x):
        reduce_dims = [0] + list(range(2, x.ndim))
        batch_mean = x.mean(dim=reduce_dims)
        batch_std = torch.sqrt(x.var(dim=reduce_dims, unbiased=False) + self.layer.eps)
        running_mean = self.running_mean.to(x.device)
        running_std = self.running_std.to(x.device)

        # gate=1 keeps the configured OFTTA prior; gate=0 falls back to source stats.
        effective_prior = 1.0 - self.gate * (1.0 - self.base_prior)
        weighted_mean = effective_prior * running_mean + (1.0 - effective_prior) * batch_mean.detach()
        weighted_std = effective_prior * running_std + (1.0 - effective_prior) * batch_std.detach()
        view_shape = [1, -1] + [1] * (x.ndim - 2)
        x = (x - weighted_mean.view(*view_shape)) / weighted_std.view(*view_shape)
        return x * self.layer.weight.view(*view_shape) + self.layer.bias.view(*view_shape)


def find_bns(model, priors):
    replace_mods = []
    bn_index = 0
    queue = [(model, "")]
    while queue:
        parent, prefix = queue.pop(0)
        for child_name, child in parent.named_children():
            module_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.BatchNorm1d):
                prior = priors[min(bn_index, len(priors) - 1)]
                replace_mods.append((parent, child_name, GatedWeightedBN1d(child, prior)))
                bn_index += 1
            else:
                queue.append((child, module_name))
    return replace_mods


class MemOFTTA(nn.Module):
    """OFTTA with class-balanced memory and batch-bias-aware adaptation."""

    def __init__(self, args, model, optimizer=None, steps=1, episodic=False):
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.num_classes = getattr(args, "num_classes", 2)
        self.label_mode = getattr(args, "label_mode", "binary")

        priors = get_priors(args, model)
        for parent, name, child in find_bns(model, priors):
            setattr(parent, name, child)
        self.bn_layers = [module for module in model.modules() if isinstance(module, GatedWeightedBN1d)]

        self.filter_K = getattr(args, "filter_K", 16)
        self.long_K = getattr(args, "mem_oftta_long_K", 32)
        self.short_K = getattr(args, "mem_oftta_short_K", 16)
        self.confidence_threshold = getattr(args, "mem_oftta_confidence_threshold", 0.75)
        self.update_gate_threshold = getattr(args, "mem_oftta_update_gate_threshold", 0.35)
        self.gate_power = getattr(args, "mem_oftta_gate_power", 1.0)
        self.probe_gate = getattr(args, "mem_oftta_probe_gate", 0.0)
        self.feature_shift_threshold = getattr(args, "mem_oftta_feature_shift_threshold", 0.65)
        self.pred_shift_threshold = getattr(args, "mem_oftta_pred_shift_threshold", 0.35)

        self.model.eval()
        self.model.requires_grad_(False)
        self.model_state = deepcopy(self.model.state_dict())

        weights, bias = classifier_weights(model, label_mode=self.label_mode)
        self.register_buffer("anchor_supports", weights.detach().clone())
        warmup_logits = weights @ weights.T
        if bias is not None:
            warmup_logits = warmup_logits + bias.detach().view(1, -1)
        self.register_buffer("anchor_labels", F.one_hot(warmup_logits.argmax(1), self.num_classes).float())
        self.register_buffer("anchor_ent", softmax_entropy(warmup_logits).detach())

        self.long_supports = empty_supports_like(self.anchor_supports)
        self.long_labels = empty_labels_like(self.anchor_labels)
        self.long_ent = empty_entropy_like(self.anchor_ent)
        self.short_supports = empty_supports_like(self.anchor_supports)
        self.short_labels = empty_labels_like(self.anchor_labels)
        self.short_ent = empty_entropy_like(self.anchor_ent)

        self.prev_feature_mean = None
        self.prev_pred_dist = None
        self.current_gate = 1.0

    def forward(self, x):
        if self.episodic:
            self.reset()

        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(self, x, self.model)
        return outputs

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.clear_memory()
        self.prev_feature_mean = None
        self.prev_pred_dist = None
        self.current_gate = 1.0

    def clear_memory(self):
        self.long_supports = empty_supports_like(self.anchor_supports)
        self.long_labels = empty_labels_like(self.anchor_labels)
        self.long_ent = empty_entropy_like(self.anchor_ent)
        self.reset_short_memory()

    def reset_short_memory(self):
        self.short_supports = empty_supports_like(self.anchor_supports)
        self.short_labels = empty_labels_like(self.anchor_labels)
        self.short_ent = empty_entropy_like(self.anchor_ent)

    def set_bn_gate(self, gate):
        self.current_gate = float(max(0.0, min(1.0, gate)))
        for layer in self.bn_layers:
            layer.set_gate(self.current_gate)


@torch.no_grad()
def forward_and_adapt(self, x, model):
    # Probe with source-anchored BN to estimate whether the current batch is safe
    # for test-stat mixing and support updates.
    self.set_bn_gate(self.probe_gate)
    probe_logits, probe_feature = forward_with_features(model, x)
    if probe_feature is None:
        return probe_logits

    probe_class_logits = logits_to_class_logits(probe_logits, self.label_mode)
    probe_probs = torch.softmax(probe_class_logits, dim=1)
    gate, confidence, pred_dist = batch_gate(probe_probs, self.gate_power)
    feature_mean = F.normalize(probe_feature.detach().mean(dim=0), dim=0)
    if detect_block_transition(self, feature_mean, pred_dist):
        self.reset_short_memory()

    self.set_bn_gate(gate)
    logits, feature = forward_with_features(model, x)
    class_logits = logits_to_class_logits(logits, self.label_mode)
    probs = torch.softmax(class_logits, dim=1)
    yhat_idx = class_logits.argmax(1)
    yhat = F.one_hot(yhat_idx, self.num_classes).float()
    ent = softmax_entropy(class_logits)

    if gate >= self.update_gate_threshold:
        update_memory(self, feature.detach(), yhat.detach(), ent.detach(), probs.detach())

    supports, labels = select_supports(self, feature.device)
    supports = F.normalize(supports, dim=1)
    weights = supports.T @ labels
    adjusted_logits = feature @ F.normalize(weights, dim=0)

    self.prev_feature_mean = feature_mean.detach()
    self.prev_pred_dist = pred_dist.detach()
    return class_logits_to_model_logits(adjusted_logits, self.label_mode)


def batch_gate(probs, gate_power):
    confidence = probs.max(dim=1).values.mean()
    pred_dist = probs.mean(dim=0)
    entropy = -(pred_dist * torch.log(pred_dist.clamp_min(1e-12))).sum()
    diversity = entropy / torch.log(torch.tensor(float(probs.size(1)), device=probs.device))
    gate = diversity.clamp(0.0, 1.0).pow(gate_power)
    # Very low-confidence batches should not aggressively update support memory.
    gate = gate * confidence.clamp(0.0, 1.0)
    return float(gate.item()), confidence, pred_dist


def detect_block_transition(self, feature_mean, pred_dist):
    if self.prev_feature_mean is None or self.prev_pred_dist is None:
        return False
    feature_shift = torch.norm(feature_mean - self.prev_feature_mean.to(feature_mean.device)).item()
    pred_shift = js_divergence(pred_dist, self.prev_pred_dist.to(pred_dist.device))
    return feature_shift > self.feature_shift_threshold or pred_shift > self.pred_shift_threshold


def js_divergence(p, q):
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    midpoint = 0.5 * (p + q)
    return float(0.5 * (kl_div(p, midpoint) + kl_div(q, midpoint)).item())


def kl_div(p, q):
    return (p * (torch.log(p) - torch.log(q.clamp_min(1e-12)))).sum()


def update_memory(self, feature, labels, ent, probs):
    confidence = probs.max(dim=1).values
    mask = confidence >= self.confidence_threshold
    if not mask.any():
        return
    feature = feature[mask]
    labels = labels[mask]
    ent = ent[mask]

    self.short_supports = torch.cat([self.short_supports.to(feature.device), feature])
    self.short_labels = torch.cat([self.short_labels.to(feature.device), labels])
    self.short_ent = torch.cat([self.short_ent.to(feature.device), ent])
    self.long_supports = torch.cat([self.long_supports.to(feature.device), feature])
    self.long_labels = torch.cat([self.long_labels.to(feature.device), labels])
    self.long_ent = torch.cat([self.long_ent.to(feature.device), ent])

    self.short_supports, self.short_labels, self.short_ent = classwise_topk(
        self.short_supports, self.short_labels, self.short_ent, self.short_K, self.num_classes
    )
    self.long_supports, self.long_labels, self.long_ent = classwise_topk(
        self.long_supports, self.long_labels, self.long_ent, self.long_K, self.num_classes
    )


def select_supports(self, device):
    supports = [self.anchor_supports.to(device)]
    labels = [self.anchor_labels.to(device)]
    ent = [self.anchor_ent.to(device)]

    if self.long_supports.numel():
        supports.append(self.long_supports.to(device))
        labels.append(self.long_labels.to(device))
        ent.append(self.long_ent.to(device))
    if self.short_supports.numel():
        supports.append(self.short_supports.to(device))
        labels.append(self.short_labels.to(device))
        ent.append(self.short_ent.to(device))

    supports = torch.cat(supports)
    labels = torch.cat(labels)
    ent = torch.cat(ent)
    return classwise_topk(supports, labels, ent, self.filter_K, self.num_classes)[:2]


def classwise_topk(supports, labels, ent, k, num_classes):
    if supports.numel() == 0 or k == -1:
        return supports, labels, ent
    y_hat = labels.argmax(dim=1).long()
    indices = []
    all_indices = torch.arange(len(ent), device=ent.device)
    for class_idx in range(num_classes):
        class_indices = all_indices[y_hat == class_idx]
        if class_indices.numel() == 0:
            continue
        _, order = torch.sort(ent[class_indices])
        indices.append(class_indices[order][:k])
    if not indices:
        return supports, labels, ent
    indices = torch.cat(indices)
    return supports[indices], labels[indices], ent[indices]


def empty_supports_like(anchor_supports):
    return anchor_supports.new_empty((0, anchor_supports.size(1)))


def empty_labels_like(anchor_labels):
    return anchor_labels.new_empty((0, anchor_labels.size(1)))


def empty_entropy_like(anchor_ent):
    return anchor_ent.new_empty((0,))


def get_priors(args, model):
    prior_max = getattr(args, "oftta_prior_max", 0.99)
    prior_min = getattr(args, "oftta_prior_min", 0.1)
    num_bn = sum(1 for module in model.modules() if isinstance(module, nn.BatchNorm1d))
    if num_bn <= 1:
        return [prior_min]
    factor = (prior_max / prior_min) ** (1 / (num_bn - 1))
    priors = []
    value = prior_min
    for _ in range(num_bn):
        priors.append(value)
        value *= factor
    return priors


def collect_params(model):
    return [], []


def configure_model(model):
    model.eval()
    model.requires_grad_(False)
    return model
