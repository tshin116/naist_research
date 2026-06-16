"""DELTA-style TBR + weighted entropy adaptation for 1D-CNN."""

from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from TTA.adapt_algorithm.common import entropy_from_logits, logits_to_class_logits, set_dropout_eval


class TBR1d(nn.Module):
    """Temporal Batch Renormalization layer adapted from DELTA TBR."""

    def __init__(self, layer, prior=0.95):
        super().__init__()
        self.layer = layer
        self.layer.eval()
        self.prior = float(prior)
        self.running_mean = None
        self.running_std = None
        self.eps = layer.eps

    def forward(self, x):
        reduce_dims = [0] + list(range(2, x.ndim))
        batch_mean = x.mean(dim=reduce_dims)
        batch_std = torch.sqrt(x.var(dim=reduce_dims, unbiased=False) + self.eps)
        if self.running_mean is None:
            self.running_mean = batch_mean.detach().clone()
            self.running_std = batch_std.detach().clone()
        r = batch_std.detach() / self.running_std.clamp_min(self.eps)
        d = (batch_mean.detach() - self.running_mean) / self.running_std.clamp_min(self.eps)
        view_shape = [1, -1] + [1] * (x.ndim - 2)
        normalized = (x - batch_mean.view(*view_shape)) / batch_std.view(*view_shape)
        renormed = normalized * r.view(*view_shape) + d.view(*view_shape)
        self.running_mean = self.prior * self.running_mean + (1.0 - self.prior) * batch_mean.detach()
        self.running_std = self.prior * self.running_std + (1.0 - self.prior) * batch_std.detach()
        return renormed * self.layer.weight.view(*view_shape) + self.layer.bias.view(*view_shape)


class DELTA(nn.Module):
    """DELTA-style entropy update with optional distribution balancing."""

    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = configure_model(model, args)
        params, _ = collect_params(self.model)
        self.optimizer = torch.optim.SGD(
            params,
            lr=float(getattr(args, "delta_lr", 2.5e-4)),
            momentum=float(getattr(args, "delta_optim_momentum", 0.9)),
            weight_decay=float(getattr(args, "delta_weight_decay", 0.0)),
        )
        self.label_mode = getattr(args, "label_mode", "binary")
        self.num_classes = int(getattr(args, "num_classes", 2))
        self.ent_w = bool(getattr(args, "delta_entropy_weight", True))
        self.dot = getattr(args, "delta_dot", 0.95)
        self.qhat = None
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
        self.diagnostics = []
        self._batch_index = 0

    @torch.enable_grad()
    def forward(self, x):
        self.model.train()
        if getattr(self.args, "disable_dropout", False):
            set_dropout_eval(self.model)
        outputs = self.model(x)
        class_logits = logits_to_class_logits(outputs, self.label_mode)
        probs = torch.softmax(class_logits, dim=1)
        pred = probs.argmax(dim=1)
        logp = torch.log_softmax(class_logits, dim=1)
        entropies = -(probs * logp).sum(dim=1)
        weights = torch.ones_like(entropies)
        if self.ent_w:
            weights = torch.exp(0.5 * math.log(max(self.num_classes, 2)) - entropies.detach())
        if self.dot is not None:
            if self.qhat is None or self.qhat.device != probs.device:
                self.qhat = torch.full((1, probs.size(1)), 1.0 / probs.size(1), device=probs.device)
            class_weight = 1.0 / self.qhat.clamp_min(1e-6)
            class_weight = class_weight / class_weight.sum()
            sample_weight = class_weight.gather(1, pred.view(1, -1)).squeeze(0)
            sample_weight = sample_weight / sample_weight.sum().clamp_min(1e-6) * len(pred)
            weights = weights * sample_weight
        use_idx = weights > 1.0 if self.ent_w else torch.ones_like(weights, dtype=torch.bool)
        if use_idx.any():
            loss = (entropies[use_idx] * weights[use_idx]).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_value = float(loss.detach().item())
        else:
            self.optimizer.zero_grad()
            loss_value = None
        if self.dot is not None:
            with torch.no_grad():
                self.qhat = float(self.dot) * self.qhat + (1.0 - float(self.dot)) * probs.detach().mean(dim=0, keepdim=True)
        self.add_diagnostic(loss_value, int(use_idx.sum().item()))
        return outputs

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.qhat = None
        self.diagnostics = []
        self._batch_index = 0

    def add_diagnostic(self, loss_value, used_samples):
        self.diagnostics.append(
            {
                "batch_index": self._batch_index,
                "delta_loss": loss_value,
                "delta_used_samples": used_samples,
            }
        )
        self._batch_index += 1

    def record_batch_labels(self, y_batch):
        record_batch_labels(self, y_batch)

    def get_diagnostics(self):
        return list(self.diagnostics)


def configure_model(model, args):
    model.train()
    model.requires_grad_(False)
    prior = float(getattr(args, "delta_old_prior", 0.95))
    for parent, name, child in find_bns(model):
        setattr(parent, name, TBR1d(child, prior=prior))
    for module in model.modules():
        if isinstance(module, TBR1d):
            module.layer.requires_grad_(True)
    if getattr(args, "disable_dropout", False):
        set_dropout_eval(model)
    return model


def find_bns(model):
    replace_mods = []
    queue = [model]
    while queue:
        parent = queue.pop(0)
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm1d):
                replace_mods.append((parent, name, child))
            else:
                queue.append(child)
    return replace_mods


def collect_params(model):
    params = []
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, TBR1d):
            for param_name, param in module.layer.named_parameters():
                if param_name in ["weight", "bias"]:
                    params.append(param)
                    names.append(f"{module_name}.layer.{param_name}")
    return params, names


def record_batch_labels(wrapper, y_batch):
    if not wrapper.diagnostics:
        return
    labels = y_batch.detach().cpu().reshape(-1).to(torch.long)
    if labels.numel() == 0:
        return
    num_classes = max(int(labels.max().item()) + 1, getattr(wrapper.args, "num_classes", 1))
    counts = torch.bincount(labels, minlength=num_classes).tolist()
    total = float(sum(counts))
    wrapper.diagnostics[-1]["batch_size"] = int(total)
    for class_index, count in enumerate(counts):
        wrapper.diagnostics[-1][f"true_class_{class_index}_count"] = int(count)
        wrapper.diagnostics[-1][f"true_class_{class_index}_ratio"] = float(count / total) if total else 0.0
    wrapper.diagnostics[-1]["true_max_class_ratio"] = float(max(counts) / total) if total else 0.0
    wrapper.diagnostics[-1]["true_num_present_classes"] = int(sum(1 for count in counts if count > 0))
