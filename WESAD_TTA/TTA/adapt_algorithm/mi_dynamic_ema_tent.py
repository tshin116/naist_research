"""MI-gated Dynamic EMA-Tent for WESAD 1D-CNN.

This method keeps the original EMA-Tent implementation untouched. It replaces
BatchNorm1d with a source / target-EMA / current-batch normalizer and controls
the current-batch contribution, EMA update rate, and entropy update by a mutual
information style gate.
"""

from copy import deepcopy
import math

import torch
import torch.nn as nn

from TTA.adapt_algorithm.common import entropy_from_logits, logits_to_class_logits, set_dropout_eval


class MIDynamicEMABatchNorm1d(nn.Module):
    """BatchNorm1d with dynamic source / EMA / current-batch variance mixing."""

    def __init__(self, layer, source_weight, ema_weight, batch_weight, momentum, eps=None):
        super().__init__()
        total = source_weight + ema_weight + batch_weight
        if total <= 0:
            raise ValueError("MI Dynamic EMA-Tent BN weights must have a positive sum")

        self.layer = layer
        self.layer.eval()
        self.base_source_weight = float(source_weight / total)
        self.base_ema_weight = float(ema_weight / total)
        self.base_batch_weight = float(batch_weight / total)
        self.momentum = float(momentum)
        self.eps = layer.eps if eps is None else eps

        source_mean = layer.running_mean.detach().clone()
        source_var = layer.running_var.detach().clone()
        self.register_buffer("source_mean", source_mean)
        self.register_buffer("source_var", source_var)
        self.register_buffer("ema_mean", source_mean.clone())
        self.register_buffer("ema_var", source_var.clone())
        self.register_buffer("last_batch_mean", source_mean.clone())
        self.register_buffer("last_batch_var", source_var.clone())

        self.source_weight = self.base_source_weight
        self.ema_weight = self.base_ema_weight
        self.batch_weight = self.base_batch_weight

    def set_mixture_weights(self, source_weight, ema_weight, batch_weight):
        total = float(source_weight + ema_weight + batch_weight)
        if total <= 0:
            raise ValueError("MI Dynamic EMA-Tent mixture weights must have a positive sum")
        self.source_weight = float(source_weight / total)
        self.ema_weight = float(ema_weight / total)
        self.batch_weight = float(batch_weight / total)

    def forward(self, x):
        reduce_dims = [0] + list(range(2, x.ndim))
        batch_mean = x.mean(dim=reduce_dims)
        batch_var = x.var(dim=reduce_dims, unbiased=False)
        self.last_batch_mean = batch_mean.detach()
        self.last_batch_var = batch_var.detach()

        source_mean = self.source_mean.to(x.device)
        source_var = self.source_var.to(x.device)
        ema_mean = self.ema_mean.to(x.device)
        ema_var = self.ema_var.to(x.device)

        mixed_mean = (
            self.source_weight * source_mean
            + self.ema_weight * ema_mean
            + self.batch_weight * batch_mean.detach()
        )
        mixed_var = (
            self.source_weight * source_var
            + self.ema_weight * ema_var
            + self.batch_weight * batch_var.detach()
        )
        view_shape = [1, -1] + [1] * (x.ndim - 2)
        normalized = (x - mixed_mean.view(*view_shape)) / torch.sqrt(mixed_var.view(*view_shape) + self.eps)
        return normalized * self.layer.weight.view(*view_shape) + self.layer.bias.view(*view_shape)

    @torch.no_grad()
    def update_ema(self, alpha):
        alpha = float(max(0.0, min(1.0, alpha)))
        device = self.ema_mean.device
        batch_mean = self.last_batch_mean.to(device)
        batch_var = self.last_batch_var.to(device)
        self.ema_mean.mul_(1.0 - alpha).add_(batch_mean, alpha=alpha)
        self.ema_var.mul_(1.0 - alpha).add_(batch_var, alpha=alpha)


class MIDynamicEMATent(nn.Module):
    """Tent wrapper with MI-gated dynamic EMA BatchNorm statistics."""

    def __init__(self, model, optimizer, steps=1, episodic=False, label_mode="binary", args=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.label_mode = label_mode
        self.args = args
        if steps <= 0:
            raise ValueError("mi_dynamic_ema_tent requires at least one adaptation step")

        self.disable_dropout = getattr(args, "disable_dropout", False)
        self.mi_gate = getattr(args, "mi_gate", True)
        self.gate_aware_ema = getattr(args, "gate_aware_ema", True)
        self.dynamic_mixture_weights = getattr(args, "dynamic_mixture_weights", True)
        self.gate_threshold = getattr(args, "gate_threshold", 0.05)
        self.mi_gate_power = float(getattr(args, "mi_gate_power", 1.0))
        self.mi_min_gate = float(getattr(args, "mi_min_gate", 0.0))
        self.loss_gate = getattr(args, "mi_loss_gate", True)
        self.probe_batch_weight = getattr(args, "mi_probe_batch_weight", 0.0)
        self.w_batch_max = getattr(args, "w_batch_max", getattr(args, "ema_tent_batch_weight", 0.3))
        self.rho_k = float(getattr(args, "rho_k", 512.0))
        self.ema_momentum = float(getattr(args, "ema_momentum", getattr(args, "ema_tent_momentum", 0.8)))
        self.eps = float(getattr(args, "mi_gate_eps", 1e-12))
        self.bn_layers = [module for module in model.modules() if isinstance(module, MIDynamicEMABatchNorm1d)]
        self.diagnostics = []
        self._batch_index = 0
        self.n_eff = 0.0

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self)
        return outputs

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.bn_layers = [module for module in self.model.modules() if isinstance(module, MIDynamicEMABatchNorm1d)]
        self.diagnostics = []
        self._batch_index = 0
        self.n_eff = 0.0

    def set_mixture_weights(self, source_weight, ema_weight, batch_weight):
        for layer in self.bn_layers:
            layer.set_mixture_weights(source_weight, ema_weight, batch_weight)

    def compute_fixed_weights(self, gate):
        if not self.bn_layers:
            return 0.0, 0.0, 0.0
        layer = self.bn_layers[0]
        batch_weight = layer.base_batch_weight * gate
        source_weight = layer.base_source_weight + layer.base_batch_weight * (1.0 - gate)
        ema_weight = layer.base_ema_weight
        total = source_weight + ema_weight + batch_weight
        return source_weight / total, ema_weight / total, batch_weight / total

    def compute_dynamic_weights(self, gate, batch_size):
        self.n_eff += float(batch_size) * float(gate)
        rho = 1.0 - math.exp(-self.n_eff / max(self.rho_k, 1e-12))
        batch_weight = float(self.w_batch_max) * float(gate)
        remain = max(0.0, 1.0 - batch_weight)
        ema_weight = remain * rho
        source_weight = remain * (1.0 - rho)
        return source_weight, ema_weight, batch_weight, rho

    @torch.no_grad()
    def update_ema(self, alpha):
        for layer in self.bn_layers:
            layer.update_ema(alpha)

    def add_diagnostic(self, diagnostic):
        diagnostic["batch_index"] = self._batch_index
        self._batch_index += 1
        self.diagnostics.append(diagnostic)

    def record_batch_labels(self, y_batch):
        if not self.diagnostics:
            return
        labels = y_batch.detach().cpu().reshape(-1).to(torch.long)
        if labels.numel() == 0:
            return
        num_classes = max(int(labels.max().item()) + 1, 1)
        if self.label_mode == "3class":
            num_classes = max(num_classes, 3)
        counts = torch.bincount(labels, minlength=num_classes).tolist()
        total = float(sum(counts))
        self.diagnostics[-1]["batch_size"] = int(total)
        max_ratio = float(max(counts) / total) if total else 0.0
        for class_index, count in enumerate(counts):
            self.diagnostics[-1][f"true_class_{class_index}_count"] = int(count)
            self.diagnostics[-1][f"true_class_{class_index}_ratio"] = float(count / total) if total else 0.0
        self.diagnostics[-1]["true_max_class_ratio"] = max_ratio
        self.diagnostics[-1]["true_num_present_classes"] = int(sum(1 for count in counts if count > 0))
        self.diagnostics[-1]["true_imbalance"] = max_ratio - (1.0 / max(num_classes, 1))

    def get_diagnostics(self):
        return list(self.diagnostics)


@torch.enable_grad()
def forward_and_adapt(x, wrapper):
    """Probe gate, score current batch, update BN affine, then update EMA stats."""
    model = wrapper.model
    optimizer = wrapper.optimizer
    model.train()
    if wrapper.disable_dropout:
        set_dropout_eval(model)

    gate_info = estimate_batch_gate(x, wrapper)
    gate = gate_info["g_mi"] if wrapper.mi_gate else gate_info["diversity_norm"]
    gate = max(0.0, min(1.0, gate)) ** wrapper.mi_gate_power
    gate = wrapper.mi_min_gate + (1.0 - wrapper.mi_min_gate) * gate
    gate = float(max(0.0, min(1.0, gate)))

    batch_size = int(x.size(0))
    if wrapper.dynamic_mixture_weights:
        source_weight, ema_weight, batch_weight, rho = wrapper.compute_dynamic_weights(gate, batch_size)
    else:
        source_weight, ema_weight, batch_weight = wrapper.compute_fixed_weights(gate)
        rho = None
    wrapper.set_mixture_weights(source_weight, ema_weight, batch_weight)

    outputs = model(x)
    if isinstance(outputs, tuple):
        outputs, _ = outputs

    class_logits = logits_to_class_logits(outputs, wrapper.label_mode)
    entropy_values = entropy_from_logits(outputs, label_mode=wrapper.label_mode)
    entropy_loss = entropy_values.mean()
    should_update = gate >= wrapper.gate_threshold
    if should_update:
        loss = entropy_loss * gate if wrapper.loss_gate else entropy_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    else:
        optimizer.zero_grad()

    alpha = (1.0 - wrapper.ema_momentum) * gate if wrapper.gate_aware_ema else (1.0 - wrapper.ema_momentum)
    wrapper.update_ema(alpha)

    pred_info = prediction_distribution(class_logits)
    diagnostic = {
        "batch_size": batch_size,
        "g_mi": float(gate_info["g_mi"]),
        "effective_mi_gate": float(gate),
        "diversity_norm": float(gate_info["diversity_norm"]),
        "sample_entropy_norm": float(gate_info["sample_entropy_norm"]),
        "w_source": float(source_weight),
        "w_ema": float(ema_weight),
        "w_batch": float(batch_weight),
        "rho": float(rho) if rho is not None else None,
        "n_eff": float(wrapper.n_eff),
        "alpha": float(alpha),
        "entropy_updated": int(should_update),
        "loss_entropy": float(entropy_loss.detach().item()),
        "gate_threshold": float(wrapper.gate_threshold),
        "gate_aware_ema": int(wrapper.gate_aware_ema),
        "dynamic_mixture_weights": int(wrapper.dynamic_mixture_weights),
        **pred_info,
    }
    wrapper.add_diagnostic(diagnostic)
    return outputs


@torch.no_grad()
def estimate_batch_gate(x, wrapper):
    probe_batch_weight = float(max(0.0, min(1.0, wrapper.probe_batch_weight)))
    wrapper.set_mixture_weights(1.0 - probe_batch_weight, 0.0, probe_batch_weight)
    outputs = wrapper.model(x)
    if isinstance(outputs, tuple):
        outputs, _ = outputs
    class_logits = logits_to_class_logits(outputs, wrapper.label_mode)
    probs = torch.softmax(class_logits, dim=1).clamp_min(wrapper.eps)
    pred_dist = probs.mean(dim=0)
    log_classes = torch.log(torch.tensor(float(probs.size(1)), device=probs.device))
    diversity = -(pred_dist * torch.log(pred_dist)).sum()
    sample_entropy = -(probs * torch.log(probs)).sum(dim=1).mean()
    g_mi = ((diversity - sample_entropy) / log_classes).clamp(0.0, 1.0)
    diversity_norm = (diversity / log_classes).clamp(0.0, 1.0)
    sample_entropy_norm = (sample_entropy / log_classes).clamp(0.0, 1.0)
    return {
        "g_mi": float(g_mi.item()),
        "diversity_norm": float(diversity_norm.item()),
        "sample_entropy_norm": float(sample_entropy_norm.item()),
    }


@torch.no_grad()
def prediction_distribution(class_logits):
    probs = torch.softmax(class_logits, dim=1)
    pred = probs.argmax(dim=1)
    counts = torch.bincount(pred, minlength=probs.size(1)).detach().cpu()
    total = float(max(int(pred.numel()), 1))
    row = {}
    for class_index, count in enumerate(counts.tolist()):
        row[f"pred_class_{class_index}_count"] = int(count)
        row[f"pred_class_{class_index}_ratio"] = float(count / total)
    row["pred_max_class_ratio"] = float(counts.max().item() / total)
    row["pred_num_present_classes"] = int((counts > 0).sum().item())
    return row


def collect_params(model):
    """Collect affine parameters of wrapped MI Dynamic EMA BatchNorm1d layers."""
    params = []
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, MIDynamicEMABatchNorm1d):
            for param_name, param in module.layer.named_parameters():
                if param_name in ["weight", "bias"]:
                    params.append(param)
                    names.append(f"{module_name}.layer.{param_name}")
    return params, names


def configure_model(model, args):
    """Replace BatchNorm1d with MI Dynamic EMA BatchNorm and freeze other params."""
    model.train()
    model.requires_grad_(False)
    source_weight = getattr(args, "ema_tent_source_weight", 0.2)
    ema_weight = getattr(args, "ema_tent_ema_weight", 0.5)
    batch_weight = getattr(args, "ema_tent_batch_weight", 0.3)
    momentum = getattr(args, "ema_momentum", getattr(args, "ema_tent_momentum", 0.8))

    for parent, name, child in find_bns(model):
        setattr(parent, name, MIDynamicEMABatchNorm1d(child, source_weight, ema_weight, batch_weight, momentum))

    for module in model.modules():
        if isinstance(module, MIDynamicEMABatchNorm1d):
            module.layer.requires_grad_(True)
    if getattr(args, "disable_dropout", False):
        set_dropout_eval(model)
    check_model(model)
    return model


def find_bns(model):
    replace_mods = []
    queue = [(model, "")]
    while queue:
        parent, prefix = queue.pop(0)
        for child_name, child in parent.named_children():
            module_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.BatchNorm1d):
                replace_mods.append((parent, child_name, child))
            else:
                queue.append((child, module_name))
    return replace_mods


def check_model(model):
    if not model.training:
        raise AssertionError("mi_dynamic_ema_tent needs train mode")
    param_grads = [param.requires_grad for param in model.parameters()]
    if not any(param_grads):
        raise AssertionError("mi_dynamic_ema_tent needs parameters to update")
    if all(param_grads):
        raise AssertionError("mi_dynamic_ema_tent should not update all parameters")
    if not any(isinstance(module, MIDynamicEMABatchNorm1d) for module in model.modules()):
        raise AssertionError("mi_dynamic_ema_tent needs MIDynamicEMABatchNorm1d layers")


def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
