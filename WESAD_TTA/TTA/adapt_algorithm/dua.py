"""DUA-style dynamic unsupervised BatchNorm adaptation for 1D-CNN."""

from copy import deepcopy

import torch
import torch.nn as nn

from TTA.adapt_algorithm.common import set_dropout_eval


class DUA(nn.Module):
    """Update BN running statistics with a decaying test-time momentum."""

    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = configure_model(model, args)
        self.decay_factor = float(getattr(args, "dua_decay_factor", 0.94))
        self.min_momentum = float(getattr(args, "dua_min_momentum", 0.005))
        self.momentum = float(getattr(args, "dua_initial_momentum", 0.1))
        self.model_state = deepcopy(self.model.state_dict())
        self.diagnostics = []
        self._batch_index = 0

    @torch.no_grad()
    def forward(self, x):
        self.momentum = self.momentum * self.decay_factor
        effective_momentum = self.momentum + self.min_momentum
        self.model.train()
        if getattr(self.args, "disable_dropout", False):
            set_dropout_eval(self.model)
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train()
                module.momentum = effective_momentum
                module.track_running_stats = True
        _ = self.model(x)
        self.model.eval()
        outputs = self.model(x)
        self.add_diagnostic(effective_momentum)
        return outputs

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.momentum = float(getattr(self.args, "dua_initial_momentum", 0.1))
        self.diagnostics = []
        self._batch_index = 0

    def add_diagnostic(self, momentum):
        self.diagnostics.append({"batch_index": self._batch_index, "dua_momentum": float(momentum)})
        self._batch_index += 1

    def record_batch_labels(self, y_batch):
        record_batch_labels(self, y_batch)

    def get_diagnostics(self):
        return list(self.diagnostics)


def configure_model(model, args):
    model.eval()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.track_running_stats = True
            module.requires_grad_(False)
    if getattr(args, "disable_dropout", False):
        set_dropout_eval(model)
    return model


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
