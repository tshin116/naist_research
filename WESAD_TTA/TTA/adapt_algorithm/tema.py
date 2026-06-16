"""TEMA: Test-time Exponential Moving Average BatchNorm statistics.

This adapts the `norm_ema` / `EMABatchNorm` idea from
`/home/shinsaku-t/work/naist_reserch/RealisticTTA` to WESAD BatchNorm1d models.
For each test batch, the model first performs a train-mode probe forward to
update BN running statistics with EMA, then performs an eval-mode forward using
the updated running statistics. No model parameters are optimized.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from TTA.adapt_algorithm.common import set_dropout_eval


class TEMA(nn.Module):
    """Test-time EMA normalization wrapper."""

    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = configure_model(model, args)
        self.model_state = deepcopy(self.model.state_dict())
        self.diagnostics = []
        self._batch_index = 0

    @torch.no_grad()
    def forward(self, x):
        # Probe pass: update BN running mean/var from the current test batch.
        self.model.train()
        if getattr(self.args, "disable_dropout", False):
            set_dropout_eval(self.model)
        _ = self.model(x)

        # Prediction pass: use the updated EMA BN statistics.
        self.model.eval()
        outputs = self.model(x)
        self.add_diagnostic()
        return outputs

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.diagnostics = []
        self._batch_index = 0

    def add_diagnostic(self):
        self.diagnostics.append({"batch_index": self._batch_index})
        self._batch_index += 1

    def record_batch_labels(self, y_batch):
        if not self.diagnostics:
            return
        labels = y_batch.detach().cpu().reshape(-1).to(torch.long)
        if labels.numel() == 0:
            return
        num_classes = max(int(labels.max().item()) + 1, getattr(self.args, "num_classes", 1))
        counts = torch.bincount(labels, minlength=num_classes).tolist()
        total = float(sum(counts))
        self.diagnostics[-1]["batch_size"] = int(total)
        for class_index, count in enumerate(counts):
            self.diagnostics[-1][f"true_class_{class_index}_count"] = int(count)
            self.diagnostics[-1][f"true_class_{class_index}_ratio"] = float(count / total) if total else 0.0
        self.diagnostics[-1]["true_max_class_ratio"] = float(max(counts) / total) if total else 0.0
        self.diagnostics[-1]["true_num_present_classes"] = int(sum(1 for count in counts if count > 0))

    def get_diagnostics(self):
        return list(self.diagnostics)


def configure_model(model, args):
    model.eval()
    model.requires_grad_(False)
    momentum = float(getattr(args, "tema_momentum", 0.1))
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            module.track_running_stats = True
            module.momentum = momentum
            module.requires_grad_(False)
    if getattr(args, "disable_dropout", False):
        set_dropout_eval(model)
    return model
