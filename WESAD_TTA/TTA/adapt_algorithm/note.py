"""NOTE-style online memory entropy minimization for 1D-CNN."""

from copy import deepcopy

import torch
import torch.nn as nn

from TTA.adapt_algorithm.common import entropy_from_logits, set_dropout_eval


class NOTE(nn.Module):
    """FIFO-memory online entropy minimization on BN affine parameters."""

    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = configure_model(model, args)
        params, _ = collect_params(self.model)
        self.optimizer = torch.optim.Adam(
            params,
            lr=float(getattr(args, "note_lr", getattr(args, "tent_lr", 1e-4))),
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        self.memory_size = int(getattr(args, "note_memory_size", 64))
        self.update_every = int(getattr(args, "note_update_every", self.memory_size))
        self.epochs = int(getattr(args, "note_epochs", 1))
        self.label_mode = getattr(args, "label_mode", "binary")
        self.memory = []
        self.seen = 0
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
        self.diagnostics = []
        self._batch_index = 0

    def forward(self, x):
        self.model.eval()
        outputs = self.model(x)
        self.add_to_memory(x)
        should_update = self.seen % self.update_every == 0 and len(self.memory) > 1
        loss_value = None
        if should_update:
            loss_value = self.update_model(x.device)
        self.add_diagnostic(should_update, loss_value)
        return outputs

    def add_to_memory(self, x):
        for sample in x.detach():
            if len(self.memory) >= self.memory_size:
                self.memory.pop(0)
            self.memory.append(sample.detach().cpu())
            self.seen += 1

    @torch.enable_grad()
    def update_model(self, device):
        self.model.train()
        if getattr(self.args, "disable_dropout", False):
            set_dropout_eval(self.model)
        data = torch.stack(self.memory).to(device)
        loss_value = 0.0
        for _ in range(self.epochs):
            outputs = self.model(data)
            loss = entropy_from_logits(outputs, label_mode=self.label_mode).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.detach().item())
        self.optimizer.zero_grad()
        return loss_value

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.memory = []
        self.seen = 0
        self.diagnostics = []
        self._batch_index = 0

    def add_diagnostic(self, updated, loss_value):
        self.diagnostics.append(
            {
                "batch_index": self._batch_index,
                "note_memory_occupancy": len(self.memory),
                "note_updated": int(updated),
                "note_loss": loss_value,
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
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            module.requires_grad_(True)
            if not getattr(args, "note_use_learned_stats", False):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
    if getattr(args, "disable_dropout", False):
        set_dropout_eval(model)
    return model


def collect_params(model):
    params = []
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            for param_name, param in module.named_parameters():
                if param_name in ["weight", "bias"]:
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")
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
