"""RoTTA-style robust online TTA for 1D-CNN."""

from copy import deepcopy
import math

import torch
import torch.nn as nn

from TTA.adapt_algorithm.common import logits_to_class_logits, set_dropout_eval, softmax_entropy


class RoTTA(nn.Module):
    """Class-balanced memory with teacher EMA consistency updates."""

    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = configure_model(model, args)
        self.model_ema = deepcopy(self.model)
        self.model_ema.eval()
        self.model_ema.requires_grad_(False)
        params, _ = collect_params(self.model)
        self.optimizer = torch.optim.Adam(
            params,
            lr=float(getattr(args, "rotta_lr", getattr(args, "tent_lr", 1e-4))),
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        self.label_mode = getattr(args, "label_mode", "binary")
        self.num_classes = int(getattr(args, "num_classes", 2))
        self.memory_size = int(getattr(args, "rotta_memory_size", 64))
        self.update_frequency = int(getattr(args, "rotta_update_frequency", self.memory_size))
        self.nu = float(getattr(args, "rotta_ema_nu", 0.001))
        self.lambda_t = float(getattr(args, "rotta_lambda_t", 1.0))
        self.lambda_u = float(getattr(args, "rotta_lambda_u", 1.0))
        self.memory = [[] for _ in range(self.num_classes)]
        self.current_instance = 0
        self.model_state = deepcopy(self.model.state_dict())
        self.ema_state = deepcopy(self.model_ema.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
        self.diagnostics = []
        self._batch_index = 0

    @torch.enable_grad()
    def forward(self, x):
        self.model.eval()
        self.model_ema.eval()
        with torch.no_grad():
            outputs = self.model_ema(x)
            class_logits = logits_to_class_logits(outputs, self.label_mode)
            probs = torch.softmax(class_logits, dim=1)
            pseudo = probs.argmax(dim=1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)

        for sample, pseudo_label, uncertainty in zip(x.detach(), pseudo.detach(), entropy.detach()):
            self.add_memory(sample.cpu(), int(pseudo_label.item()), float(uncertainty.item()))
            self.current_instance += 1
            if self.current_instance % self.update_frequency == 0:
                self.update_model(x.device)

        self.add_diagnostic()
        return outputs

    def add_memory(self, sample, pseudo_label, uncertainty):
        target_class = max(0, min(self.num_classes - 1, pseudo_label))
        per_class = max(1, math.ceil(self.memory_size / self.num_classes))
        item = {"sample": sample, "uncertainty": uncertainty, "age": 0}
        if len(self.memory[target_class]) < per_class:
            self.memory[target_class].append(item)
        else:
            worst_index = max(range(len(self.memory[target_class])), key=lambda i: self.memory_score(self.memory[target_class][i]))
            if self.memory_score(item) < self.memory_score(self.memory[target_class][worst_index]):
                self.memory[target_class][worst_index] = item
        self.age_memory()

    def memory_score(self, item):
        return self.lambda_t / (1.0 + math.exp(-item["age"] / max(self.memory_size, 1))) + self.lambda_u * item["uncertainty"] / math.log(max(self.num_classes, 2))

    def age_memory(self):
        for class_items in self.memory:
            for item in class_items:
                item["age"] += 1

    def memory_samples(self):
        samples = []
        ages = []
        for class_items in self.memory:
            for item in class_items:
                samples.append(item["sample"])
                ages.append(item["age"] / max(self.memory_size, 1))
        return samples, ages

    def update_model(self, device):
        samples, ages = self.memory_samples()
        if len(samples) <= 1:
            return None
        self.model.train()
        self.model_ema.eval()
        if getattr(self.args, "disable_dropout", False):
            set_dropout_eval(self.model)
        data = torch.stack(samples).to(device)
        ages = torch.tensor(ages, device=device, dtype=torch.float32)
        with torch.no_grad():
            teacher_logits = logits_to_class_logits(self.model_ema(data), self.label_mode)
        student_logits = logits_to_class_logits(self.model(data), self.label_mode)
        teacher_probs = torch.softmax(teacher_logits, dim=1)
        log_student = torch.log_softmax(student_logits, dim=1)
        loss_values = -(teacher_probs * log_student).sum(dim=1)
        weights = torch.exp(-ages) / (1.0 + torch.exp(-ages))
        loss = (loss_values * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_ema()
        return float(loss.detach().item())

    @torch.no_grad()
    def update_ema(self):
        for ema_param, param in zip(self.model_ema.parameters(), self.model.parameters()):
            ema_param.data.mul_(1.0 - self.nu).add_(param.data, alpha=self.nu)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.model_ema.load_state_dict(self.ema_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.memory = [[] for _ in range(self.num_classes)]
        self.current_instance = 0
        self.diagnostics = []
        self._batch_index = 0

    def add_diagnostic(self):
        self.diagnostics.append(
            {
                "batch_index": self._batch_index,
                "rotta_memory_occupancy": sum(len(items) for items in self.memory),
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
