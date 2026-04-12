from copy import deepcopy

import torch
import torch.nn as nn


class Tent(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        if steps <= 0:
            raise ValueError("tent requires at least one adaptation step")

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)


def binary_entropy_from_logits(logits):
    probs = torch.sigmoid(logits)
    eps = torch.finfo(probs.dtype).eps
    probs = probs.clamp(min=eps, max=1.0 - eps)
    return -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer):
    outputs = model(x)
    loss = binary_entropy_from_logits(outputs).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


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


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    check_model(model)
    return model


def check_model(model):
    if not model.training:
        raise AssertionError("tent needs train mode")

    param_grads = [param.requires_grad for param in model.parameters()]
    if not any(param_grads):
        raise AssertionError("tent needs parameters to update")
    if all(param_grads):
        raise AssertionError("tent should not update all parameters")
    if not any(isinstance(module, nn.BatchNorm1d) for module in model.modules()):
        raise AssertionError("tent needs BatchNorm1d layers")


def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
