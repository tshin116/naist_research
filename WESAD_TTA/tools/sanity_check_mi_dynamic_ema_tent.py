"""Sanity checks for MI-gated Dynamic EMA-Tent formulas."""

import math
import sys

import torch


def mi_gate_from_probs(probs, eps=1e-12):
    probs = probs.clamp_min(eps)
    pred_dist = probs.mean(dim=0)
    log_classes = torch.log(torch.tensor(float(probs.size(1)), device=probs.device))
    diversity = -(pred_dist * torch.log(pred_dist)).sum()
    sample_entropy = -(probs * torch.log(probs)).sum(dim=1).mean()
    return ((diversity - sample_entropy) / log_classes).clamp(0.0, 1.0)


def dynamic_weights(gate, batch_size=64, n_eff_prev=0.0, w_batch_max=0.3, rho_k=512):
    n_eff = n_eff_prev + batch_size * gate
    rho = 1.0 - math.exp(-n_eff / rho_k)
    w_batch = w_batch_max * gate
    remain = 1.0 - w_batch
    w_ema = remain * rho
    w_source = remain * (1.0 - rho)
    return w_source, w_ema, w_batch


def assert_close(name, value, expected, tol=1e-6):
    if abs(value - expected) > tol:
        raise AssertionError(f"{name}: expected {expected}, got {value}")


def main():
    uniform = torch.full((8, 3), 1.0 / 3.0)
    g_uniform = float(mi_gate_from_probs(uniform))
    assert 0.0 <= g_uniform <= 1.0
    assert g_uniform < 1e-5

    confident_diverse = torch.eye(3).repeat(4, 1)
    g_diverse = float(mi_gate_from_probs(confident_diverse))
    assert 0.0 <= g_diverse <= 1.0
    assert g_diverse > 0.99

    single_class = torch.tensor([[0.98, 0.01, 0.01]]).repeat(12, 1)
    g_single = float(mi_gate_from_probs(single_class))
    assert 0.0 <= g_single <= 1.0
    assert g_single < 1e-5

    g_low = 0.0
    alpha_low = (1.0 - 0.8) * g_low
    w_source, w_ema, w_batch = dynamic_weights(g_low)
    assert_close("alpha_low", alpha_low, 0.0)
    assert_close("w_batch_low", w_batch, 0.0)
    assert_close("weight_sum_low", w_source + w_ema + w_batch, 1.0)

    g_mid = 0.5
    w_source, w_ema, w_batch = dynamic_weights(g_mid)
    assert_close("weight_sum_mid", w_source + w_ema + w_batch, 1.0)
    assert w_batch > 0.0

    print("MI Dynamic EMA-Tent sanity checks passed.")


if __name__ == "__main__":
    sys.exit(main())
