import argparse
from types import SimpleNamespace

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset_cfg", type=str, default="./cfg/dataset/wesad.yaml")
    parser.add_argument("--algorithm_cfg", type=str, default="./cfg/algorithm/source.yaml")
    parser.add_argument("--default_cfg", type=str, default="./cfg/default.yaml")
    parser.add_argument("--target_domain", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    cli_args = parser.parse_args()

    cfg = {}
    cfg.update(load_yaml(cli_args.default_cfg))
    cfg.update(load_yaml(cli_args.dataset_cfg))
    cfg.update(load_yaml(cli_args.algorithm_cfg))

    for key in ["target_domain", "out_path", "resume", "seed"]:
        value = getattr(cli_args, key)
        if value is not None:
            cfg[key] = value

    return SimpleNamespace(**cfg)
