"""実験設定を読み込むための小さなユーティリティ。

このプロジェクトでは、OFTTA と同じように「共通設定」「データセット設定」
「アルゴリズム設定」を YAML に分けて管理する。`parse_args` はそれらを
順番に読み込み、最後にコマンドライン引数で指定された値を上書きする。
"""

import argparse
from types import SimpleNamespace

import yaml


def load_yaml(path):
    """YAML ファイルを辞書として読み込む。

    空の YAML ファイルでも後続処理が壊れないよう、`None` ではなく空辞書を返す。
    """
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args(description):
    """設定ファイルとコマンドライン引数を統合して返す。

    読み込み順は `default_cfg`、`dataset_cfg`、`algorithm_cfg` の順で、後から
    読んだ設定が前の設定を上書きする。さらに `--target_domain` など実行時に
    変えたい値は、コマンドライン引数で上書きできる。
    """
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
