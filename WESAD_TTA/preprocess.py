"""WESAD の前処理済みデータを `data/` 配下へ保存するスクリプト。

`adapt.py` のたびに raw `.pkl` を読み直すと時間がかかるため、被験者ごとに
窓分割、二値ラベル化、標準化まで済ませた `.npz` を作成する。
"""

from config import parse_args
from data_processing.wesad import preprocess_all_subjects


def main():
    """設定に従って全被験者の前処理済みデータを作成する。"""
    args = parse_args("Preprocess WESAD data into per-subject npz files.")
    saved_paths = preprocess_all_subjects(args)
    print(f"Saved {len(saved_paths)} processed files.")


if __name__ == "__main__":
    main()
