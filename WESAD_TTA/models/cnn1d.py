"""WESAD 分類用の 1D-CNN モデル定義。"""

import torch.nn as nn


class StressCNN1D(nn.Module):
    """胸部センサ 8 チャンネル時系列を入力する 1D-CNN。

    入力形状は `(batch, channels, time)`。二値分類では既存 checkpoint との互換性を
    保つため single-logit を返し、3分類では `(batch, num_classes)` の logits を返す。
    """

    def __init__(self, num_channels=8, num_classes=2, label_mode="binary"):
        super().__init__()
        self.num_classes = num_classes
        self.label_mode = label_mode
        output_dim = 1 if label_mode == "binary" else num_classes
        # 時間方向に Conv1d を重ね、BatchNorm1d を Tent の適応対象として残す。
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),
        )
        # Global Average Pooling で時間方向を集約し、分類 logits を出力する。
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
        )

    def forward_features(self, x_input):
        """TTA 手法で使う 64 次元 feature を返す。

        通常の forward と同じ畳み込み特徴を使い、最後の分類層の直前までを
        feature として取り出す。既存 checkpoint と互換性を保つため、層の定義自体は
        変更せず `classifier` 内のモジュールを再利用する。
        """
        features = self.features(x_input)
        features = self.classifier[0](features)
        features = self.classifier[1](features)
        features = self.classifier[2](features)
        features = self.classifier[3](features)
        features = self.classifier[4](features)
        return features

    def classify_features(self, features):
        """64 次元 feature から分類 logits を計算する。"""
        features = self.classifier[5](features)
        logits = self.classifier[6](features)
        if self.label_mode == "binary":
            logits = logits.squeeze(-1)
        return logits

    def forward(self, x_input, return_feature=False):
        """Conv1d 特徴抽出器と分類器を通して logits を返す。"""
        features = self.forward_features(x_input)
        logits = self.classify_features(features)
        if return_feature:
            return logits, features
        return logits
