import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

# パスの設定
current_dir = os.path.dirname(os.path.abspath(__file__))
tent_dir = os.path.abspath(os.path.join(current_dir, '../tent'))
sys.path.append(tent_dir)
sys.path.append(current_dir)

# 提供いただいたTentのプログラムをインポート
# （提供されたコードを tent.py として同じディレクトリに保存してください）
import tent 

def get_dataloader(batch_size=64):
    """
    評価用データローダーの準備
    ※ご自身のデータセットに合わせて書き換えてください。
    ※Tentはバッチ内のエントロピーを計算するため、batch_sizeは1より大きい必要があります。
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 例としてCIFAR10のテストセットを使用（本来はターゲットドメインのデータを使用します）
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return val_loader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. モデルの準備
    # ※ご自身の学習済みモデルに書き換えてください。
    model = models.resnet18(pretrained=True)
    model = model.to(device)

    # 2. Tent用のモデル設定
    # BatchNormを更新可能にし、統計量をバッチごとに計算するよう設定します
    model = tent.configure_model(model)
    
    # 3. 更新対象のパラメータ（BatchNormのScale/Shift）を収集
    params, param_names = tent.collect_params(model)
    
    # 4. テスト時適応用のオプティマイザを設定
    # Tentの論文では、Adam(lr=1e-3)などが推奨されています
    optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)

    # 5. モデルをTentでラップする
    # stepsは1回の入力に対して何回更新を行うか（通常は1）
    # episodic=Trueにすると、バッチごとにモデルの状態をリセットします（継続的適応の場合はFalse）
    tented_model = tent.Tent(model, optimizer, steps=1, episodic=False)

    # 6. 評価データローダーの取得
    val_loader = get_dataloader(batch_size=64)

    # ==========================================
    # 推論（テスト時適応）ループ
    # ==========================================
    correct = 0
    total = 0
    
    print("Starting Test-Time Adaptation with Tent...")
    
    # 【重要】通常の推論では with torch.no_grad(): を使いますが、
    # Tentは内部で backward() を呼ぶため、ここではあえて使いません。
    # （Tent側の @torch.enable_grad() でも保護されています）
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 順伝播と同時に、エントロピー最小化によるパラメータ更新（Adaptation）が行われます
        outputs = tented_model(inputs)

        # 予測精度の計算
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 10 == 0:
            current_acc = 100. * correct / total
            print(f"Batch: {batch_idx + 1}/{len(val_loader)} | Current Accuracy: {current_acc:.2f}%")

    final_acc = 100. * correct / total
    print(f"\nFinal Adapted Accuracy: {final_acc:.2f}%")

if __name__ == '__main__':
    main()