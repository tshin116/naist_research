import json
import os

file_path = 'self_learning_wesad/stress-detection-with-1d-cnn-99-accuracy.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def add_comments(source):
    commented = []
    for line in source:
        clean_line = line.strip()
        if not clean_line or clean_line.startswith('#'):
            commented.append(line)
            continue
        
        # Simple heuristics for comments based on keywords
        comment = ""
        if "import " in line: comment = "ライブラリのインポート"
        elif "DATASET_DIR =" in line: comment = "データセットのディレクトリパス"
        elif "WINDOW_SECONDS =" in line: comment = "ウィンドウの秒数"
        elif "FS =" in line: comment = "サンプリング周波数"
        elif "WINDOW_SIZE =" in line: comment = "ウィンドウサイズ（サンプル数）"
        elif "NUM_CHANNELS =" in line: comment = "入力チャンネル数"
        elif "wesad_data =" in line: comment = "データを格納する辞書の初期化"
        elif "for subj in os.listdir" in line: comment = "ディレクトリ内の各被験者をループ処理"
        elif "subj_path =" in line: comment = "被験者データのパス作成"
        elif "if not os.path.isdir" in line: comment = "ディレクトリでない場合はスキップ"
        elif "for fname in os.listdir" in line: comment = "フォルダ内のファイルをループ処理"
        elif "if fname.endswith('.pkl')" in line: comment = ".pklファイルを確認"
        elif "with open" in line: comment = "ファイルを開く"
        elif "pickle.load" in line: comment = "データを読み込む"
        elif "wesad_data[subj] =" in line: comment = "辞書にデータを保存"
        elif "data_list = []" in line: comment = "データフレーム用リストの初期化"
        elif "chest =" in line: comment = "胸部センサー信号の抽出"
        elif "labels =" in line: comment = "ラベルデータの抽出"
        elif "pd.DataFrame" in line: comment = "データフレームの作成"
        elif "data_list.append" in line: comment = "リストにデータを追加"
        elif "pd.concat" in line: comment = "全被験者のデータを結合"
        elif "df_filtered =" in line: comment = "特定のラベルのみを抽出"
        elif "label_map =" in line: comment = "バイナリ分類用のラベルマッピング"
        elif "map(label_map)" in line: comment = "マッピングの適用"
        elif "drop(columns='Label')" in line: comment = "古いラベル列を削除"
        elif "plt.figure" in line: comment = "図の作成"
        elif "sns.countplot" in line: comment = "各クラスの件数を可視化"
        elif "plt.plot" in line: comment = "信号のプロット"
        elif "np.fft.rfft" in line: comment = "高速フーリエ変換の実行"
        elif "np.fft.rfftfreq" in line: comment = "周波数軸の計算"
        elif "def segment_data_raw" in line: comment = "セグメンテーション関数の定義"
        elif "def get_window" in line: comment = "ウィンドウ取得関数の定義"
        elif "StandardScaler().fit_transform" in line: comment = "データの標準化"
        elif "train_test_split" in line: comment = "学習用とテスト用に分割"
        elif "def build_1d_cnn_binary" in line: comment = "1D-CNNモデルの構築関数"
        elif "Conv1D" in line: comment = "1次元畳み込み層"
        elif "BatchNormalization" in line: comment = "バッチ正規化層"
        elif "MaxPooling1D" in line: comment = "プーリング層"
        elif "GlobalAveragePooling1D" in line: comment = "グローバル平均プーリング層"
        elif "Dropout" in line: comment = "ドロップアウト層"
        elif "Dense" in line: comment = "全結合層"
        elif "model.compile" in line: comment = "モデルのコンパイル"
        elif "model.fit" in line: comment = "モデルの学習開始"
        elif "f1_score" in line: comment = "F1スコアの計算"
        elif "model.save" in line: comment = "モデルの保存"
        elif "roc_curve" in line: comment = "ROC曲線の計算"
        elif "auc(fpr, tpr)" in line: comment = "AUCの計算"
        
        # Append comment to line
        if comment:
            if line.endswith('\n'):
                commented.append(line[:-1] + " # " + comment + "\n")
            else:
                commented.append(line + " # " + comment)
        else:
            commented.append(line)
    return commented

# Detailed manual comments for key cells to ensure quality
# Cell 2 (Imports)
nb['cells'][1]['source'] = [
    "import os # OS操作ライブラリ\n",
    "import pickle # オブジェクト保存・読込ライブラリ\n",
    "import numpy as np # 数値計算ライブラリ\n",
    "import pandas as pd # データ分析ライブラリ\n",
    "from scipy import stats # 統計ライブラリ\n",
    "from sklearn.model_selection import train_test_split # データ分割用\n",
    "from sklearn.preprocessing import StandardScaler # 標準化用\n",
    "from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc # 評価指標計算用\n",
    "import matplotlib.pyplot as plt # 描画ライブラリ\n",
    "import seaborn as sns # 描画ライブラリ\n",
    "import tensorflow as tf # 深層学習ライブラリ\n",
    "from tensorflow.keras import Sequential # 逐次モデル用\n",
    "from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense, Input, GlobalAveragePooling1D # CNNレイヤー\n",
    "from tensorflow.keras.optimizers import Adam # 最適化アルゴリズム\n",
    "from tensorflow.keras.callbacks import EarlyStopping # 早期終了用\n",
    "from sklearn.utils import resample, shuffle # サンプリング・シャッフル用\n",
    "from sklearn.utils.class_weight import compute_class_weight # クラス重み計算用\n",
    "\n",
    "DATASET_DIR = r\"naist_reserch/self_learning_wesad/WESAD\" # データセットのディレクトリ\n",
    "WINDOW_SECONDS = 5 # 切り出すウィンドウの長さ（秒）\n",
    "FS = 700 # サンプリング周波数 (Hz)\n",
    "WINDOW_SIZE = WINDOW_SECONDS * FS # 1ウィンドウあたりのサンプル数\n",
    "NUM_CHANNELS = 8 # 特徴量の数 (ECG, EDA, EMG, Resp, Temp, ACC x,y,z)\n"
]

# Cell 4 (Loading)
nb['cells'][3]['source'] = [
    "wesad_data = {} # データを格納する辞書を初期化\n",
    "for subj in os.listdir(DATASET_DIR): # 各被験者ディレクトリに対してループ\n",
    "    subj_path = os.path.join(DATASET_DIR, subj) # ディレクトリパスを作成\n",
    "    if not os.path.isdir(subj_path): # ディレクトリでない場合はスキップ\n",
    "        continue\n",
    "    for fname in os.listdir(subj_path): # ディレクトリ内のファイルをループ\n",
    "        if fname.endswith('.pkl'): # .pklファイルを探す\n",
    "            with open(os.path.join(subj_path, fname), 'rb') as f: # ファイルを読込モードで開く\n",
    "                data = pickle.load(f, encoding='latin1') # データをロード\n",
    "            wesad_data[subj] = data # 辞書に格納\n",
    "\n",
    "print(\"Loaded subjects:\", list(wesad_data.keys())) # ロードされた被験者リストを表示\n"
]

# Cell 6 (DataFrame)
nb['cells'][5]['source'] = [
    "data_list = [] # 各被験者のデータを格納するリスト\n",
    "for subj, data in wesad_data.items(): # 辞書の各被験者データをループ\n",
    "    chest = data['signal']['chest'] # 胸部センサーのデータを取得\n",
    "    labels = data['label'].flatten() # ラベルを1次元配列に変換\n",
    "    n = len(chest['ECG'].flatten()) # データ長を取得\n",
    "    df_ch = pd.DataFrame({ # データフレームを作成\n",
    "        'ECG':   chest['ECG'].flatten(), # 心電図\n",
    "        'EDA':   chest['EDA'].flatten(), # 皮膚電気活動\n",
    "        'EMG':   chest['EMG'].flatten(), # 筋電図\n",
    "        'TEMP':  chest['Temp'].flatten(), # 温度\n",
    "        'RESP':  chest['Resp'].flatten(), # 呼吸\n",
    "        'ACC_X': chest['ACC'][:, 0].flatten(), # 加速度X\n",
    "        'ACC_Y': chest['ACC'][:, 1].flatten(), # 加速度Y\n",
    "        'ACC_Z': chest['ACC'][:, 2].flatten(), # 加速度Z\n",
    "        'Label': labels[:n] # ラベル\n",
    "    })\n",
    "    df_ch['Subject'] = subj # 被験者IDを列として追加\n",
    "    data_list.append(df_ch) # リストに追加\n",
    "\n",
    "df = pd.concat(data_list, ignore_index=True) # 全データを1つのデータフレームに統合\n",
    "df.head() # データの先頭を表示\n"
]

# Process other code cells using the generic function
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and cell['source'] and not cell['source'][0].startswith('#'):
        # Skip cells we already processed manually
        if "import os" in cell['source'][0] or "wesad_data =" in cell['source'][0] or "data_list = []" in cell['source'][0]:
            continue
        cell['source'] = add_comments(cell['source'])

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)
