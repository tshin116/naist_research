# EMA-Tent: 論文用まとめ

## 1. 研究の流れ

本研究では，感情推定の個人適応タスクに対して，学習時にtarget個人のデータを使うdomain adaptationや，未知個人に頑健な特徴を事前に学習するdomain generalizationではなく，test-time adaptation (TTA) を用いる方針を検討した。

TTAは，source modelを未知target個人のtest stream上で逐次的に適応させる枠組みである。個人差が大きい生体信号ベースの感情推定では，target個人ごとに追加学習データを収集せず，test時の入力だけから個人差へ適応できる可能性がある。そのため，本研究ではまず，代表的なTTA手法であるTentとOFTTAをWESADの個人別感情推定へ適用できるかを検証した。

その結果，TentやOFTTAはそのままでは安定に使えないことが分かった。特に，WESADのtarget streamには状態ごとのブロック構造があり，通常順序でtest batchを作成するとbatch内クラス分布が大きく偏る。このbatch composition biasにより，current test batchに強く依存するTTA手法が不安定化することが確認された。

そこで本研究では，この問題を解決するために，Tentを拡張したEMA-Tentを提案する。EMA-Tentは，BatchNorm統計をcurrent batchのみに依存させず，source統計，過去test batchのEMA統計，current batch統計を混合することで，ブロック構造を持つtarget streamに対して安定なTTAを行う。

論文では以下の順番で記述すると自然である。

1. 感情推定の個人適応にTTAを使う動機を述べる。
2. t-SNE可視化により，同一感情ラベル内でも被験者ごとに特徴分布が異なることを示す。
3. 代表的TTA手法であるTentとOFTTAを1D-CNNベースのWESAD感情推定へ適用する。
4. 予備実験として，通常順序評価とshuffle評価を比較し，既存TTAがbatch compositionに依存することを示す。
5. batch composition依存の原因として，WESADのブロック構造とbatch内クラス偏りを分析する。
6. この問題に対してEMA-Tentを提案する。
7. EMA-Tentが通常順序条件でTentの破綻を緩和し，shuffle gainを大きく低減することを示す。

## 2. 生体情報における個人差の可視化

### 2.1 目的

生体信号に基づく感情推定では，同じ感情ラベルであっても被験者ごとに信号分布が大きく異なる可能性がある。この個人差は，未知被験者に対する性能低下の一因となり，個人適応の必要性を生む。

そこで，WESADのwindow-level特徴をt-SNEで可視化し，感情ラベル内における被験者差を確認した。この可視化は，TTAを感情推定の個人適応に用いる動機づけとして位置づける。

### 2.2 可視化方法

LOSOで学習した1D-CNN source modelの中間特徴を用いた。具体的には，`S2` foldのcheckpointを固定のfeature extractorとして用い，全被験者のwindowを同一の64次元特徴空間へ写像した。その後，t-SNEにより2次元へ投影した。

各被験者・各ラベルから最大40 windowをサンプリングし，合計1800点を可視化した。

| 項目 | 設定 |
|---|---|
| Feature extractor | LOSO-trained 1D-CNN |
| Checkpoint fold | `S2` |
| Feature dimension | 64 |
| Projection | t-SNE |
| Samples | 40 windows per subject and label |
| Total points | 1800 |

出力先は以下である。

```text
logs/wesad/subject_tsne/260422_223445
```

### 2.3 解釈

論文では，ラベルごとに分けた被験者色付きt-SNEを主に用いる。

```text
logs/wesad/subject_tsne/260422_223445/tsne_by_label_subject_color.png
```

この図では，baseline，stress，amusementの各ラベル内で点を表示し，色で被験者IDを表す。同一感情ラベル内でも被験者ごとに分布が分かれる傾向が見られる場合，生体信号特徴には感情ラベルだけでなく個人固有の成分が強く反映されていることを示唆する。

ただし，t-SNEは定性的な可視化手法であり，距離やクラスタ分離を定量的な証明として扱うべきではない。本研究では，生体情報の個人差が大きいことを示す補助的な可視化として用い，その後のLOSO評価およびTTA実験で定量的な議論を行う。

論文中では以下のように書ける。

```text
To motivate personalized adaptation, we visualized 1D-CNN intermediate features using t-SNE. Even within the same emotion label, samples from different subjects tended to form separated clusters, suggesting that subject-specific characteristics remain strongly reflected in the learned physiological feature space.
```

## 3. 予備実験: TTAは感情推定の個人適応にそのまま使えるか

### 3.1 予備実験の目的

予備実験の目的は，既存TTA手法がWESADの個人別感情推定にそのまま有効かを確認することである。

本研究では，source modelをleave-one-subject-out (LOSO) で学習し，未知のtarget被験者に対してtest-time adaptationを行った。TTA手法として，BatchNorm affine parameterをentropy minimizationで更新するTentと，weighted BatchNormおよびclassifier adjustmentを行うOFTTAを適用した。

この段階では，提案手法を導入する前に，以下を確認する。

- 既存TTA手法はWESADの個人差に対して有効か。
- 通常順序のtarget streamで安定に動作するか。
- もし不安定な場合，その原因は何か。

### 3.2 実験設定

データセットはWESADを用いた。分類タスクは以下の3クラス感情推定である。

| Class | 状態 |
|---|---|
| 0 | baseline |
| 1 | stress |
| 2 | amusement |

入力は胸部センサの8チャンネル時系列である。各windowは5秒，sampling rateは700 Hzであり，1 windowあたりの長さは3500 sampleである。

| 項目 | 設定 |
|---|---|
| Dataset | WESAD |
| Task | baseline / stress / amusement の3分類 |
| Window length | 5 seconds |
| Sampling rate | 700 Hz |
| Window size | 3500 |
| Input channels | 8 |
| Batch size | 64 |
| Target subjects | S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17 |
| Evaluation protocol | LOSO |

前処理済みデータは `data/wesad/processed_3class` に保存し，設定ファイルは `cfg/dataset/wesad_3class.yaml` を用いた。

評価指標にはAccuracyとMacro-F1を用いた。Macro-F1は，各クラスのF1-scoreを単純平均した値である。本研究の3分類設定では，baseline，stress，amusementの各F1-scoreを平均する。

```text
Macro-F1 = (F1_baseline + F1_stress + F1_amusement) / 3
```

クラス不均衡を含む感情推定ではAccuracyのみでは少数クラスの性能低下を見落とす可能性があるため，主な評価指標としてMacro-F1を用いる。

### 3.3 1D-CNN source model

source modelにはWESAD用の1D-CNNを用いた。入力形状は `(batch, channels, time)` であり，8チャンネル胸部センサ時系列を入力する。

モデルはConv1d，ReLU，BatchNorm1d，MaxPool1dを3段重ね，Global Average Pooling後に全結合層で3クラスlogitsを出力する。

| Block | 内容 |
|---|---|
| Conv block 1 | Conv1d(8, 32, kernel=7), ReLU, BatchNorm1d, MaxPool1d |
| Conv block 2 | Conv1d(32, 64, kernel=5), ReLU, BatchNorm1d, MaxPool1d |
| Conv block 3 | Conv1d(64, 128, kernel=3), ReLU, BatchNorm1d, MaxPool1d |
| Classifier | AdaptiveAvgPool1d, Linear(128, 64), ReLU, Linear(64, 3) |

BatchNorm1d層はTentおよびEMA-Tentの適応対象として残している。source modelの学習は最大20 epoch，learning rate 0.0005，early stopping patience 3で行った。

### 3.4 LOSO交差検証

評価はLOSOで行った。各foldでは1人の被験者をtarget subjectとして除外し，残りの被験者でsource modelを学習する。評価時にはtarget subjectのデータのみを用い，source評価またはTTA評価を行う。

本研究のTTA評価では，target subjectのtest streamをbatch size 64で逐次的に処理する。通常順序条件では，target windowを保存順にbatch化する。これはオンライン評価に近い条件であり，未来のwindowを混ぜてbatchを作らない。

### 3.5 TTA手法の1D-CNN対応化

既存のTentやOFTTA実装は画像分類向けのBatchNorm2dを前提とする場合が多い。本研究では，WESADの1D-CNNに適用するため，BatchNorm1dを対象としてTTAを動作させた。

Tentでは，BatchNorm1dのaffine parameter，すなわち`weight`と`bias`のみを更新対象とし，予測entropyを最小化する。設定は以下である。

| 項目 | 値 |
|---|---:|
| `tent_lr` | 0.0001 |
| `tent_steps` | 1 |
| `episodic` | false |

OFTTAでは，weighted BatchNormとclassifier adjustmentを1D-CNNのfeatureに対して動作するように移植した。設定は以下である。

| 項目 | 値 |
|---|---:|
| `tta_lr` | 0.0001 |
| `tta_steps` | 1 |
| `filter_K` | 16 |
| `episodic` | false |

TentとOFTTAはいずれも，target streamを逐次処理し，過去batchでの更新が次batch以降に反映される設定とした。

## 4. 予備実験結果: 既存TTAはそのままでは不安定

通常順序条件におけるSource，Tent，OFTTAの結果は以下である。

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Source | 0.6915 | 0.5285 |
| Tent | 0.4553 | 0.3970 |
| OFTTA | 0.5231 | 0.4499 |

TentはSourceよりMacro-F1が低下し，OFTTAもSourceを下回った。したがって，代表的TTA手法をWESADの感情推定にそのまま適用しても，安定な個人適応は得られなかった。

この結果から，単にTTAを導入するだけでは不十分であり，WESADのtarget stream特有の構造に対応する必要があることが分かる。

## 5. ブロック構造が問題であると判断した理由

### 5.1 Sequential vs Shuffleによる診断

既存TTAが不安定になる原因を調べるため，target loaderのbatch構成を変更した。比較した条件は以下である。

| 条件 | 内容 |
|---|---|
| Sequential | target被験者のwindowを保存順にbatch化する |
| Shuffle | target被験者内のwindowをshuffleしてbatch化する |

ここで，Shuffle条件は実運用可能なオンラインTTAではない。target系列全体を事前に混ぜる必要があるため，実際のオンライン評価では使えない。本研究では，shuffleをbatch composition依存性を診断するための条件として用いる。

各手法について，以下を計算した。

```text
Shuffle gain = Shuffle Macro-F1 - Sequential Macro-F1
```

Shuffle gainが大きい場合，通常順序のbatch構成では性能が低く，batchを混ぜることで性能が回復したことを意味する。したがって，Shuffle gainはbatch composition依存性の指標として使える。

### 5.2 Sequential vs Shuffleの結果

| Method | Sequential Macro-F1 | Shuffle Macro-F1 | Shuffle gain |
|---|---:|---:|---:|
| Source | 0.5285 | 0.5285 | +0.0000 |
| Tent | 0.3970 | 0.6285 | +0.2315 |
| OFTTA | 0.4499 | 0.6407 | +0.1908 |

Tentはshuffle条件でMacro-F1が+0.2315改善した。OFTTAも+0.1908改善した。Sourceは適応を行わないため，shuffleによる変化はない。

この結果は，TentとOFTTAの性能低下が，単にモデル能力の不足ではなく，通常順序で形成されるtest batchの構成に強く依存していることを示す。特に，shuffleによってbatch内のクラス構成が混ざると性能が大きく回復するため，WESADのブロック構造によるbatch composition biasがTTAの破綻要因であると考えられる。

### 5.3 Batch biasの定量化

通常順序のtarget batchについて，batch内クラス分布の偏りを集計した。主な指標は以下である。

| 指標 | 意味 |
|---|---|
| `mean_max_class_ratio` | 各batchで最も多いクラスの割合の平均 |
| `mean_imbalance` | batch内クラス分布の偏り |
| `single_class_batch_ratio` | 単一クラスのみで構成されたbatchの割合 |
| `mean_missing_classes` | batch内に存在しないクラス数の平均 |

各被験者はおおむね7から8 batchで評価され，通常順序では多くのbatchが特定クラスに大きく偏っていた。例えば，多くの被験者で`single_class_batch_ratio`は約0.7143であり，7 batch中5 batch程度が単一クラスbatchであることを意味する。

さらに，batch bias指標とShuffle gainの相関を確認した。

| Target | Bias metric | Pearson r | Spearman rho |
|---|---|---:|---:|
| Tent shuffle gain | mean_max_class_ratio | 0.4057 | 0.4933 |
| Tent shuffle gain | mean_imbalance | 0.4622 | 0.4929 |
| OFTTA shuffle gain | mean_max_class_ratio | 0.4922 | 0.5433 |
| OFTTA shuffle gain | mean_imbalance | 0.6202 | 0.5500 |

TentとOFTTAでは，batch偏りが大きい被験者ほどshuffleによる改善が大きい傾向が見られた。これは，WESADのブロック構造とbatch composition biasが，既存TTAの性能変動に関与していることを補助的に示している。

ただし，被験者数は15であるため，相関分析は因果を証明するものではなく，Sequential vs Shuffleの結果を補強する分析として位置づける。

## 6. 提案手法: EMA-Tent

### 6.1 設計方針

予備実験から，WESADの通常順序評価ではcurrent batchがtarget分布全体を代表しないことが分かった。したがって，current batch statisticsのみに依存するTentは，ブロック構造を持つtarget streamで不安定になる。

EMA-Tentは，Tentの枠組みを維持しつつ，BatchNorm統計をcurrent batchのみに依存させないように拡張する。具体的には，以下の3種類の統計を混合してBatchNormを行う。

| 統計 | 役割 |
|---|---|
| Source statistics | 学習時分布に基づくanchor |
| EMA target statistics | 過去test batchから蓄積したtarget個人の統計 |
| Current batch statistics | 現在batchへの局所適応 |

これにより，current batchが単一クラスに偏っていても，source統計と過去target統計が残るため，BN統計の急激な崩壊を抑える。

この3成分を分ける理由は，それぞれが異なる失敗モードに対応しているためである。current batch statistics だけでは，時系列ブロックや単一クラスbatchに過剰適応しやすい。一方で，source statistics だけに戻すと，未知被験者の個人差を取り込めない。また，過去target統計だけを使うと，初期の偏ったbatchや過去の状態に引きずられる危険がある。

そのため，EMA-Tentでは3つを次のように役割分担する。

1. **Source statistics**: target streamがどれだけ偏っても失われない最も保守的なanchorであり，分布の暴走を防ぐ。
2. **EMA target statistics**: 単一batchではなく，過去に観測したtarget被験者全体の傾向を蓄積する長期記憶として働く。
3. **Current batch statistics**: 現在の状態や短期的なtarget変化に追従する局所適応成分として働く。

特に，sourceとEMAを分ける理由は，**「学習時分布へのアンカー」と「test時に観測されたtarget個人の蓄積情報」を区別するため**である。EMAはsource統計で初期化されるが，test streamを処理するにつれてtarget個人固有の統計へ移動していく。もしsourceとEMAを1つにまとめてしまうと，どこまでが学習時の安全なpriorで，どこからがtarget個人へ適応した結果なのかが曖昧になり，偏った初期batchに引きずられた場合に元へ戻る基準が弱くなる。

したがって，EMA-Tentの3成分混合は，単なる経験的な補間ではなく，**source anchor，target long-term memory，current local adaptation** を明示的に分離する設計である。

### 6.2 Tentとの違い

通常Tentは以下を行う。

1. BatchNorm1dをcurrent test batch statisticsで動かす。
2. 予測entropyを最小化する。
3. BatchNorm1dのaffine parameterのみを更新する。

EMA-Tentは，entropy minimizationとBN affine更新というTentの基本方針は維持する。一方で，BatchNorm統計を以下の混合統計に置き換える。

```text
source running statistics
+ past test EMA statistics
+ current batch statistics
```

さらに，現在batchの予測分布が単一クラスへ偏っている場合は，current batch statisticsの寄与とentropy minimization更新を弱める。

### 6.2.1 OFTTAとの違い

EMA-Tentは，見かけ上はsource統計とtest-time統計を組み合わせる点でOFTTAと似て見えるが，手法の中心は異なる。OFTTAは，source寄りのweighted BatchNormとT3A型のclassifier adjustmentを組み合わせた手法であり，適応の中心はtarget featureに基づくclassifier sideの更新にある。一方，EMA-TentはTentの枠組みを保ち，entropy minimizationによってBN affine parameterを更新する手法である。

両者の違いは次のように整理できる。

| 観点 | OFTTA | EMA-Tent |
|---|---|---|
| 基本原理 | weighted BN + classifier adjustment | Tent-style entropy minimization |
| 主な適応対象 | classifier sideのsupport / weight再構成 | BN affine parameter |
| BN統計 | source + current batch | source + target EMA + current batch |
| 時間方向の履歴 | 明示的なtarget BN履歴は持たない | target BN統計をEMAで保持 |

したがって，EMA-Tentは「OFTTAにEMAを追加したもの」ではない。むしろ，**current batch統計のみに依存するTentの弱点を，history-awareなBN統計設計によって補うTent派生手法**である。OFTTAがsource priorを保ったままその場のbatchとclassifier supportに反応するのに対し，EMA-Tentはsource anchor，target個人の長期履歴，current batchの局所情報を同時に使いながら，Tentと同じ目的関数で適応する。

### 6.3 EMA BatchNorm

通常Tentでは，BatchNorm統計は以下のように表せる。

```text
mean = current_batch_mean
std  = current_batch_std
```

EMA-Tentでは，以下の混合統計を用いる。

```text
mean = w_source * source_mean
     + w_ema    * ema_test_mean
     + w_batch  * current_batch_mean

std  = w_source * source_std
     + w_ema    * ema_test_std
     + w_batch  * current_batch_std
```

`ema_test_mean`と`ema_test_std`はsource running statisticsで初期化し，test batchを処理するたびに更新する。

```text
ema_mean = momentum * ema_mean + (1 - momentum) * current_batch_mean
ema_std  = momentum * ema_std  + (1 - momentum) * current_batch_std
```

主実験では，hyperparameter grid searchの結果と解釈しやすさを踏まえ，以下の設定を用いた。この設定は，平均Macro-F1最良の設定とほぼ同等の性能を示しつつ，source統計をanchorとして残す設計意図とも整合する。

| Parameter | Value |
|---|---:|
| `w_source` | 0.2 |
| `w_ema` | 0.5 |
| `w_batch` | 0.3 |
| `momentum` | 0.80 |

なお，grid searchでは `w_source=0.1, w_ema=0.5, w_batch=0.4, momentum=0.85` が平均Macro-F1で最良であったが，2番目の `w_source=0.2, w_ema=0.5, w_batch=0.3, momentum=0.80` との差は +0.0007 Macro-F1 と非常に小さかった。そのため，本研究では後者を主設定として扱う。

### 6.4 Batch bias gate

EMA-Tentでは，batchの予測分布からgateを計算する。

```text
pred_dist = mean softmax probability over batch
diversity = normalized entropy(pred_dist)
gate = diversity ^ gate_power
```

`gate`は0から1の値であり，1に近いほどbatchの予測分布が多様で，0に近いほど単一クラスへ偏っていることを意味する。

直感的には，`gate` は **「今のbatchをどれだけ信頼してよいか」** を表す量である。もしbatch内の予測が複数クラスに分散していれば，そのbatchはtarget分布の一部をある程度代表しているとみなせるため，`gate` は大きくなる。一方で，batch内の予測がほぼ1つのクラスに集中している場合，そのbatchはWESADの時系列ブロック構造に由来する単一状態区間を見ている可能性が高く，target分布全体を代表していない。このとき `gate` は小さくなる。

したがって，`gate` が小さいとは，**「現在batchは局所的に偏っており，その統計をそのまま強く使うのは危険である」** ことを意味する。EMA-Tentはこの状況を検出すると，current batch statistics の寄与を自動的に下げ，source統計とEMA統計の寄与を相対的に高める。これにより，単一クラスのブロック区間に入ったときにBN統計がその区間へ一気に引きずられることを防ぐ。

gateが小さい場合，current batch statisticsの重みをsource側へ戻す。

```text
w_batch_effective = w_batch * gate
w_source_effective = w_source + w_batch * (1 - gate)
```

また，gateがしきい値未満の場合はentropy minimizationによるBN affine更新を行わない。主設定では`ema_tent_gate_threshold = 0.2`を用いた。

### 6.5 1 batchでの処理

EMA-Tentの1 batch処理は以下である。

1. source/EMA寄りのBN統計でprobe forwardを行う。
2. batch予測分布の多様性からgateを計算する。
3. gateに応じてcurrent batch statisticsの寄与を調整する。
4. source + EMA + current batchの混合統計でadapt forwardを行う。
5. そのbatchのlogitsを評価出力として返す。
6. gateが十分大きい場合のみentropy lossでBN affineを更新する。
7. current batch統計でEMA統計を更新する。

評価に使うlogitsは，既存Tentと同じく，そのbatchのentropy update前のforward出力である。更新は次batch以降に反映される。

## 7. EMA-Tentの実験方法

EMA-Tentの評価では，予備実験と同じWESAD 3分類，1D-CNN source model，LOSO評価を用いた。比較手法はSource，Tent，OFTTA，EMA-Tentである。

通常順序条件では，`cfg/dataset/wesad_3class.yaml`を用い，target loaderの`target_shuffle`をfalseにした。shuffle条件では，`cfg/dataset/wesad_3class_target_shuffle.yaml`を用い，target loaderのみをshuffleした。

比較スクリプトでは，各手法の評価直前に同じseedを設定し直すことで，shuffle条件でも各手法が同じshuffled batch列を見るようにした。

| 条件 | Dataset config | 内容 |
|---|---|---|
| Sequential | `cfg/dataset/wesad_3class.yaml` | target windowを保存順にbatch化 |
| Shuffle | `cfg/dataset/wesad_3class_target_shuffle.yaml` | target loaderのみshuffle |

実行ログは以下である。

| 条件 | Log directory |
|---|---|
| Sequential | `logs/wesad/compare_source_tent_oftta/260421_233138` |
| Shuffle | `logs/wesad/compare_source_tent_oftta/260422_165051` |

## 8. EMA-Tentの実験結果

### 8.1 通常順序条件

通常順序条件における結果は以下である。

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Source | 0.6915 | 0.5285 |
| Tent | 0.4553 | 0.3970 |
| OFTTA | 0.5231 | 0.4499 |
| EMA-Tent | 0.7075 | 0.6136 |

EMA-TentはSource，Tent，OFTTAを上回った。特に，Tentに対してMacro-F1を+0.2166改善し，15被験者中15被験者でTentを上回った。

| Comparison | Macro-F1 difference | Improved subjects | Worsened subjects |
|---|---:|---:|---:|
| EMA-Tent - Tent | +0.2166 | 15 / 15 | 0 / 15 |

この結果は，EMA-Tentが通常順序のtarget streamにおいてTentの破綻を緩和していることを示す。

### 8.2 Sequential vs Shuffle

EMA-Tentがbatch composition依存性を抑えているかを確認するため，Sequential条件とShuffle条件を比較した。

| Method | Sequential Macro-F1 | Shuffle Macro-F1 | Shuffle gain |
|---|---:|---:|---:|
| Source | 0.5285 | 0.5285 | +0.0000 |
| Tent | 0.3970 | 0.6285 | +0.2315 |
| OFTTA | 0.4499 | 0.6407 | +0.1908 |
| EMA-Tent | 0.6136 | 0.6111 | -0.0024 |

TentとOFTTAはshuffleによって大きく改善した。一方，EMA-TentのShuffle gainは-0.0024であり，ほぼゼロであった。

したがって，EMA-Tentは通常順序条件でもshuffle条件と同程度の性能を維持しており，current batch compositionへの依存を大きく抑制したといえる。

ただし，shuffle条件ではTentのMacro-F1が0.6285，EMA-Tentが0.6111であり，EMA-TentはTentを下回った。そのため，EMA-Tentの主張は「全条件でTentより高性能」ではなく，「通常順序，すなわちオンライン時系列に近い条件でTentの破綻を緩和する」とする。

### 8.3 Batch bias相関分析

batch偏り指標とShuffle gainの相関は以下である。

| Target | Bias metric | Pearson r | Spearman rho |
|---|---|---:|---:|
| Tent shuffle gain | mean_max_class_ratio | 0.4057 | 0.4933 |
| Tent shuffle gain | mean_imbalance | 0.4622 | 0.4929 |
| OFTTA shuffle gain | mean_max_class_ratio | 0.4922 | 0.5433 |
| OFTTA shuffle gain | mean_imbalance | 0.6202 | 0.5500 |
| EMA-Tent shuffle gain | mean_max_class_ratio | 0.1339 | 0.2109 |
| EMA-Tent shuffle gain | mean_imbalance | 0.1482 | 0.2500 |

TentとOFTTAでは，batch偏りが大きいほどshuffleによる改善が大きい傾向が見られた。一方，EMA-Tentでは相関が弱く，平均Shuffle gainもほぼゼロであった。この結果は，EMA-Tentがbatch composition biasへの感度を下げていることを補助的に示す。

また，Source性能とTTA改善量の関係も確認した。ここでは，各被験者のSource Macro-F1をx軸，TTA手法の `method - Source` Macro-F1差分をy軸として相関を計算した。

| Target | x metric | Pearson r | Spearman rho |
|---|---|---:|---:|
| Tent - Source | Source Macro-F1 | -0.9026 | -0.8821 |
| OFTTA - Source | Source Macro-F1 | -0.8594 | -0.7429 |
| EMA-Tent - Source | Source Macro-F1 | -0.3057 | -0.2714 |

TentとOFTTAでは，Source性能が低い被験者ほどSourceとの差分が大きくなる強い負の相関が見られた。ただし，これは通常順序条件でSource性能が高い被験者ほどTent/OFTTAが性能を崩しやすいことも含んでおり，単純な改善効果として解釈しすぎない方がよい。一方，EMA-Tentでは相関が弱く，Source性能に対してより安定した改善傾向を示した。

### 8.4 EMA-Tent variantのアブレーション

EMA-Tentの統計混合重みを変更し，safe，default，adaptive，およびgrid search条件を比較した。

| Variant | Source weight | EMA weight | Batch weight | Momentum |
|---|---:|---:|---:|---:|
| safe | 0.5 | 0.4 | 0.1 | 0.9 |
| default | 0.3 | 0.5 | 0.2 | 0.9 |
| adaptive | 0.2 | 0.5 | 0.3 | 0.85 |
| selected | 0.2 | 0.5 | 0.3 | 0.80 |
| grid best | 0.1 | 0.5 | 0.4 | 0.85 |

| Variant | Sequential Macro-F1 | Shuffle Macro-F1 | Shuffle gain |
|---|---:|---:|---:|
| safe | 0.5676 | 0.5712 | +0.0036 |
| default | 0.5908 | 0.5912 | +0.0005 |
| adaptive | 0.6107 | 0.6096 | -0.0011 |
| selected | 0.6136 | 0.6111 | -0.0024 |
| grid best | 0.6142 | not evaluated | not evaluated |

safe，default，adaptive，selectedのいずれもShuffle gainはほぼゼロであり，EMA-Tent系の設計がbatch composition依存性を抑える傾向は一貫していた。Sequential Macro-F1ではgrid bestが最も高く，selectedがそれにほぼ同等の性能を示した。

## 9. 論文に載せる図表候補

| 図表 | 目的 | ファイル |
|---|---|---|
| t-SNE by label and subject | 同一感情ラベル内でも被験者差があることを示す | `logs/wesad/subject_tsne/260422_223445/tsne_by_label_subject_color.png` |
| t-SNE colored by subject | 全体の特徴空間で被験者差が反映されることを示す | `logs/wesad/subject_tsne/260422_223445/tsne_all_subject_color.png` |
| Sequential vs Shuffleの表 | 既存TTAのbatch composition依存とEMA-Tentの改善を示す | 本MD内の表 |
| 通常順序の平均性能棒グラフ | EMA-Tentが通常順序でTentを改善することを示す | `logs/wesad/compare_source_tent_oftta/260421_233138/source_tent_oftta_average.png` |
| Shuffle条件の平均性能棒グラフ | shuffle時にTent/OFTTAが回復することを示す | `logs/wesad/compare_source_tent_oftta/260422_165051/source_tent_oftta_average.png` |
| batch class distribution | 通常順序batchのクラス偏りを示す | `logs/wesad/batch_bias_analysis/260420_181239/batch_class_distribution_by_subject.png` |
| max class ratio heatmap | ブロック構造によるbatch偏りを可視化する | `logs/wesad/batch_bias_analysis/260420_181239/batch_max_class_ratio_heatmap.png` |
| Source Macro-F1 vs TTA improvement | Source性能とTTA効果の関係を示す | `logs/wesad/batch_bias_analysis/260423_000427/source_macro_f1_vs_tta_delta_scatter.png` |
| bias vs shuffle gain scatter | batch偏りとTTA性能変動の関係を示す | `logs/wesad/batch_bias_analysis/260423_000427/batch_bias_vs_shuffle_gain_scatter.png` |

## 10. 論文中で使える主張文

英語では以下のように書ける。

```text
We first visualized 1D-CNN intermediate features using t-SNE and observed that samples from different subjects tended to form separated clusters even within the same emotion label, suggesting substantial inter-subject variability in physiological signals. We then investigated whether representative test-time adaptation methods can be directly applied to personalized emotion recognition. Although Tent and OFTTA are effective in image-based TTA benchmarks, they degraded performance under the sequential WESAD target stream. By comparing sequential and shuffled target batches, we found that these methods strongly depend on batch composition: Tent and OFTTA improved Macro-F1 by +0.2315 and +0.1908, respectively, when target batches were shuffled. This indicates that the block-structured target stream in WESAD causes severe batch composition bias. To address this issue, we propose EMA-Tent, which stabilizes Tent by combining source BatchNorm statistics, exponentially averaged target statistics, and current batch statistics. EMA-Tent improved sequential Macro-F1 over Tent by +0.2166 and reduced the shuffle gain to -0.0024, suggesting reduced sensitivity to batch composition.
```

日本語では以下のように書ける。

```text
本研究ではまず，1D-CNNの中間特徴をt-SNEで可視化し，同一感情ラベル内でも被験者ごとに特徴分布が分離する傾向を確認した。これは，生体信号に基づく感情推定では個人差が大きく，未知個人への適応が重要であることを示唆する。次に，感情推定の個人適応に対して代表的なTTA手法をそのまま適用できるかを検証した。TentとOFTTAをWESADの1D-CNNベース3分類タスクへ適用したところ，通常順序のtarget streamではSourceよりも性能が低下し，既存TTAをそのまま用いるだけでは安定な個人適応が得られないことが分かった。Sequential条件とShuffle条件を比較した結果，TentとOFTTAはshuffleによりMacro-F1がそれぞれ+0.2315，+0.1908改善した。これは，WESADのブロック構造により通常順序batchのクラス構成が偏り，current batchに依存するTTAが不安定化していることを示す。そこで本研究では，source BN統計，過去test batchのEMA統計，current batch統計を混合するEMA-Tentを提案した。EMA-Tentは通常順序条件でTentに対してMacro-F1を+0.2166改善し，shuffle gainを-0.0024まで低減した。
```

## 11. 注意点

- t-SNEは定性的な可視化であり，個人差の存在を補助的に示す図として扱う。定量的な主張はLOSO評価とTTA実験結果に基づいて行う。
- 予備実験は「既存TTAが悪かった」という結果だけでなく，「なぜ悪かったか」を診断する実験として位置づける。
- Shuffle条件は実運用可能なオンラインTTAではなく，batch composition依存性を測る診断条件として説明する。
- EMA-Tentはshuffle条件ではTentを上回らないため，「全条件でTentより優れる」とは書かない。
- 主張の中心は，通常順序条件での破綻緩和とShuffle gainの低下である。
- 相関分析は被験者数15に基づくため，因果証明ではなく補助的証拠として扱う。
- 主設定は `w_source=0.2, w_ema=0.5, w_batch=0.3, momentum=0.80` とする。grid bestは `w_source=0.1, w_ema=0.5, w_batch=0.4, momentum=0.85` であったが，上位設定間の差は小さいため，普遍的な最適値とは主張しない。
