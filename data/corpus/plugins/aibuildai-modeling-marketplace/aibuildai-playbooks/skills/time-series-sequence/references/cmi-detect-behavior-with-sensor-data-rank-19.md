# [EN/日本語] 19th solution writeup

Competition: cmi-detect-behavior-with-sensor-data
Rank: #19
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/en-19th-solution-writeup-19

（下に日本語による説明を書いています）

We would like to express our deepest gratitude to the competition hosts! We really enjoyed the competition for the entire period. 
This write-up is separated in two parts: the observation, preprocessing and postprocessing by @chimaki821 then modeling, deep leaning by @tokkiwa. 

# 1. @chimaki821 Part (Observation + Preprocessing)

I created this discussion using [PLaMo Translations](https://translate-demo.plamo.preferredai.jp/).

## 1.1. Overview

Like most participants, I implemented the following techniques:

* Corrected left-handed data by inverting `acc_y`, `rot_x`, and `rot_y` values, swapping THM channels 3↔5, swapping ToF channels 3↔5, and inverting an 8×8 square region
* Implemented multitask learning by adding `orientation` and `behavior` as auxiliary loss terms
* Switched between IMU-only and full sensor model configurations
* Applied corrections to subjects suspected of having mounting anomalies (partially successful, details follow)
* Used weighted ensemble methods (though not included in the final submission, these proved effective in private testing)

After implementing these baseline approaches, my teammate @tokkiwa significantly improved the model through various enhancements to the pipeline and ensemble methods with multiple models. This proved extremely valuable since I had virtually no background in machine learning.

Below, I describe several key techniques that helped inform the above approach, as well as additional methods I implemented:

* Data observation and mounting correction
* Data augmentation (DA) and test-time augmentation (TTA) using rotation around the y-axis
* Creation of soft labels

I also summarize:

* What I tried during the competition that didn't work or was left unimplemented
* Additional approaches I wanted to explore but couldn't implement in time


## 1.2. Data Observation and Mounting Correction

This section was developed in collaboration with @epopaca.

We created a [notebook](https://www.kaggle.com/code/chimaki821/visualization-for-19th-place-solution) that, when provided with a specific subject and gesture, consolidates all relevant data into a single image. 81 subjects × 18 gestures = 1,458 images saved on my smartphone, constantly observing them whenever I had time.

e.g. (SUBJ_000206, gesture: Above Ear - pull hair)


Using this visualization, we conducted analyses on left-hand correction and mounting adjustments. By examining `acc_x, acc_y, acc_z` values, we determined that `SUBJ_019262` and `SUBJ_045235` likely had improper mounting. Considering potential issues with the Herios sensor's attachment, there are four main possible scenarios (including normal conditions):
(For reference on the sensor's xyz axis orientation, see this [discussion thread](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/588501#3243561)).


| Normal (Original) | 180° Rotation around Y-axis |
| ---- | ---------------------|
| |  |
| 180° Rotation around Z-axis | 180° Y-axis + 180° Z-axis |
|||

Considering the possibility that test data might also contain similar installation errors, we implemented a binary classification system to detect installation anomalies and apply correction during inference. However, since the number of relevant cases was limited in public testing, the effect was difficult to discern. Therefore, for this submission, we prioritized false positive reduction by setting the threshold to 0.9, resulting in only limited effectiveness. When lowering the threshold to 0.5 in the Late Submission, our score improved to 0.862 😢


## 1.3. DA+TTA with Y-Axis Rotation

Building on the assumption of fixed sensor orientation, we addressed the scenario where "even with proper installation, minor rotation occurs around the y-axis." Therefore, we implemented:

* During training: DA with random rotation around the y-axis using a uniform distribution between -15° and +15°
* During inference: TTA that attempts multiple small rotation angles around the y-axis and averages the results

By setting the inference angles to `[0, +5, -5, +15, -15]`, we observed an average improvement of approximately +0.002 in public LB across multiple experiments. Private LB also showed similar performance gains.

For this implementation, we carefully applied the principle that "rotation operations should be applied from the left to acceleration data and from the right to the rotation data."

## 1.4. Creating Soft Labels

This section was collaboratively developed with @mckmckmck.

While previous optimizations focused on "preventing difficult subjects from being overlooked," when we examined OOF scores by gesture type, we identified certain gestures that were particularly challenging to predict. Here is an example from the IMU-only model:

```
-----------------------------------------------------
F1 score for class[0] = 0.666 tp=417, fp=197, fn=221
F1 score for class[1] = 0.643 tp=405, fp=214, fn=235
F1 score for class[2] = 0.824 tp=548, fp=142, fn=92
F1 score for class[3] = 0.442 tp=286, fp=369, fn=352
F1 score for class[4] = 0.609 tp=384, fp=238, fn=256
F1 score for class[5] = 0.538 tp=340, fp=284, fn=300
F1 score for class[6] = 0.693 tp=444, fp=198, fn=196
F1 score for class[7] = 0.517 tp=328, fp=305, fn=309
F1 score for class[8] = 0.985 tp=2998, fp=54, fn=40
-----------------------------------------------------
Macro F1 = 0.6573605192818119
BFRB Macro F1 = 0.6164599766386722
Score = (Binary F1 + Macro F1) / 2 = 0.824077831125434
```

※ Class 8 represents non-BFRB gestures. Classes 0-7 correspond to:

```
0: Above ear - pull hair
1: Forehead - pull hairline
2: Forehead - scratch
3: Eyebrow - pull hair
4: Eyelash - pull hair
5: Neck - pinch skin
6: Neck - scratch
7: Cheek - pinch skin
```

Furthermore, we visualized the confusion matrix for the BFRB category (where element (i, j) represents the number of sequences with true label i and predicted label j):



From these observations, we identified:

* The distinction between "Eyebrow - pull hair" and "Eyelash - pull hair" proves particularly difficult
* Other similar gesture categories exist

We then introduced soft labeling after one-hot encoding, assigning small weights to similar gesture categories, which yielded modest but noticeable improvements overall.

## 1.5. Unsuccessful Approaches and Additional Considerations

**Areas for Improvement**

* **Stacking**

  * The weighted average of CV maximization weights calculated using Optuna was discarded as it resulted in lower public leaderboard scores (private score: 0.860–0.861 😢)
  * Attempted stacking with LightGBM but found no improvement in CV performance, so abandoned this approach
* **Alignment Correction**

  * Due to uncertainty about the model's robustness and to avoid false positives, I increased the threshold, which only produced limited effectiveness (see Section 1.2)

**Additional Approaches I Considered**

* **Self-Supervised Learning and Adversarial Training**

  * Although I wanted to experiment with these methods given the large number of subjects (81), lack of expertise prevented implementation
* **Pre-trained Model Utilization**

  * Referenced [this paper](https://www.nature.com/articles/s41746-024-01062-3) on self-supervised learning for human activity recognition and considered transfer learning, but ran out of time
* **Rotation Correction Around the Y-Axis**

  * Some data exhibited rotation around the y-axis (likely requiring -90° correction for SUBJ_011323 and +60° for SUBJ_032165)
  * Since omitting this correction didn't cause significant accuracy degradation compared to other subjects, I prioritized other tasks
  * If I had more time, I would have developed a model to predict y-axis rotation angles and implemented the correction

If anyone has successful implementations or insights regarding these areas, I'd be very grateful for your feedback!!



# 2. @tokkiwa Part (Deep Learning & Modeling) 

## TL;DR
- Channel-wise Convolution really matters (33rd place with single model!)
- 3D CNN for TOF
- Multitask Training 

## 2.1 Model
Our main model consists of three parts; (1) Channel-wise Feature Extractor, (2) BiGRU and (3) MLP Head. 
The latter two are not that interesting, so let's focus on the Feature Extractor. We first noticed that the channel-independent cnn models in [this notebook](https://www.kaggle.com/code/kaigaokaigao/lb-0-75-imu-only-multibranch-inference) performs better than other models. Here is the code snippet; 
```py
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    MultiScaleConv1d(1, 12, kernel_sizes=[3, 5, 7]),
                    ResidualSEBlock(36, 48, 3, dropout=0.3),
                    ResidualSEBlock(48, 48, 3, dropout=0.3),
                )
                for _ in range(num_channels)
            ]
        )
```

This model prepares cnn pipelines independently for each of the channels. As the sensor data have completely different modalities for each channels, it is reasonable to employ independent kernels to extract rich features. We further rewrite the branches with a grouped convolution, initially for the speed-up. Unexpectedly, combining the channel-independent convolution with inter-channel SE block performs much better than the fully channel-independent processing.

Additionally, our teammate @chimaki821 adopted a 3D-CNN for the TOF features to capture the spatio-temporal relationship of the sensors.

```py
           self.tof_branch = nn.Sequential(
                nn.Conv3d(tof_sensors, 24 * tof_sensors, (5,2,2), padding='same', groups=tof_sensors, bias=False),
                nn.BatchNorm3d(24 * tof_sensors), nn.ReLU(), nn.MaxPool3d((1,2,2)),
                nn.Conv3d(24 * tof_sensors, tof_out_channels * tof_sensors, (7,2,2), padding='same', groups=tof_sensors, bias=False),
                nn.BatchNorm3d(tof_out_channels * tof_sensors), nn.ReLU(),
                nn.AdaptiveMaxPool3d((None, 1, 1)),
            )
```

The features are then fed to the BiGRU and Attention Pooling, added gaussian noise and finally input to the small MLP head. Please refer to our codes for more details. 
For the submission, we trained some other variant (e.g. with FFT features added) of this model and other simple CNN model seen in the public notebooks. 

## 2.2 Traning
We used separated losses for behavior, orientation and gesture and summed them up (originally implemented by @chimaki821). 
Specifically, cross entropy was applied for orientation and gesture (for gesture, label smoothing = 0.2 was our best practice) and a `nn.BCEWithLogitsLoss` to behavior. 
We finally used Cosine Annealing LR, Adam and the following data augments: 
- random noise
- random scaling
- artificial drift
- random overwrite
- mixup

Focal loss, exponential moving average, radam and normalization were implemented but not effective for our models. 
The result of wandb sweep suggests that normalization of the columns affects badly on our models. The gain by multi-task training is also limited, but we suppose it might have a good effect on ensembling as the method enables the model to learn differently. 

---

（日本語）

# 1. @chimaki821 Part (Observation + Preprocess)

## 1.1. 概要

多くの参加者と同様に、以下を実施しました。

* 左手データの補正（`acc_y`・`rot_x`・`rot_y` の反転、THM 3↔5 の swap、ToF 3↔5 の swap、8×8 正方形領域の反転）
* マルチタスク学習（`orientation` と `behavior` を補助ロスに追加）
* IMU のみ／全センサーでのモデル切替
* 取り付け異常が疑われる被験者の補正（部分的に成功、後述）
* 重み付きアンサンブル（最終提出には未採用だが private では有効）

これらの基本実装ののち、チームメイトの @tokkiwa がモデル改善・パイプライン整備・複数モデルのアンサンブルを進めてくれたことで、スコアが大幅に向上しました。私は機械学習について何も知らなかったため、大変助かりました。

以下では、上記の方針検討に役立った取り組みと、それ以外に行った

* データ観察と取り付け補正
* y 軸回り回転による DA（Data Augmentation）＋ TTA（Test-Time Augmentation）
* ソフトラベルの作成

について述べます。あわせて、

* コンペ期間中に試したがうまくいかなかったこと・やりたかったこと

もまとめます。

## 1.2. データ観察と取り付け補正

このパートは @epopaca と共同で進めました。

subject と gesture を指定すると該当データを 1 枚の画像にまとめて出力する [notebook](https://www.kaggle.com/code/chimaki821/visualization-for-19th-place-solution) を作成しました。81人×18 gestures = 1,458 枚の画像をスマートフォンに保存し、時間があれば常にこれを観察していました。

（出力例は英語版をご覧ください。）

この可視化を使いながら、左手補正や取り付け補正を検討しました。`acc_x, acc_y, acc_z` を見ると、`SUBJ_019262` と `SUBJ_045235` は取り付けが不適切だと推測できます。Herios sensor の取り付けミスを想定すると（正常を含め）大きく4通りが考えられます（センサーの xyz 軸の向きは [この discussion](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/588501#3243561) を参照）。

（図は英語版をご覧ください。）

テスト側にも同様の取り付け方ミスがある可能性を考え、2 値分類で取り付け異常を判定し、推論時に補正をかける仕組みを用意しました。ところが public では該当ケースが少なく効果が見えづらかったため、本提出では偽陽性抑制を優先して閾値を 0.9 に設定。その結果、効果は限定的でした。Late Submission で閾値を 0.5 に下げたところ、スコアが 0.862 まで上昇しました 😢

## 1.3. y 軸回り回転による DA＋TTA

センサー軸が固定されている前提から、「正常取り付けのつもりでも y 軸回りにわずかな回転が入る」ケースを想定しました。そこで、

* 学習時：y 軸回りに一様乱数（−15°〜＋15°）で回転させる DA
* 推論時：y 軸回りの微小回転を複数角度で試し平均化する TTA

を実装。推論角度を `[0, +5, -5, +15, -15]` としたところ、複数の実験で public LB が平均で約 +0.002 改善。private でも同程度の上昇を確認しました。

この実装を行うのにあたって、「加速度データには回転作用素を左から作用、回転データには回転作用素の逆作用素を右から作用」させることに注意しました。

## 1.4. ソフトラベルの作成

このパートは @mckmckmck と共同で行いました。

これまでの工夫は「難しい被験者を取りこぼさない」ことを目指したものでしたが、OOF（out-of-fold）について gesture 別にスコアを求めてみると、予測が難しい gesture があることがわかりました。IMU Only モデルの例：

```
-----------------------------------------------------
F1 of class[0] = 0.666 tp=417, fp=197, fn=221
F1 of class[1] = 0.643 tp=405, fp=214, fn=235
F1 of class[2] = 0.824 tp=548, fp=142, fn=92
F1 of class[3] = 0.442 tp=286, fp=369, fn=352
F1 of class[4] = 0.609 tp=384, fp=238, fn=256
F1 of class[5] = 0.538 tp=340, fp=284, fn=300
F1 of class[6] = 0.693 tp=444, fp=198, fn=196
F1 of class[7] = 0.517 tp=328, fp=305, fn=309
F1 of class[8] = 0.985 tp=2998, fp=54, fn=40
-----------------------------------------------------
Macro F1 = 0.6573605192818119
BFRB Macro F1 = 0.6164599766386722
Score = (BinaryF1 + Macro F1) / 2 = 0.824077831125434
```

※ class 8 は non-BFRB。class 0〜7 は以下：

```
0: Above ear - pull hair
1: Forehead - pull hairline
2: Forehead - scratch
3: Eyebrow - pull hair
4: Eyelash - pull hair
5: Neck - pinch skin
6: Neck - scratch
7: Cheek - pinch skin
```

さらに、BFRB 部分の混同行列を可視化（(i, j) 成分は 正解が i 番目の gesture, 予測が j 番目の gesture となる sequence の数）

（図は英語版をご覧ください）

これらの観察から、

* “Eyebrow - pull hair” と “Eyelash - pull hair” は類似
* その他にも類似 gesture が存在

ことが分かります。そこで one-hot の後に、類似 gesture に小さな重みを配分するソフトラベリングを導入したところ、全体的にわずかながら改善が得られました。

## 1.5. うまくいかなかったこと／他にやりたかったこと

**うまくいかなかったこと**

* **Stacking**

  * Optuna で CV 最大化の重みを求めた重み付き平均は、public LB が低下したため選択せず（private score: 0.860〜0.861 😢）
  * LightGBM による stacking は CV 改善が見られず断念
* **取り付け補正**

  * モデルの強さに確信が持てず偽陽性を避けるため閾値を上げた結果、効果が限定的（§1.2 参照）

**他にやりたかったこと**

* **自己教師あり学習・対照学習**

  * subject が 81 と多くないため試したかったが、知識不足で着手できず
* **事前学習モデルの利用**

  * [Self-supervised learning for human activity recognition using 700,000 person-days of wearable data](https://www.nature.com/articles/s41746-024-01062-3) を参照し、転移学習を検討したが時間不足
* **y 軸回り回転の回帰補正**

  * y 軸周りに回転してしまっているデータもある（おそらく SUBJ_011323 は -90度, SUBJ_032165 は +60度の補正が必要）
  * やらなくても他の subject と比べて大幅な精度低下にはなっていなかったため優先しなかった
  * 余裕があったら y 軸周りの回転角度を予測するモデルを作り、補正したかった

これらについて成功事例・検討事例があれば、ぜひ教えてください！！

# 2. @tokkiwa part
## TL;DR
- グループ化畳み込みが有効 (1モデルで33位相当) 
- TOF に 3D CNN 
- マルチタスク学習

## 2.1 Model

自分たちのモデルは、(1)特徴量抽出、(2)BiGRU と (3) MLPの予測ヘッドからなります。後者二つはPublicにあるようなものなので、おもに、主に特徴抽出について説明します。
はじめに、[この notebook](https://www.kaggle.com/code/kaigaokaigao/lb-0-75-imu-only-multibranch-inference) にあるチャンネル独立なモデルの性能が良いことに気づきました。

このモデルは次のように、各チャンネルについて完全に独立したCNNを用意して、各チャンネルを48倍に拡大しながら畳み込み処理をおこないます。
```py
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    MultiScaleConv1d(1, 12, kernel_sizes=[3, 5, 7]),
                    ResidualSEBlock(36, 48, 3, dropout=0.3),
                    ResidualSEBlock(48, 48, 3, dropout=0.3),
                )
                for _ in range(num_channels)
            ]
        )
```
今回のセンサーデータはそれぞれ全く違うモダリティがあるので、はじめに混ぜながら処理するよりも、独立に特徴量を得る方がリッチな特徴量を得られるためだと思います。
もともとのモデルはfor文でチャンネルごとに処理をしていたので、グループ化畳み込みで高速化を行いました。この途中でわかったこととして、SEブロックのみは全チャンネルで作用させるとより性能が向上します。
したがって、最終的にIMUとTHM特徴量についてはチャンネル毎畳み込みで処理し、途中のSEブロックのみ全チャンネルで処理するというパイプラインになりました。

TOF特徴量については空間的にセンサーが並んでいるので、3D-CNNを用いて空間方向と系列方向を別に扱うような形にしました。 ( @chimaki821 のアイデアです)

```py
           self.tof_branch = nn.Sequential(
                nn.Conv3d(tof_sensors, 24 * tof_sensors, (5,2,2), padding='same', groups=tof_sensors, bias=False),
                nn.BatchNorm3d(24 * tof_sensors), nn.ReLU(), nn.MaxPool3d((1,2,2)),
                nn.Conv3d(24 * tof_sensors, tof_out_channels * tof_sensors, (7,2,2), padding='same', groups=tof_sensors, bias=False),
                nn.BatchNorm3d(tof_out_channels * tof_sensors), nn.ReLU(),
                nn.AdaptiveMaxPool3d((None, 1, 1)),
            )
```

特徴量はその後にBiGRUとAttention Pooling に入力され、ガウスノイズを掛けてMLPヘッドに入力され、推論結果が出力されます。詳しくはコードを参照してください。
アンサンブルについては、このモデルの変種 (FFT特徴量を入れたものなど)と、publicに見られたようなシンプルなCNNモデルを混ぜました。先述した回転のpostprocessing などを掛ける都合で、モデルは4~5個程度しか使えなかったので、あまり多くのモデルは使っていません。

## 2.2 訓練

behavior, orientation, gesture にそれぞれ異なるロスを設定するマルチタスク学習を行いました。( @chimaki821  が実装してくれました) gesture,  orientation に対してはCross Entropy (gestureのみlabel_smoothing = 0.2), behavior については `nn.BCEWithLogitsLoss` を用いて、すべてのロスを足し合わせたものを用いました。

最終的に、訓練には Cosine Annealing LR, Adam と、次のデータ拡張を用いました。

- random noise
- random scaling
- drift (蓄積する時間ノイズの再現)
- random overwrite
- mixup

Focal loss, 移動指数平均 (EMA), radam と正規化を行いましたが、自分たちのモデルには効果がありませんでした。特に、Wandbでパラメータチューニングを行った限りだと、正規化はしないほうが明らかにパフォーマンスが向上します。マルチタスク学習についてはCVへの寄与度はあまりありませんでしたが、アンサンブルにあたっては多様な特徴量が取れてよいのではないかと思います。
