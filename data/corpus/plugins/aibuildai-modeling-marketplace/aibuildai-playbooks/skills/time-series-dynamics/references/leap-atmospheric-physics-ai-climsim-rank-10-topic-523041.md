# 10th Place Solution

Competition: leap-atmospheric-physics-ai-climsim
Rank: #10
Source: https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim/discussion/523041

# 10th Place Solution

Thank you for holding this competition. We enjoyed a lot, as well as a lot of leaning experience. This was also really tough competition we ever participated. We literary training the model and making ensemble few hours before the competition closed, and finally we managed to win a gold medal.

The following is the summary of 10th place solution by team Updraft. More detailed writeup might be shared by each members.

## Team Members

* @tatamikenn (team leader)
* @tereka
* @phalanx
* @ryches

## Glossary

* low resolution data = LR
* high resolution data = HR

## Summary

Our solution is ensemble of each 4 member's regression models.
The models in the final submission contains 23 models with different architectures and learning conditions.
We developed the following architectures.

* Conv + Transformer (+ LSTM)
* CNN / U-Net
* LSTM

## Train/Validation Data

We basically trained our models using 1-7y LR data, and evaluate with 1/6 subsample of 8y LR data.
Some model also use:

* 8y LR data
* 1-8y HR data
* pseudo labeled old/new test data.

## Evaluation Metrics

We used 1-MSE instead of R2 after std normalized target.
We use this metrics because some member reported R2 loses correlation after host replaced the test dataset.
(However, later we found both are correlated when using sufficiently small eps for normalization.)
In normalization, we used the following formula:

```python
eps = 2.0 ** (-95)
target = target / maximum(std(target), eps)
```

## Model Design

### Bilzard's Pipeline

Basically 1/7 ~ 1/6 subsample of LR data is used because of relatively *poor* machine resource (RTX4090 x 1).
What worked most was:

* feature extraction of Climate-invariant features (RH, plume buoyancy, normalized Heat Flux etc.)[1]
* confidence-aware MSE loss
* doping hard sample of 1/1 full LR data

#### Climate-invariant feature extraction

We extracted climate-invariant features which is suggested by Tom Beucler et. al. [1].

- Relative Humidity (RH)
- Plume buoyancy
- normalized Heat Flux

Conceptually, these features are invariant under both cool and warm temperature.
So these features are expected to cancel out the effect of yearly warm-up in ClimSim dataset (which I observed by EDA).

During the competition, we couldn't train models with this features on the full LR dataset, but it showed a significant gain of ~1% on a 1/7 to 1/6 subsample of LR dataset.

Note that the late submission results shows this features still have a significant gain on the full LR dataset (see Appendix A1).

#### Confidence-aware MSE loss

We used the following loss which we called *confidence-aware MSE loss* instead of MSE.
Intuitively it has the effect of loosening the contributions of the large error by hard samples.
This idea was variant of the @ryomak 's previously shared solution[2].
Note that variance of target was regressed by model in addition to mean value. So our model's output variable is 2x larger than basic models.

```python
loss = 0.5 * (target - pred) ** 2 / var + 0.5 * torch.log(var)
```

#### Doping Hard Examples

As we shared on a public discussion[3], most of error came from extreme outliers. So we came to think that hard samples are good representatives of all dataset: if we doped small amount of hard samples from 1/1 full LR data, it would help to reduces regression error even if we used 1/n sub-sampled LR data.

We extracted ~14% of hard samples which have the squared error greater than threshold using one of our best models, and doped to train other models. We found this has ~0.x% gain using this method.
We also found adding tiny amount of neighbor time-frame sample from the same location also helps gaining accuracy.

#### Model Architecture

Basically we used the following two architecture

1. Conv-Transformer (variant of @hoyso48 's architecture[4])
2. Pixel-Shuffle stacked UNet

The second architecture was inspired by stacked hourglass architecture[5] which stacks U-Net and gradually improves the accuracy of output variable by regressing error of previous stack (like boosting). In addition, replacing ConvTranspose decoder in the usual U-Net to PixelShuffle gained ~0.x%.

#### Other modeling Tips

##### Tanh normalization

We often encounter gradient explosion when training deeper transformers (~8-12 layers).
We mitigated this issue by utilizing trick we called *tanh normalization*.
It is just inserting tanh activation before calculating softmax attention:

```python
    def forward(self, x):
        x = rearrange(x, "b c n -> b n c")

        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b n (h d) -> b h n d", h=self.num_heads)
        q, k, v = qkv.chunk(3, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.tanh() * self.attn_clip_val  # stabilize attention weight
        attn = F.softmax(attn, dim=-1)

        x = attn @ v
        x = rearrange(x, "b h n d -> b n (h d)")

        x = self.proj(x)
        x = rearrange(x, "b n c -> b c n")
        return x
```

##### Removing Dropout

In the previous competition, some Kagglers reported removing Dropout layer has significant improvement on regression tasks[6]. We observed similar phenomena on this competition.

### Tereka's Pipeline

```markdown
Public/Private LB: 0.77784/0.77023

Model
- Transformer + Convolution + LSTM
- Architecture Points
 - Learnable Positional Embedding(nn.Embedding 60)
 - several Transformer Layers
 - Final Layer LSTM

Features
- Original Dataset(standard scaler)

Dataset
- 70M(full low resolution dataset)

Loss
- Huber Loss(> MSE)

Not Worked
- Mixup
- EMA
- Deeper model, I tried some experiment, but it’s not stable.
- pretrained High resolution model, but low resolution and some high resolution data is worked in public LB.
```



### Phalanx's Pipeline



### Ryches' Pipeline

TBD

## Ensemble Method

We used tree type of ensemble method and final submission is weighted ensemble of these:

- type1) Stacking using models trained with 1-7y low resolution data
- type2) Stacking using models trained with 1-8y low resolution data
- type3) Mean ensemble of models trained with 1-8y models

For type2, basically we can't use the data which trained the seed model to fit parameter in the stacking.
So we take the following scheme:

1. calculate the ensemble weight using model trained with 1-7y data
2. re-train the model of the same architecture, same training condition except swapping train data to 1-8y data
3. using ensemble weight calculated on step2, but replacing the model from 1-7y to 1-8y data to make the test prediction

For the final submission, we ensemble the type-1 ~ type-3 model with manually fitted weight:

```python
final_prediction = type1 * w1 + type2 * w2 + type3 * w3
```

The late submission result shows assigning 100% weight for type1 model performs the best on the private LB.

Model|LB(Public)|LB(Private)|final submission
--|--|--|--
type1 * 0.5 + type2 * 0.3 + type3 * 0.2 | 0.78803 | 0.78503 | ✔️
type1 * 0.75 + type3 * 0.25 | 0.78801 | 0.78487 | ✔️
type1 * 0.6 * type2 * 0.4 | 0.78777 | 0.78535 |
type1 | 0.78771 | **0.78536** |

## Detail of Stacking

We trained ensemble weight per (target, model). This is based on the study of EDA where we found each model has their expertize: some models performs good at regressing mixing ratio and some are good at regressing wind velocity etc.

We tested the following variations and found ensemble without normalization and with bias performs the best.

normalize weight?|with bias?|LB(Public)|LB(Private)
--|--|--|--
✔️|-|0.78727|0.78532
-|-|0.78739|0.78563
-|✔️|0.78740|**0.78569**

## Appendix

### A1. LR full data + Climate Invariant Feature

During the competition, we only trained our models using climate-invariant features on a subsample of the LR data.However, after the competition ended, we successfully trained Phalanx's lightweight CNN model train model with climate-invariant features on the full sample of the LR dataset in feasible training time (~1 day) with  a relatively "low-throughput" GPU (RTX 4090).

The result of late submission shows it has significant gain of 0.05% on private LB. So climate-invariant feature has actually significant impact on this task.

model|CV(1-MSE)|LB(Public)|LB(Private)
--|--|--|--
single w/o LR-full-clim-invariant|0.78231|0.78025|0.77721
single w LR-full-clim-invariant|**0.78434**|**0.7814**|**0.78017**
ensemble w/o LR-full-clim-invariant|0.79192|0.78740|0.78569
ensemble w LR-full-clim-invariant|**0.79218**|**0.78771**|**0.78618**

## Reference

- [1] [Climate-Invariant Machine Learning](https://arxiv.org/abs/2112.08440)
- [2] [9th solution(with github code): Laplace distribution noise likelihood optimization](https://www.kaggle.com/competitions/ventilator-pressure-prediction/discussion/285353)
- [3] [Is Tropical Cylones Key to Win this Competition?](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/511274)
- [4] [1st place solution - 1DCNN combined with Transformer](https://www.kaggle.com/competitions/asl-signs/discussion/406684)
- [5] [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937)
- [6] [The Magic of No Dropout](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/260729)
