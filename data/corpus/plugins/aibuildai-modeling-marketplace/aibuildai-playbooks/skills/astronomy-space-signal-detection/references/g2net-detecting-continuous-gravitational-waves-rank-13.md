# 13th Place Solution: plain machine learning.

Competition: g2net-detecting-continuous-gravitational-waves
Rank: #13
Source: https://www.kaggle.com/c/g2net-detecting-continuous-gravitational-waves/discussion/376724

# summary
- Dataset
- Normalization
- PyFstat data generation / parameter range
- Large Kernel and FFT
- Train more with EMA
- Etc

# Dataset
- test/: used for data augmentation during training (finally not applied)
- train/ : validation data
- Generate from PyFstat: data for training

# Normalization
```
from scipy.stats import norm
def Fnormalize(X):
    X /= X.sum(-2, keepdims=True)
    return X
def Pnormalize(X):
    n = np.prod(X.shape[-2:])
    POS = min(int(n * 0.999), n - 10)
    EXP = norm.ppf((POS + 1 - np.pi / 8) / (n - np.pi / 4 + 1))
    scale = np.partition(X.flatten(), POS, -1)[POS]
    X /= scale / EXP.astype(scale.dtype) ** 2
    return X
def normalize(X):
    X = (X[..., None].view(X.real.dtype) ** 2).sum(-1)
    X = Fnormalize(X)
    X = Pnormalize(X)
    return X
```
This data normalization has a simple theoretical background.
`Fnormalize` : Changes the complex input to the sum of the squares of the real and imaginary parts of the elements, transforming the input into a `chi2` distribution.
`Pnormalize`: Scales so that the square of the `POS`th largest number of the normal distribution is equal to the `POS`th largest number of the data.
It would be a good idea to interpret `Pnormalize` for the `chi2` distribution, but since this function only scales the input, I think you'll get similar performance.
In the same model, the performance is improved by changing the normalization technique of the inference step.
|  |Private  |Public|
| --- | --- | --- |
|  Pnormalize only|0.766  |0.739|
|  Fnormalize + Pnormalize|0.770  |0.751|

# PyFstat data generation / parameter range

Before implement the dataset sampling code, I read the PyFstat library code and tracked the parameter range. However, `psi` and `phi` were not found.
Surprisingly, only 5 days before the end of the competition, I found out about the proper range [here](https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/discussion/373973), and the model performance improved.

 Setting 1(Gen1):
```
tstart: [630720013, 1861492413)
F0: [45, 600)
F1: 10 ^ truncnorm.isf(rng.uniform(), a=-100, b=3, loc=-15, scale=2)
Alpha: [0, 2pi)
Delta: [-pi/2, pi/2)
cosi: [-1, 1)
psi: [-1, 1)
phi: [-1, 1)
Tsft: 1800
SFTWindowType: "tukey"
SFTWindowBeta: 0.0001
```
 Setting 2(Gen2):
```
tstart: [630720013, 1861492413)
F0: [45, 600)
F1: 10 ^ truncnorm.isf(rng.uniform(), a=-100, b=3, loc=-15, scale=2)
Alpha: [0, 2pi)
Delta: [-pi/2, pi/2)
cosi: [-1, 1)
psi: [-pi/4, pi/4)
phi: [0, 2pi)
Tsft: 1800
SFTWindowType: "tukey"
SFTWindowBeta: 0.0001
```
|  |Validation ROCAUC|Private  |Public|
| --- | --- | --- | --- |
|  exp/v0.2-1229-1 Best valid (Gen1)|0.8569|0.761  |0.747|
|  exp/v0.2-1229-1 Last epoch (Gen1)|0.7278|  | |
|  exp/v0.2-1230-2 Best valid (Gen2)|0.8662812499999999|0.759  |0.747|
|  exp/v0.2-1230-2 Last epoch (Gen2)|0.8624|0.770  |0.756|

#  Large Kernel and FFT
I implemented [fft convolution](https://github.com/klae01/fft-conv-pytorch) to use large kernels in training. You can refer to the notebook that was released one week before the end of the competition. [link](https://www.kaggle.com/code/assign/g2net-large-kernel-inference-fft-conv2d)

## Weight Decay
It has been observed that large kernels are prone to bias and are weak. I applied a strong weight decay of 0.001 to that layer.

| |Validation ROCAUC|Private|Public|
| --- | --- | --- | --- |
|exp/v0.2-1229-1 weight decay: 1e-3 Best valid (Gen1) |0.8569|0.761|0.747|
|exp/v0.2-1230-1 weight decay: 1e-4 Best valid (Gen1) |0.8538749999999999|0.756|0.743|

## Dense stem
Some performance improvement is achieved by changing the depth-wise convolution applied per L1/H1 detector channel of the model to a normal convolution.
Compared to the best validation score, the gap is large, but the last epoch still performs better.
|exp/v0.2-1231-1|Validation ROCAUC|Private|Public|
| --- | --- | --- | --- |
|Best valid (Gen2)|0.8635312500000001|0.762|0.748|
|Last epoch (Gen2)|0.8532|0.771|0.760|

# Train more with EMA
I trained with 2x epochs with the same settings, but the model diverged. However, as the model with the highest validation score, it showed an improved score than before.
Training ended 3 hours before the end of the competition, and this is the last training.

| |Private|Public|
| --- | --- | --- |
|Gen 1|0.761|0.747|
|Gen 2|0.770|0.756|
|Gen 2 + Dense stem|0.771|0.760|
|Gen 2 + Dense stem + EMA|0.773|0.761|

# Etc
- I didn't use the idea of putting horizontal/vertical lines in the PyFstat data.
- To avoid overfitting all hypotheses to the train data, the data belonging to `train/` were only analyzed at the statistical level. Might be a dumb choice :)
- I have applied the idea of creating gaps in continuous inputs and filling them with noise, but the validation loss at the beginning of training is greatly improved, but the training becomes very unstable, and the training/validation/submission scores are all lower. It's an idea that was ultimately abandoned.
- PyFstat is slow because they uses disk. This is the bottleneck. Use `tmpfs`.
- The model was trained with 256 batches of 64k samples reused 32 times each.
- The time spent implementing the tool was too long compared to the time spent improving performance. For example, an implementation that multiprocesses a pipeline that reuses and discards data and guarantees the same order for seeds / fft convolution optimization / etc.
- Surprisingly, SNR ranges from training [6, 10]. Better than [5,10], better performance than [6, 15]. In the training phase, a low SNR range causes the model to diverge, while a high SNR range makes the model less discerning when no signal is present.

Note: This is a rough comparison of the functions that investigate the SNR mentioned above.

| PyFstat parameter ||| SNR estimation |
| --- | --- | --- | --- |
|h0|sqrtSX|cosi||
|1|[10,100]|0|[9.2,101]|
|1|[10,100]|1 or -1|[2.3,19]|
