# 11th Place Solution for the LEAP - Atmospheric Physics using AI (ClimSim) Competition

Competition: leap-atmospheric-physics-ai-climsim
Rank: #11
Source: https://www.kaggle.com/c/leap-atmospheric-physics-ai-climsim/discussion/523087

Thanks to Learning the Earth with Artificial Intelligence and Physics (LEAP) NSF Science and Technology Center and Kaggle for hosting this fun competition, which was very stable in CV, LB and PB.

Considering that my English is not particularly good, this article is being written with the assistance of chatgpt for translation. I will check and make corrections.

infer code to prove that no leak: [LEAP_11th_PB0.78491](https://www.kaggle.com/code/uesugierii/leap-11th-pb0-78491/notebook)

## Context

+ `Business context`: [https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/overview](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/overview)
+ `Data context`: [https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data)

## Overview of the approach

My final submission consisted of 6 models (with 5 different architectures) integrated together(important). All models use residual structures. All models are trained on the full dataset(low-res) from 0001-02 to 0008-01, and validated on approximately 1.7 million rows of data sampled from 0008-02 to 0009-01. Using both standardization, log transformation and embedding for input process. The models have undergone meticulous training through multiple stages(important). The weights for the ensemble of models were determined using a hill-climbing algorithm base on data from 0008-02 to 0009-01.

| Model Architecture | Number of Block | Embedding Dimension | Number of Parameters |   CV    |                                 file name                                  |  
|:------------------:|:---------------:|:-------------------:|:--------------------:|:-------:|:--------------------------------------------------------------------------:|  
|   bi-LSTM + MLP    |        7        |          8          |       5070534        | 0.71969 | [v4_1](https://www.kaggle.com/datasets/uesugierii/leap-final?select=model) |  
|   bi-LSTM + MLP    |       17        |         16          |       49089054       | 0.72703 | [v4_1](https://www.kaggle.com/datasets/uesugierii/leap-final?select=model) |  
| bi-LSTM + MLP + BN |        7        |          8          |       5238534        | 0.72493 | [v4_8](https://www.kaggle.com/datasets/uesugierii/leap-final?select=model) |  
| bi-GRU + MLP + BN  |        9        |          8          |       5286134        | 0.72536 | [v5_1](https://www.kaggle.com/datasets/uesugierii/leap-final?select=model) |  
| bi-RNN + MLP + BN  |       17        |          8          |       4511734        | 0.71692 | [v8_1](https://www.kaggle.com/datasets/uesugierii/leap-final?select=model) |  
|    1D-CNN + BN     |       11        |          8          |       16854334       | 0.71459 |  [v6](https://www.kaggle.com/datasets/uesugierii/leap-final?select=model)  |  
|      Ensemble      |                 |                     |                      | 0.73472 (PB: 78491) |                                                                            | 

## Solution Detail

### data process detail

Based on my adversarial verification experiments, there is a significant difference(AUC=1) in data between years. According to the organizer's information, there should be one billion pieces of data in theory (but there are only about 80 million on HuggingFace). Therefore, it can be speculated that the test set is from data in the later years. Considering the above information, I used data from 0001-02 to 0008-01 for training, and data from 0008-02 to 0009-01 for validation. Perhaps this is why I was able to climb up the PB ranking and enter the gold medal zone.

The original input had 556 columns. I removed 66 columns that only had one value, leaving 490 columns. For each column of data, I perform two operations: one is to directly standardize it `(X - x_mean) / x_std`, and the other is to apply log after taking the absolute value and then standardize it (CV +0.003). `(log_X - log_x_mean) / log_x_std` So the shape of the input for my model is (980,)

For each of the 980 inputs to the model, an embedding is assigned with an initialization range of [-0.05, 0.05]. This initialization range is **crucial** as other initialization methods may result in poorer performance.

By padding and reshape, the input eventually becomes (bs, 60, 25*emb_dim).

### model detail (important)

For `bi-LSTM + MLP + BN`, `bi-GRU + MLP + BN`, `bi-RNN + MLP + BN`, The model structures are all arranged in the following manner, with the only difference being the encoding layer.

```python
pre_seq = seq  # (bs, 60, 25*emb_dim)
for i in range(len(self.encoder_layers)):
    encoder_layer = self.encoder_layers[i]
    seq, _ = encoder_layer(seq)  # (bs, 60, 25*emb_dim*2)
    ffn = self.mlp[i]
    seq = ffn(seq)  # (bs, 60, 25*emb_dim)
    seq = seq.contiguous().view(bs, -1)  # (bs, 60*25*emb_dim)
    seq = self.mlp_bn[i](seq)  # (bs, 60*25*emb_dim)
    seq = seq.contiguous().view(bs, 60, -1)  # (bs, 60, 25*emb_dim)
    seq = F.gelu(seq)
    seq = seq + pre_seq  # (bs, 60*25*emb_dim)
    pre_seq = seq
```

For `1D-CNN + BN`, model structure as below

```python
pre_x = x
for i in range(len(self.cnns)):
    x = self.cnns[i](x) + pre_x
    pre_x = x
```

The complete code for the model can be found in the 'model' folder here. [https://www.kaggle.com/datasets/uesugierii/leap-final?select=model](https://www.kaggle.com/datasets/uesugierii/leap-final?select=model)

### training detail (important)

During the training process, it was found that training directly on the full dataset (low-res) resulted in lower scores. After performing EDA, it was discovered that the main cause of the unstable training was outliers.

To address this, I employed two independent methods for mitigation: 

1. First method involved clipping the value range of the input and target(`np.clip(x, -10, 10)`, `np.clip(y, -10, 10)`), first fitting on a smaller range (15 epochs, lr 1e-3), and then fitting on the original data (15 epochs, lr 1e-4). The rationale behind this is to first allow the model to learn and adapt to relatively normal weather conditions before generalizing to more extreme weather variations.
2. Second method: first let the model train on one month's data(15 epochs, lr 1e-3), then on one year's data(15 epochs, lr 1e-3), and finally on all available data(15 epochs, lr 1e-4). The main idea behind this is to have the model initially learn the climate variations over short time spans, then gradually expand the time range to ensure a smoother learning process and reduce the difficulty of learning.

To gain a slight improvement in performance towards the end, I would train the model for a few epochs on an extremely large batch size (10240) and a very low learning rate (1e-5). Some models were able to achieve further enhancements (CV +0.004).

### What didn't work for me

+ Physics knowledge, neural network methods for solving partial differential equations
    + I'm not sure if it's my lack of understanding of these knowledge that's causing the lack of effect, or if there really don't work
+ U-Net, U-Net++
+ Transformers
+ Squeezeformer
+ Assist in training using target columns with weight 0
+ More complex embedding schemes, such as bucketing + embedding, AutoDis, and so on

## Sources

ptend trick: https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/499896#2791290
