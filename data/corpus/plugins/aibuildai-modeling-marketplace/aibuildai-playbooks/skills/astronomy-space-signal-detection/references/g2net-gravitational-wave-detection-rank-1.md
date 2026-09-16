# Top 1 solution: Deep Learning part

Competition: g2net-gravitational-wave-detection
Rank: #1
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275476

We decided to make two posts to make them more or less  focused and concise. 
 DSP part  https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275507

### Baseline solution, simple Conv1D (0.88 public LB)
At the very beginning we tried  different inputs:

1. CQT - could not make it past 0.87
2. Spectrograms - with nnAudio, a bit better
3. raw signal - much better 

After applying highpass filter at 20Hz  1D conv  on stacked channels input quickly produced good val scores. Also it was much easier to make experiments as training on a single GPU took less than 1 hour. The model was very simple, just a bunch of conv-bn-silu blocks and maxpools. For convolutions we used kernel sizes (by block) 64- > 32 -> 16 -> 8. 

At that stage for augmentations we used:

channel shuffle (only Hanford, Livingston)
minor time shifts between channels, 5ms

This approach gave us 0.877 public LB score with a single 5fold model.

**SGD is better than AdamW for Conv1D without synthetic data**

Even though model capacity was quite low, the model quickly overfitted, usually after 20 epochs, which took around 30 mins. 
Switching to SGD with weight decay and nesterov momentum improved LB score to **0.880**. Other Optimizers like AdamW with whatever weight decay, MadGrad gave lower quality.

### Improved Conv1D model (0.883 public LB)
At that time the generated synthetic dataset was not good enough and we felt that 1D model could be improved. 
Following the Inception V3 approach we added different kernel sizes in each conv block, the starting block was with 32, 64, 128 kernel sizes. 
This gave us a boost from 0.88 to 0.881 on the LB.
Next step was to try even more kernel sizes as  it also made sense from DSP theory. Inception like block with 5 different kernels (16, 32, 64, 128, 256) allowed us to get 0.8823 single model LB score. 
A small ensemble improved LB score to **0.883**. 
Adding more kernel sizes did not improve CV/LB scores.
```
# Main building block for Conv1D models
class ConcatBlockConv5(nn.Module):
    def __init__(self, in_ch, out_ch, k, act=nn.SiLU):
        super().__init__()
        self.c1 = conv_bn_silu_block(in_ch, out_ch, k, act)
        self.c2 = conv_bn_silu_block(in_ch, out_ch, k * 2, act)
        self.c3 = conv_bn_silu_block(in_ch, out_ch, k // 2, act)
        self.c4 = conv_bn_silu_block(in_ch, out_ch, k // 4, act)
        self.c5 = conv_bn_silu_block(in_ch, out_ch, k * 4, act)
        self.c6 = conv_bn_silu_block(in_ch * 5 + in_ch, out_ch, 1, act)
    def forward(self, x):
        x = torch.cat([self.c1(x), self.c2(x), self.c3(x), self.c4(x), self.c5(x), x], dim=1)
        x = self.c6(x)
        return x
```

Hyperparameters
- optimizer: SGD, wd=1e-4, nesterov momentum
- learning rate: 0.1 with cosine annealing
- batch size: 128
- epochs: 40 
- input: 3 channels of raw signal filtered with butterworth filter at 20hz
- augmentations: freq masking, time masking, small shifts, channel shuffle


### Using synthetic dataset (0.886 public LB)
As soon as Denis found a more or less good approach to signal/noise generation we started experimenting with additional data. 
Overall we had 2 million noise samples and 1 million pure signal samples. During training positive sample = random noise sample + random signal sample.

From these experiments 
- mixing synthetic data with the train dataset did not work 
- augmentations are actually harmful in this case
- pretraining on synthetic data and fine tuning on the train set works great

During pretraining stage for simplicity we used the same amount of samples in epoch as in the train set.
Pre-training around 100 epochs and fine tuning 5 folds on the train set gave 8836 on the public LB for the single Conv1d model. 
As detectors, especially Virgo, have different noise distribution it makes sense to use a separate conv1d encoder for each channel. Split encoders and a linear classifier on top of concatenated features boosted the LB score to **0.8842**.

It is clear that a fully connected layer is not the best fusion approach for  the model with separate  encoders for each channel. 
That’s where resnet34 came into play and surprisingly it worked better than other 2D models. We also predicted signal parameters during pretraining  (SNR, chirp mass, Q) which also brought minor improvements. 
Pretraining 200 epochs  and fine tuning 5 folds just 1 epoch gives **0.8858** public LB score.
Augmentations during finetuning or pretraining negatively affected CV, so the best models are without any augmentations and trained with AdamW optimizer.
[model]

Input to Resnet34 looked the following way (Handford band)

[1D features]


### Segmentation
Binary segmentation using output of Conv1D predicted good masks for strong signals but did not improve recall on weak signals. In general it could be a useful tool to analyse the data, but we did not get any boost on the LB from that.
[good signal segmentation]

### Things that did not work 
There were much more experiments that I won't describe (including different frontends, training approaches etc.), but most noticeable are:

**Denoising autoencoder**

I trained different variants of autoencoders to separate noise and signals which worked great for strong signals but produced poor results on medium to low amplitude signals.

**OHEM collapse and reverse labels mystery**

I tried different versions of hard example mining to improve model performance on hard samples but usually the model collapsed and started predicting the same probability for all samples. 

Which led to an interesting experiment:
- from full OOF predictions select positive samples with low probability and negative with high probability
- train on this subset but validate on proper split
- evaluate using predicted probabilities
- evaluate using  reversed predctions (1 - p) 



That result was really confusing and at first we thought that the dataset was mislabeled. Later even with generated synthetic data we had the same problem. 
It is clear that because of the SNR wall some positive samples can be considered as just noise, but how the model generalized to predict signal from noise samples, that’s what we could not find.
