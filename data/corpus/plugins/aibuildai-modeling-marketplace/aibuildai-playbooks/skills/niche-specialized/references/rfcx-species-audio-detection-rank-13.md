# 13th Place Solution – Mean Co-Teachers and Noisy Students

Competition: rfcx-species-audio-detection
Rank: #13
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220308

Hey Everybody, I wanted to dump my solution real quick in case anyone was interested. 

It seemed to me that the critical issue is that there are a **TON** of missing labels. The provided positive examples data (train_tp.csv) has ~1.2k labels. The lb probing that @cpmpml did suggest 4-5  labels per clip on average. If the train data follows the same distribution we should expect ~21k labels,  and that's just at the *clip* level. We'd expect to see multiple calls from the same bird per clip, i.e. multiple frame labels per clip label. My best models seemed to think there were closer to 40k labels.

So my idea was to do something along the lines of [Noisy Student](https://arxiv.org/abs/1911.04252), where the general idea is to do progressive pseudo labeling where each successive model is larger and there's more noise applied to the training data. On its own, Noisy Student doesn't work very well, so I used a few other tricks. 

### 1. Mean Teacher
My first setup looks super similar to what's going on in  [Mean Teachers Find More Birds](https://www.kaggle.com/reppic/mean-teachers-find-more-birds). I train on a combo of centered positive examples and random unlabeled samples using consistency and BCE loss. Here, I'm using SED + resnet34 and some light augmentation: gaussian noise, frame/frequency dropout. This gets me to **0.865**on the public lb.

Using 5-fold mean-teacher models, I do OOF prediction to get pseudo labels over the entire training dataset. 

### 2. Co-Teaching
Now I want to train on my pseudo labels, but it's safe to assume they're pretty noisy. To deal with the bias introduced by my new, noisy labels, I do something along the lines of [Co-Teaching](https://arxiv.org/abs/1804.06872). Briefly, the idea is to train 2 models simultaneously on the same data, but with different augmentations applied to each. Then the samples with the highest loss from Model A are ignored when doing backprop in Model B and vice versa. The % of ignored samples gets ramped up slowly. The theory is that the models will learn the correct labels early in training and start to overfit to noise later on. By dropping potentially noisy labels, we avoid introducing a bad bias from our pseudo labels. 

I modified the authors idea slightly for the competition. In my setup, it's impossible for either model to ignore the good labels from train_tp or train_fp. Only pseudo labels can be ignored. I believe this helps with class imbalance issues. 

Using this setup with more aggressive augmentation and densenet 121, I'm able to get to **0.906** on the public lb. 

### 3. Heavy Mixup
Finally, using my second round of pseudo labels, I train on randomly sampled segments from all the training data. Here I apply even more aggressive augmentations and add mixup 60% of the time with a mixing weight sampled from `Beta(5,5)` (typically around 0.5). For mixup, any label present in either clip gets set to 1.0. I run this for 80 epochs. The prev 2 models were run for around 32 epochs. A 5 fold ensemble with this setup using densenet 121 gets me up to **0.940** on the public lb.

I’m able to get to 0.943 by ensembling ~90 models taking the geometric mean.

### Other Tricks
* Centering the labels from train_tp in the sampled clip segment early on seemed to help.
* When making predictions I’m averaging 4 metrics: average and max clip-wise and frame-wise predictions.
* Mixup only worked for me when it was done on the log mel spectrograms. Doing it on the audio didn't work.
* Augmentations (intensities varied) (excluding mixup): 
```
augmenter = A.Compose([
    A.AddGaussianNoise(p=0.5, max_amplitude=0.033),
    A.AddGaussianSNR(p=0.5),
    A.FrequencyMask(min_frequency_band=0.01, max_frequency_band=0.5, p=0.5), 
    A.TimeMask(min_band_part=0.01, max_band_part=0.5, p=0.5),
    A.Gain(p=0.5)
])
```

Let me know if you have any questions!
