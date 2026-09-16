# 8th Place Solution

Competition: birdsong-recognition
Rank: #8
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183223

Thanks to the competition host and kaggle teams for holding this competition and congratulations to all winners. And thanks @ttahara for providing resampled train data and @rohanrao for external data. Also thanks @hidehisaarai1213 for providing a good baseline and introduction to SED. I am excited to get my solo gold.

#### Train

The labels provided by competition host is super noisy. There is even a 40 minute audio with a single label. I think how to use the secondary label is vital in this competition. In my test, using only primary labels will only predict very view bird calls and get many nocalls.  

**Mix sound**

In order to generate more samples and get clips with more than one bird(with good labels), I choose to mix the clips from audios with no secondary labels. I use an or operation to generate the labels for mixed clips and mix sound like this:

```
mixed_sound=sound1*uniform(0.8,1.2)+sound2*uniform(0.8,1.2)+sound3*uniform(0.8,1.2)
```

**Generate pseudo strong labels**

I tried to generate pseudo strong labels with a SED method. I train SED models with 10s clips and make prediction on whole audios. However, the generated pseudo labels are still noisy and unreliable,there is too many false positives. Finally, I used them to fix labels for audios with secondary labels. 

**Train Detail**

I used logmel spectrogram as input and randomly crop 5s clips from that. Then mixed sound augmentations will be applied to audios with no secondary labels, labels for audio with secondary labels will be adjusted according to pseudo strong labels. 

Making a good validation is very difficult in this competition, since there are no reliable nocall samples and a single clip can contain multiple bird calls. I chose to use a similar mechanism with site_3 audios in testset and calculated record-wise F1 scores. 

Preprocessing: MelSpectrogram-> ToDB -> Normalize -> Resize

DataAugmentation: GaussianNoise, BackgroundNoise, Shift, Drop, Clipping,

Backbones: Resnest50, Regnety_040

Single fold Score: 0.654/0.591/CV 0.75

#### Ensemble and TTA

In my best submission, the models are 5fold resnest50d(256x512) + 5fold resnest50d(320x768)+ 4fold regnety_040 (224x512).

I used two shifted versions of sound clip as TTA, both with half hop length. However, they make no difference on LB as well as on my CV.

#### What works

Using secondary labels, record-wise F1 increases, but clip-wise F1 decreases. 

Train longer make result stable. Train with 40 epochs can also get decent scores, but the score is varying a lot. Training 80 epochs results in higher CV and stable scores in LB.

#### What doesn't work

Simulate nocall sound and train a binary classifier. There is a huge domain gap between simulated ones and real ones. The classifier just finds a shortcut. My classifier prediction is nearly all positive on competition data. 

It seems there is an additional mp3 compression for testset. I tried to finetune on compressed audios. It improves some score on public LB but not on private LB.

Mixup make my result worse.

Using more mel bins can improve local CV, but hurts LB.
