# 17th place solution

Competition: rfcx-species-audio-detection
Rank: #16
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220450

Congratulations to all the winners and gold getters, I guess those teams that broke the 0.950 wall have found the essence of this competition, which we couldn't.  
<br/>
Firstly, thanks to the host for holding quite an interesting competition. Partly labeled classification is a challenging task, which made this competition more interesting than a simple bioacoustics audio tagging competition.
<br/>

Our solution is a ranking average of image classification models and SED models. [Y.Nakama](https://www.kaggle.com/yasufuminakama), [kaerururu](https://www.kaggle.com/kaerunantoka), and I worked a lot on SED models but couldn't break 0.90 until we merge with [Taku Hiraiwa](https://www.kaggle.com/thiraiwa). His model was based on image classification similar to [@cpmpml's model](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220304). We found that quite good , so we focused on improving image classification models in the remained days.
## Image classification models
It was [Taku Hiraiwa](https://www.kaggle.com/thiraiwa)'s idea to only use the *annotated* part of the train data. To do so, we crop the image patches from log-melspectrogram of train data based on the `t_min`, `t_max`, `f_min`, `f_max` information of train\_tp.csv and train\_fp.csv, and resized the patch to fixed shape (say, 320 x 320 or so). The cropping is performed on the fly through training, so the part we crop out is randomized along with time axis.
With these image patches we trained EfficientNet models, and monitored F1 score with threshold 0.5. 
Here's the other details of image classification models.

* image size: varies from (244, 244) to (456, 456) between models
* backbone: EfficientNetB0 - B5 (used [`timm`](https://rwightman.github.io/pytorch-image-models/) and used `tf_efficientnet_b<0-5>_ns` weights).
* augmentation: GaussianNoise, Gain, PitchShift of [audiomentations](https://github.com/iver56/audiomentations) on raw waveform. Also HorizontalFlip also had positive impact on LB slightly, so we used (but don't know why it worked).
* AdamW optimizer with linear warmup scheduler
* BCEFocalLoss
<br/>
In the end, we trained a stacking model that takes the output of models below which achieve public 0.942:

1. `tf_efficientnet_b0_ns` image size 244
2. `tf_efficientnet_b0_ns` image size 320
3. `tf_efficientnet_b0_ns` image size 456
4. `tf_efficientnet_b1_ns` image size 456
5. `tf_efficientnet_b2_ns` image size 456
6. `tf_efficientnet_b3_ns` image size 456
7. `tf_efficientnet_b4_ns` image size 456
8. `tf_efficientnet_b5_ns` image size 456

## SED models
All of our SED models use the head architecture introduced in [PANNs repository](https://github.com/qiuqiangkong/audioset_tagging_cnn). The CNN encoder is either EfficientNet or ResNeSt, and they are trained with weak/strong supervision. We tried a lot of things on this, but couldn't find factors that consistently work well - which means the LB scores varied quite randomly w.r.t CV score.
Our best SED model is rank average of 11 models(public: 0.901, privte: 0.911) below - each of them differs slightly so we describe the difference briefly.

### 2 x kaerururu's model (public: 0.882, 0.873)

* Based on public starter SED (EffnB0) notebook (https://www.kaggle.com/gopidurgaprasad/rfcx-sed-model-stater)
* 3ch input
* 10sec clip
* waveform mixup
* some augmentation (audiomentations)
* pseudo-labeled datasets (add labels on tp data)
* trained with tp and fp dataset (1st training)
* trained with pseudo-labeled tp (2nd training)
* tta=10

### 5 x arai's model (public: 0.879,0.880, 0.868, 0.874, 0.870)

* Based on Birdcall's challenge 6th place (https://github.com/koukyo1994/kaggle-birdcall-6th-place)
* ResNeSt50 encoder or EfficientNetB3 encoder
* AddPinkNoiseSNR / VolumeControl / PitchShift from [My Notebook](https://www.kaggle.com/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english)
* tp only
### 4 x Y.Nakama's model (public: 0.871, 0.863, 0.866, 0.870)

* Based on Birdcall's challenge 6th place (https://github.com/koukyo1994/kaggle-birdcall-6th-place)
* ResNeSt50 encoder or ResNet200D encoder
* mixup & some augmentations
* 2nd stage training
  * 1st stage: weighted loss for framewise_logit & logit
  * 2nd stage: loss for logit
* tp only
