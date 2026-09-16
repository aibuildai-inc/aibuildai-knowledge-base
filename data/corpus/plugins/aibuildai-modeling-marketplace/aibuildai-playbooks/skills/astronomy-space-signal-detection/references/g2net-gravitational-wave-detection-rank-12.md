# 12th place solution [SiN Nakaism924]

Competition: g2net-gravitational-wave-detection
Rank: #12
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275356

First of all, I would like to thank my teammates( @naoism, @hidehisaarai1213, @yasufuminakama, @sinpcw ) for competing with me. I would also like to thank kaggle and EGO for organizing an interesting competition.
I explain my approach. My teammates will add their own approaches in the comments!
* Y.Nakama part ( https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275356#1529297 )
* SiNpcw part ( https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275356#1529331 )
* Naoism part ( https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275356#1529412 )
* Hidehisa part ( https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275356#1529639 )

## hirune part intro
I joined this competition after the SETI competition was over. This task is very similar to the SETI competition task. This task is very similar to the SETI competition task, so I reused most of the SETI code.

### preprocessing
1. Divide the all wave form by 4.6152116213830774e-20 (max(abs) of the entire train and test data).
2. Use nnAudio to run CQT. Randomly select one of flattop, blackmanharris, or nuttall to the window. This can be used in TTA. Finally, I also added CWT to this.
3. The spectrograms are combined in the frequency direction and input to the model as a single-channel image
4. Resize to 384x512 before inputting into the model, and normalize the spectrogram of the entire data set in mean, std

### Augmentation
* Before converting to a spectrogram, mixup as follows.
```
x = x1 + x2
y = y1 + y2 -(y1*y2)
```
* Randomly roll shift in time direction

### train setup
* model: timm tf_efficientnet_b4_ap
* optimizer: Adam
* Lr: 0.001
* scheduler: CosineAnnealingLR

### pseudo label
Using pseudo-labels showed some improvement, but not a lot.

### stacking
Using other team members models, and finally stacking 137 models with NN or XGB, there was a significant score increase. Other models include 1dcnn, swin transformer and efficientnet b5-8.
Team members will explain these details later.
