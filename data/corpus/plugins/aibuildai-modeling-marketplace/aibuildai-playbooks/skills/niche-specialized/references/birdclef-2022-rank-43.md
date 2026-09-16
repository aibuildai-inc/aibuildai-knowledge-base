# 44th place solution

Competition: birdclef-2022
Rank: #43
Source: https://www.kaggle.com/c/birdclef-2022/discussion/327118

Thank you for organizing such an interesting competition. It was a great learning experience for me.

## Short Summary
My solution is SED, based on the one published by @hidehisaarai1213 in BirdCLEF2021.
This ranking was achieved without modifying the model, but by devising a new data set and changing the backborn.
A big thank you to hidehisaarai1213 for publishing this wonderful notebook!

train : https://www.kaggle.com/code/hidehisaarai1213/pytorch-training-birdclef2021-starter
simple inference : https://www.kaggle.com/code/hidehisaarai1213/pytorch-inference-birdclef2021-starter
infer between chunk : https://www.kaggle.com/code/hidehisaarai1213/birdclef2021-infer-between-chunk

## Preprocessing
As you all know, the lable this time was Imbalanced, so I adopted the approach of acquiring the sound source multiple times if the length of the sound source was short.  
I have expressed the number of labels on three levels as follows.  
Few tripled and VERY FEW quadrupled their data and studied.  
As I will explain later, the data used for training is a section of the sound source(20sec), so I hypothesized that it could be used multiple times by changing the starting point.  



MANY : 'skylar', 'houfin', 'jabwar', 'warwhe1', 'yefcan', 'apapan','iiwi', 'hawcre'  
FEW : 'hawama', 'omao', 'barpet', 'akiapo', 'elepai', 'aniani'  
VERY FEW : 'hawgoo', 'ercfra', 'hawhaw', 'hawpet1', 'puaioh', 'crehon','maupar'  

## Augmentation
Weak augmentation for MANY and strong augmetation for FEW. Strong means that it transforms with a high probability.  

```
augment_strong = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    AddGaussianSNR(p=0.5),
    Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=3, max_snr_in_db=30, p=0.5),
    AddShortNoises(noise_dir, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

augment_weak = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
    AddGaussianSNR(p=0.2),
    Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.2),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),
    AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=3, max_snr_in_db=30, p=0.2),
    AddShortNoises(noise_dir, p=0.2),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.2),
])
```

## Training
First of all, if I run the above hidehisaarai1213's notebook without preprocess with backbone as tf_efficientnet_b0_ns, I get 0.6406 for public and 0.6843 for private.    
* epoch -> 30
* loss -> BCEFocal2WayLoss
* backbone -> tf_efficientnet_b0_ns
* inf threshold -> 0.02

From here, adding the above preprocess and running it, public was 0.7025 and private was 0.6916. Since public went up significantly, I determined that the strategy of increasing FEW and VERY FEW was so effective.  

Adding a second label further increased the score from this point. 0.7053 for public, 0.6871 for private.

That I tried but it did not lead to an increase in PUBLIC scores. but I did use it in an ensemble.
* more increase FEW and VERY FEW. Increased FEW by 4x and VERY FEW by 7x.  
* Increased train sound source from 20 sec to 40 sec.
    * I thought the longer section would contain more call, but it didn't WORK.
* Change backbone to tf_efficientnet_b3_ns.

What worked best.  
* "nocall" was added to label and increased from 152 to 153 classes.
    * nocall data found from here https://www.kaggle.com/datasets/kami634/ff1010bird-duration10
    * This brings the score up to private 0.7473 public 0.7572


## Choosing subs
All scores above are obtained from fold0 only. The combination of fold and weight, I reached public 0.7730 private 0.7600, but I could not choose this sub because I considered that this was overfitting to public, using only fold0 and fold2 out of 5 unfolds.  
I chose a sub that mixed all folds equally to try to reduce overfitting to the public.
