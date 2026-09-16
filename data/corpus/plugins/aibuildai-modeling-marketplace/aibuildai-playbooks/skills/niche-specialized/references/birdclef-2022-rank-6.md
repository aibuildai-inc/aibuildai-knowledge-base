# 6th place solution (human-in-the-loop)

Competition: birdclef-2022
Rank: #6
Source: https://www.kaggle.com/c/birdclef-2022/discussion/327187

First I want to thank this competition hosts and the Kaggle team for organizing such a interesting competition. And thank you to all the Kagglers.

## Overview
- Only using SED model
- Make clean data with [human-in-the-loop](https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/319813) for weak label
- Heavy post-processing (public LB 0.79 &rarr; 0.84)

Perhaps my SED is the same model as yours.
**Clean data** and **post-processing** is important in my solution.

## Clean data (important)
I made [annotations](https://www.kaggle.com/competitions/birdclef-2021/discussion/239911) in BirdClef2021. And this is effective. But hand labeling is time consuming.
Instead of hand labeling, I used human-in-the-loop method in this competition.
- First, train SED with primary label
- Second, extract high confidence SED predictions in training data
- I listen these prediction and judge correct or not correct
- Human verification (I answer yes or no)

I made 2,000 clean data by this human-in-the-loop. These data contain only scored bird.

## Training data
I used 3 type data.
- HIL_data (=human in the loop data)
  - contain [external data](https://www.kaggle.com/code/amandanavine/hawaiian-bird-species)
- other_data (not scored bird data(about 130species))
  - initial 3sec audio
  - The label is primary_label
- psuedo_data (scored bird, and not contain HIL audio)
  - initial 3sec audio
  - psuedo_label = 0.25primay_label + 0.25x1st_generation_model + 0.5x2nd_generation_model 

And training data is below ratio.
This is a best ratio.

HIL_data : other_data : psuedo_data = 1 : 4 : 1

## SED
- Backbone: eca_nfnet_l0, dm_nfnet_f0
- Only using "clipwise_output" (training & inference)
- [Basic augment](https://www.kaggle.com/code/shinmurashinmura/birdclef2022-basic-augmentation/notebook)
  - Time shift
  - Add pink noise and brown noise
  - Mix other audio dataset (ESC-50: frog, rain, airplane, crackling_fire)
- [SpecAugment++](https://www.kaggle.com/competitions/birdclef-2022/discussion/307880) (mixing Mel-spec is ESC-50)
- Label smoothing (alpha=0.1)
- Optimize with Adam
  - lr=0.0001
  - CosineAnealing (T=10)
- Epoch:30 (about 14hours in Colaboratory)
- Loss: BCEWithLogitsLoss
- Input (train & inference): 5sec
- STFT resolution:250x254
  - window_size: 1024
  - hop_size: 630
  - mel_bins: 250
  - fmin: 50
  - fmax: 14000

## Ensemble
- Ensemble is a little impact in this competition
- I compared voting vs average.
  - Voting is good score a little.
- Finally, I used voting with 10 models.

## Post-Processing (important)
My Post-Processing is similar with [12th place solution](https://www.kaggle.com/competitions/birdclef-2022/discussion/326979).
 
#### prediction time shift (=predition_TA)
prediction_now = now + 0.5next_5sec + 0.25next_10sec + 0.5previous_5sec + 0.25previous_10sec
#### TTA
Let t be the target time. I used 3 type inputs.
- [t,t+5]
- [t-1,t+4]
- [t+1,t+6]

And these ouput is used in voting.
#### Threshold optimization (=ThreshO)
- First, I tuned threhold like [this](https://www.kaggle.com/competitions/birdclef-2022/discussion/318999). These threshold is constant.
- Second, I tuned threhold each species.
#### Threshold down (=ThershD)
If models detect a certain bird once in the 1 minute audio, I lower the threshold in the audio and infer it once more. 

```
score = prediction(thresh) # 1min prediction
down_species = ["hawhaw", "hawpet1", "maupar", "ercfra", "crehon", "puaioh", "hawcre"]

for bird in down_species:
    if np.max(score[bird]) > thresh[bird]:
        thresh[bird] = thresh[bird] - decrease

score = prediction(thresh) # 1min prediction once more
```

## Ablation study
|post-processing|**Public**/Private LB
|---|---|
|not using|**0.79**/0.76
|predition_TA|**0.78**/0.75
|predition_TA + TTA|**0.78**/0.74
|predition_TA + TTA + ThreshO|**0.82**/0.77
|predition_TA + TTA + ThreshO + ThreshD|**0.84**/0.80

## Not working
- AST (equally SED)
- PCEN (equally Mel-spec)
- [ImportantAug](https://www.kaggle.com/competitions/birdclef-2022/discussion/307880)
- ArcFace as few-shot learning method
- [STFT Transformer](https://www.kaggle.com/competitions/birdclef-2021/discussion/243360)
- Cooccurrence species with [these map](https://www.iucnredlist.org/species/22708583/128101101)
- Using BirdClef2021 dataset
