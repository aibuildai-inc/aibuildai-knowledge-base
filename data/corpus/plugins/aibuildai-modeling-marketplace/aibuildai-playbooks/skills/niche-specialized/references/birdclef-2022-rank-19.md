# 19th place solution, single CNN model

Competition: birdclef-2022
Rank: #19
Source: https://www.kaggle.com/c/birdclef-2022/discussion/327175

To begin with, thanks to the Kaggle, group of organizers and other participants. 

Here you could find my solution based on the single ResNeSt50d model trained on 5 folds. One of the hardest parts in this competitions was validation. In [Cornell Birdcall Identification competition](https://www.kaggle.com/competitions/birdsong-recognition/) our team suffered badly from the shakeup thus I have tried to prevent such painful experience at all costs in this competition. 

I have cut the description of my solution as it was possible, so if you have any questions feel free to ask :)


#### **Data:**
- **Pre-training:** 
    - **BirdCLEF2021 (train_short_audio) + BirdCLEF2022 (train_audio)**
- **Finetuning data:** 
    - **BirdCLEF2022 (train_audio)**
- **Validation data:**
    - **BirdCLEF2021 (public + private lb scores)** were used to compare models after the pre-training stage + to check some of the post-processing tricks;
    - **BirdCLEF2022 (validation folds)** were used for model selection based on OOF scores calculated only for scored birds (metric: label ranking average precision);
    - **BirdCLEF2022 (public lb)** was used to tune threshold;
    - **Artificially created soundscapes** were used for the model selection.
        This dataset was created using mixtures (with different SNRs) of scored birds from the BirdCLEF2022 (train_audio) and nocalls from the part of BirdCLEF2021 train soundscapes, which weren't used as train augmentations. The segments with scored birds were filtered using nocall detector by the 0.9 percentile. I have tried hard to make this data similar to BirdCLEF2022 (public lb) data in order to tune threshold and try post-processing tricks properly, but all attempts were far from the reality. However, I found this created dataset slightly useful for the model selection (in addition to the validation folds)
            
#### **Pre-processing:** 
- 128-dim melspec + normalization, duration: 7 seconds (I have modified [notebook](https://www.kaggle.com/code/kneroma/birdclef-mels-computer-public) from @kneroma)

#### **Models:**
- The only one ResNeSt50-based CNN, 5 folds based on StratifiedGroupKFold (grouped by author);
- Resnet-based nocall detector (used only for pre-training, training losses and for the creation of artificial soundscapes).

#### **Loss:**
- BCE loss on classification head for bird classes:
    - Weights for scored birds were multiplied by 1.5 (slight boost)
    - Weight for each item was multiplied by the call probability from nocall detector [idea from top1 BirdCLEF2021 solution](https://www.kaggle.com/c/birdclef-2021/discussion/243304) (medium boost)
    - Weight=0.3 for secondary classes;
    - Label smoothing;
- Two BCE losses (weighted 0.15 for each) on classification heads for two levels of a species taxonomy: family and order ([idea from Mixit paper](https://arxiv.org/pdf/2110.03209.pdf))

#### **Augmentations:**
- Background noise from ff1010 and from the part of BirdCLEF2021 train soundscapes (this augmentation was used only for pre-training);
- MixUp;
- Pink noise;
- Bandpass noise; 
- [SpecAug based on the mixture masking](https://arxiv.org/pdf/2103.16858v3.pdf) (thanks to @shinmurashinmura for sharing)

#### **Post-processing:**
- Constant threshold;
- The probabilities of 6 underrepresented birds (based on the total duration of each bird and on the outputs of model) were multiplied by 2.0  (it has boosted my both public and private scores by 0.03);
- If the bird is found in the 5-second segment, the probabilities will be increased for this bird in two neighboring segments from each side (public and private scores were increased by ~0.02);
- Nocall threshold.
    

#### **The methods that didn’t work or worked worse:**

To be honest, there is gonna be a very long list of things, but I will reduce it to the most promising methods (based on my opinion, in descending order)

- [Mixit](https://bird-mixit.github.io) (Their separation model + my classification model have given 0.75 on private lb);
- [PASST](https://arxiv.org/abs/2110.05069) (and other transformer-based models);
- [PANNs];
- Finetuning on 79 classes only (scored birds + all birds which occur in the training audio with scored birds);
- Finetuning on 21 scored classes only;
- Quantile-based threshold for each soundscape;
- Dynamic duration for the audio during training (for each batch the duration is sampled from [5, 10] interval for example);
- Train each model for a specific daytime (for example, one model for calls in the morning, second - for calls at night, etc..);
- Using the nocall detector for the submission directly.
