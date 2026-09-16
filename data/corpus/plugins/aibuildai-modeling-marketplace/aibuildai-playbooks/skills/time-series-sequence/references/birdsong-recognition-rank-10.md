# 10th Place Solution

Competition: birdsong-recognition
Rank: #10
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183407

## Acknowledgments
First of all thanks for the organizers for this challenging and fascinating competition.
I am grateful for the relatively quick answers in the discussions (external data, domain knowledge,
it was certainly helpful.

Special thanks to
* Hidehisa Arai for showing how to use PANNS effectively
* Qiuqiang Kong et al for PANNS repo and pretrained models
* Vopani for scraping and maintaining the external XenoCanto dataset
* Jan Schlüter and Mario Lasseck for their previous BirdCLEF winning papers   

## The beginning
Seven years ago I already participated in bird song detection challenge at kaggle.
That time we had only a few hundred 10 seconds recordings to train on and similar multiclass multilabel problem but with only 19 species.
I was able to win that competition with Computer Vision template matching and Random Forests.
I thought it would be a quick & easy experiment to beat that with the available 40K bird recordings and all the available pretrained image net models.
Well it was not.

Possible reasons
* Different sample rate (16kHz vs 32kHz)
* Soundscape vs Xenocanto
* MLSP train-test split was random split across soundscapes it was possible to overfit to the same recording (e.g. crickets & rain -> Hermit Warbler))
* Different spectrogram/noise distribution

## Ornithology and LB Probing
In the beginning of the competition I was not able to submit meaningful results so I tried to understand the North American bird population better.
Just by submitting individual birds one would expect ~0.001 LB score with equal bird distribution.
From ebird.org observation data I was able to rank the 264 species based on their unique observations during the last two years.
It does not necessary reflect the distribution in the Public/Private test set but I found it better than the number of XC recordings.

E.g.
* Red Crossbill 1223 XC recordings 84K observations 0.000 LB Score
* White-crowned Sparrow 474 XC recordings 1.1M observations 0.1 LB Score (!)

## Data Preparation
Resampling everything to 32kHz and splitting the first 2 minutes of each recording to 10 second duration chunks and saving them as .npy arrays.
I used the extended dataset a
4 fold cross validation was used stratified on the author-created at to try to avoid same birds in different folds.
For early stopping I saved the best  weights based on XC validation and BirdCLEF Validation as well.

## Augmentation
I only used additive noises.
* freefield1010
* warblrb10k
* BirdVox-DCASE-20k
* Animal Sound Archive Published by Museum für Naturkunde Berlin ()

Probably should have tried synthetic noise generation too.

## Architecture
I ended up with slightly modified CNN14 (128 mel bins, mean/std standardised)
They were relatively quick to train on Nvidia Tesla T4, training a single model took 3-8 hours.
I tried PANN ResNet38, Cnn14_DecisionLevelAtt or ImageNet pretrained ResNet50 but without proper validation I got mixed results...

The dataloader handled the additive augmentations for the waveforms
* Add multiple possible 1-2-3 birds with multi-class setting
* Add same class chunks
* Add noise
* Add animal sound

the GPU created the spectrograms for the batches.

 
## Blending
During the last weekend I fixed my whole training pipeline and rented V100 to retrain a few final models and create some additional experiments.
On Sunday evening I submitted my first blended model with quite disappointing 0.570 Public Leaderboard result.
Actually it would have been enough for my final 10th place with Private score 0.649 Private Score.

In the las two days I made some desperate submissions with more models, varying thresholds
(e.g. increasing the thresholds of west coast birds, reducing the thresholds for common birds)  
They did improve my public LB score and fortunately they did not improve nor hurt the private score.
Probably with a few more submissions I would start to overfit...

## What did not work but I thought it would...
* Using secondary labels to improve primary labels with oof predictions
* Using a separate nocall classifier
* Utilizing the apriori knowledge that some birds are more frequent than others
* Blending normalized and unnormalized models
* Mixup
* More than 128 Mel bins
