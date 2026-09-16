# 2nd place solution - EfficientNetB0+augmentations+smart tiling

Competition: mayo-clinic-strip-ai
Rank: #2
Source: https://www.kaggle.com/c/mayo-clinic-strip-ai/discussion/358089

Thanks to Mayo Clinic for preparing this competition and to Kaggle for hosting it. And I'm happy that it was hosted as a code competition, because from my opinion it helps a lot to vanish the gap between kaggle competitions and real life business problems. Also I would like to congratulate guys from the first place because their solution seems much more complex and interesting than mine. 
At the same time sorry to see so low scores on the top comparing with random solution. I'm afraid that this competition might not help guys from Mayo to solve this very important problem and this is really sad.
Looking at the private leaderboard it seems like it was all about proper validation and total ignoring public scores (and a bit of luck of course). So let me quickly describe my solution.


**Data preparation**
For training I used all train images except 5 too-blurry. And also all other image with label 'Other'. I mixed them with LAA images in order to increase the amount of samples for that class, because in general I just wanted to distinguish CE from all other types.
Process step by step:
1. Resize images by factor 24. I also tried 8, 16, 32, 48, but 24 showed the best final metric;
2. Use line-to-line pixelwise difference to calc for each 28x28 block does it have blood or background;
3. For each 224x224 block with stride 28 check does it have blood or background;
4. Deduplicate tiles: remove all blood tiles that intersect by more than half;
5. Get random 20 if we have more tiles than that;
6. No color normalization. I've tried several approaches, but all of them worked worse than strong color jitter augmentation.

**Validation process**
First of all I quickly trained a small model on the same data to classify tiles by center_id. This models easily showed above-random performance (sorry, I lost exact metrics) and it was a signal for me that model potentially might overfits on similar clinics. Also [here](https://www.kaggle.com/competitions/mayo-clinic-strip-ai/discussion/347095) competition host mentioned that data was collected from 18 institutes, but in train data we have only 11. So either test has 7 other clinics or it was about data from other folder somehow. Anyway I decided to use cross-validation based on clinics. In order to handle "small-size" clinics (small test size might leads to unstable metric and potential overfit) I joined them into groups. So each group has at least 90 examples.
`(11,), (4,), (7,), (1, 5,), (10, 3), (6, 2, 8, 9,)`

**Model training**
During my experiments I had a lot of issues with overfitting and stability. So I tried to choose as simple approach as possible.
I used efficientNetB0 pretrained on imageNet as a backbone and just one FC layer as a head. Also backbone was frozen during the whole training. Model predicts only one probability "is it CE type". So basically I trained only 513 parameters. 
In order to stratify training dataset I used upsampling and also I found it useful to upsample it 4 more times (probably, it only helps with bigger batch size). Also I multiplied test dataset 20 times: if some image has 20 tiles then all of them will be used in test once, if less then some tiles from that image could be presented more than once.
Other bullet points:
- Ensemble of 6 models as a final model (one model from each CV fold);
- Bigger batch helps: 64 was my final choice;
- Learning rate is important;
- Learning process was still a bit unstable so I trained 3 models for each CV fold and got the best one;
- BCELoss as a loss function;
- Competition metric with (0.5, 0.5) weights as a main and the only metric;
- If image has no tiles (for example, too small or too-blurry) then replace their final score with 0.5.

**Data augmentations**
For my approach it was the most important part so I decided to write about it in a separate block.
I used 3 types of augmentations during the training:
1. Random flips - obvious;
2. Random adjust sharpness - by some reason helped a lot, but blur doesn't at the same time;
3. Strong color jitter (brightness=0.2, saturation=0.5, hue=0.5) - the most important one, because helped to fix color difference problem.
And also used 2 and 3 during the test. I used seeds everywhere in order to make the process reproducible. 

**Results**
Competition target metric on my CV: 0.6373838583629989
Public LB metric: 0.70822
Private LB metric: 0.66421

**What didn't work for me**
- MIL;
- More complex models;
- Color normalization;
- Final predictions clipping;
- Grayscale images;
- Gram matrices from VGG19 (see style transfer model).

Code is available on my GitHub now: https://github.com/IlyaLos/mayo-clinic-strip-ai
Notebooks (just run one by one): 
https://www.kaggle.com/code/ilyalos/2nd-place-solution-tiles-generation
https://www.kaggle.com/code/ilyalos/2nd-place-solution-train
https://www.kaggle.com/code/ilyalos/2nd-place-solution-inference
