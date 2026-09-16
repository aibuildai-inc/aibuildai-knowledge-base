# 50th place solution

Competition: birdclef-2022
Rank: #50
Source: https://www.kaggle.com/c/birdclef-2022/discussion/327217

Thank you to every competitor and host that held this competition.
This is my solution and what I tried.

## Things that I watched out
As I have posted in [this discussion](https://www.kaggle.com/competitions/birdclef-2022/discussion/321668), I believe that three aspects of this competition needed to be considered.

1. Get high CV score
This is a given, but a better model should be used to improve cross-validation scores.
I have the impression that many people used the improved PANNS model used in the Notebook published by @hidehisaarai1213 in the 2021 Bird Competition.  I also used this model and reached CV:0.789 (f1 Score), Public:0.739, and Private:0.711.
Then, in the middle of the competition, I switched to use another model (that I'll explain below.) and improved to CV:0.805(f1 Score), Public:0.7634, Private:0.7210 (best with a single model).

2.  Measures against Domain Shift  
There were several models with high CV but low LB. This is probably due to the existence of a domain shift between training data and test data. As a countermeasure, Augmentation, which inserts noise, was effective. In particular, Gaussian Noise and Pink Noise worked effectively.
Models trained with these augmentations got relatively low CV scores, but the Public scores improved.

3. Selecting the Appropriate Threshold
In this competition, since there were no Test Soundscapes (data collected under the same conditions as the test data), the threshold had to be set manually. This appropriate threshold was usually far from the value that was optimal during the Validation phase of the model, so tuning was an important step. Since the Public Score was the only available factor to choose threshold, we had no choice but to overfit the Public Score (despite this, LB did not shake as much as we expected).

## Solution 
[solution]

### Single model

As figure shows, the sounddata was converted into a 20-second segmented mel-spectrogram, and then Augmentation (Normalize, Pink Noise) was added.
Augmentation (model1: Normalize&PinkNoise, model2: Normalize&GaussianNoise&RandomVolume) was added to the input, and the features were extracted using ResNet101d(model1) and ResNeXt50d_32x4d(model2).
The output is a batch size × channel × frequency × time (four-dimensional).
Then, the features are compressed in the frequency and time directions using GeM pooling, and the result was returned to a two-dimensional tensor of batch size × channel.
Then fc layer is applied to obtain a 152-dimensional output.  
I also applied MixUp as augmentation with a probability of 0.25. I believe that the robustness of the model has been improved, albeit only by a small margin.  
As a loss function, I used Focal loss based on BCEWithLogitsLoss (https://www.kaggle.com/competitions/rfcx-species-audio-detection/discussion/213075 ).  
I also used SAM Optimizer based on Adam as the Optimizer (https://github.com/davda54/sam ).
By changing the Optimizer to SAM, the Public Score for the Sound Event Detection model improved by about 0.02 when the experiment was conducted under all conditions except for the Optimizer.

### Ensemble
Instead of passing the model-by-model average of the output predictions through the threshold, the logical OR or logical product of the true/false values after passing through the threshold was
submitted as the final prediction result. By applying an ensemble that takes Logical OR, I obtained an output result with a Private score of 0.7292, from the predictions whose private score are 0.7176 and 0.7008 (I we believe they worked effectively)

## what did not work
* Use complex models as backbone (e.g. Swin Transformer)  
* PaSST (https://github.com/kkoutini/PaSST)  
* Forcibly apply Multi head attention to the obtained features 
* Using the Psuedo label
