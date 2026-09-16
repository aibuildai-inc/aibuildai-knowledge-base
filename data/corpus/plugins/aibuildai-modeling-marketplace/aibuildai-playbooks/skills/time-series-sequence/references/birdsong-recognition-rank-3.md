# 3rd place solution

Competition: birdsong-recognition
Rank: #3
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183199

Very glad to end my journey to Kaggle GM with a 3rd place, congratz to my teamates and everybody who made it to the end !

Our solution has three main aspects : data augmentation, modeling and post-processing

#### Data Augmentation

Data augmentation is the key to reduce the discrepancy between train and test. We start by randomly cropping 5 seconds of the audio and then add aggressive noise augmentations :
- Gaussian noise

With a soud to noise ratio up to 0.5

- Background noise

We randomly chose 5 seconds of a sample in the background dataset available [here](https://www.kaggle.com/theoviel/bird-backgrounds). This dataset contains samples without bircall from the example test audios from the competition data, and some samples from the freesound bird detection challenge that were manually selected.

- Modified Mixup

Mixup creates a combination of a batch `x1` and its shuffled version `x2` : `x = a * x1 + (1 - a) * x2` where `a` is samples with a beta distribution. 
Then, instead of using the classical objective for mixup, we define the target associated to `x` as the union of the original targets. 
This forces the model to correctly predict both labels.
Mixup is applied with probability 0.5 and I used 5 as parameter for the beta disctribution, which forces `a` to be close to 0.5.

- Improved cropping 

Instead of randomly selecting the crops, selecting them based on out-of-fold confidence was also used. The confidence at time `t` is the probability of the ground truth class predicted on the 5 second crop starting from `t`.

#### Modeling

We used 4 models in the final blend :

- resnext50 [0.606 Public LB -> 0.675 Private] - trained with the additional audio recordings.
- resnext101 [0.606 Public LB -> 0.661 Private] - trained with the additional audio recordings as well.
- resnest50 [0.612 Public LB -> 0.641 Private] 
- resnest50 [0.617 Public LB -> 0.620 Private] - trained with improved crops 

Turns out that training with more data was the key, and that both our resnest were overfitting to public LB. Thanks to people who shared the datasets ! 

They were trained for 40 epochs (30 if the external data is used), with a linear scheduler with 0.05 warmup proportion. Learning rate is 0.001 with a batch size of 64 for the small models, and both are divided by two for the resnext101 one, in order to fit in a single 2080Ti.

We had no reliable validation strategy, and used stratified 5 folds where the prediction is made on the 5 first second of the validation audios.

#### Post-processing

We used 0.5 as our threshold `T`.

- First step is to zero the predictions lower than `T`
- Then, we aggregate the predictions
  - For the sites 1 and 2, the prediction of a given window is summed with those of the two neighbouring windows. 
  - For the site 3, we aggregate using the max
- The `n` most likely birds with probability higher than `T` are kept
  - `n = 3` for the sites 1 and 2
  - `n` is chose according to the audio length for the site 3.

#### Code

Everything is fully available :

- Inference : https://www.kaggle.com/theoviel/inference-theo?scriptVersionId=42527667 
- Train in Kaggle kernels : https://www.kaggle.com/theoviel/training-theo-3 (code is a bit dirty but directly usable)
- Github : https://github.com/TheoViel/kaggle_birdcall_identification  (code is cleaned and documented)


#### Final words

As we had no proper validation scheme, private LB was really a coinflip for us. The top 3 is a nice surprise ! Our best submission is actually the 5-fold ResNext50 alone, which was quite unpredictable.

Thanks for reading !

(topic will probably be updated)
