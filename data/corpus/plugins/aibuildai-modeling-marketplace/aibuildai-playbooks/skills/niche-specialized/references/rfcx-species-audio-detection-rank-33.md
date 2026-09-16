# 33rd place solution - SED model

Competition: rfcx-species-audio-detection
Rank: #33
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220753

Congrats all the winners.
I would like to thank the organizers for hosting a fun competition.

Our team's solution is an ensemble of seven models.
The ensemble includes three SED models and four not-SED models.
I will discuss the single model with the best score among them. (Public: 0.919/Private: 0.927)

### Model
My model is based on the SED model described in @shinmura0's discussion.
https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/211007

* Feature Extractor:EfficientNet-B3
* Loss Function:BCELoss
* Optimizer:SGD
* Scheduler:CosineAnnealingLR
* LR:0.15
* Data Augmentation:denoise
* CV: 4Fold multilabel-stratifiedkfold

My model is characterized by a very large learning rate, and as I reduced the learning rate from 0.15, the accuracy decreased.
This is contrary to my experience.

I thought that data expansion would be very effective, and I tried various data expansions, but most of them did not work.

The only data enhancement that worked was the denoising that @takamichitoda introduced in the discussion.
In the discussion, he de-noises all the data, but it worked when extended with p=0.1.
https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/214019

### Training data
The training data was randomly cropped from a meruspectrogram at 60/9 seconds.
I experimented with crop sizes ranging from 60/2 seconds to 60/20 seconds, and found that 60/9 and 60/10 gave good results.
Most of the discussions used a 60/10 second crop, but I think a smaller size would have reduced the probability of including missing labels.

### Validation/Test data
For the validation and test data, I used the same 60/9 seconds as the training date.

I create a total of 17 images by sliding the size of 60/9 seconds by half.
Make a prediction for each of the 17 images, and use the max value of the probability of occurrence of each label as the prediction value.



The validation was very time consuming, as we needed to infer 17 images to validate 1 clip.

### Didn't work.
Pseudo labels (train/test, soft label/hard label)
Prediction model for class3
　-> The accuracy of 3 was extremely low, so I tried to create a model specific to 3, but it didn't work.
