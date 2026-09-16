# 18th place solution

Competition: rsna-str-pulmonary-embolism-detection
Rank: #18
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193404

*I am actually quite surprised I was able to do this well in the competition, I joined way too late and trained very few models. I only had one selected submission that luckily didn't time out. It is a 5-fold efficientnet-b3 followed by a 5-fold sequential model based on its features.*

### Introduction
My solution consists of a two steps pipeline, similarly to the overpowered baseline :
- Image level efficientnet-b3, trained to classify whether the image has PE. This model is then used to extract features for each slice of the CT scan.
- A sequential is trained on the features extracted by the CNN, it predicts both the image and exam level labels, and directly optimizes the competition metric.

I joined the competition 9 days before the end, with the overall motivation of doing something similar to the winners of the previous RSNA competition. 
I was able to quickly build the CNN pipeline, and started training a bunch some models. The issue was that those models would be really long to train on my hardware (1x RTX 21080Ti), so I had to improvise.

Then, it was about quickly engineering the second part of the pipeline and the inference code, which was far from easy. 

Shortly after I joined, the overly powerful baseline was released. I did not end up really using any of the components of it, but it was an additional motivation for me to keep pushing.
I was able to come up with my first submission one day before the deadline, which *somehow* scored 23rd on the public leaderboard. 

### Data

As I could not fit the 900 Gb dataset on my computer, I solely relied on the [256x256 jpgs](https://www.kaggle.com/vaillant/rsna-str-pe-detection-jpeg-256) extracted by @vaillant.
Thanks a lot for making it possible for people like me to join the competition.

I therefore didn't experiment at all with the windowing, and simply loading the data takes a huge part of my runtime.

### First level : Convolutional Neural Networks

#### Undersampling

The issue with 2D images constructed from CT scans is that images are very similar. Therefore it makes sense not to use every slice per patient. 
As there is about 400 slices per patient on average, one epoch would take ages and this is not a path I wanted to go. 
Therefore, I only used 30 images per patient at each epoch, this is done using a custom sampler. 

Once this was done, I was able to train the models for 20 epochs in approximately 9 hours, using a 5-fold grouped by patient validation.

#### Models

Models were trained as part of a classical binary classification problem, using the binary cross-entropy
First experiments were conducted with a ResNeXt-50 model as it is usually a reliable baseline. 
I then tried to switch to a bigger ResNext-101, but results were not bigger so I quickly gave up with big architectures.
The last model I trained is an efficientnet-b3, which was chosen because a batch size of 32 could fit on my GPU. 
It performed slightly better so I sticked with this model, and I had no time left to train other models.

The efficientnet was trained for 15 epochs using a linear scheduled learning rate with 0.05 warmup proportion. 

#### Augmentations

```
- albu.HorizontalFlip(p=0.5)
- albu.VerticalFlip(p=0.5)
- albu.ShiftScaleRotate(shift_limit=0.1, rotate_limit=45, p=0.5)
- albu.OneOf([albu.RandomGamma(always_apply=True), albu.RandomBrightnessContrast(always_apply=True),], p=0.5)
- albu.ElasticTransform(alpha=1, sigma=5, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, p=0.5)
```

### Second Level

#### Model

The model I used is a MLP + BidiLSTM one that predicts both the image and exam targets using the CNN extracted features as input. 
Two 2-layer classifiers are plugged on the concatenation of the output of the MLP and of the LSTM.
I used the concatenation of average and max pooling for the exam level targets.
In addition, multi-sample dropout was used for improved convergence. 

#### Training

The model was trained using the loss function that matches the metric. 
I also used stochastic weighted averaging for the last few epochs, once again to have a bit more robustness.
A single epoch took approximately a minute. 

The validation scheme is a normal 5-fold, and my CV scores were quite close to the 0.179 score I had on the public LB.


### Inference

My inference code is available here : https://www.kaggle.com/theoviel/pe-inference-2
I used clipping to make sure the label assignment rules were respected, which dropped my score of approximately `0.003`.


### Final words

Congratz to the winners, I'm pretty sure my solution is nowhere near what the top 10 has come up with and I'm really glad I was able to finish 18th. I wanted to tackle a medical imaging challenge for a long time but was always hesitating because of the dataset sizes.

Hopefully next time I don't procrastinate too much and join a bit earlier, I'm pretty sure I'll benefit a lot from teaming up with people and spending more time experimenting.

Also, the code is available on **GitHub**, although I still have some cleaning to do, and the ReadMe to complete  :  https://github.com/TheoViel/kaggle_pulmonary_embolism_detection

Thanks for reading !
