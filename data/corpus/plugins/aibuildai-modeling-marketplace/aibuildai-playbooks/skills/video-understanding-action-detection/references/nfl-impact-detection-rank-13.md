# Team tara 13th place solution

Competition: nfl-impact-detection
Rank: #13
Source: https://www.kaggle.com/c/nfl-impact-detection/discussion/208801

[Update]

Our codes are released [here](https://github.com/kentaroy47/kaggle-nflimpact-13thplace).

First of all, thank you for the hosts for organizing this great challenge! 

Congrats to all the winners and big thanks to my wonderful teammates @tereka @hidehisaarai1213 @rishigami, it was a great opportunity to work with such talents.

We really wanted to celebrate a birth of a new GrandMaster but time shall wait! 

It was a big challenge for me too, since it was my first kaggle competition after my baby son was born..

# Overview
.png?generation=1609822336590711&alt=media)

Our solution is based on a 2-stage pipeline with the first stage detecting impact candidate boxes and the second stage classifying either the candidate box is true or false positive. 

- Detectors
We train two separate detectors with Endzone and Sideline videos, respectively. Training separate detectors held better CV in our experiments.

- PP of detections
.png?generation=1609822406670388&alt=media)

One of the challenge we faced here was that impact boxes other than the time of impact had to be filtered. To cope with this, we use time-nms, where we grouped boxes with iou>0.3 within a certain duration (30 frames) and only the median box was extracted, which should represent the time of impact and other candidates are filtered out. Also we filter boxes that do not exist in both End or Side video within certain frames.

The output of the first stage scores `Precision 0.2, Recall 0.6, F1 0.2`. The parameters are tuned so the detector will achieve a high recall. 

- Classification

Like the 3rd place team's solution, we crop +- 4 frames of helmet images around the candidate box and  classify them using Resnet3D. We implemented this based on the torchvision's video classifier model. We ensemble 8 classifiers with horizontal-flip TTA. We trained the classifiers using both Endzone and Sideline videos.

- Threshold optimization

Since the impacts/video was about 10~20, we  adaptively control the prediction threshold so that the final prediction/ video will range within 15-25 predication per video. This makes the model robust to threshold tuning and was quite important since a single video with a lot of FPs can corrupt the entire score. 

The final CV was `Precision: 0.56, Recall 0.4, F1: 0.47` which has more room for improvement!

## Training the detector

Training the detector was quite hard for us, since there were lots of options. We use mixup+cutmix augmentations and oversample images with impact during training. We use images without impacts as well, which improved the precision of the detectors.

## Training the classifier
We were inspired by the [Deepfake 3rd place solution](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/158158) and used [3d CNNs](https://arxiv.org/pdf/1711.11248.pdf) for classifiers.

We trained the classifier with all helmet images cropped in 96x96 or 128x128 images. Since there are lots of False images, we used focal loss as criterion. Also mixup contributed to the CV as well.

Since we didn't have much time to train, the detector and classifier were single-fold models.

# What didn't work (for us

TTA, ensembling of detectors (TTA improved in PB but didn't work for CV and LB). Should have dug deeper here.

Classifiers with 27-channel input efficient nets. The CV/LB was much lower for us.

Using tracking data.

The CV for Endzone videos was about `0.43` and for sideline videos, `0.5`. We could not close this gap and think that Endzone videos are fundamentally hard because of occlusions between players and wonder what other teams did to close this gap.
