# 27th place solution (0.483LB)

Competition: hpa-single-cell-image-classification
Rank: #27
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238439

Hi all! It was a very long and intensive 3 months)
First of all, many thanks to the organizers for this very challenging and interesting competition!

This was my first "real" competition at kaggle, so I'm glad to get a medal :-)

My solution consists of several parts:

1) **Cell segmentation**: I trained MaskRCNN on the HPASegmentator predictions. The only reason for this was the inference speed. MaskRCNN works really faster. A few experiments showed that solution score stayed the same when swapping HPASegmentator with MaskRCNN.

2) **Image Level classifiers**:

- EfficientNetB3 (1024x1024)
- EfficientNetB3 (only green channel) (1024x1024) (thanks @h053473666 for this idea)

3) **Cell level classifiers**:
    I used [PuzzleCAM paper](https://arxiv.org/abs/2101.11253) approach for creating CAMs. And then I used them to create pseudo labels (at this point the labels were smooth - floats in [0.0, 1.0]) for each cell within images.
    Next step was to manually set up thresholds for each class and get final pseudo labels for further training model.

- EfficientNetB3 with dropout (256x256)
- EfficientNetB3 with dropout (only green channel) (256x256)


4) **Final submission** I combined predictions from all 4 networks and get final predictions for each cell. The best result was with just an average of predictions from 4 models.

5) **Training details**
All classifiers were trained with focal_loss + lovazh_loss (like in the winning solution of @bestfitting in previous HPA competition).
Optimizer: Adam
Scheduler: ReduceOnPlatoue
I also used oversampling for rare classes and undersampling for too common classes.

Here is a summary:

[Summary]

Thanks for reading and good luck in future competitions!)
