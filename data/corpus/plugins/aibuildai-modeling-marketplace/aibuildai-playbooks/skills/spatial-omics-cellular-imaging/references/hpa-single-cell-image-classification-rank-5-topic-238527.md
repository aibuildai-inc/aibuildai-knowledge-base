# simple idea of 5th, tito's part

Competition: hpa-single-cell-image-classification
Rank: #5
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238527

First of all I would like to thank my teammates @narsil and @tivfrvqhs5. I have had a very exciting 3 weeks teaming up with you guys.

I would like to share my model idea briefly here.

## Classifier
I created a classification model that predicts labels for each image (not for each cell).

For inference, I extracted the cells one by one and made their augmented images:
[inference image]

This allowed me to solve this weakly supervised task as a normal classification task which is same as HPA2018 competition.
This makes things very simple and allows me to reuse the HPA2018 solution.


## Segmenter
The official segmenter is inaccurate for cells near the boundary (cells with no visible nucleus tend to be connected to other cells).
To avoid this effect, I created a mmdetection model using cropped segmentation image.
[segmentation for training image]
Even using official segmenter as traing label, score of this mmdetection mode is improved.
In addition, this improved the inference speed and more ensembles enabled.
