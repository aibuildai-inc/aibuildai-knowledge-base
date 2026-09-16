# 11th place solution

Competition: airbus-ship-detection
Rank: #11
Source: https://www.kaggle.com/c/airbus-ship-detection/discussion/71659

First of all congratulations to the winners!

**Data**

The main challenge of this completion, from my point of view, was very unbalanced data for ship/no ship, split/no split cases, very different types of images from distance, quality etc point of view.

For the training I took only images with ships from the training set and created two types of labels for them to prepare the separating of close ships:

* full ship contours
* only separation line

one channel for body, one contour or split

Examples:
![enter image description here][1]
![enter image description here][2]
![enter image description here][3]
![enter image description here][4]

On the beginning I used only split masks, but there were too few training images where the splitting needed and the model trained not so well. In final models I used only contour labels, but may be better idea would be to use both in different channels to improve the quality of the splitting. 

**Model**

**Classification (ship/no ship)**

First result of segmentation was not so good, as there were a lot of FP, mostly clouds and waves on empty images. As there were much more empty images than images with ships, I’ve decided to train a simple classifier for the first phase and use segmentation model only for images with ships in the second phase.

**Segmentation**

Encoder: Resnet34, se_resnext50

Decoder: hypercolumn, scSE, classification-based attention

**Training image augmentation**

From imgaug I used: flips(horizontal and vertical), 
PerspectiveTransform, CropAndPad, Affine(scale, translate_percent, rotate, shear), 
One of (ContrastNormalization, Color(Multiply, Grayscale))
One of (GaussianBlur, AverageBlur, MedianBlur, BilateralBlur, AdditiveGaussianNoise, ElasticTransformation)

**Training**

Optimizer: Adam 

Loss function: Lovasz(elu+1)

step1 30-60 epoch (with reduce on plateau)
step2 10-20 epoch fine tuning with smaller LR

9/10 images were used for training, 1/10 for validation. 

StratifiedKFold was used to split folds by ship size, close ships.

The training was running on the crops 224x224. Every batch was the mix of random crops around the centers of ships and totally random crops.

The validation was done on full size images.

**TTA**

Original image + vflip+hflip

**Postprocessing**

The watershed was used to split the labels

**Final ensemble**

The local validation score for se_resnext50 based models was much better, than Resnet34, but on public LB vice versa. Looks like better encoder se_resnext50 just overfitted on the this data for me.

The final score is a simple average of 3 x Resnet34 and 1 x se_resnext50.

**Hardware**

1 x 1080Ti

**Software**

Before the leak break I was using Keras, but during the TGS Salt moved to pytorch. Final models are only pytorch.

**Did not work for me**

In the postprocessing I tried to do the labels “more rectangle”, but it made the score only worse.

**Should try**

Another heavier encoders, like dpn, densenet, with more augmentations against overfitting. 


  [1]: https://storage.googleapis.com/kaggle-forum-message-attachments/421773/10682/c_edec35a72.png
  [2]: https://storage.googleapis.com/kaggle-forum-message-attachments/421773/10683/c_52554d6ee.png
  [3]: https://storage.googleapis.com/kaggle-forum-message-attachments/421773/10680/s_edec35a72.png
  [4]: https://storage.googleapis.com/kaggle-forum-message-attachments/421773/10681/s_52554d6ee.png
