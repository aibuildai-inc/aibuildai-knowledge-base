# 9th place solution

Competition: siim-acr-pneumothorax-segmentation
Rank: #9
Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/108060

Hello everyone.
Firstly thanks to the host and Kaggle for this interesting competition, and congrats to all the winners. My solution is quite chaotic, so thanks also to everyone who can read it to the end:)

## First attempts(not very successful):
At the beginning I decided to use a two-stage approach in this competition - train separate classification(pneumothorax/non-pneumothorax) and segmentation models. 
Therefore, I trained several common classification models: resnet34, resnet50, seresnex50, seresnext101, dpn98, they all gave approximately the same accuracy ~0.9. And I could not improve this result  in any way.  After that I trained several unet models (only on images containing pneumothorax) with different backbones and got the score in the gold zone. But this approach was not very stable and results on LB did not correlate with my local CV.
And after several unsuccessful attempts to improve this solution, I was disappointed and stopped participating in this competition for about a month.

## Final solution: 
After returning to the competition, I decided to try a slightly different approach, which eventually has became my final solution. It also has two steps:

### Classification:
For this purpose at that time I decided to use Unet model. And determined whether there are pneumothorax in the pictures by the threshold of the number of pixels. Based on my validation the best threshold was 2000. 
Backbone: seresnext50
Data: For this step I used all images and balanced batches(pneumothorax/non-pneumothorax) it greatly accelerated convergence
Data splits: 5 folds and 10 folds stratified by pneumothorax area.
Input size: 768x768
Loss: BCE
Augmentations: hflips, rotations(up to 10 degree), random brightness, contrast and gamma, blur
Lr scheduling: reduce lr on plateau with patience=3 epochs.

This approach gave a pretty good result,  but there were remained a rather large number of False Negative examples. Therefore, I decided to use classification models that I trained in the beginning and marked additional pictures as containing pneumothorax if all models attributed them to such.

### Segmentation:
A core element at this stage was also Unet model) But this time, model predictions were made only in pictures that were identified as containing pneumothorax in the сlassification stage.
Backbone: seresnext50
Data: only images containing pneumothoraxes
Data splits: 5 folds and 10 folds stratified by pneumothorax area
Input size: 928x928, 768x768
Loss: BCE + Dice
Augmentations: same as in the classification stage
Lr scheduling: reduce lr on plateau with patience=5 epochs.

Update:
Code https://github.com/scizzzo/kaggle-siim-pneumothorax
