# Trust CV -- 1st Place Solution

Competition: tensorflow-great-barrier-reef
Rank: #1
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307878

Thanks to the organizers and congrats to all the winners and my wonderful teammates @nvnnghia and @steamedsheep 

This is really very unexpected for us. Because we don't have any NEW THING, we just keep optimizing cross validation F2 of our pipeline locally from the beginning till the end.

# Summary

We designed a 2-stage pipeline. object detection -> classification re-score. Then a post-processing method follows.

Validation strategy: 3-fold cross validation split by video_id.

## Object detection

* 6 yolov5 models, 3 trained on 3648 images and 3 trained on 1536 image patches (described below)
* image patches: we cut original image (1280x720) into many patches (512x320), removed boxes near boundary, then only train yolov5 on those patches with cots.
* modified some yolo hyper-parameters based on default: `box=0.2`, `iou_t=0.3`
* augmentations: based on yolov5 default augmentations, we added: rotation, mixup and albumentations.Transpose, then removed HSV.
* after the optimization was completed by cross validation, we trained the final models with all the data.
* all models are inferred using the same image size as trained.
* ensemble these 6 yolov5 models gives us CV0.716, in addition the best one is CV0.676.

## Classification re-score:
* crop out all predicted boxes (3-fold OOF) into squares wich conf > 0.01. The side length of the square is `max(length, width)` of the predicted boxes, then extended by 20%.
* we calculate the iou as the maximum of the iou values of each predicted box and GT boxes of this image.
* classification target of each cropped box: iou>0.3, iou>0.4, iou>0.5, iou>0.6, iou>0.7, iou>0.8 and iou>0.9 Simply put, the iou is divided into 7 bins. e.g.: `[1,1,1,0,0,0,0]` indicates the iou is between 0.5 and 0.6.
* during inference we average 7 bin outputs as classification score.
* then we use BCELoss to train those cropped boxes by size 256x256 or 224x224.
* a very high dropout_rate or drop_path_rate can help a lot to improve the performance of the classification model. We use `dropout_rate=0.7` and `drop_path_rate=0.5`
* augmentations: hflip, vflip, transpose, 45° rotation and cutout.
The best classification model can boost out CV to 0.727
* after ensemble some classification models, our CV comes to 0.73+

## Post-processing

Finally, we use a simple post-processing to further boost our CV to 0.74+.
For example, the model has predicted some boxes B at #N frame, select the boxes from B which has a high confidence, these boxes are marked as "attention area".
in the #N+1, #N+2, #N+3 frame, for the predicted boxes with conf > 0.01,  if it has an IoU with the "attention area" larger than 0, boost the score of these boxes with `score += confidence * IOU`

We also tried the tracking method, which gives us a CV of +0.002. However, it introduces two additional hyperparameters. We therefore chose not to use it.

# Little story

At the beginning of the competition, we used different F2 algorithms for each of the three members of our team, and later we found that for the same oof, we did not calculate the same score.
For example, nvnn shared an OOF file with `F2=0.62`, and sheep calculated `F2=0.66`, while I calculated `F2=0.68`.
We finally chose to use the F2 algorithm with the lowest score from nvnn to evaluate all our models.

https://www.kaggle.com/haqishen/f2-evaluation/script

Here's our final F2 algorithm, if you are interested you can use this algorithm to compare your CV with ours!

# Acknowledge

As usual, I trained many models in this competition using Z by HP Z8G4 Workstation with dual A6000 GPU. The large memory of 48G for a single GPU allowed me to train large resolution images with ease. Thanks to Z by HP for sponsoring!
