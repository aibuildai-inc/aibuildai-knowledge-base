# 1st Place Solution

Competition: severstal-steel-defect-detection
Rank: #1
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114254

Congratulations to all winners in this competition! This is my first competition in Kaggle and also the first competition for my teammates @zhuhongbo  and @paffpaffyu. They were my former classmates in the university. We are all very surprised that we won the competition. 

First of all, thanks to @hengck23  for sharing his insightful ideas. We followed some of his posts and got a huge improvement. And also thanks to @lightforever, our best final solution used several models from him. And thanks to @pavel92  for his convenient segmentation library.

**Classification**
Classification is an important part of this competition. Even though classifiers can only slightly improve your score after 0.915 in public LB, they can work as a preliminary screening by filtering out around half of images with no defects. This will enable us to ensemble more models in segmentation part. 
We trained our classifiers with random crop of 224x1568 and do inference on full size. This random crop gives slightly improvement on accuracy. 
Augmentations:
Randomcrop, Hflip, Vflip, RandomBrightnessContrast (from albumentations) and a customized defect blackout
Since this is a sematic segmentation task, we know exactly where the defects are. As a result, these defects components can be randomly blacked out and the label for this image will also change from 1 to 0 if all defects are blacked out. This augmentation indeed works on local CV and public LB. Here are some graphs of the training process of a ResNet34 classifier.



Batchsize: 8 for efficientnet-b1, 16 for resnet34 (both accumulate gradients for 32 samples)
Optimizer: SGD
Model Ensemble:
3 x efficientnet-b1+1 x resnet34
TTA: None, Hflip, Vflip
Threshold: 0.6,0.6,0.6,0.6

**Segmentation**
We have to admit that we used models from @lightforever these models improved our score from 0.907 private LB to our current score. 
Train data: 256x512 crop images 
Augmentations: Hflip, Vflip, RandomBrightnessContrast (from albumentations)
Batchsize: 12 or 24 (both accumulate gradients for 24 samples)
Optimizer: Rectified Adam
Models: Unet (efficientnet-b3), FPN (efficientnet-b3) from @pavel92 segmentation_models_pytorch
Loss: 
BCE (with pos_weight = (2.0,2.0,1.0,1.5))
0.75*BCE+0.25*DICE (with pos_weight = (2.0,2.0,1.0,1.5))
Model Ensemble:
1 x Unet(BCE loss) + 3 x FPN(first trained with BCE loss then finetuned with BCEDice loss) +2 x FPN(BCEloss)+ 3 x Unet from mlcomp+catalyst infer
TTA: None, Hflip, Vflip
Label Thresholds: 0.7, 0.7, 0.6, 0.6
Pixel Thresholds: 0.55,0.55,0.55,0.55
Postprocessing:
Remove whole mask if total pixel &lt; threshold (600,600,900,2000) + remove small components with size &lt;150

**Pesudo Label**
We did 2 rounds of pseudo labels in this competition. The first round is generated from a submission with 0.916 public LB, maybe it is too early? The second round was done several days before the end of this competition, generated from a submission with 0.91985 public LB. With pseudo label and public models, we finally improved from 0.91985 to 0.92124 on public LB and from 0.90663 to 0.90883 on private LB. 
The pseudo labels are chosen if classifiers and segmentation networks make the same decisions. We got this idea from Heng. An image will only be chosen if the probabilities from classifiers are all over 0.95 or below 0.05 and it gets same result from segmentation part. According to this rule, 1135 images are chosen and added to trainset.

**Predictions on public LB:**
Defect 1: 97(128)
Defect 2: 2(43)
Defect 3: 611(741)
Defect 4: 110(120)
Sum Pos: 820(1032)
