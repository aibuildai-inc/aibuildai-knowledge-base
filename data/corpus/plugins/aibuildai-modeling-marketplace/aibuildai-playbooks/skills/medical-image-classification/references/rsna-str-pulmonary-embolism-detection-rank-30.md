# 30th Place Solution

Competition: rsna-str-pulmonary-embolism-detection
Rank: #30
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193423

Thanks to the Kaggle team and organizers for this challenging competition. Huge props to my teammates @wuliaokaola @bacterio, they worked very hard!! We have learned a lot and now we are exhausted both computationally and physically 😃, just like many of you. This solution is developed within 15 days and it's far away from the top winning solutions. But hopefully there's something helpful in this!

## TL; DR
* **Efficientnet B1** for image level, **Efficientnet 3D CNN** + **Resnet 3D** for study level. **CNN + Transformer** for image and study level.

* Used TFRecords by the organizers and windows function by @vaillant.

* Applied mask to hide left/right side of the image

* Optimized blending weights based on OOF calculation.

## Timeline

* We started the competition about 15 days ago.   

* Trained **Efficientnet B0** on image level and blended with the mean predictions by @osciiart, aka the public notebook scored 0.325. 0.4 * mean prediction +  0.6 * Efficient B0 along with the mean prediction for study levels. This gives us LB: `0.292`. 

* Developed **Resnet3D** that scored `0.370` LB. We ran a local validation for this model and found out that this model did well on the study level but poorly on the image level. Hence, for study level, we replaced then mean prediction with our Resnet3D and achieved `0.253` LB.

* Improved **Efficientnet B0** with masks(masking one half of the image). And achieved `0.248` LB. 

* The amazing public baseline came out and we were frustrated because it scored `0.233` and it made us panic. We tried to inference that and it took forever so we decided to not incorporate that into our pipeline.

* We added @bacterio to our team and blended with our existing results(take the mean) and got `0.226` LB. @bacterio had a **CNN + Transformer** at that time, which is completely disparate from our approach. It calmed us down a bit and we knew that if we continue improving both image level and study level, we will get better.

* We improved by training three **Efficientnet B1**s with similar techniques and @wuliaokaola came out with a new architecture **Efficientnet 3D CNN** and we got `0.212` LB. 

* Oh! We only had **two days left**. What should we do? We realized that it's quite unrealistic add in another architecture. So we should work on upgrading existing models. We added TTAs to **Efficientnet 3D CNN** and fine-tuned **Efficientnet B1**, **Resnet 3D**, **Efficientnet 3D CNN** and we got `0.204` LB.

## Modeling
In this section I will show our final models.
### Efficientnet B1(TF)
* Trained with TPU using TFRecords provided by the organizer
* 3 windows as 3 channels by @vaillant
* Configurations:
   * Batchsize: 768
   * Epochs: 17
   * Scheduler: 9 epochs **1e-3**, 2 epochs **1e-4** and 6 epochs **1e-5**.  Three phases, pick the best model at each phase and continue with a new learning rate next phase.
* Augmentations:
   * rotation, shear, hzoom, wzoom, hshift, wshift by @cdeotte. https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96
   * mask, basically a cutout, this has **significantly boosted our valid loss**. Implementation shown as below @wuliaokaola.


* Results:
   * Around `0.22` image level weighted loss

### ResNet 3D(TF)
* Randomly take 256 images from each study distributed by position; if the study does not contain 256 images, tile
* Input shape: [256, 256, 256, 3], output shape: 256(image level) + 9(study level)
* ResNet 50 Backbone
* Change Conv2D to Conv3D
* Keep z dim as input and use it for image level label
* It can predict image level labels as well but the performance is not as we expected
* Results:


### EfficientNet 3D CNN(TF)
* Connect 256 **EfficientNet B0**s with a **TimeDistributed** layer 
* use the same identity_block and conv_block like ResNet on the top
* It can also predict image level labels as well but the performance is not as we expected
* Results:
.png?generation=1603765158757751&alt=media)

### CNN + Transformer(Fastai v2 + Pytorch)
* Resnet34 Backbone
* 6 Stacked Reformers head 
* Trained with 256x256 images with a **single window** but stacking 5 **consecutive slices as channels**
* Affine Augmentations
* Results: around `0.245` on image level and `0.270` on exam level with 5xTTAs

### Blending
* Blending **EfficientNet 3D CNN** and **ResNet3D** for study level based on OOF:
.png?generation=1603765267593556&alt=media)
Here the exam level loss is around `0.19` with weight `[0.71, 0.05, 0.07, 0.38, 0.28, 0.78, 0.71, 0.13, 0.45]` for each label.
* Blended with 
* Along with our existing image level models(**EfficientNet B1**), we can get around a 0.2 CV

## Hardwares
Yes, this competition is really heavy on hardwares. We also found the problem of IO bottleneck, i.e. GPU cannot run at full power, as shown below. Here are a list of hardwares we primarily use:
* Kaggle TPU, GCP TPU for Tensorflow development
* V100(32G) * 2 for Fastai and Pytorch development
* Rtx Titan(24g) for casual testing


## Additional Tips
* When **blending Tensowflow and Pytorch** models, it's often a good practice to write them as **scripts** as show below.

## Fin
Thanks for reading this solution. Happy kaggling!
