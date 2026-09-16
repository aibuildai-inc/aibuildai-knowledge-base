# 10th place solution

Competition: severstal-steel-defect-detection
Rank: #10
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114465

**Overall approach**
1) Separate models for different defects (since the number of classes is small - it is feasible)
  * + can use completely different architectures/pipelines/samplers etc
  * + can combine trained models for different defects instead of retraining a single model
  * + encourages to decompose the problem (4-defect detection) into smaller subproblems (4 single-defect detections) – it allows to focus on perfecting them separately, without the fear of spoiling results of a different subproblem
  * + easier to reuse code for binary segmentation
  * + no need to balance defect class losses
  * - wastes valuable submission time → smaller size/number of models can be ensembled
  * - more training time
  * - no potential for synergy between detectors of different defect types

Doing this was a “greedy solution” - I already had experience with binary segmentation, and I’m a bit sad that I missed out on a chance to learn more new stuff within the scope of this competition. On the other hand, it maximized my chances of winning, so if I had to make a decision again – I would do the same thing. 
Besides, I can still try to match/beat the benchmark of my current approach with a single model for multiple defects after the end of the competition, even though there would be less motivation for doing that :)

2) Not predicting defect2
  * + less models → less training / research / inference time
  * + trying to predict defect2 was always decreasing the pubLB for me, so not predicting it → higher score
  * - if private test set contains more instances of images with defect2 – score would degrade

Initially I was going to predict only defect3 and defect4, but at some point I hit a wall in trying to improve my score, and tried predicting defect1 and defect2. Surprisingly for me, defect1 allowed a noticeable gain in score, even though it was less than a half from theoretically possible one, so I decided to keep it. Defect2, on the other hand, was even more rare, and I have been unable to get any gain from predicting it. Anyway, to protect myself against a scenario where it is more frequent in private test set, I have included a model for defect2, tuned to produce predictions only when it is very confident, and reserved a second selected submit for it. After the competition deadline it turned out that private test set didn’t seem to contain more of defect2, but I think that it was still a reasonable use of a second submit selection.

3) Training on full size images
  * + model has more context → easier to make a correct decision (avoid a false positive/negative)
  * + no need for an extra fine-tuning stage on full images / less domain mismatch if no fine-tuning is done
  * - more time and GPU RAM is required → less experiments, smaller batch size
  * - less “unique” samples seen by model → more potential overfitting

I’ve tried training on crops as well as on full sized images, and in all my experiments training on crops was inferior. I’ve seen that some solutions mention that when they were cropping images with defects they constrained the crops to contain defect – it might be the key to proper training on crops, but I haven’t tried it.

4) 3-fold validation
  * + larger size of validation set → more stable metric estimates
  * +“less similar” models → ensembling is more beneficial
  * +- less models produced – this is ok, since I am anyway constrained in how many models I can fit into prediction time quota
  * - smaller size of training set → potentially weaker individual models  
  * - longer validation time

This might have been important for avoiding negative consequences of shake-up in this competition. Keeping in mind that my validation set is 2x larger than public test set, I was highly reluctant to base any decisions about my solution on the impact on public leaderboard score, and was “trusting“ only changes that improve both CV and pubLB. 
At some point I started to worry that I’m overfitting to pubLB anyway, since adding more models to my ensemble decreased pubLB, indicating that I got lucky with my submit, and small variations of predictions caused a drop in pubLB. I decided to add RandAugment-type (more details in Augmentations section) to the validation set, hoping that this way I would select models/hyperparams that are more robust under different conditions, besides it would allow artificially increasing the size of validation set.
  
  
**Encoder**
SE ResNeXt-50 / EfficientNet-b3, pretrained on ImageNet. First model still seems like a good tradeoff between size/speed and accuracy, and efficientnet is relatively new and I wanted to try using it for segmentation tasks. Ironically, its implementation in PyTorch is not very efficient (at least, compared to the one in TensorFlow, see [this github issue](https://github.com/lukemelas/EfficientNet-PyTorch/issues/19)). This limited usage of “upscaled“ versions of it, and I settled on b3, which was close to seresnext50, although the batch size that I was able to fit into GPU RAM was smaller (3 vs 4) and it potentially decreased batchnorm performance.
  
  
**Model**
Experimented with Unet, FPN and PSPNet from great [Segmentation models](https://github.com/qubvel/segmentation_models.pytorch) by @pavel92 . Unet and PSPNet performed noticeably worse with default settings and were discarded from the experiments at the early stages of competitions.

I compared FPN with several variations that I considered to be reasonable, listed below.
FPNA: vanilla FPN.
FPNB: take not the 4 uppermost layers of encoder, but 4 lowermost. Intuition was that if a defect is a local artifact, then using stronger semantic information is unnecessary, and using higher-resolution layers instead would allow to better locate small defects. In my experiments it outperformed FPNA at 128x800 resolution, but underperformed at full resolution – perhaps receptive field was insufficient.
FPNC: take all (5) levels of encoder, this might help to handle different scale levels properly. Unfortunately the increase in GPU RAM usage and training time didn’t translate to improvement in performance in my tests.
FPND: Instead of summing outputs of encoder layers, concatenate them (reducing the number of channels in decoder layers’ output to keep final number of channels the same) - this is similar to hypercolumns, which worked well in TGS competition. This version seemed to be comparable with FPNA, and outperformed it in defect4 detection, perhaps due to the fact that this defect had significantly larger scale variation compared to other 3.

I experimented a bit with deep supervision (with a aux output for each defect class presence), and even though I initially have gotten some improvement using this scheme, I have later managed to match this result without DS, and decided to drop it to avoid additional complexity.

I have also tried to train classifiers for rejecting empty images before classification to avoid false positives, but haven’t been able to get more accuracy than I already had with segmentation models, in the end the only classifier that I used was for defect1 (taken from Heng CherKeng’s starter guide) – it seemed to have a better precision than my defect1 models.
  
  
**Augmentations**
Did a “grid search” for optimal augmentations, training same small model with a single augmentation for a fixed number of epochs and validating without augmentations.
Surprisingly, “base augmentation” (no augmentations) was always better, even compared to horizontal flip. This was not something I have encountered before, and main guess was lack of model capacity to handle the variation after introducing augmentations. Proper way of testing it was running the experiment again with a bigger model, but I didn’t have time/resources to do it properly, so I decided to make an assumption that horizontal flip doesn’t destroy important information and shouldn’t lead to overgeneralization, and other “generic” augmentations should be treated carefully.

I also tried few “custom“ augmentations: “SteelShift” (cyclically shift image, handling steel sheet border separately) to simulate a capturing steel sheet at a different time on a moving conveyor belt, and “BorderLoc” (encode distance to steel sheet border in a separate channel, black areas and images with no black areas are handled separately). Motivation for adding 2 custom augmentation, targeted at images with steel sheet borders was the fact, that even though there was correlation between CV and pubLB, CV was always better, and the only significant difference between 2 sets was the fraction on images where steel sheet border was visible.

Impact of these 2 “custom” augmentations was somewhat hard to asses, I included some models with “SteelShift” into final ensemble to increase diversity of predictions, and didn’t include “BorderLoc”, because it would require either applying it to all images (and discard models trained without it), or rewriting inference code to apply different augmentations to different models in ensemble)

I ended up with follwing augmentation “modes”:

a) only horizontal flip
b) [RandAugment](https://arxiv.org/abs/1909.13719) - up to 2 augmentations chose randomly from the list below ([albumentation](https://github.com/albu/albumentations) names):
* HorizontalFlip
* VerticalFlip
* ISONoise
* IAAAdditiveGaussianNoise
* CoarseDropout
* RandomBrightness
* RandomGamma
* IAASharpen
* Blur
* MotionBlur
* RandomContrast
  
  
**Loss**
0.75 * BCE + 0.25 * LovaszHinge
Lovasz worked significantly better than Dice. Perhaps a good idea was a separate fine-tuning stage, where only Lovasz was used, but I never got around to testing that.
Also wanted to try replacing BCE with Focal, but didn’t have time and also was worried about proper coefficient balancing – focal is usually significantly smaller than CE, and magintude might change during training.
  
  
**Post-processing**
- binarize averaged predictions using threshold, determined on validation set
- zero masks with number of predicted pixels less than a threshold, determined on a validation set

I have also thought about filling holes and removing small components in predictions, but after looking at predictions on the public test set decided that it will not change the score much, but might backfire in unexpected way on unseen data.
  
  
**Pseudo-labeling**
After many teams started to score more than 0.920 on pubLB it became evident that at least some of them are using semi-supervised learning. It didn’t seem to be extremely beneficial, because size of public test set in this competition wasn’t very big compared to train set size, but it should at least help to reduce to domain mismatch between train and public test sets, so I decided to give it a go, after plateauing in score on pubLB.
First I’ve tried to include all predicted test negative samples (this was easier, since I didn’t need the masks) into the train set (keeping folds unchanged and not including pseudo-labeled images into validation set, to be able to track improvement) and fine tune the models I had using this data. It didn’t help much, so I moved on to including all predicted test set images the same way into training set and training from scratch. This allowed to get a single model score equal to the ensemble score on pubLB, but ensembling such models didn’t provide any improvement – it made sense, since most likely the models were just “remembering” masks for all the samples from the public test set, and there was no variation between predictions on the public test set images.
Finally, I decided to do it the “proper” way, and included only confident samples into the training set. I measured confidence similar to [approach of winners of TGS competition](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69291#latest-592781) - count the number of pixels with confidence &lt; 0.2 or &gt; 0.8, and consider images that have more than N confident pixels to be reliable. I’ve noticed, that no matter how certain/good the predictions are, there is always an uncertain region along the predicted defect boundary (which makes sense, since the ground truth annotation was fairly arbitrary in terms of boundary). To avoid this uncertain area affecting the “prediction confidence” estimate, I’ve applied morphological operations to “grow” confident regions – this allowed to “close” uncertain areas along the boundaries, and still kept the “really” uncertain defect predictions. This was done differently for each defect type, since they had different areas and boundary uncertainties.
This approach helped me to move past 0.917 on pubLB to 0.919, with most of this change attributed to improved predictions for defect3. I have also applied pseudo-labeling to defect1 and defect4, but been unable to get improvement for defect1 (makes sense, since my predictions for it were far from being perfect), and unclear results for defect4 (not really sure why, maybe due to more skewed defect/nondefect distribution of data).

**Closing thoughts**
Thanks to organizers of this competition and all participants - it was a really fun and intense experience! This is a second competition where I have been using [DVC](https://dvc.org/) and [albumentation](https://github.com/albu/albumentations) libraries, and first for using [Catalyst](https://github.com/catalyst-team/catalyst), [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch) and [Weights &amp; Biases](https://app.wandb.ai/) - really enjoying these, if you haven't tried them yet - I strongly encourage you to.
