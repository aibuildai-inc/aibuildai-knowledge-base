# 12th place solution overview

Competition: severstal-steel-defect-detection
Rank: #12
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114309

Team: @delpro @pavel92 @ilyadobrynin 
We joined finally the last day of merging deadline and start woking on joint solution. At that time we have quite popular approaches with first step as classification (to remove most of non defective images) and second step as segmentation for defects localization. But at the end came up with 3-step solution. Our multilabel FPN and PSPNet segmentation models perform not so good in pixel-level segmentation, but works awesome to reduce number of false positive predictions after first classification step, so we decided use them as intermediate classification step. 

Following you will find more detailed explanation of each step.

### Stage 1. Multilabel classification
Implemented by @ilyadobrynin 
On the Stage 1 there was a multilabel classifier to detect images with at least one type of defect.
#### Model:
- Best 3 of 5 folds Senet154 trained on resized images (128x800) with simple BCE loss

#### Augmentations:
- Normalization
- Resize
- h-flip, v-flip
- no TTA

After training we have found thresholds for binarization with maximizing f1-score for the given class, so each class have its own threshold. It allowed us to exclude almost half of the images and speed up inference.

### Stage 2. Multilabel segmentation
Implemented by @delpro @ilyadobrynin 
On the Stage 2 there was some overfit magic. There we have mean ensemble of the multilabel segmentation nets:
- 4 folds of PSPNet on se_resnext101_32x4d (Awesome Qubvel implementation), trained on full images with BCE + Jaccard loss;
- Custom FPN on senet154 trained on crops (256x256) with simple BCE loss
- Custom PSPNet on senet154 trained on crops (256x256) with simple BCE loss

#### Training
- Pretrain on crops (256x256), batch size 32, BCE + Jaccard
- Fine-tune on full size with 0.1 * lr, batch size 4 with gradients accumulation up to 20 images

#### Augmentations:
- Normalization
- CropNonEmptyMaskIfExists: crop image with defect area if one exists, else make random crop (we have contributed this transformation to [Albumentations](https://github.com/albu/albumentations/pull/342) during competition)
- h-flip, v-flip
- only h-flip on TTA

#### Details:
There was a trick in the custom FPN and PSPNet training. First of all, we have trained a multilabel Senet154 classifier on the crops. After that we use this classifier as a backbone for the segmentators. It speed up training significantly with the same quality.
#### Post processing:
First of all, we removed small objects and holes from the image. Then we assume that there is no objects if the sum of positive pixels is less than a minimum threshold (unique for the each class). It boosted our score on the public leaderboard, but could cause overfit. On this stage we have no thresholds optimization for the ensemble of models, since the default thresholds give us better result.
Second stage allows us to remove many of the False Positive images.
### Stage 3. Binary segmentation
Implemented by @pavel92 
On the Stage 3 we use binary segmentation models, trained on non-empty masks for each class: 2 Unet(seresnext50), 2 FPN(seresnext50), 1 Unet(SeNet154)

#### Training:
- loss: SoftDice + BinaryFocal(gamma=2.)
- optimizer: RAdam
- sheduler: reduce on plateau
- sampling: non-empty
- crop: 256x768

#### Augmentations:
- Normalization
- h-flip, v-flip
- rotate180
- random brightness/contrast
- jpeg compression
- random scale (limit=0.1)

### Tips:
- Pseudolabels. For the classification step we use the most confident predicts from the previous submission. The confidence score for each image: `np.mean(np.abs(np.subtract(prob_cls, 0.5)))` where prob_cls - probability for each class on the image
- Use `torch.jit` to serialize your models (it helps a lot to transfer models to kaggle without pain)
- Trick that helps to improve models quality for about 0.3-0.5 % points for all models - best checkpoints weights average (**weights!, not predictions**). During traning 5 best checkpoints have been saved and then differnet combinations of them evaluated to find the best candidates to average

### What did not work:
- binary classification on the Stage 1
- multilabel segmentation on the Stage 3
- multiclass segmentation with aux classification output
- gated multiclass segmentation with aux classification output and `mask = mask_output * cls_output` (during traning and inference)
- we have also tried to fuse conv+batchnorm to reduce models inference time, but get only minor improvements, for `se` models it is just about ~5% speedup :(

### Usefull libs you have to know:
  - [PyTorch] https://github.com/qubvel/segmentation_models.pytorch
  - [PyTorch] https://github.com/qubvel/ttach
  - [Keras] https://github.com/qubvel/segmentation_models
  - [Keras] https://github.com/qubvel/efficientnet
  - [Keras] https://github.com/qubvel/tta_wrapper
  - [Everywhere] https://github.com/albu/albumentations
 
### Useful papers:
  - Gradients accumulation: https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
  - SWA: https://towardsdatascience.com/stochastic-weight-averaging-a-new-way-to-get-state-of-the-art-results-in-deep-learning-c639ccf36a
  - Pseudolabels implementation: https://arxiv.org/abs/1904.04445

Special thanks to my awesome teammates!
