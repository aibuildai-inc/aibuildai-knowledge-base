# 8th place solution [0.49 on private LB]

Competition: google-ai-open-images-object-detection-track
Rank: #8
Source: https://www.kaggle.com/c/google-ai-open-images-object-detection-track/discussion/65120

# Using out-the-box solutions (pre-trained models) 
To make initial submit we used pre-trained *faster_rcnn_inception_resnet_v2_atrous_oid* model from [TF Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.mdhttps://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) . We achieved 0.216 on public LB. As this model was frozen with threshold 0.3 and we decided to refroze this model without any thresholds. After this model gave us 0.36 on public LB and we use it as base for our final submissions.

# Data and Metric Analysis 
* **Unbalanced data.** First thing we noticed that data is very unbalanced. For example Top-50 classes (by count of b-boxes) is 87.9% of training b-boxes or 84.3% of training images. On the other hand we have less than 1% of b-boxes and images for Bottom-50. It means that we have 200K samples per class for Top-50 and only 200 samples per class for Bottom-50. Due to that fact that evaluation metric is mAP all classes have the same impact on the final score.
* **Hierarchy structure.** All classes in OID organized into one huge classes tree. Therefore there are a lot of classes that share common features and this is must help a lot for model, but such hierarchical structure also rise new problems: model confused by almost same classes (what class will have Girl age of 19?  Girl or Woman?) and meta-classes start to contain some trash (can not define is this Woman or Man? Let’s mark as Person) 
* **Bounding Boxes.** We analyzed b-boxes by its area size and aspect ratios but didn’t find strong correlation. By default aspect ratios in RPN Anchor Generator are 1, 2, 0.5. We added ⅓, 3, ¼, 4 ratios but it did not give notable improvement.
* **Missing annotations.** We visualized classes b-boxes and looked threw some training images. We noticed that there are many images which contains partially annotated instances for large number of classes. Missing annotations would lead to false negatives during training. [Image]
* **Validation dataset.** We used validation set proposed by organizators. During submissions we noticed that results on Public LB are 0.13 lower than on validation. This gap left almost unchangeable from submission to submission.

# Detection Architecture
As our experiments based on framework from tensorflow/models we choose Faster R-CNN + backbone. Also we tried Single Shot Multibox Detector approach but it gave much worse results for OID (even for small amount of classes).
Training single model on the whole dataset
At first stage we tried to train on all 500 classes and to figure out what accuracy we can achieve with single model. We experimented with different backbones for Faster R-CNN:

* **Inception v2.** One of the fastest models for training/inference (except MobileNet) in TF Model Zoo. We knew that we won’t achieve comparable results to Inception ResNet v2 but we could check different solutions relatively fast. Best mAP: 0.298 on public LB
* **Inception ResNet v2.** As it had pre-trained weights on OID we fine-tune it but stuck on mAP ~0.38 on public LB
* **NasNet Large.** Consume large amount of resources during training/inference (2-3x in comparison to Inception ResNet v2). Best mAP ~ 0.42 on public LB

# Dataset Split
Taking into account that data is very unbalanced we split it into 6 subsets according to its data distribution. Then we trained Faster R-CNN + ResNet-101 on each subset and gained improvement about 0.12 in comparison to single model.

* **Bottom 0-100:** 263 imgs/class
* **Bottom 100-200:** 691 imgs/class
* **Bottom 200-300:** 1552 imgs/class
* **Bottom 300-400:** 4206 imgs/class
* **Bottom 400-450:** 14375 imgs/class
* **Bottom 450-500:** 202171 imgs/class

# Missed annotations:
Open Image Dataset contain a lot of unlabeled object. This is produce noisy and some time even wrong training signal for our models. To leverage this problem we tried following ideas:

* **Pseudo-labeling.** This is common approach that used on competitions to get some additional data and to fix bad annotation. We applied this trick to only to bottom 0-100 classes subset with prediction threshold &gt; 0.5. This trick increase mAP for bottom 0-100 classes subset by 1%. 
* **Overlap Soft Sampling (OSS, [paper](https://arxiv.org/abs/1806.06986)).** Another idea is to reweight training samples depending on highest IOU with any ground of truth. Core idea here is when sample contain significant part of any annotated object then lower probability of wrong training signal. This approach in average increase performance on 0.3%, but if look closer, than we can sees that some classes decreased in performance up to 50% AP and some classes increased in performance up to 50% AP, so this is was really good news, in terms of ensembling.  

# Ensembling
* **Best only.** In this approach for each class we simply select predictions from model that predicts this class the best, based on validation.  We used it at starting point. 
* **Performance-based with restore procedure.** This approach share same idea as “Best only”, but instead selecting just one model to predict some class, we select several models that perform best for this class and add they predictions. Before adding some predictions to final model we rescore all confidences of this model for this class by following formula: [Image]
Here performance is class average precision and N - number of models that predict this class. We achieved ~1% improvement in comparison to base strategy (Best only) 

# Result Pipeline
1. Split training set into 6 subset based on training sample (see dataset split topic).
2. Training models. Generally we train Faster R-CNN with ResNet-101 backbone for each subset.
3. For each class we selected models (based on evaluation results) that will be used to predict this class. Note that we used pretrained Faster R-CNN with 
Inception-ResNet v2 backbone as base model.
4. Then we concatenate all selected prediction from each model, expand all classes with meta-classes predictions and apply Soft-NMS to suppress multiple predictions.

# Hints and tricks:
* We increase output size for RPN and Result predictor from 300 to 500. Improved mAP ~0.1%.
* We expanded predicted b-boxes from leaves on superclasses. Improved mAP ~0.5%
* Using [Soft-NMS](https://arxiv.org/abs/1704.04503) above default NMS improved mAP ~2.5% 

# Other approaches that did not improve performance:
* Classify image classes and suppress false positives from detectors
* Split prediction head into classification and regression
* RMSProp/Adam instead SGD with momentum
* Dropout
* Using ImageLevel annotations during training
* Augmentations: Random Distort Color, Random Adjust Brightness,  Random Black Patches, Random Gaussian Noise
