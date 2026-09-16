# 8th place solution + code

Competition: rsna-intracranial-hemorrhage-detection
Rank: #8
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/118780

First, I'd like to thank the whole team for the great collaboration: @meanshift, @tgilewicz, @nordberdt, @dmytropoplavskiy, it was a pleasure working with you on this competition. Following is a summary of our solution and experiments.

Code: https://github.com/ambrzeski/kaggle-rsna-2019

# Preprocessing
## Gantry tilt correction
In order to reduce the impact of radiation on patient’s eyes during head CT, sometimes scans are performed with a tilted gantry. Such studies have slices not aligned properly in the y axis, which causes the distortion in the 3D volume, visible on the sagittal view. To correct the distortion, slices must be shifted in the y axis with a shear transform. The angle of the shear can be determined from “Image Orientation (Patient)” DICOM field.


*Saggital view before and after gantry tilt correction.*

We observed that gantry tilt is quite common in this dataset, occurring in more than 50% of studies in the train set and 90% in stage1 test set. We haven’t measured the impact of gantry tilt correction on models’ performance. Supposedly it doesn’t matter too much when inputting slices as channels to 2D model, but when using wider 3D context it could make a difference, especially considering the fact that the angle of the shear varies among studies.

## Windowing
First, HU values were clipped to [-400, 1000] range. Then, instead of using fixed size windows, we apply a non-linear transform, which stretches out most valuable HU ranges and compresses less interesting ranges. The exact shape of the transform function is manually designed by us. We’ve tried modeling this function as a cumulative distribution function of histogram of pixel values in the dataset, but setting the values by hand allowed us to enhance specific ranges containing hemorrhages, and consequently worked slightly better.


*Mapping from HU values to normalized values*


*From left to right: image clipped to [-400, 1000] range, brain window (L:40, W:80), non-linear transform*

We’ve also tried fixed and learnable windows, but our non-linear transform gave the best results, while also being the most convenient to use.

# Models
All of our models are various variants of 2D CNNs with consequent slices being fed to the network as image channels, predicting classes for the middle slice only. Models were trained on 5 folds with no patient overlap between folds. As an optimizer, we used RAdam. For data augmentation we used standard spatial transformations. Any transforms messing with pixel values resulted in a decrease in performance.

For some of the models, slices (or groups of slices) were forwarded separately through the network backbone and their feature maps concatenated and combined by a convolutional layer. As such approach for 5 slices would increase 5x the training time, for the first 4 epochs, the model has been trained on the current slice only with weights of combining 3d convolution related to other slices set to 0. For two more epochs, the model has been trained on all 5 input slices. Combining model outputs improved results more (around 0.006-0.01) compared to multiple inputs to single model.




Summary of all the models in the ensemble is presented below.

*in stage2 we had a bug in validation, which may cause the results to be slightly inaccurate

As many have noticed, scores on the final leaderboard are significantly lower than stage 1 leaderboard. We also observed a large divergence between model ranking on local cross-validation and private leaderboard. Specifically, simpler models, like resnet18, seem to perform better on private leaderboard, compared to more complex ones.

# Segmentation masks
We hand-labeled 196 studies with segmentation masks. For the labeling we picked studies with the highest log loss on out-of-fold predictions. We’ve tried multi-task learning, using pre-computed mask as attention masks or additional features for a classifier, but all of these approaches failed to provide any significant boost to log loss metric. Anyway, we included one of the models trained with masks to a final ensemble, hoping to at least increase the variety within the ensemble. Segmentation based model performed slightly better on other metrics like F1.

The model has been trained to do both classification and segmentation (for labeled samples), with segmentation samples oversampled for initial few epoch and switched to the original samples distribution after.



# TTA, ensembling and second-level model
For each model we perform 5-fold ensembling and test-time augmentations. Then, final predictions are obtained using a L2 linear model, which takes as an input 1st level predictions for 5 consecutive slices from each model and returns final predictions for a middle slice. The L2 model gives a very slight improvement (0.0002 difference from simple averaging on LB).

# What didn’t work
- We’ve tried transfer learning with 3D CNNs from https://github.com/Tencent/MedicalNet, but trainings were taking too much time and we had to abandon this idea, but it still might be worth trying given more time
- Experiments with segmentation masks
- Class balancing
- Multiple trainable windows
