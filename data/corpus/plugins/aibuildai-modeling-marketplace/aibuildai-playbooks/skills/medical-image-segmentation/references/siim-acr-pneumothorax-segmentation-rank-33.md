# 33 place solo model solution

Competition: siim-acr-pneumothorax-segmentation
Rank: #33
Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107546

All with the end of the competition and congratulations to the winners.

We tried different architectures, starting with resnet18, resnet34, vgg16, seresnext50, dpn92 and densenet169.

The best results were shown by dpn92 and densenet169. As the final solution, we chose densenet169 by the top LB. dpn92 lagged just a little bit behind, but the model's weight was very large (approximately 500MB per fold) so that they did not risk to load them.

**We did not retrain the model in the second stage on the new train.**

**Details of the final decision:**

Trained on 5 folds of 50 epochs 3 times.

1. IMAGE_SIZE - 1024
1. Used a sampler: 60% with masks.
1. Loss: BCE + DICE + Focal.
1. OPTIMIZER: Adam, LR = 0.0001
1. SCHEDULER: CosineAnnealingLR
1. BATCH_SIZE: 4
1. GRADIENT CLIPPING: 1

Transforms, we used albumentations:
1. Horizontalflip
1. Randomcontrast
1. Randomgamma
1. Randombrightness
1. ElasticTransform
1. Griddistion
1. OpticalDistortion
1. ShiftScaleRotate
