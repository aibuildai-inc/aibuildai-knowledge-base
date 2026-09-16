# [8th place] Solution Overview + Code

Competition: siim-acr-pneumothorax-segmentation
Rank: #8
Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107522

Congratulations to all of the winners and participants. 

@felipekitamura and I are happy to share our solutions. 

Full documentation: https://docs.google.com/document/d/108xK_J4WVTtuuwMfxBINR7FLXYRNxldnAQV5cLfO-1g/edit?usp=sharing

Code:
https://github.com/i-pan/kaggle-siim-ptx

Data split: 10% holdout for ensemble, 10-fold CV on remaining 90%, stratified by pneumothorax size

Architecture: DeepLabV3+

Backbone: ResNet50/101 and ResNeXt50/101 with group normalization

Loss: weighted BCE (trained on all images) or soft Dice (trained on positives only)

Optimizer: Vanilla SGD, momentum 0.9

Training: batch size 4, 1024 x 1024 and batch size 1, 1280 x 1280 for pure segmentation (did not retrain on stage 2)

Schedule: cosine annealing, 100 epochs, 5 snapshots, initial LR 0.01 to 0.0001

Ensemble: 
    - 12 models total (x3 snapshots/model)
    - 4 trained on positives only with soft Dice
    - 8 trained on all images with weighted BCE
    - I used 4 of the models trained on all images as "classifiers"
        - Max pixel value was taken as classification score, averaged across 4 models
        - Multiplied pixel-level scores from 4 models trained on positives only by this classification score, then averaged
    - Final ensemble: multiplied score as above averaged with pixel-level scores based on other 4/8 models trained on all images
    - Hflip TTA

Post-processing:
    - Remove images with total mask size &lt;2048, 4096 pixels 

What did not work/insights:
- Ensembling model with Felipe: he got to 0.8750 with Unet, EfficientNetB4 kernel using 512 x 512 trained on soft Dice loss, all images. We weren't able to properly ensemble our models and ended up dropping it in favor of my slightly better ensemble. 
- I personally tried lower resolutions and Unet, LinkNet, PSPNet, EncNet, HRNet, etc. All did worse than DeepLab, 1024 x 1024
- Whole-image classifiers: did not work as well as segmentation
- SGD worked better than Adam, Adabound optimizers
- Hflip TTA + small mask removal helped 
- Weighted BCE loss was most stable for me, Lovasz, soft Dice were unstable or did not converge at all

Local: 0.8737
Stage 1 LB: 0.8780 (remove 4096 pixels)
Stage 2 Private: 0.8627 (remove 2048 pixels)
