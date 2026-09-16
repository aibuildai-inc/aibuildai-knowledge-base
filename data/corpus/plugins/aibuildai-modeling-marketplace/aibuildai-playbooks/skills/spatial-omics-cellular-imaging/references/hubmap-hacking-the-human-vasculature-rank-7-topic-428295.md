# 7th Place Solution

Competition: hubmap-hacking-the-human-vasculature
Rank: #7
Source: https://www.kaggle.com/c/hubmap-hacking-the-human-vasculature/discussion/428295

Thanks to HuBMAP for hosting the exciting competition, and congrats to all prize and medal winners!

- Summary
  - Mask R-CNN model trained with dataset1, 2, 3 (pseudo labels)
- Training pipeline
  - Train models with dataset 1 (5 folds)
  - Create pseudo labels for dataset 2, 3 using the above models (for each fold)
  - Train models with dataset 1, 2, 3 (5 folds)
    - For dataset 2, both original (dilated) annotations and pseudo labels were used
- Model
  - Mask R-CNN (Swin Transformer backbone, HTC RoI head)
- Augmentation
  - Random resize (768-1536), flip, Rot90, RandomBrightnessContrast, HueSaturationValue
- TTA
  - Resize (1024, 1536), hvflip
- Ensemble
  - Ensemble on both region proposal and RoI head
  - See "ensemble detection model" part of this solution
    - https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/discussion/298146
- Post-processing
  - Dilation
  - Remove small masks
  - Remove masks that contain glomerulus regions
- Does not work for me
  - Train with test images' pseudo labels (train in submission)
  - External dataset https://data.mendeley.com/datasets/m2t49zf6xr/1
  - YOLOv8
  - Puzzle in submission
    - https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/discussion/417314
- Dilate or not dilate
  - I have found experimentally that when using only dataset1 for training, the score is higher without dilation than with dilation. Therefore, I suspected that the success of dilation was brought from noisy dataset2 and was an overfitting method to LB. Thus I have tried to minimize the difference in score with and without dilation by using pseudo labels and dilated annotation masks for dataset2.
  - In the first submission, the dilation score was 0.1 better than without dilation, but in the final submission, the difference was reduced to 0.02. However, the submission with dilation was still better for both public and private LBs.
