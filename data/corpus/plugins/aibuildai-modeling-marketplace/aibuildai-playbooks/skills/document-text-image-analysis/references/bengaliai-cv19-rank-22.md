# 22 place Solution

Competition: bengaliai-cv19
Rank: #22
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/136870

I almost gave up hope, cos no matter what i tried the accuracy was not improving on public LB. Regardless the learning here is deeply trust in your local CV and so much more to learn since the accuracy gap between (1, 2, 3, 4th place) and the rest is significant.

Solution Outline -
- Start with transfer learning, Densenet201, One cycle learning
- Unfreeze layers and re-train (Included custom learning rate for individual layers)
- 3 separate model approach
- Albumentation Image Augmentation
- Local CV

Other details -
- Training epochs - 5 fold CV with callback each fold training with min-20, max-32 epochs
- Pre-processing - Converted the image to size 256*256 and applied 
- Image Aug - basic image augmentation like VerticalFlip=False, img_size of 256*256, - - - BrightnessContrast, RandomGamma, fastai's jitter
- Error_metric - Default fastai error_metric for each class
- No post processing except used the average of the 5 fold probabilities
