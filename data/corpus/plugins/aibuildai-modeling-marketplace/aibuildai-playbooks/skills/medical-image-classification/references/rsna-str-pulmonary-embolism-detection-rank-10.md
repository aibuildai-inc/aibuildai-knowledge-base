# 10th Place Solution with code

Competition: rsna-str-pulmonary-embolism-detection
Rank: #10
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193505

**code** - https://github.com/OrKatz7/RSNA-Pulmonary-Embolism-Detection
**Full Pipeline**


**Overall Strategy**
1. Train an image-level 2d CNN and save to hard drive features
2. Train an exam-level 3d CNN and save to hard drive features
3. input the 2d and 3d features into sequence model

**2D CNN Modeling**
1. Data pre-processing - based on Ian Pan: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/182930
2. augmentation - RandomBrightnessContrast, HorizontalFlip, ElasticTransform, GridDistortion, VerticalFlip, ShiftScaleRotate, RandomCrop
3. cnn models - efficientnet-b3, efficientnet-b4, efficientnet-b5

**3D CNN Modeling**
based on @boliu0 pipline https://www.kaggle.com/boliu0/monai-3d-cnn-training

**Sequence Model**
Input - Slice embeddings from multi 2d models + exam embeddings from 3d models
loss - rsna metric
