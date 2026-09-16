# 10th place solution

Competition: happy-whale-and-dolphin
Rank: #10
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/319941

Congratulations to all the winners. And thank you to kaggle host and all participants for this exciting competition. 
This is My First Solo Gold Medal. 
I'm very impressed to win the gold medal among so many kaggle GMs, masters.  

Here is my 10th place solution.

# Dataset  
I used 2-type datasets and especially thank @jpbremer, because of his sharing valuable approaches and datasets.  
 - fullbody dataset  
   - public dataset by @jpbremer
   - private dataset  
     - trained yolov5 and WBF with bboxes by Detic  
 - backfin dataset  
      - public dataset by @jpbremer　　

# Model
I trained 2-type(fullbody/backfin) models with image sizes 512 or 784.
fullbody/backfin model: trained with fullbody/backfin dataset

When we search nearest neighbors,  
 - for the specices with backfin -> use concatenated embeddings by fullbody models and backfin models   
 - for the specices without backfin (like Beluga, ...) -> use embeddings by fullbody models

- backbone  
  - efficientnetv2_m
  - efficientnetv2_l
  - convnext_base
  - convnext_large

In my experiments, EfficientnetV2 > EfficientnetV1 ≧ ConvNext, but ensembling them boosted my CV/LB scores. 
And I concatenated outputs of conv-layers before pooling layer, and then forward this to the neck of the model. 
This also works well. 

# Augmentation
The main augmentation is below.  
 - HorizontalFlip
 - ImageCompression
 - ShiftScaleRotate
 - RandomBrightnessContrast
 - HueSaturationValue
 - MotionBlur  
### Mixup  
 - mixup the embeddings (not images) and Arcface with soft label worked (CV:+0.003-0.005)

# Loss 
In my case, some aux-loss helped.
Loss = ArcfaceLoss + FocalLoss + SpeciesLoss
 - SpeciesLoss: classification of 26-species

### progressive dynamic margins  
At some early epochs, training didn't go well if the margins are somewhat large. 
Inspired by https://arxiv.org/pdf/2010.05350.pdf, I re-designed the function to increase margins gradually. 
- 1~5 epoch: increase coefficient of margins linearly from 0.2 to 1
- 6~20 epoch: coefficient of margins = 1 (That is, this function is equal to original-dynamic margins)  

# Pseudo Label
This is also the key to raise the scores.  
At first, I trained models on pseudo-label, 
and then trained on original training dataset using this pretrained weights.

# Ensemble
I concatenate weighted embeddings, this operation works a little (CV:+0.001-0.002)

# Thresholding 
To predict `new_individual_id`, search the best threshold by the predicted top1-species on validation datasets.  


Thank you for your reading.
Any questions are welcome !
