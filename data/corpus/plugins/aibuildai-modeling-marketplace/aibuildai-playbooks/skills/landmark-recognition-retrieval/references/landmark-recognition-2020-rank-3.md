# 3rd Place Solution — A Pure Global Feature Approach

Competition: landmark-recognition-2020
Rank: #3
Source: https://www.kaggle.com/c/landmark-recognition-2020/discussion/187757

Thanks to the organizers and congrats to all the winners! Our ( @haqishen @boliu0 @garybios @alexanderliao) solution is a pure global feature metric learning approach with some tricks.

## architecture: sub-center ArcFace with dynamic margins
ArcFace ([paper](https://arxiv.org/abs/1801.07698)) has become a standard metric learning method on Kaggle over the past two years or so. In this competition, we used Sub-center ArcFace ([paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)), a recent improvement over ArcFace by the same authors. The idea is that each class may have more than one class center. For example, a certain landmark's photos may have a few clusters (e.g. from different angles). Sub-center ArcFace's weights store multiple class centers' representations, which can increase classification accuracy and improve global feature's quality.

The classes in GLD dataset are extremely imbalanced with longs tails. For models to converge better in the presence of heavy imbalance, smaller classes need to have bigger margins as they are harder to learn. Instead of manually setting different margin levels based on class size, we introduce *dynamic margins*, a family of continuous functions mapping class size to margin level. This give us some major boost. Details to be described in paper.

We train the model using ArcFace loss only.

## validation scheme
Stratified 5-fold. Training using 4 folds, validate on only 1/15 of 1 fold to save time. Use different folds for different single models, for maximal diversity in ensemble.

We use GAP metric to validate. CV GAP is very high compared to LB. All model's CV GAP are over 0.967. Good news is that CV GAP and LB have high correlation.

## test set predicting strategy
- If we use model's ArcFace head to predict test set, our best single fold model's public LB is only **0.564**
- A better strategy is to calculate the global feature cosine similarity of each [private train image, private test image] pair, and use the top1 neighbor and corresponding cosine similarity of each test image as prediction. This gives us **0.604** public LB for the same single fold model
- Instead of using top1 neighbor only, we can improve it by combining top5 neighbors and their similarities. Best combining function is 8th power. 
E.g. let's say test image A's top5 neighbors and their cosine similarities are: class 1 (0.9), class 2 (0.8), class 2 (0.7), class 1 (0.5), class 3 (0.45). Then class 1's total score is `0.9**8 + 0.5**8 = 0.434`; class 2's total score is `0.8**8 + 0.7**8 = 0.225`. So we predict class 1 with p=0.434
LB increases to **0.610**
- We can further improve it by incorporating ArcFace head's predictions with 12th power. In above example, assuming image A's ArcFace head give class 1 score = 0.75, class 2 score = 0.88. Then class 1's total score becomes `(0.9**8 + 0.5**8) * 0.75**12 = 0.0138`; class 2's total score becomes `(0.8**8 + 0.7**8) * 0.88**12 = 0.0486`. Now we predict class 2 with p=0.0486
LB increases to **0.618**


## augmentations
We resize all images to square shape without cropping.
```
import albumentations as A
A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_lower=99, quality_upper=100),    
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        A.Resize(image_size, image_size),
        A.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
        A.Normalize()
    ])
```

## pretraining and finetuning
- In cGLDv2 (cleaned GLDv2), there are 1.6 million training images and 81k classes. All landmark test images belong to these classes.
- In GLDv2, there are 4.1m training images and 200k classes, among which 3.2m images belong to the 81k classes in cGLDv2.

We noticed that (1) training with the 3.2m data gives better results than only the 1.6m competition cGLDv2 data, (2) pretraining on all 4.1m data then finetuning on 3.2m data gives even better results.

## 3-stage training schedule
- Stage 1 (pretrain): 10 epochs with small image size (256) on 4.1m data
- Stage 2 (finetune): about 16 epochs with medium image size (512 to 768 depending on model) on 3.2m data. Number of epochs varies by model and ranges from 13 to 21. 
- Stage 3 (finetune): 1 epoch with large image size (672 to 1024) on 3.2m data

Note: above schedules are for Sub1 (private 0.6289). Sub2 has higher score (private 0.6344) but more complex schedules, i.e. longer with more rounds of finetuning. Both submissions are 3rd place.

## ensemble
7 models: EfficientNet B7, B6, B5, B4, B3, [ResNeSt101](https://github.com/zhanghang1989/ResNeSt), [ReXNet2.0](https://github.com/clovaai/rexnet)

For global feature neighbor search, we concatenate each model's 512-dimension feature; for ArcFace head, we take simple average of each model's logits.

Best single model is EfficientB6. **private = 0.6005, public = 0.6179** (We didn't submit all the epochs, there may be higher ones)
Sub1 is 7 model ensemble. **private = 0.6289, public = 0.6604**
Sub2 is 9 model ensemble (B5 and B6 twice). **private = 0.6344, public = 0.6581**


### [update 10/12/2020]
paper: https://arxiv.org/abs/2010.05350
repo: https://github.com/haqishen/Google-Landmark-Recognition-2020-3rd-Place-Solution
