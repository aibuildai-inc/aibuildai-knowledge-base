# 8th place solution

Competition: sartorius-cell-instance-segmentation
Rank: #8
Source: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/297998

My solution points
1. cascade mask rcnn with resnext152 backbone
2. pretrain on LIVECell dataset
3. pseudo labeling
4. WBF and WMF(weighted masks fusion)
5. train and inference on each image class

Pipeline is [this](https://drive.google.com/file/d/1Bl5AzzTsm9_tjwezlXUvyEprOwmZcWSw/view?usp=sharing)

Training strategy : 
I tried to train the model in all classes, but it did not work, so I trained and inferenced with each class.
I pretrained models on LIVECell dataset and finetuned with competition data. Then, I inferenced on train-semi-supervised data to generate pseudo labels.  And finally finetuned these models with competition data and pseudo labels.
The ways to pretrain and finetune are similar to  these codes. [pretrain](https://www.kaggle.com/markunys/sartorius-transfer-learning-train-with-livecell), [finetune](https://www.kaggle.com/markunys/sartorius-transfer-learning-train)

Inference strategy:
I ensemble the boxes predicted by the cascade mask rcnn and yolov5x with WBF, and use the boxes to generate masks. I use WMF, which is WBF applied to the mask ensemble, and ensemble the masks of folds.
Inference code and WMF code is here.[Inference code](https://www.kaggle.com/markunys/8th-place-solution-inference), [WMF code](https://www.kaggle.com/markunys/ensemble-boxes)
WMF code is identical to [this](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) except for including WMF.

I have spent hundreds of hours on this competition, and I am very happy with the results!!
