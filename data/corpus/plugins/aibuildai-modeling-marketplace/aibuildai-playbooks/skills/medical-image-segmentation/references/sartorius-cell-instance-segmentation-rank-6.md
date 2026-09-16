# 6th place solution. Higher resolution is all you need.

Competition: sartorius-cell-instance-segmentation
Rank: #6
Source: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/297986

# Preprocessing

I only used `train` images, without semi-supervised or LiveCell.

All train images were sliced into smaller size, with window size (208, 281), and stride (104, 140). So a (520, 704) image can be slices to 16 smaller images.



# Training

## Model

It seems that model does not really matter in private LB. My res2net-101, resnext-101, detectoRS-r50 can get 0.350 in private LB, even though their public LB are very different.


Ensemble cannot improve both public and private LB significantly. For me, single model is enough.


## Hyperparams

I trained with mmdet, default 1x schedule and 1x swa training. Image scales were set as (1333, 1333)-(800,800).



# Inferencing

Test images were also sliced as training images. The test image scales were [(1333,1333), (1024,1024), (800,800)].

Masks were iterated from higher score to lower score. Scores lower than class-wise threshold and areas lower than class-wise pixel-threshold were removed. 



When dealing with overlaps, I remove mask whose overlapped part was more than 20% of itself.



RCNN's NMS was replaced by Weighted cluster-NMS with DIoU.

# Submission

My final submission was ensemble of HTC-Res2Net101 (trained with all data), HTC-ResNeXt101 (trained with all data) and HTC-Res2Net101 (trained with fold0)

# Weakness

LONG inferencing time, 2h per model.

# Github Repo

[https://github.com/CarnoZhao/mmdetection/tree/sartorius_solution](https://github.com/CarnoZhao/mmdetection/tree/sartorius_solution)
