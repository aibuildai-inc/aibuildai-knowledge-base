# Private 22nd Place Solution

Competition: hpa-single-cell-image-classification
Rank: #22
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238387

Congratulations to all the winners, and thanks so much for hosting such an interesting competition!!
This task was really challenging in mostly two points: weak-labels and class imbalance.
I spent hard time on solving them, and learned a lot in the middle of it.



# Summary
・I tackled this competition as a classification task (I didn't use any segmentation models other than HPA Cell Segmentator).
・Environment: Kaggle Notebook and Datasets, TPU training, GPU inference, PyTorch
・Cell tiles: 'nucleus BBox center' chosen as tile center, 'cell BBox short side' chosen as tile one side length→score improved!!
・2-Stage Training Pipeline (For 2nd stage, pseudo-labels, thresholding and sampling methods were used.)
・Green Image Level Label prediction further added, shared by @h053473666 



# Training
My pipeline is the following.
CV: multilabel stratified group kfold (group by image id)
augmentation: flip, random rotate, shift scale rotate
loss: BCEWithLogitsLoss
optimizer: Adam
scheduler: cosine annealing
number of cell tiles used as input: about 70000 (1st stage), about 75000 (2nd stage)
epochs: 5eps w/o early stopping
training time: 1~2 hrs per model

[Screen Shot 2021-05-12 at 12 54 37]



# Inference
I used @samusram fast segmentator with a little modified.
Classification predictions by model above was combined with segmentator instance segmentation result.
https://www.kaggle.com/drtausamaru/hpa-ct-ill-inference-private
[Screen Shot 2021-05-12 at 12 48 40]



# What didn't work for me (score dropped)
・Cell tiles other than my approach (whole cell tiles, cell BBox long side length, conversion to all values = 0 of the area outside the targeted cell, only green signal used...etc)
・focal loss
・BCEWithLogitsLoss with pos-weight argument > 1.0
・label smoothing
・pseudo hard labels (0/1)
・For 1st-stage, using >=4 cell tiles at once
・MLSKF (not grouped)
・Models: ResNet200D, SEResNeXt101
・cellline classification model



# What I didn't try
・RGBY
・cell tiles: >256x256, uint16 
・other segmentation models (object detection)
・other augmentations (mixup, brightness modification, cutout...etc)



# In the end
I really enjoyed this competition because the task itself is interesting and challenging, there were a few public kernels of just ensembling or forking, and I was able to compete with top Kagglers.
Thanks for reading :)
