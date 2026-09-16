# Post Competition Architecture Discussion

Competition: passenger-screening-algorithm-challenge
Rank: #8
Source: https://www.kaggle.com/c/passenger-screening-algorithm-challenge/discussion/45805#261169

My winning solution was an ensemble of 11 models. All models are end-to-end, i.e., they take as input 16 images per subject and spit out 17 probabilities per threats. I did not use any pretrained model or weights. All models are hand-crafted CNN based models. All models were trained on only one Titan X gpu.
  
**Data pre-processing**
- I only used APS format.
- For most models, we down-sampled the dataset to 256  by 330 
- We also rotate images 90, 180 and 270 degrees and train separate models on the rotated dataset.
- For couple of models we used the original 512 by 660 images and the rotated dataset.
- Global zero-mean and unit-variance normalization were used. 

**Data Augmentation**
We perform different image augmentations including: rotation (10-15 degrees), width and height shift (10-15%), shear, zoom (10-15 %) as well as random erasing [Ref1]. I tried to use brightness augmentation but the models were completely confused with any changes in the brightness so I dropped it from augmentation.

**Architecture**
Model architecture as seen in the figure is basically a conv2D to extract features from 16 image slices one at a time, flattening features, average over 16 image slices and then generating 17 Sigmoid outputs (one per zone). I did not know about the MVCNN paper but it seems there is similarity with that method. 

**Insights**
Random erasing augmentation provided some improvements. Building separate models on 90,180,270 rotated dataset was also very effective. It is also better to use the original dataset as opposed to the down-sampled data. Although, it is more time consuming to train.  

**Ref1**
Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, Yi Yang “Random Erasing Data Augmentation”
