# 6th place Solution Summary(0.549)

Competition: hpa-single-cell-image-classification
Rank: #6
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/239166

We would like to thank the competition host(s) and Kaggle for organizing the competition and congratulate all the winners, and anyone who benefited in some way from the competition. Special thanks to my teammates @zehuigong @thedrcat @dschettler8845 @felipebihaiek 

I am sorry for late share,because I am busy with two competitions at the same time, I only spent the last 3 weeks preparing for the competition. If I start earlier, I still have a lot of ideas that I haven’t been able to realize.
# Solution Components:
- Cell level models
- Image-level Model
- Gridify  and Gapmask Inference


# 1. Cell level models
Baseline solution:  The settings of our baseline is as follows:
• Backbone: EfficientNet-b3;
• Data: Only training data, cell-level classification.
• Optimizer: Adam, BS = 128, LR=0.001, warm up + constant learning, 16 epoch.
• Image size: 320 for training and inference.
• Augmentations: The optimized transform policy of the previous third place solution
• Focal loss (gamma=2)
• Using the offline SWA to generate the inference weights.
The above model achieved 0.351 LB score.

#### Tricks on the baseline:
(1)	Label smoothing with gaussian random noise, eps=0.05, the smooth label are generated as follows: (0.351 -> 0.412)
[random]
(2)	Mixup augmentation (0.412 ->0.428);
(3)	Attention class head (boost ~0.002);
(4)	Train + public (except for class 0 and 16) + label smoothing + mixup: 0.507
(5)	Deep supervision, add supervision on the middle layer of the backbone network, 2 layer supervision (boost ~0.004, 0.507-> 0.511), 3 layer supervision (0.511 -> 0.513). 
(6)	Merge the cell-level prediction with image-level(RGBY+G) prediction.



# 2. Image-level Model

• Backbone: EfficientNet-b7;
• Data: training +public data(RGBY+G), img-level classification.
• Image size: 600 for training and inference.
• Focal loss (gamma=2).







#### Inference

We use 8-TTA for inference, and the inference image size is the same as training image, e.g., 320.Segmentation Model is original HPA Segmentator
#### Post-processing
We use a threshold of 0.001 to filter the predictions, and merge the image-level and cell-level classification results, which achieves an improvement of 0.04 mAP, with the weight of 0.7 for cell-level, 0.3 for image level predictions, respectively (0.513 -> 0.553). 


# 3. Gridify  and Gapmask Inference
A detailed breakdown of this step has been posted in a separate [discussion](https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238365)


## Final Results
1. Original HPA Segmentatorse+2 layer  +3 layer supervision cell level+img-level +Gridify  and Gapmask Inference(  Public: 0.5771 , Private: 0.5499)
2. Faster HPA Segmentatorse+2 layer  +3 layer supervision cell level+img-level +Gridify  and Gapmask Inference(  Public: 0.5724 , Private: 0.5410)



## Ideas in mind
   - OOF models.
   - Use metric learning technique, such as ranking loss. (paper: Improving Pairwise Ranking for Multi-label Image Classification)
## Things didn’t work for us
(1)	Using GAP and GMP(global max pooling), and concatenate two output, before the final classification layer, we add two more fc layers.
(2)	Currently, our cell-level image are all multiply with the corresponding cell mask, we train a model with cell-level images without multiplying the cell mask and, adding the cell mask as an additional input channel;
(3)	BCE + Focal
(4)	Class aware training sampling, we assign the sampling ratio for each of the training images according to the class frequency. This will lower the sampling ratio for the major class images, while improve the ratios for the minor class images.
