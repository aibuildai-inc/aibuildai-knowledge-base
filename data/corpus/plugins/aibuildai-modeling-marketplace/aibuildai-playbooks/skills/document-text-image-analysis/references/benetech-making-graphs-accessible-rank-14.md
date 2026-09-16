# 14th Place Solution

Competition: benetech-making-graphs-accessible
Rank: #14
Source: https://www.kaggle.com/c/benetech-making-graphs-accessible/discussion/418323

First, congratulations to the winning teams! This task was very interesting and challenging. I hope to learn a lot from the top solutions.

My solution consists of chart_type classification, template matching for scatter and pix2struct model for the rest chart_types.

# Chart_type Classification
I finetuned resnet18d with about 70,000 images in the ICDAR dataset, resulting in the accuracy of 0.995 for extracted data.

# Template matching for scatter
Scatter often requires a large number of datapoints, and long token lengths are needed. I addressed this problem by object detection, OCR and template matching. This approach scores 0.51 on extracted data. The algorithm is as follows:

1. object detection of data points and axis texts by YOLOX (im_size=480)
2. generate a pattern of datapoints using the highest confidence among the detected points
3. identify the shape of the pattern based on the distance between the RGB values of the pattern and the background color
4. scan the entire image and calculate the similarity score (L2 distance) with the pattern
5. select approximately 1000 pixels with the highest similarity score as candidate points
6. fill the rectangular area containing the data points with background color and use it as a base for reconstruction
7. take out the candidate points in order from the one with the highest similarity score, and place a pattern if the following two conditions are satisfied
  1. the L2 error with the original image is smaller when a pattern is placed than when a pattern is not placed
  2. the overlap with the existing pattern does not exceed 30%.



Since simple linear regression is vulnerable to outliers, I tried to use some form of robust regression. Considering that methods such as RANSAC and huber require a parameter on the scale and that the parameters tend to overfit the data set due to small data size, I implemented a simple robust regression without any hyper parameters. All two point pairs are connected by a straight line and the slope and intercept are calculated. The desired regression line is obtained using the median value for all computed slopes and intercepts. 



## Pix2Struct model for other types
- labels are in the form xyxy
- decimal part of numerical is rounded to 4 significant digits
- augmentation for translation, rotation, noise, hue, etc.
- pretrained model: MatCha-base
- hyperparameters
  - max_length: 512
  - max_patches: 2048
  - lr: 1e-5
  - scheduler: get_cosine_schedule_with_warmup
  - num_warmup_steps: 1000
  - weight_decay: 1e-5
- Training data
  - extracted (1118)
  - generated (999)
  - ICDAR (labeled 1286+ pseudo&hand labeled 2988)
