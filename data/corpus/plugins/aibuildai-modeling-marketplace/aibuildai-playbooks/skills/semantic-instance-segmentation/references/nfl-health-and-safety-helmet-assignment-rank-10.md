# 10th place solution

Competition: nfl-health-and-safety-helmet-assignment
Rank: #10
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/284945

Thanks NFL and kaggle for hosting this interesting competition.
I am happy that I get a gold in this very challenging problem.
So, Here is my approach.

Firstly, I use Yolov5 to detect helmet, then I use a regression network to match the detected helmet with tracking data. This regression network is an efficientnetb0 encoder with an Unet decoder. By using yolov5 and my regression model, I am able to reach a CV of 0.7 (0.68 public and 0.7 private). 
The output is then being postprocessed by tracking algorithm (I use deepsort and SiamRPN), apply tracking boost my CV to 0.9, public LB 0.85 and private LB 0.86.

The yolov5 and deepsort is quite popular in the public notebooks and discussion, so I won't write more about it here. I just use the public source code and optimize it to this competition.

The regression network is quite similar to a normal segmentation network.
- Encoder: EfficientnetB0
- Decoder: Unet
- Input: 2-channel image (2x256x256)
- Output: 2-channel regression mask (2x256x256). Channel 1 predict the x-coordinate and channel 2 predict y-coordinate.
	- For example, the point at position (x',y') in the output mask is responsible for predicting value (x,y).
	- Loss function: L1 loss. Loss is only calculated at the pixels where tracking data exist (white points in channel 2).  


*My real implementation is a little different where I use an extra segmentation channel to guild the network for better training. I simplified the idea here to make it easier to understand*
