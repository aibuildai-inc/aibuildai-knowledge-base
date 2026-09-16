# 25th place solution (Trained DeepSort + Hyperparameter Sweep + Gmmreg + Hungarian Algorithm)

Competition: nfl-health-and-safety-helmet-assignment
Rank: #25
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/285153

# 25th place solution

First of all, I would like to thank the hosts for this amazing competition it was truly interesting and challenging!

Our solution followed the following 4 steps pipeline:
1. Helmet detection using YoloV5
1. Bounding box fusion using WBF
1. Helmet tracking using DeepSort
1. Label assignment (i.e. mapping tracking data to video data)

##1) Helmet detection    
We trained 5 yolov5x6 using 5-fold CV for 10 epochs at full image resolution (~1h per epoch). You can find the whole notebook [HERE](https://colab.research.google.com/drive/1tI0_nMW_V_tAWd_AIS3zc71nNQt5xZZB?usp=sharing)

##2) Bounding box fusion    
We then fused the boxes using WBF

##3) Tracking    
We trained DeepSort using comp data (which boosted LB by a bit). We trained that by simply using the cropped boxes and training a classification model (with 1k+ classes). You can find the training notebook [HERE](https://colab.research.google.com/drive/1APloA_9FTE3ou9PBCDajOBtoBMoJIPxl?usp=sharing)   
We also did hyperparameter tuning using WandB Sweeps, you can find the optimization notebook [HERE](https://colab.research.google.com/drive/1nGuWjZRCxF6xtmO2u4hnEgR0C42wrXpp?usp=sharing)


##4) Label assignment    
By far the most challenging part was doing the label assignment. You can see our algorithm in the image below.

By far the most important part of this algorithm was label averaging (using tracking information) and the linear sum assignment (Hungarian algorithm).
Registration was performed using gmmreg.

You can see the whole code [HERE](https://www.kaggle.com/coldfir3/nfl-label-assignment?scriptVersionId=77839706)

There are a few hyperparameters that we tuned using the amazing WandB sweep tools, you can find the notebook [HERE](https://www.kaggle.com/coldfir3/nfl-label-assignment-sweep)



## Things that didn't work for us:    
1. PointNet: could not make the network to learn anything useful ([Notebook1](https://www.kaggle.com/coldfir3/point-net-v0?scriptVersionId=77205269) and [Notebook2](https://colab.research.google.com/drive/1Wx9uLYX1pZFTPHc55A82Yqklh5mtTZ8y?usp=sharing)) 
1. ProbReg registration algorithm: it works but `gmmreg` is much better
1. Classification based on image: too many classes and too much occlusion (network can learn on train set but do not generalize well)
