# 41st place Solution (7th Public LB) YOLO-Z Inspired Model with Tracking

Competition: tensorflow-great-barrier-reef
Rank: #38
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307807

I want to start by saying that I am not experienced in data science. I don't exactly know all the bells and whistles in data science but I love to learn and experiment. In terms of this competition, I learned a ton about object detection and it was really fun!

# Data
I used a groupkfold split on sequences since the data in private LB will be new sequences, the best way to validate a model (at least in my noob eyes) is with new sequences it has not seen before. I only trained models on the same fold and did not ensemble or use validate models via oof predictions. 

# Model
My solution revolved around the [YOLO-Z](https://arxiv.org/pdf/2112.11798.pdf) paper that looked into improving the original YOLO5 architecture in small object detection. It looks like their proposed model is somewhat included in YOLO5v6. They mention that in their experiments modifying the "width" multiplier to that of the next tier (using L or M "width" for M or S models) they achieved better results in small object detection. Since YOLO5v6 already includes smaller layers (see [this](https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/303766) discussion thread), I went into the direction of modifying YOLO5v6 models to increase the width multiplier (to see if this does better than the original). This increases the number of parameters but not the number of layers. This would keep inference time lower than using a bigger model and give the best of both tiers.

|     Model     | Layers | params (M) | FLOPs |
|:-------------:|:------:|:----------:|:-----:|
| YOLOv5s6      | 280    | 12.6       | 16.8  |
| YOLOv5s6-wide | 280    | 27.6       | 35.8  |
| YOLOv5m6      | 378    | 35.7       | 50.0  |
| YOLOv5m6-wide | 378    | 62.6       | 86.5  |
| YOLOv5l6      | 476    | 76.7       | 111.4 |

To not have to re-train the models, I had to use to transfer weights from the next model (if I want to use YOLOv5s6-wide, use YOLOv5m6 weights, etc.). I still wanted to pre-train weights since I deleted a lot of layers in the process.

I only selected YOLOv5s6-wide and YOLOv5m6-wide to train and use.

# Training
Because the "wider" models deleted many layers I thought re-training would help better fit the models. After searching a bit I found [Detecting Underwater Objects](https://github.com/chongweiliu/DUO) dataset based on competitions listed [here](https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/304367). 

I trained for 100 epochs with heavy augmentations (see [wandb output](https://wandb.ai/outwrest/gbr/reports/m-wide-duo-1280--VmlldzoxNTYzMDk0?accessToken=q1nh3kbk1u39ydfs53h1ffnuchpig7sntec09i7rhkszjd2dcjw6wenilritoyno)). 

I used the COT mask dataset to make 10,000 "fake" images of cots pasted in places with different opacity levels. I found that training with this dataset increases recall in the final model (thanks @alexandrecc). But I don't 100% know if it helps in public LB.
See this [notebook](https://www.kaggle.com/outwrest/augmentation-using-image-blending-cot-masks) and [results](https://wandb.ai/outwrest/gbr/reports/m-wide-fake-1280--VmlldzoxNTYzMTQz?accessToken=341vkrc709f82dx5xs0vxlv0j17wqj5asq8wptfc20x7jyun27ud4r2208pu9448).

Trained again on sequences with 3000 IMG size -> and fined tuned again by training with a much lower LR and less augmentations. I used simple norfair tracking notebook as a postprocessing technique.

# Overfitting Public LB
I noticed that by submitting 2x img size (6000) without TTA I was able to overfit LB pretty easily and achieved 7th place in LB with some CONF tunning. 

The best overfit was YOLOv5s6-wide with 7200 img size and no TTA trained on 3600 img size, 0.769 Public-0.641 Private.
The best notebook I had is YOLOv5m6-wide with 6000 img size and TTA trained on 3000 img size, 0.673 Public-0.706 Private.

# TLDR
Modified YOLOv5m6 model to include more parameters -> pretrained on DUO dataset -> pretrainedx2 on a "fake" dataset (I am not 100% sure this helped) -> trained on sequences -> finedtuned with lower LR again.


Hopefully, this helps. I feel like the score can be improved even with an ensemble and more conf/iou tuning since I did not explore that much. Thanks to other competitors and I hope to learn from your writeups.
