# public 3rd place solution

Competition: dfl-bundesliga-data-shootout
Rank: #3
Source: https://www.kaggle.com/c/dfl-bundesliga-data-shootout/discussion/360236

Congrats to all the winners, especially for Team Hydrogen( @philippsinger, @ybabakhin, @ilu000) and @kmat2019, your scores are really impressive!

I’ll write up a summary of my solution.

## Baseline
It ends up being a very simple end2end 2.5d model.
Backbone is EfficientNet with TSM[1], and head is simple 1D UNet.

But there are many tricks in both training and inference.

---

## Training

### 1. Pretrain with ball detection
Even though I used ImageNet pretrained model for efficientnet, inserting TSM module changes model structure drastically, so it was hard to control training. (ex. easy to overfit or diverge.) And intuitively it seems important to teach a model to focus around the ball, so I used ball annotation in the SoccerNet tracking dataset[2] to pretrain the model.  

Pretrain set up is following:
- Efnet + TSM (+ shallow conv2d head)  
- Output is the same resolution as the final feature map of efnet, 1/32 scale heatmap.  
- Label is 2d gaussian heat map like CenterNet, but didn’t add regression branch for simplicity  

At first I was not sure whether we can use SoccerNet or not, so I decided to annotate balls for the video clips by myself, it ended up being more than 20,000 frames. But after confirming we can use SoccerNet, I switched to it. Pretrain performance is better than mine, but the final score is not much different.

Prediction is like this. I tried to use it as additional input to the next stage, but no successful result.


### 2.Label design
1d gaussian heat map +- 5 frames around the event. Here is an example for challenge at frame 13. (num_frames=16)

Also create a binary mask to represent whether the frame is in eval interval or not.

### 3.Sampling
Half pos and half neg. Pos means at least 1 event in the label and neg means randomly sampled from eval interval. 

### 4. Augmentation
Basic ones like RandomHorizontalFlip, RandomRotation, ColorJitter.
Also manifold mixup[3] works well. I applied mixup on image-wise features.

### 5. Model architecture
- EfficientNet + TSM at the end of each blocks
- Simple 1d UNet (some variants have LSTM and GRU at the end)
- LayerNorm instead of BatchNorm in the UNet (BN didn’t converge maybe due to small batch size)

### 6. Loss
Base loss is focal loss[6]. 
The tasks for play, challenge and throwin are actually independent. But challenge and throwin are difficult to learn as there are much fewer samples than play. So I adjusted the alpha factor in focal loss based on the number of samples in each class. It’s inspired by class balanced loss[7], but the formulation is modified. This stabilizes training, especially before using the ball pretrained model.
```
def _calc_weight(self, samples_per_class, beta=0.9999):
    effective_num = 1.0 - beta**samples_per_class
    weights = (1.0 - beta) / effective_num
    weights = weights / torch.sum(weights)
    weights = weights.unsqueeze(0).float()
    return weights.cuda()
```

### 7. Fine tune with long sequence
After training with a relatively short clip like 32 frames, I fine tune the model with a long sequence like 128 frames, but with freezing backbone so that it fits in gpu memory.

### 8. Others
- LR warmup was important somehow
- EMA[4]
- Gradient checkpointing[5] helps to fit large model in memory
- Mixed precision + cast image to float16 before feeding to the model

---

## Inference
### 1.Video loading
cv2.VideoCapture + multi worker pytorch dataloader doesn't work well. Video decoding can be easily bottleneck, so I brought imutils[8] library’s threading implementation and modified it a bit.  
Also, I used multi image sizes [(540, 960), (480, 800), (360, 640)]. It’s faster to create small images by resizing (540, 960) image after transferring it to the GPU.

### 2.Overlapping predictions
My model predicts 64 frames at the same time. But I realized that predictions around edges are relatively inaccurate, so I decided to drop them and take only center 48 frames out of 64 frames. Below diagram shows how it works. 


### 3.Postprocess
Simply apply nms. Thresholds are [12, 6, 6] for [challenge, play, throwin] respectively.

### 4.Ensemble
To maximize the effect of the ensemble, I prepared various models with different folds.
- Image size is one of [(540, 960), (480, 800), (360, 640)] 
- Backbone is one of [b2, b3, b4]
- Sequence length is one of [32, 64, 128]  

Single best model scores 0.768.   
((360,640), efnet-b2, seq_len=128, all data)  
Ensemble 6 models with various combination reaches 0.827.

## What didn’t work

Too many to list up.
- Crop around ball area
- Crop out no player area
- Feed ball heatmap as additional input
- Transformer head like ActionFormer or VideoMAE
- Larger backbone
- Pseudo label
- Many others..


## Reference
[1] https://github.com/MIT-HAN-LAB/temporal-shift-module
[2] https://github.com/SoccerNet/sn-tracking
[3] https://arxiv.org/abs/1806.05236
[4] https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/utils/model_ema.py
[5] https://github.com/cybertronai/gradient-checkpointing
[6] https://arxiv.org/abs/1708.02002
[7] https://arxiv.org/abs/1901.05555
[8] https://github.com/PyImageSearch/imutils
