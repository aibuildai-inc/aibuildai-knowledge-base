# 1st place solution

Competition: deepfake-detection-challenge
Rank: #1
Source: https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721

# Keep it simple
I used a frame-by-frame classification approach as many other competitors did. 
Tried a lot of other complex things but in the end it was better to just use a classifier.

### Data preparation
- extracted boxes and landmarks with MTCNN and saved them as json
- extracted crops in original size and saved them as png
- extracted SSIM masks with difference between real and fake and saved them as png

### Face-Detector
I used simple MTCNN detector.
Input size for face detector was caluclated for each video depending on video resolution. 
- 2x resize for videos with less than 300 pixels wider side
- no resize for videos with wider side between 300 and 1000
- 0.5x resize for videos with wider side &gt; 1000 pixels
- 0.33x resize for videos with wider side &gt; 1900 pixels

### Input size
As soon as I discovered that EfficientNets significantly outperform other encoders I used only them in my solution. 
As I started with B4 I decided to use "native" size for that network (380x380). 
Due to memory costraints I did not increase input size even for B7 encoder.

### Margin
When I generated crops for training I added 30% of face crop size from each side and used only this setting during the competition.


### Encoders
I tried multiple variants of efficient nets at the beginning:

- solo B3 (300x300) - 0.29 public
- solo B4 (380x380) - 0.27 public
- solo B5 (380x380) - 0.25 public
- solo B6 (380x380) - 0.27 public (surprisingly it was worse than B5 and I have not tried B7 until competition last week)
- solo B7 (380x380) - 0.24 public 

In the end I used two submits with:
- 15xB5 (different seeds) with heursitic overfitted for Public LB which was trained with standard augmentations - 10th place on private
- 7xB7 (different seeds) with more conservative avergaing heursitic and trained with hardcore augmentations - 3rd place on private 

### Averaging predictions:
I used 32 frames for each video.
For each model output instead of simple averaging I used the following heuristic  which worked quite well on public leaderbord (0.25 -&gt; 0.22 solo B5).

```
def confident_strategy(pred, t=0.87):
    pred = np.array(pred)
    size = len(pred)
    fakes = np.count_nonzero(pred &gt; t)
    if fakes &gt; size // 3 and fakes &gt; 11:
        return np.mean(pred[pred &gt; t])
    elif np.count_nonzero(pred &lt; 0.2) &gt; 0.6 * size:
        return np.mean(pred[pred &lt; 0.2])
    else:
        return np.mean(pred)
```

I.e. I used only confident predictions for averaging if they passed some thresholds. 
Though it worked well on public leaderbord to be safe for the second submit (3rd private) I used more conservative thersholds for real videos. 

### Validation strategy

I used 0-2 folders as holdout at the beginning. 
But 400 videos from the public test had more correlation with Public LB and I switched to this approach.

I tracked two log loss metrics (on averaged probabilities per video)
- logloss on real videos 
- logloss on fake videos 

From the validation I can say logloss on FAKE videos was much lower than on REAL. I.e. fakes were too easy to spot.
Which was not encouraging. I guess public leaderboard also has this characteristic as my validation had strong correlation with public LB. 
It is not the case on private set as it would contain new/different FaceSwap methods. 

### Augmentations

I used image compression, noise, blur, resize with different interpolations, color jittering, scaling and rotations
```
def create_train_transforms(size=380):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        IsotropicResize(max_side=size)
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ]
    )
```
### Generalization approach
I expected it won't be enough for generalization and spent a few weeks working on domain specific augmentations. 

I wanted to push models to learn the following properties:
- visual artifacts (models learn that easily even without any augmentations)
- different encoding of face from other part of the image. Big margin helps with that. 
- face warping artifacts. Big marging helps with that as well. Related article https://arxiv.org/abs/1811.00656 to catch FWA
- blending artifacts - here we need either to predict blending mask (https://arxiv.org/abs/1912.13458) but it's not possible to obtain ground truth mask as there is no big difference in pixels/SSIM on the edge due to blurring and other techniques used to reduce blending artifacts or come up with augmentations that **destroy** visual artifacts.

To catch face blending artefacts:
1. removed half face horisontally or vertically. Used dlib face convex hulls. 
2. blacked out landmarks (eyes, nose or mouth). Used MTCNN landmarks for that.
3. blacked out half of the image. To be safe I checked that it will not delete highly confident difference from masks generated with SSIM. 



I don't really know if it helped on private leaderboard though.

### Training schedule
- sampled fake crops based  on number of real crops i.e. `fakes.sample(n=num_real, replace=False, random_state=seed)`
- 2500 iterations per epoch
- SGD, momentum=0.9, weight decay=1e-4
- PolyLR with 0.01 starting LR
- 75k iterations
- used Apex with mixed precision
- trained on 4 GPUs with SyncBN and DDP. Batch size 16x4 for B5, 12x4 for B7.
- label smoothing - for me label smoothing with 0.01 eps was optimal on public leaderboard. Also it allowed not to use clipping at all. 

 

### Hardware
Unfortunately I did not ask aws credits from organizers, hoped that my home workstations would be enough -  I was wrong!
I have two devboxes: one with 2xTitan Vs, the other with 4xTitan Vs

Huge thanks to hostkey provider (https://www.hostkey.com/) that gave me a grant http://landing.hostkey.com/grants?_ga=2.3307699.2051560741.1587714719-1038061670.1587714719 !
I got a devbox with 4x2080Ti for two months! Additional 4 GPUs helped me a lot to iterate faster as my models required all of them for training a single model. 


### Things that I tried but did not work well enough
- Metric learning - was worse than a simple classificator
- UNet with SSIM difference prediction - got the same LB score.
- Self supervision with blending real faces. Extracted convex hull for real faces warped/resized/compressed face and then blended it back and changed label to fake. Models learned that quickly but that did not give any boost on validation/public leaderboard.
- Self supervision with rotations like here https://stanford-cs221.github.io/autumn2019-extra/posters/110.pdf - did not improve my score
- Temperature scaling - risky, no big difference. Decided not to use it. 
- Using steganalysis. Did not imporve validation score.
- RNN - I guess it would work, but due to CPU bottleneck it would be hard to extract multiple chunks in kernel. My first model with RNN + Resnet34 scored around 0.4 on public and a simple classifier was better. 


### No External data
That was a huge concern for me because nothing is basically allowed.  I decided to be safe and did not use any external data.


### Github
https://github.com/selimsef/dfdc_deepfake_challenge
