# Public 2nd (Private 30th) Place Solution

Competition: deepfake-detection-challenge
Rank: #2
Source: https://www.kaggle.com/c/deepfake-detection-challenge/discussion/140236

Long 4 months have passed. Applaud to participants who have been working hard and thanks to hosts who arranged this dataset and competition.

In social aspect, automatic deepfake detection algorithm will be a must in near future. In personal aspect, this competition was held in a good timing for me. So dfdc took priority in my head for the last 4 months.

Luckily, I somehow managed to attain 2nd place in public leaderboard. While waiting for the private leaderboard to be revealed, I'll share my solution. 

Note that all of the following are what ***I*** have done for this competition. (This post does not include what my teammates have done)

For those who are curious, I'll first share the major methods that I came up with, which improved public lb score.

# Ingredients for Public LB

### 1. Augmentations

Extensive augmentations significantly improved public lb score and reduced cv-lb gap. I guess it makes model robust on varying data.

### 2. Face Margin

I set margin when cropping face, where `new_face_width=face_width*(1+margin)` (same with height). Margin of 0.5~0.7 worked significantly better than 0 and further reduced cv-lb gap. I guess model learns to detect some inconsistency between manipulated region and surrounding region.

Tuning augmentations and face margin were the two main ingredients that boosted public lb significantly.

### 3. Model Capacity

Efficientnet-b4 did quite better than efficientnet-b0. With extensive augmentations, appropriate model size improved the score.

### 4. Multi-task Learning

I used Unet with classification branch for model architecture. I calculated pixel-wise difference between fake and corresponding original video, generated mask with pixels that have outstanding differences, and used it as segmentation part target. By using mask information, the model is able to know which part of the image is modified and I guess it helped boost performance.

### 5. Conservative Fix

Train data and test data have quite different distribution, as can be seen in cv-lb discrepancy. Multiplying constant (&lt;1) to the logit then taking sigmoid helped improve logloss in this situation.

### 6. Ensemble

Ensemble always helps.

### 7. N Frames when Inferencing

I used simple average of frames to get probability, so the more frames extracted from the video, the better, until it hits 9 hours restriction.

<br><br>

Now, I'll go into details of my journey. I'll separate it into 10 phases, each of where I concentrated on certain subject. (number inside parenthesis in the title is approximate public lb score at that phase)

# Phase 1. Setting Pipeline (0.69)

This was my first encounter to video type vision task and deepfake detection. So at the very beginning of the competition, I started searching google for articles and papers, and also there were nice posts in kaggle discussion that introduced related papers.

At first, I tried to go with network architectures that take care of temporal information, but I found out that they lack pretrained weights and are very heavy to train, so I decided to make baseline based on XceptionNet which was introduced in faceforensics paper( https://arxiv.org/abs/1901.08971 ).

At my initial experiments, I tried to integrate audio part in the image model by using spectrogram as second input, but it didn't help that much and complicated the process, so I abandoned audio part. Anyway my teammate wanted audio information, so I set up the pipeline which uses ffmpeg with subprocess to extract audio stably in both train and public test videos. (I used PyAv first, but it wasn't stable with public test videos.)

Reading video was the main bottleneck for the runtime, so I struggled with optimizing the process. At last, I selected method kindly provided at https://www.kaggle.com/c/deepfake-detection-challenge/discussion/122328 which was the fastest among those I tried. I extracted 20 frames evenly. Duration is 10 seconds so I extracted frame every 0.5s.

Next step was to select fast but accurate face detector. I searched and found out that `retinaface` is the best available face detector evaluated on widerface dataset( http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html ). Thereafter, I used its pytorch implementation https://github.com/biubug6/Pytorch_Retinaface .

To save data loading time when training the network, I first processed each `original` videos and saved them as compressed joblib files at disk. I extract 20 frames from each video, detect face, crop them, and with additional meta information extracted, save them. The saved video will be a numpy array with shape `(n_people, n_detected_frames, height, width, 3)`. There were a lot of trivial issues and bugs when processing videos. I'll introduce some.

* Cropped frames don't have equal size -&gt; Just resized to match the maximum size within one video
* Some videos have 2 people -&gt; Set threshold to confidence score and if the confident score of second confident face detected is bigger than the threshold, confirm there are 2 people
* Detector finds face in some frames but doesn't in other frames within one video -&gt; just ignore no-detected frames (so the output can have &lt;20 frames)
* In some videos, detector fails to detect face -&gt; lower the confidence threshold for first confident face

After processing all original videos, I used the bounding box information of original videos to process corresponding fake videos. I could use multiprocessing to speed up this process, since it didn't use cuda. After I processed all videos, I updated given metadata with the new extracted information.

It took about 12 hours on my computer to process all of the train dataset. Then I trained the model with processed data.

Inferencing in Kaggle notebook was another obstacle. Initially my notebook failed many times with varying error messages. I concluded there are some corrupt videos in public test set, so I introduced `try except` block and the submission went well thereafter.

# Phase 2. UNet Architecture (0.6~)

In the Understanding Cloud Organization competition( https://www.kaggle.com/c/understanding_cloud_organization/discussion ), I learned that multi-task learning with classification and segmentation helps improve model's classification score.

I could generate masks by selecting pixels from the fake video that have large differences from corresponding real video. All of the training videos are compressed, so the masks were not perfect, but the strategy worked in my validation set and public test set (about 0.01 improvement in lb). 

By using mask information, the model was able to know which part of the image is modified and I guess it helped boost performance.

I used UNet with classification branch. Also, since efficientnets are the state of the art architectures in imagenet, I used them as the encoder of Unet.

At this moment, I was using first frame from each videos.

# Phase 3. CV-LB Discrepancy (0.6~)

Then this huge problem came. My validation score and public lb score didn't correlate and had large gap (0.2 vs 0.6). It was the main problem in this competition from the start to the end. I managed to decrease the gap to ~0.05 at the end, but still don't fully understand how public test data differ from train data.

# Phase 4. Augmentations (0.40)

I was only using horizontal flip since I thought augmentation will distort features that were introduced by manipulation. But it was wrong after all. Augmentation made the model robust on public test set.

I added some basic augmentation such as shift, scale, rotate, rgbshift, brightness, contrast, hue, saturation, value. Also, referencing dfdc preview paper( https://arxiv.org/abs/1910.08854 ), I added JpegCompression and Downscale augmentation too.

Local validation score improved a bit but the public score jumped massively to 0.4.

# Phase 5. Large Models &amp; Learning Rate (0.33)

I was using `efficientnet-b0` and I switched to `efficientnet-b3`. It scored 0.36, then I switched to `efficientnet-b7` and it scored 0.33.

By that time I thought the improvement was solely due to model capacity, but some of the improvement was actually due to learning rate - batch size relationship.

# Phase 6. Experiments (0.3)

I thought I had some correlation with cv and lb by this time, so I tried a lot of experiment with local validation. I'll introduce some.

### What Didn't Work for LB

* Different learning rate between encoder - decoder
* Pad rather than resize when feeding the image
* CNN-LSTM architecture
* Construct model so that it uses statistical information across frames (ex. mean, std)
* Tune segmentation loss weight
* Guarantee fake-original pair to exist in each batch
* Use scse option in decoder
* Decrease face margin
* Use Vggface2 pretrained inceptionresnetv1 provided by FaceNet pytorch as encoder
* Validating with compress/downscaled validation set
* Split validation set with actors grouped by face encoding + KMeans
* One cycle learning rate scheduler
* Integrate retinaface confidence score as additional feature
* Stacking across frames with LGBM

### What Worked for LB

* Tune learning rate considering batch size (batch size 24 - lr 0.0002)
* RAdam + ReduceLROnPlateau
* Tune ratio of decreasing the learning rate when plateau (0.1-&gt;0.3)
* Use different frames of each video every epoch (#0-&gt;#7-&gt;#14-&gt;#1-&gt;#8-&gt;...)
* Increase resolution
* Use `conservative fix`: multiply constant(&lt;1) to the logits than take sigmoid. It helps improve logloss when the training and testing distributions differ.

# Phase 7. More Augmentations (0.27)

After testing the tuned model on public lb, I once again found out cv and lb didn't correlate at most experiments I did. I needed to select cv vs lb at this time, and decided to value lb more, since the hosts implied that the private test set will be different than train set. So I started to validate experiments on public lb. Also, I decided to use b5 for my experiment, since I judged b0 had too low capacity when augmentations are applied.

Then I remembered I got significant boost when I did more augmentations, so I added extensive augmentations including noise and blur, and increased the degree of augmentations. Score jumped to 0.269. I did harder augmentations but it didn't improve, so I stopped tuning augmentations.

# Phase 8. More Experiments (0.27)

I did more experiments, but sadly nothing worked.

* Mixup fake-original pair
* Label smoothing
* Undersample rather than weighted loss
* Generate a mask using landmark information extracted by retinaface, and use it as an augmentation
* Use difference itself as a segmentation target, not (difference&gt;threshold)

# Phase 9. Increase Face Margin (0.214)

I had an experience of increasing the face margin from 0.05-&gt;0.1, and it seemed to have increased overall public lb score. So I decided to increase face margin more.

I experimented face margin of 0.5, and single b5 scored astonishing 0.222. So I increased face margin to 1.0 and started to tune how much I should crop the margins.

With margin crop of 0.15 (margin of 0.7), I could get 0.218, and could get 0.214 with b4.

# Phase 10. Ensemble (0.199)

I trained 7 b4s and 3 b5s with different seeds and did simple average. Also I increased nframes to 30 when inferencing. It resulted in my final public lb score of 0.199.
