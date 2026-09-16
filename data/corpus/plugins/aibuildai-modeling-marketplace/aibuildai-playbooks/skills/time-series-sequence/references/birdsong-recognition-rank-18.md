# 18th place solution: efficientnet b3

Competition: birdsong-recognition
Rank: #18
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183219

First of all I want to thank the host and Kaggle for this very challenging competition, with a truly hidden test set.  A bit too hidden maybe, but thanks to the community latecomers like us could get a submission template up and running in a couple of days.  I also want to thank my team mate Kazuki, without whom I would probably have given up after many failed attempts to beat the public notebook baseline...

**Overview**

Our best subs are single efficientnet models trained on log mel spectrograms.  For our baseline I started from scratch rather than reusing the excellent baselines that were available.  Reason is that I enter Kaggle competition to learn, and I learn more when I try from scratch than when I  modify someone else' code.  We then evolved that baseline as we could in the two weeks we had before the end of competition.

Given the overall approach is well known and probably used by most participants here I will only discuss the items that may be a bit different from what others did.

**Training data clips**

It was clear from host that training on random 5 second clips had a drawback: some of the clips may not contain the target bird.  We then used a simple hypothesis: clips were stripped, i.e. periods without a song at the beginning or at the end were removed for the sake of reducing storage needs.  We therefore trained on first 5 seconds or last 5 seconds of clips, assuming these would contain the target bird.  We preprocessed all data to be sampled at 32 kHz.

**Noise**
We added noise extracted from the two test sequences made available, a bit like what Theo Viel did.  But we used the meta data to extract sub sequences without bird call, then we merged these sequence with a smooth transition between them.  We then added a random clip of the merges sequences to our training clips

**No Call**
We added the freefield1010 clips that were labelled as nocall to our training data.  We added a 265th class to represent the no call.  As a result our model could predict both one or more birds, and a nocall. Adding this data and the nocall class was probably the most important single improvement we saw in CvV and LB scores.  It is what led us to pass the public notebook baseline.

**Multi Bird Clips**
The main documented difference between train and test data is that train is a multi class data while test is a multi label data.  Therefore we implemented a mixup variant were up to 3 clips could be merged.  This is not really mixup as the target for the merged clip is the maximum of the targets of each merged clip.

**Secondary labels**
Primary labels were noisy, but secondary labels were even noisier.  As a result we masked the loss for secondary labels as we didn't want to force the model to learn a presence or an absence when we don't know.  We therefore defined a secondary mask that nullifies the BCE loss for secondary labels.  For instance, assuming only 3 ebird_code b0, b1, and b2, and a clip with primary label b0 and secondary label b1, then these two target values are possible:

[1, 0, 0]

[1, 1, 0]

The secondary mask is therefore:

[1, 0, 1]

For merged clips, a target is masked if it it not one of the primary labels and if it is one of the secondary labels.

**Loss Function**

We use binary cross entropy on one hot encoding of ebird codes.  Using bce rather than softmax makes sense as bce extends to multi label seamlessly.  We tried dice loss to directly optimize F! score, but for some reason this led to very strong overfitting.

**Class Weights**

The number of record per species is not always 100.  In order to not penalize the less frequent one we use class weights inversely proportional to class frequencies.  And for the nocall class we set it to 1 even though it was way more frequent than each bird classes to make sure the model learns about nocall correctly.

**Model**

Our best model was efficientnet on log mel spectrograms.  We resized images to be twice the size of effnet images: 240x480 for effnet b1, 260x520 for effnet b2, and 300x600 for effnet b3.  We started from efficientnet_pytorch pretrained models.  We tried the attention head from PANNs models but it led to severe overfitting.  I am not sure why to be honest, maybe we did something wrong.

**Training**
Nothings fancy, adam optimizer and cosine scheduler with 60 epochs.  In general last epoch was the one with best score and we used last epoch weights for scoring.


**Log Mel Spectrogram**

Nothing fancy, except that we saw a lot of power in low frequencies of the first spectrograms we created. As a result we clipped frequency to be at least 300 Hz. We also clipped them to be below 16 kHz given test data was sampled at twice that frequency.  

**Augmentations**

Time and pitch variations were implements in a very simple way: modify the length of the clipped sequence, and modify the sampling rate, without modifying the data itself.  For instance, we would read 5.2 seconds of a clip instead of 5 seconds, and we could tell librosa that the sampling rate was 0.9*32 kHz.  We then compute the hop so that the number of stft is equal to the image width for the effnet model we are training.  We also computed the number of mel bins to be equal to the height of the image.  As a result we never had to resample data nor resize images, which speed up training and inferencing quite a bit.  There was an issue with that still: this led us to use a high nftt value of 2048 which lead to poor time resolution.  We ended up with nfft of 1048 and a resize of the images in the height dimension.

**Cross Validation**

We started with cross validating on our training data (first or last 5 seconds of clips) but it was rapidly clear that it was not representative of the LB score.  And given we had very few submissions, we could not perform LB probing.  We therefore spent some time during last week to create a CV score that was correlated with the public LB.  Our CV score is computed using multi bird clips for 46% of the score, and nocall clips for 54% of the score.  We added warblrb and birdvox nocall clips to the freefield1010 for valuating no call performance.  We tuned the proportion of each possible number of clips in multi bird clips, and the amount of noise until we could find a relatively good CV LB relationship, see the picture below: x is cv, y is public lb.



The correlation with private LB is also quite good:



We then used it to guide our last few submissions training and thresholding.  Our CV said 0.7 was best, and late submissions proved it was right.  We ended up selecting our 2 best CV submissions, which were also the 2 best public LB submissions, and also the two best private Lb submissions.

**Conclusion**

These were probably the most intensive two weeks I had on Kaggle since a long time.  My first two subs 14 days ago were a failure, then a LB score of 0. Kazuki had started a bit earlier, but not much. I am very happy about where we landed, and I am not sure we would have done much better with few more days.  Maybe using effnet b4 or b5 wold have moved us higher but I am not sure.  I am looking for gold medalist solutions to see what we missed.  I'm sure I'll learn quite a bit.
