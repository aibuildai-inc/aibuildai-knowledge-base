# 10th-place solution - Agnostic loss for semi-supervised learning

Competition: rfcx-species-audio-detection
Rank: #10
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220319

Thank you to the organisers, Kaggle, and to everyone who shared ideas and code for this competition. I learned a lot, as I'm sure many of you have, and I thought I would break down my approach since I know many competitors couldn't find a way to use the False Positive labels. I'm thrilled to have secured my first gold (and solo gold) in a competition, so it's the least I could do!

### Summary

On a high-level, my approach was as follows:

* train models using existing labels
* generate pseudo-labels on train/test
* isolate the frames which had a very high/low prob across an ensemble of models
* conservatively threshold these and use as new TP/FP values for the relevant class
* repeat

This gradually increased the amount of data I had to work with, until I had at least 2 frames from each recording with an identified TP and/or FP. The growing data diversity allowed successive models to generalise better.

### What Worked

**Fixed Windows**

I used a window of 5 seconds centred on the TP/FP. For predicting on test, I used overlapping windows with a step of 2.5 seconds. This ensured that train/test had identical preprocessing for their inputs. The maximum value was taken for each class across all of these windows. For some submissions, I used the average of the top 3 predictions but this didn't seem to notably change the LB score.

**Agnostic Loss**

Perhaps a better term already exists for this, but this was what I called my method for using both the TPs and FPs in a semi-supervised fashion. The problem we face with unlabelled data is that any spectrogram can contain multiple classes, so setting the target as 0 for everything apart from the *given* label will penalise true positives for other classes. We can only know for sure that one class is present or absent, and the loss needs to reflect this. So I excluded all non-definite targets from the loss calculation. In the target tensor, a TP is 1 while an FP is 0. Unlabelled classes are given as 0.5. These values are then excluded from the loss calculation. So if we had 5 classes (we have 24, but I'm saving room here) and this time window contained a TP for class 0 and an FP for class 3:

```
y = torch.Tensor([1., 0.5, 0.5, 0., 0.5]) 

```
And in the loss calculation:
```
preds = model(inputs)
preds[targets==0.5] = 0                    
loss = BCEWithLogitsLoss(preds, targets)
loss.backward()
```
Thus the model is 'agnostic' to the majority of the potential labels. This allows the model to build a guided feature representation of the different classes without being inadvertently given false negatives. This approach gave me substantially better LB scores.

The figure of 0.5 is arbitrary and could've been any value apart from 0 or 1: the salient point is that the loss resulting from unlabelled classes is always constant. Note that this kind of inplace operation is incompatible with `nn.Sigmoid` or its functional equivalent when performing backprop so you need to use the raw logits via `torch.nn.BCEWithLogitsLoss()`.


**ResNeSt**

I found EfficientNet to be surprisingly poor in this competition, and all of my best scores came from using variants of ResNeSt (https://github.com/zhanghang1989/ResNeSt) paper available [here](https://arxiv.org/abs/2004.08955).

For 3-channel input I used the `librosa` mel-spectrogram with power 1, power 2 and the `delta` function to capture temporal information. With some models I experimented with a single power-1 spectrogram, delta and delta-delta features instead. While quicker to preprocess, I noticed no impact on scores.

I also incorporated it into the SED architecture as the encoder. This showed very promising metrics during training, and while sadly I didn't have time to run a fully-convergent example its inclusion still helped my score. In future competitions this could be a very useful model. ResNeSt itself only takes a 3-channel input and has no inbuilt function to extract features, so I had to rejig it to work properly: I'll be uploading a script with that model shortly in case anyone is interested.

**Augmentations**

From Hidehisa Arai's excellent kernel [here](https://www.kaggle.com/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english), I selected `GaussianNoiseSNR()`, `PinkNoiseSNR()`,  `TimeShift()` and `VolumeControl()`. I was wary of augmentation methods that blank out time windows or frequency bands like [SpecAugment](https://arxiv.org/abs/1904.08779). Some of the sounds occur in a very narrow frequency range (e.g. species 21) or in a very narrow time window (e.g. species 9) and I didn't want to make 'empty', positive samples that would coerce the model into learning spurious features. I also added some trivial augmentations of my own:

* swapping the first and last half of the audio vector
* adding a random constant before spectrogram normalisation (occluding the relevant features)
* 'jiggling' the time window around the centre of `t_mid`, at a maximum of 1 second offset in either direction

**Eliminating species 19**

A minor point, but this class was so rare that setting all of its predictions to zero usually improved the LB score by 0.001. There were many unrelated sounds that would presumably cause the model to produce false positives. My best submission didn't do this however; it was a simple blend of models including ResNeSt-50, ResNeSt-101, EfficientNet-b1 and the SED architecture described above. I used weighted averaging roughly in proportion to the individual models' performance.

### What didn't work

* Using separate models for different frequency ranges (these models never gained an adequate feature representation, and produced many false positives).
* EfficientNet alone gave poor results, but helped as part of an ensemble.
* Larger models (ResNeSt101, EfficientNet b3) didn't improve scores.
* TP-only training.
* Models that worked on all windows for a single clip - these were slow and produced inferior results.

Otherwise I was quite lucky - I thought about my methodology for a while and most of what I tried worked well on the first attempt. If I'd had more time, I would have liked to try:

* automatically labelling some species as an FP of a similar class (e.g species 9 & 17)
* probing the LB for class distribution (I suspect you could get +0.9 by only predicting the most common half of the classes and ignoring everything else) - I realised the importance of this too close to the deadline
* experimenting with different encoders for the SED architecture.
* using a smaller window size (<=3 seconds) for greater fidelity. 

The overall class prediction histograms for my final submission were as follows:



Some classes gave me particular trouble. I used my own, simple scoring metric during training that recorded the proportion of positive cases that were predicted above a certain threshold. I never satisfactorily made a model that could detect the rarer classes like 6, 19 or 20 in a reliable fashion.

Overall I had an interesting time exploring how to work with the application of CNNs to spectrograms, and with large amounts of unlabelled data. In the next audio competition, perhaps I'll aim a little higher! I'm looking forward to seeing how those who scored > 0.97 managed to achieve their results.

If you have any questions I'll do my best to answer them!
