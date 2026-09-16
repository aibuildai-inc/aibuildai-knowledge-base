# 8th solution

Competition: g2net-gravitational-wave-detection
Rank: #8
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275433

I want to thank Kaggle and Host for this amazing challenge. Topic was great, and teaming with @onodera and @titericz was great too.

We led the competition for a while then all of a sudden teams started to pass us.  In hindsight we know it is thanks to using 1D models and/or resnets. Unfortunately for us we kept working with efficientnet v2 as we were having good results so far.  We have no excuse given both ideas were shared in the forum soon enough to be leveraged.  To be honest we tried resnet34 but it did not match effnetv2 results for us.  I wonder why it is different for other teams.  We also tried 1D models briefly, but, same, they were low quality compared to effnets.  We should have submitted a blend obviously.  

Here is a snapshot of our solution.

**Data processing**

Data was multiplied by 1e19 (1e21 later) to ensure that CQT and CWT could be performed correctly using FP16.

CQT and CWT (results are very similar).  Best model  (public 0.8826 private 0.8805)  used these setting for CQT:

```
sr = 2048
hop_length = 5
fmin = 22
fmax = 22*16
bins_per_octave = 8
n_octaves = 4
n_bins = n_octaves * bins_per_octave
fscale = 1
cqt = CQT1992v2(sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax,
                        n_bins=n_bins, bins_per_octave=bins_per_octave * fscale,         
                window=('kaiser', 14), filter_scale=1/fscale,
               )
```

We used fscale = 2 (i.e. filter_scale = 0.5) in few models, it improves a bit but we could not train them as much as our best model.  We also varied hop length to get different image sizes. We then concatenated the 3 CQT images on the frequency dim, getting images of size 96xL where L depends on hop length (L = 820 for the above setting).  

**Signal To Noise EDA**

We found the best frequency range by doing a signal ratio analysis.  We computed the average of CQT images for positive samples (POS) , and divided it by the average of CQT images for negative samples (NEG).  This gives us what we called a mask.  Here is the mask for above settings

[mask]

We see where the waves are on average.

By taking mask average over detectors and time we see that we can crop images left and right safely (y axis labels should be multiplied by 3, sorry):

 [time]

Cropping sides gets rid of the CQT border artifacts.

By taking the mask mean over detectors and time we get signal to noise ratio per frequency (y axis labels should be multiplied by 3, sorry):

[freq]

x is log scale, but doing the math we see that snr is 1 outside 22Hz-352Hz

We tuned bins per octave so that the image height is a multiple of 16 (for speed), and 32 bins overall was a very good tradeoff.

**Input images**

We used nnAudio CQT (or CWT) as the first model layer.  Then 3 outputs are stacked on frequency axis. We then divided the image by NEG (average of negative sample images) to get rid of noise on average.  This was quite better than whitening. We finally applied a log scaling, a bit similar to amplitude to db in audio signal processing.  An example of resulting image is shown below (ignore frequency axis labels):

[image]

Depending on the model we added the mask as an additional channel. We then scaled data to have multiple of 16 dimensions.  For instance resizing 96x820 to 1x96x768

**Model**

We mostly used efficientnet_v2s_s and efficientnet_v2_m from timm package.  Models were trained using pytorch.cuda.amp.  By using batch size and image dimensions that are multiple of 16 we use of RT cores and speed up training significantly on GPU V100.

We used BCEWithLogitsLoss, Adam and OneCycleLR most of the competition. We later found that retraining models was helping. Our best model went through 4 training cycles.  Using cosine annealing from the start would probably have been a good choice.

**Augmentations**

Kazuki shared most of our augmentations here: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275335

We also used a random time shift in addition to these.

**Pseudo Labeling**

Pseudo labeling gave us a 0.0002 boost on LB when we used it.  We used non rounded predictions from teacher model for test dataset and concatenated to training fold.  It improved CV by 0.0008 on average but LB by only 0.0002.  However, our best model was not a pseudo labeling model.

**Ensembling**

We kept oof predictions for all our models and trained second level models on them. We used logistic regression on logits, scipy optimize on ranked predictions, and XGBoost on ranked predictions.  They yield similar results, and averaging them did not help really.  

Ensembling moved us from 8826 for best single model to 8832 on public LB.

**What did not work**

- Giba tried hard to generate additional positive samples.  Issue was to calibrate these to be of the same distribution as training data.  We did not find the right way in time.
- Resnet34.  As written above, we could not meet effnetv2s performance.  And speedup was only 2x anyway.
- We tried many other things that did not work well enough, like triplet models (a backbone on each detector then a common head), or using detectors as RGB channels.

That's it.  We clearly missed 1D model, but for the rest we did everything we could.  And I learned from my team mates!  Teaming is good!.
