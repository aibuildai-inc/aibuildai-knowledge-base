# 18th place solution: DenseNet + RNN based

Competition: prostate-cancer-grade-assessment
Rank: #17
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169178

First of all, thanks to the organizers, Kaggle team and all of the participating kagglers!

That's the first time when I get so high place! I'm really glad that all of my work done in these 3 months is rewarded.
And as usual I learned a lot during this challenge!


My solution is quite different from the concat pooling (by Iafoss) based mainstream approach.

It is:

0) image similarity clustering via image hashing (splitting clusters into tr/val sets, not images)
1) rotation of a whole (middle resolution) image to arbitrary angle with crop preventions
2) extraction of tissue tiles (256x256)
3) Global Contrast Normalization (across all of the extracted tiles from single image)
4) DenseNet121 backbone (imagenet pretrained) -&gt; Dense feature extractor -&gt; 2 GRU layers -&gt; single head ISUP grade regression (logcosh loss)
5) Multiple generations of discarding the "hard or wrong labelled" images by MAE&gt;2.5 threshold
6) 5-Fold CV during training, keeping the gleason_score frequencies balanced while splitting the train and validation.
7) 3 stage training:
     - backbone frozen, long tile sequence 64 tiles
     - all unfrozen, shorter tile sequence of 16 tiles
     - backbone frozen, long tile sequence of 64 tiles again
8) short train batch size of 2 (for regularizing effect)
