# 31st place solution + source code

Competition: humpback-whale-identification
Rank: #31
Source: https://www.kaggle.com/c/humpback-whale-identification/discussion/82393#latest-482789

Source code: https://github.com/suicao/Siamese-Whale-Identification . I'll try to update this repo later but it should be simple enough to follow for now.

Our approach was simple. We took the amazing solution by @martinpiotte and added a few twists:

- Using RGB instead of grayscale images.
- Changing the feature extraction CNN with Imagenet trained models. My teammate @iafoss was able to achieve 0.937 single model with a DenseNet121 encoder. At the last few weeks he also noticed a severe bug in my code where I froze the branch model instead of the feature extractor in the first few epochs, which helped boosting the score.
- Simply training on bigger image size worked, but we didn't have the resources needed to try anything bigger than 512x512.
- Adding TTA made the results worse, we haven't got time to investigate this just yet.

That's it, at the end I made an ensemble of a few high scoring models trained by my teammates using [this method][1].  Big thanks to @matthewa313 .

I didn't even have any accesses to GPUs for the majority of this competition, and one of my other teammate couldn't compete either due to hardware problems, so this is quite unfortunate for us. 

Anyway I'm happy with the final standings, congrats everyone, I wish you do *whale* in the future competitions.


  [1]: https://www.kaggle.com/matthewa313/ensembling-algorithm-for-average-precision-metric
