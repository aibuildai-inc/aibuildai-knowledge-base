# Which approach works best?

Competition: imaterialist-fashion-2019-FGVC6
Rank: #3
Source: https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/discussion/91536#latest-547636

I'm using Mask-RCNN so far (https://github.com/facebookresearch/maskrcnn-benchmark) and it seems to work OK, and the code is quite easy to modify. Looks like current top-2 are also using Mask-RCNN?
Another possible approach would be to do segmentation with something UNet-alike, and segment into instances (should be easy here). And maybe also have a separate classification step to determine attributes.
Main difference is that for Mask-RCNN, segmentation and classification are more explicitly separated, while for pure segmentation tasks, they are performed jointly. Although it seems that Mask-RCNN is likely to have worse quality masks, due to fixed per-object mask size and resampling.
Also attributes seem really hard to predict to me so far.

What do you think?
