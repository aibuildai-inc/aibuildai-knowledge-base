# 2nd place solution

Competition: blood-vessel-segmentation
Rank: #2
Source: https://www.kaggle.com/c/blood-vessel-segmentation/discussion/475657

What's happened? My name is written at 2nd place? I remember my public score was 0.43 and my place was 1052...

# Overview

My solution consists of U-Net3D (128x128x32), threshold adjustment, and <strong>post-processing</strong> to remove unconnected vessels as they are false positives.

* data augmentation using random rotation (and position), same as 1st place solution.
* U-Net3D. I assumed that 3D would be more accurate because it provides more information. I think this assumption might be wrong, since the 1st place solution uses 2.5D, .
* Binary-focal loss. Since there are a few positive data.
* Adjusting threshold. Since the volume ratio of blood vessels are not so different betwwen persons, the threshold is set according to the ratio.
* <strong>Post-processing</strong>. Since blood vessels are supposed to be connected, extract small chunks with depth-first-search and remove them.

https://github.com/tail-island/blood-vessel-segmentation
https://www.kaggle.com/code/ojimaryoji/sennet-hoa-2nd-place-solution?scriptVersionId=159388443

# Data

To make it easier cutting out the data, I created a 3D Numpy array and adjusted the scale. I created *all* and *dense* data because I was planning to do curriculum learning in the order of *all* to *dense*. However, since it took a long time to learn in my PC, I only trained on sparse data this time. Also, I did not normalize or clipping the data because I thought there should not be a big difference since the data is visible to the human eye.

https://github.com/tail-island/blood-vessel-segmentation/blob/main/src/create_volumetric_images.py

# Train

I generate data from random positions and rotations in each *n* epochs. To reduce data generation time, I used multiple processes.

https://github.com/tail-island/blood-vessel-segmentation/blob/main/src/dataset.py

The neural network is U-Net3D.

https://github.com/tail-island/blood-vessel-segmentation/blob/main/src/model.py

I used binary-focal loss. Optimizer is AdamW and learning rate is scheduled by cosine-decay.

https://github.com/tail-island/blood-vessel-segmentation/blob/main/src/train_0.py

# Submit

Prediction is made by tiling. get_candidate() finds candidates with a given ratio and <strong>get_blood_vessels()</strong> removes small unconnected chunks.

https://github.com/tail-island/blood-vessel-segmentation/blob/main/src/submit.py

Searching the big blood vessel chunk (and clip) version, private score is 0.756793 and public score is <strong>0.000000</strong>...

https://github.com/tail-island/blood-vessel-segmentation/blob/main/src/submit_.py

Other scores...



# What's happend?

What's happened? My name is written at 2nd place? I remember my public score was 0.43 and my place was 1052...

# Overview

My solution consists of U-Net3D (128x128x32), threshold adjustment, and <strong>post-processing</strong> to remove unconnected vessels as they are false positives.

* data augmentation using random rotation (and position), same as 1st place solution.
* U-Net3D. I assumed that 3D would be more accurate because it provides more information. I think this assumption might be wrong, since the 1st place solution uses 2.5D, .
* Binary-focal loss. Since there are a few positive data.
* Adjusting threshold. Since the volume ratio of blood vessels are not so different betwwen persons, the threshold is set according to the ratio.
* <strong>Post-processing</strong>. Since blood vessels are supposed to be connected, extract small chunks with depth-first-search and remove them.

https://github.com/tail-island/blood-vessel-segmentation

# Data

To make it easier cutting out the data, I created a 3D Numpy array and adjusted the scale. I created *all* and *dense* data because I was planning to do curriculum learning in the order of *all* to *dense*. However, since it took a long time to learn in my PC, I only trained on sparse data this time. Also, I did not normalize or clipping the data because I thought there should not be a big difference since the data is visible to the human eye.

https://github.com/tail-island/blood-vessel-segmentation/blob/main/src/create_volumetric_images.py

# Train

I generate data from random positions and rotations in each *n* epochs. To reduce data generation time, I used multiple processes.

https://github.com/tail-island/blood-vessel-segmentation/blob/main/src/dataset.py

The neural network is U-Net3D.

https://github.com/tail-island/blood-vessel-segmentation/blob/main/src/model.py

I used binary-focal loss. Optimizer is AdamW and learning rate is scheduled by cosine-decay.

https://github.com/tail-island/blood-vessel-segmentation/blob/main/src/train_0.py

# Submit

Prediction is made by tiling. get_candidate() finds candidates with a given ratio and <strong>get_blood_vessels()</strong> removes small unconnected chunks.

https://github.com/tail-island/blood-vessel-segmentation/blob/main/src/submit.py

Searching the big blood vessel chunk (and clip) version, private score is 0.756793 and public score is <strong>0.000000</strong>...

https://github.com/tail-island/blood-vessel-segmentation/blob/main/src/submit_.py

Other scores...



# What's happend?

~~Maybe public data is for the first *x*% of the images, I think. It contains only the end part of the blood vessels; the root part is not included. So, it seems that my post-processing would have resulted in a lower score.~~



~~I gave up to improve my program quite early because my public score was too low. And some other Kagglers continued to improve their programs for the end part of the blood vessels in public data. I suspect that their improvements were not very effective in the private data that contains the root of the blood vessels. In other words, my 2nd place would be caused by LUCK...~~

What's happened?
