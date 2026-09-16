# 4th place solution

Competition: sp-society-camera-model-identification
Rank: #4
Source: https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/49298

Here is a brief summary of my steps.

Edit: I only used the central 80% crop of the train data because the boundaries are often statistically very different from the test data. For example, if the original image size is 1000x1000, only the central 800x800 crop is used. This center-cropping applies to train data only, and it gave around 1% higher accuracy than training on the original size.

 1. Finetune a pretrained inception_v3 with random 480x480 crops. The provided training set and Gleb's data were used. My data augmentation include the eight possible manipulations but no transpose, rotation or flipping as I believe they should not help in theory. JPEG compression is always aligned (the 8x8 grid) as I bet re-compressions were done before cropping. This achieved Public LB 0.976 and Private LB 0.972. 

 2. Predict the test set ('unalt' images only) using the finetuned model. Use the predicted probabilities as pseudo-labels for test data and merge the test data with the training set.  Continue tuning with the merged set. After the pseudo-labeling, the performance improved to Public LB 0.983 and Private LB 0.976.

 3. Group the 'unalt' images in test set by predicted labels and estimate the sensor noise patterns for each camera in test set (totally ten reference patterns). Then match each of the 'unalt' images with the ten reference patterns, and correct the predictions when the correlation between an image and a reference pattern is larger than a certain threshold. I also corrected the 'manip' part by matching their sensor noises with the augmented (by the eight manipulations) reference patterns. The last step gave the largest boost: Public LB 0.986 and Private LB 0.987.

Thanks to Kaggle and IEEE SPS for hosting this interesting competition.
Thanks to everyone who generously shared their data and ideas.
