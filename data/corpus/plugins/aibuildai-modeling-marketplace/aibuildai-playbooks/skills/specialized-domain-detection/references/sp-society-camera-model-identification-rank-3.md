# 3rd place solution

Competition: sp-society-camera-model-identification
Rank: #3
Source: https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/49602

**Data**

The training set has been augmented with both private images and images downloaded from Flickr. 
All images were preliminary sifted based on the EXIF header meta-data information,
to keep only images with meta-data coherent with the corresponding camera of the training set. 
All images that were processed by an editing software or with restrictive license were removed. 

Finally we obtained a dataset of 450 images per model (about 15 GB). 400 images have been used for training, 50 images for validation.

**Training**

During training patches were randomly extracted from the central part of image (1024x1024 pixels) and randomly rotated with steps of 90 degrees. 

We trained separate networks for the model identification of unaltered and manipulated images. 
For the first problem, we included only unaltered images in the training set. 
On the contrary, the networks for manipulated images were trained using both unaltered and manipulated images, as we found this solution to provide consistently better results.

**Solution overview**

Fusion of different CNNs models (arithmetic mean of the scores). 

For unaltered images we considered 5 networks trained at different patch size: XceptionNet (96), XceptionNet (299), Inception v3 (139), InceptionResNet v2 (299), DenseNet121 (224). 

For manipulated images we considered the same networks above but used various input patch sizes, for a total of 14 models. 
We tested many alternative solutions to improve performance, two of them were eventually applied: 1) network training on small patches and fine-tuning on larger ones, 2) identification of problematic classes (JPEG 70 and Resizing 0.5) and design of dedicated detectors.

**Hardware**

NVIDIA Tesla P100 GPU with 16GB of RAM.

**Special mention** 

It is interesting that using only a single model for unaltered (XceptionNet on patches of dimension 96) and manipulated images (XceptionNet on patches of dimension 96 fine tuned on 299) gave a public score = 0.986666 and a private score = 0.981309. 

**Lessons learnt**

Do not trust the public LB too much. Use a better validation set!

This was a great experience for us (our first time in a Kaggle competition) and we really learnt a lot. Many thanks to the organization and to all the teams sharing their work.
