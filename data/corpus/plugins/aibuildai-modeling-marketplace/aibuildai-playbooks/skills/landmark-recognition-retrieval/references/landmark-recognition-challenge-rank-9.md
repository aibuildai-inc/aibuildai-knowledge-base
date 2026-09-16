# [Source code shared] Single ResNet50 LB0.1+

Competition: landmark-recognition-challenge
Rank: #9
Source: https://www.kaggle.com/c/landmark-recognition-challenge/discussion/57152

**[UPDATE] The missing file world.py has been uploaded.**

**[UPDATE] I decide to share my code. Anyone who is interested can take a look at the attachment. Don't laugh at my messy code. :)**


Recently on the forum I mentioned that I achieved LB 0.1+ using a single ResNet50. Some people showed their interest about how I did it. Here let me share the experimental configuration about one of my model:

**LB:** 
0.111

**Architecture:** 
ResNet50

**Input size:** 
224x224

**Data augmentation:** 
Resize the original images to 256x256; 
Crop at random position;
Randomly horizontally flip it;
Demean and normalize it.

**Batch size:**
32

**Initial weights:**
Pretrained on ImageNet

**Initial learning rate:**
1e-4

**Learning rate decay:**
Learning rate is halved at Epoch 5 and halved again at Epoch 7

**Max Epochs:**
8

I used 1/8 training images for validation.
At test stage, center-crop is used instead of random-crop.
