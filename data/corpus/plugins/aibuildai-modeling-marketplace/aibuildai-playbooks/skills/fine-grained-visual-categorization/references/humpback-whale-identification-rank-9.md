# 9th place solution or how we spent last one and a half month

Competition: humpback-whale-identification
Rank: #9
Source: https://www.kaggle.com/c/humpback-whale-identification/discussion/82427#latest-505626

**TL;DR** Adam, Cosine with restarts, CosFace, ArcFace, High-resolution images, Weighted sampling, new_whale distillation, Pseudo labeled test, Resnet34, BNInception, Densenet121, AutoAugment, CoordConv, GAPNet 

We’d like to share our solution as a story how we gradually improve our models. 

But first of all I’d like to thank my teammate [Vladislav][1] for fruitful collaboration, kaggle community for motivation and ods.ai for kind support:)

To start with, it was obvious idea to consider whale’s flukes as human faces. Fortunately there are tons of papers for face identification, re-identification and verification. 

So in the beginning of this competition this paper https://arxiv.org/abs/1804.06655 helped us a lot. It features comprehensive survey of state-of-the-art face identification techniques. 
According to it softmax-based losses look really promising. Due to their classification nature and the fact that we already have classification pipeline from Protein Atlas and Draw challenges we decided to focus on them. 

Among others **Cosface** and **Arcface** stand out as newly discovered SOTA for face recognition task. The main idea is to bring examples of the same class close to each other in cosine similarity space and to pull apart distinct classes. Training with cosface or arcface generally is classification, so the final loss was CrossEntropy. One can read more details in their papers: https://arxiv.org/abs/1801.09414, https://arxiv.org/abs/1801.07698. After we train cosface or arcface net we took embeddings and calculate cosine similarity between train and test images. Then average similarities for each class in train and took 5 most similar.

In the beginning of every competition you should always devise robust validation procedure. We did it poorly. But nevertheless, to do so we select about 1000 sample from classes with number of instances greater than 3, one example for each class. Also we chose about the same number of new_whale images. This setup show good correlation between local score and public LB score. Unfortunately threshold for new_whale derived from local validation was slightly biased. That was really bad, because threshold was unreliable. Another way for threshold determination was to adjust it so top-1 new_whale percentage should be around 30%. 

To keep up with kaggle community in conquering LB we decided to construct our pipeline in the following way:

 1. Decrease training time of model as much as possible
 2. Test a lot of hypothesis as much as possible

To do that we restrict image size to 256x256 and number of epochs up to 64. This kind of restrictions gave us a model that can be trained in 2 hours on 1080ti or even faster on 2080ti. This setting let us iterate quickly in testing new hypothesis or optimizing hyperparameters. After we have established our general training we start endless array of experiments for low-res images. 

Let’s divide our experiments into two broad groups:

 1. Model engineering: what to train
 2. Training engineering: how to train

Here comes Model engineering. We started with some heavy encoders such as inceptionv4, seresnext50 etc. But it appeared that for us in classification task they seem to overfit a lot. Then we decided to main some light networks such as resnet34, bninception and densenet121. After several competitions I begin to realise that sometimes when you don’t have much data light encoders may really boost your score. Like they don’t tend to overfit much to rare classes and label noise. This is just a hypothesis that need to be carefully verified.

To get final models after initial 64 epochs on 256x256 images we increase image size up to 1024 for resnet34, up to 512 for bninception and up to 640 for densenet121 and train for 64 epochs more. 

To boost model performant we tried a lot of modification. According to our findings **CoordConv** https://arxiv.org/abs/1807.03247 and **GapNet** architecture https://openreview.net/forum?id=ryl5khRcKm helped to improve resnet34 score. Unfortunately we didn’t have time to test this mods on bninception and densenet121. Also adding some sophisticated convolution blocks to our nets didn’t help. So Squeeze-and-Excitation, Convolutional Block Attention Module didn’t help. That was a sad story because a lot of time was spent on trying to optimize model architecture instead of optimizing training itself. 

When it comes to training one of the first things that comes to mind is how not to overfit to training data. Especially when one working with zero and few-shot learning. Inspired by AutoAugment paper https://arxiv.org/abs/1805.09501 we search augmentation space by random sampling and came up with the following augmentations:

 1. HorizontalFlip
 2. Rotate with 16 degree limit
 3. ShiftScaleRotate with 16 degree limit
 4. RandomBrightnessContrast
 5. RandomGamma
 6. Blur
 7. Perspective transform: tile left, right and corner
 8. Shear
 9. MotionBlur
 10. GridDistortion
 11. ElasticTransform
 12. Cutout

Those augmentations we took from albumentations https://github.com/albu/albumentations and Augmentor https://github.com/mdbloice/Augmentor modules. Firstly we thought that this is to much for our networks. But actually for our models it was essential part not to overfit to training data.

Cosface and Arcface parameters was optimised as well. Cosface: S = 32.0, M=0.35. Arcface: M1 = 1.0, M2 = 0.4, M3 = 0.15.

We experimented a lot with optimizers and their hyperparameters: Adam, AdamW, SGD, SGDW. But the best optimizer for us appeared to be good old **Adam with Cosine annealing**.

In the end we tried different kinds of TTA, but it didn’t help to improve the score. 
Mixed precision learning didn’t show good results either. 

Starting from the beginning we realised that it is essential to do something with new_whales in order to incorporate them into training process. Simple solution was to assign each new_whale probability of each class as 1 / 5004. With help of weighted sampling technique it gave us some boost. But then we realised why don’t we use softmax predictions for new_whales derived from trained ensemble. So we came up with **distillation**. We choose distillation instead of pseudo labels, because new_whale is considered to have different labels from train labels. Though it might not to be really true. 

To further boost model capabilities we add test images with **pseudo labels** into train. Eventually our single model can hit 0.958 with snapshot ensembling. Unfortunately ensembling of models trained in this way didn’t give score improvement. Maybe it was due to less variety because of pseudo labels and distillation. 

In the end I should mention that this competition was really interesting and give us an opportunity to develop face/fluke recognition skills. Thank your Kaggle!
	
	


  [1]: https://www.kaggle.com/vlad0922
