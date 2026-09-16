# 4th place solution: Stain Normalization is all you need

Competition: hubmap-organ-segmentation
Rank: #4
Source: https://www.kaggle.com/c/hubmap-organ-segmentation/discussion/354851

First, we'll thank organizers for their effort in this amazing competation. And then, we will thank @hengck23 , we learnt a lot from your code and notebook.

This is my second competation in kaggle, and my first competation on image segmentation, I learnt a lot from this, thanks all of you.

In this competation, we mainly rely on two-stream model and some tricks on models ensemble and stain normalization at infernce, without using any pseudo label or ext data, maybe that's why we missed prize hahaha :)

## Models

At first, we followed the code share by @hengck23 that's an amazing baseline code. Besides, in our experience, we found that the knowledge learned from other organs may influence the performance in lung images, so we proposed two-stream model, one stream for lung images and the other model for other organs. And then we tried some other encoders and decoders, and the final ensemble models are shown as follows:

- Encoder: Coat-lite-medium; Decoder: daformer+unet
- Encoder: mit (Segformer); Decoder: daformer+unet
- Encoder: mpvit; Decoder: daformer+unet

## Stain Normalization

That's the most import part in our training and inference step, for we think the biggest challenge in this competation is how to bridge the gap between HPA and HuBMAP images. The methods we used for stain normalization are from Staintools.

### Train

First, in training step, we normalize the HPA image to the only one HuBMAP test image by using reinhard and vahadane normalization randomly. So that the model can study both features in HPA feature space and HuBMAP feature space.

### Inference

However, just using stain normalization in training step is not enough, for stain normalization still can't transfer feature space perfectly, we also need using stain normalization in inference step for better bridging the gap between two datasets.

In inference step, we select a representive image from training images for five organs respectively. And for all of HuBMAP images, we normalized them to HPA images according to their organ labels by using vahadane and reinhard methods, and then average their predictions to get the final predicted mask.

The final codes will be released after a few days, thanks for reading!
