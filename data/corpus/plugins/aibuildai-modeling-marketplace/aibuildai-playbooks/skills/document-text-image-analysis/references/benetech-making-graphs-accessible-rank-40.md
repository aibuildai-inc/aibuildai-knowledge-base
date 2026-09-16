# 40th place solution + code

Competition: benetech-making-graphs-accessible
Rank: #40
Source: https://www.kaggle.com/c/benetech-making-graphs-accessible/discussion/418331

# Code

I hope you find the training and inference codes neat and tidy, I did my best effort to not make a mess!
- [Matcha Training Code](https://www.kaggle.com/code/alejopaullier/benetech-matcha-train-0-74)
- [Matcha Inference Code](https://www.kaggle.com/code/alejopaullier/benetech-matcha-inference-0-74)

# TL;DR

Here is a brief explanation of our team solution with @cody11null.

Our solution consisted of a pipeline of object detectors (EfficientDet) and image-encoders/text-decoders (Matcha and Donut):
1. A Donut model which acts like an OCR (Object Character Recognition) model is used to extract the axis labels of the chart (red)
2. An object detector (EfficientDet) detects the data series (green). In the end, we only used this for scatter plots.
3. A second object detector (EfficientDet) detects the plot's chart area (area where the data points are)(purple).
4. Knowing the chart's limits from (3), the bounding boxes of the data series from (2) and the axis numerical values from (1) you can with a simple cross multiplication get the data series values. This approach is essentially as described in my [discussion](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/396773).
5. For the other chart types (`vertical_bar`, `horizontal_bar`, `line` and `dot`) we used a Matcha model.

</img>


# Solution in detail

### 1. Donut: Axis labels detection

We train different image-encoder/text-decoder architectures to detect the axis labels. I trained a Donut model (which surprisingly got the best results) and two other Matcha models which got worse performance. Despite training with different `max_patches` results were rather similar for all. Most complications were associated to shared origins in the axis labels, long sequences, floats with many decimal places, etc. In this step, the most important is to get the minimum and the maximum values right for each axis so the cross multiplication is done right.

### 2. EfficientDet: Object Detector for Data Series

This was one of the most time consuming parts for me. At the beggining of the competition until Nicholas shared his Donut model I didn't know that image-encoder/text-decoder models could perform that well in this competition so I focused on an object detector model which could detect well data series. To do so I lack bounding boxes training data, so I started manually labelling thousands of images. These took me a lot of days. My EfficientDet object detector did extremely well on generated images but not as well on extracted. I labeled as many images as I could. My OD has ~50% exact matches accuracy for scatter plots. It performs much better for other chart types, but since scatter plots can have *a lot* of points it's reasonable to be a tougher task.

Here you can see some of my ODs predictions:

</img>

### 3. EfficientDet: Object Detector for Chart bounding box

As part of the pipeline I had to detect the chart's bounding box so I could map pixel coordinates to real numerical values from the axis labels. This step was rather easy and its basically the same code from part (2) applied on the bounding boxes of `plot-bb` from the annotations. We had this training data available inside the JSONs files and it's a simple task so the OD achieves high accuracy.

### 4. Matcha and Donut models

At the beggining of the competition I started making several changes to Nicholas' Donut model and soon discovered that you could achieve higher performance by perfoming some basic postprocessing. However, even after training more epochs, perfoming data augmentation and other tricks I realised I couldn't achieve much higher performances. I then tried implementing a Matcha model and with a lot of effort I made it work thanks (again) to Nicholas which raised the issue of the GitHub discussion. Once I got the Matcha model working I tried a ton of different stuff until I could squeeze as much performance as I could. So here is a little roadmap to achieveing 0.74 in the public LB:
- **0.20:** a model with only Object Detection and Donut for axis labels.
- **0.47:** Nicholas model but just modifying post processing, like using max length of xy series instead of the min.
- **0.48:** doing additional post processing like filling values with average of the data series mean values.
- **0.49:** using a mixture of Donut and the Object Detector.
- **0.50**: training Donut for 10 epochs instead of 5.
- **0.56:** BOOM! big increase by making the Matcha model work. Super vanilla, no fancy stuff.
- **0.61:** combining vanilla Matcha with my Object Detector for scatter plots.
- **0.64:** training Matcha for more epochs (10 epochs).
- **0.69:** training Matcha with Bartley's generated images from code. These improvement increase the performance a lot of all chart types except scatter with a relatively low number of additional images per chart type (+5k per image).
- **0.71:** increase `max_patches` of Matcha from 512 to 1024.
- **0.74:** trained Matcha on 100% of the extracted images instead of the 75% I usually used so I could get a 25% holdout set for validation.


# Things that didn't work

So many things that didn't work! I will write the ones I can remember:
1. I tried training at an early stage **one different image-encoder/text-decoder model per each chart type** instead of training a general model. I believe that this didnt work for two reasons: 1. I think I trained a donut model for this 2. I didnt have enough images (at that point I didnt use Bartley generated images).
2. I tried **different pre-trained weights** for the Matcha model and ALL got worse results. These were: `statista`, `chartqa`, `plotqa`. Best performance came from `matcha-base`.
3. Tried training **two Matcha models for each axis**. I really thought this idea could work and until today I don't know why it didnt! My reasoning is the following: it's harder to predict a longer sequence, so why don't I train a model to predict x-axis series and another for y-axis? In the end it gave worse results than a combined model.
4. Tried **training with A LOT of generated images**. I tried training with additional 5k, 10k and 25k more images per chart type. So essentially I trained the model with the competitions 60k plus the additional generated images. So I trained some models with a total number of 180k images! What I soon realised was that if there is not that much variance in additional generated images no matter how many images you add the model will have reached its capacity. I discussed this [here](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/415470#2294381), which relates to the Law of Diminishing returns.
5. Trained with **Balance Sampler/Oversampling**. I trained with a Balance Sampler, which guarantees that on every batch, at least one minority class image will be present. My minority class images are the extracted images. This has two effects: it oversamples the minority class and (in theory) makes the model converge faster. I didn't see any benefit from oversampling (no surprise here).
6. **Data augmentation**. I tried augmenting data mostly by applying color-related augmentations like `RGBShift`, `RandomBrightness`, `ColorJitter`, etc. 
7. **Increased the `max_patches` parameter** to 1536 (halfway between 1024 and 2048). So, as I saw an improvement by increasing `max_patches` from 512 to 1024 I thought, why don't I increase it a bit more and see what happens? Well, it didn't seem to improve the score and of course consumed more VRAM and computation time. Increasing `max_patches` increases a lot the VRAM consumption so the batch size had to be lowered and training took longer.
8. Trained with **different schedulers**. I ended up training with `OneCycleLR` but before I tried some other, even constant LR, and never saw an increase in performance by modifying the LR scheduler.
9. For the Object Detector I tried using **Weighted Boxes Fusion** to increase the exact matches of the data series. One of the greatest challenges of this competition is to get the exact number of data series points right. My OD sometimes produced an excess of low confident bounding boxes. I thought that maybe if I could fuse them with higher confidence bounding boxes I could get higher accuracies but couldn't quite make it. I ended up tunning the probability threshold for keeping/discarding the bounding boxes. In the end, bounding boxes with p>0.22 where kept. Would be glad to know how anyone did this!
10. Different Object Detector backbones. I used `tf_efficientnetv2_s` as my OD's backbone, but tried larger models without match success like `tf_efficientnetv2_l`.
11. **Automatic Mix Precision**. Not that it didn't work but didn't provide any performance by training with mixed precision using brain floating point tensors (`bfloat16`). However, since you use less VRAM compared to `float32` you can train with larger batch sizes which is a recommended practice (I heard that Karpathy said that it's always the best to train with as larger batches as you can, I may be wrong here).
12. Many other things that I don't remember right now.


# Hardware 

It's barely impossible to train models like Matcha and Donut with the P100's provided by Kaggle. We ended up paying Google Colab Pro+ which provides A100 GPUs with 40 GB of VRAM since neither of us had DL GPUs like an RTX 3090/4090. With the A100 we were able to train large models and run many experiments faster. I trained more than 30 different models!

# Conclusions

Even though I would have liked to get inside the gold zone (always hoping for the best) I am grateful for the result we obtained and mainly for all the lessons learned throughout the competition. I firmly believe that if you want to learn something try changing it! Code something from scratch! Best way to learn by far. I am really looking forward to read other teams solutions. I would also like to know if anyone tried changing the Vision Model and the Text Model from Pix2Struct. I don't know if it's even possible or how its done so if you know please leave in the comments.

Thanks all!
