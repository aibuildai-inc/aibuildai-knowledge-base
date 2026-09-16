# Step by step (9th place solution)

Competition: severstal-steel-defect-detection
Rank: #9
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114297

It will be a long story (just like I did with [Porto Seguro](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44659#251185)) so grab yourself a coffee :)

## Premature optimization is the root of all evil

What was the goal of this competition? In my opinion the goal was to predict mask for specific image and specific defect type. That's the way data is constructed - one row of data is combination of image and defect type. 

But for some reason (almost) everyone assumed that the goal was to build model which predicts 4 masks from 1 image. You may think it's not really a difference, I will explain why I think it matters.

[Two months ago](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/106099#610085) I wanted to build 4 classification models and 4 segmentation models - one for each class. In this thread @phoenix9032 proposed three Pipeline Strategies. Please look at strategy 2. 

We agree we need some kind of classifier. Why we need it? What should it do? Classifier takes image as input and predicts probability that class has a defect. And what we should do with that information? 

We can use that information after segmentation to say that segmentation was pointless. But in that case what was the reason to start segmentation at all? We have only 1 hour to predict everything.

So we need to use that information before segmentation. When classifier predicts that there is specific defect we can call segmentation, if not - we can skip segmentation and save time. But wait - if your segmentation model is predicting 4 masks at once you don't need information about defect type, because you will call segmentation anyway if at least one defect is possible.

But what if you build single classifier with 4 outputs and then you use separate segmentation model for defect? Then you need to execute segmentation only for the defect type with high probability.

How do you select which model is the best one? The one with best metrics, right? But how do you know which model has best metrics if you average 4 defects? What if I told you that predicting each mask is independent task? When you split competition to 4 different parts you can work on each of them independently - which makes solution much more stable - and then you can even survive the shakeup.

And that's not all. How to handle class inbalance? I read many posts that people can't predict class 1 and class 2. Some people dropped class 2. In my case each class was separate training. On different training data.

You can assume otherwise - that predicting defects is not independent task, there are no pixels with two different defects, so you may say model which predicts all defects at once is smarter. Like softmax with 5 class (I had no time to try that). But what are you doing after predicting mask and before saving submission? You are executing some kind of postprocessing - and then you can remove defect at all if it's too small.  And again this is independent for each class.

## metrics

Soon I realized that [something is wrong with the metrics](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/107182). I wasn't sure, maybe there was some mistake in my thinking - thanks to @zavareh89 for confirmation.

"The leaderboard score is the mean of the Dice coefficients for each ImageId, ClassId pair in the test set." 

It means that score in this competiton was calculated separately for each image and for each defect type. So if ground truth is mask with zero pixels and you will predict zero pixels - you have score 1, but if you predict at least one pixel - your score is 0. It is crucial to filter out empty masks.

Default metrics used in kernels (at least in keras kernels) takes whole batch of images, four masks together and flattens that. Then intersection and union is calculated in array of one dimension. In which case ground truth is mask with zero pixels if you flatten all images and all masks? Only when all 4 masks in all images in batch are empty - which is rare case. This metrics is totally wrong.

## loss

If there is something I learned from TGS Salt competiton it is the lovasz loss.

The problem with binary crossentropy is that it works in the beginning but then the loss is decreasing but the metrics is not increasing. Lovasz loss is much better for segmentation, but it needs good model to start, because the training is slower.

I tested lovasz_hinge and lovasz_softmax. 

So the correct workflow is:
- train with BCE 
- take best model (using metrics not loss!)
- train with lovasz_hinge

It always works. 

## segmentation models

Thanks to @cdeotte for [information how to use segmentaton models](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/103367#latest-649639)

Thanks to @bibek777 for [info about EfficientNet.](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/111110#640168)

And of course thanks to @pavel92 for creating segmentation models :)

In the beggining I was using unet with resnet34. Then I was trying different networks - resnet50, densenet, etc... but performance was lower. Then I tried efficientnet and it finally it was better.

I used resnet34 classifier to the end of the competition, but for segmentation I switched to b0, then b1, then b2.

## PyTorch

Everything was good, I was doing my deep learning and then Kaggle announced new feature - now time of GPU usage is limited to 30 hours per week. You may think 30 hours is plenty of time, truth is I was able to use half of this time during one day. [So I wrote this.](https://www.kaggle.com/general/108481#624663) 

I borrowed GTX 1080 and to my surprise I was quickly able to make everything work locally, so few days later I bought RTX 2070. From that time all my training was performed on my GPU. Funny thing: last competiton when I was using my home computer instead Kaggle kernels was Porto Seguro (the one with gold medal).

I had no issues with speed of my local GPU - it was similar to Kaggle kernel. But [there was another kind of problem.](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/109185). Cheap GPU has smaller memory than the one used by Kaggle. So I wasn't able to fit same batch. And my model started to overfit. Thanks to @bfishh for confirmation. Thanks to @ostamand for info about gradient accumulation in PyTorch. Also thanks to @sandeepsign and @ashwan1 for info about float16.

I started to learn PyTorch. spoiler: I was never able to try float16 and gradient accumulation didn't work correctly. But I realized PyTorch is much much better than TensorFlow/Keras, I had much more control over everything. I know guys in Google are smart and that's why they designed TensorFlow 2.0 but I had no time to check it, so I use PyTorch.

Funny thing: I could fit larger batch in PyTorch than in Keras for same architecture. Maybe it is related to framework itself, maybe it is related to segmentation models implementation.

## classifier and class inbalance

On first October I trained 4 folds resnet34 classifier. As I wrote before classifier is used to skip segmentation for some images. But in my case it is also used to handle class inbalance.

When classifier predicts that image may have defect this image is later processed by segmentation model. Let me rephrase this sentence - when classifier predicts that image doesn't have defect - it is not processed by segmentation model. We can conclude from that something about distribution of data used in segmentation model.

For each class:
- take all images with mask
- add all images with high prediction of classifier
- train segmentation only on this data

I started from 0.5 threshold, final threshold was 0.01.
This is my final trainset for each class:
(targets means ground truth mask, outputs means classifier &gt; threshold, all is union of them)

```
class 1
all 2405 / 12568
outputs 2377 / 12568
targets 897 / 12568
all-targets 1508 / 12568
class 2
all 949 / 12568
outputs 941 / 12568
targets 247 / 12568
all-targets 702 / 12568
class 3
all 8398 / 12568
outputs 8381 / 12568
targets 5150 / 12568
all-targets 3248 / 12568
class 4
all 1603 / 12568
outputs 1591 / 12568
targets 801 / 12568
all-targets 802 / 12568
```

## final training workflow

In the EfficientNet thread @hengck23 wrote following sentence:

"My efficientnet b3 and b5 with fpn gives good results. Input size is crop of 256x400. You may want to use this as reference"

But for my crop was always worse than full size. You can read it [here](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/111110#648254). 

And then @phoenix9032 wrote following sentence:

"Yes Chris . With qubvel models I can train with 256x256, and then predict with 256x1600. Or even run 256x256 for 30 epochs , load the model and finetune with 256x512 by passing that as input size and then finally finetune stage 3 with 256x1600 as well . "

Theoreticaly there is no new information here. I was also using crop for training and full size for prediction. But finetuning on different sizes? Why not? So I tested it. And there was no improvement.

The main problem is overfitting. First model generalizes but at some point it overfits to training data. By using augmentation we can push it more, but then it overfits to all possible augmentatons and there is no longer improvement on validation.

But why when I train on 256x416 and then I train on 256x800 it needs time to overfit? If the model is already dumb it should stay dumb with 256x800 too. Instead it looks at larger input data and learn new stuff. Validation metrics was not increasing, but...

What if I use different crop sizes not to finetune but as an augmentation?

The problem was that augmentation was in DataGenerator which produces single image and for training I needed whole batch of same resolution. So I could look into PyTorch docs because it's probably very easy to implement custom DataLoader but it was October 17th and I really had no time to fix potential bugs in my code.

I was already using two DataGenerators: one for the training and one for validation. Why not create 4 DataGenerators and 4 DataLoaders? (PyTorch FTW!)

My final training workflow is:
- train on 256x256 crops (all images from train dataset) batch_size = 12
- train on 416x256 crops (all images from train dataset) batch_size = 12
- train on 608x256 crops (all images from train dataset) batch_size = 8
- validate on full size (all images from valid dataset) batch_size = 4
- call it an epoch

I verified that it works in each class - always result is better than full size training.

It's important to decrease lr:
```
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, eps=1e-09, verbose=True)
```

For BCE stage:
```
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
```

And for lovasz stage:
```
optimizer = optim.Adam(model.parameters(), lr = 1e-4)
```

## final days

After October 18th I used 24h per day only to train my final models.
Last one was trained few hours before competition deadline.
Without it I would be probably in the silver zone.

My top two submissions from public are also my top two submission from private. I believe it is because my solution is stable.

and now some details from my winning kernel:

### classification - 4 folds resnet34 (BCE loss in name)

```
20191003-classifier-class-2/model_class_a_56_0.072748.pth
20191003-classifier-class-2/model_class_a_46_0.074594.pth
20191003-classifier-class-2/model_class_a_51_0.065235.pth
20191003-classifier-class-2/model_class_a_37_0.063228.pth
```

### segmentation - 5 folds FPN efficientnet-b2 (my metrics in name)

```
20191019-class-3/model_lovasz_class_3_fold_0_c_25_0.787906.pth
20191020-class-3-fold-1/model_lovasz_class_3_fold_1_c_38_0.800861.pth
20191021-class-3-fold-2/model_lovasz_class_3_fold_2_c_30_0.805968.pth
20191022-class-3/model_lovasz_class_3_fold_3_c_28_0.809485.pth
20191024-class-3-fold-4/model_lovasz_class_3_fold_4_c_28_0.817969.pth

20191024-class-1/model_lovasz_class_1_fold_0_d_1_0.829611.pth
20191024-class-1/model_lovasz_class_1_fold_1_d_1_0.831025.pth
20191024-class-1/model_lovasz_class_1_fold_2_e_1_0.836173.pth
20191024-class-1/model_lovasz_class_1_fold_3_c_30_0.867282.pth
20191024-class-1/model_lovasz_class_1_fold_4_c_20_0.833884.pth

20191020-class-2/model_lovasz_class_2_fold_0_c_7_0.877366.pth

20191023-class-4/model_lovasz_class_4_fold_1_d_21_0.869015.pth
20191023-class-4/model_lovasz_class_4_fold_0_e_3_0.879779.pth
20191023-class-4/model_lovasz_class_4_fold_3_d_1_0.884751.pth
20191023-class-4/model_lovasz_class_4_fold_4_d_2_0.901342.pth
20191023-class-4/model_lovasz_class_4_fold_2_e_9_0.849709.pth
```

### classification thresholds

```
class 1: t = 0.3
class 2: t = 0.3
class 3: t = 0.25
class 4: t = 0.1
```

### classification results

```
class  1
idx_outputs_positive  191
class  2
idx_outputs_positive  71
class  3
idx_outputs_positive  781
class  4
idx_outputs_positive  176
```

### segmentation thresholds

for each class: min mask size = 1000
pixel thresholds:

```
class 1: t = 0.55
class 2: t = 0.7
class 3: t = 0.53
class 4: t = 0.5
```

### segmentation results

```
class 1: found masks:  92
class 2: found masks:  9
class 3: found masks:  586
class 4: found masks:  113
```

## summary

- Kaggle is still the best place on the planet
- to me PyTorch is better than Tensorflow 1.x / Keras
- using segmentation models was very good idea
- reading everything on forum is extremely important
- I had 3 months for this competition and I trained everything in the last week

Thanks Severstal for an awesome competition.
Thanks everyone for great fun!
See you in another competition, hopefuly soon, not in another 2 years :)
