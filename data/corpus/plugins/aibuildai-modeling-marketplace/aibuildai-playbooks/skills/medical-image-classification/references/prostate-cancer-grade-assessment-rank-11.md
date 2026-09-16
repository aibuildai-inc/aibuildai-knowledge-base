# [11th place solution] I have survived in this storm

Competition: prostate-cancer-grade-assessment
Rank: #11
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169205

## Summary
- Tile extraction is based on my [public pipeline](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/146855) with **128x128x128** tiles from intermediate resolution layer
- Label nose removal gives **~0.005 public and 0.01+ private LB boost**
- tile cutout + tile selection augmentations
- [kappa loss](https://arxiv.org/pdf/1509.07107v2.pdf)
- majority voting ensemble of 8 **ResNeXt50** based models (**0.917 public and 0.930 private LB**)
- more advanced tile selection could give **~0.004 boost** at private LB on average (and the maximum private LB score of **0.941**)


## Introduction

To begin with, I would really like to express my gratitude to organizes and kaggle team for making this competition possible. It was really enjoying working on it and learned many new things. By sharing some of my ideas in this competition, such as [tile pooling base pipeline](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/146855) used by many participants, gave me 3 kernel gold medals, so I have reached the kernel grand-master rank. And I have received my first solo competition gold medal. Also, I would like to say congratulations to all winners and people who received medals.

However, this day is quite sad for many participants, especially ones who worked very hard throughout the entire competition and got down at private LB. The **chose of the metric by organizers could be done more wisely**: 500+500 test set is definitely not enough for QWK. It is not really normal when LB score is changing by 0.005+ when a different seed is used. The things got deteriorated when the third digit became available for LB score: many people got seduced by overfitting LB noise.

Below I outline the main things that worked for me. I have tried many more, but most of them have never worked, and I couldn't get any further improvement of my LB during the last month.

## Main challenges

This competition to a large extent was about dealing with noisy data and train/test bias: as reported by organizers, the Redbound train data has only about **0.853** QWK, and I expect that Karolinska train data has 0.95-0.96 QWK. Beyond this, since Redbound data is graded by students, and Karolinska data is graded by only a single expert, while the test data is graded by 3 experts, there could be train/test bias because of the subjective opinion of people performing grading the train set. Therefore, **solely relaying on CV was not really good strategy in this competition**: at some point I saw a consistent decrease (~10 different models) of LB score when I ran training for longer, while CV was increasing. It confirms the hypothesis about the bias, and the trick was to train models only for limited number of epochs (even if CV could be increased), 32-48 depending on the setup, to **prevent learning the bias**.

Meanwhile, LB was also not the best thing to trust because of severe noise, but some ppl tried to fit random seed as a hyperparameter 😄. The right thing, in my opinion, in this competition was to find the balance between CV and LB, and **trust to your intuition and the experience gained in the previous competitions**.

## Noise

It is the most important part of this competition, in my opinion. After organizers have disclosed that there is a substantial level of noise, especially in Redbound train data, I have explored a number of techniques to deal with the noise: progressive label distillation, JoCoR (Joint Training with Co-Regularization), Co-teaching, negative learning, excluding hard examples from the batch, etc. However, most of them didn’t really work well here. The additional challenge is the bias between train and test and unstable LB. The thing I found to be the best for this data is removal of the uncertain examples from training set based on the out of fold predictions. I excluded ~1400 Redbound and 300 Karalinska data, so my clean training set contains about 8700 items. Relabeling the excluded images didn’t improve the performance. At the end of the competition [some ppl have discovered this trick as well](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/161909), so I got nervous about my LB position 😬 
**This trick gave ~0.005 public LB boost and 0.01+ private LB boost.**

## Pipeline

The method I have used is mainly based on my [tile pooling pipeline](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/146855) with several additional tricks:


Based on my public kernel, one could reach ~0.90 public and 0.91 private LB averaged (over different submissions) using 36x256x256 tile setup and the kappa loss (see below) without any other changes.

[**kappa loss**](https://arxiv.org/pdf/1509.07107v2.pdf): I have used one minus


(both predictions and labels are centered based on the mean value of labels). In my experiments I found that kappa loss &gt; sorted [binning loss](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/155424) &gt; binning loss &gt; MSE &gt; CE. The only issue with the loss is that bs should be sufficiently large: I needed to pretrain models on low resolution and then continue training on intermediate resolution with bs =6-8 (**progressive resizing**), while at bs 1-2 I couldn't get convergence. The predicted value is limited within [-0.5,5.5] as `yp = 6*sigmoid(p) - 0.5`. In addition, I have CE aux for prediction of the Gleason score with 0.08 weight.

**tile cutout**:  **Instead of using all tile tiles, why not to randomly select part of them** (let's say 96 out of 128). So, I can use large bs and the model is regularized in the same way as if cutout is used. It gave me quite good boost for CV and quite fair boost at LB.

**128x128x128 tiles from intermediate resolution**: It appeared that many smaller tiles work better than 36x256x256. I think that it helps to select the tissue areas more effectively and at the same time prevents overfitting. 

**tile selection augmentation** The idea is quite simple: instead of [generating a single tile set](https://www.kaggle.com/iafoss/panda-16x128x128-tiles), I can generate 4 with adding sz/2 padding to x, y, or both before cutting the image into tiles and selecting ones having the most of the tissue. So, each tile in these 4 datasets will be different, but it is important not to mix tiles from them. During training I select the dataset by random, so effectively I have x4 data. It is an approximation of tile selection with random offset each time, which would be even more effective (based on my experience in Severstal competition), though, too slow to be used with intermediate res images. I also tried TTA based on tile selection, (as well as selection of the tile set with the largest tissue area out of 4), but I couldn't get any statistically significant improvement.

**The above tricks gave ~0.005 boost over my baseline if I consider multiple submissions**. Though, score from submission to submission could change quite a bit.

**Advanced tile selection**: In addition to my main pipeline I also tried to use the method proposed by @akensert [here](https://www.kaggle.com/akensert/panda-optimized-tiling-tf-data-dataset) with 128x128 tiles (but didn't use it for my final sub). It gave ~0.934 private LB single model performance on average for 8 different single 4 fold model subs (**and maximum 0.941 private LB**) and ~0.910 average public LB (~0.912 maximum). Too bad that I didn't create an ensemble based on this method. The trick that I have used in the model for training with such tiles is **n-pooling**: at test I apply the pooling only to nonempty tiles (n for a particular image, while the batch may contain some extra empty tiles for padding), and training is done with random selection of 96 tiles with repetitions (so I don't consider white tiles, which can change the mean statistics at pooling).

**High resolution**: I tried to train several models on high res/2 resolution, 128x256x256 tiles. With tile cutout I could use batches of size 4 (and include 64 random tiles). However, the results were slightly worse than ones for 128x128x128 tiles from the intermediate resolution layer. It indicates that **going to higher resolution would likely provide only a minor boost**, even if I try to optimize my pipeline for training with small batches. Some idea I had is based on having two conv parts for intermediate and high res tiles. First pass through the low res model selects tiles having the highest uncertainty. Next, the selected tiles (but in high res) are passed through the second conv part. The produced feature maps are downscaled twice and replace the low res feature maps that had high uncertainty. Finally, pooling and head are applied to produce the final prediction. This method would allow to keep overall statistics of tiles with only correcting ones that model is not confident about. However, too large level of noise in the training set, noisy LB inconsistent with train labeling, and small potential gain, which would likely be overshadowed by the noise, have prevented me from going into this direction. Also, more complicated pipeline is more likely to be broken under such competition, where there is no certain way to evaluate the performance.

**Augmentation**: I have used Albumentations with the following parameters:
```
Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9, 
                         border_mode=cv2.BORDER_CONSTANT),
        OneOf([#off in most cases
            MotionBlur(blur_limit=3, p=0.1),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        OneOf([#off in most cases
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        OneOf([
            HueSaturationValue(10,15,10),
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(),            
        ], p=0.3),
    ], p=1)
```

**Model**: All my models are based on **ResNeXt50**, similar to my public kernel, with batch norm in the head replaced with Group-norm. The optimizer, best model selection based on CV, and other things are similar to my public kernel, and I was using 32-48 epochs, depending on the setup. In addition, I tried ResNet34, ResNeXt101, and EfficientNet, while all of them were performing worse. I think ResNet34 may be not capable enough for this task, while ResNeXt101 is too large to do training on my computer with sufficient bs. However, I would say that **the model is the minor thing in this competition, and the main role is played by considering the noise and by optimizing the pipeline: there is no magic model, but there are hard work and solid understanding of the task and the data**.


## Final ensemble

The submission that gave me the 11th place (**0.930 private LB/0.917 public LB**) is based on a majority voting ensemble of 8 models (4 fold) with 6 TTA. They are trained with different train/val splits and other modifications in the training procedure. On average each of the models trained in such manner gave **~0.930 private and ~0.910 public LB** single model 4 fold performance (with the **maximum of 0.938 and 0.916**, respectively). So, I got quite fair score, not good or bad luck (and my LB position almost haven’t changed). However, the large number of models was a way to survive in this storm. My another ensemble of 11 models, not selected as a final one, got 0.934 private LB. And as I mentioned above, more advanced tiling gives about **0.004 boost** on private LB (while similar public score as my main approach based on 128x128x128 tiles), with the average of **~0.934** and the maximum of **0.941 private LB**, but unfortunately, I haven’t built an ensemble based on them for my final submissions.

**the code snippets are available at:** https://github.com/iafoss/PANDA

And I would like to congratulate all participants and wish the best luck in the next competitions. I hope some of my tricks would be useful to you.
