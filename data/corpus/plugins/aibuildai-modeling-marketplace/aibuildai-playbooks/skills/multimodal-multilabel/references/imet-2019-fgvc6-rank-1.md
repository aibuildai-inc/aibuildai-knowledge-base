# Solution | Private Top 1

Competition: imet-2019-fgvc6
Rank: #1
Source: https://www.kaggle.com/c/imet-2019-fgvc6/discussion/94687#latest-570986

Congrats to all who finished in the gold zone. Now, while we are waiting for second stage results let me share my solution part. I do not expect shake up, so I think leaderboard will be stay the same.

First of all, what is the main challenge in this competition? Of course it is noisy data and we should find a way to work with it. 
So, i can divide my solution into several stages:

Hardware: 6x 1080 ti with 36 cores and 120 gb RAM in total.

**Stage 0. The same part for all stages:**
Models: SENet154, PNasNet-5, SE-ResNext101 (all pretrained from cadene repo)
CV: 5 folds with Multilabel Iterative Stratification
*Data Augmentations:*
```
        HorizontalFlip(p=0.5),
        OneOf([
            RandomBrightness(0.1, p=1),
            RandomContrast(0.1, p=1),
        ], p=0.3),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),
        IAAAdditiveGaussianNoise(p=0.3),
```
*Data Preprocessing:*
This part was a great finding which speed up convergence and increased score very good. So, analysis of different models with Crops(224), Crops(288), Crops(300) and so on shows that it influence on tags a lots. Actually, lets imagine that you have picture 500x300 and this image labeled with tag «person» if there are persons somewhere at this image. There is a not so low probability that you crop it =&gt; you data becomes more and more noisy. 
So, I decided to use CropIfNedeed + Resize. The main Idea is perform crop if it possible (for example crop 600x600 from 300x500 image — 300x500. from 300x900 — 300x600). Code below:

```
class RandomCropIfNeeded(RandomCrop):
    def __init__(self, height, width, always_apply=False, p=1.0):
        super(RandomCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, h_start=0, w_start=0, **params):
        h, w, _ = img.shape
        return F.random_crop(img, min(self.height, h), min(self.width, w), h_start, w_start)
``` 
So, I used the following couple:
```
RandomCropIfNeeded(SIZE * 2, SIZE * 2),
Resize(SIZE, SIZE)
```
With SIZE = 320 for SEResNext101 / SENet154 and SIZE = 331 for PNasNet-5

*TTA*: Using the previous approach for data preprocessing we can just use TTA2 (original + hflip image)

*Scheduler*: manual, the correct scheduler increased accuracy a lot. I found it by analyzing plots with metrics / losses. 
I started with LR: 0.005 then train 15 epochs, drop it by 5 times and train some epochs more.

**Stage 1. Training the zoo:**

*Loss*: Focal
*Sampling* with logarithmic weights
*Batch size*: 1000-1500 (accumulation 10-20 times for different models)

**Stage 2. Filtering predictions:**
Drop images from train with very high error between OOF predictions and labels (I consider them as very noisy and incorrect)

*Loss*: Focal
+ Hard negative mining (Sample 5% of hardest samples each epoch)


Re-train models from scratch.

**Stage 3. Pseudo labeling:**

*Loss*: Focal

The simplest version of pseudo labeling, add the most confident predictions (highest  np.mean(np.abs(probabilities - 0.5)) ) to the training dataset.

Re-train models from scratch.

**Stage 4. Culture and tags separately:**

I found 2 main things about cultures and tags:
1. tags are less noisy than cultures (and starting from some epoch the culture accuracy did not increased very much)
2. some tags classes are very similar to ImageNet classes

So, using already trained weights as pretrain I continue training model for 705 classes (tags only) 

*Loss: Focal* -&gt; BCE (yes, here I switched the loss, I helped too)

**Stage 5. Second-level model**

I construct the binary classification dataset: I took 1103 (number of classes) rows per each image and trying to predict that this class relates to this image (0 or 1). So it means the length of my train data becomes `len(data) * 1103`.

I extract next features:
- probabilities of each models, sum / division / multiplication of each pair / triple / .. of models 
- mean / median / std / max / min of each channel
- brightness / colorness of each image (you can say me that NN can easily detect it — yes, but here i can do it without cropping and resizing — it is less noisy)
- Max side size and binary flag — height more than width or no (it is a little bit better for tree boosting than just height + width in case of lower side == 300)
- Aaaaand the secret sauce: ImageNet predictions ;) As I already mentioned — some tags classes similar to ImageNet classes, but ImageNet much bigger, pretrained models much more generalized. So, I add all 1000 (number of ImageNet classes) predictions to this dataset

So, then I trained LightGBM on all this data.

**Hints / Postprocessing:**
- Different threshold for cultures and tags models. 
- EDA shows that tags are fully labeled and cultures may be not. So, I binarize predictions using the following code:

``` 
culture_predictions = binarize_predictions(predictions[:, :398], threshold=0.28, min_samples=0, max_samples=3)
tags_predictions = binarize_predictions(predictions[:, 398:], threshold=0.1, min_samples=1, max_samples=8)
predictions = np.hstack([culture_predictions, tags_predictions])
```

(thank you to @lopuhin for the binarize_predictions code in his kernel)

Total training time: ~10-15 days (that's why I submitted not very often ;) )


P.S. I did not submit all this ensemble to the private stage… Due to I faced with kernel limit time :facepalm: (for example I used LGBM for pseudo-labeling but did not use it at the next cycle of test predictions)
