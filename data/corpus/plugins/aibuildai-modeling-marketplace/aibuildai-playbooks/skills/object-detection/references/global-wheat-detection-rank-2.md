# 2nd Place Solution with Code [MIT-Compliant]

Competition: global-wheat-detection
Rank: #2
Source: https://www.kaggle.com/c/global-wheat-detection/discussion/175961

[update] The code is now available at https://github.com/liaopeiyuan/TransferDet .

Now that the results are out, I've decided to share a bit more than [my previous post](https://www.kaggle.com/c/global-wheat-detection/discussion/172433).

First of all, huge thanks to @rwightman for his PyTorch implementation of EfficientDet and @shonenkov for his amazing starter kernel. I've also learned a lot from the discussion posts.

This is an interesting challenge with a very noisy dataset and huge domain shift between train/test, and within train/test as well. The kernel running time limit also gives possibility to do semi-supervised learning, which can in a way reduce the aforementioned problem.

The first choice I made was the ways I validate and perform model selection. This is a tough choice because 1. object detection is inherently a hard task and 2. the training data are noisy and unstable.
Eventually my decision was "relative detection loss" on a 20% stratified split with bad boxes removed (by manual inspection), as hinted [here](https://www.kaggle.com/c/global-wheat-detection/discussion/166564#927359). My reasoning is as follows:

We know that train and test distributions are wildly different, so mAP to me would make little sense because it's a heavy processed result (with hyperparameters in between of nms, and which mAP to look at, etc.) And detection loss we use to train our networks can capture the overall classification and regression performance. In addition, I don't use this loss to compare across models, and instead I accumulated a list of procedures that I considered as "comparable under loss," by pure empirical results and a little bit of instinct, and used them to select minor choices such as augmentations. For model selection overall, I took the public LB with a grain of salt, and in a way because I ran out of time to train diverse models, I didn't have to make those huge choices of model ensemble. I believe that I can investigate into this more once the entire set is released and cleaned.

Now, on to data insights. We can all agree that the data, even with blatantly bad boxes removed, are really noisy, in two forms: noisy anchor and noisy boxes. 
- For noisy boxes, this is the direct result of different understanding of wheat heads by annotators (including an automatic one, Yolov3). So I ran a few experiments to ascertain the noise. I degraded the boxes' x,y coordinates by 10% and 5% respectively (kind of the norm on articles robust object detection), and trained with my previously best pipeline. Interestingly, for both degradation we've seen a drop in model performance, so in a way the data is more informative than a 5% shift. My models were extremely costly to run, so I didn't run any more tests or designed experiments that could reveal more about bbox noise, but I had a basic idea of how to proceed. 
- Noisy anchors, on the other hand, is a little tricky, because the problem is not just about disagreements between labellers. Remark that our training data are 1024 crops of larger pictures, and (correct me if I'm mistaken) annotation is only done on crops, with the wheat heads on edges only getting annotated if 1/3 of the head is visible. This is kind of crucial because APs in this competition has a huge impact on the resulting mAP performance. Models that are too confident over the border (identifying all wheat heads no matter if it's over 1/3 or not, sometimes even leaves mistakenly) or too pessimistic both degrades final performance. This also explains my jigsaw trick, which I will explain in the following paragraphs. Finally, the source images are really different. One could train a simple CNN classifier and classify the source with validation accuracy 99.99%. Even so, you can learn a metric learning model on say 4 sources and achieve 80%+ homogeneity when extracting embedding on the remaining 3 sources that the model has never seen.

Therefore, all the models I developed are trying to address these issues. I first tried DetectorRS, but only managed to get pass 0.72 LB. So I moved on to the EfficientDet pipeline. I ran some simple tests from D4 to D7, and concluded that D6 with COCO-pretrained weight is the best. Here are my reasonings.

D0-D5: not large enough
D7: Remark that D7 got its precision by larger input size (and hence larger anchors). I really couldn't fit 1536-resized images on my GPU and get reasonable results.
D6 with noisy student/AdvProp pretrained cls backbone and random-initialized FPN/box, cls net: Object detection is quite hard. I couldn't make these models converge.

The next thing I did was trying to improve baseline performance of my EfficientDet-D6 model. I tried over 10 different techniques by dissecting the timm-efficientdet repo line by line and hacking into them, but turned out simplicity and elegance prevailed. The best performance is achieved by the default setting of the D6 model (with anchors unchanged; I tried KNN clustering of anchors in train set but it gave worse results), with default huber loss and focal loss. Augmentation wise, it's just an empirically deduced recipe:

```
 A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 1024), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
	    A.Transpose(p=0.5),
	    A.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
	    A.OneOf([
		A.Blur(blur_limit=3, p=1.0),
		A.MedianBlur(blur_limit=3, p=1.0)
	    ],p=0.1),
            A.Resize(height=1024, width=1024, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )
```

and for `__getitem__`:

```
if self.test or random.random() > 0.33:
            image, boxes = self.load_image_and_boxes(index)
elif random.random() > 0.5:
            image, boxes = self.load_cutmix_image_and_boxes(index)
else:
            image, boxes = self.load_mixup_image_and_boxes(index)
```

Remark that mixup is the 1:1 version; I didn't get the beta version to work (I tried to reweight the boxes during loss calculation; I got worse results). Then I tuned the WBF over TTA (8x, because 16x has duplicates) pipeline a bit with basic sanity checks and exclusion of boxes that have low score or are too big or small. The overall baseline gave public LB of around 0.753.

Then, I realized that similar to the TGS competition I participated few years back, we can get more data by doing jigsaws. You can check out one of the popular public kernels for how to restore the original full-sized images. After this, I realized two more problems: 

1. Boxes over the edge are fractured in the sense that if a wheat head appears in half in one image and in half in another, there will be two boxes in the final image. This will generate huge amount of noise and will certainly degrade model performance.

2. This is more subtle. Remark that in the data insight section I talked about the conundrum of boxes over the edge. Now, if we are now cropping 1024 patches from the larger image, the resulting distribution is actually different than the one represented by what we now can see as crops at special positions. I didn't really document the changes in model performance, but it is indeed significant enough for training.

My solutions is as follows:

1. I tried to correct every single label manually but got bored after 30 minutes. So I wrote an automatic procedure that does that for me:
  - Get all boxes that have one of the coordinates divisible by 1024
  - Prune those over the edge of the big picture
  - Generate the segments over the edge and match it to box (aka we only care about the side on the edge)
  - Calculate pariwise-IoU over edges
  - Pair boxes using a greedy approach
  - Fuse boxes
I then used the corrected data to train my models. And, to encourage APs gain for my models, The images have a 0.5 chance of feeding into the network without cropping, e.g., resizing an 2048x2048 down to 1024x1024 instead of cropping it from the larger image. I did some ablation experiments and this does improve model performance.

2. What I did here is basically treating images obtained from jigsaw as "pseudo-labels," or simply images that come from different distributions.
   - The first model I developed is based on the paper "A Simple Semi-Supervised Learning Framework for Object Detection". Basically, the resulting loss is a weighted sum of loss calculated by original image and jigsaw image separately. 
   - Then, I remembered the trick introduced in the paper "Adversarial Examples Improve Image Recognition", where images of different distributions goes through different BatchNorms. So I copied the batchnorm stats from baseline and replicate it twice, and during 2-nd stage training original data and jigsaw data passes through different norms, and the stats are calculated separately. During inference, I use the distribution of the original data. I tried extended scenario such as a 7-split norm by source, or average of different splits of norms during inference, all of which gave improvements locally, but not on public LB, so I didn't investigate further.

I also designed a detection variant of FixMatch, but I couldn't fit it in my GPU with image size = 1024.

Finally the semi-supervised learning part was pretty mundane, as fancy techniques (STAC, SplitBatchNorm, Pi-Model, Mean Teacher, etc) did not fit under the kernel eval time frame. All I did was to freeze the backbone to allow training with 16GB vram, and different hyperparameters to generate the pseudolabels. You can see the details in my open-sourced kernels. Most choices are more empirical than insightful.

Sorry for this messy solution journal as working alone gives you a bad habit of trying too many stuff and not documenting well enough. Also, since I'm working alone, I didn't have enough time to investigate in depth some of the ideas I brought up above. I will seek to improve the solution journal in the upcoming days, and maybe write a tech report if I have time.
