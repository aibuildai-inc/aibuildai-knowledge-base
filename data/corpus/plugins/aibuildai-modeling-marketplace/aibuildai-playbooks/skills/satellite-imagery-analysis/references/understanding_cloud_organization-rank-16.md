# 3rd silver place key points

Competition: understanding_cloud_organization
Rank: #16
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/118065

First I would like to thank Max Planck Institute and Kaggle for hosting this interesting competition.

I would like to share some of the key points of my 3rd place (silver) solution :-) It sounds cool right?  (well I love to make that top silver medal the most out of it, forgive me :P)

### 1) Cutmix augmentation
Naturally thinking, cutmix is the best way to deal with this competition. We can cut a part of this image and paste to another image. This idea came off from my mind without knowing its existence academically, which I later found an official paper about it.


How to do that in code?
I search for some augmentation package, but find it hard to flexibly code it my way. So I decided to do it manually.

The different of doing cutmix or not is just an extra section of  `__getitem__` in data generator. Here, `indexes_augment` is the random indexes pick from the training data, `w_cutmix` and `h_cutmix` are width and height of the crop. So I just get the random starting position of width and height in the original drawn image (`X`) and insert part of other image (`Xc`) into it. 

**Cutmix boosted both LB and CV by 0.004.**

### 2) Pseudo-label
Pseudo-label only works if we correctly select good samples, as well as the correct number of samples. I did this by assessing the `quality` of each predicted validation image by calculating:
`quality = (number of pixels with probability &gt; top) + w*(number of pixels with probability &lt; bot)`. Here, `top` can take values from [.7, .75, .8, .85, .9], `bot` can take values from [.1, .15, .2, .25, .3], and `w` is the weight of low-value pixels as compared to high-value pixels, which can be taken from, say, [.1, .5, 1, 2, 10]. 

I get the `quality` of all validation data, rank it, and select `nb_samples` most confident samples from it, and see the score. I search through a full set of validation data and had a result something like this


So, I can manually decide `bot`, `top`, `w`, and `nb_samples` as long as `nb_samples` are reasonable with the corresponding score. For example, `bot=.1`, `top=.7`, `w=1`, and `nb_samples`=1000 (with corresponding `dice=0.77xx`), which means the most 1000 confident predictions out of 5546 train images can have that good dice. Then I can pick up the same ratio of images from test predictions, which is (1000/5546*3698).

**Pseudo labelling boosted around 0.003 on both CV and LB.**

### 3) Estimating private LB distribution and decide to trust CV
First, I did a test on private LB, based on [this topic](https://www.kaggle.com/c/understanding_cloud_organization/discussion/109793#latest-631950). 



From that probing result, we need to make 1 assumption:
"Train set and full test set should have the same distribution of classes."

Then, by submitting each class as empty (others as 1-pixel masks), we can know the percentage of each class in the public LB. Then the assumption I make will allow us to know the percentage of each class in the private LB. The result is:


Train data: Fish 49.85%, Flower 57.35%, Gravel 47.00%, Sugar 32.36%.
Private test data: Fish 49.94%, Flower 56.76%, Gravel 47.20%, Sugar 31.30%.

As you can see, the distributions of private test and train very similar, allowing me to completely trust CV. Therefore during the whole competition, I never probed LB by submissions, but only stick with full k-fold to search for post-processing parameters. **This is important, as it guides the way we do everyday in the competition**. And you can see that I jumped on private LB, and I also selected my possibly best submission.

Finally, I still would like to emphasize again that late sharing should not be encouraged. I have a bad thought that whenever I see the excessive sharers around in future competitions, I would be very disappointed, and discouraged from competing. In other words I am somehow "scared" of their existence. 

Thanks for reading!
