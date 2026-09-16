# 3rd place solution with code: ArcFace

Competition: humpback-whale-identification
Rank: #3
Source: https://www.kaggle.com/c/humpback-whale-identification/discussion/82484#latest-502552

## UPDATE: code available on github
[https://github.com/pudae/kaggle-humpback](https://github.com/pudae/kaggle-humpback)

---

Congrats to all the winners.
Thanks to Kaggle and hosting team for an interesting competition.

Here is my solution summary.
Solution Summary
==============

Dataset
---------
- **Validation set**: randomly sampled 400 identities that has 2 images + 110 new whales (= 400 * 0.276).
- **training set**: all images except new whales.
- I doubled up the identities by horizontal flip.

Model
-------
**bounding box &amp; landmark**

- I used annotations by [Paul Johnson](https://www.kaggle.com/c/humpback-whale-identification/discussion/78699) and [Radek Osmulski](https://www.kaggle.com/c/humpback-whale-identification/discussion/76281). (Thanks to Paul and Radek. Without your contribution, I couldn't achieve such high score.)
- I made 5 fold CV and trained 5 models using them.
- IOU: 0.93

**whale identifier**

- [ArcFace](https://arxiv.org/pdf/1801.07698.pdf) approach is used.
- Following the paper, the layers after last convolution were replaced to flattening -&gt; BN -&gt; dropout -&gt; FC -&gt; BN.
- densenet121
- m 0.5 (the default value of the paper)
- weight decay 0.0005, droupout 0.5

Augmentation
-----------------
- average blur, motion blur
- add, multiply, grayscale
- scale, translate, shear, rotate
- align or no-align

Training
---------
- adam optimizer
- learning rate of 0.00025 -&gt; 0.000125 -&gt; 0.0000625

Inference
-----------
**getting embedding feature for identity**

- For each images, I got multiple feature vector by using 5 bounding boxes and landmarks.
- For each identities, the center of all feature vectors was used as final embedding feature.

**getting embedding feature for test image**

- For each images, multiple feature vectors were generate and the center of the feature vectors was used.

**computing similarity**

- The cosine similarity of above two feature vectors was used as the measure of similarity.

**selecting threshold**

- The threshold for new whale was selected so that the proportion of new whale is about 0.276.

The process to the final method
========================
Followings are the process to the final method.

**without landmark**

At first, I excluded the identities having only one image and new whales from the training set. For inference, the identity of the most similar image of the training set was used as the predicted identity.

 &gt; Public LB: 0.90, Private LB: 0.90 

After using the center of all feature vectors in the same identity, I got

&gt; Public LB: 0.942 / Private LB: 0.939

After using weight decay 0.0005

&gt; Public LB: 0.946 / Private LB: 0.946

After including the identities having one image to training set

&gt; Public LB: 0.963 / Private LB: 0.961

**with landmark**

When I used aligned image, network was trained faster but the score was not improved.

&gt; Public LB: 0.962 / Private LB: 0.959

The bounding boxes and landmarks of some images are very poor and it seems to prevent improving scores. So I also used non-aligned images.

&gt; Public LB: 0.965 / Private LB: 0.961

Finally, I doubled up identities by horizontal flip. Flipped images have different identities but visually very similar. So I set the logit value of flipped to zero to prevent flowing gradient.

&gt; Public LB: 0.968 ~ 0.971 / Private LB: 0.965 ~ 0.968

Congrats to winners again.
Thanks.
