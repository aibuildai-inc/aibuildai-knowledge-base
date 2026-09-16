# 43rd : Positive-Unlabeled Learning based Solution

Competition: hpa-single-cell-image-classification
Rank: #43
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238504

Congratulations to the winners and thanks for all the competitors, great hosts!🎉
We are members of Harada Laboratory at Tokyo University.

This competition was very interesting in that classification should be done with weakly supervised labels.
And I thought even not high-score solution is worth to be made public.

**Digging and saying "here was nothing" is also important.**

I'll share our Positive-Unlabeled based solution.

---

# Overview

- cut images into cells, treat labels as Negative-Unlabeled then optimized AUC
- optimization of AUC which uses sigmoid was hard and the score didn't improved enough
- treating images as i.i.d. samples could induce not sufficiently tight loss

We are members of the same laboratory. We participated this competition to search our master course theme. m0ka took care of image level classifier solution and I searched image-level classifier solution. Considering PublicLeaderBoard scores, we decided to use the image-level classifier as our final model.

# Pipeline

- ensemble of 10 ResNet34 models.
- AUC optimization under Negative-Unlabeled label setting

[image]

# Purpose

- train models without label noise
- Optimization minimize PR-AUC
- validation of models can be done under label noise

We had image-level labels, but we didn't have cell-level labels. If we classify cell images in this situation, we would suffer from falsely added labels. And even after training, we can't validate our models with these noisy labels.

Even under this situation, we can train our models without such bias if we use statistical machine learning technique. Below I will explain our solution.

# Assumption and Setting

- treat labels as Negative-Unlabeled.

Let's consider cell-images with 0-17 class labels.

Because these labels were added to original whole image, there are many False-Positively added labels. Then we could assume that

1. Added labels can be actually negative.
2. Not-Added labels are always negative.

In this point of view, we can see this competition setting as Negative-Unlabeled Setting. Negative label is always negative and positive labels are always positive.

**[Edit]** After this competition, I found a paper, [[Peng and Zhang, 2019]](https://arxiv.org/abs/1905.12226) which treats this setting😅. However, this paper's method requires class prior P(y_{k}) for the unbiased estimation, which can't be accessed. To resolve this issue and optimize AUC rather than Bayes-Risk, we introduce AUC based solution. 

# Optimization of AUC

- Optimize AUC so that PR-AUC improves.
- we don't have to know class-prior

In normal setting, mAP is enhanced by optimizing some losses like BCE Loss. However because labels are noisy, this loss can be hard to optimize.

Rather I decided to optimize AUC which resembles ROC-AUC.

Notate sample x's class i score output as fx. P is positive samples' probability and N is negative one. AUC of class i is calculated as

[image]


I don't write precise theory here, but in PU Learning setting, it is known that

- using symmetric loss, which satisfies l(x) + l(-x) = const. can be reduce bias caused by Falsely added labels.

Combining AUC optimization and symmetric loss leads object function, which we don't have to use class prior. we can notate it as

[image]

# Good points & Bad points

- Good points
    - we can use whole images!
        - only 1 labeled images are limited.
    - there was correlation between LB and validation score relatively.
        - this wasn't true when I used BCE Loss and score improved +0.06pt which was large in this competition.
- Bad points
    - optimization of sigmoid was hard.
    - taking bags of image apart could make loss loose.

After training this method, I couldn't improve our model much. I tried hard to solve issue caused by sigmoid which is difficult to opmize, ended up first 3weeks solution.

8th and 9th also use image-based classifier, but they treated image-level labels as no-noise labels. This can be tight loss compared with this solution.

# Others

If we could, we wanted to give some contribution to HPA community like bestfitting did in the last competition. We couldn't and he did it again, congrats!🎉

We can't say our solution worked enough, but I hope you enjoyed this solution. Thanks!

## Appendix

### solutions' background

positive label is described as y=1, negative is y=0.
unlike PU setting, we can't assume P(y=1) = P(y=1|s=1). So I changed some assumptions.

[image]
