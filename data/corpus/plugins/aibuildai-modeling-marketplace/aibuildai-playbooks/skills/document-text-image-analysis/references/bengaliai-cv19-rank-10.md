# 10th Place: Single Model Using Snapshot Ensemble w/ code

Competition: bengaliai-cv19
Rank: #10
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/136815

## UPDATED
I added the additional study at the end of this post.  
* link to discussion about DataAugmentation
* links to discussion and notebook about visualization of sSE Block 
* tables of my late submission results (w/o RandomErasing, w/o sSE Block, w/ common sSE Block, and w/ **_Chris's Magic_**)
* link to the source code repository on GitHub

----

I was more surprised than pleased when the private leaderboard uncovered. I didn't imagine such a big shake happens.
I still don't understand why got this place (sorry, since I had only 4 days, I was not able to do enough experiments) , and want to investigate it by late submission.

Although there remain some mysteries, this is my first solo gold medal. I'm so glad to share my solution as a gold place one and become Competitions Master!

Lastly, congrats to all the teams got the medal and participants finished this competition! And thanks to Kaggle and Bengali.AI for hosting this competition!

<br>
Here, I'd like to share my solution's summary.

### Resources
Kaggle notebooks and a local machine (GTX1080ti x 1)

## Approach
At first, for tackling the unseen  graphemes problem, I split train data by multi-label stratified group K-fold (regarding a grapheme as a group) and tried training models. However, I was not able to train models successfully. I suspect this is because some components are contained only in one or few grapheme(s).

My approach is very simple － training a model using all train data (because I want to include all components in training). This way, However, has a risk of overfitting. For preventing worsening of generalization performance, I train a model using cosine annealing scheduling and do snapshot ensemble.

There is no special pre-processing nor post-processing.

## Model

The model is based on SE-ResNext50 and not so special, but has something unique **in global pooling step**. The figure below is the model overview.

[model overview]

Instead of simply applying Global Average Pooling to feature map and using its result as **common input** of each component's head, I used **sSE Block** and Global Average Pooling **for each component**.

If you want to know more details about sSE Block, read the original paper:

[Concurrent Spatial and Channel Squeeze &amp; Excitation in Fully Convolutional Networks(Abhijit Guha Roy, et al., MICCAI 2018)](https://arxiv.org/abs/1803.02579)

**NOTE**: For passing 1-channel images to the pre-trained model(3-channel), I use a simple way. See [the comment bellow](https://www.kaggle.com/c/bengaliai-cv19/discussion/136815#781819) for more details.

## Training
### Data Augmentation
applying augmentations  by the following order, implemented by [albumentations](https://albumentations.readthedocs.io/en/latest/)

[Original Image: 137x236]
-&gt; Padding(to 140x245) -&gt; Rotate(rotate limit=5, p=0.8)
-&gt; Resize(to 128x224)    -&gt; RandomScale(scale limit=0.1, p=1.0)
-&gt; Padding(to 146x256) -&gt; RandomCrop(to 128x224, p=1.0)
-&gt; **RandomErasing(mask-ratio-min=0.02, mask-ratio-max=0.4, p=0.5)**
-&gt; Normalize(by channel-wise mean and std of train data)
-&gt; [Model Input: 128x224]

As you see, **I did not use CutMix/MixUp.**

### Loss
calculating softmax cross entropy for each component, and averaging them by weights (`grapheme_root`:`vowel_diacritic`:`consonant_diacritic` = 2 : 1 : 1) 

### Optimization
- Optimizer : SGD + NesterovAG(momentum: 0.9) with weight decay(1e-04)
- batch size: 64
- epoch: 105
- learning schedule: cosine annealing (3 cycle)
  - 35 epoch per cycle
  - learning rate: max=1.5e-02, min=0.0

## Inference
### transform
applying the following transforms to images before feeding them into models

[Original Image: 137x236]
-&gt; Padding(to 140x245) -&gt; Resize(to 128x224)
-&gt; Normalize(by channel-wise mean and std of train data)
-&gt; [Model Input: 128x224]
 
### ensemble
Obtaining models(35, 75,  and 105 epoch)' outputs using softmax by component, simply averaging them and applying argmax by component.

## Why got this place ?
I don't still understand, but have some hypotheses.

### RandomErasing is Good?
Many participants used CutMix or MixUp. While these techniques seem effective in making combinations of components, not effective in disentangling interdependence of components in original combinations, especially about MixUp.

Because of this, may be, simple masking method like RandomErasing is good.

### sSE Block is Good?
I'm not sure whether or not sSE Block is the best. But simply applying global average pooling looks not good for me because I suspect each component has spacial dependence.

I suppose a kind of weighted global average pooling **for each component** is good.

### Training a model by all data and Snapshot Ensemble is Good?
In previous competitions, training K-fold and K-fold averaging was the very effective way for me. But in this competition, it was very difficult for me to split K-fold because of unseen graphemes problem and rare components, I trained a model using all train data.

Here are each cycle's and snapshot ensemble's scores.

|       model         |  Public Score (Rank) | Private Score (Rank) |
|:-------------------:|:-------:|:--------|
| cycle 1 (35epoch)   | 0.9754 (288th)  |  0.9438 (23rd) |
| cycle 2 (70epoch)   | 0.9821 (160th) |  0.9499 (14th) |
| cycle 3 (105epoch)  | **_0.9843 (124th)_** |  0.9497 (14th) |
| snapshot ensemble   | 0.9840 (131st) |  **_0.9536 (10th)_** |

The model cycle 3 achieved the best public score, but this slightly overfitted.
Snapshot ensemble achieved the best private score, which is **0.0037** higher than the best single model(cycle 2).

I think snapshot ensemble is good when you want to train models **using all train data**.

<br>
That's all. Thank you for reading!

<br>

----

## Additional Study

### RandomErasing is Good?
- Discussion: [CutMix/MixUp is **Not** All You Need?](https://www.kaggle.com/c/bengaliai-cv19/discussion/137029)

### sSE Block is Good?
- Discussion: [Key of the 10th solution? : Where sSE Block looks](https://www.kaggle.com/c/bengaliai-cv19/discussion/137552)
- Notebook: [Visualize 10th place model: Where sSE Block looks?](https://www.kaggle.com/ttahara/visualize-10th-place-model-where-sse-block-looks)

### Late Submission

#### All the results
- w/o RandomErasing : not using RandomErasing
- w/o sSE Module : simply apply GAP to feature map extracted from SE-ResNeXt50 and feed it into each component's head (Dense -&gt; ReLU -&gt; Dropout -&gt; Dense)
- w/  **_Common_** sSE Module : apply one common sSE-Pooling to feature map and feed it into each component's head

| model |       cycle       | Public Score | Private Score  |
|:-------:|:-------------------:|:-------:|:--------|
| w/o RandomErasing |  1 (35epoch)   | 0.9425  |  0.9187  |
| 〃 | 2 (70epoch)   | 0.9479 |  0.9201  |
| 〃 | 3 (105epoch)  | 0.9648 |  0.9345  |
| 〃 | Snapshot Ensemble | 0.9643 |  0.9373 |
| w/o sSE Module |  1 (35epoch)   | 0.9746 | 0.9431 |
| 〃 | 2 (70epoch)   | 0.9824 |  0.9490  |
| 〃 | 3 (105epoch)  | 0.9836 |  0.9485  |
| 〃 | Snapshot Ensemble   | **0.9843** | 0.9517 |
| w/ **_Common_** sSE Module | 1 (35epoch) | 0.9739 | 0.9433 |
| 〃 | 2 (70epoch) | 0.9809 | 0.9474 |
| 〃 | 3 (105epoch) | 0.9827 | 0.9496 |
| 〃 | Snapshot Ensemble | 0.9832 | **0.9527** |

#### Compare results using Snapshot Ensemble (&amp; **_Magical_** Post-Processing)

| model | Public Score (rank) | Private Score (rank)  |
|:--------:|:----------- -:|:---------------|
| w/o RandomErasing | 0.9643 (1210th) | 0.9373 (66th) |
| w/o sSE Module | 0.9843 (124th) | 0.9517 (12th) |
| w/ **_Common_** sSE Module  | 0.9832 (149th) | 0.9527 (12th) |
| final sub model (**_component-wise_** sSE Module) | 0.9840 (131st) |  0.9536 (10th) |
| w/ **Chris's Magic**(-0.6,-0.6,-0.4) |  **_0.9848 (112th)_** | 0.9623 (4th) |
| w/ **Chris's Magic**(-0.8,-0.8,-0.7) |  0.9828 (153rd) | **_0.9653 (3rd)_** |

OMG! It's a really magic!

If you want to use this magic (it is very easy to use but effective!), check this discussion: https://www.kaggle.com/c/bengaliai-cv19/discussion/136021

----

## code

I've published the source code on GitHub:
https://github.com/tawatawara/kaggle-bengaliai-cv19
