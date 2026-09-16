# 14th place solution: data and metric comprehension

Competition: airbus-ship-detection
Rank: #14
Source: https://www.kaggle.com/c/airbus-ship-detection/discussion/71664

My team and I are happy to achieve a 14th place. Congratulations to everyone else in medal positions! And many thanks to my team mates [radek][3] and [kerem][4] for inspirational collaboration.

This is my first attempt at a Kaggle competition with tiers/points. 12 months ago I had not heard of kaggle or pytorch, deep learning was a mystery, and python a snake, so it feels gratifying to do well. For that, I must thank both *kaggle* and *fast.ai* for creating communities and tools that allow autodidacts such as myself with a single GPU to progress so rapidly, and sponsors like Airbus for being so brave as to be open with data. I give back by explaining our solution ...

**The TL;DR; Solution**

The solution is based on the idea of

 - careful data preparation (read: exploit data issues)
 - understanding the balance between false positive rate and segmentation score (read: exploit the metric)
 - local validation (read: trust local results not public scores)
 - likelihood of the public leaderboard and private leaderboard differing in score-effecting ways (read: exploit LB carving)
 - using dead simple out of the box models until the above had been exhausted

**Data Prep**

I was glad to learn the train data wasn't being re-released as the overlaps and leaks within it created opportunity. I created mosaics slightly differently to others, by using the train-masks rather than the train-images, which possibly made it much simpler and perhaps even better as these smaller ship-dense mosaics, separated by sea, could be stratified more homogeneously. I couldn't find tools to do this, I just used python hash values of the masks and re-assembled tiles by position. A 768px window was slid over mosaics in 256px steps to create a 50k image train set, striped into 5 folds.

**Binary Classifier**

A simple 256 resnet34 classifier, best used at the end not the start of a prediction pipeline.

**Segmentation**

A rudimentary resnet34 encoded unet, direct from [fast.ai v1][1], which uses Leslie Smith's   [1cycle policy][2] for learning. Training directly on 768px for 4 frozen then 32+ unfrozen epochs, small batch size of 4.  Dihedral augmentation with some limited brightness/contrast augmentation in training.

I used a single GPU with only 8GB memory use, albeit on a 2080 ti with mixed precision training (thanks again to fast.ai v1) and 16 hour training runs. With this and the good data, single folds of 0.849 were achieved. Evidently a 5-fold CV didn't help any on the private LB despite big public LB differences (.729-.741).

Our team ensembled this with resnet18 and resnext50-se models for a 0.001-2 boost. Our one regret is we didn't explore these ensembles further, due to public LB disappointment, and lack of time for local validation examination, but I am confident higher scores could have been achieved. Our best ensemble submission would have scored 11th if selected.

We used dice loss, focal loss, BCE loss, and mixed versions, and do not consider the choice of loss function to have been significant.

**Post processing**

The intuition here was that the severe penalty (0 score) for labelling a ship where there is none meant it was necessary to sieve the results based on ship size, f2 score for that size, number of ships, circularity, whether the blob was on an edge, and classifier confidence for the image so as to avoid the penalty. 

And that the difference in the ship-laden public LB and the ship-barren private LB created an opportunity, albeit a simple algebraic one yet one that seems to have passed many people by.

Ships &lt;100px had f2 scores of just .2-.3 in validation meaning 1 false positive image in every 4-5 would wipe out any gain. Whereas ships &gt; 5000px had an f2 of .7 and are unlikely to trigger a false positive. So in local validation we took the c2900 ship images that the segmenter said existed and used a crude heuristic to sieve out c500 ships: small ships if not very confident (.99+) with the binary classifier, and medium ships if not somewhat confident (.95+). We only submitted predictions for 2400 ie 2/3 of the 3700 images with ships (from the .765 private LB).

While the curious data situation and puzzling LB split created an opportunity and helped us achieve a good result in this case, I plea for well provisioned data and more representative LBs going forward, if only to preserve sanity.

**Attempted but didn't seem to help**

- More complex models didn't seem to help as much as understanding the data, the metric, and local validation results did.
- We trained a 'coast and structures' classifier 
- We trained a segmentation model for 20-100px ships
- A [Circle Frequency Filter][5] gave cool results on images, but didn't have time to include in a model and to refine ship widths in post processing 
- Fitting BB's, going so far as to try IOU measured BBs over blobs.

**Things we didn't have time to explore enough that may have helped**

- Varying the pixel selection threshold away from 0.5 for different sized ships
- Ensembling strategies
- Stacking ship characteristics with segmentation predictions
- Other mask strategies e.g. trimaps

**What wasn't attempted**

- More test time augmentation
- Breaking apart conjoined ships
- A 'wake detector' that may have helped binary classification of very small sub 20px ships.
- Probing to see a ship-size makeup of the public LB

**fast.ai**

As an addendum, I can't sing the praises of fast.ai enough. It really does make deep learning 'uncool' and accelerates experimentation around problem solving, not model making. No need to be a programming whiz. No need to subscribe to paid DL services. Indeed I don't need to make any code public, as the training needed to get 14th place was as simple as:

    #prep data
    #define unet learner
    learn = get_learner(data)
    lr=2e-3
    learn.freeze_to(1)
    learn.fit_one_cycle(4, lr, div_factor=100, pct_start=.3)
    learn.unfreeze()
    learn.fit_one_cycle(32, [lr/64,lr/8,lr], div_factor=25, pct_start=.3)

  [1]: https://www.fast.ai/2018/10/02/fastai-ai/
  [2]: https://arxiv.org/abs/1803.09820
  [3]: https://www.kaggle.com/radek1
  [4]: https://www.kaggle.com/keremt
  [5]: http://www.itfind.or.kr/Report01/200302/IITA/IITA-2015-037/IITA-2015-037.pdf
