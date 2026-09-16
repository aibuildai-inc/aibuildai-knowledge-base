# s3 class postprocessing (+0.008 on private LB)

Competition: rfcx-species-audio-detection
Rank: #25
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220314

As it was mentioned by others, s3 was a very interesting class. Looking more into this, our team figured out that we should scale up predictions for s3 class. We checked a number of ways to do this, but eventually a simple scale of 1.5x worked best for us (our predictions were all positive numbers, that scaling probably wouldn't work if your predictions are in logits).

This scaling lifted our final ensemble from 0.933 to 0.941 (+0.008) on private LB.

The intuition to convince ourselves that this was not a small public LB fluke (because it similarly helped on public LB) was as follows:
- Each class roughly has similar number of TP labels 
- But you can see on OOF train predictions and on test prediction that classes are very imbalanced (let's say looking at frequencies of different classes being top1 or top3 predictions). And s3 class is actually clearly the most ubiquitous. 
- Now because of specific labeling of this competition, we are in a situation where s3 is very common class, and hence very often present next to other labels, and model gets 0 target for s3 class (if you did not do any corrections, like loss masking)
- So model learns to be extra cautious at predicting s3, when in fact it should be quite aggressive in predicting it

Hence we applied post-processing here. 
We could not leverage this on other classes, but s3 really stood out as a clear outlier for us.
