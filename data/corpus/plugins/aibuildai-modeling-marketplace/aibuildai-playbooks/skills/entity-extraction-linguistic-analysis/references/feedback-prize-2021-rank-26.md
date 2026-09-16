# 26th Place Solution: datasaurus, ln & Tom

Competition: feedback-prize-2021
Rank: #26
Source: https://www.kaggle.com/c/feedback-prize-2021/discussion/313274

I’d like to thank the organisers for a great dataset and competition, and I’d especially like to thank my teammates @nazarov and @tikutiku who I have learnt a huge amount from.

# CV Strategy
We used a 5 fold `StratifiedGroupKFold` method  grouped by text id. Initially I was not validating with post-processing, but since PP was a major part of the final score I switched to use the `min_thresh` and `proba_thresh` methods used in the public notebooks. Initially our CV-LB agreement was good, but it seemed to diverge past the 0.710 LB boundary.

# Models
Like most other teams, we quickly found that although Longformer and BigBird were convenient, they were quite inefficient and models like deberta seemed to perform much better. @nazarov did a great job of building a pipeline to split and merge the long sequences using weighted overlaps (Tom also had a similar method before we merged) that enabled us to use the following models:
- deberta-large CV: ~0.697 (<- best single model public LB 0.704)
- deberta-xlarge CV: ~0.698-0.700
- bart-large CV: ~0.680-0.692
- muppet-large CV: 0.670
- luke-large CV: 0.691

# Training strategy

## Parameters
Typically trained for 5 epochs. We found that the results were highly sensitive to batch size, and BS=4 seemed to be a sweet spot, using a LR=1e-5 & cosine decay. I was able to train deberta-xlarge on my GPU using DeepSpeed, BS=2 and gradient accumulation=2.

Our models also used multisample dropout and my model used an attention layer in the head. We also trained a few models using the BIEO scheme. We also also replaced all the "\xa0" characters with a space.

## Loss
We used a range of loss functions for diversity, Cross Entropy, a modified Dice Loss (since Dice and F1 are equivalent) and Tom was using [Conditional Random Fields (CRF)](https://pytorch-crf.readthedocs.io/en/stable/)

## Augmentation
Random token masking with probs in the 0.03-0.07 range.

# Post-processing
The only methods we used were the `min_thresh` and `proba_thresh` methods which were optimised using OOF predictions, the link evidence method and a method to ensure that there was only a single “Lead” prediction. Tom also made a heroic effort in the last few hours of the competition to use the essay topic clusters in PP, but we ran out of time. Unfortunately, we missed the gradient boosted trees post processing notebook.

# Ensemble
The ensemble was tricky since the CRF method after the Viterbi decoding outputted hard labels instead of logits/probabilities . We found that taking the simple mean of the OHE CRF predictions with the softmax probabilities of the other models worked better than linear or 2nd stage stacking models. 

I also managed to get a pipeline to create accurate token to word mappings which enabled the majority of the prediction splitting using tensors instead of expensive for/while loops. This gave quite a bit of speed up over the public notebook methods and allowed us to ensemble more models in the time limit.

To choose the final ensemble models, I used a similar technique I used in CommonLit, where I start with a basket of all the models, and iteratively remove one model at a time based on how much it impacts the CV F1 score, until I have a basket of the most informative models that will fit into the time limit.

Our final two subs used:
- 6 models (2x deberta-xlarge, deberta-large-mnli, deberta-large (CRF & BIEO), bart-large (CRF), muppet-large), using 3 out of 5 folds (18 checkpoints). CV: 0.716, Public: 0.710, Private: 0.723
- 11 models (as above plus a few more variants of deberta & bart), using 2 out of 5 folds (22 checkpoints), CV: 0.717, Public: 0.710, Private: 0.721

# What did not work
- [Mixup Transformers](https://arxiv.org/pdf/2010.02394.pdf)
- Deberta versions 2 & 3 (we built a custom wrapper for sentencepiece to get offset mappings)
- Using essay topic clusters as either an additional target or input features
- 2nd stage stacking models
- Larger batch sizes

# Final words
I’d like to again thank my amazing teammates and I really enjoyed the competition.

Unfortunately my long wait for the GM title will have to be a little longer since I will be taking a 6-7 month break from Kaggle to thru-hike the full Pacific Crest Trail 🎒️. I’ll see you all again on the leaderboard towards the end of the year 💪️
