# 20th place solution

Competition: asl-fingerspelling
Rank: #20
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434658

Thanks to Kaggle, Google and other organizers for hosting this exciting competition. Hope this kinds of competition open in kaggle more often.

**TLDR**

My solution is a single large model trained with CTC loss with no ensemble(39MB). The output shape of the model is (batch, sentence length, class).

**Data Preprocessing and Augmentation**

Basically the same as the previous 1st solution. But I found using (x, y, z) and pose landmarks gives better score. I used the following landmarks: left hand, right hand, eye, lips and pose landmarks.

**Model**

I used previous 1st solution. However, I used the branchformer. Also, I changed the multi head attention to relative multi head attention. My model consists of 1 dense layer - 4 conv1d layer - 6 branchformer layer. In addition, I used stochastic depth.

**Training**

Epoch = 150
bs = 64
Lr = 8e-4
AWP = Epoch * 0.1
Schedule = CosineDecay with warmup ratio 0.1
Optimizer = Adam with Lookahead
Loss = Inter CTC loss(https://arxiv.org/abs/2102.03216)
