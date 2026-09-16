# Share your approaches?

Competition: landmark-retrieval-2019
Rank: #11
Source: https://www.kaggle.com/c/landmark-retrieval-2019/discussion/94494#latest-543837

I would like to break the silence here and started to share what we have done, and hopefully more others will share as well, especially from the top teams that we can learn :)

In fact, we have nothing fancy in the end (all fancy things didn't work for us this time , unfortunately :( We followed similar paths of last winner's approach, and final model is ensemble of three (inception v3 + Atten&amp;Gem pooling, as well as David's Xception  and SEResNet101 with VLAD pooling, each has 512 in dim) 

We used DBA for each model to enhance image vectors quality, and finally go through QE (1 round) using the concatenated vectors from three models, which boosted performances quite a bit. 

We tried different poolings, different ensemble approaches, but they didn’t give much difference or improvement, and the best one seemed to be GeM (with some CNN Attention) and VLAD pooling, with one round of DBA and one round of QE, that’s it. When we trained models, we started with classification training, followed by fine tuning using Siamese fashion, which further improves a bit more.  

However, I guess the biggest (possible) mistake was that we only trained using old train data set only, and ignore the set of new train, mainly because of the stage 1 LB shows so much correlation with old train, and we wrongly assumed that stage 2 would be similar, and final Stage 2 turned out to be from image set whose distribution was so different from old train (much more noisy) and much closer to new. 

In the end, we found a very good approach to remove most of the distractors images from test + index, but surprisingly for us, simply removing those distractors didn’t help at all for our current pipeline (DBA + QE). It could be because non-distractors will match well with non-distractors, and won’t be affected much by distractors that we are trying to remove. 

I found out this excellent QE code was very useful - https://github.com/fyang93/diffusion, only that they shouldn’t use ANN at all even for large dataset, which unnecessarily increases computations.
