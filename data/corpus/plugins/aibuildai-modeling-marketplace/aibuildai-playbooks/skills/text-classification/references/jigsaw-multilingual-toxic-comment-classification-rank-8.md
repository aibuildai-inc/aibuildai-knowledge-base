# 8th place solution

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #8
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160937

Thanks a lot to Kaggle and Jigsaw to host this competition. I am very happy to achieve my second solo gold medal within a week :)

My main goal when starting this competition was to learn how to use TPUs on Kaggle. I started with Pytorch, switched to Tensorflow, and then switched back to Pytorch. I think overall the usability is still very shaky, and 3 hour limit is also limiting things quite a bit. There is constant progress on it though, and I hope that they will be better usable in the future. I made a separate [topic](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/159723) before.

The whole test set was visible in this competition, so subbing was very easy and blending many models was very important. Unfortunately, I could not select my best sub because it included a blend with a [kernel](https://www.kaggle.com/shonenkov/tpu-inference-super-fast-xlmroberta) where no training script was available. I am happy I followed my heart to not select it even though I would have ranked a few spots higher with it. I really would encourage people to stop posting inference only kernels, because logically people will use them if they score well, but the learning potential is quite limited imho.

### Models

There is not much differences in the models I fit to those that have been posted throughout the competition. I could not get any better results compared to using XLM-Roberta-Large. I also tried uni-language models without any improvements over XLM-Roberta.

I also do not believe that any advanced sampling technique, label smoothing, augmentation, or other stuff floating around was giving any improvements. 

The only real important thing was to translate the training data to the languages at hand. This is in line with the experiments reported in the original [XLM-Roberta paper](https://arxiv.org/pdf/1911.02116.pdf).

I blended models by doing rank-averaged blending.

### Data

I only used provided data and translations of it.
I think the most untapped potential was to use talk pages from Wikipedia dumps to further train the models. I did not have time to explore this, but I hope someone else did and can report on it.

I actually think one of the main issues of the provided English data was that it included barely any Bot comments, but the test data contained quite many of them, specifically Turkish which is also why I think the score is better there than for others.

### Post-processing

After blending, post-processing was quite important for me to reach high scores. As the metric is a global AUC over six different languages, it became quite quickly obvious to me that there should be room to optimize the global score by shifting predictions of individual languages up or down. 

I tested this on CV with the three languages and managed to get very good boosts there. I further tested how a simulated public and private dataset would behave in terms of optimization, assuming a random split between public and private. So what I did was to repeatedly select random 30% of CV, optimize the thresholds for the 30% and apply them to the remaining 70% and check how far off they would be from the optimal thresholds of the 70%. Overall, I saw that there was not much difference, which is why I was quite confident that I can optimize the thresholds on public LB and they should work well for private LB. Optimizing on CV alone does not work because CV only has 3 languages. This is an example plot of such an analysis:



The x-axis is the scaling factor for the language at hand, the y-axis is the improvement of the score over no scaling. The blue line is the improvement of the 30% sample and the orange of the 70% sample. Note that the improvement is not 0 for factor 1.0 in this case as another language has already been optimized.

Additionally, I optimized a range of values for which I optimize the scaling factors. I found that not scaling the top predictions improved scores, so I only scaled predictions between 0.25 and 0.875 (rank average values). You can get some further minor improvements by averaging the predictions for similar texts.

My whole PP routine boosted scores by roughly 0.004, so quite a lot.
