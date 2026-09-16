# 49th place (Weighted Loss etc.)

Competition: feedback-prize-english-language-learning
Rank: #49
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369793

I'm new to the NLP and this is my first NLP competition.
So I learned a lot of things from the competition. 
Thanks for hosting the competition.

# **OverView**
I ensemble 15 models (with 4 folds).
I used orthogonal initialization, reinitialize the last 1 layer and layer wise learning rate decay for my all model.
These methods also worked well for me.

And Weighted Loss slightly improved CV and LB for some models.
The  Weighted loss is maybe my unique point.
I shared [my code](https://www.kaggle.com/code/taruto1215/49th-place-deberta-v3-base-weightedloss/notebook) for Weighted Loss.
The Weighted loss tend to improve the score of targets which can be easily predictable.
For model diversity, I wanted to make two model (One is with Weighted Loss, other is without Weigted Loss) for each models, but There was no time to make all these model.

Then I tried applying some method (AWP, SiFT, MIXout, and so on..) into the Deberta-v3-base model. But all of them didn't work.
However at the all most end of this competition, I found SiFT and MIXout  worked well for Deberta-v3-small (Without Weighted Loss).
I wanted to apply these method into other models (large, xsmall, Roberta and so on), but there was no time to train...
So these methods could be applied into only Deberta-v3-small model.

Models used for the my best submission is below.


For deberta-v3-XXX set the max_len=1462, others set the max_len=512

# **What Worked**
- Orthogonal initialization [High impact]
- Mean pooling [High impact]
- Reinitialize the last layers [High impact]
- Layer-wised learning rate decay [High impact] 
for deberta-v3-base lr_mult=0.9, for the small lr_mult=0.8
- SiFT [It depends on model (maybe it depends on hyper-parameter cause it sensitive for hyper-parameter)]
Only for deberta-v3-small / xsmall it works (start at 2 epoch, learning_rate=1e-3, init_perturbation=1e-2
[https://arxiv.org/pdf/1911.03437.pdf](https://arxiv.org/pdf/1911.03437.pdf)
[https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/sift/sift.py](https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/sift/sift.py)
- MIXout [Medium impact for some models]
Only for the deberta-v3-small/xsmall, it works (mixout_prop=0.075)
[https://arxiv.org/abs/1909.11299](https://arxiv.org/abs/1909.11299)
- Weighted Loss   [Medium impact]
Please reference below and [my code](https://www.kaggle.com/code/taruto1215/49th-place-deberta-v3-base-weightedloss/notebook)

[https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)
I guess the weighted loss improve the score of targets which can be easily predictable.
Considering the characteristic of weighted loss the results reasonable.


- For the ensemble, below methods works (these method did not  improved CV and LB)
-- Multi-sampled dropout (For the deberta-base / large, dropout_rate=0.05)
[https://arxiv.org/abs/1905.09788](https://arxiv.org/abs/1905.09788)
-- AWP (Adversarial weight perturbation) (But in ensemble it work as small weight)
[https://arxiv.org/abs/2004.05884](https://arxiv.org/abs/2004.05884)
-- Weighted CLS pooling for the last 4 layers.
-- Add NewLine ([BR]) token  (CV improved 0.005, but LB worse, but in ensemble it works)

# **What Didn't Work:**
- Pseudo labeling using FB2+3 ( CV was much improved, but LB rank was very low, so I didn't use)
- concatnate/weighted different layers mean pooling 
- Attention pooling
- Masked language modeling (Pretrain)
- SWA (Stochastic Weight Averaging)
[https://arxiv.org/abs/1803.05407](https://arxiv.org/abs/1803.05407)
- FGM (Fast gradient method)
[https://arxiv.org/pdf/1806.01477.pdf](https://arxiv.org/pdf/1605.07725.pdf)
- Random token dropout

 AND SO ON....

# **Ensemble strategy:**
1. Optimize ensemble weights (target-wise) using optuna. (Trust CV)
If I use optuna for my OOF. the optuna set some Deberta-v3-base weights ZERO. 
**I was afraid to submit, but I believed in my CV and this is the best in the PB!**
2. Manually optimize weights considering LB rank. (Trust LB)
3. Just averaging 1. + 2. (Trust CV and LB))

# **CV  strategy**
- Just using MultilabelStratifiedKFold, Fold4, seed42

Thanks.
