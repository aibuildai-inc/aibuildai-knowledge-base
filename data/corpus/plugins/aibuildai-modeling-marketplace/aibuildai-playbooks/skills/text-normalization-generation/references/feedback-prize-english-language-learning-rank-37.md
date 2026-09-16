# 37th place solution

Competition: feedback-prize-english-language-learning
Rank: #37
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/371602

First of all, thanks to the host for an interesting competition and congratulations to all the winners!

**Overview**
- Our final submission is ensemble of 13 models. The weights of the models are determined using nelder-mead.
- cv0.44384/PublicLB0.435426/PrivateLB0.435417



**What Worked**
- Ensemble including roberta and funnel
          - The best PrivateLB for the deberta-only ensemble model was 0.435807.
          - The best PrivateLB in every submit was 0.435103.

- Optimization by nelder-mead
　　　- Our bestCV(0.44287) and bestPrivate(0.435103) were ensemble models with weights optimized by nelder-mead.
　　　- In the final submission, the best model is the ensemble model of cv0.44384,Private0.435417 optimized by nelder-mead

**What Didn’t Work**
- adding LSTM to model
- Embedding SVR with fine-tuned models
- Different loss functions

**Important ciations**
- [FB3 Deberta Family Inference](https://www.kaggle.com/code/kojimar/fb3-deberta-family-inference-9-28-updated)
- [FB3 / Deberta-v3-base baseline](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
- [FB3 / Deberta-v3-base baseline [inference]](https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-inference/notebook)

**Thanks and Acknowledgements:**
Thanks to our hosts for hosting the competition. And thanks to the Kaggler's for sharing their helpful notebooks and discussions.Special thanks to my teammates for such a perfect teamwork

**Team Members:**
- @rachisotaro 
- @shinuk
