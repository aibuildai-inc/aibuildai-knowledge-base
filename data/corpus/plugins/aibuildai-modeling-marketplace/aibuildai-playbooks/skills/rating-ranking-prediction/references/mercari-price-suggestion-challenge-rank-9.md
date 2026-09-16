# 9th place solution 0.3975

Competition: mercari-price-suggestion-challenge
Rank: #9
Source: https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50319

9th place solution is available now

Code Link: https://www.kaggle.com/leeyun/ensemble-model?scriptVersionId=2460425

My final submission is the 106th version, It's an ensemble of Conv1d, FastText and FM+FTRL. 

 **1. Offline Validation and Data Preprocess**

Many thanks to **chenglong** 's share: https://github.com/ChenglongChen/Kaggle_HomeDepot, I really learn a lot from this tutorial, such as cross validation, feature engineer, hyperopt, stacking and so on.

My offline code is based on this tutorial, I use 5 fold cv. I use hyperopt to find good parameters.

**2. NN models**

For neural network, I've tried some models from public kernels, many thanks to the authors: **yanglu** ,  **ololo**  and **nzw** .

GRU: https://www.kaggle.com/yyll008/gru-25-12-12-with-keras-512-64-relu-sgdr-lb0-432?scriptVersionId=1956235/code

Conv1d: https://www.kaggle.com/agrigorev/tensorflow-starter-conv1d-embeddings-0-442-lb

Fasttext: https://www.kaggle.com/nzw0301/simple-keras-fasttext-val-loss-0-31 

I put many effort on GRU. But it's accuracy is lower and time consume is higher compared to conv1d and fasttext. So I abandoned it. The conv1d is a very simple revise version from ololo, it achieve 0.413 in public LB. The fasttext was inspired by anttip's topic: https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/49430. I just remove the conv1d layer from my conv1d NN, and achieve 0.414 alone and it's very fast. The ensemble model of conv1d NN and fasttext can achieve 0.405.

**3. Non NN models**

For Non NN models, I've tried many linear models, such as ridge and lasso, many non linear models, such as XGBoost, LightGBM, , and large scale sparse model FTRL, FM+FTRL,   Many thanks to the public kernel's authors: **ynqa** , **Bojan Tunguz** , **anttip** . Especially thanks to **anttip** 's work, which is a really useful tool.

Ridge: https://www.kaggle.com/object/more-effective-ridge-script/code

LGBM: https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44823?scriptVersionId=1855115/versions

FM+FTRL: https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/47295

**Konstantin**  give a kernel https://www.kaggle.com/lopuhin/eli5-for-mercari demonstrate the usage of ELI5,  but i'm too stupid to improve my models by this usefull tool :P. 

I can't improve linear model into 0.425. The LGBM's performance is very bad and time consuming. The XGB's performance is even more terrible. I can't improve FTRL into 0.43, so I abandoned it. 

The FM+FTRL is a very powerful tool, it achieve 0.407 alone. the only risk is out of memory, so it need carefully monitor the memory usage offline with simulate data. 

**4. Conclusion**

I really learn a lot from this competetion. The sharing spirit, the effective usage of memory and time, the skill to model neural networks, the conception of FM and FTRL, cross validation, feature engeneer, and most important, keep learning and improving. 

Thanks my teammates, they give me many help and valuable suggestions, especially when I felt frustrated, depression, and lose my way. Thanks all your guys that compete togethor in the last two month, It's a great honor for me to fight with you！
