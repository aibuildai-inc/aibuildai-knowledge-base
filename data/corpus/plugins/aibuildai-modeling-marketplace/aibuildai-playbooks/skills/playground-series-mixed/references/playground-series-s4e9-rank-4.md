# #4 solution | Beating a dead horse: blending works, but "blending" doesn't

Competition: playground-series-s4e9
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s4e9/discussion/536973

There was a discussion [**here**](https://www.kaggle.com/competitions/playground-series-s4e9/discussion/536917) just hours before the deadline asking whether blending will work. I will summarize the gist of my discussion in that thread and give a broad overview of my solution.

Proper ensembling, which includes blending, has been used for a long time and on a variety of datasets and problems. It works. That's not me saying it, but rather a known fact.

Guessing weights to combine public solutions is not blending in the original sense, so I will call it "blending." That one may or may not work, and it is to a good degree dataset- and luck-dependent. The luck was not with us with this dataset, and I thought that would be the case after about 3 days working with this dataset - see [**here**](https://www.kaggle.com/competitions/playground-series-s4e9/discussion/531968). This "blending" approach will work here and there, but not in general. Maybe that's good enough for some of us, but I learned after about 3 competitions that it is not my cup of tea.

I selected for scoring my 1st and 3rd best submissions, so very happy about that. Both of them were hill climbing ensembles of 32 models each. The better of the two ensembles (they differ by 8 RMSE points) was made completely of my models. The second ensemble included about half my models, and the other half came from public notebooks that I thought were done well, and most importantly the notebooks that had out-of-fold predictions. Can't do proper ensembling without those!

**EDIT:** Thanks to @roberthatch for reminding me [**here**](https://www.kaggle.com/competitions/playground-series-s4e9/discussion/536969#3003452) that I didn't give proper credit to the groups whose OOF models were part of my second ensemble. I also used LAMA's OOF models from [**here**](https://www.kaggle.com/competitions/playground-series-s4e9/discussion/531884), @cdeotte from [**here**](https://www.kaggle.com/code/cdeotte/nn-starter-lb-72300-cv-72800/output) and @ravaghi from [**here**](https://www.kaggle.com/code/ravaghi/s04e09-ensembling-techniques-for-regression).

Ensembles must be diverse to work, so both of mine included Keras factorization machines (FMs), xLearn FMs, Lasso, CatBoost, LAMA NNs, AutoGluon ensembles, and a huge array of individual tree models that were pulled out from AutoGluon directories (a mix of RandomForest, ExtraTrees, XGBoost and LightGBM models). A couple of FastAI NN models, which I don't even know how to make, were pulled also from the cache of AutoGluon models. The first four approaches modeled all the features as categoricals, and generally worked the best. I also used AugoGluon-encoded datasets (with some NLP-generated features) for separate modeling with factorization machines, and that gave a small boost.

Tried feature engineering, which gave slightly better CV scores but not LB scores. Didn't use any of it in the end as I suspected overfitting. One thing I do regret is adding the original dataset too late, as it seemed to benefit the overall score. I had only 2 days to incorporate those models with original used car data, and they pushed my ensembles by about 20 points. I suspect that would have been more productive if I started earlier.
