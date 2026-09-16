# 22nd place - NNs were again better than GBMs

Competition: playground-series-s6e2
Rank: #22
Source: https://www.kaggle.com/c/playground-series-s6e2/writeups/22nd-place-nns-again-better-than-gbms

My thanks to everyone who vigorously participated in conversations, and particularly to @mirko45 and @masayakawamata for their expert discussion topics. Special thanks to @omidbaghchehsaraei for his excellent [**RealMLP notebook**](https://www.kaggle.com/code/omidbaghchehsaraei/the-best-solo-model-so-far-realmlp-lb-0-95397), and in general for all his NN contributions. Props also to these gentlemen for their LB placement!

I am writing this for 2 reasons: 1) to extend [**the argument from a month ago**](https://www.kaggle.com/competitions/playground-series-s6e1/writeups/2nd-place-nns-sometimes-work-better-than-gbms) where I concluded that NNs often work better than GBMs; 2) to remind everyone that **ensembling is fundamental, but blind blending is detrimental!**

I started modeling only in the last week and didn't decide to compete until 48 hours ago. While I don't approve of 10 submissions per day that was reinstated this month, it worked to my advantage this time as I managed to submit ~15 single models and 5 ensembles. Those were 5 RealMLP, 4 TabM, two each of CatB and XGB, a Keras-FM and an FFM model. I will list below only the best scoring model from each group.

| Model | CV | Public LB | Private LB |
| --- | --- | --- | --- |
| RealMLP | 0.955747 | 0.95388 | 0.95531 |
| TabM | 0.955742 | 0.95389 | 0.95530 |
| CatBoost | 0.955630 | 0.95381 | 0.95522 |
| XGBoost | 0.955647 | 0.95376 | 0.95521 |
| Keras FM | 0.955661 | 0.95379 | 0.95520 |
| xLearn FFM | 0.955511 | 0.95368 | 0.95508 |

Even though the types of models I chose were diverse, their predictions weren't. I always check my models by multiple criteria other than just their scores and simple correlations, and they were similar. Also, there didn't seem to be much improvement by using feature engineering, even though I did it in limited fashion. For example, a fairly simple Keras factorization machine, which treats all features as categorical without any other modification or feature engineering, scored just 0.00009 worse than my best models. Heck, it scored only 0.00015 lower than the best score on private LB. So after I made the initial batch of models, I ran several long HPOs for the top 4 models, which in the end moved both CV and LB scores by 0.00002. My general observation is that NNs were more competitive than GBMs, just like last month. Now, "more competitive" in Kaggle terms usually means "the same" in real-world terms, because we are talking here about differences on 5th decimal place. Still, it is fair to say that RealMLP and TabM have leveled the playing field in Kaggle competitions. My thanks for that to @davidholzmueller and the Yandex team (the original creators of TabM). And to all of you being brought up to use only XGBoost or LightGBM: it is time to consider adding other things to your toolbox!

There was nothing else in my approach other than keeping up with good ML practices: 10-fold cross-validation with the same fold split, and at least 3 different ensembling approaches. Hill climbing and NN ensemble solutions were my two picks, while the CatB ensemble had an identical 5-decimal score but was placed higher. All of my top 5 solutions had the same 5-decimal scores of 0.95533, so it couldn't have been that much of a difference between them. Still, I selected models 3 and 5, which was not optimal. One of the top two solutions had a really low CV and there was no rational way to pick it.

I want to remind everyone not to fall for the trap that is blind blending. All those high-scoring notebooks mean nothing because they overfit to the public LB. I already showed [**here**](https://www.kaggle.com/competitions/playground-series-s6e2/discussion/679364) how most blind blenders dropped precipitously on private LB. A zoom into the same plot for the top 300 competitors shows that hardly any of them retained their LB positions. All those blue dots below the diagonal were competitors who went up by hundreds of spots.


