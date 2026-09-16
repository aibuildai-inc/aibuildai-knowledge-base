# 1st place - (I've ran out of catchy phrases :V)

Competition: playground-series-s6e1
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s6e1/writeups/1st-place-ive-ran-out-of-catchy-phrases-v

First of all, I would like to thank Kaggle for a very nice dataset this month. The CV/LB correlation was even better than S5E8/S5E11 and there was a decent amount of signal. I would also like to thank all the people who shared their code and insights, including but not limited to: @cdeotte, @yekenot, @omidbaghchehsaraei, @siukeitin, @masayakawamata, @mikhailnaumov.

Now my solution:

#FE

I have 2 main feature sets that I used for my models:

Feature set 1 (this did best with NNs):
- Cyclical features from @yekenot's notebooks
- The formula from [here](https://www.kaggle.com/competitions/playground-series-s6e1/discussion/665915)
- A categorical copy of each numerical base feature
- Digit features
- Certain combinations of features + TE (mean/std/skew)
- Certain combinations of digits + TE (mean/std/skew)

Feature set 2 (this did best with GBDTs):
- The same formula as feature set 1
- Ordinal mapping of categorical features from @cdeotte's notebook
- A categorical copy of each base feature (including categoricals as they are mapped to numerical)
- Digit features

#Models

I focused on making stronger models rather than a lot of different models this time. This resulted in my best single model placing 7th! Their CV/LB scores (without post-processing) are as follows:

| Model type | CV | Public LB | Private LB |
| --- | --- |
| RealMLP | 8.58742 | 8.54280 | 8.58005 |
| XGBoost | 8.59480 | 8.55235 | 8.59251 |
| TabM | 8.59651 | 8.55740 | 8.59279 |
| CatBoost | 8.60027 | 8.56047 | 8.59537 |
| DeepTables | 8.60147 | 8.55467 | 8.59020 |
| LightGBM-dart | 8.60870 | 8.56743 | 8.60437 |
| LightGBM | 8.60910 | 8.56951 | 8.60670 |
| Keras MLP | 8.61511 | 8.57016 | 8.60957 |

There are some other weaker models like Resnet, FTTransformer, xLearn FFM, etc. but those are not listed here (partially because I'm kinda lazy :V).

#Ensembling

My final ensemble has a total of 190 models ensembled with Ridge: CV 8.56634, LB 8.53096, PB 8.57273 (with post-process). Other ensemblers like HC, Autogluon, CatBoost, etc. gave worse scores. 

#Post-processing

Somewhere in the middle of the comp, I tried using Isotonic Regression to post-process the oofs passed to Ridge as well as after and it gave a decent improvement. Turns out, my final ensemble without the post-process is only slightly worse on LB and a lot better on PB: CV 8.56959, LB 8.53099, PB 8.57152. I tested it on my best single model as well and both LB and PB are worse as well.

#Hyperparameter tuning

I also want to note that adding a lot of models with different hyperparameters worked really well. My boost from around 8.5335 down to 8.5309 was due to hyperparameter tuning all my best models (RealMLP, TabM, GBDTs) and saving all the oofs from all the trials. I would recommend using WandB sweeps as you get very handy visualizations and can parallelize the tuning on multiple machines (at least I think so, I've never tried it :V).

#Conclusion

I've said it many times and I'll say it again, ensembling lots of diverse models and some FE is all you need. Best of luck for the next comp (which I will probably not be participating in) and happy Kaggling!
