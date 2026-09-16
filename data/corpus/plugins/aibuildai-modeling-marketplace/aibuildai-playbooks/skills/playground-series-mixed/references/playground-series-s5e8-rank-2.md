# 2nd place - Yet another ensemble

Competition: playground-series-s5e8
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s5e8/writeups/2nd-place-yet-another-ensemble

TLDR: My approach for this competition is basically the same as S5E6: generate a bunch of diverse oofs (ideally with different models rather than same model and different hyperparams/FE) and slowly add them to the ensemble.

Firstly I would like to thank the people who shared their code and insights, including but not limited to: @cdeotte, @omidbaghchehsaraei, @yekenot, @itasps, @siukeitin and @tilii7.

Now onto my solution: 
#Models
My final submission is an ensemble of 59 models, ensembled with Catboost and with scores (of the best model) as following (note: not all models were fully optimized/run with the best feature set, so scores can probably be improved by a lot):
| Model type | CV | Public LB | Private LB |
| --- | --- |
| TabM | 0.976810 | 0.97765 | 0.97750 |
| XGBoost | 0.976543 | 0.97741 | 0.97707 |
| LightGBM | 0.976013 | 0.97693 | 0.97660 |
| RealMLP | 0.975983 | N/A | N/A | 
| Catboost | 0.975066 | 0.97590 | 0.97571 |
| DeepTables | 0.974459 | 0.97579 | 0.97559 |
| TabR | 0.973597 | 0.97580 | 0.97518 |
| Gandalf | 0.973107 | 0.97438 | 0.97410 |
| Random Forest | 0.972771 | N/A | N/A |
| GRN | 0.972365 | 0.97418 | 0.97380 |
| FTTransformer | 0.972252 | 0.97439 | 0.97398 |
| CNN | 0.970265 | 0.97447 | 0.97386 |
| Bartz | 0.968910 | 0.97250 | 0.97205 |

#Feature Engineering

My best feature set had the following:
- TE mean and CE on bigrams (competition targets and orig targets)
- Products of bigrams (for numerical features)
- Cyclical features from @yekenot

#Ensembling

I tested a few ensemblers (namely Ridge, LightGBM, Catboost and HC) and Catboost was the best one on the Private LB. Their scores are as follows:

| Ensembler | CV | Public LB | Private LB |
| --- | --- |
| Ridge | 0.977207 | 0.97804 | 0.97786 |
| LightGBM | 0.977408 | 0.97816 | 0.97796 |
| Catboost | 0.977432 | 0.97817 | 0.97796 |
| HC | 0.97740 | 0.97807 | 0.97789 |

After reading a few writeups, it seems ensembling with Autogluon might have provided a boost.

#Conclusion

From the past 3 episodes (we're obviously excluding S5E7 :V), I can say that the formula to getting in the top 3 is a big, diverse ensemble and maybe a trick or two. Thanks to everyone who has helped me get this far, and happy Kaggling :D!
