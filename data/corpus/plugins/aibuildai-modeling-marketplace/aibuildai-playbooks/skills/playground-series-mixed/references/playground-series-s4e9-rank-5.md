# #5 Solution | 🚀 AutoGluon submission 1 on day 1

Competition: playground-series-s4e9
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s4e9/discussion/537173

Our 5th place final solution was our very first submission to the competition, around 8 hours into the competition launch on day 1. We chose this submission despite it not being one of our top public LB submissions because we had good reason to believe it would do well on private LB, given we did extensive cross-validation testing. 

Unfortunately, it didn't get a good public LB score (72221.55, aka rank 626 in public LB), and this ended up leading to us getting [6th in this competition's Grand Prix standings](https://www.kaggle.com/competitions/playground-series-s4e9/discussion/531971) since it is scored on the public LB and not the private LB, which ultimately led us to [getting 2nd in the Grand Prix overall by a single point delta](https://www.kaggle.com/automl-grand-prix******) :joy:. Oh well, them's the rules 🤷‍♂️.



We used pre-release AutoGluon with some experimental features toggled on:
1. We dropped the original features in the L2 stackers, using only the stack features to train the models, this was shown to be better in CV: `ag_args_ensemble = {"use_orig_features": False}`
2. We hacked in stratified cross-validation splits by treating the numeric label column as the stratification column. This only worked because there were multiple instances of each label in the data. To do it properly you'd want to bin it, will add this in a future feature to AutoGluon. This helped avoid major distribution shifts between folds due to outliers.
3. We used an experimental 2024 version of AutoGluon's zeroshot-HPO portfolio which is slightly stronger.
4. We did not use any additional data. It didn't seem to help in CV and we felt it was too risky.
5. We disabled AutoGluon's ngram feature generation and text special feature generation, as we were not confident it led to an improvement (it was within noise, but removing them sped up model fitting):

```python
_feature_generator_kwargs = {
    "enable_text_special_features": False,
    "enable_text_ngram_features": False,
}
```

You can see our full write-up [here](https://www.kaggle.com/competitions/playground-series-s4e9/discussion/532635).

P.S: We were driven mad during the Grand Prix portion trying to figure out why our public LB scores were so poor despite how confident we were based on our internal CV, to the point of us even going over our code in-depth for any possible bugs. We are glad that it turns out we were right to trust our CV, and as @tilii7 has been saying in other posts, overfitting on the public LB can often be counterproductive. I'm guessing the main reason for low correlation in public and private is due to the density of outliers in the public vs private test splits.

Cheers,
Nick and Lennart ( @lennartpuruckerisg ) on behalf of the "AutoML Grandmasters"
