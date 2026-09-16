# 6th place solution in 3 days - AutoGluon + AutoXGB

Competition: playground-series-s3e8
Rank: #6
Source: https://www.kaggle.com/c/playground-series-s3e8/discussion/392820

6th place with only 8 submissions and 3 days to deadline, a record for me 🎉 also started the first submission with 7th place, lucky entry as well :)

I used AutoGluon framework and XGBoost with Autoxgb framework in a weighted ensemble based on the CV scores!

For the AG I used two solutions, one trained 8 fold-1lv stacked and finally inference with LGBM,CatB,XGB,RF,NNTorch,NNFastAI and the other solution with psuedo labels from the first models. This took approx. 13h together.

Then the best ~24h HPO tuned XGB 8 fold model was ensemble with above,

Both the datasets where used, orginal and generated.

That's it!
