# 15th place solution

Competition: optiver-trading-at-the-close
Rank: #15
Source: https://www.kaggle.com/c/optiver-trading-at-the-close/discussion/486086

I used an ensemble of GBTs with a single online component. Published my training code [here](https://github.com/osyuksel/kaggle-optiver-2024).

# Details

**Ensemble:**
- 3 x LGB, offline
- 4 x XGB
- 1 x LGB, online (re-trained every 5 days with a window of 60 days)

**Training:**
- Used n-fold cross validation by date_id with 5-day gaps between the folds.
- Hyperparameter selection: used public notebooks as a reference, optuna + manual tweaks as the final pick.

**Features that I haven't seen in public notebooks:**
- Revealed_target
- Intraday revealed_target using the wap from previous time steps
- Features based on [revealed stocks](https://www.kaggle.com/code/lognorm/de-anonymizing-stock-id):
	- sector_id
	- embeddings based on historical open/close/high/low data
- Group features
	- performance against the mean
	- performance against the sector

- At-the-money call price estimate with "expiry" at the end of auction
- Inferred price based on tick size


**Post processing:**
- Replaced "zero-sum" with subtraction by index-weighted mean targets.

**Other:**
- Dropped stock_id and relied on the embeddings based on historical performance and sector to reduce the effects of delisting and other drastic changes.


**Stuff that didn't work:**
- Most rolling features I introduced caused a drop in LB, so I omitted those such as:
	- Rolling cross-correlation
	- Rolling z-score
- Clustering stock_ids based on correlated wap, target
- Stock embeddings from neural networks


I got the biggest public LB boost from online learning, post-processing and hyperparameter tuning.


# Acknowledgements

I re-used feature engineering code from:
https://www.kaggle.com/code/meli19/lgb-kf-baseline
https://www.kaggle.com/code/zulqarnainali/explained-singel-model-optiver/notebook
https://www.kaggle.com/code/judith007/lb-5-3405-rapids-gpu-speeds-up-feature-engineer
https://www.kaggle.com/code/verracodeguacas/fold-cv

Thanks to Kaggle and Optiver for the interesting competition. This is the first time I saw a competition through and learned a lot during the process.
