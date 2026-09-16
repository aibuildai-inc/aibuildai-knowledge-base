# 6th Place Solution

Competition: neurips-open-polymer-prediction-2025
Rank: #6
Source: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/6th-place-solution

**Overview**

My solution is a fork of a public XGB-based notebook with Tg predictions shifted (+30). While I had more aggressive Tg shifts submitted, I wasn't sure how to gain confidence that this would be a winning strategy in the end and limited the increase. I felt strongly that it was likely to have some impact in the private test set, so I picked one notebook with a mild shift and a notebook without any Tg shift. Regrettably, while I found it a lot of fun to experiment with neutral nets, I did not find a path that was strongly competitive with gbm-based approaches in the time I had available for this challenge.

I've published the notebook fork I used at [https://www.kaggle.com/code/rob1080ti/tg-modified-fork-of-lb0-65-xgb-5-fold](https://www.kaggle.com/code/rob1080ti/tg-modified-fork-of-lb0-65-xgb-5-fold)

**tl;dr**
`gbdt_predictions_df['Tg'] += (303-273)*1`

**Unsuccessful attempts**

- Small TabTransformer

When the [LEAP - Atmospheric Physics using AI (ClimSim)
](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/overview) competition ended last year, I found it very interesting that I could shrink the winning network design and still keep roughly the same results. So, I thought it would be interesting to shrink the embedding dimension (6), number of heads (6), number of encoders (6), and increase the feed-forward hidden dimension (768).

While this led to a strong improvement: LB 0.089 PB 0.103 -> LB 0.077 PB 0.094; I didn't think it was competitive with XGBoost or LightGBM, etc.

- Data augmentation with LAMMPS

This is more of my own fault; I decided fairly quickly that I didn't have the time to really invest in this.
