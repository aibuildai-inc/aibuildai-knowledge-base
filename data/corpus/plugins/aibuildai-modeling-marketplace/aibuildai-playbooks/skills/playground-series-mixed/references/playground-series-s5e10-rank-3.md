# 3rd Place: From Base to Stacking: A Multilevel Ensembling Solution

Competition: playground-series-s5e10
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s5e10/writeups/3rd-place-from-base-to-stacking-a-multilevel-ens

Congratulations to all the winners!

This was my very first Playground Competition, and I truly enjoyed the collaborative spirit in the forum.
A huge thank-you to everyone who contributed — the quality and variety of discussions and shared code were amazing!

My solution was fairly straightforward, and with a bit of luck, it got me to 3rd place!

## Level 1: Base Models

I started with five base models, inspired by several excellent public notebooks:

- TabM: thanks to @masayakawamata for showing [how powerful this model can be](https://www.kaggle.com/code/masayakawamata/s5e10-single-tabm-tuned)!
- TabM over residuals – credit to @cdeotte (as usual !) for teaching/reminding us of this [great technique](https://www.kaggle.com/code/cdeotte/xgb-boosting-over-residuals-cv-0-05595)! 
- XGBoost 
- LightGBM
- MLP using TabM framework

All models were trained using a Stratified on target 7-Fold Cross-Validation  with variations in models random seeds.
Interestingly, testing 7 Folds vs. 5 Folds and Stratified vs. Non-Stratified gave a small but consistent boost in OOF scores.

## Level 2: Stacking

At the second level, I trained a stacking neural network that used the predictions from the base models as input features.

## Level 3: Meta Model with YDF

For the third level, I trained a YDF model that used both: 

- The predictions from the previous stacking level

- The original base features

Thanks to @mikhailnaumov for sharing the [excellent YDF baseline](https://www.kaggle.com/code/mikhailnaumov/road-risk-single-ydf), which helped a lot in setting this up!


## Level 4: Final Ensemble

The final submission was a 50/50 blend of  second-level stacking model and third-level YDF model with CV: 0.05585 and LB: 0.05564


## Final touch

Finally, I merged my submission with best public LB  which gave a final 0.05563 LB
