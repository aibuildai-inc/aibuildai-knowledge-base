# 5th Place Solution (Full Write-Up)

Competition: equity-post-HCT-survival-predictions
Rank: #5
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/568339

(Note: I shared a [write-up on day one](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/566541), but didn't try to cover everything. I thought it would be nice to publicly share my documentation write-up using the template Kaggle provided for prize-winner's documentation)

# A. MODEL SUMMARY
# A1. Background on you/your team
- Competition Name: CIBMTR - Equity in post-HCT Survival Predictions
- Team Name: Robert Hatch
- Private Leaderboard Score: 0.69881
- Private Leaderboard Place: 5th
- Name: Robert Hatch
- LinkedIn: https://www.linkedin.com/in/robhatch/ 
# A3. Summary
- Model types used: LGB hand-tuned, CatBoost with params from AutoGluon and public notebooks, TabM hand-tuned based off public notebook, ODST Pairwise NN from public, and AutoGluon directly, which mainly also included XGBoost and some fairly simple NN models.
- 42 models used:
  - 20 Catboost (11 AG, 2 public notebooks, 7 personal variations)
  - 13 LightGBM (7 AG, 5 hand-tuned, 1 public notebook)
  - 4 XGBoost (4 AG)
  - 3 TabM (3 hand-tuned from public notebook starting point)
  - 1 NN (1 AG)
  - 1 ODST Pairwise NN (1 public notebook)
- Final Kaggle Notebook: https://www.kaggle.com/code/roberthatch/cibmtr-5th-place-official-submission 
- Targets used:
  - I used the A and B model split that was so successful for all the top 7+ competitors. With my version of the formula:
  - P = (a * b) - ((1 - a) * (S_RATIO))
  - A was a custom variant of predicting the ‘efs’ label. B was a custom variant of predicting the ‘efs_time’ rank for efs==1 case only.
  - Additionally, I trained against a 3d grid search of over 1000 variants of a NelsonAaler target. With variables for reducing the target by a percentage for efs==0 case, t[efs==0] *= y. Another for then shifting target by a flat value, t[efs==0] -= x. And finally a third variable for reducing the sample weights for efs==0 case. The best performing targets and ensemble targets were then trained against all model types.
- Feature engineering used:
  - GBDT: recalculate HLA sums, but make hla_nmlp_6_new and do NOT replace the original hla_nmlp_6 feature.
  - NNs: Remove all hla _6 and hla _8 sums other than the hla_nmlp_6.
  - NNs: Force all but 2 features as categorical.
  - Round the 2 numerical age features.
  - Rejected many features in my own testing. But for diversity sometimes included public notebook WITH original feature engineering in my larger ensemble.
- The final trimmed down models took around 6 hours of GPU and around 14 hours of CPU to train. However, that was really part of a much larger multi-stage ‘ensemble selection’ that included 1780 LGB models, several hundred AutoGluon models, as well as quite a few TabM and Catboost models against the main 6+ targets I used. So arguably there was around 15 hours GPU and around 60 hours CPU worth of models fed into the full ensemble to select the final models, even without counting the many other experiments and tuning that was done.
# A4. Features Selection / Engineering
- The most important feature was the target.
- Targets:
  - Main targets were A and B, see below.
  - P = (a * b) - ((1 - a) * (S_RATIO))
  - S_RATIO was a constant (~0.42456) that simply balanced the relative value of b=1.0 (The most risky entry) vs b=0.5 (average risk efs==1) vs an average efs==0, in terms of expected concordance wins and losses. So it was mathematically derived from the distribution of train data, and the nature of the concordance metric’s formula.
  - For the variable ‘a’, instead of predicting chance of efs==1 (vs efs==0) directly, I predicted the chance that efs_time was lower than the ‘tipping point’ where the fewest rows in train data were on the ‘wrong’ side of that point. This value for this dataset was EFS_SPLIT = 13.326. For this Classifier, I removed (censored) if both efs==0 & efs_time < EFS_SPLIT. That still left around 28600 rows. (99%). With more time, I would have preferred to ALSO train all the models against the more standard efs==1, and used both in my final ensemble if train CV of that ensemble showed improvement.
  - Similarly, for the variable ‘b’, I used ONLY both efs==1 AND efs_time < EFS_SPLIT, and converted to evenly distributed 0.03 to 0.97. Then took the logit of that value and trained RMSE regressors against that value. LGB was able to continue barely improving for thousands and thousands of iterations. Unusually, I used efs==0 observations in train but with sample_weight=0, which somehow seemed to help the model groupings. LGB was especially good with this target. Public LB score was lower than expected, so there was concern about overfitting. However, that ended up being a mirage and it’s possible LGB could’ve been pushed even further to accurately predict rank. Like with ‘a’, with more time, training a more traditional predictor of rank across all efs==1 might’ve been helpful, including the 2% with efs_time above the EFS_SPLIT tipping point.
  - In addition to the main targets, training a diverse set of 1d targets as proxies for the 2d ‘efs’ and ‘efs_time’ was also helpful. Key insights:
  - Using NelsonAalen, 1-cumulative_hazard was a good starting point and was always used unmodified with sample_weight = 1 for efs==1 datapoints. For efs==0, the question was how to shift the data to better predict the true risk OR to better ensemble with other targets and model variations. As starting point, instead of sample_weight = 1, I set sample_weight to the cumulative_hazard value. So a very early censored event would get a very low weight. To further refine efs==0, the three dimensions I used were:
  - Flat shift (X) of target as in public notebook(s). Increases separation to better distinguish efs==0.
  - Multiplier (Y) aka percentage of target. Intuition is that 50% of target would be the average score if efs==0 later became an efs==1 at equal likelihood with the overall population’s distribution.
  - Sample_weight (WM) lowered further by a multiplier aka percentage of original sample_weight.
  - In the end, the most helpful was 2 targets at opposite extremes, one with all efs==0 condensed near each other and shifted super far from efs==1, but efs==0 still with a low sample weight so it can learn both efs==1 ranking, but due to the huge gap, still have a strong emphasis on efs==0. Paired as an equal ensemble with one that didn’t shift efs==0 target at ALL, only reduced their weight. They were x=0, y=1.0, WM=0.35 and x=1.0, y=0.1, WM=0.4.
  - Another surprisingly effective pairing was X=0.7, y=0.6, WM=0.15 (and WM=0.1 was also good), this one was the best single target for pairing with the ODST Pairwise NN public notebook for whatever reason.
  - Cox Loss from Andrew’s public notebook was also used. The only change I made was a post-processing change. I ensemble all models with raw scores, I don’t ensemble their rankings. Since I use the raw model scores, unlike the public notebook I got this model from, it was important that I noticed that the Cox Loss model I was using had predictions in logit form. It ensembled MUCH better once I converted from logit to probabilities using the “expit” function.
  - The ODST pairwise notebook used concordance score directly, so another model that used both efs and efs_time as labels directly.
- For competition focus, I don’t look closely at feature selection or feature importance, other than looking at other people’s public EDA notebooks. Models are best at self-optimizing without overfitting. Overfitting can happen if making decisions based on feature selection or importance. Explainability was not my focus.
- How did you select features?
  - I AVOID selecting (as in removing), not helpful for GBDT based models. Adding can be good, but for this competition with small data and synthetic data wasn’t my focus and no one reported more than the smallest success with FE on the forums or in public Notebooks.
- Did you make any important feature transformations?
  - Not really. Hla_recalculate and rounding may have been slightly helpful.
  - For rounding, I rounded the clearly not-that-important donor age to the nearest even year. The intuition is that it could matter whether someone is 60 vs 30, but whether they’re 19 years and 2 months or 19 years and 7 months is much too tiny of a difference compared to the overall noise of the dataset. The goal is to balance “obviously doesn’t matter” with “let the model figure it out”. For Age at HCT I rounded to the first decimal place, so approximately to the nearest month. Again, maybe how many months old a person is could matter, but no way it matters whether person A was born 3 days before or after person B. That’s clearly an irrelevant detail.
# A5. Training Method(s)
- What training methods did you use?
  - LGB hand-tuned
  - Catboost from public hyperparameters, and just a little hand-tweaking of hyperparameters for my own models
  - TabM based on a public notebook, then spent considerable time updating and tuning. I used Cosine LR with Warm Restarts and increasing period (like the original paper suggested, with T=10 and doubling each restart).
Perhaps because the model has a lot of random variation and noise on later epochs, and “inspired” by the strangely high public LB score of the original notebook that accidentally did NOT have early stopping working, and because the warm restart paper suggested that early stopping is not necessary with this LR method, I tried both with and without early stopping, and ended up NOT using early stopping, but instead going a fixed number of epochs.
  - Advantages: CV is not inflated by picking the ‘best’ random variation. Different folds blend better for both OOF CV and test preds, they’re likely to be more similar as training epochs is the same. Might avoid overfitting CV if variation is mostly noise.
  - Disadvantage: potentially just a little less optimal than picking the best epoch especially if later epochs are worse score due to normal overtraining causing overfitting.
  - ODST Pairwise NN:
  - Although I tried a lot of things, I stuck with the same basic model and a different seed. The different seed was picked to be less crazy overfit (higher CV, lower LB), but also I speculated that the model was genuinely “smart” some runs and “failed to find a good optimum” other times, so picking a lucky seed might(?) be plain better than a not-so-lucky seed. So I balanced a fear of overfitting LB and perhaps even overfitting CV (because I took a good CV and second-best LB I got on a single fold) with the fear that if I took a more average seed or average over 3 runs, I could be reducing some way in which it was smart on public LB which might translate to private LB. So I didn’t necessarily think it was best, but I wanted to hedge by keeping it strong on public LB.
  - The model uses SWA (Stochastic Weight Averaging), which had a weird interaction with the public notebook’s “checkpoint”. If you loaded weights from the checkpoint, it seemed like you completely lost the SWA (I think?). So it ended up being another model where I trained it for a fixed number of epochs, without early stopping. Though on this model you then get a blend of the last 15 epochs through SWA.
  - Adding SWA to TabM would probably improve it by a small but significant margin. With the TabM interface it may or may not be trivial to implement. The SWA example was using PyTorch Lightning, and TabM had a lot of its own boilerplate code, so it didn’t look trivial in the least to port TabM to PyTorch Lightning, and I didn’t have any code example in front of me on doing SWA without PyTorch Lightning doing all the work under the hood.
  - AutoGluon v1.1.1 with 150 zeroshot model/hyperparameters presets.
  - I trained 112 (if CPU training only, then skip NNs) or 150 AutoGluon models on each of 5 folds against each of 6 different targets. I also ran target encoding against target A. (and target encoding model runs on other targets, but this was on final day and due to unknown pipeline issue, ran out of subs and abandoned testing adding these additional models to the ensemble).
  - I didn’t use stacking or ensembling via AutoGluon’s built in logic, I did ensembling myself.
- Did you ensemble the models?
  - Yes, I use my own variant of Greedy Ensemble Selection (GES), based on and very similar to AutoGluon’s internal ensembling technique. It might be very marginally less powerful than other ensembling techniques, but can reduce the number of models in the final ensemble and is less prone to overfitting out of the box, so it’s very effective when intentionally adding thousands of models as I did in this competition.
  - Not only was this technique used for around 50+ models I build and hundreds of AutoGluon models to pare down to the final set of models in the final solution. But it was also used with the 1780 LGB models with same LGB model and same features, just different target and sample weights, to select the best targets.
  - I had 3 model types: A, B, and normal. A and B I ensembled separately using the logit preds, and only after both were ensembled I converted to predictions from logits, then applied the formula to get the true AB preds. Normal preds was a third separate ensemble. After normal pred ensemble was done, I would ensemble it with the AB preds to get the final weighting of the ensemble of ensembles.
- If you did ensemble, how did you weight the different models?
  - Weighting with the GES method is take the best model with weight=1, then check every possible addition also with weight=1 (so 50/50%) and take the best, and keep going. I use train valid split and early stop when validation score stops improving. I decided that the competition metric was too noisy and used concordance index of all data as the metric to optimize. Using race groups drastically reduces the total pairwise pairs, and the standard deviation is arguably mainly controlled by irreducible noise. So the simple concordance index score was what I trusted throughout the competition, and was used for early stopping in many cases.
# A6. Interesting findings
- Besides using the A + B targets in the first place, which was the golden trick for the entire top 5, post-processing those A and B predictions was key for my final score.
- First, when I remember and bother to take the time - and it seemed very important for this split prediction case - I firmly believe in keeping predictions as logits as long as possible, and averaging and ensembling the logit values rather than the predictions. This more closely matches how GBDT models do their own internal additive ensembling, and is a mathematically appealing approach.
- So for post-processing, I found that multiplying and/or adding to one or both of the logits A and/or B for some reason greatly increased the CV score. But it didn’t increase public LB score, so for my final submission, one had no post-processing, and one had a conservative 1.5 multiplication to both A and B, with 0.25 +/- additive boost away from 0 for A only. This likely helps the model better rank and emphasize LOCAL concordance relative pairwise risk, as global pairwise score doesn’t matter for adjacent predictions, the ideal would be to swap adjacent predictions if and only if the lower datapoint is more likely to be the head-to-head higher risk winner, regardless of which one is more likely to win or lose the most GLOBAL pairwise rankings.
- Knowing that public LB was not super trustworthy, and CV ended up being very trustworthy for everyone, this can be pushed at least a little further with CV optimization compared with the conservative value I used.
# A7. Simple Features and Methods
- Simple Ensemble: 
  - Catboost was most important for A, and LGB for B.
  - There was a highly effective 4 model ensemble:
  - Catboost + TabM for A
  - LGB + Catboost for B
  - Post-processing (2.0 Confidence)
  - By itself, these 4 models get a score of 0.69812, within 99.9% of the full 42 model final ensemble’s score of 0.69881.

# A8. Model Execution Time
- How long does it take to train your model?
  - 20 hours (6 GPU, 14 CPU) for the 42 models. Or around 75 hours if including all the inputs to the ensemble selection model.
- How long does it take to generate predictions using your model?
  - Maybe 45 minutes for submission pipeline, but some of the time is for target encoding in Yunbase and my own target encoding.
- How long does it take to train the simplified model (referenced in section A6)?
  - About 4 hours.
- How long does it take to generate predictions from the simplified model?
  - Probably 5-10 minutes or less for submission pipeline.
# A9. References
- Papers:
  - Greedy Ensemble Selection:
  - CARUANA, Rich, NICULESCU-MIZIL, Alexandru, CREW, Geoff, et al. Ensemble selection from libraries of models. In: Proceedings of the twenty-first international conference on Machine learning. ACM, 2004. p. 18.
  - TabM:
  - TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling
Y Gorishniy, A Kotelnikov, A Babenko
arXiv preprint arXiv:2410.24210

Kaggle public sources and inspiration:
- AutoGluon zeroshot config: Taken and modified from AutoGluon team and https://github.com/AutoML-Grandmasters/Fourth-AutoML-Grand-Prix/blob/main/tabrepo_2024_custom.py 
- TabM initial notebook: @i2nfinit3y Having a starter notebook for TabM was a really big help, I was able to focus more on learning rate schemes and hyperparameters and batch sizes. https://www.kaggle.com/code/i2nfinit3y/cibmtr-tabm-nn-model-cv-0-6769-lb-0-685 
- TabM public work on top of initial notebook: Thanks for the inspiration: especially this notebook demonstrated grid search of shifted targets: @mtinti https://www.kaggle.com/code/mtinti/tabmhazard-na-from-i2nfinit3y 
- cdeotte baseline NN: https://www.kaggle.com/code/cdeotte/nn-mlp-baseline-cv-670-lb-676 
- @cdeotte metric explanation and discussion: https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/550003 
- Very important target creation visualization: @ambrosm  https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/550835 
- The incredible ODST Pairwise Loss NN model: Not only the public model of the competition, but very impressive custom engineering work to create it in the first place, and tune it to the level it performed at. @dreamingtree https://www.kaggle.com/code/dreamingtree/single-nn-with-pairwise-ranking-loss-0-689-lb 
- I used Andrew’s Notebook as the basis for my own notebook, though mine got modified a hundred times since then. And his Cox Loss (number 2) was a small but significant piece of my final ensemble. Thanks for the clear and modular baseline @andreasbis https://www.kaggle.com/code/andreasbis/cibmtr-eda-ensemble-model 
- Yunbase apparently didn’t do well on private LB, but as a small part of my final solution it seemed fine, I haven’t checked if it technically hurt my private LB or not. I used Catboost and (barely) LGB. But I didn’t convert any of the public models to A/B format, they made up a good chunk of my “normal” models. Big thanks for sharing the code and inspiring me to finally try my own target encoding as well :) @yunsuxiaozi  https://www.kaggle.com/code/yunsuxiaozi/cibmtr-yunbase
