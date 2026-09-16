# 13th Place Solution Part II

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #13
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/161998

First of all, I want to thank all my wonderful teammates. Without your effort, we won't be able to make this happen. @euclidean @nullrecurrent @strider1125 @sherryli94 

Congratulations to all the medal winners, it's been a tough competition for all of us.

For the overview of our solution, you can refer to Part I written by @strider1125 here:
https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/161974

I will talk more about some details of our solution and where they came from:
- CV
- Stacking / Blending
- Exotic Experiments
- Things I Wish We Tried
- Final Thoughts


# CV
- We started the competition using the validation set as purely holdout. In this phase, we reached around 9380 public.
- Later it became clear that we need to use the validation set and we first tried to do group-3fold by language, which gave significantly underestimated CV score and we reached 939x public.
- Then the different distribution among languages seemed strange when building OOF predictions and we decided to switch to random 4-fold split, which helped improve the stability of OOF predictions. 
- Finally, we settled for 5fold with (language x toxic) stratified and used this CV setup until the end of the competition.
- It seems that @christofhenkel found some very reliable and elegant CV setup:
https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160980

# Stacking / Blending
- The reason we spent so much time on CV was that the CV score being so unstable, especially in stacking/blending.
- Stacking / Blending methods we tried
HillClimb / Greedy
https://www.kaggle.com/hhstrand/hillclimb-ensembling/
Unintended 3rd place @sakami 's blending method
https://github.com/sakami0000/kaggle_jigsaw/blob/master/compute_blending_weights.py
Unintended 9th place @khyeh0719 's stacking using ExtraTreeClassifier
https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100530
Toxic (and this time!) 1st place @leecming 's discussion
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557
GM @kazanova 's StackNet
https://github.com/h2oai/pystacknet
- In terms of the second level, we mainly tried logistic regression, LGBM, and ExtraTreeClassifier. LGBM stacking brought us back to silver-zone with 9464 at some point, but then it seemed too overfitting. Eventually, we chose ExtraTreeClassifier as it was more stable and gave a better lb score. There is also some intuition here that our models seemed to have high variance and ExtraTreeClassifier is a good candidate to reduce that variance while keeping the predictive power of all base models.
- Our file structures allowed for a good enough stacking workflow (i.e. saving all OOF predictions and test predictions per model), but I have to say StackNet works very well and I highly recommend people to give it a try.
- The most painful part of stacking/blending is that every time you get a higher CV score from parameter tuning or feature selection, thinking you are going to save the world, the LB score will beat your ass. Eventually, after tons of failures, I realized that doing blind parameter optimization and feature selection under such an unstable CV setup was not going to help. 
- The consistent trend we observed was that whenever we add more diverse models to the stack it improved our LB score and the CV-LB gap reduced, so we were pretty confident that our solution would be stable enough to survive the shake-up (but you might notice I'm saying this after getting the medal :-).
- Although our CV was not stable if we look at each individual score/experiment, on average our distribution of models has a good correlation with lb score and since we are using something like ExtraTreeClassifier, we can to some extent guarantee that it converges in probability to our LB score (I usually refer to this as my faith to Law of Large Numbers). Then it became clear that we should create more diverse models that have good scores (that's where the 1900 oof predictions came from).

# Exotic Experiments

- Averaging multiple XLMR's weights (similar to SWA), this gave our best single model (lb 9471).
- Adding the language tag (tr/fr/en/es etc.) as the first token after CLS in XLMR. So your tokens would look like [CLS] + [tr/fr/en/es...] + [other sentence tokens], this simple trick helped create one of our best single models. One could also do it in the way 2nd place winner @xiwuhan suggested: https://www.kaggle.com/xiwuhan/jmtc-fine-tune
- Random masking of tokens for data augmentation. 
- Using aux columns in stacking, quite a few aux predictions were selected when doing blending(forward selection) and we believe improved stacking diversity as well.
- Null permutation and RFECV for stacking feature selection (didn't help).
- Adding more 2nd level OOF in StackNet (didn't help much).
- Pseudo Labeling in stacking (didn't help).
- Roberta large on English (score low, probably that's why we missed the monolingual models most other top teams touched: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160862)

# Things I Wish We Tried

- Freeze embedding layers / different learning rate per layer
- PL with soft labels and more iterations
- Putting monolingual models into our stacking
- Trying GPT-2 / XLNET etc.
- Post-processing mentioned by other teams (adjusting average toxicity per language)

# Final Thoughts

- It's not easy to get a gold medal. I read the winning solutions from Unintended so many times that I almost remember who used which trick/technique. All those discussions were super helpful and I learned a lot from them. And now we finally get the chance to write our own gold solution.
- Keep trying new things, keep doing new experiments, come up with new hypotheses, and validate them with a proper CV/LB. Whenever I get stuck I go back reading winning solutions, and they always gave me some new thoughts.
- I love this game!


