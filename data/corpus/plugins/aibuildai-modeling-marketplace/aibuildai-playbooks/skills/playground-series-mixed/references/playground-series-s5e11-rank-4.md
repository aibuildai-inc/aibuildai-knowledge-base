# 4th Place Solution

Competition: playground-series-s5e11
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s5e11/writeups/4th-place-solution

Wow! I finally finished in the Top 5. More special because I had just turned 26... perfect timing for a gift. 😁

I want to thank Kaggle for continuously organizing these competitions. I’ve been competing for a year now, and I’ve learned a lot from the Playground Series. You get to learn from great, skilled people like @optimistix, @tilii7, @ravi20076, and the two geniuses @cdeotte and @siukeitin. My participation in the Playground Series was a major reason why I later won 3 silver medals. I genuinely recommend newcomers to start here.
 
Thanks to: @sagarnagpure1310, @mikhailnaumov, @masayakawamata, @yeoyunsianggeremie, @yekenot and others  for the public codes.

**My approach: Two Models + Huge Feature Engineering**

I used LGBM as the main workhorse along with a strong DNN. I started late (last 10 days), so I began from public notebooks and optimized upon.

First, I trained about 8 models (cv_xgb 0.9276, hgb 0.9273,, catboost 0.9272 .. etc), and ensembled them in multiple ways, but the gains were negligible. So I decided to invest more into FE.

To avoid throwing away my earlier ensemble effort, I used the ensemble predictions over the test set as pseudo-labels. This was very effective (+0.0004 on CV).

 
**Then each of the two models (LGBM, DNN) followed the same pipeline:**

1. Large and diverse engineered feature space (about 100 new features). Surprisingly,  n-way interactions didn’t help me 🤷‍♂️.
2. Pseudo-labeling based on ensemble predictions.
3. Stratified 5-fold CV training.
4. N_SEEDS = 5 independent training runs.
5. Differential-evolution hill-climbing for ensemble weighting (optimal convex combination of the 5 seeds using `scipy.optimize.differential_evolution`). This increased CV_DNN by +0.0004 over the best single seed, and CV_LGBM by +0.0002.
6. Final test predictions by applying the optimized seed weights.

For the final submission, I used a weighted average of LGBM and DNN.

| Model | CV     | Public LB |
| ----- | ------ | --------- |
| LGBM  | 0.9284 | 0.9279    |
| DNN   | 0.9274 | 0.927     |

Public LB: 0.928
Private LB: 0.92915

**Some notes/tips:**

* I noticed many people weren’t happy with DNN/CatBoost performance. For me they worked well, but you need to treat them differently. FE sometimes is everything; it determines how the architecture "sees" the data.
* Always trust your CV score but avoid leakage, especially with TE. I repeatedly see public notebooks getting this wrong.
* Monitor LB when the test data is large enough (0.2 × 254,569 is a good indicator), **but mix it with CV**.
* Add features gradually and check if CV improves. Dropping a batch of features all at once could cancel each other.
* (Personal view; no idea if others agree) LLMs are totally useless for creativity. They just memorize and spit it nicely. Don’t waste time asking them to "optimize" anything unless your baseline is terrible. Searching with them or writing code is fine, but that’s it.
* I strongly recommend reading @cdeotte’s previous solutions.
