# S5E10 | 14th place solution

Competition: playground-series-s5e10
Rank: #14
Source: https://www.kaggle.com/c/playground-series-s5e10/writeups/s5e10-14th-place-solution

Final Scores: Public LB: 0.05536| Private LB: 0.05564

Approach: Simple average of 3 HC ensembles (pooling ~100 diverse models) + 1 Ridge ensemble. 
Key Insight: Trust your CV, but carefully assess also overfitting risk. With 70+ Hill Climbing ensemble variants, each with different model combinations and weights, I faced critical decision: trust the lowest CV score or hedge against overfitting?  

I had around 20+ submissions all scored 0.05539 on public LB with CV scores ranged from 0.055822 to 0.055833. All these submissions would ensure me private LB scores of 0.05562 or 0.05563. Lower CV scores below 0.055800 came from more aggressive HC optimization – ensembles from 30+ models with finely-tuned negative weights ranging from -0.30 to +0.65 allowing to correct correlated errors and improvements at the 1e-7 tolerance level.
 


My Hill Climbing ensembles were diverse and included: 
- XGBoost variants (with original data, residual-based, features-rich, with target encoding);
- LightGBM variants (Optuna-tuned, residual-based, with statistical features);
- CatBoost variants; 
- Neural Networks (TabM, RealMLP, FastAI, PyTorch);
- Tree ensembles (Random Forest, Extra Trees, YDF).

This created predictions that captured different aspects of the data, reducing correlation and improving generalization. 

From this competition I learned that lower CV doesn't always mean better - context and overfitting risk matters. The winning submissions were in my arsenal, but they seemed too conservative. 

Published final solution is simple average of the 4 ensembles (~40 lines of code) – 3 HC greedy ensembles (21-26 models each, with negative weights) and Ridge stacking ensemble with no post-processing or calibration. All OOF and test predictions were generated with 7-fold CV, saved as NumPy arrays, and combined using basic NumPy operations.

Thanks to the Kaggle competitors who published strong public notebooks (@cdeotte, @masayakawamata, @mahoganybuttstrings, @mikhailnaumov). The diversity in my solutions came from testing many different ideas and discussions posts published by @tilii7 and @siukeitin.
