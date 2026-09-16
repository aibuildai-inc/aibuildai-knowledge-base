# 5th Place Solution - (XGB+LGBM+TabM)*5SEEDs+AG

Competition: playground-series-s5e11
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s5e11/writeups/5th-place-solution-xgb-lgbm-tabm5seeds-ag

First of all, huge congratulations to @mahoganybuttstrings on the 1st place!
Also, thank you to everyone for the insightful discussions and notebooks throughout this month. I learned a great deal from interacting with all of you.

In this competition, I felt that **Feature Engineering** played a more significant role than anything else. My solution consists of a single effective feature set used to train XGBoost, LightGBM, and TabM (5 seeds each), blended with AutoGluon's OOF/test predictions using Ridge Regression.

# Score Overview

| Model | CV | Public LB | Private LB |
| :--- | :--- | :--- | :--- |
| LGBM | 0.92798 | 0.92795 | 0.92908 |
| XGB | 0.92790 | 0.92799 | 0.92908 |
| TabM | 0.92785 | 0.92779 | 0.92888 |
| Ridge Stack | 0.92803 | 0.92800 | 0.92911 |
| **+AG (Final)** | **0.92805** | **0.92800** | **0.92912** |

# Feature Engineering
Most of my FE ideas are based on @cdeotte's past solutions. Thanks!

1.  **DIGIT Features:** Created new features extracting digits from all positions of numerical columns.
2.  **ROUND/BIN Features:** Created features by rounding or binning numerical columns.
3.  **Interaction Features:** In my public notebook, I only created interaction terms from the original features (BASE). However, for the final solution, I added 2-way, 3-way, and 4-way interactions within `DIGIT_CATS` and within digits derived from the same numerical feature.
4.  **ORIG Features:** Introduced Mean Encoding and Count Encoding to capture signals from the original data.

# Feature Selection
Since the generated features likely contained noise, I performed a lightweight feature selection.
* **Method:** I added features one by one to the BASE features. If the accuracy improved over the baseline, I selected it; otherwise, I discarded it.
* **Validation:** Due to computational costs, I used `train_test_split` (test_size=0.2) instead of K-Fold for this selection process.

While this method might still retain some redundant features, I did not perform further selection due to time constraints and computational resources. I assumed that GBDTs are robust enough to handle a certain amount of noise.
(I also tried optimizing the `top-n` feature importance using Optuna, but I couldn't get it to work effectively.)

# Hyperparameter Tuning with Optuna
I performed parameter tuning using Optuna for XGB, LGBM, and TabM.
* **XGB & LGBM:** 5-fold CV
* **TabM:** `train_test_split` (test_size=0.2)

# Seed Averaging & Ridge Stacking
I performed **5-seed averaging** for XGB, LGBM, and TabM.
The main reason was to stabilize the prediction and validation scores. Since the validation score fluctuated slightly just by changing seeds, I decided not to rely on single-seed validation.

I used the OOF and test predictions to calculate the final prediction via **Ridge Regression**.
* **Why Ridge?** I didn't spend enough time creating diverse OOFs that would benefit from non-linear stackers like MLP or XGBoost. My time was focused on building one strong feature set. In such cases, using XGB or MLP for stacking often leads to overfitting. Therefore, I chose a linear model like Ridge to prioritize stability over aggressive accuracy gains.

# Data Generator Notebook
Following @greysky's approach from the Podcast competition, I created a Data Generator Notebook.
In my implementation, since I wanted to calculate `test_pred` using a 5-fold average, I needed to create datasets for each fold (especially for Target Encoding).
As the number of TE features increased, the time required for TE per fold also increased. By pre-generating and simply loading the data, I significantly improved the efficiency of Optuna tuning and multi-seed training.

* **Note on Full Data:** I did not train on the FULL DATA because, in my early experiments, the Public Score was consistently lower than the fold average. Since CV-LB correlation was excellent in this competition, I decided to trust my CV and avoided full data training.

# Notebook Links
I have published almost all the code used for my submission, including the TabM implementation. I hope this helps!

* **Data Generator:** [s5e11-data-generator](https://www.kaggle.com/code/masayakawamata/s5e11-data-generator)
    * *Used Version 4.*
* **Ridge:** [s5e11-ridge-ensemble](https://www.kaggle.com/code/masayakawamata/s5e11-ridge-ensemble?scriptVersionId=278279429)
    * *(Note: The CV score in the TabM filename inside the code differs from the table above due to a minor oversight where I put the score of the 5th seed in the filename.)*
* **LGBM:** [s5e11-single-lgbm-tuned](https://www.kaggle.com/code/masayakawamata/s5e11-single-lgbm-tuned)
* **XGB:** [s5e11-single-xgb-tuned](https://www.kaggle.com/code/masayakawamata/s5e11-single-xgb-tuned)
* **TabM:** [s5e11-single-tabm-tuned](https://www.kaggle.com/code/masayakawamata/s5e11-single-tabm-tuned)
    * *I haven't verified if TabM runs successfully on Kaggle kernels. It might hit OOM or the 12h timeout.*

**Discarded Models:** I also implemented CatBoost, xRFM, and RealMLP with the same feature set. However, they didn't match the accuracy of LGBM/XGB/TabM and didn't contribute much to the ensemble, so they were excluded. (Though xRFM and RealMLP might have potential if tuned more thoroughly).

Thank you for reading. **Happy Kaggling and Trust your CV!**
