# 3rd Place Solution in Three Words!

Competition: playground-series-s5e2
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s5e2/discussion/565653

**Determination – Creativity – Luck**

I'll leave it to you to decide what luck is

**Hello everyone!**

My name is Sebastian, and first of all, I would like to thank **Mr. Paweł Godula – narsil (jobs-in-data.com)** for spreading the Kaggle ideology in Poland and, most importantly, for playing a key role in helping me discover, after many years, what I will pursue in life and in which field I will become one of the best in the world.

**Chris Deotte** – Thank you for what you do and how you do it. **WooHoo!!!**

---

## Main Stages of the Solution

An essential part of my solution is the code that was shared by other competitors. The main characters are: @cdeotte, @masayakawamata, @mikhailnaumov and @vyacheslavbolotin .

1. **Feature Engineering**
    1. **Distance Features (`feh_distance`)**
        - For the raw datasets `train_raw` and `test_raw`, new variables are computed based on the distances between selected attributes (after mapping them to numerical values).
        - Columns such as `_2_1`, `_2_2`, ... `_5_1` are created, which represent the square roots of the sums of squared differences of selected (mapped) attributes, e.g., (x1 - x3)² + (x2 - x4)², etc.
        - This yields a dozen or so features expressing the “similarity/difference” within the backpack.
    2. **Creation of Combined (COMBO) Features**
        - For each original categorical column (e.g., `Brand`, `Style`, `Color`, etc.), a new feature is created by combining it with `Weight Capacity (kg)`.
        - Example: `new_col = Brand * 100 + Weight Capacity (kg)`.
        - This helps capture the interactions between the categories and the backpack’s carrying capacity.
    3. **Statistics from an External Dataset (`orig_price_*`)**
        - Based on an external dataset (`Noisy_Student_Bag_Price_Prediction_Dataset.csv`), the following values are calculated: `orig_price_mean`, `orig_price_std`, `orig_price_min`, `orig_price_max`, and `orig_price_median`.
        - The variable `orig_price_missing` is used to capture cases when a given combination did not appear in the external dataset.
    4. **Group Aggregations and Target Encoding**
        - **GPU-accelerated** grouping (using `cudf`) was employed for faster computation of statistics such as mean, std, min, max, median, count, and skew.
        - Multiple aggregations were performed, including grouping by `Weight Capacity (kg)` and the COMBO features.
        - Additionally, **Target Encoding** (implemented via `cuml.preprocessing.TargetEncoder`) was applied to the columns in `BASE_FEATURES`. As a result, each feature is replaced by the (smoothed) average `Price` within its category.
    5. **Missing Value Indicators (`_NaN_*`)**
        - For each of the 7 main categorical features, a missing indicator was defined (e.g., `_NaN_Brand` = 1 if `Brand` equals ‘Missing’).
        - Additionally, the column `_7_NaNs` sums up the number of missing values across all key fields.
2. **Autoencoder for Feature Extraction with a Supervised Layer**
    - An **autoencoder** (built with Keras + TensorFlow) was constructed and trained, featuring two main output branches:
        1. Reconstruction of the original numerical features (reconstructed = `Weight Capacity (kg)`),
        2. Prediction of the target value (`Price`) in the supervised branch.
    - Consequently, the hidden layer (`latent`) contains a representation of the numerical features that also “knows” how to assist in predicting the price.
    - The `latent` vector becomes a valuable feature appended to the final input of the models.
3. **Models and Ensembling**
    - Four **tree-based** models were utilized: LightGBM, XGBoost, XGBoost (with a different configuration), and CatBoost.
    - Finally, **stacking** was applied: the prediction vector from (LGBM, XGB, XGB2, CatBoost) is used as input to a **BayesianRidge** model, which finalizes the ensemble prediction.
    - The coefficients of the tree-based models and BayesianRidge are tuned to minimize the RMSE.

---

## Training and Prediction

- **10-fold KFold** for result stabilization,
- Each iteration generates Out-Of-Fold (OOF) predictions from 4 models (LGBM, XGB, XGB3, CatBoost),
- **Stacking** (using BayesianRidge),
- The final test prediction is the output of the Bayesian model applied to the stacked predictions.
- The submission that achieved the best result was carefully blended with several public submissions

---

**In Summary:**

The notebook makes intensive use of feature engineering – both *classical* (group aggregations, target encoding, handling missing values) and more *advanced* (autoencoder, distance features, external price data). The final ensembling combines several tree-based models and leverages BayesianRidge in the last layer, which further stabilizes the results and reduces RMSE.

---

**A Few Loose Thoughts – from a Newbie to Newbies**

I’ve done many things in life, but none of them had anything to do with IT. About three months ago, I started learning Python and SQL, and only two months ago did I discover Kaggle, so everything I write about my competition experiences might contain errors, and I might be mistaken.

I’ve played many games, and often the deciding factor was whether a game was challenging enough. Kaggle is the most challenging game I know, and the satisfaction from it is on a completely different level.

**A Few Loose Thoughts – from a Newbie to Newbies**, which ran through my mind at the finish of the **Backpack Prediction Challenge**. Perhaps tomorrow I’ll have different conclusions, so please don’t attach too much importance to them. The reflections from the end of the competition are mainly non-technical, as I’ll only be ready to tackle those in a few months.

- **Never spend too much time on the fundamental understanding of key aspects of the competition, even those that seem the simplest.** If I thought I understood something, I often discovered there was more to it. Even if we don’t find anything entirely new, delving into topics such as evaluation metrics or exploratory data analysis (EDA) can inspire ideas that seem to come from a completely different area.
- **Chasing the leaderboard score from the very beginning** might not be the best strategy.
- **Optimization** – it allows for rapid testing of ideas, and sometimes it can prove crucial at the end of the competition, when models start growing and merging.
- **The desire to win** serves as a fantastic learning method, or at least I hope so.
- **Theory vs. Practice** – it seems to me that training dozens, or even hundreds, of different models offers more insights than reading a few books. Concepts I repeatedly read about before but couldn’t grasp now seem obvious.
- **Creativity and unconventional ideas** are rewarded, but they should be applied at the right moment – for instance, in the middle of the competition when I couldn’t enhance the signal found by XGB using standard methods. After testing a dozen very strange solutions, I discovered a function that significantly improved the score. This function amplified the deviation from the median prediction—the greater the deviation, the higher the value. However, as I refined the model, I noticed that this idea ceased to work as the prediction quality improved.
- **Experimentation** – I believe that I will once again discover something interesting that no one else has thought of, but now I know it’s best to search and experiment later, rather than in the middle of the competition.
- **Hardware** – I performed the vast majority of computations on the free resources provided by Kaggle (my own hardware is much weaker). Although I spent most of the last few days of the competition on an ultimately unsuccessful attempt to optimize my automatic feature selector, I believe this is not a barrier. I’d bet it was possible to win even on a CPU, plus a great idea.
- **Overall Planning** – if I were to start this competition over again, I would do everything differently, focusing primarily on refining one model and methods for discovering new, strong features, and perhaps incorporating more models at the end for a better outcome.
- **Selection of Kaggle Materials** – it’s crucial to carefully select the materials we use. Gathering three or four excellent ideas from public notebooks, at least at the Playground stage, already provides a lot.
- **Chris Deotte** – if you come across his posts, just stick around longer.

---

What do you think? What experiences do you have?

---

Good luck to everyone in future competitions.

Thank you all for a wonderful competition.

See you on the trail and at the top of the LB.

---

Sebastian Kruszek
automatylicza@gmail.com

---

**Codziennie Silniejsi!**
