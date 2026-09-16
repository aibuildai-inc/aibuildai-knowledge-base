# 19th Place Solution: 108 OOFs blending with RidgeCV

Competition: playground-series-s5e8
Rank: #19
Source: https://www.kaggle.com/c/playground-series-s5e8/writeups/19th-place-solution-108-oofs-blending-with-ridgecv

I'm thrilled to share our solution for the recent Kaggle Playground Series competition, **"Binary Classification with a Bank Dataset" (Season 5, Episode 8)**, where we managed to secure a **final position of 19th** out of thousands of participants.

This competition was a classic tabular data binary classification problem. The key to our success wasn't a single complex model, but a meticulously crafted **blending ensemble of 108 diverse models**, regularized using RidgeCV. Here’s a breakdown of our approach.

#### **The Core Strategy: Diversified Blending**

Our final solution was a weighted average (blend) of predictions from many individual models. Instead of a simple average, we used **Ridge Cross-Validation (RidgeCV)** to learn the optimal weights for combining these models, effectively allowing the algorithm to decide which models to trust more.

The entire architecture of our solution can be visualized as follows:



#### **How We Engineered Diversity**

The fundamental rule of a successful ensemble is **diversity**. We aimed to have models that made mistakes, but made them *differently*. A model that is wrong in a unique way can have its error corrected by the consensus of others. We generated this diversity in several key ways:

1.  **Feature Engineering:** We created multiple datasets with different sets of engineered features. Some included the original features heavily, while others relied more on new creations.
2.  **Target Encoding:** This was a major source of diversity. We applied a wide variety of encoding strategies (mean, median, standard deviation, variance, min, max, etc.) on categorical features, each creating a slightly different representation of the data for the models to learn from.
3.  **Model Types & Hyperparameters:**
    *   **~70% Gradient Boosting (XGBoost, LightGBM, CatBoost):** These were our workhorses, providing the strongest individual scores. We varied their hyperparameters, depths, learning rates, and random seeds extensively.
    *   **~30% Neural Networks (TensorFlow/Keras):** While their individual performance was lower than the best GBDT models, they were crucial. Their error patterns were highly complementary to the tree-based models, providing the blend with a different "perspective" and boosting our score significantly.
4.  **Randomness:** Simply changing the `random_state` for data splitting or model initialization produced meaningfully different models that could be added to the ensemble.

#### **The Final Stretch: A Linear Intuition**

In the final hours, we experimented by adding a simple **Linear Regression model** (which had poor performance on its own) to the RidgeCV blend. Interestingly, it **improved our local CV score by +0.000016**. However, on the public leaderboard, it caused a **drop of -0.00017**.

This was a critical lesson in overfitting and the importance of trusting the consistency between CV and the public LB. We decided to submit two solutions:
*   **Final Submission (19th Place):** Blending *without* the Linear Regression model.
    *   **CV Score:** 0.977242 | **Public LB:** 0.97784 | **Private LB:** 0.97744
*   **Alternative Submission (~21st Place):** Blending *with* the Linear Regression model.
    *   **CV Score:** 0.977258 | **Public LB:** 0.97767 | **Private LB:** 0.97740

#### **Key Takeaways & Lessons Learned**

**What worked well:**
*   **Vigilance against Data Leakage:** We were meticulous, especially with our target encoding, ensuring it was correctly calculated within each cross-validation fold to avoid leakage. This built trust in our CV scores.
*   **CV-LB Correlation:** Our local CV score was a reliable indicator of public leaderboard performance. When one moved, the other followed in the same direction. This gave us confidence in our experimentation.
*   **Community Engagement:** Actively learning from the competition's Discussions thread was invaluable. We incorporated many new techniques shared by others throughout the competition.

**What to improve next time:**
*   **Strict CV Scheme Adherence:** Some of the 108 models incorporated from other contributors used different validation splits. While their OOF predictions were valuable, this inconsistency prevented us from exploring more advanced techniques like stacking. Next time, I will prioritize retraining all models under a unified CV scheme.
*   **Model Management:** With over 100 models, organization becomes a challenge. I need a better system for tracking them—more descriptive naming conventions and perhaps a dedicated tool or spreadsheet to log each model's parameters, features, and performance. **(I'd love to hear how others manage this! Please share your tips in the comments.)**

This was an incredibly fun and educational competition. A huge thanks to Kaggle for hosting the Playground Series and to all the participants who shared their insights and fostered a collaborative environment, especially @cdeotte and @tilii7.

Congratulations to @optimistix and the other winners!
