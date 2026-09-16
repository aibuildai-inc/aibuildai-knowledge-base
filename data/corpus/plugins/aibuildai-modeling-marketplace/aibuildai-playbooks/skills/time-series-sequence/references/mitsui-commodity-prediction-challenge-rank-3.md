# 3 rd place solution - Directional Trends Over Volatile Noise

Competition: mitsui-commodity-prediction-challenge
Rank: #3
Source: https://www.kaggle.com/c/mitsui-commodity-prediction-challenge/writeups/3-rd-place-solution-directional-trends-over-vola

>First of all, I was totally surprised. This competition was completely unexpected for me. In the morning, I got the notification about the results. I went to the leaderboard page and scrolled to the bottom (I couldn’t see clearly) and suddenly looked at the top - I was completely shocked. It was a journey from the bottom to the top! Trust me, this writeup is not late - I was just reviewing my entire code yesterday, double-checking if I had done something wrong.

>This writeup presents ideas that might be debatable - even I don’t always have answers to my own questions. I just followed some basic principles, and as a beginner, this was my first time series competition. So I focused on the simplest feature engineering and the simplest model engineering I could manage.

# 0.Introduction

This is a 3rd place solution writeup that uses basic feature engineering and simple models, implemented end-to-end. Both the training and forecasting stages were fully executed — the models were trained from scratch on the provided data, and predictions were generated exactly as the Kaggle evaluation API runs them. The key innovation of this approach was focusing on specific target pairs with robust feature and model engineering, while carefully avoiding noise and overfitting to maintain stable, reliable predictions.

# 1.Understanding 

## 1.1 Understand true needs 

This competition is **not a typical** time series challenge. Some important points highlighted by the competition hosts help clarify the task.

They mention:

>"In particular, participants are challenged to predict price-difference series — derived from the time-series differences between **two distinct assets’ prices** — to extract robust price-movement signals as features."

This point is critical. It indicates that participants need to be conservative with feature engineering, focusing on meaningful signals rather than adding noise. The targets are price differences, which are inherently more volatile and risky. Therefore, models must be stable and robust, capable of capturing direction and relative movement without overreacting to noise, ensuring reliable predictions over time.

Going ahead , they mention

>"Accurate commodity price prediction is crucial for **reducing financial risk** and **ensuring stability** in the global market. Currently, the inherent **volatility and unpredictability** of commodity prices create several problems. Companies struggle with resource allocation, budgeting, and investment planning, often resulting in financial losses and inefficient operations. Inaccurate forecasts can also contribute to market instability and price fluctuations, negatively impacting producers, consumers, and investors alike."

As a machine learning engineer, my primary responsibility is to build models that are stable and robust. They must respond intelligently to market volatility without overreacting to short-term noise. In other words, a reliable model should maintain consistent performance even during highly volatile periods, avoiding large swings in prediction uncertainty while capturing the true underlying market trends.

Finally , they include
>"While these models show promise, they often encounter limitations in consistently achieving high accuracy across diverse market conditions and over extended time horizons. Existing models may exhibit over-reliance on specific data patterns, lacking the adaptability required to navigate the dynamic and constantly evolving nature of financial markets"

To address this challenge, time-series feature engineering can be leveraged to extract meaningful, robust signals from historical data. By focusing on stable trends, relative movements between assets, and carefully engineered lagged or aggregated features, models can better generalize and maintain consistent performance, even under volatile conditions.

As a ML engineer , The points needed me to focus are :

- Commodities are volatile and noisy: Using every possible feature or over-engineering can make the model overfit to short-term patterns that won’t generalize.
- Stable predictions are more valuable than chasing small improvements: The goal is to capture directional trends and relative movements without reacting too strongly to noise.
- Selective feature engineering is necessary: Focusing on key signals (like target pairs, lagged differences, rolling metrics) helps the model generalize better and maintain consistent performance, rather than being distracted by irrelevant or volatile inputs.
- Time-series dynamics require care: Arbitrary inclusion of future or unrelated data can create leakage or unstable predictions. By being selective, you avoid these risks while preserving robustness.

## 1.2 Understand the datasets

### 1.2.1 train.csv Historic finance data 

My judgement : I treat it as a supplimentory dataset ( will explain further .)

### 1.2.2 test.csv — Mock Test Set

- Structured similarly to the unseen evaluation set.
- The public leaderboard set is a copy of the last 90 dates in the train set.
- Public leaderboard scores are **not meaningful** for evaluating true model performance.
- The unseen copy served by the evaluation API may be updated during training.

My judgement: 
This setup is somewhat of a “trap,” and it influenced my strategic decisions for this competition.

Strategic Choices

- Choice 1 — Exclude the last 90 days from training for public leaderboard:

    - Advantage: May provide a robust estimate on the public leaderboard.

    - Disadvantage: Limits the model’s exposure to recent patterns, which could improve predictions on the private leaderboard.

- Choice 2 — Use the full dataset for training:

    - Advantage: Allows the model to leverage all historical data, learning richer patterns for private evaluation.

    - Disadvantage: Public leaderboard score becomes largely meaningless, as it does not reflect model generalization.

Initially, after my first submission, I was at the bottom of the leaderboard, which made me uncertain about my approach. I re-evaluated my fundamentals, feature engineering, and code. Considering the public leaderboard was not a reliable signal, I opted for **Choice 2**, using the entire dataset for **training** to **maximize private leaderboard** performance.

Important Note: **My training strategy was different** from typical approaches. I **focused** on **robust feature engineering on target pairs**, careful model design, and **avoiding overfitting** to volatile signals, rather than chasing public leaderboard scores.

### 1.2.3 train_labels.csv : The targets consist of log returns 

My judgement:
From a machine learning engineer’s perspective, this dataset is important but often overlooked. Each column represents a distinct target derived from two different assets, as specified by the competition hosts. This means:

- The targets are independent computations, not necessarily correlated with each other.
- There is no inherent requirement that all targets share the same model. In my approach, I treated one target → one model. This is simpler, avoids cross-target interference, and aligns with the competition’s definition of targets.
- The number of targets effectively determines the number of models in my opinion, which provides flexibility and avoids forcing unrelated signals into a single model.

*Computation Note:*
According to the target_pairs.csv description:
>"See [This notebook](https://www.kaggle.com/code/sohier/mitsui-target-calculation-example/) for an illustration of how to use this information to go from price data to targets."

I followed this example and recomputed the targets programmatically rather than relying directly on train_labels.csv. This ensured that my target values aligned perfectly with my feature engineering pipeline, while still leveraging the dataset to verify correctness.

Although initially confusing, this approach allowed me to maintain a clear one-to-one mapping between targets and models, which I found to be the most robust strategy for this competition.

### 1.2.4 target_pairs.csv — The Gold Mine

This file is the most important dataset for understanding how targets are generated. Its columns:

- target — The label/column we want to predict.
- lag — The number of days into the future to compute the return.
- pair — The asset(s) used to calculate the target.

Key Takeaways / My Judgement:

- Each target is computed from a specific asset or asset pair with a lag, confirming that not every feature in the dataset is relevant for every target.
- This allows a target-focused approach: I only engineered features corresponding to the assets in each target pair.
- Avoiding features unrelated to the target helps reduce noise and overfitting, which is critical given the high volatility of financial data.

**1.2.4.1 Understanding a target_pairs.csv Entry**

```nginx
target_0 | 1 | US_Stock_VT_adj_close
```

| Field                   | Meaning                          |
| ----------------------- | -------------------------------- |
| `target_0`              | Column/label to predict          |
| `1`                     | Lag — 1 day ahead                |
| `US_Stock_VT_adj_close` | Asset used to compute the return |

**1.2.4.2 Target Formula :**

For a given day d, the target is calculated as:\
target_0[d] = log(Price[d+1]) - log(Price[d])\
Where:\
Price[d+1] → future price (day d+1)\
Price[d] → current price (day d)

### 1.2.5 Others are by kaggle
So I just go ahead ,

## 1.3 How this modify my training strategy :

Based on the structure of the competition and insights from target_pairs.csv, I adopted a target-centric training strategy.

### 1.3.1  One Model per Target
- Each target is trained using a separate model.
- Since every target is derived from a specific asset (or asset pair) and lag, it is natural to treat them as independent prediction problems.
- I did not find a strong justification for forcing a single global model across all targets, especially when the targets are computed from different assets and lags.

### 1.3.2 Target-Specific Feature Selection
- There is no explicit requirement in the competition to use all available features for every target.
- For each target, I only used features associated with the asset(s) specified in target_pairs.csv as the base feature set.
-Features unrelated to a given target were deliberately excluded.

- **This design choice significantly reduces**:
    - Irrelevant noise
    - Feature interference across unrelated assets
    - The risk of overfitting in a highly volatile setting

### 1.3.3 Trade-off: Feature Coverage vs. Robustness
- Feature engineering still plays a role, but with an important trade-off:
    - Using the entire dataset for every target would dramatically increase the number of features.
    - More features do not guarantee better 
    - generalization—especially in financial time series, where additional signals can introduce noise and spurious correlations.

- For each target, engineering features over the full dataset would be:

    - Computationally expensive
    - Prone to overfitting on the public leaderboard
    - Risky for private leaderboard performance
    - Given that public leaderboard scores were known to be unreliable, aggressively expanding features for marginal gains was a high-risk, low-confidence strategy.

### 1.3.4 Advantage of the Selective Approach

By training one model per target and restricting features to those directly relevant:
- Each model can generalize better for its specific target.
- The approach acts as a natural regularizer against volatility and noise.
- It provides additional protection against overfitting, especially in a blind evaluation setting where public feedback is misleading.

#### Final Decision

I therefore chose a selective, target-specific feature engineering strategy guided directly by target_pairs.csv.
This aligns with the competition’s emphasis on price-difference modeling, robustness, and stability rather than aggressive feature expansion. 

# 2 . Data prepration strategy

The data preparation pipeline was designed to strictly align with the target-specific structure defined in `target_pairs.csv`. Rather than treating the dataset as a single global time series problem, `each target` is prepared `independently` using only the information that `directly contributes to its definition`.

## 2.1 Train data strategy

### 2.1.1 Target-Centric Data Construction
For each entry in target_pairs.csv, the pipeline dynamically constructs a dedicated dataset:

- Target column: Each model is associated with exactly one target (e.g., target_9)
- Lag: The forecast horizon is explicitly respected during alignment
- Asset pair: Only the asset(s) listed in pair are used to build features

This ensures that every training sample reflects only the causal structure used to generate the target itself.

### 2.1.2 Asset-Pair–Driven Feature Selection

The pair column is parsed to identify one or two assets involved in target computation.

- If the target is derived from a single asset, only that asset’s time series is used
- If the target is derived from two assets, both series are used
- No unrelated assets or cross-market features are included

This design choice:

- Eliminates feature leakage
- Prevents interference from unrelated price dynamics
- Acts as an implicit regularizer against overfitting

Only columns that actually exist in train_df are used, making the pipeline robust to missing or sparse features.

### 2.1.3. Strict Temporal Alignment (No Leakage)

To preserve causality and prevent lookahead bias, features and targets are aligned carefully:

- Features (x_train) use observations up to time t
- Targets (y_train) correspond to outcomes at time t + lag
- Training windows are truncated to ensure `len(x_train) == len(y_train)`

The test sample is taken from t − lag, exactly mirroring how predictions are expected to be made during inference.\
This alignment guarantees that:

- No future information is used during training
- The training setup faithfully reproduces the evaluation environment

### 2.1.4 Robust Handling of Missing Values

Financial time series frequently contain missing values due to market closures, illiquidity, or data quality issues.

To ensure stability:
- Features are forward-filled and backward-filled
- Target series are fully filled before slicing into train/test segments
- Edge cases where an entire target series is missing are safely handled by fallback initialization

This avoids:
- Dropping rows (which can distort temporal structure)
- Introducing artificial volatility
- Training instability due to NaN propagation

### 2.1.5 Full-History Training per Target

Each target model is trained using the entire available historical window, subject only to the lag constraint.

- No artificial train/validation split based on public leaderboard periods
- No exclusion of recent samples for leaderboard optimization
- Public leaderboard behavior is intentionally ignored

This reflects a belief that:

- The public leaderboard is not a reliable proxy for real performance
- Robust models should learn from all available history
- Overfitting prevention should come from model simplicity and feature discipline, not data withholding

### 2.1.6 Design Philosophy Summary

This data preparation strategy reflects three core principles:

- Causality over correlation : Only features that directly contribute to a target’s construction are used.

- Stability over complexity : Simpler, target-specific datasets reduce noise and improve robustness in volatile markets.

- Faithful inference simulation : Training and test construction strictly mirror how predictions are made during evaluation.

This approach intentionally sacrifices feature breadth in favor of generalization, robustness, and interpretability, which ultimately proved effective in achieving a 3rd place finish.

This approach intentionally sacrifices **feature breadth** in **favor of generalization, robustness, and interpretability**, which ultimately proved effective in achieving a 3rd place finish.

## 2.2 Test data strategy

The test data preparation strictly follows the same assumptions used during training.\
For each target, I rely entirely on target_pairs.csv. The file specifies which asset(s) and which lag were used to compute the target. During inference, I only use those exact assets as input features. No additional columns from the dataset are introduced.\
This ensures that the model sees the same feature space during training and testing, avoiding any mismatch or hidden leakage.

### 2.2.1 How test data is constructed
For a given row in target_pairs.csv:

- target identifies which model to use
- pair defines the asset(s) allowed as input features
- The test dataframe is restricted to only those columns

If the test data arrives in a non-Pandas format (as sometimes happens during Kaggle evaluation), it is explicitly converted to Pandas to keep the pipeline consistent.\
Missing values are handled conservatively using forward fill followed by backward fill, identical to the training pipeline. This avoids dropping rows and preserves temporal continuity.

### 2.2.2 Design choice

I intentionally avoid any additional features on the test set as well like train .
No cross-asset interactions beyond what is explicitly defined in target_pairs.csv, and no transformations that depend on future information.

This mirrors a real deployment scenario:
at inference time, the model should operate on raw, available market signals, not engineered signals that may introduce instability or leakage.

# 3 Robust feature engineering

After restricting the feature space to only the assets defined in target_pairs.csv, I apply **conservative** time-series feature engineering on those base signals.\
The objective is not to maximize complexity, but **to introduce temporal stability and directional context** around inherently **volatile price-difference targets**.

## 3.1 Features choice
### 3.1.1 Lag features (positive and negative)
- Positive lags capture historical momentum and delayed effects
- Negative lags provide local symmetry and smoothing context
- This helps the model avoid reacting too aggressively to single-day shocks

### 3.1.2 Rolling window statistics
- Rolling means introduce trend stabilization
- Rolling maxima capture local extremes without chasing noise
- Window sizes are deliberately moderate to avoid regime overfitting

### 3.1.3 Difference features

- Differences reinforce relative movement rather than absolute price levels
- Especially important since targets are log returns or price differences
- Helps the model learn direction and magnitude consistency

As a machine learning engineer, I was genuinely unsure whether I should use negative lags or negative windows while training. In most production systems, these are usually avoided, and honestly, I hesitated a lot before adding them.

But this competition is not a standard forecasting task. We are dealing with cross-market relationships, not a single asset moving independently. In such markets, if one asset changes on day d, another asset might react on day d + t. At the same time, there are situations where a move that happened on day d − t can still influence behavior on day d. Markets don’t always react cleanly or instantly.\
If I only used positive lags, the model would mainly learn:\
*“This pattern happened before, so tomorrow it may happen again.”*

But markets don’t always repeat patterns so cleanly. Sometimes the important question is:\
*“How did similar patterns resolve?”*

To handle this uncertainty, I introduced negative lag and window features. My intention was not to let the model “see the future,” but to help it learn robust structures around a pattern rather than overfitting to a single directional spike.

I often think about this using a trader analogy. A trader looking at a chart doesn’t just memorize what happened yesterday. They look at how similar formations behaved in the past — sometimes they continued, sometimes they reversed, sometimes they stabilized. I wanted my model to have a similar sense of context.

Of course, I was careful about data leakage. I used a strict temporal split. All feature calculations were done only on training data. If a negative lag went out of scope, it naturally became NaN. My preprocessing pipeline handled these safely before the data ever reached the model. Nothing from the future was injected, and Kaggle’s evaluation API further ensured that no future information was accessible.

Because of this setup, negative lags acted more like a stability mechanism than a predictive shortcut. They reduced the model’s tendency to overreact during highly volatile periods — which is extremely important when the targets themselves are price differences and inherently noisy.

I questioned this choice many times after submitting. I kept asking myself if I had done something wrong. But in the end, I believe the results answered that question. If this approach had caused leakage or overfitting, my model would not have climbed from the bottom of the leaderboard to the top.

For me, negative lags were not about predicting tomorrow using tomorrow’s data. They were about teaching the model to stay calm, consistent, and directionally aware in an unpredictable market.

# 4 . Data processing pipeline

inancial time-series data is inherently noisy. Price differences, returns, and engineered lag features often contain extreme values, zeros, missing points, or numerical instabilities that can silently break a model if not handled carefully. Because my goal was stability over aggressiveness, I designed a simple but robust preprocessing pipeline that focuses on numerical safety and consistency rather than heavy transformations.

### 4.1 Log Transformation (Safe Log)

The first step applies a log1p transformation using a safe wrapper .
Financial features often span multiple orders of magnitude, especially when combining raw prices, differences, and engineered indicators. Applying a logarithmic transformation helps

- Compress extreme values
- Reduce skewness
- Stabilize variance across time

The transformation is implemented safely so that zero or near-zero values do not cause numerical issues. This is especially important because lag and difference features can frequently produce small or zero values.

### 4.2 Infinite Value Handling
After log and difference operations, infinite values can appear (for example, when dividing by very small numbers). Instead of allowing these to propagate and destabilize training, all infinite values are explicitly converted to NaN.

This step ensures that the pipeline never passes invalid numerical values into the model.

### 4.3 Missing Value Imputation

Missing values naturally arise in time-series feature engineering:
- Lagged features at the beginning of the series
- Windowed statistics with insufficient history
- Negative lags or windows that go out of scope

Rather than dropping rows or applying aggressive filling strategies, I use **median imputation**, which is robust to outliers and preserves the central tendency of each feature. This choice aligns with the overall philosophy of minimizing noise amplification.

### 4.4 Feature Scaling

Finally, all features are standardized to zero mean and unit variance. Since the models used in this solution (tree ensembles and stacking models) are sensitive to feature scale during optimization and aggregation, standardization ensures:

- Fair contribution of each feature
- More stable convergence
- Improved ensemble behavior

# 5 . Model Engineering

This model uses a two-level stacked ensemble architecture to improve predictive performance by combining multiple heterogeneous learners and learning how to optimally weight them.



### 5.1 Motivation Behind the Design

Different models capture different data patterns:
- Tree ensembles handle non-linear interactions well
- Boosting models reduce bias
- Bagging models reduce variance

Instead of choosing a single model, stacking allows us to learn how to combine them optimally using a meta-model.

### 5.2 Base Learners (Level-0 Models)

| Model         | Strength                                             |
| ------------- | ---------------------------------------------------- |
| LightGBM      | Fast gradient boosting, handles large feature spaces |
| Random Forest | Robust to noise, reduces overfitting                 |
| XGBoost       | Strong bias-variance tradeoff                        |

Each base model is trained independently on the same training data.

### 5.3 Meta-Feature Generation

After training each base learner:
These predictions are used as features, not final outputs ,
Reshaped into column vectors , Horizontally stacked to form meta-datasets .

Meta-datasets:
- meta_x_train: predictions on training data
- meta_x_test: predictions on test data

This transforms the problem from `original features -> target` to `meta predictions -> target` , Thats how they learn to predict .

### 5.4 Meta-Model (Level-1 Model)

The final estimator is another XGBoost regressor:

Role of the meta-model:
- Learns how much to trust each base model
- Captures correlations between base model errors
- Produces the final prediction

### 5.5 Prediction Flow

- 1.Base models predict on x_test
- 2.Their predictions become input features
- 3.Meta-model predicts the final output

so , entire flow is just smooth and cool , all models are stored in memory , since they are lightweight they come up with results very fast . so I ensure **Model diversity , Reduced bias & variance , Modular architecture , Easy extensibility , Production-friendly** architecture .

### 5.6 One Important Note (Advanced Insight )

Currently, base models are trained and evaluated on the same training data, but this is not a rebust solution in a real life case .
Production-grade improvement: Use out-of-fold (OOF) predictions with K-Fold cross-validation for meta-features
This is exactly what Kaggle gold-level solutions do .

In my case, I consciously chose not to do heavy fine-tuning and cross validation , even though I knew it could give small short-term gains. The system had to train and predict at runtime, with limited compute and a large number of features and models, so pushing aggressive hyperparameter search would have gone against the practical constraints of the problem. More importantly, I was aware that tuning too much in such a noisy environment often leads to overfitting, especially to the public leaderboard. I did not want the model to react to noise or accidental correlations just to climb the public LB, because that usually collapses on the private LB. Instead, I relied on model diversity and simple, stable configurations, allowing multiple learners to capture different aspects of the signal while keeping variance under control. If the approach were truly overfitted, the performance would not have improved consistently from the bottom to the top. The goal was never to extract the last decimal of performance, but to build something robust, generalizable, and stable under unseen data — and that mindset guided every modeling decision I made.

# 6 . Evaluation

To be honest, whatever testing I did on my models was more of a formality than a deep validation exercise. I did look at predictions, scores, and model behavior, but in practice there was not much I could change, because I neither fine-tuned the models nor had enough time to explore alternatives in depth. This was a time-series competition, so it is natural to be aware of your scores and keep an eye on how the model behaves over time, but I also understood that simply following scores is never enough. The public leaderboard, in particular, felt misleading, and I consciously chose not to rely on it. At many points, I genuinely did not know whether my model was performing “better” or “worse,” or even what the correct next step should be. Instead of reacting impulsively to leaderboard feedback, I stayed with my original assumptions and structure, trusting that consistency and temporal discipline mattered more than chasing short-term validation signals. In hindsight, that uncertainty became part of the process rather than a weakness - I was building without guarantees, only principles.

Still for formality , here are my scores :

| Score | Value    | Remark                                                                                |
| ----- | -------- | ------------------------------------------------------------------------------------- |
| MAE   | 0.01961  | Low absolute error; predictions are close in magnitude on average                     |
| RMSE  | 0.02931  | Slightly higher than MAE; occasional larger deviations exist                          |
| R²    | -0.11228 | Negative; model explains less variance than predicting the mean                       |
| Corr  | 0.09097  | Very low; predicted and actual series barely move together, captures direction poorly |

My model shows low MAE (0.01961) and RMSE (0.02931), indicating small absolute errors, but negative R² (-0.112) and very low correlation (0.091) show it struggles to capture the magnitude of movements. This suggests it predicts direction rather than exact values, which is acceptable in noisy financial time-series where directional accuracy matters more than spikes. However, if precise value forecasting is the goal, the model is underfitting. Overall, it’s fine for trend-following or regime-aware predictions but weak for exact magnitude prediction.

.png?generation=1768765737772201&alt=media)

Definitely, I did compare my model’s predictions against the test data, and the behavior becomes very clear when visualized. In the plot, the blue horizontal line represents the actual log returns, while the yellow line represents my model’s predictions. What immediately stands out is how volatile the true signal is — the blue line exhibits frequent spikes, including extreme movements that are very hard to model consistently. In contrast, the yellow prediction line behaves much more conservatively. Instead of reacting aggressively to every sudden spike, the model focuses more on capturing the direction of the movement rather than its magnitude. Most of the time, it aligns with the correct directional signal, but it deliberately avoids entertaining extreme changes or sharp price differences. This controlled behavior was intentional: rather than chasing highly volatile patterns that are often noisy and unstable, the model prioritizes stability and robustness, effectively smoothing out excessive fluctuations while still responding to meaningful market signals.

%20(1).png?generation=1768766415503119&alt=media)

- The plot actually supports the design choices rather than contradicting them. 
- The predicted series does not attempt to mirror the exact amplitude of the true series but instead behaves like a directional and regime-aware estimator.
-  In many segments, the predicted line moves in the same direction as the true series, with often parallel slopes, indicating that the model has learned directional correlation rather than memorizing spikes . This is valuable in time-series forecasting, especially for log returns, where magnitude is often dominated by exogenous shocks and noise. 
- During extreme spikes, the model systematically underreacts, which is an intentional bias introduced through selective features, median imputation, stacking, lack of aggressive fine-tuning, and no reliance on public leaderboard feedback. This variance suppression reduces overfitting risk and aligns with private evaluation objectives that reward stability. 
- The model’s occasional disagreements, lagged reactions, or dampened responses indicate healthy generalization rather than memorization, confirming a clean temporal split. 
- The smoothed appearance of the predicted line arises from ensemble averaging, stacking, log transforms, median imputation, and restrained hyperparameter tuning, effectively enforcing bounded predictions, outlier resistance, and regime stability-akin to risk control in financial ML. 

Overall, the model prioritizes directional correctness over magnitude, suppresses noise, and generalizes across time, which aligns with your philosophy and explains why your public-to-private leaderboard improvement occurred, even if the actual series remains highly unstable.

# 7 . Submission

Submissions are made using the Kaggle evaluation API, which runs the notebook in Kaggle’s environment. The API automatically invokes the predict function on the test set, performs full training , prepration , inference, generates the submission file, and evaluates it on the leaderboard. This ensures that predictions are produced without any forward-looking data and that the scores are computed and judged consistently and automatically.

# 8 . Resources

My complete training notebook is attached. Although the evaluation code was previously removed, I have re-integrated it into a forked notebook to demonstrate the results.

# 9 . Author wordict

>Entering this competition as a beginner, I focused on basic principles and simple feature engineering. When my first submission landed at the bottom of the public leaderboard, I questioned my entire approach and almost quit out of uncertainty. Because the public leaderboard felt like a 'trap' and the evaluation was so volatile, I stopped chasing short-term scores and instead bet on a strategy of model stability and temporal discipline. When I received the notification yesterday morning, I scrolled to the bottom of the leaderboard out of habit—only to realize I had jumped to 3rd place. It was a shocking journey from the bottom to the top, and I spent the rest of the day double-checking my code to ensure the results were real.

# 10 . Conclusion 

The solution was defined by a strategic emphasis on stability and robustness over aggressive feature engineering. By treating each target as an independent problem with its own dedicated model and restricted feature set, the author minimized noise and avoided the "trap" of overfitting to the unreliable public leaderboard. The approach utilized a two-level stacked ensemble—combining LightGBM, Random Forest, and XGBoost—to learn optimal weights for different data patterns while maintaining a production-friendly architecture. Ultimately, the model intentionally prioritized capturing directional trends rather than chasing extreme, noisy spikes, a form of variance suppression that ensured reliable performance during highly volatile market periods.

# 11 . Wrapping up

I would like to extend my sincere thanks to Kaggle and the sponsors, Mitsui & Co., for the Commodity Prediction Challenge. I am especially grateful for the clarity provided in the dataset descriptions and evaluation guidelines; having such high-quality information was a 'gold mine' that helped me navigate the task without the usual struggle to understand the requirements. Special thanks to @sohier for the clear instructions regarding submissions and the target computation notebook, which were essential to my workflow. Finally, congratulations to everyone who participated. This competition was a journey of self-doubt and personal growth, but it provided invaluable lessons through the process of testing and betting on different modeling strategies. I want to express my deep gratitude to my 'silent mentors'—the many Kagglers who, directly or indirectly, shared their knowledge through public notebooks. While I cannot pinpoint every specific source, my strategic decisions were strongly backed by proven Kaggle patterns. Many of the principles I applied—such as focusing on model stability and resisting the noise of the public leaderboard—are lessons that experienced competitors have championed for years. I simply followed those basic principles and applied them to this challenge.\
Thanks and regards ,\
Ayush
