# 18th Place: Many OOFs, Neural Networks over GBDTs, and Greedy Features

Competition: playground-series-s6e3
Rank: #18
Source: https://www.kaggle.com/c/playground-series-s6e3/writeups/18th-place-many-oofs-neural-networks-over-gbdts

I would like to thank Kaggle and the community for the valuable insights shared throughout the competition. My final solution is an ensemble of 140 models, stacked and combined using Ridge regression. Both Ridge and Hill Climbing proved to be the most effective ensembling methods. I also experimented with non linear stacking using models such as LightGBM, CatBoost, and various neural networks, but these consistently underperformed compared to linear approaches.

### Summary

- 140 model ensemble stacked with Ridge
- Linear stacking > nonlinear stacking
- Heavy reliance on feature diversity
- Minimal hyperparameter tuning
- Final rank: 18 / 4000+

---

### Best Models
Hill Climbing is an effective way to select models for an ensemble. However, I don't like that Hill Climbing drops some models that could potentially contribute some signals. 

| Model | Weight | Info |
| --- | --- |
| tabkit_2 | +0.5 | Feature Engineering from top public notebooks + excluding statistics from ORIG + (one_hot_cat_size, n_ens, embedding_size tuned) |
| tabm_adv | +0.2 | TabM with bi-tri grams + ORIG + Little to no tuning |
| tabkit_tuned | +0.1 | Tuned the model used in <code>tabm_adv</code> |
| base_xgb | +0.1 | max_bin=33000 + NUMS_AS_CATS + NO TE + NO ORIG | 
| xgb_extended | +0.1 | max_bin=33000 + NUMS_AS_CATS + TE + NO ORIG + Bitwise TE |
| tabtransformer_2 | +0.1 | Same configurations as <code>tabkit_2</code> |

---

### Feature Engineering 
I did not use a very large feature pool. At most, the feature set consisted of around 247 features, including those derived from top public notebooks as well as some of my own, along with bi and tri gram features. While the feature engineering was not particularly sophisticated, I include the key ideas below for completeness.

 - Discretized tenure as a categorical feature like <code>df['tenure_rounded_cat'] = df['tenure_rounded'].astype(str).astype('category')</code>
The cardinality is pretty small and with 33000 bins it contributed significantly in the training of most of my GBDT models.
 - Domain specific features like <code>df['isolated_cust'] = ((df['Partner'] == 'No') & (df['Dependents'] == 'No')).astype(int)</code>
 - Using the number of services as a categorical column for Target Encoding.
 - Digits of numerical features
 - Combination of digits and base features
 - KBinsDiscretizer (for linear models/NNs)
 - Using modular features like <code> df['MonthlyCharges']%7</code> and also experimenting with TE on their categorical counterpart.
 - Genetic features did not provide a significant improvement overall, but I experimented with them under the assumption that a larger feature pool can sometimes be beneficial. In my case, they improved a base Logistic Regression model from 0.90510 to 0.90699. While the gain is modest, it is still notable for a linear model. I also built a simple wrapper to reuse this approach across tabular competitions. Here is the function set I used for the <code>SymbolicTransformer</code>

```py
function_set = [
    'add', 'sub', 'mul', 
    'div', 'sqrt', 'log', 
    'abs', 'neg', 'inv', 
    'max', 'min', 'sin', 
    'cos', 'tan'
]
```

 - Bitwise TE as mentioned in <code>xgb_extended</code>'s key features. For binary columns, I tried creating XOR/OR/AND combinations hoping for some gain. TE on their categorical counterparts was beneficial.

```py
from itertools import combinations

BINARY_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines']
BINARY_FE_COLS = []

for col1, col2 in combinations(BINARY_COLS, 2):
    for op, symbol in [('xor', '^'), ('or', '|'), ('and', '&')]:
        col_name = f'{col1}_{op}_{col2}'
        fe_col_name = col_name + '_fe'

        train[col_name] = eval(f'train[col1] {symbol} train[col2]')
        test[col_name]  = eval(f'test[col1]  {symbol} test[col2]')
        
        train[f'{col1}+{col2}'] = train[col1].astype(str) + train[col2].astype(str)
        train[f'{col1}+{col2}'] = train[f'{col1}+{col2}'].astype("category")
        
        test[f'{col1}+{col2}'] = test[col1].astype(str) + test[col2].astype(str) 
        test[f'{col1}+{col2}'] = test[f'{col1}+{col2}'].astype("category")

        if op != "and":
            train[fe_col_name] = train[col_name].astype(str).astype('category')
            test[fe_col_name]  = test[col_name].astype(str).astype('category')
    
            BINARY_FE_COLS.append(fe_col_name)
```

 - I realized that <code>OneHotEncoder</code> significantly outperforms base TE for a stacking pipeline with LGB+XGB passed through the final estimator (LogisticRegression). I think the only thing that helped me climb so up in the leaderboard was this sort of diversity in the final ridge ensemble.

```py
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder

_CONFIGS: list = [
    [
        ('cats', OneHotEncoder(), list(CATS)),
        ('nums', StandardScaler(), list(NUMS))
    ],
    [
        ('cats', TargetEncoder(), list(CATS)), 
        ('nums', StandardScaler(), list(NUMS))
    ],
]
```
As mentioned above, <code>_CONFIGS[1]</code> outperformed <code>_CONFIGS[0]</code>

 - Other feature engineering i used were taken from top public notebooks and this [discussion](https://www.kaggle.com/competitions/playground-series-s6e3/discussion/679407).

---

### Base Models

I have trained several models on several feature configurations. Some of them are listed below. 

| Model | CV | LB |
| --- | --- |
| RealMLP | 0.91919 | 0.91787 |
| Bartz | 0.9164 | 0.91712 |
| XGBoost (Base) | 0.9152 | 0.91695 | 
| Catboost (Tuned) | 0.9182 | 0.91690 |
| LGBM-Dart (Base) | 0.917387 | 0.91534 |
| MLP (ALL_CATS) | 0.91823 | NA |
| CatBoost_L1_norm | 0.91871 | NA |
| HGBC | 0.91621 | NA |
| Logistic Regression | 0.91495 | 0.91367 |
| ExtraTrees | 0.91431 | NA |
| ExtraTrees Gini | 0.91399 | NA |
| Random Forest | 0.91437 | NA |
| RandomForest Gini | 0.91417 | NA |
| DeepFM | 0.91368 | NA |
| GNN | 0.91305 | NA |
| Tabnet | 0.91191 | NA |

---

### What Did Not Work
 - Pseudo labeling 
 - Categorical Interactions that go beyond bi/tri.
 - Autoencoders
 - Nonlinear stacking

---

### Closing
I went to sleep ranked around 150 and woke up at 18 :p

Averaging weak OOFs turned out to be a mistake, and selecting stronger OOF predictions would likely have improved the final result further. I did not rely on extensive hyperparameter tuning or tools like Optuna. In future competitions, I would focus more on careful OOF selection and deeper tuning. Thanks to the community for the insights and shared ideas.

Good Luck.
