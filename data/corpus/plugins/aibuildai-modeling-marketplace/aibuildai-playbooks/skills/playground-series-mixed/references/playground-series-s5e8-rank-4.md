# 4th Place Solution

Competition: playground-series-s5e8
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s5e8/writeups/4th-place-solution

## Introduction

This month's Playground competition was incredibly fun! The CV-LB relation was solid, and a fantastic group of competitors gathered, making it enjoyable from the beginning of the month to the very end.

I want to express my deepest gratitude to the Kaggle organizers and everyone who contributed useful discussions and notebooks.

In particular, the posts and comments from @tilii7, @mahoganybuttstrings, @optimistix, @cdeotte, and @siukeitin were extremely informative and inspiring. Thank you all very much.

## TL;DR

-   **L3-Hill Climb:** A final blend using 4 stacking models.
-   **L2 Models:** Four models in total—two XGBoost and two Neural Networks. Each model type was trained in two ways: one using pseudo-labeled data as new rows, and another using it to create new columns.
-   **L1 Models:** A base layer of over 200 Out-of-Fold (OOF) predictions.
    -   ~100 OOFs were generated using AutoGluon on various feature-engineered datasets, keeping only the OOF and test predictions.
    -   ~100 OOFs were from a collection of my own custom-built models.

---

[MAIN]

## Timeline - A Month's Journey

For the first 20 days of the competition, I dedicated my time to experimentation. I would generate predictions, save the OOF and test predictions, and then feed them into an XGBoost model without much deep thought. I repeated this cycle until my Public Score reached around 0.9781.

With 10 days remaining, I decided to reassess my approach. I ran experiments on areas I was curious about (e.g., whether fold-averaging was better than refitting). In the last 7 days, I finalized my strategy, made some fine-tuning adjustments, and improved my score from 0.97808 to **0.97828**.

## L1 - 200+ OOFs!!

[L1]

I wanted to create as many diverse models as possible with minimal effort, so I turned to AutoGluon. As mentioned in the official AutoGluon documentation, spending time on feature engineering (FE) is the most effective way to use the library. I created many variations of feature sets and trained an AutoGluon instance on each, which allowed me to easily generate a large number of OOF predictions. The CV scores for these ranged from **0.95 to 0.9758**.

Since AutoGluon is primarily focused on ensembling and not parameter tuning, I also used FLAML separately to create optimized LGBM models for each feature set. One of the notebooks for this is [here](https://www.kaggle.com/code/masayakawamata/s5e8-diverse-lgbm?scriptVersionId=257214221). In addition to this, I created two or three more diverse LGBM models for generating OOFs. These models had CV scores between **0.971 and 0.976**.

Furthermore, as demonstrated in @mahoganybuttstrings's excellent notebook, creating 2-way interaction features and applying Target/Frequency Encoding was extremely effective. I wanted to extend this to 3-way or higher interactions, but I ran into memory issues in the Kaggle Notebook environment.

To overcome this, I used Optuna to search for up to 150 effective 2-way to 6-way feature combinations. Due to the 12-hour runtime limit, a full search was not possible, so I ran the search process multiple times (about 5) to compensate. This approach yielded CV scores between **0.973 and 0.9755**.

I also experimented with other, more unconventional models, mostly for fun. These included '1D-CNNs' and models like 'NN-SVC_Head' and 'NN-XGB_Head', which were trained on features extracted from a base neural network. These models were trained on the 2-way TE/CE features and all achieved a CV of around **0.974**. However, they didn't provide a significant boost to the final blend this time.

Discussions I found helpful:
* **1D-CNN:** [https://www.kaggle.com/competitions/lish-moa/writeups/tmp-2nd-place-solution-with-1d-cnn-private-lb-0-01](https://www.kaggle.com/competitions/lish-moa/writeups/tmp-2nd-place-solution-with-1d-cnn-private-lb-0-01)
* **NN-... Head:** [https://www.kaggle.com/competitions/petfinder-pawpularity-score/discussion/276724](https://www.kaggle.com/competitions/petfinder-pawpularity-score/discussion/276724)

Ultimately, I implemented every idea I could think of, including things I had tried in the past and interesting solutions I had seen in other competitions. For all L1 models, I used the **fold-average** of predictions from multiple seeds within each fold, rather than refitting on the entire dataset.

## L2 - XGB+NN - Pseudo Data as Rows/Cols

[L2]

I trained XGBoost and Neural Network models using the 200+ OOF predictions generated in L1.

During this stage, I added pseudo-labeled data from the test set where `test_pred > 0.99` or `test_pred < 0.01`. The predictions used for this pseudo-labeling came from a weighted average of all L1 models, with the weights calculated via a Hill Climb algorithm. I considered using the final model's test predictions, but I opted for the weighted L1 average to reduce the risk of overfitting and high model bias, hoping it would better absorb the quirks of individual models.

I used this pseudo data in two different ways:
1.  **As Rows:** For each fold, I simply concatenated the pseudo-labeled data as new rows.
2.  **As Columns:** I first trained a k-NN model on the data augmented with pseudo-labeled rows. Then, I used this k-NN to generate new features for the original data based on its nearest neighbors. To prevent leakage, I used nested k-folds, similar to how one would implement Target Encoding.

I tried to create a diagram for this, but it didn't turn out well. For the k-NN column approach, please refer to this [Notebook](https://www.kaggle.com/code/masayakawamata/s5e8-l2-pseudoknncolgrn).

The idea for using pseudo data came from @cdeotte's write-up here: [https://www.kaggle.com/competitions/playground-series-s5e6/writeups/chris-deotte-1st-place-fast-gpu-experimentation-wi](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/chris-deotte-1st-place-fast-gpu-experimentation-wi).

| Model              | CV         | Public LB | Private LB |
| ------------------ | ---------- | --------- | ---------- |
| L2-pseudoColXGB    | 0.977431   | 0.97816   | 0.97774    |
| L2-pseudoRowXGB    | 0.977418   | -         | 0.97767    |
| L2-pseudoColNN     | 0.977494   | -         | 0.97762    |
| L2-pseudoRowNN     | 0.977471   | 0.97802   | 0.97757    |

## L3 - Hill Climb

For the final layer, I performed a weighted average of the four OOF predictions from the L2 models. The weights were calculated using a Hill Climb algorithm.

The final weights were:
`pcol_XGB*0.1 + prow_XGB*0.2 + pcol_NN*0.55 + prow_NN*0.15`

At a model-type level, this simplifies to **XGB\*0.3 + NN\*0.7**.

| Model  | CV         | Public LB | Private LB |
| ------ | ---------- | --------- | ---------- |
| L3-HC  | 0.977594   | 0.97828   | 0.97790    |

---

## Appendix

### Fold-Average vs. Refit Comparison

I'm sure many people debated whether to use the average of fold predictions or predictions from a model refitted on all training data. I ultimately chose fold-averaging for all my models, but I ran a few experiments to make this decision.

I split the training data 80:20. The 20% split was held out as a validation set. I used the 80% portion to perform cross-validation and refitting, and then compared the performance of both methods on the held-out 20%.

When refitting, I tried using `1.2x` and `2.0x` the average `best_iteration` from the folds, but in all cases, the **fold-average** method produced better and more stable results.

Looking at Chris Deotte's experiments, it seemed he saw a decent improvement from refitting. I personally speculate that this was likely due to an increase in the number of models (refit + multiple seeds) rather than just the increase in training data. That is, `refit + multi-seed average > fold-average > refit (1 seed)`. Since training multiple seeds for each fold (which also increases the model count) gave me performance very similar to `refit + multi-seed`, I adopted that approach.

### How I Used AutoGluon

When I used AutoGluon after extensive feature engineering, I frequently ran into "disk space exceeded" errors. To manage this, I limited the number of models by using `excluded_model_types` or `included_model_types`. I also set `num_stack_levels=0` to ensure I was only getting the L1 base models.

```python
predictor = TabularPredictor(label=label, eval_metric=metric, groups='fold').fit(
    train_data,
    excluded_model_types=['XGB'],
    # included_model_types=['GBM', 'FASTAI'],
    time_limit=time_limit,
    presets='extreme',
    num_cpus=4,
    num_gpus=2,
    num_stack_levels=0,
)
```

By restricting the models to lighter ones like `included_model_types=['GBM', 'FASTAI']`, it was even possible to train up to L3 within the environment.

For information on how to retrieve OOF predictions, the official docs and @ravaghi's notebook were very helpful: [https://www.kaggle.com/code/ravaghi/s05e05-calorie-expenditure-prediction-automl](https://www.kaggle.com/code/ravaghi/s05e05-calorie-expenditure-prediction-automl).

---

I would be delighted to hear any feedback or suggestions for improvement in the comments.
