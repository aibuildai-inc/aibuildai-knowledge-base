# [Private LB 8th] solution

Competition: jane-street-real-time-market-data-forecasting
Rank: #8
Source: https://www.kaggle.com/c/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution

First of all, thanks to the host for this amazing competition! It was a rare and exciting opportunity to apply deep learning to tabular data and test how well models can adapt to new data in real-time, simulating real-world conditions. I really enjoyed participating and learning throughout the process. Thanks also to all the participants who contributed to public discussions - I learned a lot from you! @victorshlepov, @lihaorocky, @johnpayne0, @shiyili

Link to the code https://github.com/evgeniavolkova/kagglejanestreet
Link to the submission notebook https://www.kaggle.com/code/eivolkova/public-6th-place?scriptVersionId=217330222

## 1. Cross-validation

I used a time-series CV with two folds. The validation size was set to 200 dates, as in the public dataset. It correlated well with the public LB scores. Additionally, the model from the first fold was tested on the last 200 dates with a 200-day gap to simulate the private dataset scenario.

## 2. Feature engineering and data preparation

## 2.1 Sample

I used data starting from `date_id = 700`, as this is when the number of `time_id`s stabilizes at 968. I experimented with using the entire dataset, but it did not result in any score improvement.

## 2.2 Data preparation

Simple standardization and NaN imputation with zero were applied. Other methods didn't provide any improvement.

## 2.3 Feature engeneering

I used all original features except for three categorical ones (features 09–11). I also selected 16 features that showed a high correlation with the target and created two groups of additional features:

- Market averages: Averages per `date_id` and `time_id`.
- Rolling statistics: Rolling averages and standard deviations over the last 1000 `time_id`s for each symbol.

Besides that, I added `time_id` as a feature.

Adding these features resulted in an improvement of about +0.002 on CV.

## 3. Model architecture

## 3.1 Base model

Time-series GRU with sequence equal to one day. I ended up with two slightly different architectures:

- 3-layer GRU
- 1-layer GRU followed by 2 linear layers with ReLU activation and dropout.

The second model worked better than the first model on CV (+0.001), but the first model still contributed to the ensemble, so I kept it.

MLP, time-series transformers, cross-symbol attention and embeddings didn't work for me.

### 3.2 Responders

I used 4 responders as auxiliary targets: `responder_7` and `responder_8`, and two calculated ones:

```python
df = df.with_columns(
    (
        pl.col("responder_8")
        + pl.col("responder_8").shift(-4).over("symbol_id")
    ).fill_null(0.0).alias("responder_9"),
    (
        pl.col("responder_6")
        + pl.col("responder_6").shift(-20).over("symbol_id")
        + pl.col("responder_6").shift(-40).over("symbol_id")
    ).fill_null(0.0).alias("responder_10"),
)
```

These are approximate rolling averages of the base target over 8 and 60 days, respectively. As described in detail in [this discussion](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/555562) by @johnpayne0, `responder_6` is a 20-day rolling average of some variable, while `responder_7` and `responder_8` are 120-day and 4-day rolling averages of the same variable, with some added noise. Given an N-day rolling average, we can easily calculate N*K-day rolling averages.

A separate base model was used for each auxiliary target. The predictions from these models were then passed through a linear layer to produce the final target output, `responder_6`.

The sum of losses (weighted zero-mean R²) for each responder was used to train the model.

Adding auxiliary targets improved both CV and LB scores by about +0.001.

Models were trained using a batch size of one day, with a learning rate of 0.0005.
For submission, I trained models on data up to the last date_id, using the number of epochs equal to the average optimal number of epochs on CV.

### 3.3 Ensemble

I ran both models on 3 seeds and took a simple unweighted average of predictions from those 6 models. This resulted in an LB score of 0.0112 (vs best single model LB 0.0105).

## 4. Online Learning

During inference, when new data with targets becomes available, I perform one forward pass to update the model weights with a learning rate of 0.0003. This approach significantly improved the model’s performance on CV (+0.008). Interestingly, for an MLP model, the score without online learning was higher than for the GRU, but lower with online learning.

Updates are performed only with the `responder_6` loss, without auxiliary targets.

Updates are applied for the entire dataset provided during submission, including rows with is_scored = False.

I also considered performing a full online retraining on the data up to the start of the private dataset. This would make sense because there is a significant gap between the training data and the private dataset. However, retraining the model would require distributing the training process across multiple inference steps, as the one-minute time limit between dates would not be sufficient. I believe this would have been feasible but I decided not to spend time on it, although my tests suggested that it could provide a +0.001 improvement in the score. Still, I find it amazing that, instead of a full model retraining, performing one-day updates for almost a year is enough, and the model continues to perform well.

## 5. Technical details

### 5.1 Inference Speed

Inference speed was critically important, so I spent a significant amount of time optimizing my code, particularly data processing and calculation of rolling features.

For my final submission, it takes 0.06 seconds to run one inference step (`time_id`), 0.02 of which are spent on data processing. Updating model weights once per `date_id` takes 3.6 seconds.

I used PyTorch, but since TensorFlow is said to be faster, I tried switching to it. However, after a few days of experimenting, I couldn't achieve better performance, so I decided to stick with PyTorch.

### 5.2 Technical stack

Due to RAM requirements, I switched from Google Colab to vast.ai and was extremely happy with the decision. I wrote code locally, enjoying all the perks of VSCode, and then ran a script to push the code to github, pull it on the server and execute scripts remotely.

I also used WandB to monitor experiments, which helped me keep track of scores and easily revert to an older version of the code if something went wrong.

To debug my submission notebook and estimate submission time I used [synthetic dataset](https://www.kaggle.com/code/shiyili/js24-rmf-submission-api-debug-with-synthetic-test) by @shiyili.

### 6. Scores

|                                                          | CV fold 0 | CV fold 1v | Fold 1 with 200 days gap | CV avg |
| -------------------------------------------------------- | --------- | --------- | ------------------------ | ------ |
| GRU 1 without both auxiliary targets and online learning | 0.0161    | 0.0062    | 0.0011                   | 0.0112 |
| GRU 1 without auxiliary targets                          | 0.0235    | 0.0148    | 0.0136                   | 0.0190 |
| GRU 1                                                    | 0.0249    | 0.0153    | 0.0147                   | 0.0201 |
| GRU 2                                                    | 0.0262    | 0.0166    | 0.0161                   | 0.0214 |
| GRU 1 + GRU 2                                            | 0.0268    | 0.0169    | 0.0163                   | 0.0218 |
| GRU 1 3 seeds                                            | 0.0258    | 0.0164    | 0.0152                   | 0.0211 |
| GRU 2 3 seeds                                            | 0.0267    | 0.0175    | 0.0163                   | 0.0221 |
| GRU 1 + GRU 2 3 seeds                                    | 0.0270    | 0.0175    | 0.0162                   | 0.0222 |

Fold 0: `date_id`s from 1298 to 1498.
Fold 1: `date_id`s from 1499 to 1698.
