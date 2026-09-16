# 11th Place Solution

Competition: m5-forecasting-uncertainty
Rank: #13
Source: https://www.kaggle.com/c/m5-forecasting-uncertainty/discussion/163219

Congrats to all the winners! Below is a summary of our solution. Hope it will be useful to you in some way.


## Overview of Solution

Our solution combines three types of models:
1. Autoregressive RNNs that model the probabilistic distribution of the sales directly.
1. Autoregressive RNNs to obtain the “inherent” uncertainty of the forecasted sales.
1. Tree-based models to obtain bottom-up point estimates of the sales.

The reason for having three types of models stems from the hierarchical nature of the time series -- certain models are more capable of modeling the uncertainty for a given level in the hierarchy.



## Our Solution (with a little more detail)

As mentioned, we are dealing with hierarchical time series. On page 3 of the M5 competitors' guide, we are given a breakdown of the aggregation levels. Based on the behavior of the data, it seemed natural to us to divide the hierarchies into two types:
1. The aggregated sales of individual products (series levels 10, 11, and 12) exhibit more “count-like” behavior. They are also generally more intermittent. For these series, modeling the uncertainty as a poisson or negative binomial seemed to make sense.
1. The higher-level sales aggregations (series levels 1 through 9) exhibit a more continuous behavior. Modeling the sales distribution using some continuous probability distribution (e.g. Gaussian) seemed natural in this case.


### - 1st Hierarchy Type

For the first hierarchy type, we directly modeled the distribution of the sales using an autoregressive RNN, specifically [DeepAR](https://arxiv.org/pdf/1704.04110.pdf). In a nutshell, DeepAR attempts to model the conditional distribution of the future of a time series given its past. During training, the model learns to parametrize a likelihood function of a chosen probability distribution (in our case, we chose a negative binomial distribution given the count-like nature of the data). Then, during inference, we can draw samples from the selected probability distribution parametrized by the model and compute quantiles from the drawn samples. 

We developed our DeepAR model largely from this great [kernel](https://www.kaggle.com/steverab/m5-forecast-compet-uncert-gluonts-template). 

### - 2nd Hierarchy Type

For the second hierarchy type, we initially tried to model the distribution directly using DeepAR with a Gaussian distribution. While the results were decent, we found that there was a high sensitivity towards the random seed used and performance occasionally varied substantially across folds. It turned out that we were unable to get the DeepAR model to capture the center (median) of the distribution consistently.

We opted instead for a bottom-up approach, in which we predict the individual product sales and sum the respective predictions to get the predicted median of the aggregated sales. We found that LightGBM models from the M5 accuracy competition worked well for this. In particular, we adapted the LightGBM model from [this kernel](https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50). Key changes made were:
- Removed “magic” multipliers
- Tuned hyperparameters to mitigate overfitting
- Feature engineering pertaining to price volatility

Clearly, the bottom-up approach only gives us part of what we want (the median). To obtain the rest of the required quantiles, we turned back to our DeepAR (with Gaussian distribution) approach, where we used the ratio of the quantile prediction to the median prediction as a multiplier. For example, for a given series and prediction day, if the ratio of (say) the 25th percentile to the median in the DeepAR model is 0.95, then to get the 25th percentile of the corresponding LightGBM prediction, we simply multiply that prediction with 0.95. And we do this individually for all of the aggregated series across each time step in the horizon.

We felt that this approach made sense in that the DeepAR model captures the “inherent uncertainty” in the sales, which should “carry over” to the predictions from the LightGBM model. Admittedly, this justification is hand-wavy and lacks any statistical rigor. Empirically, however, this approach resulted in stable improvements across folds and was certainly preferable to manually tuning “quantile multipliers”.

Our approach can be summarized as follows:





## Validation Strategy

We realized early on that it was unwise to rely on a single 28-day validation horizon given the volatility of the competition metric, coupled with the ease of overfitting. Intuitively, we also felt that the recency of the folds matters. Bearing these findings/thoughts in mind, we ended up using the three most recent 28-day periods as our validation “folds” for our model development.



## Some Stats and Findings

We report the weighted scaled pinball loss (WSPL) on our validation folds for some of our experiments:

| Experiment                                                                        | D1858 - D1885 | D1886 - D1913 | D1914 - D1941 |
|-----------------------------------------------------------------------------------|:-------------:|:-------------:|:-------------:|
| LightGBM + “point to uncertainty kernel”                                          | 0.1967        | 0.1739        | 0.1672        |
| + DeepAR (Gaussian) ratio multipliers for 2nd hierarchy type series               | 0.1935        | 0.1656        | 0.1567        |
| + DeepAR (Negative Binomial) for 1st hierarchy type series                        | 0.1744        | 0.1512        | 0.1414        |
| + Ensemble LightGBM with modified “darker magic” kernel from accuracy competition | 0.1683        | 0.1499        | 0.1408        |

The model in the last row was our final submission, yielding a private LB score of 0.16721.

Some noteworthy findings:
- The magnitude of the WSPL can vary quite noticeably across different folds. However, the directionality of the loss is generally quite consistent across all folds (i.e., improvement in one fold is typically accompanied by the same relative improvements in all other folds).
- We found that training separate DeepAR networks for each individual product aggregation (series levels 10, 11, and 12) resulted in consistent small improvements in WSPL across all validation folds (~0.002). This could be due to DeepAR being a global model, and so by separating the series by their hierarchies, the separate models can better capture the nuances within each hierarchy. 
- Training the same models using different seeds and ensembling them did not give any noticeable improvement. Neither the naive way of simply averaging the quantile predictions of each model, nor drawing samples simultaneously across each model and then computing the quantiles from the pool of samples worked.
- In the final days of the competition, we ensembled our LightGBM bottom-up model with the one from [this kernel](https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic), with slight modifications to reduce overfitting. As seen in the table above, this led to slight improvements in score.





## Stuff that did not quite work/didn’t get to try

We attempted to model the distribution non-parametrically using a network called [DeepTCN](https://arxiv.org/pdf/1906.04397.pdf). While performance was decent, we couldn’t find a way to ensemble the predictions with our existing solution.

Modeling the different hierarchies separately discards potential information gain from combining higher and lower aggregation levels. Unfortunately, we did not have the time to explore forecast reconciliation methods.



## Closing Remarks

Special thanks to @steverab, @kneroma, and @kyakovlev for the aforementioned kernels that we leveraged. And of course, we would like to express our gratitude to Kaggle and the competition host for the opportunity to participate in this competition.
