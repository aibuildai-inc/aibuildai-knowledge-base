# 6th Place Solution with AutoML☘️

Competition: planttraits2024
Rank: #6
Source: https://www.kaggle.com/c/planttraits2024/discussion/510143

Hello everyone, we would like to extend our gratitude to the organizers, participants, and you, the reader of this article.

We are a group of Kaggle novices who have some computational resources. Therefore, we chose an AutoML framework, [AutoGluon](https://auto.gluon.ai/stable/index.html), to simplify our training process and use some settings as our reference. The following understanding of AutoGluon comes from our interpretation of the documentation and source code, I hope there's no inaccuracies.

AutoGluon has a module for handling multi-modal data, which extracts features from various types of data using different models and then combines these features (specifically through concatenation) to make predictions. In this competition, we used models from the TIMM library to extract image features and combined them with tabular data features for the final predictions. Our tests showed that this strategy outperformed using image models alone.

For tabular data, AutoGluon provides [FT-Transformer](https://arxiv.org/abs/2106.11959) and MLP to extract features from them. Although GBDT models like XGBoost perform well with tabular data, they are not suitable for feature extraction.

For the fusion architecture, MLP and Transformer were the two options. Our tests showed that the Transformer slightly outperformed the MLP.

For multi-label processing, we used a chaining method. This involved training a model on the dataset without labels to predict the first label. For the second label, we trained a model using the original dataset plus the prediction of the first label as new data, and so on for subsequent labels. The order of the labels is determined based on the r2 values in the paper, with more easily predictable variables placed first. This approach is intended to prevent errors in predictions from affecting subsequent predictions. In our experiments, this method performed better than directly outputting multiple labels, though it required more computational resources and time. We believe this chaining method better considers label correlations, but our control over other variables was not strict enough, potentially introducing some errors.

However, image models were still the decisive factor for scoring. We tested the performance of various image models and found that the EVA series performed exceptionally well for this task. Using [eva_large_patch14_336](https://arxiv.org/abs/2211.07636) scored 0.483 on the private leaderboard, and [eva02_large_patch14_448.mim_m38m_ft_in22k_in1k](https://arxiv.org/abs/2303.11331) scored 0.486 (incidentally, I’ve always wondered if EVA’s naming come froms Neon Genesis Evangelion's Evangelion, as extracting EVA fro" as extracting EVA from "Explore the limits of Visual representation at scAle" seems really unusual). Other notable models included vit_large_patch16_384 with a score of 0.43 and swin_large_patch4_window12_384 with a score of 0.419. AutoGluon also performs model soups by merging the three highest-scoring models from the validation set. Due to limited computational resources and time, we didn’t do much hyperparameter tuning. Additionally, AutoGluon’s tuning framework exhibited some unexpected behaviors, which impacted our results.

Next, we performed stacking. To avoid overfitting, we chose to train on the original validation set, although we were not entirely sure if this was the right choice. We continued using AutoGluon, but this time its tabular module. The tabular module offers various models (including advanced GBDT and neural network models) with different hyperparameter configurations, and can perform bagging and weighted averaging (although we did not opt for additional stacking in our task, it is available for use in other tasks). The experiment demonstrates that it outperforms other meta-learners.. We eventually found that incorporating the version using MLP as the fusion model into stacking also improved the score significantly, ultimately achieving a score of 0.526, placing us 6th. We believe stacking effectively leveraged the strengths of different models and better handled outliers in the dataset.

In terms of data preprocessing, we didn’t introduce much innovation. We observed many excellent analyses of outliers and applied their removal to our workflow after testing.

Overall, I believe AutoML frameworks like AutoGluon indeed bring a lot of convenience and provide valuable references for those not yet technically mature. However, they still require some learning effort to read through documentation and source code, and they might lack flexibility when facing specific customization needs, often requiring deep modifications to the source code, consuming significant time.

Additionally, we attempted to create our dataset by requesting TRY’s public data and images from iNaturalist through an API. We tried using five images instead of one to provide more material for the model to learn from. However, due to time constraints, models based on this dataset are still training, and the results are yet to be determined. We will update this post with the results after submission, so feel free to follow up if you are interested.

Edit: I chose to merge multiple images, but this strategy didn’t lead to better scores. Perhaps using multiple images for each species would be more suitable for converting the problem into a classification task?

Lastly, thank you sincerely for reading all the way through!
