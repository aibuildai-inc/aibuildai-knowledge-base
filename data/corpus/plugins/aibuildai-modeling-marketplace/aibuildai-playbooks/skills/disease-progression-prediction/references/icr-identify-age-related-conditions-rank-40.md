# 40th Place Solution for the ICR - Identifying Age-Related Conditions Competition

Competition: icr-identify-age-related-conditions
Rank: #40
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/432726

I'm really surprised by my position in the private leadrboard having gained over 3283 position :D


**Context section**

- Business context: https://www.kaggle.com/competitions/icr-identify-age-related-conditions

- Data context: https://www.kaggle.com/competitions/icr-identify-age-related-conditions/data

**Overview of the approach**

The difficulty of this challenge arose from the very limited number of available rows. 
This also posed significant challenges in defining a suitable cross-validation scheme.

As a cross-validation strategy, a stratified scheme was chosen based on the Alpha column.

The main component of my solution was to employ a contrastive learning approach. The reason for opting for contrastive learning was that this way, the scarcity of available observations would be compensated by a large number of simulated rows.

A substantial portion of the approach is derived from Setfit (source link at the end).

The model utilized was an LGBM, which, starting from the absolute difference of all features, aimed to classify whether two features belonged to the same class or not. In addition to the initial features on which the absolute difference was computed, the following additional features were added:

- number_zero: % of features with zero difference
- mean_diff: mean of the absolute difference
- std_diff: standard deviation of the absolute difference
- median_diff: median of the absolute difference
- diff_mean: absolute difference between the means of all initial values
- diff_std: absolute difference between the standard deviations of all initial values
- diff_median: absolute difference between the medians of all initial values

**Details of the submission**

For training the model, pairs were sampled in a 1:5 ratio (class 0 vs. 1) to ensure a balanced dataset. For each observation, a certain number of random examples from the same class and from different classes were selected. Any duplicate combinations were removed.

The metric used to determine the appropriate number of rounds was AUC.

During the inference phase, the following post-processing steps were performed:

- Predict the probability that a new observation belongs to class 0 (by comparing it with all class 0 observations).
- Calculate the probability that it belongs to class 1 (by comparing it with all class 1 observations).

With prob_0 and prob_1 (calculated as the mean of each previous calculated probabilities), calculate prob = prob_1 / (prob_0 + prob_1). This functions as a kind of ensemble.

This way, for each individual observation to be predicted, 617 different predictions need to be made.

The section to confront each new observation with the train observation is the given function:

```
def get_retrieval_dataset(
        test: pd.DataFrame, target_example: pd.DataFrame, 
        feature_list:list
    ) -> Tuple[pd.DataFrame, list]:

    test_shape = test.shape[0]
    target_example_shape = target_example.shape[0]

    test_x = test[feature_list].to_numpy('float32')

    target_example = np.concatenate(
        [
            target_example
            for _ in range(test_shape)
        ], axis=0
    )
    test_x = np.repeat(test_x, target_example_shape, axis=0)
    index_test = np.repeat(test.index.values, target_example_shape, axis=0)

    retrieval_dataset = fe_pipeline(
        dataset_1=target_example,
        dataset_2=test_x, feature_list=feature_list,
    )
    retrieval_dataset['rows'] = index_test

    return retrieval_dataset

```


What didn't work:

- Metric Learning using Neural Networks, both DNN and TabNet, did not work regardless of the chosen metric (cosine similarity, mse, contrastive loss, etc.). Unfortunately, I couldn't achieve better CV results than the LGBM, likely due to the extremely limited number of available observations playing a key role.
- Contrastive learning on Alpha column
- Use a weight for each given training observation the observation

**Sources**
[Inference Notebook](https://www.kaggle.com/code/stenford23/icr-inference-contrastive-retriever/notebook?scriptVersionId=130640859)

[SetFit](https://huggingface.co/blog/setfit)
