# 43rd place: Every cloud has a silver lining

Competition: equity-post-HCT-survival-predictions
Rank: #43
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566577

I would like to start this article by thanking users Ken Lee(@kendontcare11) and Albanito, (@albansteff) your public notebooks inspired me a lot, I used both for ensemble and achieved the highest score of all my solutions in a very short time. **But very sadly (or unfortunately), I didn't choose it as my final submission, probably because I think shorter time schemes are easy to overfit**😭😭😭
This is the highest private score out of all my solutions：

[https://www.kaggle.com/code/aristotlechen/top-30-solution](url)

*This version doesn't output a score because I ran out of my GPU time, but this private score is at least a top 30 solution*

-------------------------------------
*As I am not a computer-related professional, the following content represents only personal opinion, if there are some errors, please point out that*

1. I've found that most programs don't include LGBM, but intuition tells me that many times the things we ignore are exactly what we need.CatBoost is used for categorical feature processing, LightGBM is suitable for large-scale data, XGBoost performs more consistently on small datasets, and the integration of the three reduces the bias and variance of a single model.


2.In medical data processing competitions, tabular data (e.g., ISIC2024, Child Mind Institute - Problematic Internet Use, etc.) is the main form of data, and TabNet is able to capture the relationship between these features well, and TabNet's feature selection capability can help to improve the generalization performance of the model.
overfitting problems may occur if the amount of training data is not enough (this training set is estimated 30MB.).
It can fill in data by learning patterns of missing values without additional preprocessing steps. (I didn't use it because my model in Child Mind Institute - Problematic Internet Use had a huge SHAKEUP!)

3.
**Comparison of weighted average methods and ranking-based integration methods**:
Versus Weighted Average Methods If the prediction results of individual models are already better, weighted averaging can further improve performance
**Sorting methods lose specific information about the original predicted values, which may lead to a decrease in the predictive power of the model. Although C-index is based on sorting, Stratified C-index also requires that the model's predictions be consistent (i.e., have a small standard deviation) across racial groups. Sorting methods may disrupt the model's predictive consistency across groups, resulting in larger standard deviations and thus lower final scores.**
