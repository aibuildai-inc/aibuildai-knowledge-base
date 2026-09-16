# 37th place solution - TabPFN with only comp. data and basic FE.

Competition: playground-series-s5e3
Rank: #37
Source: https://www.kaggle.com/c/playground-series-s5e3/discussion/571139

My approach for the solution: 

As the dataset was extremely small TabPFN model is a great choice, also skipping the feature engineering, e.g. creating more features, to reduce the risk of overfitting on the small dataset, only handling basic FE. Also skip adding any other dataset as it could affect the distributions between the small train and test dataset as they were generated from the same LLM.

Model framework: https://github.com/PriorLabs/TabPFN

Result: 

Using only TabFN model and framework, simple feature engineering and only the synthetic generated data gave the best score on the private score.

Other:

Next best score I had with XGB, trained with adding the extra original data to the generated.
XGB is usually the best pick with binary dataset.

In the mirror one maybe should have ensembled them both and had a better score.

-----------------

That's it! Happy Kaggling!
