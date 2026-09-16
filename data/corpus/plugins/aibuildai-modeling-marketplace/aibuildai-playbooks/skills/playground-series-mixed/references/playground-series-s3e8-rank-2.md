# 2nd Place Solution

Competition: playground-series-s3e8
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s3e8/discussion/392828

### Overview

The goal of this competition was to minimize Root Mean Squared Error (RMSE) for gemstone price predictions. My solution was a 2-level stacked model. The first level of the stack consisted of 60 different models utilizing XGBoost, CatBoost, and LightGBM. First level model selection was based on a search through model space using RMSE as a guide - I started with collections of first level models that had local CV RMSE scores of 574 or lower, and stepped this threshold down by increments of 0.1, building second level Ridge models and searching for the model combination that gave the smallest local CV score for the second level. The optimal cutoff for selection of first level models was 572.6. All models were generated with training data split into 10-folds, predictions were made out-of-fold (OOF), and metrics were gathered OOF. Feature engineering is discussed in more detail below.


### What Worked

* **[High Impact] Feature Engineering**
    * Result: CV lift of about -1.5
    * Reasoning and context: cut, clarity, and color features, with aspect ratio features was better able to segment the data into various price ranges.
    
* **[High Impact] Use of Original Dataset**
    * Result: CV lift of about -1.6
    * Reasoning and context: additional data provided more samples to learn from.
    
    
### What Didn't Work

* **Exploiting Duplicates in the Original Dataset**
    * Result: public LB lift of about -0.05, but drop in private LB of about +0.1
    * Reasoning and context: duplicates between the original data and test data existed. Using the price from the original data as predictions for the test duplicates appeared to provide good lift on the public LB, but didn't hold in private LB. See more below in additional context.
    

### Additional Context

For this competition, I generated a total of 1,816 models. There were 1,709 models that were first level models - a mixture of XGB, CatBoost, Ridge, and LightGBM models.  The first level models were exploratory - they were focused mainly on testing engineered features along with hyper-parameter searches. I generated a total of 105 second level models using Ridge regression, blending, and neural networks. Most second level models were Ridge models that were searching for optimal selection of first level models. Second level blended models performed poorly compared to Ridge regressions. While second level neural networks provided a slightly better CV metric compared to the best Ridge model, all my neural net models resulted in a hard rebound on the public LB, suggesting bad overfit (which was subsequently confirmed via the private LB). 

Aside from the model stacks lending stability to the predicted results, the two key insights were feature engineering and duplicate handling. In terms of duplicates, the previous competition revealed duplicates between the training set and the test set, the handling of which turned out to be critical for top scoring solutions. In this competition, there were 398 rows from the original data that were duplicated in the testing data, and 1,440 rows from the training data that were duplicated in the testing data. I attempted several ways of overlaying the prices from the original dataset and the training dataset onto the testing dataset as a post-processing step. Using public LB scores as a guide, it appeared that overlaying the 398 rows from the original dataset onto their testing counterparts generated lift. However, I remained very skeptical of the results, and hedged bets on final submissions, using my best Ridge model with and without the exploit applied. In the end, the better model was the one without the exploit.

The second insight - feature engineering - was particularly important. In [my EDA](https://www.kaggle.com/code/craigmthomas/play-s3e8-eda-models) I looked at several different engineered features. The first insight was that in the real world, the cut, clarity, and color categoricals correspond to a numbered scale, and that this scale is used to make quality comparisons between diamonds. Simple tests showed that considerable lift was possible if we converted the categoricals to a numeric scale, and made sure that LightGBM, CatBoost, and XGBoost didn't treat those columns as categorical in nature. This was confirmed by using different encoding methodologies on the categorical values, and comparing their relative scores (see figure below). While the color scale feature didn't appear to provide lift in my EDA, it did have a small impact when it came to tuned versions of models that used it. Another feature engineering insight came with respect to aspect ratio and carat value. By dividing up the standardized aspect ratio of the gemstone by an exaggerated version of the carat value, we could tease apart low value gemstones from higher value ones. 

Here is a summary that explained some of the different approaches and their results when used to generate untuned LightGBM models. Again, the specific details of each model type and what they were exploring is explained in more detail in my EDA.



### Thanks and Acknowledgements

Once again, thank you to the Kaggle organizers for continuing to deliver the Playground series. Again, having access to such a large number of competitions in such a short timeframe is so valuable. And again, thank you to everyone who competed and took part in the discussions - it's always great to learn about the different approaches that everyone is using to approach the problem, and hear about what is working and what they are learning.
