# 3rd Place Solution

Competition: playground-series-s3e7
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s3e7/discussion/390979

### Overview

The goal of this competition was to maximize AUC ROC score for hotel booking cancellations. My solution implemented a 2-level stacked model. The first level of the stack consisted of the best 123 performing CatBoost, XGBoost, and LightGBM models that I generated. Each model was trained using a different selection of hyper-parameters. No engineered features were used, and the original dataset was mixed in for training purposes. All models were generated with training data split into 10-folds, predictions were made out-of-fold (OOF), and metrics were gathered OOF. The second level of the stack was a simple TensorFlow neural network. Objective was `SigmoidFocalCrossEntropy` from TensorFlow Additions. The neural network was trained using the training predictions from the first level, and predictions were made using the testing predictions from the first level. Post processing on the final set of predictions was performing utilizing the train / test duplication inversion technique discussed below. 

* **Model Stack Diagram**



* **Neural Network Configuration**



### What Worked

* **[High Impact] Train / Test Duplication Prediction Inversion**
    * Result: public LB increase of about 0.0141
    * Reasoning and context: inverting scores for duplicates in the testing data to lift performance - see discussions from @icfoer [here](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/388851), and from @sergiosaharovskiy [here](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/388992)
    
* **[Medium Impact] Stacking with Neural Network**
    * Result: CV increase of about 0.0026 compared to single models
    * Reasoning and context: best neural network configuration combined 123 models, and had an increase over the best single model of 0.0026.
    
* **[Medium Impact] Using Original Dataset**
    * Result: CV increase of about 0.002
    * Reasoning and context: I added the original dataset into the training set, making sure to perform CV metrics on competition data only.

### What Didn't Work

* **[Medium Impact] Any Feature Engineering**
    * Result: CV drop of about 0.004
    * Reasoning and context: exhaustive investigation into features along with model testing revealed no significant gains with any engineered features (e.g. time based features). See more below in additional context.
    

### Additional Context

My first big insight - no feature engineering - was key. As I mentioned above, I failed to find any way to boost class separation through feature engineering. My public facing EDA ended up being incomplete, because I stopped updating it when I began a more in-depth hunt for engineered features that would result in better separation between the positive and negative classes. After spending a lot of time digging through various ways of combining features, my conclusion was that I couldn't find anything significant that I could qualitatively or quantitatively measure as giving any type of lift to my models. I wasn't happy with that result, so I spent time doing the opposite - proving to myself that most of the feature combinations were hurting performance. Any time-based feature engineering such as adding day of week, quarter, day of year, etc. worked to bring down local CV scores. My best model using time-based features scored 0.900011 compared to my best model without them of 0.904148. 

My second biggest insight in this competition came from the agreement between local metrics and the public LB. Adversarial validation between train and test suggested that the two datasets were nearly identical. Given that we had a fairly generous ratio of positive cases vs negative cases, and that we had a decent number of training and testing samples to use (plus additional data we could mix in), my initial hypothesis was that local CV and public LB should match up fairly well. To test this, I submitted a number of models - each one meant to see how an increase or decrease in local CV was reflected on the public LB. About 10 models in, it became apparent that my metrics and the LB were in good agreement.

With regards to the testing / training duplication and the inversion post-processing, after testing out agreement between my local CV and the leaderboard, I looked at the discussion of duplicates between the training and testing data, the original from @icfoer [here](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/388851) and a followup from @sergiosaharovskiy [here](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/388992). I implemented an inversion on the testing duplicates and saw a good increase on the public LB by doing so. For every model I generated, I made sure to generate a set of testing predictions that contained the inversion as well, and tested to make sure that all notebooks that inverted the data saw lift on the public LB, which they did. Still not trusting the public LB entirely, I hedged bets, and for final model selection used the best model that inverted the predictions for the duplicates in test and train, and the model that just kept them as was. The inverted prediction model won third place.

To ascertain whether I should use the original data, I performed an adversarial validation that resulted in an AUC ROC score of 0.7062. This suggested that there were differences between the datasets, but local CV testing suggested that there was a performance benefit to using it, and the public LB scores remained in-line with the CV results. The trick was ensuring that the CV metrics were done using only the competition data, so that I could compare my models to each other correctly.

In total, I generated 623 models - 553 of them first level, 67 second level, and 3 of them third level. First level models were a mixture of CatBoost, XGBoost, and LightGBM. The 553 first level models were exploratory in nature, usually testing out features and random combinations of hyper-parameters. Second level models were combinations of the first. I attempted blending and stacking of first level models. For blending, I used between 6 and 10 of my top performing models and searched for the best co-efficients to blend them together that maximized AUC. I would then drop out any models that had 0 weight, and swap in another. Best blend had CV increase over best single model of 0.0007. I then swapped to using a stacking technique using a Ridge regression. The best Ridge model had a CV increase over best single model of 0.0019. Finally, because my first level model predictions were already ranged nicely between 0 and 1, I explored regression using neural networks. Best neural network configuration combined 123 models, and had an increase over the best single model of 0.0026.


### Thanks and Acknowledgements

Thank you to the Kaggle organizers for continuing to deliver the Playground series. I very much appreciate the time and effort that goes into finding the datasets and generating novel ones from them so that we can compete and learn! Thanks to @sergiosaharovskiy and @icfoer for sharing their insights about the duplicates in the training and test datasets. Finally, thanks to everyone competing for great notebooks and discussions - always great to see such great engagement and enthusiasm!
