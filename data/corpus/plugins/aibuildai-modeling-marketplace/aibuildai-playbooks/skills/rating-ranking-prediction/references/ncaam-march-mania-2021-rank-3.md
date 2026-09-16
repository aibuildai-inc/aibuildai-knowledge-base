# 3rd place solution | NCAAM 2021

Competition: ncaam-march-mania-2021
Rank: #3
Source: https://www.kaggle.com/c/ncaam-march-mania-2021/discussion/231063

Thank you Kaggle for hosting this March Madness Mania 2021 competition!  I received my first award on Kaggle, solo gold 🥇 scoring in 3rd place in this competition.

I have made my solution public here in this Kaggle notebook.

[https://www.kaggle.com/timsereno/ncaam-march-mania-2021](url)

**Input:**

I began with code from a public notebook with a dataset comprised of features already engineered from just minimal data files provided by Kaggle for the competition.  This did not include the Massey Ordinals.  My goal was to focus my attention on the model tuning rather than on data wrangling.

**Process:**

I used PyCaret library to create a classification model. Tried various parameters to improve LogLoss during model training.  I ended up enabling the PyCaret parameters for feature engineering, imputing missing data, and boruta feature selection.  The combination of all of these settings made a significant difference in reducing the LogLoss metric during model training.

Using PyCaret functions, performed 10 fold cross-validation, and blended (ensembled) three models using Extra Trees, LDA, and NB linear regression algorithms.  I tuned the model with 10 iterations using the out-of-the box grid search settings within PyCaret, but optimizing specifically for LogLoss.  Finally, I finalized the model with PyCaret function to train on all the data instead of just the training/testing data which helped generalize the model.

Catboost and LightGBM were really slow to train model even with GPU acceleration and metrics were not improving, so decided to go with the faster classic algorithms.  

Please see notebook for settings that provided the lowest LogLoss metric during training.  However, I'm sure that ensembling various other algorithms and tuning for more iterations may further reduce LogLoss during training the model.

**Output:**

Please note that I did not manually override any of the predictions submitted from the finalized model.  I wanted to see how well a generalized model could perform against 707 total participants.



Let me know if you have any kind of confusion/questions. Happy to answer!
