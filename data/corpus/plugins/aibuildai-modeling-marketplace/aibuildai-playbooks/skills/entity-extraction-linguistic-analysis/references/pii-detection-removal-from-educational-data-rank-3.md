# 3rd Place Solution

Competition: pii-detection-removal-from-educational-data
Rank: #3
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497482

Thank you Kaggle and The Learning Agency Lab for another interesting competition!

# **Context section**
Business context: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/overview
Data context: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data

#  **Overview**
My final solution consists of a single backbone - deberta-v3-large - that was trained with multiple seeds in two stages. The key was getting the right mix of data, post-processing  and ensuring that the model performed well for all classes (including those that had few or no samples in the training data). A huge thank you to @nbroad and @mpware for the datasets that they shared. The solution was implemented using the following steps:
- Pre-training with mpware dataset
- Fine tuning on competition + nbroad dataset
- Post-processing

# **Data**
There were quite a few datasets that were shared during this competition. I experimented with varying combinations and eventually ended up with what I’ve described above - using mpware for pre-training and a combination of the competition data and nbroad’s dataset for finetuning. 

The mpware dataset was very useful but using it directly reduced the model's performance for some of the classes. However, using it for pre-training increased both CV and LB.

This was already discussed during the competition  - there were zero or very few examples of some classes (like `PHONE_NUM`) in the training data and public LB. As a result, it was hard to get a sense of how the model was performing for these classes by doing any validation on the train data or by using the public LB. I ended up using some of the shared external datasets as well as generating some of my own data to evaluate how my model was performing for these classes and that seemed to have worked well.

# **Training and Post Processing**
The final solution uses only one backbone - deberta-v3-large. I tried a few others but none of them improved performance. I spent some time tuning the hyperparameters based on my CV and then retrained the model on the full dataset with different seeds. Adding  `\n\n`, `\t\r` and `\n` as new tokens to the tokenizer was helpful. Other hyperparameters:
- learning rate: 1e-5
- batch size: 2
- gradient accumulation: 2
- warmup: 0.1
- epochs: 2

For post-processing, I used the same strategy as most public notebooks and used a threshold for the `O` labels. This worked well but there were false positives so I developed a set of rules based on my oof predictions to filter these false positives -  remove extremely short predictions for some classes, remove common instructor names from `NAME_STUDENT`, remove predictions of Mr, Mrs, Dr for `NAME_STUDENT` etc. 

I also included some other post processing based on my oof predictions. For instance, in a few essays the same name would appear as `B-NAME_STUDENT` and `I-NAME_STUDENT` like the name 'Leroy' in the essay below. The model tended to mix up the predictions in these cases and it had to be fixed with post-processing. 

# **Validation**
I used stratified 5 folds for cross validation. The stratification was based on the presence or absence of PII. CV and public LB were well correlated. 

However, as I mentioned earlier, there were some classes that had zero or few samples in the training data and public LB so both CV and public LB were not reliable indicators of how the model was performing for these. It seemed quite likely that these classes would exist in the private test so I generated essays using Mistral and used these and some of the shared external datasets to evaluate my models performance. This was definitely helpful. 

# **What did not work**
- Relabelling training data
- Other backbones
- Models for individuals classes 
- Using more external data
- Using a second stage model to identify false positives

Thanks again to Kaggle and The Learning Agency Lab for hosting such a great competition. And congratulations to all the winners!

Inference code: [https://www.kaggle.com/code/rai555/pii-data-detection-inference](https://www.kaggle.com/code/rai555/pii-data-detection-inference)

Training code: [https://github.com/srai9/pii-data-detection-3rd-place-solution](https://github.com/srai9/pii-data-detection-3rd-place-solution)
