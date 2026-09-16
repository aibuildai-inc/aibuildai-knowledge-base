# #2 solution 0.52521

Competition: playground-series-s3e13
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s3e13/discussion/407829

After returning from my holiday, I was pleasantly surprised to find I won second place on the private leader board! This was my first competition, and it was an awesome experience. I would like to thank Kaggle for organizing this series of competitions, which have been super helpful for us newcomers. Also, thanks to everyone who shared their code, solutions, and experiences - you guys inspired me!

Getting back to the main point, I initially wanted to share my code, but I found it to be so messy that I didn't even know where to start organizing it. So, I've decided to share some tricks that I use.

I didn't focus much on the model, just went with XGB, and used StratifiedKFold while training. Here are my parameters:
`XGBClassifier(objective = 'multi:softprob',
                           tree_method = 'exact',
                           colsample_bytree = 0.6,  
                           gamma = 0.8,  
                           learning_rate = 0.01, 
                           max_depth = 6,
                           min_child_weight = 3, 
                           n_estimators = 300, 
                           subsample = 0.6)`
`StratifiedKFold(n_splits = 4, random_state = 42, shuffle = True)`
In terms of feature engineering, I first borrowed @sergiosaharovskiy 's clustering method for symptoms within the same category.
 
And, I combined every pair of features and performed AND, OR, and XOR operations, generating over 6000 new features in the process. 

The reason I did this is because I think different people might have different symptoms for the same illness. For instance, when I have a cold, I don't experience a sore throat but do have a headache, while others might have the complete opposite or both symptoms. Also, the features provided in the data are in 'one hot' format, and since I'm not well-versed in medicine, I went with the rather simple method mentioned above.

Next, I carried out the feature selection process. Starting with the original features as the initial set, I added the newly created features and used the model's performance(MAP@3) as a criterion to filter out the following additional features:
'cluster_0',
'cluster_1',
'cluster_2',
'cluster_3',
'weakness_or_yellow_skin',
'jaundice_or_abdominal_pain',
 'weakness_or_yellow_eyes',
 'stomach_pain_or_abdominal_pain',
 'back_pain_or_yellow_skin',
 'toenail_loss_xor_bullseye_rash',
 'weakness_or_light_sensitivity',
 'yellow_skin_or_prostraction',
 'coma_or_yellow_skin',
 'inflammation_or_light_sensitivity',
 'weakness_or_urination_loss',
 'weakness_or_slow_heart_rate',
 'abdominal_pain_or_irritability'

Finally, train and submit.

That's all for my work. During the competition, I also tried some feature selection methods, such as Sklearn's SelectKBest and the Boruta-SHAP package, but their performance on this dataset was not as good as the method I used above. Additionally, I've been focusing on researching various feature selection techniques recently, so I'm very eager to discuss them with all of you. If you have any great suggestions, please be sure to let me know!
