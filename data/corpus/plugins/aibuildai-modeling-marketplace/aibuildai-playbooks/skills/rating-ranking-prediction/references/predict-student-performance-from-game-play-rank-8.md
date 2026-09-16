# 8th Place Solution and Code

Competition: predict-student-performance-from-game-play
Rank: #8
Source: https://www.kaggle.com/c/predict-student-performance-from-game-play/discussion/420528

The competition was really exciting and it gave us a chance to practice feature engineering. I'm very thankful for the support and help from my team @shinomoriaoshi  @hoangnguyen719 and @martasprg. They were always there for me and together we made a big difference.

I would like to thank the hosts, and special thanks to @cpmpml and @pdnartreb for identifying the issue of data leak, which made the competition right back on track.

Special thanks to @cdeotte for his great starter notebooks and insights that helped me  in the early phase of the competition.

##Overview
Here's an overview of what each of us worked on:
·         My main focus was on improving the XGBoost model and handling feature engineering.
·         Minh Tri Phan worked on a Transformer model with a CV (cross-validation) score of 0.699 and a public leaderboard (LB) score of 0.7.
·         Hoang processed the external data.
·         Martin worked on selecting the most relevant features.

In our final submissions, we ensembled the XGBoost and Transformer models, which helped us achieve the gold position. Our ensemble submission had a public LB score of **0.705** and a private LB score of approximately **0.7025**. Additionally, we had two other submissions with single XGBoost models, where one had a public LB score of **0.705** and a private LB score of **0.700**.

##My Part

Code: The code is a bit uncleaned, apologies for that. For any queries, contact me on [LinkedIn](<https://www.linkedin.com/in/priyanshu-chaudhary-ba0b23199/) 
**FE code:** <https://www.kaggle.com/code/chaudharypriyanshu/mb-fb5-train-xgb-25-11-external-data/notebook>
**Inference code:** <https://www.kaggle.com/code/chaudharypriyanshu/inference-xgb-25-11-17/notebook>
**Training code:** <https://www.kaggle.com/code/chaudharypriyanshu/mb-fb5-train-xgb-25-9-training/notebook>

### Overview
I created a 5-fold XGBoost model for each question (a  total of 90 models). I used Kaggle kernels only to train XGBoost since it took only 45 mins on Kaggle’s P100 GPU to train all 90 models.
The single XGBoost model achieved a Public leaderboard (LB) score of 0.705 and took 45-50 mins for inference, but it didn't perform as well on the private LB. When we included Hoang's external data, the model's score improved to **0.704** on the private LB. However, we decided not to use it because the public LB score was unusually low at **0.702**.

###Feature engineering 

1. **Session length:** simply accounts for the total length of the session per level group.

2. **Instance features:**  I created Object click-based features (first object click, room coordinates of that click, I called them Instance features)that were most important and gave an improvement of 0.0007, when I added them with standard features. I created a total of 36 features since there were 12 instances where object clicks were present.

3. **Magic bingo features:** Inspired from the public notebooks. I created more such features for all 3 level groups and it improved the CV by **0.0003**.

4. **Standard features:**

      a) **Count features:** I created count features based on `Fqid, text_Fqid, room_fqid, level, and event_comb`. These features capture the frequency of specific events or combinations. 

      b) **Binning of indexes:** I performed binning on indexes with bin sizes of approximately 30 or 50 in sorted order. Raw indexes worked better on the private LB, while binned features yielded better results on the public LB.

      c) **First and Sum features:** I generated first and sum of elapsed_time_diff  for all categorical columns. I found that min, max, and std did not work well in my case. 

      d) **Aggregations based on hover duration.**

5. **Top Level Group Features:** Used top 15-25 features (according to feature importance), Duration and instance features across different level groups.

6. **Meta features:** Using past questions predictions to predict the current question. i.e. for question` t` I used all predictions for questions `(1 to t-1)`. Using them gave an improvement of around **0.001**.

###Feature Selection (Martin's Part):
1. I eliminated features that had zero importance based on their Gain and Shapley feature importance scores.
2.  After performing feature selection, I made adjustments to the learning rate by reducing it from **0.05** to **0.03** and adding more features. 
3. Additionally, I removed duplicate features and features with more than **95%** values as null.


###External data:
1. We used publicly available data. It had about 7500 sessions where all 18 questions were answered.
2. Adding this external data improved our model's performance by 0.0005 in cross-validation and 0.002 on the leaderboard.
3. Hoang also created processed external data that worked well on the private leaderboard (score of 0.704). If we had included it, our single XGBoost model could have reached a top 5. position. However, we decided not to use it because of its lower Public leaderboard score (a bad decision).

###Inference:

1. We made improvements to retain the original order of the sequence during inference.

2. We found that there are approx. 250 sessions with abnormal indexing(interestingly all of them are from the 5th and 6th December 2020)

3. Created a function to preserve the original sequence for 99.5% of sequences, with only a small portion (0.5%) having misplaced events not more than 4-5 positions of the actual index.

4. Reindexed these abnormal sessions which improved or scored on LB slightly.

###Things not worked:
1. Ensemble with LGBM, Catboost didn’t work.
2. Created a custom eval metric that uses benchmark true positives and negatives a model should have. It increased the CV by 0.001 but LB  decreased probably due to overfitting.
3. Different thresholds for each question. (increased CV decreased LB).


The below tables list our experiments with the best results.

|External Data Used| CV |Public LB  | Private LB | final Sub|
| --- | --- | --- | --- | --- |
| No |0.6996  |0.701| 0.698 | No |
| No|0.7001 |0.702| 0.700 | No |
| No |0.6996  |0.701| 0.698 | No |
| Yes((Public ED) |0.7015 |0.705| 0.700 | Yes |
| Yes(Hoang's ED ) |0.7019 |0.702| 0.704 | No |
| Yes (Hoang's ED) |0.7022 |0.703| 0.703 | No |

##Tri's Part:

The model is shown in the following figure:



Particularly, it consists of 2 parts,
(i) Training a neural network, then extracting the embedding.
(ii) Concatenating the embedding from the neural network to a set of aggregated features, then training a gradient boosting model (XGBoost, CatBoost, LightGBM).

###Neural network

I was inspired by the RIIID competition and @letranduckinh’s solution, in which he customized the multi-head attention mechanism to adopt the time gap between 2 actions. In my opinion, if we have to relate the problem to an NLP problem, RIIID competition is like a token classification task (e.g., NER), meanwhile, this competition is like a document classification task. Therefore, I decided to use a transformer and some other recurrent network types.

I used the encoder-only structure as I didn’t see any motivation to have the decoder. 
However, the transformer encoder alone didn’t work so well, so I decided to add some more (3) GRU layers in front of the encoder. The detailed architecture (Pytorch code) is given here (<https://github.com/minhtriphan/Kaggle-competition---Predicting-Student-Performance/blob/main/Transformer/model.py>).

###Some remarks about training:
1. I used 3 models for 3 level groups. At each level, I used the sequence of previous levels (e.g. The model for the 0-4 level uses the 0-4 sequence, the model for the 5-12 level uses the 0-4 and 5-12 sequences, and so on.)
2. I used all the given features to train the model,

```python
NUM_COLS = ['index', 'time_diff', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration']
TXT_COLS = ['level', 'event_name', 'name', 'text', 'fqid', 'room_fqid', 'text_fqid']
```

3. I think the performance of a student, for example, in level 13-22 could carry some information to predict his/her performance in level 0-4. This is what I call the “global knowledge” of a student, and I want the network to capture that. Therefore, the neural network is trained in a multi-tasking manner, in which in the main output is the set of questions in the corresponding level (e.g., for level 0-4, the main output is 3-dimensional for questions 1, 2, and 3), the auxiliary head is used to predict all other questions. This trick helps to gain **+0.002** in CV.

Overall, the NN gets **0.695/0.700** in CV and public LB (before the API crisis, after that I never check how the NN works in the public LB anymore as it was combined always with XGBoost)

###Gradient Boosting

However, the NN in my case was not super satisfactory. I then decided to extract the embedding from the trained NN, concatenate them into a set of aggregated features, then use XGBoost to train the model. This helped me to get a huge boost in both CV and LB.

Overall, the scores of this approach are shown below,

|External Data Used| CV |Public LB  | Private LB|
| --- | --- | --- | --- |
| No |0.6993  |0.702| 0.697 |
| Yes |0.6989  |0.701| 0.699 |


Unfortunately, as I didn’t observe any gain in CV and public LB with external data, I decided not to choose that model to add to our model pool.

**Links:**
Training code: <https://github.com/minhtriphan/Kaggle-competition---Predicting-Student-Performance----part-of--8th-solution.git>

**Inference code:** 
**Without external data:** <https://www.kaggle.com/code/shinomoriaoshi/psp-v7b-infer>
**With external data:** <https://www.kaggle.com/code/shinomoriaoshi/psp-v9a-infer>

## Hoang's Part: 
Hoang has described his work in a separate thread that describes the preprocessing of external data, experimental results and why to trust CV over LB.
Link to Hoang's Part: <https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/420315>
