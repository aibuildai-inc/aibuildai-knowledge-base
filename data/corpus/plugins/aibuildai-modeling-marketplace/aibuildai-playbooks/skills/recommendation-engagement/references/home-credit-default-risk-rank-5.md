# Overview of the 5th solution [+0.004 to CV/Private LB of your model]

Competition: home-credit-default-risk
Rank: #5
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/64625

## Acknowledgements ##

We would like to thank:

 - Kaggle/Homecredit - for providing such an interesting and non-anonymized dataset
 - Neptune guys - who generously shared their solutions and magnificent [code][1]
 - Olivier for sharing his knowledge and kernels that helps us with feature optimization problem.[Your ‘Null Importance’ - improved model][2] made it less time consuming and gave slight improvement on CV (first time it was feature reduction from 4500 to 920)
 - Silogram for your educational and inspiring ['few notes'][3]  
 - All competitors. One man’s dream come true is another man’s nightmare, but you are “the salt of this earth” called Kaggle, and your presence and pressure makes us all better in Data Science. Your chasing made us dig deeper into data and find out more and more interesting features.

## Overview of the solution ##

Our best single model scored 0.80550 on Private LB, which would be enough for 3rd place. Unfortunately, it was rather impossible to select it among our solutions- it had neither the best CV (0.8064) nor LB (0.8150). Our best submission was a weighted average of 3 models (0.25, 0.25, 0.5), but scored the same on LB as one of the models in the mix. Maybe the fact that we didn’t put much attention to stacking (and that stacking gave us no improvement, and was much worse that the best single model) was our biggest mistake. We invested everything into feature engineering, because the data was so interesting to analyze. 

We did a lot of hand-crafted features (around 8.000), out of which we selected around 3.000 using mechanism based on Olivier’s work. 

There are few main ideas that we think we could have done differently vs. others, and which gave us the lead on public LB.

 1. Using deep learning to extract interactions among different data sources
 2. Using “nested models”, with (2.1) different approach to credit card balance features
 3. Using a model to predict the interest rate/duration of loans in current applications
 4. Using scoring model based on banking mechanism taken from articles and meetings
 5. Some additional ideas

First 3 points (apart from 2.1 which caused an overfit) resulted in ~60 features that gave us 0.005 boost on cv / 0.010 boost on LB. We share them in krakowlublinzhabinka_feats.csv attached to this post so that you can check if they improve your models too. Their format is the following: First 307511 rows belong to train cases, and the following rows belong to test cases. The order of IDs is preserved within train/and test subsets, so that you can use pd.concat([your_df, krakowlublinzhabinka_feats], axis = 1).

Below we elaborate on all of the points. 

## 1. Using deep learning to extract interactions among different data sources ##

We wondered how we can capture the interactions between signals coming from different data sources. For example, what if 20 months ago someone was rejected in external Bureau, had a late payment in installment payments, and applied for a loan at Homecredit? These types of interactions are very hard to capture by humans because of the number of possible options. So, we turned to deep learning and turned this problem into an image classification problem. How? Attached is the sample “user image” that we fed to the neural network (user_100006_image.xlsx ). 

You can see that we created a single vector of user characteristics coming from different data sources for every month of user history, going as far as 96 months into the past (8 years was a cutoff in most data sources). Then we stacked those vectors and created a very sparse “user image”. 

The network architecture was as following: 

 - Normalization - division by global max in every row
 - Input (the “user image” in format of n_characteristics x 96 months - we looked 8 years into the past)
 - 1-D convolution spanning 2 consecutive months (to see the change between periods)
 - Bidirectional LSTM
 - Dense
 - Output

This model scored 0.72 AUC on cv, without any features from current application. It trained rather quickly = around 30 mins on GTX 1080. We have put oof predictions from this model as a feature into our LGBM model, which gave us around 0.001 on CV (improvement on an already very strong model with &gt;3000 features) and 0.001 on LB. This means that the network was able to extract some information on top of &gt;3000 hand-crafted features.

## 2. Using nested models ##

One of the things that bothered us throughout the competition was a somehow arbitrary nature of various group-bys that we performed on data. For example, we supposed that an overdue installment 5 years ago is less important than 1 month ago, but what is the exact relationship? Traditional way is to test different thresholds using cv score, but there is also a way for the model to figure it out. 

In order to explain our approach, I will use an example of installment_payments data source. 
What we did: 

 1. Add a “TARGET” column to installment_payments, by merging on SK_ID_CURR with application_train
 2. Run a lgbm model to predict which records have 0 or 1 target. This step allows a model to abstract “what type of behavior in installment payments generally leads to default in loan in current_application”. You can see in the attached example (2. installment_payments_example.xlsx), that the same “fraudulent” behavior (like missing money = installment amount - payment amount) long time ago receives lower score compared to more recent similar behavior. We concluded then that the model learned to identify fraudulent behavior at “behavior” level (without the need to aggregate or set time thresholds)
 3. From step 2 we receive bunch of oof predictions for every SK_ID_CURR on row-level in installment payments. Then we can aggregate: min, max, mean, median, etc. and attach as features to main model.

We ran the same procedure on all data sources (previous application, credit card balance, pos cash balance, installment payments, bureau, bureau balance). OOF aggregated features on all of them added value, apart from Bureau Balance, which actually decreased cv and we didn’t use it in the end.
We received very low auc on those “nested models”:

 - previous application: 0.63
 - credit card balance: 0.58
 - pos cash balance: 0.54
 - installment payments: 0.58
 - bureau: 0.61
 - bureau balance: 0.55

The low auc scores for these models were hardly surprising, as they carry enormous amount of noise. Even for default clients, majority of their behaviors are OK. The point is to identify those few behaviors that are common across defaulters. 

This gave us 0.002 improvement on CV/ 0.004 improvement on LB, on our strong model with &gt;3000 features.

## 2.1  Another approach on credit card balance features ##

We have observed that there is a significant difference in train and test sets in ‘loan types’.
The cash_loans to revolving_loans ratio in train and test set is respectively ~90/10 and ~99/1.  That is why we have decided to use features generated on credit card balance set only in one final submission. 

Our observation was that in the case of ‘nested feature’ generated on credit card balance set by one model was 0.58, in other case (using different approach of Light GBM) the AUC score was ~0.64. 
Despite the fact that only one third of the train and test sets we labeled by this feature we have trained model with only one additional feature and improvement on CV was shocking! 
From 0.805 to 0.811! This corresponds with the ratio we have mentioned before. It appears to be confirmed in our final result - that it is an overfit caused by cash_loans / revolving_loans ratio.


## 3. Model to predict the interest rate ##

We were wondering for a long time why the organizers did not share cnt_payments for current_application. And then it was clear - if we know amount of credit, amount of annuity and number of payments, we can calculate interest rate from the following formula:
Annuity x CNT payments / Amount of Credit = (1+ir) ^ (CNT payment /12).
This formula assumes annual payments, while in reality the payments were done in monthly fashion, but it was good enough as a proxy.

The interest rate is the measure of risk that a current Homecredit model assigns to a customer (especially within given duration of loan). Hence, knowing interest rate means to some extent knowing the risk assessment from current HomeCredit model. From that moment on we set out on a journey to guess the interest rate of loans in train and test sets.

**3.1 First iteration: prediction of interest rate based on business understanding of credit products**
The key to predict interest rate is to understand that credit duration (or cnt payment) is not a continuous variable - it belongs to a very specific set of values. When you look at cnt_payments from previous application, you see that majority of loans have a duration which is a multiple of 6. It makes sense - you don’t take a loan for 92.3 days, rather you take it for half a year, one year, 2 years etc. So, at the very beginning I assumed that duration can belong to a following set of values (in months): [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]. 

What can you do now? Let me explain it using a loan with 35.000 credit value and 1.000 annuity (both of these values are known for applications in train and test).What you can do, is iterate over different durations and check if interest rate makes sense. 

Have a look at attached file 'ir_table.jpg'. From this table you can see that only two interest rate values make sense: 10% or 11% (I assumed, based on previous applications, that around 8.5% is the lowest interest rate that a loan may get at Homecredit). Which one is correct? We don’t know, so we can compute various characteristics of “possible interest rates”, like min, max, median, etc, and let LGBM find the best correlation with TARGET. It turned out, that a minimum value of such a “possible” set of interest rates was a very good predictor of the actual interest rate, until…

**3.2. .. we understood, that you can actually use previous applications to build a model to predict interest rate.**

For majority of previous applications, you can calculate interest rates, because you have cnt_payment, credit amount and annuity. Then, you can treat interest rate calculated in such a way as a Target, and use features common between current application and previous application as features to explain it. But still, the most important features are those from the table above: min, max, mean, median over possible set of interest values. Features like credit amount, annuity amount etc. are only ‘helper’ features, letting the model choose properly between different possible values of interest rates.

We achieved 0.02 rmse on model to predict interest rate and 0.50 rmse on model to predict duration, both based on oof predictions tested on previous application. These models used only features common between previous application and current application, so that we could use them to predict interest rate and duration for loans in current application.

Interest rate prediction gave us 0.002 improvement on CV/ 0.004 improvement on LB, on our strong model with &gt;3000 features.

## 4. Using scoring model based on banking mechanism taken from articles and meetings ##

Another issue that we were exploring during the competition was the broad idea of currently used credit-risk-scoring banking models. We read some articles and conducted a series of meetings with people involved in credit risk scoring in banks. Thus, we have established that what we could create is a basic scoring models – logit and probit.

Firstly we have prepared the list of features from application_train, previous_applications sets and aggregate them into groups.

Then we have created two models:
 - logit
 - probit
 
The scores of those models are respectively 0.56 and 0.59. Nevertheless features created from those models (OOF) improve CV score by 0.0013 -&gt; 0.001 LB


## 5. Some additional ideas ##

 - Dealing with categorical features as numerical. For example, app_data.replace({'NAME_EDUCATION_TYPE': {'Lower secondary': 0, 'Secondary / secondary special': 1,  'Incomplete higher': 2, 'Higher education': 3, 'Academic degree': 4}}), prev_app_df.replace({'NAME_YIELD_GROUP': {'XNA': 0, 'low_action': 1, 'low_normal': 1, 'middle': 3, 'high': 4}})
 - While making aggregation features, GroupBy data not only by one feature but also by combination of features. For example, by (‘CREDIT_TYPE’+’CREDIT_ACTIVE’).
 - Using grouping by ‘time windows’. This allows you, for example, follow the change of credit card balance in different periods of time.

## Summary ##

Again, we encourage you to try out attached features in krakowlublinzhabinka_feats.csv and check out their impact on your score.

That’s it -  it’s over now. We must admit, it was quite a journey for us, and we feel we moved mountains to get these features and scores. Thank you, and see you again in the following competitions. We will be back, stronger with this experience.

We would like to finish with this citation from Churchill and dedicate it to everyone in this competition:

*It is not the critic who counts; not the man who points out how the strong man stumbles, or where the doer of deeds could have done them better. The credit belongs to the man who is actually in the arena, whose face is marred by dust and sweat and blood; who strives valiantly; who errs, who comes short again and again, because there is no effort without error and shortcoming; but who does actually strive to do the deeds; who knows great enthusiasms, the great devotions; who spends himself in a worthy cause; who at the best knows in the end the triumph of high achievement, and who at the worst, if he fails, at least fails while daring greatly, so that his place shall never be with those cold and timid souls who neither know victory nor defeat.* 

Pawel, Michal and Aliaksandr, aka “Kraków, Lublin and Zhabinka”


  [1]: https://www.kaggle.com/c/home-credit-default-risk/discussion/57175
  [2]: https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
  [3]: https://www.kaggle.com/c/home-credit-default-risk/discussion/58332
