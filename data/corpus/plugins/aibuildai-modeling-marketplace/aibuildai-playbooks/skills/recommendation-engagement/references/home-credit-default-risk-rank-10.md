# 10th place writeup

Competition: home-credit-default-risk
Rank: #10
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/64598

# Our Team Members:
@fatihozturk @lingchensun @shivrajp @a31314431 @kingychiu

Congrats everyone and thanks Kaggle and Home Credit for hosting such an amazing competition! 

# My Top Features and Some Final Words @fatihozturk 
I want to talk about my part first and some final words. I also used lightgbm as my best single model. During competition, I read papers and blogs about existing credit scoring methods and factors signing loan default. Got some insights and generated features based on them. Some didn't work, but most of them worked. I won't go much in details but will mention about my top features.
Because we have sort of time series data, it made sense to generate features using dates close to current application.

LATE_PAYMENT feature: I think most of you used installment['DAYS_INSTALMENT']-installment['DAYS_ENTRY_PAYMENT'] as a new feature since it shows number of delays. It was a already good feature but I realized that, when you take last 365 days of installment data and aggregate this delay as a new feature, it added much more value to model. Here is a simple code:

    installment_temp = installment[installment.DAYS_ENTRY_PAYMENT &gt;= -365]
    installment_temp['LATE_PAYMENT'] = installment_temp['DAYS_INSTALMENT']-installment_temp['DAYS_ENTRY_PAYMENT']
    late_payment_feature = installment_temp.groupby('SK_ID_CURR')[['LATE_PAYMENT']].min().reset_index()

CREDIT_UTILIZATION feature: In blogs and papers I've read, most of them was saying credit utilization is a strong indicator for a risky customer and you can calculate it simply by dividing Credit card balance by credit card limit. It was making sense but didnt work when I calculated on full credit card dataset. Again I tried checking results on more recent credit card data. Finally I've found that last 2 months' data was more valuable for my model. You can see the simple code and check yourself with different months:

    month = -2 
    cred_temp = cred_card_bal[cred_card_bal.MONTHS_BALANCE &gt;= month]
    cred_temp['CRED_UTIL'] = cred_temp['AMT_BALANCE'] / cred_temp['AMT_CREDIT_LIMIT_ACTUAL']
    cred_util_feature = cred_temp.groupby('SK_ID_CURR')['CRED_UTIL'].max().reset_index().rename(columns={'CRED_UTIL':'CRED_UTIL_'+str(month*-1)})

Most important buraeu feature: Debt ratio was the most important feature I found in bureau file. I also tried using more recent bureau data for this feature but it was the best when I used all data.

    buro['DEBT_RATIO']=buro['AMT_CREDIT_SUM_DEBT']/buro['AMT_CREDIT_SUM']
    debt_ratio_feature = buro.groupby('SK_ID_CURR')['DEBT_RATIO'].max()

I was also at Santander competition and had a very intense 3,5 months. For me, our team was amazing. Now I have 4 new friends at different continents. I recommend you to team up with someone not only having a good single model score but also having full time for the competition and nice 128 cores machines ;)

# Interest Rate Features &amp; Cash Loan Model @a31314431 (FuLin)
Again want to say thank you to my teammates, sponsors and everyone for making this competition a great experience for me. Here is my two cents:

**Interest Rate Features**

Interest rate seems to be an important feature in predicting loan default.(You can see this from the lending club dataset) We were not given interest rate directly in either Main Application data or Previous Application data but we could approximate it: 

Previous Application - The key thing here is that AMT_ANNUITY includes interest. Based on AMT_CREDIT, AMT_ANNUITY, and CNT_PAYMENT we can derive interest rate.

    prev_app['INTEREST'] = prev_app['CNT_PAYMENT']*prev_app['AMT_ANNUITY'] - prev_app['AMT_CREDIT']
    prev_app['INTEREST_RATE'] = 2*12*prev_app['INTEREST']/(prev_app['AMT_CREDIT']*(prev_app['CNT_PAYMENT']+1))
    prev_app['INTEREST_SHARE'] = prev_app['INTEREST']/prev_app['AMT_CREDIT']
		
Then by calculating max, min, mean of features above for each customer, I got several top features for Previous Application table.

Main Application - We were not given CNT_PAYMENT in Main Application data and but we could predict it from Previous Application table. I built a lightgbm from Previous Application table by using AMT_CREDIT, AMT_ANNUITY, AMT_CREDIT/AMT_ANNUITY to predict CNT_PAYMENT. The model got a RMSE ~2.78. After applying the model on Main Application table, I got predicted CNT_PAYMENT(EXP_TERM) for each record and then I created exact same features as I did for previous app table. After applying the model on Main Application table, I got predicted CNT_PAYMENT(EXP_TERM) for each record and then I created exact same features as I did for previous app table.

    main_app['INTEREST'] = main_app['EXP_TERM']*main_app['AMT_ANNUITY'] - main_app['AMT_CREDIT']
    main_app['INTEREST_RATE'] = 24*main_app['INTEREST']/(main_app['AMT_CREDIT']*(main_app['EXP_TERM']+1))
    main_app['INTEREST_SHARE'] = main_app['INTEREST']/main_app['AMT_CREDIT']
	    
Overall, several of these interest rate related features joined my top 10 lgb features and gave ~0.0006 boost in CV.

**Cash Loan Model**

Looking at the NAME_CONTRACT_TYPE in Main Application table, we have ~90% cash loans while ~10% revolving loans. I had a feeling that the default behavior may vary across this two types of loans so why not built a separate model only for cash loans and use this new model to correct the cash loan prediction from the big model? By blending this new cash loan model with the original model, one single lgb model could get a ~0.0006 boost on CV, and ~0.0004 on LB and PB.

# Records Filtering and Division @kingychiu (nlgn)
Basically, I used nested for loops, for tables other than application.csv, they were filtered into sub-tables:

    active_bureau_df = bureau_df[(bureau_df['CREDIT_ACTIVE']=='Active')]
    closed_bureau_df = bureau_df[(bureau_df['CREDIT_ACTIVE']=='Closed')]
    ...

For each sub-tables, then they were divided by “time”:
2 different “time” definitions were used:

1. n months before the application, (3, 6, 18, 30, 42, 54, 66 months before)
2. The latest k records, the earliest K records, k was the ratio (0.1, 0.2, 0.3, 0.4)

Then the tables were grouped by [‘SK_ID_CURR’] and [‘SK_ID_CURR’, ‘SK_ID_PREV’] and aggregated mainly by mean and max, sometimes sum, min and seldom median. Trend aggregation was also used. It was inspired by the Open Solution.

My model: https://github.com/kingychiu/Kaggle-Home-Credit-Default-Risk

# Predicting Missing Values @lingchensun (Chad)
Thanks again for my teammates’ excellent work! The only thing I want to share is how to improve cv by predicting EXT source. As we know, EXT features are the most important features in this game, however, they are filled with too many NA. 

So, I built a lightGBM model to predict EXT 1 and 3 with similar features of the default model. My prediction helped all of my teammates boost CV at least 0.0005. I also attached the prediction files, feel free to use them to boost your model too! Enjoy boosting!

# Boosting Best Single Model @shivrajp (Shivraj)
As our single model was pivot for us finishing so well when the dust settled, my two cent contribution was to fix the lower folds CV, which got a bump while using the same best params but by changing max_depth =-1 and blending them together always helped fix our CV on folds scoring low. 
PS: Our single model could have won Gold by itself ;) 

#Feature Selection
- Lightgbm's gain importance was used initially to cut down features, at that time our best single model performance was: CV: 0.80368, Public lb: 0.80386, Private lb: 0.80048.

- Olivier's script was used later, the best single model became: CV: 0.80500, Public lb: 0.80533, private lb: 0.80207

# Blending/Stacking
For a long time, we only used weighted blending predictions based on out of fold files. Our CV and LB went highly correlated. However, after we hit 0.806 CV score, Our Lb Score started to came 805 and it was coming lower 805 at each improvement. This part was kind of annoying. Even with 0.807 CV we were getting 0.805. 

Then, last day and actually the last hours @fatihozturk suggested stacking with a simple tree model and adding that model also to blending. Thanks God, it gave us 0.806 LB but very close to 807 with 0.807+ CV . This submission happened to be our best private score.

@fatihozturk's quick notes for stacking: Having different folds and seed number for different model is not a problem and never use early stopping during stacking, it ruins oof score of 2nd level model.

We didn't share our top features among us and we all had single 0.80+ CV models. This resulted in a good blending score which was 0.807+ since we had diverse models. However, our private score came 0.803+ and don't know what caused this.(I think chance took its part here).

# Final Submissions:
1. Blend (stack + best public score blend): CV: 0.807420, Public lb: 0.80674, Private lb: 0.80354
2. Blend of 12 models: CV:0.807640, Public lb: 0.80590, Private lb: 0.80346
