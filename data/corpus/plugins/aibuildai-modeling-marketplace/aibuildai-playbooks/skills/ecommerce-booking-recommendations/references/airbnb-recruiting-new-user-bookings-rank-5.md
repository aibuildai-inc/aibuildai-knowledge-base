# 5th Place Solution

Competition: airbnb-recruiting-new-user-bookings
Rank: #5
Source: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/discussion/18918

Hey,

I thought I'd share with you all some of what I did for this competition. I liked this one a lot because it allowed for some creative feature engineering. My final model was an ensemble of 8 XGBs, 4NN (nolearn Lasagne), 1 GLMNET, and 1 RandomForest (using 'ranger' package in R). I used my out-of-fold CV predictions to create a second layer XGB model. 

Feature Engineering:

+ Users:  
I extracted all the normal things from the dates. Then I created dummy variables for all of the categorical variables. Nothing fancy here.  

+ Sessions:  
Filled in blanks with "NULL" so it was a new factor level.  
  
 For each user:
 + Number of actions taken 
 + Number of
   unique Actions, ActionTypes, ActionDetails, Devices   
 + Sum, mean, min,
   max, median, s.d., skewness, kurtosis of seconds_elapsed   
 + Entropy of
   Actions, ActionTypes, ActionDetails, Devices, Secs_elapsed of the
   first, second, last, second-to-last, and third-to-last Actions
 + Ratios of unique Actions, ActionTypes, ActionDetails, Devices
 + Ratios of entropies 


 Then I casted the sessions frame so each level of Action, ActionType, ActionDetail, and Device had its own column. I did this twice, once using the count and the second time using the sum of seconds_elapsed. Sometimes I would use the raw count and sec_elapsed values for the actions, actiontTypes, etc. and sometimes I would convert them to proportions. Additionally I sometimes computed similarity/distance matrices, and used nearest neighbors to diffuse the matrices. Information on this can be found [here][1].

I said in another thread that I tried using n-grams to capture the sequence of actions and it didn't really help at all, but my single model that scored the highest on the private board used n-grams, so maybe it did help a little.

Then I created 12 "helper" columns, one for each destination. Basically what I did was find the average of each of my features for each country_destination, then found for each country_destination which features were most different based on standard deviation. For example, say the countries were [NDF, US, FR, IT] and their averages for "feature_1" are [1.04, 5.40, 0.50,  0.98], then "feature_1" would be one of the features I use in creating a helper column for the US. I find 30 of these columns for each country and sum them together and that becomes the "helper" column.



  [1]: https://www.kaggle.com/c/walmart-recruiting-trip-type-classification/forums/t/17632/department-features-correlation-graph
