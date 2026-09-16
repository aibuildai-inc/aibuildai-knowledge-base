# 12th place insight/detailed approach

Competition: recruit-restaurant-visitor-forecasting
Rank: #12
Source: https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/49251

First, I want to thank Recruit and Kaggle for hosting this exciting competition. As a relatively new Kaggler, it was a really enjoyable competition.  
Secondly, I want to encourage all other new Kagglers. I have no ML/stats/math background. If I can get a gold medal, you can, too :)  

Now, I want to present some of my findings which got me a gold medal, and then my detailed approach.  

<h1>Findings</h1>  
##Next day is holidays##
![distribution of visitors][1]  
You can see from this image, that the number of visitors on holidays are,  
(a) Close to weekend mean.  
(b) If you divide the holidays to whether the next day is a holiday or not, it will become quite close to a Saturday or Sunday.  
I dug further into this finding, and I found that it was a very consistent pattern accross the year.  

Features from this finding:  
 - whether the next day is a holiday or not   
 - "dows" which converts all holidays into Saturdays(5) or Sundays(6), depending on  the next day  

Score improvement: 
0.003, securing my gold medal.  
![Score difference with the feature][2]

What I missed:
- I should have done a "dowsf" feature, converting every day before a holiday into Friday(4)  


##Reserve data##
![most_reserve-visit_datediff_is_within_1week][3]
As you can see from the image, most of the reserves are made within one week from visit.  
That means that if you train your model based on the raw reserve data, your model will predict that visitors in May will be really low, because it has low reserve numbers (which should not be the case, because the reserve numbers are low just because we don't know yet on 2017-04-22)

Features from this finding: (I trained 5 models, split based on the week number it was going to predict)  
- shifted(lagging) reserve data:  
shift the reserve data by n weeks, n being the number of weeks the model was trying to predict into the future.  
- shrinked reserve data:  
only use the reserve data, which is within the models prediction. That means that I would use reservations that are "reserve_and_visit_date_diff &gt; 7" for my week2(2017-04-29 to 2017-05-06) model.

Score improvement:  
![massive_overfit][4]
- avoided a massive overfit, which let me climb 20 places in the private leaderbaord  
- roughly 0.001 to 0.004 improvement with reserve data, depending on the week  

#Detailed approach  
Before my detailed approach, I would like to thank @huntermcgushion for his weather data, @maxhalford for a lot of idea sharing, and @gertjac for his wonderful documentation of the Rossman competition.  

<h2>General Approach</h2>    
As a software engineer, I assumed that my strength would be in implementation skills.  
Basically, I tried to implement any idea that I could come up with, and then delete ones that would make my model overfit.
I am also able to read English (never underestimate what you can do with that ability), so I read a lot of winner's solution on Kaggle. I based my initial model on the 1st place solution in the Rossman competition.  

<h2>Basic model idea</h2>    
Based on lightgbm (best library in 2018 for liberaly throwing in data), treated the problem as a time series problem.  
I made two submissions, one was a simpler version, and one was complex ensembled variation using the same features.  
The simple version was a weekly model, which means I had 5 models in total.   
I also had a daily model for week2. The reason is that the number of reserve decays exponentially (see above image), so the reserve data is important on closer-to-public-leaderboard days.  

In the complex version, I trained stores that have data before 2016-03-01 (about 300/820 of the stores do) separately.
Those "good" store's prediction was based on the separate model, and the other "bad" stores were trained all together with the "good stores. I also trained my whole model without outliers (thanks @maxhalford). Finally averaged by equal weight, which is my final submission.  
The complex version improves my score by &gt;0.001. 

<h2>Features</h2>    
- basic features    
air_store_id, air_genre, air_area  

- rolling features  
groupedby (air_store_id), and (air_store_id, day_of_week)  
for 1, 5, 15, 55 weeks.  
mean, median, max, skewness  
shifted by n week (n being the week number that model was predicting)  

- temporal features  
is_holiday, is before holiday, dowh(dow(0-6), holiday as 7), dows(described in "findings" section),  
day of week, month, year, holidays in week, next week number of holidays, prev week number of holidays.  

- reserves  
air/hpg reserve sum, shifted by n weeks (n being the week number that model was predicting)  
air/hpg reserve sum, cut by number of weeks (so for week3, I only used reserve done 21days before visit)  

- regression  
Got this idea from the Rossman competition.  
For extrapolation, I blended in regression(Lasso and Ridge)  
Divide the whole data by months, and then for each month, do a regression grouped by air_store_id, using the previous 3months, previous 13months. Only use two features for regression: (visitors) is y and (day-delta, day of week) is x  

- extra ideas  
rolling medians divided by another rolling median 
exponential weighted mean  
weather data: percipitation and avg_temperature (improved my model roughly 0.001, thanks @huntermcgushion)  

<h2>Features that didn't work</h2>   
I did a manual greedy backward elimination of features. Yep, I just invented that term :)
That means that I commented out some of my features and tested if it would make my score better. If i made my model better, that would mean that the feature is bad, and they would go to garbage.  

Week of year, day of month were very overfitty.  
Rolling-kurtosis and log-tranformed features were slightly bad.  
Rolling-min and a lot of my dow-grouped features were slightly bad to no effect.(I dropped them for simplicity)  
To my surprise, the "prefecture" and "city" that I extracted from the area name gave me a worse score.  
The total of air + hpg reserve was also bad for my model.  

<h2>Validation</h2>   
I had 4 validation period, basically training with data till day n, and testing with day n+1 to n+7.
I made minor tweaks, like testing with n+8 to n+14, but the results were pretty consistent, so I stuck with this setup in the end.
train:(2016-01-16 to 2017-03-11) / test:(2017-03-12 to 2017-03-19)  
train:(2016-01-16 to 2017-04-01) / test:(2017-04-02 to 2017-04-08)  
train:(2016-01-16 to 2017-04-08) / test:(2017-04-09 to 2017-04-15)  
train:(2016-01-16 to 2017-04-15) / test:(2017-04-16 to 2017-04-22)  
I started my train period from 2016-01-16, because a lot of my rolling features were null in the beginning of 2016.  
I also want to note that I was extremely careful in information leaks.  
I am really surprised that the public kernels performed so well, after having so much information leak from the future.

I also did some manual holdout-validation for the golden week.  
Training with the same periods, but holding-out a few goldenweek-days from the train data.  
example: omit 2016-05-03, train with the 4 weeks mentioned above, then validate using 2016-05-03.  
The more goldenweek train days I included, my score improved, so I was kind of confident about my goldenweek.  
However, I did not make any features based on this validation, because we only have one year worth of golden-week, and I feared overfitting.  

My final submission was trained on the (2016-01-16 to 2017-04-15) / (2017-04-16 to 2017-04-22) split.  
The reason being that there were stores that appeared only from 2017-03, and I wanted to squeeze in any available data for that store.  

<h2>What I missed</h2>     
I should have had "air_genre" grouped rolling features as well. How the **** did I not think about that?  

<h1>Final words</h1>     
Question: Why are there so many national holidays in Japan?  
Answer: The government is upsampling holidays, to make kagglers better predict the number of restaurant visitors on holidays.  
(Joke from twitter)

  [1]: https://kaggle2.blob.core.windows.net/forum-message-attachments/279665/8521/holiday_difference.PNG
  [2]: https://kaggle2.blob.core.windows.net/forum-message-attachments/279665/8520/dows_difference.PNG
  [3]: https://kaggle2.blob.core.windows.net/forum-message-attachments/279665/8525/reserve_most_in_a_week.PNG
  [4]: https://kaggle2.blob.core.windows.net/forum-message-attachments/279665/8526/huge_overfit.PNG
