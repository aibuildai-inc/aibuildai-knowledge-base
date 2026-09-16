# 36th place solution

Competition: ga-customer-revenue-prediction
Rank: #36
Source: https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/82746

# Just show me the code!

1. Initial data analysis: https://www.kaggle.com/returnofsputnik/data-analysis-using-r
2. Model: https://www.kaggle.com/returnofsputnik/gstore-r-only-test-data

# Abusing the leak before the reset

The first thing I did when I saw the Google Analytics Demo Account was posted online was to try to familiarize myself with the environment. I was searching around trying to understand it, and then I clicked the User Explorer tab. I found out that the Client ID for each user actually had the visitID of the first session of the visitor. And I found that if I click on this user, I can map each visit to how much he spent based on the number of pageviews he made in that session. So, I simply exported all of this information, and was changing my submission based on what this user export said! This is how I was able to skyrocket to #1 on LB before the reset, and it spawned the "Insight or Dodgy?" discussion that led to this competition's reset.

# Construction of solution

## Validation Set

First we must realize that the test set was going to be graded on December 1 2018 to January 31 2019. Therefore I reserved December 1 2017 to January 31 2018 as my validation dataset, instead of using the standard May 1 2018 to October 15 2018 that was issued as the "test set". I did this because I thought this period was holiday/Christmas period, and therefore I thought the behavior of the customers during this period would be significantly different from the behavior of the customers during other periods.

## Data Preprocessing

First, I convert all columns with 2 unique values into binary. I then impute some NAs based on what they should logically be. I updated some values, that for anyone who "bounced" from the website, their sessionQuality was set to 1 (the lowest possible) and their timeOnSite was set to 1 (the lowest possible).

For everyone who didn't bounce, and there was no logical imputation for NA, I imputed with the median. I created new binary columns representing if the column was originally NA or not.

## Feature Engineering

I did a lot of feature engineering in this competition:

1. Time until final visit
2. Time of last visit (max visit)
3. Time of first visit (min visit)
4. Difference between time of last and first visits
5. Difference between current visit and first/last visits
6. Ratio of 5 and 4
7. Do the same for visitId (I do this because visitId doesn't always go up by 1 for each visit, so maybe a gap in visitId signified something)
8. Take log1p(abs(x)) of these feature engineered to homogenize values
9. Take the length of the gclId 
10. Obtain the last 3 characters of the gclId (this can be a new category)
12. Consolidate browsers into fewer categories based on research. Include a new category called "nonhuman" for spiders and scrapers
13. Consolidate operatingSystem into fewer categories
14. Get the timezones for each country, being more specific when countries have multiple timezones by looking into the city
15. Obtain the "local" visitStartTime by converting the UTC visit start time into the local time
16. Obtain hour, day of year, weekday, week, month, and year values from the local visit time.
17. Append GDP data to all countries
18. Append binary column to say if English is the country's official language
19. Using Bing Maps API I was able to get the longitude and latitude of every city. Append the longitude and latitude as features
20. Compute Haversine distances from San Francisco given the coordinates
21. Compute Bearing distances from San Francisco given the coordinates
22. Consolidate adwordsClickInfoSlot
23. Replace unseen countries with similar countries (e.g. Samoa replaced with American Samoa, Micronesia replaced with Vanuatu, etc.)
24. Consolidate trafficSource links (e.g. anything that has "amazon" in it, just let it be "amazon.com"; anything that has "kik" in it, let it be "kik.com". etc.)
25. Binary column if trafficSource has the word "google" in it
26. Binary column if trafficSource has the word ".com" in it
27. Binary column if referralPath has the word "google" in it
28. Binary column if the referralPath has the word "youtube" in it
29. Binary column if the referralPath has the word "intl" in it
30. Binary column if the referralPath has the word "html" in it

## Aggregation to user-based level

Recall this is at a visit-based level, but we need to make user-based predictions. Therefore I had to aggregate all these features by taking maxes, mins, sums, means, standard deviations, heads, tails and modes. The aggregations I chose were based on what I thought would be important. I also made some aggregations based on time, e.g. I took the mean of the customer's lifetime and the mean from the last 6 months, because I saw that features like these can improve the model from the previous Home Credit competition.

I took logs of some aggregated features if I found their distribution was better. This was determined using kernel density plots.

## Validation Modeling

There were two parts to my solution:
1. First, make a model to predict if the buyer will make a purchase in the future
2. A regression that predicts how much they will spend in the future

First, I tuned my parameters using the validation set. I predicted 0 or 1 did you make a purchase or not. Performance was looking good.

## Prediction Modeling

Once I was satisfied with my validation, I retrained the model on the entire dataset using 4 StratifiedKFold. This is one of my errors--I should have used TimeSplitFold instead.

I generated binary OOFs to feed to the regression, and I achieved an AUC of 0.95255 and Log Loss of 0.0020877.

Add these predictions to a LightGBM fitted to minimize MSE. This got my mean absolute error to 0.01147 and my root mean square error to 0.320785.

Run this procedure 4 times using 4 different seeds and average the results. Then you obtain my submission!

Things I tried that didn't work:

1. Tons of ratios between the aggregate features, based on the 6 month aggregate versus the lifetime aggregate.
2. I had this idea in my head that perfecting the CV on my validation set wasn't a good idea, because I would be "overfitting" the validation set instead of allowing it to generalize to a test set. I should have looked into observing the difference between my training losses and validation losses to see how well the models were generalizing, and based on this make my decision.

Things that I know to do for next time:

1. Obviously, the assumption of seasonality for the test period was false based on EDA. I should have explored this before making the assumption, and if I did, I would have made a bunch of windows to train on like many other top competitors did. This would have added model strength
2. I should have tried multiplying the probability of a purchase by the regression, instead of simply feeding that as a feature to LightGBM. In fact, the #1 submission in this competition and the #1 in the recently ended Elo Merchant competition both used this technique of multiplying the regression by probability.
3. I should have looked into reducing the dimensionality of my featureset. I doubt every feature was useful, and because there were so many features I think this harmed the ability of LightGBM to perform well.
