# 2nd Place | Single LightGBM and Target Encoding

Competition: playground-series-s5e4
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s5e4/discussion/575840

My solution consists of a single LightGBM model and target encoding of features. I've already shared 90% of the work publicly.

I've never been great at writing up solutions, so I just shared the data generation [notebook](https://www.kaggle.com/code/greysky/podcast-dataset-generator).

The final processed training dataset contains 1552 features and 794868 rows. Fortunately, with careful data type casting and avoiding unnecessary copy operations, I was able to train the model on Kaggle CPUs. Training takes about 4 hours. I trained on all the data using 5 different seeds.

#### LightGBM Hyperparameters
* objective = 'l2'
* metric = 'rmse'
* n_iter = 12000
* max_depth = 15
* learning_rate = 0.008
* num_leaves = 480
* colsample_bytree = 0.25

#### New Features 
* Mul_Hpp_Elm = Host_Popularity_percentage * round(Episode_Length_minutes)
* Mul_Gpp_Elm = Guest_Popularity_percentage * round(Episode_Length_minutes)
* Rounded_Episode_Length_minutes = round(Episode_Length_minutes) // 2 
* Rounded_Host_Popularity_percentage = round(Host_Popularity_percentage) // 2
* Rounded_Guest_Popularity_percentage = round(Guest_Popularity_percentage) // 2

#### Target Encoded Features | pair_size = [1, 2, 3, 4, 5, 6]
* Podcast_Name
* Episode_Length_minutes
* Episode_Num
* Episode_Sentiment
* Host_Popularity_percentage
* Guest_Popularity_percentage
* Number_of_Ads
* Publication_Day
* Publication_Time
* Rounded_Episode_Length_minutes
* Rounded_Host_Popularity_percentage
* Rounded_Guest_Popularity_percentage

#### Descriptive Statistics on Target Encodings (Column-wise)
* Mean, standard deviation, min, max (aggregated globally)
* Mean, standard deviation, min, max (aggregated by pair_size)
* Mean, standard deviation, min, max (aggregated by source column)
