# Not real 4th place solution

Competition: nyc-taxi-trip-duration
Rank: #4
Source: https://www.kaggle.com/c/nyc-taxi-trip-duration/discussion/39553

Hi, all

Thanks to the hoster and all Kagglers that contributed to the discussion and kernels. Because of the data leak, this rank is not precise. But still, I'd like to share my solutions.

**Features:**
Thanks to beluga's sharing. Most of my features are based on his strategy.

Apart of this part, I also clustered the center coordinate of the trip and created similar aggregated features such as average speed and count of center cluster based on time series.

I discretized the feature of direction into 4 bins (southwest, southeast, northwest, northeast) and 8 bins.

For the date_time feature, I only created a new feature called is_rest_day which assigns 1 to weekends and holidays, 0 to business days.

A new average speed feature using total_travel_time in fastest_route data set divided by trip_duration.

More aggregated features (average speed and count) are created through grouping date_time features(hour, week_hour, is_rest_day), cluster features(pick_up_cluster, drop_off_cluster, center_cluster), vendor_id and discretized directions. I totally have 18 different groups of aggregated features. Since there are lots of correlations among them, I divided them into 4 different data sets.

I rotated the latitude and longitude to make the Manhattan road direction correspond with the new direction. For example, the whole 5th Ave will be on the same longitude. Then I calculated the trip distance on new North-South direction and East-West direction respectively. Two new direction feature: (North or South) and (East or West). I also clustered the new coordinate and  generated some aggregated features.

For the fastest_route data set, I applied three solutions. 

 1. Firstly, I created a dict for each trip of their routes and relative
    distances, for example: {'5th Ave': 500, 'W 26th St': 180}  (Just an
    assumption, not real number). Then I used DictVectorizor to create a
    sparse matrix. The final step is applying SVD on this sparse matrix
    and using the top 50 features which explained about 87% of the
    variance.
 2. I aggregated routes frequency by hour and week_hour. For
    example, W 26th St occurs 200 times in Wednesday hour 16. Such
    frequency may represent the traffic situation of roads within
    specific time. Then, for each trip, I calculated their mean and sum
    frequency for the routes it went through.
 3. Following the second solution. I multiplied the frequencies by route distances of each trip, created new dictsm transformed into a new sparse matrix and applied SVD.

**Model:**
A simple 2-layer stacking. 
1st layer: Since I have 4 different aggregated data sets. For each data set, I applied a XGB, a RandomForest, a ExtraTrees, a DecisionTree and a Linear Regression. Totally 20 models

2nd layer, just a carefully tuned XGB.

That's all what I did. Thanks!
