# [22nd] place solution: Only Wifi Models + Public Post Processing + Pseudo labels

Competition: indoor-location-navigation
Rank: #22
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/240010

I would like to thank kaggle for hosting such an interesting competition and my amazing teammates @shivamcyborg @dehokanta @shivammittal274 @nooblife .Also,  Congratulations to Tom & dott's team for another win and @mamasinkgs team for dominating the competition in most half of the competition. 
I would like to share our approach which did quite well for us and led us to secure a decent position on leaderboard.
Firstly, all our models were trained on just wifi features BSSID & RSSI with diverse approaches. We couldn't find any better way to include more features into our model, most of them performed poorly on both validation score & leaderboard, hence we decided to stick completely with Wifi data.

#### Models:
1.Fastai/ RNN : This was our best model with highest weightage in ensemble, baseline was built by @nooblife, I was really amazed by its performance. One of the reason it performed better was generating the wifi data within the last 5 seconds for each position and sending them sequentially to our model. With some more parameters tuning and slightly different architecture, we managed to reach 5.26 with single model without any post processing.

2. Site Wise RNN training: We also trained similar lstm model site wise, although its result were not as good as lstm on complete data, but it did good boost in ensemble because of low correlation. 

3. MLP: It also did quite well in our final ensemble of models, giving slight boost when merged with lstm models.

Our models ensemble without any post processing scored **4.8-4.9** on public leaderboard.


#### Post Processing

Our final solution is solely based on public post processing with some additional tweaks and tricks. Although, we come up with brilliant way to boost local as well leaderboard score by 0.2-0.3 through post processing but with the help of hand labelled waypoints.  The idea behind it was doing post processing in loops with different thresholds for snap to grid, we might have reach a lot better position than current if hand labelled grids were allowed, I hope no team is using it in current standings.
Our models score with post processing and optimised cost minimisation : **3.57**

#### Pseudo Labels

The idea behind it was simple, as some of the waypoints were being changed by a good margin after post processing, we planned to retrain some of our models along with train+test data to get better predictions for some paths.
This led to our final score reach: **3.43** after adding pseudo label models in ensemble and making us jump to 22nd position on private leaderboard.

Baseline version of rnn: https://www.kaggle.com/nooblife/indoor-location-rnn-v2
Preparing data: https://www.kaggle.com/nooblife/indoor-location-rnn-data-v2


### Things that didn't worked: 
- Transfer learning by using other buildings than test data.
- Imu+wifi features models
- lgbm
- post processing based on shapely

### Things we couldn't finish:
- Automated grid points
- Efficiently use other data with wifi features
- Using time features efficiently 
- in depth EDA of predictions

#### References 
https://www.kaggle.com/kokitanisaka/lstm-by-keras-with-unified-wi-fi-feats by @kokitanisaka
https://www.kaggle.com/saitodevel01/indoor-post-processing-by-cost-minimization by @saitodevel01
https://www.kaggle.com/robikscube/indoor-navigation-snap-to-grid-post-processing by @robikscube
https://www.kaggle.com/tomooinubushi/postprocessing-based-on-leakage by @tomooinubushi

It was really nice competing in this competition, I wish I could have joined it earlier instead of last 2-3 weeks. I think we did pretty well without use of hand labels and small data. Thank you for reading our solution.
