# 7th place main ideas

Competition: optiver-realized-volatility-prediction
Rank: #7
Source: https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/276506

Firstly, many thanks to Optiver for sponsoring this competition. It allowed me to gain a lot of knowledge about finance and data science in general, and it ended up being very interesting.

##### About time_id aggregations
It's probably not a surprise to anyone that my public LB placement can be mostly attributed to "The Leak", so credit to whoever posted [this](https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/256725) (and then deleted their account).

I used a similar approach to the one described in @nyanpn's [post](https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/274970), with a few minor differences such as [this] (basically I also re-ordered the NN time-ids and derived some features from there)(https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/274970#1531808).

It goes without saying that such a solution could not actually be implemented in production, so this post will focus on all the other features & techniques I've learned about and that I haven't seen mentioned elsewhere.

I'd also like to say that I'm quite inexperienced in both data science and quantitative finance, and I don't know what my LB placement would be without time_id aggregations, so don't take this for a top solution. These ideas improved my CV and LB but I'm not sure if they would, by themselves, get me anywhere in the top LB spots.

##### Baseline & models
My submission started as a fork of the ["lgbm baseline"](https://www.kaggle.com/alexioslyon/lgbm-baseline), thanks to @alexislyon for providing such a helpful learning resource.

Although I changed most the the feature engineering, my final model is still based on a 50/50 LGBM and FFNN ensemble.

##### Trend
Most public solutions are using aggregations by seconds_in_bucket over 100 second time periods. I've found out that instead of passing those features as is, it is more efficient to fit a linear regression model on those features, and pass the mean, slope and error:

Function to get mean, slope, error of input array

```
def get_dir_stats(arr):
    mat = np.array([np.ones(len(arr)), np.arange(len(arr))]).transpose()
    coefs, err, _, _ = lstsq(mat, arr)
    return np.array([coefs[0] + (len(arr)/2)*coefs[1], coefs[1], err])
```
 
Whenever we would like to analyse the way a feature behaves over time, we would calculate that feature over 100 second time periods, and pass the array of those features (for example [rv_0-100, rv_100-200, ...]) to the above function.

This ended up improving my CV and LB by about 0.00100, and I believe this also helped improve readability, which in turn helped me for feature selection.

##### Regarding stock_id
Stock_id as a categorical feature wasn't used in my final submissions because I believe it wouldn't scale well with future data. As an extreme example, imagine what would happen if one of the stocks in our dataset was GameStop and the dataset was taken before 2021. 

I did use stock_id aggregations, but only over the N closest time_ids. This does require price denormalisation but I still think that would be the way to go even if we only had access to past time_ids for each prediction.

This probably isn't such an original concept but I haven't seen it discussed elsewhere. I'm interested to see what would be the most successful approach without time_id aggregations.

##### Feature selection
I used BorutaShap with an XGBRegressor model for feature selection. After running it on a kaggle kernel overnight I was still left with about 60 features I was unsure about. I tried re-running it again with the unimportant features removed, but I kept getting diminishing returns. If I had more time I could try to save the state of the program and resume it in a new session every 8 hours. Alternatively, I could try to get access to a decent CPU and run it locally.

Interestingly, most of the features deemed unimportant were domain specific (most quarticity features, a lot of trade features from the "lgbm baseline" notebook etc.) and not part of the increasing amount of aggregated features.
