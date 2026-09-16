# 9th Place Solution - Inference

Competition: g-research-crypto-forecasting
Rank: #9
Source: https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/324180

Hello everyone,

Firstly, I want to thank you everyone in this competition, it was a great journey for me and I'm very happy to win a gold medal for the first time.

I actually shared my inference [notebook](https://www.kaggle.com/code/bturan19/lgb-3fold-rollingagg-lagtarget-submissioninference/notebook) long ago. Now I want to give some information about it..

My most precious feature was Hull moving average:

```#@jit(nopython=True)
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.append(np.array([1]*n), ret[n - 1:] / n)[1:]

#@jit(nopython=True)
def calcHullMA_inference(series, N=50):
    SMA1 = moving_average(series, N)
    SMA2 = moving_average(series, int(N/2))
    res = (2 * SMA2 - SMA1)
    return np.mean(res[-int(np.sqrt(N)):])

row["hull"] = last_close - calcHullMA_inference(f[asset]["all_close"][-260:], 240)
```

Other important note about creating lag feature, is the window size. I choose my windows with a Fibonacci sequence: fibo_list = [55, 210, 340, 890, 3750]. You can see that in notebook.

Last important explanation about my feature is using lag target. Firstly I tried to use the official calculation but then I just gave up and use this:

```row["target_return"] = (last_close / f[asset]["all_close"][-16]) -1```

and for a market indicator, I collected new "target_return" for every batch, and get the average of last available list.

Last important note for my work is models. I have 3 different LightGBM models that were trained for different market conditions. Up market, down market and relatively more stable market. Then I get the average of them.

I thought about how could I improve my models. First thing is: I could do parameter optimization, I almost used default parameters. Then I could add different kind of models, I thing my submission time was around 6 hours. lastly, I had actually more feature in my mind.

Well, thanks to everyone again.. See you in next competition.
