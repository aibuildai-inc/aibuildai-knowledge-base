# 15th Place, Interesting Features - No Phishing

Competition: optiver-realized-volatility-prediction
Rank: #25
Source: https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/276137

My thanks to Optiver for hosting a really interesting competition that I very much enjoyed digging into. I’d like to share some of the more compelling features I've found that haven’t been discussed. I've also made my 15th place notebook [public](https://www.kaggle.com/jacobyjaeger/15th-place) though I can't say much for it's readability. '

The features I will discuss include extensions of weighted average price that incorporate all book orders, measures of orderbook liquidity, and a feature derived from the previously mentioned that had a remarkably high correlation with volatility.

## Extending WAP

First of all, the weighted average price definition we were given minimizes the following function: `f(x) = -bid_size*log(x-bid_price) - ask_size*log(ask_price-x)`  over the range `[bid_price, ask_price]`. Given this fact we can extrapolate to a wap definition that includes multiple levels of market depth `wap =argmin_x(sum_i[bid_size_i*-log(x-bid_price_i) + ask_size_i*-log(ask_price_i-x)  ])`.

I identified this by starting from the following assumptions:

1. There exists some two functions `g_ask(x)` and `g_bid(x)` such that `WAP=argmin_x[ bid_size*g_bid(x - bid_price) + ask_size*g_ask(x - ask_price)]`
2. `lim(g_ask(x), x->0)=inf` and `lim(g_bid(x), x->0)=inf`

The first assumption is made because it is necessary if I am going to have a hope of finding a useful function minimized by WAP and because it is true of a traditional weighted average (with `g_*(x)=x^2`).

The second is made because logically the WAP should never be greater than the lowest ask or lower than the highest bid, and for that to always be the case, g_*(x) must have a vertical asymptote at 0.

Of course, `log()` is not the only function with a vertical asymptote, and this leads to a family of alternative wap definitions: `wap_k =  argmin_x(sum_i[ bid_size_i*(x-bid_price_i)**-k + ask_size_i*(ask_price_i  - x)**-k`. 

Unfortunately features derived from these alternative waps did not provide me much of a performance benefit. However they did lead me to a very important other set of features: liquidity.

## Defining Liquidity

In finance, liquidity describes how much of an asset can be purchased or sold immediately without moving the price significantly. A reasonable numerical definition of liquidity then should start from the following assumptions: 

1. moving bids closer to asks always increases liquidity
2. increasing order sizes always increases liquidity
3. adding additional orders to the book always increases liquidity

It happens that if we take the minimum of any of our alternative WAP functions, that is to say the value of the function at its respective WAP it will meet all of our conditions for a good liquidity measure. For example the liquidity derived from wap_1 is:

`liq_1 = sum_i[ bid_size_i/(wap_1 - bid_price_i) + ask_size_i/(ask_price_i  - wap1)]`

or for wap_2:

`liq_2 = sum_i[ bid_size_i/(wap_2 - bid_price_i)**2 + ask_size_i/(ask_price_2  - wap1)**2]`


## Trade Volume Per Liquidity (TVPL)

Now, liquidity is a useful feature on its own, but what’s really compelling is trade volume relative to liquidity. Trade volume divided by liquidity has a remarkably high correlation with volatility with `log(TVPL_2))` and `log(vol1)` having a correlation coefficient of about 0.88. I think it’s safe to say that trade volume per liquidity is most of what creates volatility in the short term. 

Recognizing the relationship between TVPL and volatility, there are a few more ideas we can explore. First, between trade volume and liquidity, trade volume is the far more noisy feature whereas liquidity is much more stable. So, for the sake of making projections, we should calculate trade volume over the full time window or at least most of it, but we can take liquidity over only a small portion to the latter end.  Many of my most powerful features play on this, some also incorporating aggregations across stock_id's. 


These have been the most interesting and helpful features I've found, but there are also other unique aspects to my notebook, so let me know if you are interested in having any more of them explained.
