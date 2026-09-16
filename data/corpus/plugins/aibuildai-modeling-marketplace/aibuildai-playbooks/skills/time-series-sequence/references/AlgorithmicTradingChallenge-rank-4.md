# Winning Algo/Code

Competition: AlgorithmicTradingChallenge
Rank: #4
Source: https://www.kaggle.com/c/AlgorithmicTradingChallenge/discussion/1236#7778

<p>This was an interesting contest! &nbsp;Many thanks to the organizers &amp; other competitors, and congratulations to IIdefons. &nbsp;Before discussing models, I thought I'd start a discussion about the data itself &amp; how it generally impacted peoples' modeling approaches.
 &nbsp;So here are some observations of my own, in no particular order: &nbsp;</p>
<p><span style="text-decoration:underline">Observations</span></p>
<p>1. Bids/asks from T=1...T=47 seemed to provide little predictive value. &nbsp;My variable-selection algorithms dropped them. In the forums, I noticed others mentioned that they also saw little value in using these prices.&nbsp;</p>
<p>2. The error contribution right at the market open (at 8AM) was extremely large. For one model, I found 12% of squared error for the entire trading DAY occured in the first MINUTE of trading. I trained a seperate model for the open (the naive benchmark worked
 better than a regression at the open, for example) and got about a &nbsp;0.0050 improvement, best case. &nbsp;</p>
<p>3. I didn't see price &quot;resiliency&quot; that the organizers discussed. Some of the examples the organizers posted showed stock prices bouncing back to pre-liquidity-event levels; we did not see this on average. Looking at the trade data in aggregate (via time
 averages, and various PCAs), we saw that for buys, the ask price jumped up immediately due to the liquidity event, and the bid price jumped up one time period later, and then both the bids &amp; asks rose very slowly. The opposite happened for sells.</p>
<p>4. For some of our models, we found that training a separate model for each stock _underperformed_ training a general model for all stocks. So a per-stock model was not necessarily a big winner, as we first suspected.</p>
<p>5. Prediction accuracy varied across time. Using a holdout set &amp; one of our models, I found that the error rose as you got farther from the liquidity-event trade. The RMSE was about 0.4 at T=52, rising to over 1.6 at T=100. RMSE rose roughly with sqrt(t),
 which, to me, implied some random-walk behavior away from the known prices at T=51.</p>
<p>6. The &quot;liquidity event&quot; trades did not seem to impact prices very much. Roughly 99.7% of the time, the VWAP was exactly equal to the best bid or ask at T=50. If there was a huge trade that ate through multiple levels of bid or ask prices, I would expect
 the VWAP to be different than the inside bid/ask immediately after the trade. It might have been somewhat more interesting if the trading data had some more large, market-moving trades.</p>
<p><span style="text-decoration:underline">Suggesetions for Improving the Contest</span></p>
<p>There were a few things that I thought could be changed to improve this contest; others have mentioned these, but I'll reiterate them:</p>
<p>7. The sampling methods used to create the testing &amp; training were different, and from my perspective, it would have been easier if they were sampled same way. &nbsp;The proportions of each security in testing vs training differed, of course. &nbsp;Also, the testing
 set was in random order, so why not also randomize the training set? &nbsp;One could correct for these testing vs training set differences by using different, per-stock weights for each row of data, or creating per-stock models. But this seemed like extra work
 that could have been avoided with uniform sampling. &nbsp;In the end, it took time away from focusing on the main goal of predicting the price behavior of the stocks.</p>
<p>8. The average prices for stocks in the dataset varied by a couple order of magnitudes, and when this was combined with the RMSE metric, this meant that high-price stocks (which contributed most to RMSE) dominated. For example, stock 75 -- with the highest
 price -- gave 36% of all squared error for one of our models. If the price data we were given was normalized (say, by dividing all prices by their VWAP), then perhaps the resulting models would be more generalizable across all stocks, &nbsp;regardless of price..</p>
<p>Everything considered, I thought this was an interesting contest in a &quot;hot&quot; area in finance. &nbsp;I look forward to reading about what others found &amp; did to create their models!</p>
