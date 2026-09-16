# 24th place solution: Couple more tricks, and how I kinda sorta cheated

Competition: jane-street-market-prediction
Rank: #23
Source: https://www.kaggle.com/c/jane-street-market-prediction/discussion/224079

Inspired by @dmitryvyudin's recent post, I thought I'd also comment on a couple of tricks I used that I haven't seen anyone else mention. I think the first couple are good ideas, while the last is getting into a bit of a cheating grey area, and is certainly not deployable. Curious to hear if anyone else tried any of these!

**1. Dynamic Label Smoothing based on resp**
From very early on, I was frustrated that regression approaches were not working, and felt that classification was throwing away valuable information contained in the Resp targets. When I came across label smoothing (1 -> 0.99, 0 --> 0.01), I had the idea to use resp to control how much smoothing to do. So my classification labels were actually sigmoid(a*resp), where a is a large, tunable parameter. This way, if resp was very close to zero, the 0/1 label would actually be blunted to something a bit closer to 0.5. This was repeated for all 5 resp targets, fed into multi-target NN, and I simply averaged the final predictions. This had a measurable benefit to my CV scores.

**2. Log the weights in my CV.**
The most challenging part of this contest IMO was measuring which models were actually doing well. I would constantly get conflicting advice on different runs, even when ensembling many random seeds together or repeating experiments. In addition to focusing on metrics like AUC, something else that helped was log-transforming my weights for my internal CV utility function. This seemed to help mitigate the effects of large-weighted trades without ignoring them entirely.

(and as many others have mentioned, I almost entirely ignored the public LB. Both my submissions were in the bottom half in public LB rank.)

**3. Use weights to triage to faster/slower models in deployment.**
I thought this was pretty clever, but it's unclear how much it actually helped me. If you look at the weight distribution and the utility score, it turns out that a few observations make up the vast amount of the total weight. Because of this, you can actually deploy much better/slower models on just a few obs, and get massive benefit. For example, obs with a weight of 30 or more only make up 3 % of the observations, but a third of the totalweight. I threw an ensemble of 20 MLPs, 5 Densenets, and 5 resnets at these observations. The lowest weighted obs barely mattered, so they got 1 MLP, and ran in 1/30th the time. In the end, my deployment run time was FAR below the max allowed, whereas probably would have timed out if I had thrown the full 30-model ensemble at every observation. 

**4. Head I win, tails you lose.**
From early on, I had decide that I was going to use my two submission for opposing thresholds. 

(An aside: I was never convinced that the utility score actually penalized risk-taking like the hosts intended. The sharpe ratio balances risk and return, but because it is multiplied by total return, the return term is actually in the utility equation twice, and thus outweighs the volatility penalty in the denominator in practice. I found no evidence that a "conservative" approach in my submissions ever helped. But maybe I'm missing something...)

All that is to say: I think 0.5 is the best/correct threshold if you have to pick one. However, because we have two submissions, I submitted one that was "bearish" and one that was "bullish". So basically one submission with a threshold of 0.49, and one with a threshold of 0.51.

But I took this one step further: If feat_0 is in fact long/short or buy/sell, then shorts should be more aggressive in a bad market, and longs should be more aggressive in a good one. So in the end, my "bearish" submission uses a 0.49 threshold for feat_0 == 0 obs, and a 0.51 thresh for feat_0 == 1 obs, and my other submission does the opposite. If the market booms relative to the training data, one of my submissions will do well. If it crashes, the other will do well. I could have gone more aggressive with something like 0.47 and 0.53, but I also wanted to do reasonably well if the market did exactly as well as it did during the training data. 

If Jane Street can find a way to trade in two parallel universes, and only suffer the consequences of the one that works out better for them, I'd highly recommend this trading strategy.

Other than these tricks, everything I did was pretty similar to stuff found in published notebooks. I ensembled MLPs, densenets, and resnets, with the most weight on MLPs. 3-4 layers, dropout, swish, very few epochs. First 85 days INCLUDED, and weight == 0 observations included. I'm not sure why so many folks took this data out.

Curious to hear feedback, and looking forward to the next 6 months of suspense!
