# My Approach - Ehsan ☃️

Competition: santa-2020
Rank: #27
Source: https://www.kaggle.com/c/santa-2020/discussion/216425

Hi everyone,

It was a very interesting run with lots of joys and fears down the road. I really enjoyed the ride, having a new idea every day, and coming to appreciate that how such a seemingly simple problem can go deep, having multiple aspects. I already see very interesting ideas are being shared in the forum. I hope all of you have enjoyed it as well. I'm going to describe my approach here and I'm really curious to know yours. Especially the top teams who did not use any machine learning tool, as we see some are really fast, i.e. only simple calculations are involved. 

# My Approach
main points:
- estimating the thresholds
- using opponent's pull information
- generating math/statistics features
- data scraping & training regression model
- exploitation vs exploration

The main core of my approach was to estimate the thresholds and select the bandit with the maximum threshold at each step. I tried both estimating the original thresholds and then correcting for decay, and estimating the current threshold directly. Using the first approach, if you have a slight error in the estimation of \\(thr_0\\) it will magnify when multiplied by decay. In other words, if you estimate the initial thresholds with 0.9 rank correlation, it does not necessarily lead to 0.9 correlation to the current thresholds. On the other hand, estimating the current threshold is somewhat harder. So how to estimate thresholds.

## Method 0
This is what gave me a boost in mid-December. It has two main ideas: incorporating the decay, opponent action consideration.

in the Bayesian ucb approach \([here is a good starter notebook](https://www.kaggle.com/isaienkov/santa-2020-starter)) you estimate the threshold by updating Beta distribution parameters by updating posteriors. then the boundary is \\(\mu + \sigma * c\\). I incorporated the decay by updated the winning posterior with a value more than 1, and the failing bandit by a value less than 1:

```
    win = (1/d) ** n[bandit] * reward
    fail = d ** n[bandit] * (1-reward)
    post_a[bandit] += win
    post_b[bandit] += fail
    total[bandit] += win+fail
    total[op_bandit] += 1
```
I also kept track of the total values, i.e. win + fail + opponent, and used this information in the boundary as:
[bound = \mu + \sigma * \frac{(total + 1)}{post_a + post_b}]

his way if my opponent is pulling a bandit more than me, it increases the multiplier to standard deviation, and the chance of pulling this bandit increases.

## Method 1
The previous method was not the best way to incorporate the decay. For that, we need careful mathematics and going to basics. I published a [notebook ](https://www.kaggle.com/safavieh/santa2020-maximum-likelihood-estimation-agent) taking the maximum likelihood into account. There I explained how using the data or observations as {0,1,X} series we can calculate the probability of occurring the data having a certain threshold, i.e. \\(P(data|thr/100=p)\\). But we know that the interesting probability distribution to us is the \\(P(thr/100=p|data) \\). So we need the Bayesian rule, and consequently Beta distribution to update the probability distribution after each update in the data. Generally as:
[P(thr/100=p|{a_1,a_2,...,a_k}) = \frac{P(a_k|thr/100=p) \times P(thr/100=p|a_1,a_2,...a_{k-1})}{c}]
which should be of the form:
[P(thr/100=p|{a_1,a_2,...,a_k}) = \frac{p^m (1-p)^n \times P(thr/100=p|a_1,a_2,...a_{k-1})}{c}]
if we assume \\(P(thr/100=p|a_1,a_2,...a_{k-1})=Beta(a,b)\\) then we can update the beta parameters like \\(Beta(a+m, b+n)\\)
Then the next step is to calculate m, n based on the current observation and current estimate of the. Here we can deviate and decide if we want to estimate the initial threshold or the current threshold. Let's continue by estimating the initial threshold:
as discussed in my [likelihood notebook](https://www.kaggle.com/safavieh/santa2020-maximum-likelihood-estimation-agent) if the initial threshold was (p) then the probability of having reward in (k^{th}) observation is (p\times d^{k-1}) and the probability of fail is (1-p\times d^{k-1}). So we should calculate (m) and (n) in a way that:
[p^m = p\times d^{k-1},] [(1-p)^n = 1-p\times d^{k-1}]
i.e.
[m = 1 + (k-1) . \log_p d] [n = \log_{1-p} (1-p.d^{k-1})]

but, (p) is actually unknown and its distribution has been calculated so far. So in order to calculate (m) and (n) we have two solutions: 1-put the expected value of p in the formula, 2-calculate the expected value of the formula given the beta distribution of the threshold. scipy has nicely provided the `beta.expect` functionality. 

Using this approach, and incorporating the total as method0, I could survive in top teams until the end of Dec.

But such methods have a fundamental problem because when calculating 
[P(thr/100=p|data) = \frac{P(data|thr/100=p) \times P(thr/100=p)}{P(data)}]
we consider (P(thr/100=p)) unchanged during the game, while in reality, assuming the sanity of the opponent, we observe good bandits more often. I tried incorporating such factors into statistics, but the best way to do it was using machine learning.

## Machine Learning
I developed many approaches like the above-mentioned method0 and method1, totally around 10 methods, and each one could reveal some features from the game, for example, the average, mode, and std of the probable distribution of the threshold. Taking all of such features plus simple features like number of pulls of bandit by the opponent and my agent, ratios, number of consecutive pulls, step, etc. I could train machine learning models and estimate the threshold. I had in the order of 100 features that I chose 25 out of them for training the models. 

For data generation, I used scraped episodes from my agent against other agents and generated a dataset with all of the above-mentioned features. Then I trained regressor models mainly Ridge, Lightgbm, XGBoost, CatBoost, and MLP models. Some of them were fast during prediction and some slow, so I had to play around to find the best combination of them while avoiding the timeout. 
For the ensembling of the models, I also tried many different approaches, like simple aggregation, weighted average, random/weighted-random selection. The best one for me was taking (average + std \times c) as the aggregated output of n-models.

## Exploration/ Exploitation Dilema
A little bit of Exploitation worked for me. I set a rule that at the beginning of the episodes I would exploit the bandits if my estimated probability of successful pull was greater than 0.6. Other than that, neither exploration nor exploitation worked for me. The best strategy for me was to select the bandit with the maximum estimated success probability.

## lessons learned
- there is always a better way/ method
- data is more important than the model. I learned this the hard way, I only realized in the last week that taking a good sample of episodes for training was more beneficial than focusing on the ML/ feature engineering part. As a result of this issue, I had two types of agents: ones that were good against good agents and performing badly in the early rounds and thus could hardly get to the top; and ones that were good in the beginning and could not survive beyond 1150.
- I had to work more on the strategy rather than threshold estimation.
