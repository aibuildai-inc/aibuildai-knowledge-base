# 538 abuse — solo cash gold solution

Competition: march-machine-learning-mania-2024
Rank: #7
Source: https://www.kaggle.com/c/march-machine-learning-mania-2024/discussion/493841

First of all, huge thanks to Kaggle and hosts for giving us the opportunity to compete.

This year, for the first time, we didn't have the ratings from 538. Why not just solely rely on the previous years data?

**Main idea**

My main idea was as follows — to map the current strength of the teams (publicly available rankings from various sports agencies) onto the historical ratings of 538 and use only them to form probabilities through the ELO formula.



The current rankings reflect the real power of participating teams, and using multiple of them adds diversity. The 538 ratings are just a solid set of numbers to map onto, when further applying the ELO formula.

**Submission strategy**

To make 2 diverse submissions I used strategy that was previously discussed in the MMLM competitions. Predictions were replaced as follows \\((p,1-p)\rightarrow (0.64, 0.36) \backslash (0.36, 0.64)\\), when \\(p>0.77\\) for men and \\(p>0.7\\) for women in one submission and vice verse in the other. It was used later to make it possible to achieve high LB placement under log-loss, so I'm not sure it is the best possible option under the new metric.

I have simulated 100k brackets for both women and men, manually removing brackets where low-seeded teams won the championships,  resulting in approximately 99k brackets for women and 95k brackets for men. Notebook is avalable [here](https://www.kaggle.com/code/samson8/fork-of-prefinal-mm-9875d3)

**Data usage**

I have used [this dataset](https://www.kaggle.com/datasets/raddar/ncaa-men-538-team-ratings) (+ [this ](https://www.kaggle.com/datasets/raddar/ncaa-women-538-team-ratings) for W), [ESPN rankings](https://www.espn.com/mens-college-basketball/rankings), [KenPom](https://kenpom.com/index.php), [Massey](https://masseyratings.com/cb/ncaa-d1/ratings), [SM](https://sonnymoorepowerratings.com/m-basket.htm) (both W and M), [JSokol](https://www2.isye.gatech.edu/~jsokol/lrmc/about/) (both W and M).

**Honorable mentions**

I would like to thank @lennarthaupts for creating [simulation code](https://www.kaggle.com/code/lennarthaupts/simulate-n-brackets) — it was used by me for tournament simulations. Also kudos to @roberthatch for creating [this notebook](https://www.kaggle.com/code/roberthatch/mm24-follow-live), which helped us tracking our score online.
