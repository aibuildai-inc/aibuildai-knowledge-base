# 7th place solution

Competition: playground-series-s5e5
Rank: #7
Source: https://www.kaggle.com/c/playground-series-s5e5/discussion/582591

First off, congratulations to all who survived the massive shake-up. I can't say that it took me by surprise. I wrote a post about a potential upcoming shake-up [here](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/578016) and shared my thoughts on the matter. The dataset this month was very large, and while I didn't expect a shake-up at first, that quickly changed after I made my first few submissions.

# Data Preprocessing

I trained models both with and without the original dataset, but most were trained without it. As I mentioned [here](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/576094), the original dataset didn't bring much improvement. Models trained with additional data were primarily part of early experiments to evaluate its usefulness.

Aside from that, I didn't do much preprocessing. I trained my models without any feature engineering or preprocessing, except for the `Sex` column, which I converted from categorical to integer values.

# Modeling

I used both standard gradient boosted models and AutoGluon. The latter wasn't very competitive this month, so I only used four models from early experiments. Most of my models were CatBoost, as it proved to be the strongest single model.

# Ensembling

I experimented with Ridge, Lasso, AutoGluon, and hill climbing for ensembling. Hill climbing showed the best CV score, but Ridge turned out to be the winner in the end. AutoGluon didn't perform well in terms of CV or LB scores. Interestingly, I had a submission where I used AutoGluon as an ensembler, and it could have secured 3rd place. However, I didn't choose it because it had neither strong CV nor a strong LB score.

# Final Words

If you take anything away from this discussion, let it be this: **trust your CV and avoid mindless blending**. Blender notebooks have become a significant problem in playground competitions. They heavily overfit to the LB and may mislead beginners into thinking they're effective strategies for winning. It's becoming increasingly difficult to find quality public notebooks, as many of the top ones are blends with manually tuned weights aimed at overfitting to the public LB.

I hope this trend changes, and that the community begins to focus more on learning and adhering to data science best practices instead of mindlessly blending others' work.

**I wish you all the best.**
