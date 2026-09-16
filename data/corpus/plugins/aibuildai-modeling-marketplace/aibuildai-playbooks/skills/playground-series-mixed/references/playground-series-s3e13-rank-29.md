# Top 29 Using Baseline

Competition: playground-series-s3e13
Rank: #29
Source: https://www.kaggle.com/c/playground-series-s3e13/discussion/406366

Hello folks!
I think the shaking of the leaderboard is predictable and can be observed by using adversarial validation. Although I knew there would be a shake, I didn't have a strategy to avoid it.

It was quite shocking that my CatBoost baseline model achieved a private score of 0.49890 with a public score of 0.35651. It literally just fit the data directly, without any FE and preprocessing.
Here's the [link](https://www.kaggle.com/code/eryaww/top-29-using-baseline/notebook) to my notebook about this baseline. 

My best submission is the submission I pick luckily, it was produced by [Ensembling](https://www.kaggle.com/competitions/playground-series-s3e13/discussion/406041) my top submission of diverge approach. Fortunately, I didn't include that base model on my ensemble 🙃.  It turns out pretty effective, best csv I put inside my ensemble is scored private 0.481, turn into private 0.502.

This diverge approach include
1. [Medical approach](https://www.kaggle.com/competitions/playground-series-s3e13/discussion/402933) by [Tanoi](https://www.kaggle.com/antoinerogeau)
2. [Cluster Engineering](https://www.kaggle.com/code/belati/vector-borne-disease-how-to-engineer-the-data) by [belati](https://www.kaggle.com/belati)
3. [Dimensionality Reduction](https://www.kaggle.com/competitions/playground-series-s3e13/discussion/402752) by [tetsu213](https://www.kaggle.com/tetsutani)

What a great experience! I'm looking forward to the next competition.
