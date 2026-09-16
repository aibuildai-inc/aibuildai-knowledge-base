# 5th Place Solution

Competition: march-machine-learning-mania-2025
Rank: #5
Source: https://www.kaggle.com/c/march-machine-learning-mania-2025/discussion/572909

**Private Leaderboard Score**：5th

**Private Leaderboard Score：** 0.10748

[notebook](https://www.kaggle.com/code/aziel888/updated-goto-conversion-winning-solution/)

I will start by congratulating all participants and thanking Kaggle and the organizing committee for this really nice contest year after year!



#### Background

This was my first competition on kaggle and I had predicted results in other sports' competitions before.



## Methodology

By analyzing the high-scoring solutions of the past few years, there were some solutions that were seen simple but work well. For example,  a lot of people used Nate Silver's data for reference, and do some  [manual fine-tuning](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/discussion/400116) in 2024. So this time I also chose the similar approach.

My work is mainly according to the analysis form the [visual page](https://wncviz.com/NCAABrackets/KaggleBrackets.html), then manually adjust some probabilities based on the output file from @kaito510's notebook. I don't think the past data can be linearly extrapolated to the present, they are not very correlated. But historical data can be used as a baseline. At the same time I think the price ratio is a good reflection of the probability, which is what the stock market often says if the information is completely transparent, then the price contains the information.

Here are some of my fine-tuning methods:

* Refer to the results from some popular voting websites that predicted the winners of the men's and women's groups. And it was also mentioned in the comment sction that the probability of the men's group was more random, while the women's group was more stable.Therefore, for the men's group, I did not choose the championship favorite Duke.
* Fine-tune the results based on the odds
* Fine-tuned based on the ELO rating system and Nate Silver's predictions
	Since most of this year's tournaments weren't particularly "crazy," this approach worked out well

Till next year!
