# #2 solution

Competition: tabular-playground-series-nov-2021
Rank: #2
Source: https://www.kaggle.com/c/tabular-playground-series-nov-2021/discussion/291903

Thanks to @chaudharypriyanshu and @dlaststark for their notebooks on neural networks, and to @motloch for his work on flipped targets, and to @grayjay for pointing out the data is chunked.

All I did was notice the proportion of flipped targets was slightly different in each of the chunks in the training set. Across the entire dataset it was about 25%, but each chunk ranged from about 24.8 to 25.2% or something like that. It's a small variation, but I thought it might be enough to gain a small increase in accuracy. 

The idea was that if a chunk in the test set had a slightly higher rate of flipped targets then this chunk would contribute more to the overall error, and I should push the predictions of that chunk towards 0.5. 

I ran a neural network and submitted the raw results as a baseline, and then probed the 9 chunks in the test set. I made 9 copies of the baseline, and for each copy I rescaled a different chunk by averaging the target with 0.5. Chunk 6 turned out to benefit the most from being rescaled, while chunks 1 and 8 got worse. For the final submissions I rescaled each chunk of my NN using the equation (c*target + 0.5)/(1+c) with different values of c for each chunk.

One thing I learned from this competition is to look closer at the datasets and not take anything for granted. It seems like @motloch and @grayjay saw the simple patterns that the neural networks missed.
