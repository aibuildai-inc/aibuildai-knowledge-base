# 2nd Place Solution

Competition: mercari-price-suggestion-challenge
Rank: #2
Source: https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50499

So here's our second place solution (the kernel we are publishing is almost similar to the winning kernel)
The kernel scored 0.38888 on public and 0.38948 on private and run in 2800s

[https://www.kaggle.com/mchahhou/second-place-solution][1]

As a preprocessing step, we kept maximum 60 words for all names and descriptions and apply the same cleaning process on all the texts
We  labelEncoded all brand and categories and concatenated the encoded values with the name feature
   
We made 4 models : 
1) Ridge model : trained on 1-ngrams and custom bigrams.
      1-ngrams is using both name and description. Our custom bigrams work as follows :  we concatenate name with the first 5 words from description,  then apply np.unique() on the list of word to sort them and remove duplicates,  then create all 2-way possible combination of words. 
The ridge model is able to score .418 on public LB
   
2) Sparse NN model trained on a CountVectorizer with ngram_range=(1,2). The NN is fit with sparse data from name and description features

3) a fastText NN model with a shared_embedding layer for name and description features

4) another Sparse NN model fit with character Ngrams using name and description features.

We waste half our time building an LGB model that didnt help even though it was our best model. We found that using different models with the same data representation is not going to work. We then choose to use different NN with different inputs

To make our NN fast, we double the batch size after each epoch. This is quite similar to reducing the learning rate after each epoch with the advantage of speed gain

  [1]: https://www.kaggle.com/mchahhou/second-place-solution
