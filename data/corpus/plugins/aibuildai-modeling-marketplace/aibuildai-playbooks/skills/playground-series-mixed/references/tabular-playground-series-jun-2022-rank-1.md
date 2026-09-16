# #1 Solution - Denoising Autoencoder

Competition: tabular-playground-series-jun-2022
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-jun-2022/discussion/334331

As most people noticed during this competition, the significant challenge was estimating the conditional distribution of F4 where two or more values were missing in the given row. To solve this, I used a denoising autoencoder, which estimates the distribution of missing values given a mask of where the values are missing. I mean-imputed F1 and F3, and completely ignored F2.

A pytorch implementation of my notebook can be found [here ](https://www.kaggle.com/code/sebastianvangerwen/1st-place-solution-tps-jun-denoising-ae) (it scored poorly because I messed up the submission dataframe 😅).

Below, I have included a drawing of this architecture.



**Random Mask**
I initially impute the data with zeros, and create a source null matrix that contains the locations of the original data nulls. Then I create a binomial random mask (where each row has at least one value) that I multiply by the original data to randomly set values to zero. The model takes the masked data and an input mask - the input mask is the combination of the source null and random mask vectors.

**Feature-wise embeddings**
The model embeds both the features and the mask and adds them together. This is so it can learn a representation of where the nulls are set to zero, and therefore where to focus to impute. This method performs much better than simply applying dropout to the masked data. For these embeddings, I linearly projected each feature in the input and mask vectors into an embedding dimension (16 in the final submission), added them together, and then flattened them.
I also tried dot-product attention between the mask and linear embeddings (as per [this paper](https://arxiv.org/abs/2106.16057)), but this performed worse than simply adding them together.

**MLP Architecture**
The feature/mask embeddings are then sent through an MLP. I used layer normalization with skip connections, with 7 dense layers using the mish activation function. The performance increased with larger layer sizes, but I only tried up to a size of 2048, since computation time became a problem and I was having to decrease my batch size dramatically.

**Output Computation**
I send the MLP output through a final dense layer to match the dimensions with the input shape. Then, the final output is conditional on whether the value was masked. If the value was masked, we use the final dense layer output, otherwise we just use the model input. This causes the gradients for non-masked inputs to equal zero with the MSE loss, so the network only updates parameters for their contribution to the imputation of the mask. Like the mask embeddings, this helps the network focus on the values that should be imputed.

**Masked MSE Loss**
This step is very similar to the output  computation, and is just another way of conditionally setting the loss function to zero. Here, I used an MSE loss that sets the value to zero according to a mask. This mask is the true data null values - thus we never calculate a loss on values that were originally in the training data, but we can still learn from those rows without creating bias by using a naive imputation method during training. A similar method is used [here ](https://arxiv.org/abs/2002.08338)(equivalent to setting the loss to zero at the data nulls). This step could have been included as part of the modified output computation, but I only wanted to include this step during training - otherwise the implementations would have to pass a vector of zeros during the prediction step (which felt a bit clunky).

**Conditional Ensemble**
I ran the DAE 3 times in pytorch and tensorflow - tensorflow runs performed significantly better, but the average of all performed the best. This model scored 0.83351 on the private LB. To improve this slightly, I ensembled single-attribute prediction runs only when the row-wise null count for F4 was 1. This conditional ensemble gave the private LB score of 0.83343.
