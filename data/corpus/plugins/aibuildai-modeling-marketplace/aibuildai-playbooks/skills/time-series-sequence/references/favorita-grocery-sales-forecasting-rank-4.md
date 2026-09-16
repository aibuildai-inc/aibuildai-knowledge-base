# 4th-Place Solution Overview

Competition: favorita-grocery-sales-forecasting
Rank: #4
Source: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47529

First off, congrats to the winners and thanks to all those who contributed to the forum discussions.  I'm definitely disappointed with the result, but I might as well share my solution anyways.  I'll only discuss my contribution to my team's solution, which is a single model which scores .499 on the public leaderboard.  

### Model Architecture
We framed forecasting as a sequence to sequence modeling problem, where the encoder reads in a portion of a time series and the decoder sequentially emits estimates of the subsequent 16 values. The encoder and decoder do not share parameters, but they are both similarly parameterized as a stack of ~30 dilated, causal convolutions.  Dilated convolutions allowed the receptive field to span the entirety of the 4+ year train period, while still maintaining a small number of parameters (~1 million).  To allow the decoder to use future onpromotion values, it was also augmented with a bidirectional LSTM which encoded information from prior and future onpromotion data.  The model struggled to accurately predict zero, so we modified the architecture to additionally emit a bernoulli parameter at each timestep (corresponding to the probability that the output was zero) and the final predictions were adjusted accordingly.  Weighted RMSE of the log-transformed data was directly minimized.

### Input Representation
The input to the network consisted of the raw time series values, embeddings of the categorical variables, and some manually engineered features.  The manual features included lags, diffs, rolling statistics, date features, and conditioning time series (i.e. average sales for a given product/store/etc.).  Promotion data required some special handling, which I'll describe later.

### Validation
5% of the time series were held out, and the model was evaluated on random periods from the last 365 training days of these held out series.  The purpose of this was to prevent the model from overfitting a validation set which was biased toward any sort of weekly or monthly trends.

### Promotion Data
As discussed in the forums, if you naively impute missing onpromotion values with 0, the sales distribution conditioned on onpromotion values will differ in the train and test sets.  I didn't find a way to remedy this, but thankfully my teammates were able to come up with a clever solution which improved my score by about .004.  The missing values were imputed randomly: 1 with probability p and 0 otherwise.  Determining p was a rather tedious task, since we had no data to model it and we had to resort to trial and error leaderboard submissions.  We ended up setting p separately for each day, and it was computed as the mean onpromotion rate for each day, scaled by a learnable factor estimated by stochastic leaderboard descent (leaderboard probing...).  All credit goes to my teammates for this, and they can probably explain it better than I can.

### Source Code
I normally share my source code, but since the core model is nearly identical to one I've already shared, I'll just refer you to that instead: https://github.com/sjvasquez/web-traffic-forecasting

Finally, I'll admit I'm mostly disappointed with Kaggle.  I raised concerns about the unseen items in the test set and also about the onpromotion issue earlier in the competition.  Ultimately, the competition was a lottery.  For the sake of others, I hope Kaggle will begin to demonstrate a moderate level of care in data preparation and show that they understand that whenever they deviate from the assumption that train set is an unbiased sampling of the test distribution, they induce randomness which turns competitions into lotteries.   This will be my last Kaggle competition, as my time will be better spent focusing on research from now on.  Best of luck to everyone.
