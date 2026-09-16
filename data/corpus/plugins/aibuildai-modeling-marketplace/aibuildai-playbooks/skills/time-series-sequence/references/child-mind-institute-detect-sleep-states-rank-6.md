# 6th Place Solution - BiLSTM-UNet

Competition: child-mind-institute-detect-sleep-states
Rank: #6
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459604

Thanks a lot to the competition host and kaggle for hosting this fun competition!

# Data Processing

I split the data into different separate days with a padding of 60 minutes. The days are shifted by 500 minutes (so that one day starts and ends at 15:40). I did this to avoid wakeup/onset events near the end or start of a day, which I hypothesized to be benefical to the model (since there is only limited padding in both directions). I discarded 8 series ids which were mostly filled with nan values, and hand picked a first-n-day cutoff for a few series (removing the end of the series whenever there were no events but still regular activity). Since events only ever happened on full minutes, I constructed features for every one-minute bucket of data and generated an array of size (1440, n_feats). The features were mostly basic: anglez, enmo, anglez*enmo, hdcza were selectively paired with min, max, mean, std aggregations to generate 10 features, where hdcza was inspired by this discussion post:
https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/453267

Since it was discovered by @ymatioun made the great discovery that events had different frequencies at different steps in the 15 minute cycle. Therefore, one additional feature is the (step // 12) % 15.
Since there were large repeated periods in the data, I used an algorithm that utilized hashing and hashmaps which looked for such patterns in the data:


Such a feature proved incredibly useful since there were many, non-obvious such repeating patterns in the data.
Here is a plot of enmo (blue), awake (red), and this new features, which indicates whether the data occurs somewhere else in the series or not (black):


As you can see, there were many parts of the data which were copy pasted from somewhere else. I am sure some models will have learned this anyway from the sometimes apparent cuts in the signal, but this of course works much better. The important thing about this feature is, that the patient is almost always sleeping when it is 1 (>99%). I didn't do too much in depth testing on how this affected the score, but local cv indicated that it improved the score by roughly ~0.015. Since the data was repeated and likely not valid anyway, I null out all anglez and enmo values where this features is 1 to prevent overfitting.

# Model Architecture
I used 4 model architectures which were all very similar and mostly based on Bidirectional LSTM layers. I used a 6 fold split. 

My main model used a UNet LSTM/Transformer architecture with a fully connected dimension of 1200 and an LSTM/Transformer dim of 300. In contrast to the typical UNet implementation, I added the layers in the skip connection rather than concatenating.

## Main model: 

This is the model pipeline used by the best single models. The other 3 architectures I used are very similar (with varying layer counts and dimensions), except 1 architecture which doesn't use the UNet.





1 day of data in 1 minute intervals (60 minute padded in both directions) shape: (batch_size, 1440 + 60*2, n_feats) -> fully connected layers -> BiLSTM -> max pooling (pool size 15) -> 2 lstms layers -> max pooling (pool size 2) -> transformer block -> upsampling (factor 2) -> skip -> 2 lstm layers -> upsampling (factor 15) -> skip -> lstm layer -> fully connected layers -> output layer

The 'skip' layers indicated the the output from the same stage at the downsampling stage is added onto the current sequence in the upsampling stage.

The minute-mod-15 feature was used to calculate an embedding which was added to the sequence before, and after the UNet. Also, I used the mish activation most of the time.

The other models had slight variations, e.g. leaving out the UNet, changing dimensions, increasing layer count.
Some variations also include a sub-lstm layer, which applies an LSTM layer on all 180 minutes intervals within a day, with 120 minute overlaps between the intervals.

## Loss function and output
The model predicted if an onset or wakeup event occurred within 0, 1, 3, 5, 7 minutes of the current step. I use a weighted BCE loss which weights the loss of the different minute thresholds by 1/(threshold+1). The prediction is the sum of all the probabilities that an event is in the 0, 1, 3, 5, 7 threshold. This is done for wakeup and onset separately so the model had 10 outputs per step.

# Postprocessing
The model output itself had dimension (minutes_in_series, 10). This is the process of retrieving predictions from the model output:
1.	Concatenate the model predictions from each day in a series.
2.	Take the sum of the prediction that an event is in 0, 1, 3, 5, 7 minutes range at each step, for onset and wakeup respectively.
3.	Until there are no predictions larger than 5e-5 left:
a.	Select the highest prediction.
b.	Null out all predictions within a range of 10 minutes of the prediction.
4.	Since we are still in minute space, multiply by 12 to obtain the correct steps.
5.	As was pointed out in the discussion forums, predictions for full minutes are punished since the metric doesn’t count a prediction as within a threshold, if it is directly on a threshold. Therefore, to maximize the number of possible thresholds we may be in, it is beneficial to add or subtract 1. This is decided by looking at the neighboring prediction.
6.	Use the summed predictions from step 2. As scores for the retrieved steps.

Thanks to @maruichi01 for pointing that out! The discussion related to this is here:
https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/446919#2487136

# Training
I used the AdamW optimizer with a weight decay of 1e-7 and lr of 1-e3 that exponentially decayed. It helped training stability to clip the norm of the gradient to 1e-2. Regular dropout was used and a kind of dropout that randomly masked a 15 minute segments with a probability of 15-25%. I trained on 4 seeds and picked the best seed of each architecture within each fold. I also refitted the best 2 architectures using the entire data.

# Other stuff
Like Chris mentioned in his write-up, the discovery that we can make a lot of predictions per event was also very important. I first had a decoder model, which used the previous predictions the generate the next one using a probability distribution over all future steps. This worked quite well but only if the maximum prediction would have been counted. So I discarded this idea after discovering that we can make a lot of predictions per event without getting punished too much.
My local cv is ~0.833.

# Things that didn't work
Like I said earlier, I first worked on a decoder model to predict the next change event, which didn't work so well since it was beneficial to make many predictions per night. I also worked on stacking the model with an LGBM, but that only gave tiny improvements so I avoided the idea to not add too much complexity, though if I had more time I would have probably reconsidered it.
Since the ranking of the predictions was important I tried using other loss like e.g. a pearson correlation loss but that did not work better.

I hope I covered the most important parts and didn't forget anything. I tried a lot of things I may add later. I am very happy I managed to get my second solo Gold today, thus becoming a Kaggle master 😃

Edit:
Link to git repo:
https://gitlab.com/fritz_cremer/cmidss_final
