# 1st place solution

Competition: mlb-player-digital-engagement-forecasting
Rank: #1
Source: https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/274255

First of all, many thanks to MLB and Kaggle for hosting this great competition! It was a very funny and interesting competition with a really surprising ending for me. I completed the first solution that I was happy with almost at the latest possible moment and thus my first submission was on the day before the deadline. In fact, I seriously considered not submitting at all...🙈 
I provided the link to the final notebook used for my best submission at the end of this post. Unfortunately it is certainly not as clean and nicely documented as a typical public notebook. But I hope the following documentation will be sufficient to get the basic ideas used for the solution.

# Overview 
Frist a very short overview for those who don't have time or are not interested in the details.
I used a neural network consisting of a GRU and several processing layers which was trained from scratch (the first 5 hours of the 6 hours inference time) in the Kaggle kernel (with GPU runtime). The main idea was to build a model that sequentially processes the provided “raw“ input information related to a certain time span to gather the relevant information for the final prediction (so that no noteworthy feature engineering is needed). 
Since I didn't have any other models for a real ensemble, I used at least a snapshot ensemble for the submitted prediction.

# Modeling approach
Heuristically one would expect that the past targets of a player coupled with the player performance features (related to the “playerBoxScores“ - information)  would be the most crucial information to predict future values. However, while the "performance features" are available for the complete inference time span, the targets are only available until the last day of the training data. One of the basic assumptions was that a suitable and consistent simulation/treatment of this time-gap during training would be crucial for a well performing  model.
The model uses features related to a 96 day interval previous to the target day as input. 
First of all the past targets (scaled with log(1+x) to reduce their partly extremely "peaked nature") were provided as input. To simulate the time gap without target information the target values for the last n days (where n is chosen randomly and n<=32) are simply set to 0. In addition a target mask was added to mark if the target information was provided on the respective day or not:
[[MLB.png]](https://postimg.cc/SncW68s4)
Moreover, (like in almost all approaches) the daily performance features were added (e.g.  “playerBoxScores“  like “groundOuts“, “runsScored“ and “homeRuns“) as well as some other features like the “positionCode“, “teamId“, “statusCode“, "numberOfFollowers" and the "month". Non categorical features were scaled to roughly the same order and missing values were replaced by -1 (otherwise the features take non-negative values).
[[MLB-model-2.png]](https://postimg.cc/QKBcW0Sq)
First the input features are fed to an Embedding Block consisting of several Dense layers with leaky relu activation and processed by 2 (linear) Residual Blocks so that for each of the input days reasonable embedding vectors can be created (So up to this point there was no exchange of information between different days).
Afterwards, the day-wise feature information is processed in positive time direction by the GRU layer with hidden size 128 (motivated by the picture of an analyst who observes the performance of a player and the fan engagement over a certain time span to gather the relevant information for the prediction of future values). 
To simplify the conservation of the day-wise input information, the output of the GRU layer is concatenated with its input. Finally, the output of the previous steps is processed by a sequence of 3 Dense layers and returned with relu activation.
For regularization Dropout layers were added.

# Model training
The model was trained from scratch in the Kaggle kernel (GPU) for a period of 5 hours so that the remaining hour could be used for the inference process (no data apart from the competition data was added). The epoch length was defined such that the training time corresponds to roughly 30 epochs and for validation May 2021 was used.
Adam optimizer was used for the model training and the learning rate was reduced with exponential decay after each epoch where the base was chosen such that the learning rate starts at 2e-3 and ends at 5e-4. The batch size was set to 64 and for regularization weight decay of 6e-5 was added.
For each training sample a random player is chosen as well as a random time gap for the time span between the day for which the targets shall be predicted and the last day where the information of past targets is added to the model input. Corresponding to the announced length of the evaluation period the maximal value for this time gap was set to roughly one month (32 days). For a more efficient usage of the data also the targets for the period previous to the actually considered day belonging to this time gap were predicted. Probably it would have been smarter to predict all days of the maximal time span (i.e. 32 days) in each iteration. Otherwise model bias to shorter time gaps (where the correlation to the past target information is probably larger) might be introduced because for the applied approach targets with a time gap of a few days are predicted almost in each iteration while those related to a time gap of 32 days are only predicted when the random interval takes its largest possible value. 
Moreover, the assumption that the model is able to reliably capture the time span between the last day with target information and the day for which the targets shall be predicted, is possibly too optimistic. Perhaps the addition of positional encodings (providing information about the length of the respective time gap) would be helpful in this regard and could lead to a further improvement of the model.

# Inference
The inference is done in the final hour of the kernel runtime. Since only one model architecture and type was trained, at least a snapshot ensemble was used to introduce a bit of diversity. To this end, after each training epoch the model weights and the related validation loss were saved. The 6 best “snapshot-models“ (with the lowest validation loss) were used for predicting the targets during the test period. These predictions were simply averaged, clipped to the relevant range (0-100) and submitted as final prediction.

# Final comments 
Many other architectures with alternatives to the GRU layer were used like LSTM, CNNs or a transformer. A direct comparison is hard because obviously the optimal parameters (layer units, number of layers, learning rate,…) are typically different for different architectures but the GRU layer outperformed the alternatives at least under the chosen circumstances. 
Also the addition of different normalization methods (batch norm, layer norm,…) was tested, but without success.
Most time was clearly invested for the optimization of the model and only few time was used for feature selection. So there is probably still potential for improvement by dropping or adding certain features.
Finally, many thanks to the great input from many participants in this competition. 🙏E.g. this [notebook](https://www.kaggle.com/ryanholbrook/getting-started-with-mlb-player-digital-engagement) was a great help to get started with this competition and some stuff in my final notebook is certainly still based on it. 

My final notebook: [https://www.kaggle.com/ph0921/mlb-final-1](https://www.kaggle.com/ph0921/mlb-final-1)
