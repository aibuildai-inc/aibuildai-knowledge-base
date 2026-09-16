# 5th place solution

Competition: LANL-Earthquake-Prediction
Rank: #5
Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94484#latest-544389

Hello kaggers!

There have been many good solutions posted already, but I haven't yet seen some of the things I did, so I'd like to share some of my insights in this competition.

 - First thing I realized is that no public kernel or my own features can distinguish between high and low quake intervals. Models usually settle for something that can be described as the mean quake interval multiplied by a factor between 0 and 1 with 1 being far from the next quake and 0 being right before the quake.

 - Knowing this, I decided to eliminate the influence of the time between quakes on the target and instead predict that value between 0 and 1. I called it time fraction and it's `time_to_failure/time_between_quakes`. Then I just multiply the predicted values by the mean time between quakes (actually, a slightly lower value worked better, so I treated it as a hyper-parameter). This allowed my models to run longer before early stopping, converging to a better solution.

**Feature engineering **

- I used ~20 features in the end. Here is the picture of features' permutation importances:
[permutation importances]

- Most of them involve some sort of signal filtering followed by some feature extraction. To filter the signal, I used Butterworth filters (`filtered_f1-f2_feature_name`), wavelet decompositions of various levels (`wavelet_name_decomposition_level_feature_name`), or removing large peaks. My most common features were histogram-based. Basically, it's the summed values at the peak or the tail of the histogram.  Some other features are peak-based or from public kernels.

- I did extensive feature selection, which improved my result significantly. I basically tried all possible wavelets in the pywt library with histogram-based features and did a stage-wise forward feature selection on them. To speed up feature selection, I used a simple SVR with a gaussian kernel instead of my final model.

 **Model** 
I used a simple feed-forward neural net in pytorch, I found that it works much better with my features than LGBM. I added some bells and whistles to it, like adjusting learning rate on plateaus, batch normalization, adding 0 mean, 0.05 std gaussian noise to the training batches, training several models on each fold and picking the best one to avoid unlucky random weights initialization.

 **Cross-validation** 
I loaded training data on a per-quake basis, so there was no overlap between quakes. I didn't test anything else, it just made sense to me to do it that way. I did 15-fold quake-based CV uniting the first and the last bits to the neighboring quake. So, my folds were [0,1],  2,  3,  4,  5,  6,  7,  8, 9 10, 11, 12, 13, 14, [15, 16]. Also, I extracted features from 150,000-point chunks with 25,000 shifts, which resulted in a significant overlap between adjacent chunks, but it worked fine in my NN model.

 **Scaling** 
I identified the best scaling factor for the train set (by treating it as a hyper-parameter), then adjusted it based on the info about the test data (upscaled it a bit). I used 2 coefficients: ~1.05 and ~1.13. The first one gave me a better public LB and it correlated well with my CV, so I used it to check my model. The way I came up with 1.05 is pretty random and makes little sense, but this is the multiplier that brings the mean of my CV predictions to the target mean ttf in the train set. The second coefficient is based on what I expected the mean test set ttf to be.

Overall, it was a fun but stressful ride. I think predicting time fraction instead of ttf could be useful to other winning models here and it could be a better target to predict in real life, as it indicates the relative state of the system (kind of like a danger level).

Finally, here is my CV predictions plotted:
[CV predcitions]

UPDATE: Just adjusted my best submission's mean from ~5.7 to 6.32 and my score improved to 2.24127 :O Oh well, this is the lottery we all played.
