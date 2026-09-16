# 16th Place Solution

Competition: child-mind-institute-detect-sleep-states
Rank: #16
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/460371

I want to thank the organizers for such a great competition that provided us with the opportunity to delve into the very intresting domain of detecting sleep events using time series data from an accelerometer. I hope the participants' solutions provided meaningful and valuable insights to Child Mind Institute that will speed up and enhance their research. 
This was my first competition where I participated seriously, and I have achieved 13th place with a gold medal on the Public data and 16th place with a silver medal on the Private data (unfortunately, the shuffling on the private  had other plans for my gold medal).

**Overview**

Data Processing/Feature engineering:
- Remove gaps without labels from the training data.
- Apply the log1p transformation to enmo and anglez/90.
- A Combination of spectrograms with different window lengths and rolling features.
- Gaussian distributions around events as the target labels.

Models:
- 5-fold GRU + CNN model for detecting events.
- 5-fold LSTM + CNN model for detecting events.
- 5-fold TCNN model for detecting events.
- 5-fold LSTM + CNN model for segmenting states (sleep or awake).

Post-processing:

- Find mini peaks (7 minutes distance).
- Find macro peaks (400 minutes distance) to capture multiple micro peaks (1 minute distance) around them.
- Apply a segmentation probability map using a custom equation to rescore detections.
- Detect and remove dummy segments.
- Maximum increase overlap between input windows to models

**Data processing**
 
I noticed that models perform better in detecting events outside of dummy segments when trained on filtered data. The filtration process involved identifying gaps without any labels, removing them from the time series, but retaining 300 minutes after the last wakeup before the gap and 300 minutes before the first available onset after the gap. 300 minutes retaining was done to preserve context for models around events. Then, filtered time series were split into chunks of 700 minutes with an overlap of 350 minutes and shuffled into Group 5 Folds for training and validation. Predictions on overlap segments were averaged. 

**Feature engineering**

Feature engineering was initially to take log1p of enmo, anglez/90 and then make spectrograms with window lengths 12 and 60 of them as well as rolling features such as absolute difference median, std, and median with window lengths of 30, 60, and 120.  And the sin of minutes was included as an additional feature. 

Labels for the event detector task were made by Gaussian with 20 minutes around events with sigma=0.75. And labels for segmentation were made by taking segments between pairs of onset and wakeup. 

**Event detection models and training**

I achieved the best CV using LSTM with an initial dense layer of 32 dim (increasing dimensionality gave worse results), after LSTM, CNN head was added. Then, I found out that incorporating a TCNN with 7 residual layers (kernel size 3, dilation 3^(layer-1)) into the ensemble boosted the LB score ~ 0.004.  

To add a little diversity into the ensemble,  I thought that because GRU has less gates and information flow differently from LSTM I included it into the ensemble using the first architecture replacing LSTM and got ~0.002 boost on LB.  

Trained models with CyclicLR with the length of cycle = the number of batches, max lr = 0.005 and min lr = 0.0001 gave the best results. I suppose that it may be related to the cyclic nature of time series data. 

**Segmentation models and training**

After visualizing predictions, I realized that event detection models sometimes overlook long-term dependencies. For instance, they may assign high probabilities to steps preceding a gap, even when a long real sleep state segment appears in a while.

With the wish to include more context I decided to train a segmentation model to segment  sleep/awake state and apply its probability map via custom equations which will be explained in the post-processing part. 

After some experiments with 1d unet, cnn, rnn I found out that LSTM with CNN head (the same architecture as for event detection model) showed the best iou. CyclycLR was used as well with the same parameters as for event detection models.  

Below are predictions of event detection and state segmentation models on a random segment. 



**Postprocessing** 

I found that detection of peaks with 7 minute distance between them gave the best score. 

But because of the strange nature of ground truth with its offsets I found that it is beneficial to additionally find micro peaks with distance 1 minute between them around dominant macro peaks with 400 minutes distance, doing this I increased probability that I did not skip more close to ground truth peaks. It gave about a 0.003 LB  boost. 

 Then, as you can see at the visualization of predictions, models can predict peaks before or after some gaps, and before or after segments with fluctuations during sleep (short wakeups during sleep periods). If peaks close to the ground truth have a higher score, these small peaks should not impact the overall score. However, if peaks close to the ground truth have a score similar to those further away, it can significantly decrease the score. Therefore, I needed to rescore the peaks using a larger context provided by segmentation models(like low probabilities above short not sleep segments and higher above real sleep segments). To do this rescoring I came up to such equations:

****To rescore onsets I used the following equation:****

new onset score = score for peak of onset from event detection model * 
                                    (1 -MeanSegm(-240 minutes) + MeanSegm(+240 minutes))/2

Where: 1−MeanSegm(−240 minutes) is 1  minus the mean of the probability of segmentation map on 240 minute segments before onset peaks. The logic is that the segmentation model should give less probability for time series to be in sleep state before real onsets. The same logic applies in reverse for  time series after onset peaks, denoted as MeanSegm (+240 minutes ) .


****To rescore wakeups I used the same equation but with inverted signs:****

new wakeup score = score for peak of wakeup from event detection model * 
                                     (1 -MeanSegm(+240 minutes) + MeanSegm(-240 minutes))/2

Where: 1 -MeanSegm(+240 minutes) -  is 1  minus the mean of the probability of segmentation map on 240 minute segments after wakeup peaks. The logic is that the segmentation model should give less probability for time series to be in sleep state after real wakeups. The same logic applies in reverse for time series after wakeups MeanSegm(-240 minutes). 

The rescoring provided me with a 0.006 LB boost. 


****To filter dummy data****

To filter dummy data, I decreased the frequency of timestamps to every ten minutes, then  took windows for each step with length of 40 and searched for similar windows by euclidean distance with threshold < 0.75. Then, added 400 minutes before the start of the detected dummy segments for onsets as dummy data and the same but after dummy segments for wakeups. It gave about a 0.05 LB boost!. 

****Maximum overlap****

I also noticed that interference with a large overlap boosted score (it allowed model to look at events with a little bit different context). So, with a window of 700 minutes, I selected an overlap of 650 minutes to fit into the 9 hour limit.


**What did not work**

- any augmentation like random noise, empty gaps with random constant value for anglez between -90 to 90 and zero for enmo to simulate common pattern in data where  multiple false positives appeared, short segments with high variation to simulate common pattern during sleep, amplitude scale.
- More complicated models like 1dUnet with RNN after each downsample.
- Rescore of close peaks by XGBoost rank method. 
- The second stage RNN run on short segments around peaks.
- Hierarchical RNN from this paper - https://arxiv.org/pdf/1809.10932v3.pdf
