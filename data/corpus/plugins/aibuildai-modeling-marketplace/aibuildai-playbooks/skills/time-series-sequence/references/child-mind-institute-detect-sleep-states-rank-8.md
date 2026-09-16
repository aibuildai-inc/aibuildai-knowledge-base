# 8th Place Solution for the Child Mind Institute - Detect Sleep States Competition

Competition: child-mind-institute-detect-sleep-states
Rank: #8
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/460617

# 8th Place Solution for the Child Mind Institute - Detect Sleep States Competition
I thank CMI and Kaggle for hosting this interesting competition, and other competitors who worked hard to push up LB improving the quality of the event detection methods. I got my first medal 🦾 and I've learnt a lot from it.

## Context
Study context: [https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/overview](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/overview)
Data: [https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data)

# Overview of the approach

The main idea of my solution is to use as little pre/postprocessing as possible, and try to detect sleep/wake events in an end to end fashion. This is because I've observed that different postprocessing methods influence the mAP inconsistently on different folds (increase on some and decrease in others), probably due to the inconsistencies in the labels.

My pipeline contains two kinds of models, one to detect the accurate position of events (aka "Regressor"), and one to detect the probability density of the event happening inside a day (aka "DensityNet").


# Details
## Regressor
This is a simple 1D [unet](https://arxiv.org/pdf/1505.04597.pdf) which **only** uses *local information* and *anglez* to detect where the event occurs. This is motivated by [Faster-RCNN](https://arxiv.org/pdf/1506.01497.pdf) and subsequent bounding box RPN regression methods like [YOLO](https://arxiv.org/pdf/1506.02640.pdf). As we are working with 1D data and the events are well-separated, it is enough to predict two values, (onset, wakeup) per step.

### Training
A fixed hyperparameter "width" is chosen for training. The data loader will shuffle and load the onset and wakeup events of the training series_id(s), and the time series interval between $ [\text{event} - \text{width}, \text{event} + \text{width}] $. The target to optimize the model against is the following:



- Each box corresponds to a time step (5s)
- Coloured box is the ground truth location of the event
- i.e model predicts the relative position of the current step vs location of the event

Since the data is noisy, I found out that Huber loss works the best, just like the smooth L1 regression loss in [Fast-RCNN](https://arxiv.org/pdf/1504.08083.pdf), so it is less sensitive to outliers.

### Inference
For inference, the Regressor network will be run on the whole time series, to predict relative (onset, wakeup) values per step, which gives locations of interest. Gaussian kernels with std=12 and centered at the locations of interest will be accumulated together. 



 - First row: Relative location predictions by model
 - Second row: Time step
 - Blue color: Current iteration of accumulation
 - Green color: Location of interest at current iteration
 - Graph: Accumulated score 

There will be two accumulated scores, one for onset and one for wakeup. The peak of the scores gives the possible locations for the onset and wakeup events. I used the simplest peak detection method:
`
locations = np.argwhere((score[1:-1] > score[:-2]) & (score[1:-1] > score[2:])).flatten() + 1
`

An extra NMS postprocessing step so that predicted locations must be at least 6mins from each other.

### Model architecture
The Regressor is a simple 1D Unet, with 1-channel input and 2-channels output, with a [1D ResNet](https://arxiv.org/pdf/1512.03385.pdf) backbone. The hidden channels are 2, 2, 4, 8, 16, 32, 32, and 2 ResNet blocks between each pooling operation. I did not include [SE](https://arxiv.org/pdf/1709.01507.pdf) modules and used BatchNorm1D instead of InstanceNorm/GroupNorm/LayerNorm to make the network insensitive to global changes.

### Ensembling
I trained 3 models on the whole training dataset, with width 120, 180, 240 (10, 15, 20 mins resp.). The score of the 3 models are subsequently averaged before obtaining the locations.

### Local location prediction quality
To validate the performance of this model, I computed the argmax of the scores of the steps within 120 steps (10 mins) of each event. The CDF (5 fold out of fold) looks as follows:


 - x-axis percentile
 - y-axis distance in steps
 - model trained with Huber loss performs best

Around 85% of the argmax predictions lie between 3mins of the actual event. As there may be multiple peaks in the 240 step interval around the event, we expect the min distance to have less error:


 - same x, y-axis
 - min distance between events and all predicted peaks
 - model trained with Huber loss
 - ["Huber", "Gaussian" , "Laplace"] is the shape of the kernel to reconstruct the score

## DensityNet
Another network (aka "DensityNet") is needed to assign scores to each event. This network has to determine which bout of wake and sleep is the most likely (pick the longest one within a day), and honor the 30-min length and interruption rules as stipulated by [https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data](url). Therefore a longer context is necessary. 

To this end, I added transformer encoder modules in the deepest feature level of the 1D Unet to model global information. Since there are at most one event per day, the DensityNet will predict the probability density of the onset/wakeup event in the window of time. I used symmetric [ALiBi](https://arxiv.org/pdf/2108.12409.pdf) encodings to make the transformer encoder blocks translation equivariant.

### Training
I found it useful to train the model on a larger interval than that for inference to include more global context. However, my model does not predict onset/wakeup probabilities per step. Motivated by signal processing, the unknown onset/wakeup signal is a random variable within a fixed interval of interest. As such, the DensityNet is fitted against the ground truth onset+wakeup locations in a 2-day interval. 

This is simply the cross entropy loss with 12 * 60 * 24 * 2 = 34560 classes. Since the labels are noisy (and are clipped to nearest min), the target probability distribution is smoothed with the Laplace distribution. Pseudocode:

```
# (N, T, 2),     N = batch_size,    T = 34560
target_distribution = get_distribution(interval_min, interval_min + 34560, series_onset_lbls, series_wakeup_lbls)
pred_logits = model(time_series) # (N, T, 2)
loss = cross_entropy_loss(pred_logits.permute(0, 2, 1), target_distribution) # torch cross entropy acts on 2nd axis
```

Since the data is imputed with fake intervals when the watch is taken off, the DensityNet also predicts two probabilities - whether there is any onset/wakeup event respectively in the whole interval.

 - Trained 3 kinds of models
   - With anglez input only
   - With enmo input only
   - With anglez + time input (time randomly shifted with uniform distribution [-30min, 30min] to avoid overfitting)
 - Random flipping
 - Random elastic deformation

### Inference
The model now does inference only on the center 1-day subinterval of the 2-day interval. To assign scores to the events predicted by the Regressor, we use the conditional probability

$$p(t|\text{actual event in Regressor events}) = \frac{p(t)}{\sum_{t' \in \text{Regressor events}} p(t')}$$

where p(t) is the probability density predicted by the DensityNet. To account for the fake intervals, the final scores are

$$\text{score} = q * p(t|\text{actual event in Regressor events})$$

where q is the predicted probability that the interval contains some event (onset, wakeup respectively).

The scores for the events in entire series are computed by shifting the prediction window over the whole series and averaged when each prediction window overlaps.

The conditional probability can easily be computed by restricting the softmax to only on the logits of the suggested locations by the Regressor.

## Postprocessing
### Shifting predictions
Similar to other teams, I shifted the events to xx:xx:15, xx:xx:45 to increase the mAP. Note that xx:xx:30 is undesirable too as there is a 7.5 min window in the mAP score.

### Augmentations
I am surprised not much top solutions used this trick to increase the mAP. It stands to reason that events cannot be accurately labelled to 3mins / 1min precision, a slight move of the mouse will perturb the label by 1min. Below are the local 5fold out of fold mAP scores (computed with bad series and portions with missing labels removed):

(no augmentation)



(with augmentation, ~+0.002 mAP)

 - Similar performance for tolerance >3 min
 - Gains for tolerance 1, 3 mins

### Use matrix profile to remove fake data
Matrix profile can detect exact repetitions. I added that as an extra postprocessing step to remove predictions located in fake data.

# Possible things to improve
## Use result from matrix profile as input
Some top solutions added a binary feature (1, 0) to indicate whether the step lies inside the fake data, or trained only on the intervals with clean data. This should make the model more performant as compared to my current approach of letting the model predict if the events are present in the window.

## Make use of per-15 min labelling error distribution
Many top solutions made use of the pattern (uneven distribution of events per 15min). This suggests including the pattern either as input to the model or as a postprocessing step could improve the performance.
