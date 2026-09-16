# 14th place solution. Custom Mask and LSTM encoder/decoder

Competition: lyft-motion-prediction-autonomous-vehicles
Rank: #14
Source: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/201143

Great thanks to Lyft and kaggle team for hosting this competition. Congrats to all the winners. And a special thanks to all my teammates for teaming up!! The time I spent discussing together was very insightful and most enjoyable.

# Create Mask

Our solution’s point is that we made the custom mask.
Agent data one frame away from the same scene is very similar. So, in order to learn efficiently, we created a custom mask so that the data is loaded every N frames. For example, by sampling data every 5 frames, the training data is reduced by a factor of 5, allowing for faster learning.
To make it more efficient, we balanced the data according to the target distance. About 65% of the data had a target distance of less than 6 meters, and they were easy tasks. So in order to efficiently train the difficult data, we downsampled the data below 6 m. This halved the data. In the end, the number of data was roughly 80,000 iterations x 64 batch size.

# Model architecture


In addition to CNN, we used LSTM encoder and LSTM decoder.
- Image size: 300x300
- Optimizer: Adam
- Scheduler: Onecyclelr
- Num history: 10
- Target: positions + cosine/sine of yaw

First, we trained the model with the target distance balanced mask. Additional learning with the non-balanced mask. We used log10-scaled loss when additional training, because train/validate loss goes up and down with a large margin repeatedly. For two days of model training by using this method, we got a public score under 13, and a private score about 12.


# Didn’t work
history_positions more than 10 (Public score around 15, slow convergence)
metalabeling
Seq2seq like LSTM decoder

----------
# Discussion

## Stacking

As discussed in [this thread](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/199531#1091632), we found that stacking boosted the score, but had no time left integrating the best model to stacking...

## Satellite image

Until halfway through, I had mistakenly used satellite_debug as the 3 channel map information instead of semantic_map. Interestingly, this model got a public score of 13.3.
I didn’t pursue it further because I realized the mistake, but it seems that satellite maps can score somewhat well in this task. Try it if you like it!!!

## 3D-CNN(R2+1D)

Rasterized image data in the form of RGB×T×H×W, then fed them into R2+1D.
Validation score was equivalent to resnet18, but the process of rasterizing and the 3D-CNN model were so heavy that the training required twice ~ triple as much time.

## Only sequence model

We trained Seq2seq model not only with the target agent’s history position, but also with the surrounding vehicle’s history position. Despite no map information, this model got a validation score of 24. We integrated this trained seq2seq model into the trained resnet18 model, and fine-tuned the concat model. Sadly, improvement of the score was marginal.
