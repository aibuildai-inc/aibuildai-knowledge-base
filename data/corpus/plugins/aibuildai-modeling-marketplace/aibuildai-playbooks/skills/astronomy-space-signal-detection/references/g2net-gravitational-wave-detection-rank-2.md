# 2nd Place Solution: trainable custom frontend [EventHorizon]

Competition: g2net-gravitational-wave-detection
Rank: #2
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275341

First of all, I would like to express deep gratitude to organizers and all the teams for making this competition so interesting and exciting. Also, I want to express a big congratulations to the first place, who dominated this competition with a single ResNet34 model :)

Code is available at: https://github.com/analokmaus/kaggle-g2net-public

# Common settings
num epochs = 8
optimizer = Adam
scheduler = CosineAnnealingWarmRestarts(8)
loss function = BCE
cross validation = target stratified 5 fold cross validation

# Frontend architectures
Neural network architecture played the most important role for improving the performance.
Here are the frontend architectures I used. 
Trainable frontend in general outperformed fixed frontend.





# Preprocessing
I applied bandpass filter to all networks. [16, 512] for CWT-CNN and Trainable frontend CNN, [30, 300] for 1d-CNN. Whitening did not work.

# Augmentations
I tested several types of augmentations on wave and spectrogram, and only a few of wave augmentations worked. I added gaussian noise for 2d-CNN networks, and flipped wave amplitude for 1d-CNN.

# Pseudo-labeling
Re-training on soft(continuous) pseudo-labelled test dataset improved AUC by ~0.001. Label smoothing during pseudo-label also helped a bit. 

# Stacking
I kept oofs and predictions from all my experiments. 
Cross validated Ridge regression model was used to combine the outputs from neural network models. 
A constant improvement in both CV and LB was observed as I add more model into the stacking model.
Finally, I run greedy model selection and chose 20(/10/5) models to maximize CV. 

**Stacking 20 models: CV 0.88283 / Public LB 0.8845 / Private LB 0.8829**
Stacking 10 models: CV 0.88270 / Public LB 0.8842 / Private LB 0.8827
Stacking 5 models: CV 0.88242 / Public LB 0.8839 / Private LB 0.8825

## Appendix: all networks 


