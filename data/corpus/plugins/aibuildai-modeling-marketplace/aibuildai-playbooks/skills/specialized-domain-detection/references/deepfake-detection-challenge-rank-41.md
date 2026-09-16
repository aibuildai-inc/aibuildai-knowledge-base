# 43rd place private LB solution

Competition: deepfake-detection-challenge
Rank: #41
Source: https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145841

Here is a brief description of the solution that led us to the 43rd place on the private LB!
The core idea is described in our paper "Video Face Manipulation Detection Through Ensemble of CNNs" available on [arXiv](https://arxiv.org/pdf/2004.07676.pdf). To reproduce the paper results, please refer to our [GitHub repository](https://github.com/polimi-ispl/icpr2020dfdc). Kaggle notebook for inference is also now available [here](https://www.kaggle.com/nicobonne/43-rank-ispl-ensamble-10-model).

#Model
We started from EfficientNetB4 and we tweaked it a bit, by adding an attention mechanism in the middle of its convolutional blocks chain. We call it EfficientNetB4Att.

#Training
We resorted to two different training paradigm, end-to-end and Siamese training with triplet loss, both considering frames as samples.

We then trained 5 instances for each model in a 5-fold strategy based on folders. For each fold, we selected 40 consecutive folders for training and the remaning 10 for validation. There was no overlap between the folds.
By doing this we ended up with 10 models, 5 trained in a end-to-end fashion and 5 trained in a siamese fashion.

During training, we considered only frames with one face, keeping the best face found by Blazeface.
As augmentation we used addictive noise, change of saturation, brightness, downscale and jpeg
compression. We trained using Adam for 20k iterations max (more details on this in the [paper](https://arxiv.org/pdf/2004.07676.pdf)).


#Inference
At inference time, we considered 72 frames per video and looked at all the faces found by Blazeface, keeping only those with a score above a certain threshold. In case a frame had more than one face above the threshold, but with discordant scores, we took the maximum score above them. The rationale is that if we have multiple faces and just one face is fake, we want to classify the frame as fake. We then averaged the scores of all the networks, the scores of all the frame of the video and computed the sigmoid.
You can look into the inference code in details [here](https://www.kaggle.com/nicobonne/43-rank-ispl-ensamble-10-model).


What a journey! Big big thanks to all my teammates from [Image and Sound Processing Lab (ISPL)](http://ispl.deib.polimi.it/) of Politecnico di Milano.
@unklb197 @edoardodanielecannas @saramandelli @bestagini
