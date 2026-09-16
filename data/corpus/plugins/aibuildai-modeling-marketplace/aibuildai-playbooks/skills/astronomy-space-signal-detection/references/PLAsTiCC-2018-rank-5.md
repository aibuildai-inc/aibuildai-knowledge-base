# 5th Place Partial Solution (RNN)

Competition: PLAsTiCC-2018
Rank: #5
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75040

First of all, thanks PlAsTiCC for holding such an interesting competition.
Thanks for the initial probing in the post by @titericz and others, save us all some submissions :D
Thanks for teammates for the nice cooperation, we've learned a lot from each other!
Here is solution of my parts, my teammates might share theirs too.
I'll update a github repo later.

**Basic Solution**

Basically, my parts involoves in building RNN model, since I've already got teammates proficient in lgb :D

1. For each timestamp, encode with vector composed of flux + flux_err + one-hot encoded passband + mjd-diff
2. Manual RNN Architecture Search (gru\lstm, several rnn layers, dropout, attention, conv1d with avg\max pooling): 
   - Training set is small, this is a quick trial for me.
   - Best RNN Architecture: Time-Series Branch + Meta Branch.
   - Time-Series Branch: 64-bigru + Spatial Dropout (Prob=0.1) + 64-bigru + Attention.
   - Meta Branch: Simple Fully Connected NN.
3. Add Gaussian Noise to have run-time augmentation, so that training won't fit to train set too quickly.
4. OOF from RNN with time-series branch only help building better lgb models from @cpmpml and @marcuslin
5. Single Model Scores.
   - Based on raw features only: Private 0.93560, Public: 0.91956, CV: 0.585
   - With features from @marcuslin:  Private 0.85371, Public: 0.84372, CV: 0.474
6. How the model looks like: see appendix.

**What I've tried\researched ?**

 1. RNN autoencoder: 
   - The latent vectors and reconstructed light curves (change different timing sampled) does not help in lgb modeling.
 2. Research for open-classifcation\unknown class detection problems:
   - https://arxiv.org/abs/1610.02136: Use max softmax probability for class99. Not worked for me. Oliver's approach still make sense to me more and perform better.
   - https://openreview.net/forum?id=ryiAv2xAZ: " Use GAN to generate adversarial samples to calibrate the probability. For adversarial samples, make the confidence predicted for known classes the same =&gt; Maximize (1-p0)*...*(1-p13) =&gt; Maximize prob for class 99. Seems an interesting idea to me. But did not try it eventually, don't think I could code GAN with RNN in time.
   - https://arxiv.org/abs/1704.03976: Virtual Adversarial Training. Use some tricks in loss to calibrate the confidence prediction. Still in half-way modifying the loss...


**What did not work for me ?**

 1. Adversarial Classification, predict the probabilities of how the samples are similar to test. Then use the probability as the weight for training set.
 2. CNN: Conv1d, Conv2d, dilated CNN...
 3. Using the RNN autoencoder to pretrain time-series branch, and use the pretrained weights as initial weights with warmup for classification model training.
