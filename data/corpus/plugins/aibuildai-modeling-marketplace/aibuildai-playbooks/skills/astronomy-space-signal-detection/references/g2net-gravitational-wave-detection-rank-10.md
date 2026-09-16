# 10th Place Solution: CWT->1D Conv + CNN

Competition: g2net-gravitational-wave-detection
Rank: #10
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275353

First of all I want to thank the host and Kaggle for organizing this amazing competition. We learned a lot in this two months. 

<h1>Brief summary of my part</h1>

<h3>Pipeline</h3>
CWT/CQT is a kind of 1D Conv in nature. But it is only one layer and the kernel is too large. So making it trainable directly doesn't work. I used a little trick as below.

* CWT + Resnet34/EfNet, 488x512, 1/5 fold, train 5 epochs -> valid_auc: 0.87x
* Replace CWT with a multiple layers 1D Conv, the same size as CWT, train 20 epochs -> valid_auc: 0.880x
* Fine-tune the model with full data, 5 epochs
Single model can get 0.8790/0.8808 (private/public)

Using different parameter sets(bandpass, time shift, random channel off etc.) to train 10 models above. Ensemble them all can get 0.8803/0.8818 (private/public)

Then ensemble with my other models, the result is 0.8816/0.8828 (private/public) Unfortunately we didn't choose it as our final score.

<h3>Augmentation</h3>
* Trim the original wave to (3, 3904), Random shift +- 65
* Random shift each channel +- 5
* Random turn off one channel
* Random switch channel of two waves (target=0 only)

<h3>Others</h3>
* Using Tensorflow and TPU
* CWT/CQT conversion is on-the-fly
