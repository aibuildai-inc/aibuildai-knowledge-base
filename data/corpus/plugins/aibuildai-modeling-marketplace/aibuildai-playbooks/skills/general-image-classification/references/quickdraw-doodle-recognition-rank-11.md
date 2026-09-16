# 11th place solution with limited hardware resources up to 2xP40

Competition: quickdraw-doodle-recognition
Rank: #11
Source: https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/73808

Our solution is pretty simple.
 **1. CNN**
- We tried some relatively small models with 100k/class and 64x64, 128x128 and 224x224 input size first.
- Then we retrained with full data + weighted loss on se-resnext101 (.944),  se-resnext50 (.943), se-resnet50 (.942),  resnet50 (.942), densenet169 (.939), xception(.938), densenet121(.934) with batch size &gt;= 400, 128x128 input size.
- It took 4 weeks.

**2. RNN**
 - We tried some public kernels and modified them (deeper, bigger and stronger, replacing LSTM by attention/GRU, using timestamp on raw data) with 100k/class first.
 - Then, we trained some best of them with full data and got best results around 0.93x.
 - It took nearly 2 weeks.

**3. Inference**
 - TTA (hflip) + Ensemble (0.8 * CNN + 0.2 * RNN) + optimization(Secret Sauce + magic wand).
 - Private LB without optimization: 0.94701

**4. Our weakness**
 -  Limited on something as in the article, we were not able to try some bigger input size with big enough batch size.
 - We did not find a suitable way of filtering the training dataset on unrecognized images.
 - Only depend on optimization made our model roughly overfit. However, the best model on the public LB was our best model on private LB, too. Big thank to God for that!
