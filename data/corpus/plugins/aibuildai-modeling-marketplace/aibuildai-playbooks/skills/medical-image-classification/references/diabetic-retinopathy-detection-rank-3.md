# 3rd Place Solution Report

Competition: diabetic-retinopathy-detection
Rank: #3
Source: https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15845

We’d like to thank the California Healthcare Foundation and EyePACs for sponsoring and providing data for the competition. Many thanks to the Kaggle community. We leaned heavily on work shared by past competition winners.

Our solution was an ensemble of 9 convolutional neural networks. We used a variety of model architectures. Our best performing were variations of [Simonyan and Zisserman][1], closely followed by models featuring the [fractional max pooling layers][2] developed by the 1st place finisher ([Graham][3]). Also included was a model utilizing [cyclic pooling][4] as described in the winning solution to the National Data Science Bowl ([Dieleman][5]). We found that using large image sizes and combining information from both eyes were key to getting good performance.

A more in-depth summary and submission code are attached.

Edit: Fixed documentation around cyclic pooling model. See v2 of the PDF.


  [1]: http://arxiv.org/abs/1409.1556
  [2]: http://arxiv.org/abs/1412.6071
  [3]: https://www.kaggle.com/btgraham
  [4]: http://benanne.github.io/2015/03/17/plankton.html
  [5]: https://www.kaggle.com/sedielem
