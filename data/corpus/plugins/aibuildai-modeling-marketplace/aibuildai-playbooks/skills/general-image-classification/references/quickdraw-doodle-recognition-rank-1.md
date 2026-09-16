# 1st place solution

Competition: quickdraw-doodle-recognition
Rank: #1
Source: https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/73738

Big thanks to Google for hosting this flawless competition and collecting such a great dataset. I'm also very excited to become top-5 in overall user ranking and even more excited for my teammate [Pavel Ostyakov][1] who got his second 1st place in a row! 

CNN
---

First of all, Pavel did what he does best - trained a bunch of pytorch classification models. Here is the list of architectures: resnet18, resnet34, resnet50, resnet101, resnet152, resnext50, resnext101, densenet121, densenet201, vgg11, pnasnet, incresnet, polynet, nasnetmobile, senet154, seresnet50, seresnext50, seresnext101. 

One and three channels preprocessing were used as well as different image sizes starting from 112 and up to 256. The best model got 0.946 score, in total there were around 40 models. However, the gold could be achieved with a single model.

RNN
---
I trained a couple of LSTM models based on the [best public kernel][3]. Tweaked the architecture a bit, got rid of dropouts and achieved 0.893 score. Would love to hear in the comments how you got better results.

LightGBM
--------

How do you ensemble models with too many classes? This issue has been already resolved during [Cdiscount’s Image Classification Challenge][4]. The idea is the following: for each sample and for each model you collect top 10 probabilities with the labels, then convert them into 10 samples with the binary outcome - whether this is a correct label or not (9 negative examples + 1 positive). It's easy to feed such a dataset to any booster because the number of features will be small (equal to the number of models). On top of that, I also added some time-specific features. The most significant was maximum timestamp from the raw representations of the strokes.

Secret sauce (aka "щепотка табака")
-----
As it was mentioned by [Heng CherKeng][5] a month ago [classes in a test set were equally distributed][6]. It was a very important clue which seemed to be lost in the depths of the forum. I also did not see this comment but arrived at the same conclusion by noting that (112199+1)/340=330 (number of samples in the test set plus one is divisible by the number of classes). Knowing the structure of the test set gave us an average boost of 0.7% for every model. 

The algorithm behind postprocessing is the following: for the most popular class decrease all the probabilities iteratively by the same small value until it is no longer the most popular, repeat this procedure until all classes become equal. This technique was also used in [one of the previous competitions][7] (see github link for the code).

Blending
-----

After struggling for a week and producing 17 different balanced submits Pavel left me with the 5 last attempts to improve our public score of 0.956. I used [this public ensembling kernel][9] and scored 0.957 after the first attempt. Changing weights from 5-i to 1/(i+1) gave us a slight additional boost (it mimics map3 weights) and the 1st place. 

Data
----

We used 34000 random samples as the overall holdout set and 1 mln samples for building second layer models. All first layer models were trained on 49 mln simplified samples. Raw data features were only added to LightGBM model.

Key takeaways
-------------

 - Read forum carefully, especially when [Heng CherKeng][10] is present
 - Study past solutions from similar competitions


  [1]: https://www.kaggle.com/pavelost "Pavel"
  [2]: https://www.kaggle.com/pavelost "Pavel"
  [3]: https://www.kaggle.com/huyenvyvy/bidirectional-lstm-using-data-generator-lb-0-825
  [4]: https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/45733
  [5]: https://www.kaggle.com/hengck23
  [6]: https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/70540#416772
  [7]: https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/49334
  [8]: https://www.kaggle.com/pavelost "Pavel"
  [9]: https://www.kaggle.com/paulorzp/ensemble-weighted-voting
  [10]: https://www.kaggle.com/hengck23
