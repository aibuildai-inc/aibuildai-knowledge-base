# [2nd place solution] stacking with CNN

Competition: imaterialist-challenge-fashion-2018
Rank: #2
Source: https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/discussion/57934

I'd like to thank [DungNB][1], my teammate and also my co-worker, who have contributed 80% of our work in this competition (all single models are trained by him).
Our solution is stacking many CNN models: densenet201, inception_v3, inception_resnet_v2, resnet50, nasnet, xception by another CNN model and a xgboost model. Details below.

**Single models**

We built many CNN models with the same structure: base model -&gt; GlobalMaxPooling2D -&gt; Dense(2048, activation='relu') -&gt; Dropout(0.5) -&gt; Dense(2048, activation='relu') -&gt; Dropout(0.5) -&gt; Dense(NUMBER_OF_CLASSES, activation='sigmoid').

In training time, we applied some augmentations to avoid overfitting: Flip, rotate, AddToHueAndSaturation, GaussianBlur, ContrastNormalization, Sharpen, Emboss, Crop (thanks to [imgaug][2])

In testing time, we applied only one augmentation: flip.

Best single model scores *0.64xx* on public LB

p/s: we didn't optimize threshold for f1 score, just using a simple threshold 0.3 for all classes.

**Stacking**

Stacking is the key to this competition. Thanks to [Planet: Understanding the Amazon from Space competition][3] and the winner [bestfitting][4]. We have read his [solution][5] carefully and tried to create our new method.
We found that the correlation between labels is really important. In Planet competition, bestfitting built 17 models to learn the correlation between 17 classes. Should we do that for 228 classes in this competition?
No, we didn't. Instead, we used a single CNN.

We had 9 single models, each model predicted on 9k samples in the validation set. Then we concatenate and reshape each output sample to [number of models, number of classes].
We built a stacking CNN with the structure: Input(shape=(number of models,number of classes,1)) -&gt; Conv2D(filters=8, kernel_size=(3, 1)) -&gt;  Conv2D(16, (3, 1)) -&gt; Conv2D(32, (3, 1)) -&gt; Conv2D(64, (3, 1)) -&gt; Flatten() -&gt; Dense(1024) -&gt; Dense(NUMBER_OF_CLASSES, activation='sigmoid').

With a window size (3,1) we hope the CNN can learn the correlation between the prediction of single models. And the last Dense layer can learn the correlation between 228 labels.
Training this model with Kfold (k=5) on the validation set, we can get *0.714x* on public LB.

We also tried to use xgboost and MultiOutputRegressor (supported by sklearn).
Training this model with Kfold (k=5) on the validation set, we can get *0.703x* on public LB.

Simple weighted CNN model and xgboost model give us final score *0.719x* on public LB.

**What didn't work**

Too much augmentation on testing time reduces our score.

[Problem Transformation, Adapted Algorithm][6] supported by skmultilearn give poor result in this competition.
 


  [1]: https://www.kaggle.com/nguyenbadung
  [2]: http://github.com/aleju/imgaug
  [3]: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space
  [4]: https://www.kaggle.com/bestfitting
  [5]: http://blog.kaggle.com/2017/10/17/planet-understanding-the-amazon-from-space-1st-place-winners-interview/
  [6]: https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/
