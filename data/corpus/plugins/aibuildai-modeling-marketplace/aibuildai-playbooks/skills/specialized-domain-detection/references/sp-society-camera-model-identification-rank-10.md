# 10th Place Solution

Competition: sp-society-camera-model-identification
Rank: #10
Source: https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/49366

Here is a brief overview of our modeling and solution. Please feel free to ask any questions.

1. We only used the data provided by Kaggle and Gleb´s links. We had some concerns about the Gleb's dataset license of use for some of the images, to be honest, but most of them seem to be good. 

2. We first tried some CNN models from scratch, but discovered that the well known pretrained models converge faster and gave better accuracy. Our best single model was a DenseNet201 trained  on patches of size 336x336 scoring 0.963/0.967 (public/private LB). As the accuracy was pretty high, we decide to give a try to pseudo-labeled images from the testset using the predictions from this model.

3. We applied a threshold of p&gt;=0.999 and used only unaltered images. This gives us 1116 new images to train with.

4. The same DenseNet201 model, trained now with this extra dataset, scored 0.975/0.973. After intense TTA, it gave us 0.980/0.975.

5. We trained a few more models (another DenseNet201, InceptionResnetV2, etc.) , but we couldn't improve much by just blending their predictions (and also being worried about overfitting to the LB), we decide to try stacking predictions from different models, and to train a second layer model with them.

6.  Since we only had about a week to train them, and since we didn't have too many GPU resources, we decided to do stacking with just 2-folds.

7. While we were training NN models, we tried to extract image features to train a non-NN model. We basically followed the ideas from this paper:

["Using sensor pattern noise for camera model identification" Thomas Filler, Jessica Fridrich, Miroslav Goljan][1]


8. We extracted approx. 3000 features from image linear pattern and noise pattern. We trained XGB, LightGBM and MultilayerPerceptron after applying PCA, scoring 0.898/0.881. (We found out that PCA reduced the training times of the models, but the raw dataset was trainable as well, achieving similar performance, if not better.)

9. Our final model is an XGB model of stacked predictions from 9 L1 models:
        - From images: DenseNet201, VGG16, Xception, DenseNet121, InceptionV3, ResNet50 (we tried others but we discarded them)
        - From extracted features: XGB, LightGBM, MultilayerPerceptron

10. This model (A13) scored 0.9877/0.9866. We trained the same model but excluding pseudolabeled data. We thought that they could introduce a strong bias toward these images, and after all, they were already used for all ours L1 models. This second model (A13a) scored  0.9798/0.9847, and we made the random decision of select this submission for the final, finishing 10th instead of 5th. (Our 2nd final submissions was a optimized weighted average of several submissions scoring 0.9885/0.9835, but we weren't very confident about it).


  [1]: https://www.semanticscholar.org/paper/Using-sensor-pattern-noise-for-camera-model-identi-Filler-Fridrich/e9d9d89d81e49c8fcbb8b2cc960efa4c91434319
