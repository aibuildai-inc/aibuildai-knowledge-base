# Third place: my approach

Competition: imaterialist-challenge-fashion-2018
Rank: #3
Source: https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/discussion/57984

First of all I want to congratulate [radek][1] and *cybercore.co.jp* who have achieved first and second place respectively. I thank the sponsors for organizing the competition as well.

My approach can be grouped into 3 parts: CNN models, stacking and F1 score optimization.

**CNN models.**
I have used the following models: 2 inception_resnet_v2, 3 Xception, 2 DenseNet201, NASNetMobile, restNet50, DenseNet121, DenseNet169, VGG16, restNet50, inception. All with keras. In the training process I applied some methods of image augmentation, using the ImageDataGenerator function of keras. Since the training set is more than 1 million images, instead of using all the images to train, I tried different values ​​of steps_per_epoch, which significantly reduced the calculation time (each epoch with all the images lasted 9 hours approximately), that way I could try different models. For some models, I started by fixing some layers, then I added a small regularization on each layer and I continued to train with all the layers. For more information, please see the following example: https://www.kaggle.com/dingkun/xception-model-training-pipeline-lb-0-9798

**Stacking**
Using the predictions of the CCN models, I trained in the validation set boosting models using cross validation. I used the LightGBM library.

**F1 score optimization**
Finally using the code in the contest planet competition: Understanding the Amazon from Space that you can see it in the following link: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475, my F1 score improved significantly.

*They did not work for me:*

 - I tried to test TTA in validation and test, but they did not improve the score.
 - I tried changing the objective function, giving more weight to the classes with less frequency, but neither did the score improve.

Jahaziel Ponce Sánchez


  [1]: https://www.kaggle.com/radek1
