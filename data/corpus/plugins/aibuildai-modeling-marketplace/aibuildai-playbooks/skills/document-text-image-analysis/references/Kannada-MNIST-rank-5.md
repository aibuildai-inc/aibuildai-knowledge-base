# 5th place Solution: Pseudo-Labeling + Dig-MNIST training

Competition: Kannada-MNIST
Rank: #5
Source: https://www.kaggle.com/c/Kannada-MNIST/discussion/122160

Thanks a lot to Kaggle and @higgstachyon for hosting this competition, it's such a great experience for me to learn and acquire knowledge from other competitors, congratulations to all the winners!

## The overall approach:
**CNN (Convolutional Neural Network)**

## What worked for me:
**1. 5-fold cross-validation, average the results (or majority vote)**

**2. Image augmentation:**
        translation, rotation, shearing and scaling

**3. CNN with Squeeze-and-Excitation Block (SE block)**
       some simple explanation from my [kernel](https://www.kaggle.com/ccchang801023/se-net-my-top1-baseline-model-with-pytorch)

**4. General training tricks**
      Reduce learning rate (ReduceLROnPlateau), early stopping

**5. Pseudo-labeling on test data (most effective)**

**6. Trick when training Dig-MNIST dataset**
        As discussed in [post1 ](https://www.kaggle.com/c/Kannada-MNIST/discussion/111462)and [post2](https://www.kaggle.com/c/Kannada-MNIST/discussion/112933),  you may notice that your model will get worse if you add Dig-MNIST dataset directly to your training dataset. There are some issues in Dig-MNIST dataset such as border cut-off effect, non-native user's style... which may cause poor performance to your original model.
       
My baseline model(trained only with 60k training data) got ~99.8% accuracy on training set but got only ~90% accuracy on Dig-MNIST dataset. It seems difficult to generalize well on that 10% rest of images, but it also means that your model is still able to predict most of the images correctly. So I trained with Dig-MNIST by following steps:
(1)Train a baseline model(with 60k training data) 
(2)Use model to perform pseudo labeling on test images (now you have 65k training data)
(3)Use model to predict Dig-MNIST dataset, add the correctly classified data to the training set (says 9200 images, now you have 74200 training data)
(4)Train a new model on new training dataset, replace baseline model with new model
(5)Repeat steps (2) to (4), and see if there's improvement on Dig-MNIST prediction

the flowchart is illustrated as below:



Pseudo-labeling and Dig-MNIST dataset training obviously improved my LB score
from (private/public) : 0.9890/0.9894  to 0.99160/0.99120.

**Some implementation details:**
- Optimizer: Adam
- Epoch of each fold: maximum 200
- Initial learning rate: 1e-3
- Data augmentation (in pytorch): 
     torchvision.transform.RandomAffine(degree=15, translate=(0.25,0.25),scale=(0.7,1.1),shear=8)
- Model training time:  One set of training for 5-fold cross validation takes about 4 hours on my PC (RTX 2080+i9-9900KF)

##What didn't work for me
**1. More folds cross-validation (up to 20 folds)**

**2. Further image augmentations:**
     colorjitter (brightness, contrast... ) will make model worse, also, you might want to 
     avoid vertically flip since it will cause wrong classification for similar digits such as 6 and 9.

**3. Critical subclasses classification(0 and 1,  6 and 9)**

**4. Classify after "Native" classifier**
     In order to deal with "newbie effect" mentioned in this [post](https://www.kaggle.com/c/Kannada-MNIST/discussion/116064), before classifying the digits, I also tried to predict whether the numbers are written by native user or not. Assumes that training set are done by native user, label 1; Dig-MNIST set are done by non-native user, label 0, then train two different models separably for these two kinds of image, but this method doesn't improve in my case.

**5. Several optimizers and learning rate reduce methods:**
     RMSProp, Adamax, AdamW, SGD, SGDR(SGD+CosineAnnealingWarmRestarts)...

**6. Several architecture of neural network**
     Resnet18, Resnet34, Resnet50, SE_Resnet(SE block +residual network) ...

**7. Transfer learning**

**8. Snapshot ensembling**

##What I want to try but got no time:
**1. Border cut-off effect augmentation**

**2. Different ensemble methods**

**3. Tune hyperparameters in more systematically, or in more smarter ways**

##Final Submission
My final models were trained over ~75K images (train set 60k + test set 5K + Dig-MNIST data 10k) and got 0.9924 on the private LB. In order to avoid overfitting on the Dig-MNIST dataset, I also submitted another models which were trained without Dig-MNIST dataset and got 0.9916 on the private LB.

Thanks for reading!
