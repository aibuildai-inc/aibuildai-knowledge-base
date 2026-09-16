# 3rd place Solution

Competition: Kannada-MNIST
Rank: #3
Source: https://www.kaggle.com/c/Kannada-MNIST/discussion/122167

Thanks a lot to Kaggle and @higgstachyon for hosting this competition, it's suitable for me to start kaggle competitions, congratulations to all the winners!

My method is very sample, just try to avoid overfitting. 
And the two versions code are available in my github: there are 10 tricks in the code, 3 final used.
https://github.com/HaiyangLiu1997/Kaggle_Kannada_MNIST_Pytorch

### What worked for me:
1. pseudo-labeling, increase my score from 99.280 to 99.420(private score)
2. 5 model embedding, average voting. increase my score from 99.180 to 99.280(private score)
3. using all data to train, after choosing best model base on my local 5 KFold CV 
4. fix the random seed at first. 

### What didn't worked for me:
**the didn't work, just mean no significant improve, of course the result will increase 00.020 sometimes**
1. warm-up
2. multi-lr in different layers
3. more complex model, such MobileNetV3, DenseNet
4. small batchsize
5. TTA
6. Focus loss
7. label smoothing

### My consideration
**I'm curious about the First and Second place methods, I think it will have more useful things than my simple solution**
More details are in github. Thanks for reading!
