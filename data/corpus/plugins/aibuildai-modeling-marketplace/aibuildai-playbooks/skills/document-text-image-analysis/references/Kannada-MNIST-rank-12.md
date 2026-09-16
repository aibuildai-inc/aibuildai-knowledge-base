# 12th place Solution

Competition: Kannada-MNIST
Rank: #12
Source: https://www.kaggle.com/c/Kannada-MNIST/discussion/122131

First, I would like to thank the competition organisers @inversion at Kaggle and @higgstachyon who created the Kannada-MNIST dataset. It has been an interesting challenge.

### The Challenge of Validation
Initially I expected to only briefly take part and implement a CNN I built 18 months ago for the original MNIST competition. However, once I got into this competition I was fascinated by the fact that the public leaderboard accuracy was much lower than my validation accuracy and much lower than in the other competition. I made a couple of posts about that [here](https://www.kaggle.com/c/Kannada-MNIST/discussion/111460) and [here](https://www.kaggle.com/c/Kannada-MNIST/discussion/111462). I'm writing this in the last hours before the result is announced. That means I am still unsure what was the best approach. I tried cross-validation as per the kernels I shared e.g. [here](https://www.kaggle.com/datahobbit/cnn-for-digit-recognition-in-the-kannada-script). I also used the Dig-MNIST dataset for validation and tried ways of getting a validation score which was a blend of the validation score in cross-validation and on the Dig-MNIST out of sample validation set. I did not use the Dig-MNIST data for training as others reported it did not improve accuracy. Also, I wanted to keep it as a holdout set. Using cv my public score got to 0.9906 but I could not get past that. I then started experimenting with train test split validation and got a single model to 0.991 on the public leaderboard but I could not recreate the result. 

### Improving the best single model
After that I started saving my model weights in keras using the .h5 format. Eventually I was able to get a single model to 0.9908 with saved weights, but even with ensembling I could not overtake the 0.9910. Eventually the thing that helped me move forward on the public leaderboard in the last few days was that I tried pseudo-labelling thanks to some very helpful comments from @nandor65 in [this thread](https://www.kaggle.com/c/Kannada-MNIST/discussion/117670). I was able to get my single model score to 0.9916 based on the output of the 0.9908 model. However, I am not sure how reliable this is because running the base model with the same settings most times led to a score around 0.9884 - 0.9898. I think that the pseudo-labelling is an improvement but I am not sure how much and if there is overfitting to the public leaderboard I suspect that the pseudo-labelling will make that worse. I found this idea quite late, and, I was training the model with hundreds of epochs on an annealing learning rate. This meant I decided there was not time to repeat the whole cross-validation. 

### Choosing 2 Different Approaches for the Final Submissions
I also decided to take 2 very different approaches for my 2 submissions to the competition. One approach was using full cross-valiation and ensembling several 10 fold cv models. The other approach was using an ensemble of many models. This did not have cross-validation or an agreed validation set from within the training data. Instead I used the public score and the holdout set to validate. I would not recommend this approach in real life. From a Kaggle perspective I was quite close to the top of the leaderboard so I was trying to tweak things to see if I could make it even higher. Often it is recommended to use your highest cv and your highest public score as your competition entries and that is what I did in the end. In the last days pseudo-labelling took me to 0.9924 and 4th place on the public leaderboard. Interestingly by the end of this process my holdout validation score of 0.9282 on the 0.9924 public score was slightly higher than the 0.9272 holdout score I got from the cv approach which scored 0.9906 on the public leaderboard.

### Things I Tried
I tried adjusting lots of things such as batch size, number of epochs, starting learning rate, model architectures, I tried adding a conv1d layer. I tried a denoising autoencoder but the results were not good. I also tried stacking with knn and svc on the proba outputs. I also tried adding the pca outputs to that. With pca the results were terrible but without the results were about the same as the unstacked, occasionally a tiny bit higher on the holdout set. Once in a while a negative weight in the blend made the holdout score better but in the past I have not been successful with negative weights so I did not pursue it for the final entry. I also tried to implement the swish activation in the last few hours but the versions of it I found online seem to have been for tensorflow v1.x and I could not make it work in 2.0. I didn't want to waste my last hours on that, maybe it was something simple I missed, I plan to return to that in future. 

### Shakeup Expectations
I expected that the safe 0.9906 score should not suffer much, if at all, in the shakeup and hoped that would make it into the top 30 (it would have been around 30th on public scores). I was also hoping that there would not be too much shakeup and my best public score resulting in 4th place would be maintained +/-1 to the private leaderboard position.

### Last Minute Ideas
In the last few hours I suddenly thought of trying mixup which is a way of blending images. I found a way to implement it [here](https://github.com/yu4u/mixup-generator) and tried to combine that with pseudo-labelling. I really didn't have time to test this properly or train it for as many epochs as I would like. The validation scores I saw in keras looked good but the holdout scores on the Dig-MNIST were not great. Maybe with more time they would have been better. In the last 3 hours I tried mixup without pseudo-labelling. 

### The Results
Most of this was written before the private leaderboard was revealed. Now that it is revealed I can see that I finished 12th on the private leaderboard after being 4th on the public. The pseudo-labelling approach did score better. My best entry would have finished 4th on the private, but in most competitions there is always an entry we didn't choose that could have done better. The choice between my best entries came down to the holdout score which was very close. I have a little bit of food for thought about why the best entry might have been better than the one I did choose. Still this is a top 1% finish so I guess that is pretty good. In a competition that awards medals 12th would have just been enough for gold.

CV with no pseudo-labelling entry 0.9906 -&gt; 0.9914 approx 30th public to approx 21st private
pseudo-labelling without cv entry 0.9924 -&gt; 0.9918 4th public to 12th private

Congratulations to all the winners and whatever result you got I hope you enjoyed it and learnt something along the way.

[edited to acknowledge final results and add details of 2 final submissions]
