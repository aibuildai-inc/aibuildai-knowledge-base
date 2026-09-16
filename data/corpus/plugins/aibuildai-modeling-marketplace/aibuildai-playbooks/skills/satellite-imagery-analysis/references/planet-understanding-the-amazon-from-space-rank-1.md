# My brief overview of my solution

Competition: planet-understanding-the-amazon-from-space
Rank: #1
Source: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36809

Congratulations to Team plant and Team Russian Bears.<br>
Special thanks to @Heng CherKeng for your share on forum,your codes and reports are very helpful,and the competition would not be so interesting without your generous contributions.
<br>Here is the brief overview of my solution.<br>
<h2>Challenges</h2><br>
1.Submissions are evaluated based on F2 score,so we must search thresholds after predictions and we can not find perfect thresholds.<br>
2.Label noise.We can only control the side effect of the noise.<br>
3.The public score are very close,and the division of the test data can cause shakeup..<br>
<br>
<h2>Solution</h2><br>
My solution are similar to Heng’s,so I will only introduce the differences<br>
1.I write a F2 loss function in pytorch,(thanks to @Bruno G. do Amaral,https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/34484) and add it to logloss,and I found it was helpful,but I did not check how much improvement we can get because I only compared them by using resnet 34 with 15 epochs and did not predict and submit.<br><br>
2.I used ＂Single Image Haze Removal using Dark Channel Prior＂ from http://kaiminghe.com/ ,I found that it was very useful,my networks can '<b>see</b>' the images more clearly.I hesitated to use it on weather labels,but I found that it’s quite good on all kinds of labels,especially road/water/habitation... <br><br>
3.I trained a hard example mining network on simple-net (based on heng’s code ) ,I selected 1/3 examples with largest loss and do back-propagation,I also trained other hard example methods but they did not help a lot ,so I did not use it.<br><br>
4.I trained different networks with 64*64 224*224 256*256 inputs and I used dilation of the in a resnet 34 network.Different networks have different capabilities on different labels.For example,SimpleNet 64 have good performance on Label:clear.<br><br>
5.Because I have networks with difference capabilities,so I do Ridge regression on them to predict each label separately,I mean,I have 17 regression models.<br><br>
6.My final submission are based on some strong models(based on CV),My best submission are voted based 9 models.<br><br>
7.I selected tiff images with correct labels (20000+ in train set),although they are more clear but I found they have little value ,I guess my dark channel pre-processing were strong enough. And I was afraid of the differences of  distribution of train and test set will be harmful to F2 thresholds,so I did not use them.<br><br>
8.I also found I can get big images like ZFturbo did(https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36738#latest-205616) but I did not use it because I thought it will be of little value because convolutional network models are strong enough.<br><br>
9.I tried bayesian inference but I found it was not helpful.<br><br>
10.I compared the perfect F2 thresholds with thresholds searched,this can lead to 0.001 difference .What is perfect F2 thresholds?If the distribution of test data is exactly same as train set ,then the thresholds are perfect.I tried a lot of ways(such as normalization the prediction,controlling ratio of the labels of the test set)But I found that I could not get good result,and it seemed to be very difficult to reach perfect thresholds for all of us, so I just let it be.<br><br>

<h2>Shake up</h2><br>
I tried my best to evaluate  the possible shake up in final stage of this competition.<br>
I simulated the test set use half of train set,and I split the simulated test set to 66:34 using different random seeds,I found as the seed change ,the value of public-private score will change 0.001 to 0.0025.<br>
![gap][1]<br>
the x is seed index,y is simulated public-private (I sort the seed by the value:public-private)
![dist][2]
the distributions of simulated public-private value <br><br>
As we can see,the shake up  happened on real Leardboard.
<br>
I thought carefully,I checked public LB,I must persuade myself that the public scores were not perfect metric of the capabilities of models,it was difficult,because at first glance,we have 40000+ images in public test set,it should be stable.<br><br>
But<br>
1.The gap between us were very small(0.0005-0.001)<br>
2.I thought the public-private score changed along with seed is caused by Hard Examples,and these Hard Examples can only be random guess,as they are caused by label noise.<br><br>
Finally,I persuaded myself by a guess:If we label all the images ourselves 3 times,some of then images will have different labels,because we can not say for sure some images are haze or cloudy,road or water,slash burn or not,blooming or not.....so the noise are random noises.<br>
And If the noises are random,the score get from public LB will loss in private LB. <br>
So, I adjusted my goal to keep myself in TOP 10,and decided not to care about public LB in last week and tried to seek most stable way to ensemble my models,I threw away any models may cause over-fitting,and I used just vote and ridge regression.And the final result is a big surprise to me.<br>
<br><br>
By the way,I entered this competition partly because I used Google satellite map for years when I preparing my Mountain bike trip,it is very useful when we cross mountains.I once thought I can benefit from my capabilities of distinguish water/road,habitation or not,but I found this kind of skills can only be used in my trip.:)<br>
Thanks to kaggle and organizer !

[1]:https://raw.githubusercontent.com/bestfitting/kaggle/master/amazon_demos/gap.jpg
[2]:https://raw.githubusercontent.com/bestfitting/kaggle/master/amazon_demos/dist2.png
