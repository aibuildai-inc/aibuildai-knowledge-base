# 36th place solution

Competition: birdsong-recognition
Rank: #36
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183222

At first, congratulations to the winners and all participants who finished this competition. And also thanks to @shonenkov for kindly posting the kernel for submission.

In the early stage of this competition, I was worried about the shake in private LB. But I continued with this competition, changed my thought as it is a relatively stable. I think this task was relevant to real issues and very interesiting. I would like to thank @stefankahl,  @tomdenton, @holgerklinck and Kaggle.

  
So let me shamelessly share a simple solution.
The figure below is the overview of my model pipeline. As I'm sure there are already some great solutions out there, and will be more to come, so I'd like to give just a few points of my own.



  
1) Event aware extraction
As the participants noticed, not all the time of an audio had bird voices. Therefore, if we randomly extracted some parts of an audio (e.g. 5 sec or so), there would be no birdsong at all, resulting in a form of mislabeling for training. To mitigate that, I used a naive algorithm to extract the signal parts of the audio as shown below.




2) LogMel Mixup
When I mixed 2 LogMels, once I convert it back into the power domain and then did a logarithmic transformation again. This is because LogMels are logarithmic values, and just linear summing is not the sum of the powers.
a * log(X1) + b * log(X2) != log(a * X1 + b * X2)
And labels were not scaled with the mixup coefficients used in features, but an union of them.

3) Binary Model
I made a binary model (ResNet18) to classify call/nocall audio chunks. This slightly improved my score in publicLB, but as a result, it was a big improvement in privateLB.

4) Multi Label Model (Multi Task Learning)
Although the primary label was provided in the data for this competition, it was clear that there were actually other bird calls in the background that fell under the competition's predictive label. So, in training the model, I did multitasking learning by splitting the model top into two parts, one for the primary label and the other for the background. And I trained the models as the learning strategy below.

Step1: primary only
Step2: primary + background
Step3: psuedo soft labeling using background predictions

Last step3 improved my score by 0.004 in public and 0.006 in private, respectively.


Now, that's what I'm going to share with you. I'll see you at the next competition somewhere else. 
Until next time!
