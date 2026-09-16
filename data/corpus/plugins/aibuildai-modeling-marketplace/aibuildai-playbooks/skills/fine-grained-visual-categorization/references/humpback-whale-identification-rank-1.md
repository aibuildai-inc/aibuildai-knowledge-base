# 1st solution(classification) && code

Competition: humpback-whale-identification
Rank: #1
Source: https://www.kaggle.com/c/humpback-whale-identification/discussion/82366#latest-523382

First of all, thanks to all of my teammates, Venn, Tom and Alex.

**- Overview**
At the very beginning, we utilized softmax + fixed threshold to train the model but didn’t get a good result (&lt;0.9). In order to use new_whale images in our network, we decided to do 2-class classification for each whale class. 
After several weeks’ experiments, senet154 performs the best and we’ve got a 0.96 (both public &amp; private) result (single model). 
For further improvements, we added some tricks (will discuss later) and gets 0.969, added 4 fold cross validation with class balance post processing to achieve 0.973.
We also tried to ensemble our se154 with other networks like seresnext101, dpn131 but didn’t get any boost.

![enter image description here][1]
**- Network input and training steps**
input size is (512, 256)
We use 4 channels, RGB + masks (trained by 450 open source labels) as our input.
Step 1: Training within all labels with &gt;10 samples (this step helps to converge faster and easier)
Step 2: Training with all samples, and fixed all of the networks except the last two layers.
**-Flip images (+0.006)**
Thanks to Heng’s idea, we flip images and consider flipped id-whales as different whales and keep new whales as the same. 
![enter image description here][2]

**- Pseudo labels (+ 0.001)**
We added around 2000 test images (with confidence &gt; 0.96) into our training set
**- Class balance (+0.001 ~ 0.002)**
During our continuous improvements (from 0.8+ to 0.96), we found that the number of labels are correlated with scores. Thus we use the follow strategy to further balance our predictions:
For top 5 predictions class1 to class5, if: conf class1 – conf class 2 &lt; 0.3, and class 2 is not used in all top 1 predictions, and class 1 has been used in top 2 predictions for many times, we switch class1 and class2’s positions.

Finally, congrats to all participants, especially Heng and Dene . Congrats to 3 new GM, SeuTao, David and Weimin!

code of model
![enter image description here][3]

--**code**
https://github.com/earhian/Humpback-Whale-Identification-1st-


  [1]: https://storage.googleapis.com/kaggle-forum-message-attachments/481042/11474/network_.png
  [2]: https://storage.googleapis.com/kaggle-forum-message-attachments/481042/11466/hengck.png
  [3]: https://storage.googleapis.com/kaggle-forum-message-attachments/481042/11472/1551411492(1).png
