# A brief summary

Competition: state-farm-distracted-driver-detection
Rank: #1
Source: https://www.kaggle.com/c/state-farm-distracted-driver-detection/discussion/22906

Firstly, I would like to thank Kaggle and State Farm to host such an interesting competition. I would also like to thank ZF Turbo, Dong Jiao for their starting code and many others who have kindly share their valuable ideas and experiences on the forum. I have learned a lot during this competition.

Also, sorry for the late summary, I was quite occupied in the past two weeks.
	
I understand that data set in this competition displays two unique characteristics: 

 1. The test set is much bigger than the train set, so it is easy to get overfit.
 2. The images are correlated as they are produced as frames from a video clip.  Images taken from a single video segment are put into the same category, although if judged individually, a few of them might be labelled differently.

My solution focuses mainly on these two characteristics.
	

 - **Models**

I used two models in this competition: pre-trained Vgg16, and Vgg16_3, a modified version of Vgg16 I devised to deal with overfit.

The best LB score I got with a single Vgg16 model is around 0.3 as a result of overfit. It seemed to me that the model memorized some other features rather than the desired ones. To help the model to concentrate on more essential features, I manually selected two regions of interests for each train image. One is the head region, which I believe is the most informative part of the whole image. And the other is the bottom-right quarter where the appearance of a driver’s right hand cross is highly correlated with c5.

The original train image, together with the two selected regions of interests, are the three inputs of Vgg16_3. 

The original vgg16 network except the last output layer are shared cross the three inputs. The three outputs of the second-to-last layer (the second full layer), which correspond to the three inputs, are connected with three separate final output layers. 

I used Vgg16_3 to fine-tune saved model files from Vgg16. It improves the single model score by around 0.05.	

 - **K_Nearest_Neighbor Average**

I used output of the last Maxpool layer (pool5) of Vgg16 with pertained weights to map each test image to a 512*7*7-dimensional coordinate, and use distances in this space (pool5-feature space) to define similarity among test images. A weighted average of the predictions of each image, together with their 10 nearest neighbors in pool5-feature space, would generally improve a single model score by 0.10-0.12. 
 

 - **Ensemble average**

Ensemble average was applied for each category separately.   
From confusion matrices of the CV set, I noticed that some trained models did a decent job at distinguishing a certain category from the rest, while quite confused among the rest categories. So for each category, models among top-ten percent of cross-entropy loss associated with this specific category are chosen and used for calculating ensemble average. This category-specific average outperforms significantly compared with simple arithmetic/geometric average according to my experience.

 - **Segment average**

The last step, segment average, the same as K-nearest-neighbor-average, focuses on the second characteristics of the competition as I mentioned above. I divide the test images into small groups according to their similarities in the 512*7*7-dimensional pool5-feature space. If images in one group display consistent and also confident predictions, I receive this as a strong positive signal and renormalize all the images in that group to share the same predictions. 

As I read in the forum, some of you have achieved a single-model LB score below 0.15, which is admirable. I believe you can reduce that by around 40% by using K_Nearest_Neighbor Average alone, single model. I hope all of you had enjoyed this competition as much as I did, thanks again.
