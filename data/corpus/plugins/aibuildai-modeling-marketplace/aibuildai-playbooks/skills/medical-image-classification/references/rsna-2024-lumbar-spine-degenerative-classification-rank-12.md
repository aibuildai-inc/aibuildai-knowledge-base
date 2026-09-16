# 12th place solution

Competition: rsna-2024-lumbar-spine-degenerative-classification
Rank: #12
Source: https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/540001

First of all, I would like to sincerely thank the organizers for hosting this competition. I have learned a lot from it. 

My approach is very simple and follows a two-stage method. In the first stage, a point detection model is used to extract the ROI based on the predicted points. In the second stage, classification is performed on the ROI.

**First stage**

In fact, I used a separate point detection model for each type of disease. It is important to note that for subarticular conditions, i can locate the corresponding vertebrae from the spinal points. Therefore, the point model for subarticular only involves **localization on the x and y axes**, which helps differentiate between left and right during the subsequent classification.



**Second stage**

Since the first stage involves differentiating between left and right, there is no need for the model to perform this distinction in the second stage. I trained a separate classification model for each disease (2.5D CNN + GRU + AttentionHead), and developed a combined multi-modal model for the three diseases (3*2.5D CNN + GRU + 3*AttentionHead). The final results are derived from the weighted average of these models. It is noteworthy that, in order to simulate potential inaccuracies in point localization from the first stage, I introduced **random jitter** within a specified range during the training of the second stage. This approach allowed the ROI to shift slightly within a confined area, thereby enhancing my performance. Furthermore, to address the issue of label distribution imbalance, I implemented oversampling of the data from the severe category. I used ResNet50 and SE-ResNeXt50 as the backbones for my models.

**Post-process**

I multiplied all the probabilities by a temperature value of 1.3.

Finally, I reviewed the approaches shared by everyone, and I found that some of my methods are also included among them. I won't elaborate further here. Thank you all, and I look forward to seeing you in the next competition!
