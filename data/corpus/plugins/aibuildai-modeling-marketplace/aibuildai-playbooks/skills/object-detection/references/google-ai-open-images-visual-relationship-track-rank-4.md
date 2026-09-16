# 4th place solution

Competition: google-ai-open-images-visual-relationship-track
Rank: #4
Source: https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track/discussion/64642

First of all, thank you Google AI and Kaggle team for hosting this great competition. I really enjoyed it and learnt a lot from the impressive dataset.  

My approach consists of 2 models. The first model is to solve "is" visual relationship. The second is "non-is". I took this way as I wanted to start simple in the beginning and to leverage a model established for the object detection track.

**1. “is” relation**

The distinct triplets of “is” relation were only 42 therefore I treated each of them as a separate class and trained SSD Resnet50 ( Retinanet ) with the 42 target classes. I chose this model as it was relatively lightweight and it was compatible with GCP TPU. 

https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config

In the given triplet labels challenge-2018-train-vrd.csv, there were 194K “is” relations out of 374K samples. I generated my “is” only model training set based on these triplets for the positive labels, mixing-in 10K images which don’t have any “is” relations as negative labels.

This first model gave me 0.09 on LB. 

**2. “non-is” relations**

Next, I moved to “non-is” relations such as “under”, “on”, “holds” etc. I trained another Retinanet with all the 62 target classes just to detect single boxes. This second training set was all the images with challenge-2018-train-vrd-bbox.csv positive labels. Then I calculated the probabilities of possible valid box combinations in each image for the most 100 confident boxes. In other words, for each image, I got the most confident 100 combinations out of 10000 combinations of boxes ( 100 x 100 ). I used this formula to estimate the confidence for each combination C_c. 

C_c = F_r x sqrt ( C_box1 x C_box2 )

Where C_box1 and C_box2 are the confidences for each box given by the box detection model. F_r is the relationship coefficient function for each box1-box2-relation combination. I used GBDT ( LightGBM ) to calculate this coefficient using simple features like box labels, relation label, Euclid distance of boxes, relative distance ( distance divided by sum of total box area ), relative x-y position of box1 to box2, etc.  

This second model gave me 0.16 on LB.

**3. Ensemble**

This is just a concatenation of the first and second model outputs. This is simply ok as they detect different types of relations. There shouldn't be any conflict or performance degradation.

The result was 0.25 on LB.

**Resources**

I didn't use any external dataset. I used only the given open image v4 dataset and labels. The 500 USD GCP credit was very helpful even though it was quite challenging to manage everything within the GCP credit for this size of dataset. GCP worked great flexibly and it allowed me to do all the work. My local env ( macbook pro ) was just too small for this dataset :) I just feel it’s quite impressive that this quality and size of computer vision dataset is publicly available for ML experiments.

Thank you all,
