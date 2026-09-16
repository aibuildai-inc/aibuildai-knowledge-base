# 2nd solution

Competition: hotel-id-to-combat-human-trafficking-2022-fgvc9
Rank: #2
Source: https://www.kaggle.com/c/hotel-id-to-combat-human-trafficking-2022-fgvc9/discussion/328345

Thanks to host for organizing this challenge.
 
 ### Mask
 Statistics of the mean, variance of the size and position of the mask in the test set. Generate mask with the statistical values in training.
 
 
 ### data preprocess
 50k+fgvc9<br/>
 * data clean<br/>
 Calculate the md5 of all the data, and delete the data with the same md5 but different categories, keep data in fgvc9(A total of 45,769 categories remain). 
 
 * direction<br/>
 Using the direction model, rotate the image to face upward.
 
 
 ### train step
 step 1. <br/>
 train all data (10~20 epoch)<br/>
 step 2. <br/>
Fine-tuning the data of 3116 categories in fgvc9 (40epoch) (improve ~0.03)
 
 
 
 ### loss function
 Because the same id of the hotel varies greatly, there are bedrooms, bathrooms, etc. so we use sub-center Arcface(k=3, Dynamic Margin) as the loss function.
 
 
 ### predict
 There is little difference between using logits and retrieval in my experiments, so I choose the more convenient logits as the prediction.
 
 
 ### Model
Backbone | input size | public | private
---|---|---|---
swin-base-384|384|0.707|0.688
swin-large-384|384|0.704|0.692
eca_nfnet_l1|576|0.700|0.688
efficientNetV2-l|480|/|/
efficientNet-b5|576|/|/
ensemble|/|0.732|0.717


 ### dit not work
 Pseudo Labeling of fgvc8
