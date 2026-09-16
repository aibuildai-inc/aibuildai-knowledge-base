# 6th Place Solution

Competition: rsna-2023-abdominal-trauma-detection
Rank: #6
Source: https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/discussion/448208

First of all, Thank you to RSNA and Kaggle for hosting this competition.  
Congratulations to all competitors. 
My solution is based on my mistakes in past RSNA competitions and solutions I learned from great competitors.




## Data
I use the datasets from @theoviel. Thanks for him.  
I make 5 folds based on patient id (n=3147).  



## Models
I divided task based on the label type.
* **Organ Model** : Seg Label(nii) + Study Label
* **Bowel Model** : Seg Label(nii) + Study Label + Image Label
* **Extra Model** : Study Label + Image Label


### 1) Organ Model
First, I trained 3D segmentation model for generating masks.  




I used Qishen's 3D segmentation code. https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362607
And I cropped organ and get 15 slices for each one. Because I got many ideas from previous RSNA competitions, I started to use adjacent +-2 channels.  
And I just tried only 1 slices with 5 channels because I want to see how different, but it performs better.  
So finally I used this way. But, I think the original method makes more sense.

And then I trained CNN + sequence model With cropped volumes and study label.  



#### Model:  
1. 3D segmentation : generate masks and crop (15 slices in each organ) 
    * resnet18d
2. CNN 2.5D + sequence : train Organ classifier with study label.
    * efficientnetv2s + LSTM
    * seresnext50_32x4d + LSTM
     


### 2) Bowel Model
The 3D segmentation part is same with above.    



The only difference is I cropped 30 slices for bowel.   

I trained also CNN + sequence model with cropped volumes and study and image label.  



#### Model:  
1. 3D segmentation : generate masks and crop (30 slices in each organ)
    * resnet18d
2. CNN 2.5D + sequence : train Organ classifier with study label and image label.
    * efficientnetv2s + LSTM
    * seresnext50_32x4d + LSTM


### 3) Extra Model
For Extra model, I got slices with stride 5 and +-2 adjacent channels. 
For example, each image shape is (5, size, size) and 5 channels are [n-2, n-1, n, n+1, n+2].  
Also I just resized images to 384. I tried the other ways like 512 size, cropped images, but not working well. 

Extra Model is based on 2 stage.  
First, I trained Feature extractor and got feature embeddings.   
Second, I trained Sequence Model.   
These are enough for gold zone.   




In addition, thanks to Ian's bbox label, I could improve Extra model more.  

In my experiment, training detector with bbox label is not working.  
So I used this label to make model to focus on extravasation region.
I added segmentation head to feature extractor and it worked well.

This idea to add segmentation head comes from the previous Siim competition.

#### Model:
1. Feature Extractor
    * seresnext50_32x4d
    * efficientnetv2s
2. Sequence
    * GRU

## Things that did not work
* Yolov7 + Ian Pan extravasation boxes. Training detector to crop bboxes is not working well.
* seperate organ model.  



I truly appreciate the many competitors who produce and share great solutions every time. 
Thanks to, I was able to learn so much and become a Kaggle master. 
Also, Thank you to host and everyone who contributes to the best solution.


## Code
inference code : https://www.kaggle.com/madquer/rsna-inference-6th-solution
training code : https://github.com/sushi58373/RSNA_abdominal_trauma_6th_solution
Demo Notebook : https://www.kaggle.com/code/madquer/rsna-inference-6th-solution-clean-version/notebook
