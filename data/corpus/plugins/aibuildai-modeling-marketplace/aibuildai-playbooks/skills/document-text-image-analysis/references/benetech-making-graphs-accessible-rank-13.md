# #13th place solution

Competition: benetech-making-graphs-accessible
Rank: #13
Source: https://www.kaggle.com/c/benetech-making-graphs-accessible/discussion/418321

Thank Kaggle and the host to organize the competition. I want to do a quick write-up of my simple solution. Everything is basic, all steps can be done using Kaggle and Google Colab (Pro) computing resources.

Final Submission notebook: https://www.kaggle.com/code/namgalielei/benetech-eval-and-infer-v12/notebook

## 1. Overview
Chart type classification + Plot BBox prediction ->  OCR -> Chart data instance detection / segmentation (line) -> Data association 

## 2. Modules
### a. Chart type classification + Plot BBox prediction:
- Segmentation model pytorch Unet with backbone eca nfnet l1, auxilary classification head. I do binary segmentation for Plot BBox prediction, and softmax classifcation for Chart type classification 
- Ensemble 3 folds.
Training code: https://colab.research.google.com/drive/1WdlqUr1ONcntWfaIksDp6Z6BD4MXDJg4?usp=sharing
### b. OCR:
- Based on Paddle ppOCRv3
+ Text detection: Fine tune the light weighted mobilenetv3-dbnet on the competition's data. The target is at word group level. The ground-truth for this is provided on the json file.
+ Text recognition: Use the pretrained model. 
+ Modify some logic: First, the default post processing of Paddle TextDet (mask to quad) may result in some polygons with only 3 points (triangle instead of quadrilateral). I fixed this by finding the minimum bounding rotated rectangle. Second, when perspective-transform a cropped text, it might be rotated, so the text recognition is not able to read them well. I predict twice (no rotation and clock-wise 90deg rotation) and take the one with higher confidence score. 
### c. Chart data detection: 
- VBar, HBar, Scatter and Dot detector: Mask RCNN Resnet50 model. Training code: https://colab.research.google.com/drive/14X97mTwAU9kxS__xaehDU7SQbe_2ivP0?usp=sharing
- Line detector: a single Unet++ model with backbone efficientnet b5. Training code: https://drive.google.com/file/d/1OFoogWWWP3vHAHe9g0UsY0cCfiHKOeMp/view?usp=sharing
I annotated some images and train these models.
### d. Data association:
- Some rule-based logic to associate the chart data (bar, scatter, dot) to its corresponding ticks. 
- Project the pixel coordinates of ticks and chart data onto the axes-relative scale.

Some thing not having time to try yet: External dataset, a wider range of instance segmentation / object detection models, Generative models
