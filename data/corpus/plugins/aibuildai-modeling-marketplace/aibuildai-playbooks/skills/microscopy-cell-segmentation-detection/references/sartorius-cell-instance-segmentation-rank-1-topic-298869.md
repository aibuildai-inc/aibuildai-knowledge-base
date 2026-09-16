# 1st place solution

Competition: sartorius-cell-instance-segmentation
Rank: #1
Source: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/298869

We would like to thank Kaggle and Sartorius for organizing such a great competition.

# Overview
Our solution is as follows. In many respects, it is similar to the 2nd place solution.

[[img.png]](https://postimg.cc/tskb7m27)

At the very beginning of the competition, we decided to build a solution using box-based instance segmentation, and focus more on the bbox detection part. We think the mask prediction performance is mainly limited by annotation quality so we did not pay much attention to it.
During the competition, we used COCO mAP as our validation metric, we believe that high mAP and proper thresholding would give a high LB score. Following is the validation score we achieved at the end of the competition

Evaluating bbox...                                                                                                                                                                                                                                                  
Loading and preparing results...                                                                                                                                                                                                                                    
DONE (t=0.09s)                                                                                                                                                                                                                                                      
creating index...                                                                                                                                                                                                                                                   
index created!                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                    
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=2000 ] = 0.396                                                                                                                                                                                    
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=2000 ] = 0.764                                                                                                                                                                                    
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=2000 ] = 0.364                                                                                                                                                                                    
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=2000 ] = 0.354                                                                                                                                                                                    
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=2000 ] = 0.305                                                                                                                                                                                    
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=2000 ] = 0.572                                                                                                                                                                                    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.386                                                                                                                                                                                     
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.491                                                                                                                                                                                     
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=2000 ] = 0.579                                                                                                                                                                                    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=2000 ] = 0.550                                                                                                                                                                                    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=2000 ] = 0.552                                                                                                                                                                                    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=2000 ] = 0.757                                                                                                                                                                                    
                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                    
+----------+-------+----------+-------+----------+-------+                                                                                                                                                                                                          
| category | AP    | category | AP    | category | AP    |                                                                                                                                                                                                          
+----------+-------+----------+-------+----------+-------+                                                                                                                                                                                                          
| shsy5y   | 0.334 | astro    | 0.399 | cort     | 0.456 |                                                                                                                                                                                                          
+----------+-------+----------+-------+----------+-------+                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                    
Evaluating segm...                                                                                                                                                                                                                                                  
Loading and preparing results...                                                                                                                                                                                                                                    
DONE (t=0.57s)                                                                                                                                                                                                                                                      
creating index...                                                                                                                                                                                                                                                   
index created!                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                    
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=2000 ] = 0.362                                                                                                                                                                                    
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=2000 ] = 0.767                                                                                                                                                                                    
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=2000 ] = 0.294                                                                                                                                                                                    
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=2000 ] = 0.308                                                                                                                                                                                    
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=2000 ] = 0.419                                                                                                                                                                                    
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=2000 ] = 0.461                                                                                                                                                                                    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.349                                                                                                                                                                                     
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.441                                                                                                                                                                                     
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=2000 ] = 0.516                                                                                                                                                                                    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=2000 ] = 0.514                                                                                                                                                                                    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=2000 ] = 0.469                                                                                                                                                                                    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=2000 ] = 0.465                                                                                                                                                                                    
                                                                                                                                  
                                                                                                                                  
+----------+-------+----------+-------+----------+-------+                                                                        
| category | AP    | category | AP    | category | AP    |                                                                        
+----------+-------+----------+-------+----------+-------+                                                                        
| shsy5y   | 0.328 | astro    | 0.302 | cort     | 0.456 |                                                                        
+----------+-------+----------+-------+----------+-------+                                                                        



# Bbox part
We found YOLOX performed impressively well without hyperparameter tuning. We used train/val split to monitor validation scores, and then trained models using all training data.
Things worked:
Strong feature extractor (CB DBS-FPN, EffDetD7, CSPDarknet-YOLOXPAFPN)
Large input size (1536)
Livecell pretrain

In the livecell dataset, some images have thousands of instances, and this amount could be doubled by mixup. Some operations in SimOTA would cause OOM frequently when we use large input sizes and have many ground-truth instances. @tascj0 optimized SimOTA with some CUDA extensions to save memory and speedup training.

# Segmentation, Mask R-CNN part
We used two Mask R-CNN models with CB-DBS as the backbone. According to the results of [Revisiting Mask-Head Architectures for Novel Class Instance Segmentation](https://ai.googleblog.com/2021/09/revisiting-mask-head-architectures-for.html), training was done using GT's bbox and mask. In our experiment, the score was comparable to the one using the RPN head proposal as usual.
At the time of inference, we passed the ensembled bbox to the mask head to get predictions.

However, in the end, this part only slightly boosted the score.　The main part on the mask side is UPerNet.

# Segmentation, UPerNet part
In this part, we also used the data cropped by GT's bbox and mask to train. Since the instances are small in pixel size, it’s important to do cropping and pasting accurately. Thus we used ROIAlign to crop&resize input image&mask and used grid_sample to paste a prediction to its bbox location before thresholding. It’s also important to resize training target(mask) using bilinear interpolation then threshold it, instead of using nearest neighbor interpolation. Basically, the point is to follow the setting of Mask R-CNN mask head.
We trained 4 UPerNet models with Swin or ResNet101 as the backbone, pretrain on livecell and finetune on competition data. We used ensembled bboxes for inference. In mask ensemble, we simply averaged the probs of each model's (UPerNets and Mask R-CNNs) prediction.

# Reranking
We re-scored predicted instances by the score of bbox × average of score of mask (prob >= 0.5). This improves validation COCO mAP of astro by 0.01.

# Post process
Simple thresholding, overlap removal and dropping small number_of_pixels instances.

# Code
We used mmdetection and mmsegmentation from open-mmlab to build our pipeline. We are very grateful that such an easy to use tool is being developed in open source.

Here we release a minimum version of our solution to show to implement the ideas given those awesome tools

https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution


# Acknowledge
takuoko is a member of Z by HP Data Science Global Ambassadors. Special Thanks to Z by HP for sponsoring me a Z8G4 Workstation with dual A6000 GPU and a ZBook with RTX5000 GPU.
