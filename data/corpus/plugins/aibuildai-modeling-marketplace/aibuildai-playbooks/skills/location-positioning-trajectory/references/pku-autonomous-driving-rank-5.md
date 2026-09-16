# (Another part of) 5th place solution

Competition: pku-autonomous-driving
Rank: #5
Source: https://www.kaggle.com/c/pku-autonomous-driving/discussion/127145

This is a summary of me and @erniechiew's approach as part of our 5th place solution. For the other part, please see [https://www.kaggle.com/c/pku-autonomous-driving/discussion/127065].

## Quick Overview

Our approach is based on 2D bounding boxes: a 2D object detector (Faster R-CNN) is fine-tuned on this dataset to detect the neighboring cars. A separate network then regresses the 6D position of each car based on raw features from the bounding boxes, as well as image features from the bounding box crops.


### Our Solution in More Detail

There are two key components to our solution:
1. 2D bounding boxes
2. 6D-pose regression

.png?generation=1579744111028907&amp;alt=media)




## 1. 2D Bounding Boxes


The “backbone” of our approach is 2D bounding boxes. Because the training data does not provide labeled bounding boxes, we first modify this kernel [https://www.kaggle.com/hypocrites/simple-eda-with-imageai-object-detection] to obtain bounding boxes for the training set. The key change we made to the kernel is that we used the provided 3D car models to obtain car dimensions, and adjusted our bounding boxes based on the car dimensions for a given car label (whereas the kernel uses a generic car dimension for all cars). We call the boxes obtained from this method as “ground truth boxes”.

For the test set, we cannot obtain bounding boxes with this method, since the method requires the very data points we are trying to predict! We therefore resorted to 2D object detectors. First, we fine-tuned Faster-RCNN (from maskrcnn-benchmark) on the BDD100k dataset. The BDD100k tuned model is then further fine-tuned on our “ground truth boxes”. 


We found that training with larger input images resulted in noticeably better validation MAP and LB score. In the end, we trained 3 separate detector backbones with the following configurations:
1. X-101-32x8d-FPN (input size 1373x1100)
2. R-101-FPN (input size 2000x1602)
3. R-50-FPN (input size 2499x2002)



The 3 models are then used to predict bounding boxes for cars in the test images. Within each model, overlapping boxes are discarded based on an NMS threshold of 0.70, and any remaining boxes below 0.70 confidence are further discarded.

The boxes from the three models are then ensembled using weighted boxes fusion (https://github.com/ZFTurbo/Weighted-Boxes-Fusion) with equal weights. We found that by including only the boxes that are successfully fused/merged by all three models, our score improved. We call the boxes predicted by the object detectors as “predicted boxes” (as opposed to ground truth boxes).


## 2. 6D-pose regression

With the ground truth boxes and predicted boxes in hand, we then regress the cars’ positions using a downstream network that we simply call Q-Net (Q for Quaternion).

### Input:

Q-Net consumes the following two inputs:

1. Car bounding box crops (RGB images, resized to 128x128)
2. 10 numerical features from the bounding boxes
- Box width
- Box height
- Box width to height ratio
- Box area
- X-coordinate of the box center
- Y-coordinate of the box center
- “2D-distance” of box center from camera 
- “2D-angle” between box center and camera
- Whether the box is close to the left boundary of the image
- Whether the box is close to the right boundary of the image


Both inputs are standardized appropriately.


### Output:

Car translation (x, y, z)
Car rotation (quaternion)



## Brief Network Architecture

The bounding box crops are fed into a pre-trained DenseNet121 model from torchvision, while the numerical features are connected to FC layers. Both of these “input paths” are then connected to 2 “output paths”, one for predicting car rotations, and another for car translations.  

The intuition here is that both bounding-box crops and numerical features work hand-in-hand in regressing the 6D-pose, and so information from both input components should be “communicated” or “shared” with both output paths.



## Some Training Details

We trained Q-Net on the ground truth boxes, and made predictions on the predicted boxes. Initially, we thought that training with out-of-fold predicted boxes would yield better test set generalization, but this was not the case. This is most likely because in the latter case, there was a challenge to map the predicted boxes with the corresponding ground truth pose information for a given car.


For translation we used L1-loss. For rotations, as mentioned, we converted the angles into quaternions, and used Dot Product Loss (https://arxiv.org/pdf/1901.09366.pdf).

We trained Q-Net on 10-folds and averaged the predictions from each fold. For translation, we simply averaged the predicted x, y, and z values respectively. For rotation, we averaged the quaternion predictions using (https://github.com/christophhagen/averaging-quaternions) before converting back to Euler angles.



## Score Summary

Our best model attains 0.124 on public LB and 0.116 on private LB.


## Final Ensemble

Thanks to our team member @uiiurz1's Centernet model, we had relatively diverse models to ensemble. However, ensembling the two models of different nature was slightly challenging. We used weighted points fusion (weighted boxes fusion but with IOU replaced with 3D-distance) to ensemble our predictions, credit to @uiiurz1. By ensembling, we obtained a decent boost in score: 0.131 public LB, 0.123 private LB.


## Closing Remarks

We would like to congratulate all the winners, and really anyone who benefited in some way from the competition. We want to thank our teammate @uiiurz1 for his great insights and collaborative spirit. Finally, we would also like to thank the competition host(s) and Kaggle for organizing the competition. Putting aside issues regarding the unknown competition metric and labelling methodology for cropped cars, this competition was intriguing enough to keep our minds occupied thinking about car position estimation during our drives home from work the past couple of months :)
