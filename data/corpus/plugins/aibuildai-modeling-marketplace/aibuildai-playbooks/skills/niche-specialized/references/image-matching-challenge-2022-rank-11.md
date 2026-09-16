# 11th Place (Public: 0.847 / Private: 0.845)

Competition: image-matching-challenge-2022
Rank: #11
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/328887

First of all, thank you so much to the organizers for hosting this competition. 
This competition was educational with much of resources from organizers, and the responses to the competition's questions were very clear and fast.
Also, I can't say this lightly, but thank you so much for handling this competition even during this extremely difficult time in Ukraine. I sincerely hope the peace in Ukraine as soon as possible.

## **[Overview]**
My approach is similar to other solutions.
**In short: QuadTreeAttention LoFTR + Magic Leap’s SuperPoint/SuperGlue with Multi-Scale TTA & fp16.**
Both models are pre-trained ones and didn’t be fine-tuned or trained from scratch with the training dataset.

After trying various pre-trained models, I selected two models, QuadTreeAttention LoFTR and MagicLeap’s SuperPoint+SuperGlue, as my final submission. In the ensemble of these two models, I applied multi-scale TTA for each model.
(LoFTR and DKM were also candidates because they showed good accuracy, but for creating a simple solution I decided not to use DKM and excluded LoFTR from one final submission. However, now I think I should have incorporated DKM into final submissions for boosting the final performance.)

The resize parameters of the longer side are set to 832, 1024, 1152 for QuadTreeAttention LoFTR, and 840, 1200, 1400 for SuperGlue. When increasing the image size for QuadTreeAttention LoFTR, the number of output correspondences also increases in proportional to image size and becomes imbalanced with that of the original image size (i.e. 800), so I adjusted the confidence threshold for suppressing output correspondences for each image scale.  The confidence parameters are set to 0.20, 0.40, 0.50 for 832, 1024, 1152 image size.
The score of each model is here:
| Method | Image Size (longer side) | Public LB / Private LB|
| --- | --- | --- |
| SuperGlue (Single) | 840 | 0.721 / 0.715 |
| QuadTree (Single) | 832 | 0.788 / 0.787 |
| QuadTree (3 Ensemble) | 832, 1024, 1152 | 0.811 / 0.821 |
| QuadTree (3 Ensemble) + SuperGlue (3 Ensemble)  | 832, 1024, 1152 + 840, 1200, 1400 | 0.847 / 0.845 |


Also, when resizing images for QuadTreeAttention LoFTR, the aspect ratio of an image is kept as the original one, and the width/height size is adjusted to be divisible by 64 with zero-padding. Using these images for outputting correspondences, the inference coordinates of correspondences are possible to be out of the original image size, so I removed out-of-border points with simple post-processing.

To reduce GPU memory consumption, I also used torch.cuda.amp.autocast() for each model and converted the model’s parameters to fp16. QuadTreeAttention code internally uses CUDA C++ modules, so before passing variables to CUDA C++ modules, I converted variables to fp32 and after receiving variables from CUDA C++ modules I re-converted those variables to fp16.

## **[Training or No-Training]**
When I first looked at the dataset, the training data looked like PhotoTourism Dataset, and the test data looked like GoogleUrban Dataset, so I thought about how we could fill in the gaps between the training data and test data. 

However, after the appearance of the LoFTR 0.721 kernel by @ammarali32 (It’s a great kernel, really appreciate it), I thought that just using MegaDepth pre-trained models seems enough for this competition, and decided to concentrate on searching for nice pre-trained models in behalf of training/fine-tuning some models. 
In addition, I didn’t have enough time & GPU machine to create strong image matching models, so I just had to give up training. If there are enough resources to train models, I wanted to try PosFeat (CVPR2022) for example.

## **[Handling the time cost of MAGSAC++ calculation]**
As described in VSAC paper Table4, the worst calculation time of MAGSAC++ is larger than that of other RANSAC variants and had a bad effect on the 9hrs inference time limitation, so I tried various other RANSAC variants which can replace MAGSAC++. 
However, they all performed worse than MAGSAC++, so I tried the approach of dividing/decomposing MAGSAC++ into 2-stages, i.e., first with fewer iterations (e.g. 10000) and then with more iterations (e.g. 90000). 
Having applied this approach, the accuracy drops a bit (-0.002), but the calculation time for a difficult image pair (e.g. test image pair No.3) decreases from around 1.5s to 0.2s.

I didn’t incorporate this approach into the final submission because I prioritized accuracy, but I think it could be a useful technique if improved well.

## **[Plot results for test images]**
Yes, I agree with this post :). I also want to see the visualization results of each team.
https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/328715
The results are here. I used https://postimages.org/ service to upload these images.
  

## **[What didn’t work in my experiment]**
・Pre-Processing approaches like gamma correction, contrast correction, histogram equalization, etc.
・Post-Processing with segmentation mask to remove moving objects (e.g. car, person, sky) as used in IMC2021 winner solution. I tried mmsegmentation’s pspnet/segformer which was trained on Cityscapes, but the results were slightly worse than the original one, and also it’s time-consuming, so I decided not to use this approach.
・Other pre-trained models like MatchFormer, SuperPoint+SGMNet, SE2-LoFTR, etc. However, I think SE2-LoFTR is useful for tackling the rotation-invariant matching problems. In this competition, as denoted on the data description page, all images are correctly prepared as upright, so I guess SE2-LoFTR didn’t have much gain for test images.
