# 7th place solution (brief summary)

Competition: image-matching-challenge-2022
Rank: #7
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/329015

## Acknowledgement 

Firstly, we'd like to thank the organizers( @eduardtrulls, @oldufo)and Kaggle staff for opening this remarkable challenge and giving us an opportunity to learn, suffer(😏) and grow without any barriers. They were kindly up to responding quickly to questions and leaving us helpful references and tutorials. Personally, although I'm quite a newbie here in Kaggle, I've never seen this before. 
  
 Secondly, thanks for sharing your notebooks( @remekkinas, @radac98) and ideas( @kirderf, @johanedstedt) so we, beginners, could learn from it and keep moving forward. I'm looking forward to seeing me publishing notebooks or discussions like these one day:
- [https://www.kaggle.com/code/remekkinas/detector-free-local-feature-matching-w-transformer](https://www.kaggle.com/code/remekkinas/detector-free-local-feature-matching-w-transformer) 
- [https://www.kaggle.com/code/cbeaud/imc-2022-kornia-score-0-725](https://www.kaggle.com/code/cbeaud/imc-2022-kornia-score-0-725)
- [https://www.kaggle.com/code/radac98/public-baseline-dkm-0-667](https://www.kaggle.com/code/radac98/public-baseline-dkm-0-667)

- [https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/324805](https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/324805)
- [https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/321177](https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/321177)
  
## Short Impression
 This is my second competition in Kaggle, but this one is the first I seriously dedicated my time and energy to.(Nights after work and weekends..😂) And, my teammates and I joined this competition with almost zero background knowledge and learned a lot from this competition, as I believe many of us here are the same. So this is a very meaningful result for me 😊

## Summary of our solution
Actually, we were surprised that other top teams ensembled at least more than 3-4 models(though we only used 2), so we learned from it that ensemble almost always works. Will use it for the next competitions. 
  
#### 0. Models
1. LoFTR([https://arxiv.org/abs/2104.00680](https://arxiv.org/abs/2104.00680))
2. DKM([https://arxiv.org/abs/2202.00667](https://arxiv.org/abs/2202.00667))

  
#### 1. K-means and Crop

We observed that the matching points tend to be concentrated on promising regions and spread out on doubtful regions such as roads. So why not cropping those promising regions? We firstly used **K-means** to cluster the matching points, found the most promising cluster based on the number of points, and picked the center point of that cluster for cropping. We **cropped** that region (512x512) from image-1, **extracted** the corresponding region(512x512) from image-2, **set** them as a new image pair, and **ran** the model to get more meaningful matching points. This method boosted our score even without ensemble.
  
Furthermore, we tried to crop two promising regions and get more and more matching points. It worked and boosted our score at first. However, this method takes so much time for inference and restricts the resource we could use for experimenting with other methods, so we turned our way. 

#### 2. Warping (back and forth) matching

We have got this idea from the previous IMC2021 challenge winning team([https://github.com/PruneTruong/DenseMatching](https://github.com/PruneTruong/DenseMatching)) that they firstly warp the image-1 based on the coarse matched homography matrix and then get more accurate one with the new warped image pair. This method was similarly implemented in our way that we first calculate the homography matrix using **cv2.findHomography** for the original image pair**(1->2 and 2->1)** with LoFTR, get the new **two image pairs(warped_image1/image2 & image1/warped_image2)**, run the model for both pairs, and concatenate the resulted matching points. 


#### 3. Ensemble

Ensembling LoFTR and DKM definitely helped to boost our score. LoFTR was more robust and strong for situations where general and global context-based understanding is required. On the other hand, DKM worked better in local and promising regions. Hence, we used LoFTR for the non-cropped image pairs to get matching points in the first place and then used both LoFTR/DKM to get matching points from the cropped image pairs.   

#### 4. VSAC(🤔?)

We found this issue([https://github.com/ivashmak/vsac/issues/5#issuecomment-1145685290](https://github.com/ivashmak/vsac/issues/5#issuecomment-1145685290)) in the official repository of VSAC and decided to somehow implement it on our notebook regardless of our lack of knowledge of how to do it without the internet. After the painful weekend, we made it by manually installing all the dependencies with dpkg install and revising python binding parts, etc.
  
It didn't improve our score with the default setting, so we tried to find better settings/parameters to at least improve the speed without any loss of score. This is the final setting that we used for the submission, but we believe this is not optimal and there should be better ones. 
  
  
 ` params = pvsac.Params(pvsac.EstimationMethod.Fundamental, 0.1, 0.99999, 100000, pvsac.SamplingMethod.SAMPLING_PROGRESSIVE_NAPSAC, pvsac.ScoreMethod.SCORE_METHOD_MAGSAC)`
`params.setParallel(True)`
`params.setLocalOptimization(pvsac.LocalOptimMethod.LOCAL_OPTIM_INNER_LO)`
`params.setPolisher(pvsac.PolishingMethod.MAGSAC)`
`params.setLOSampleSize(10*params.getSampleSize())`
`params.setLOIterations(20)`
`F, inliers = pvsac.estimate(params, f_mkpts0, f_mkpts1)`
  
  
The public LB score seemed to get worse a little bit with gaining some speed. We were not sure at the moment, so just decided to include this one for the final submission. It turns out to improve the speed (way faster than cv2.MAGSAC) without loss of score. However, it is not what we expected in the first place. (we expected more for the time we spent for it LOL) 

#### 5. Radius NMS 
Since there are so quite a lot of matching points that are redundant to each other, we thought NMS-type of filtering would work to filter those points. It certainly helped to not only decrease the time spent for ransac, but improve our score. We found that applying NMS for 0.5~1.0 pixel resolution works the best. We just implemented it based on the public GitHub repository([https://github.com/luigifreda/pyslam](https://github.com/luigifreda/pyslam)). 

#### 6. Resize 
We tried several different image sizes for inference such as 640, 840, 1080, and 1280 for the longest side of the image. 1080 was our selection considering both memory resources and the score. We found that scaling up the image to some extent helps LoFTR find more matching points. 

## Ideas that worked 
- K-Means/Crop
- Warping (back and forth) matching
- Ensemble
- Some tuning on VSAC
- Radius NMS
- Multiple Scales Matching (worked in **private score** but not selected)
- Confidence based Filtering (worked in **private score** but not selected)
- Resizing
  
## Ideas that didn't work
- Multiple Crops
- Adaptive-size Crops (based on original image size)
- YOLOX-based Filtering (filtering cars/people/bikes, etc.)
- Rotation and Projection based TTA

## Lessons learned 

- **Ensemble** almost always works great. I should exploit its benefits more for the next competitions for sure. 
- Learning from challenges is fun, great, and very helpful for growing as an engineer, but very painful at the same time.
- Team play works better than solo play. (at least for me) 
- There are wonderful people out here I can learn from.
- I like coffee.
