# 26th place

Competition: image-matching-challenge-2022
Rank: #26
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/328848

First, I would like to thank organizers for such a great competition, it was really easy to participate, great job! 

And congratulations to all the winners! 

My plan for the competition was to see how far I can get with limited resources (Colab Pro+) just to try what is out there on the market, but I’ve ended up training some models...)))

The main thing to solve as for me are: scale and matching/smart outliers removal. 

****

**IMAGE**

1. For LoFTR I didn’t resize, but crop the image a little bit from the right/bottom which didn’t affect the keypoints' locations that were found. 
2. Multiple scales (for LoFTR and SuperGlue), all keypoints after were resized to the original image size

****

**RECIPE**

Public 0.827, Private **0.828**

For LoFTR, TTA: left-right, and merge keypoints
1. LoFTR (pre-filtered keypoints by confidence)
For multiple scales, I rescale uniformly (1.2, 1.2; 1.6,1.6) and filter out by the biggest number of outliers (with running MAGSAC, on conf.>0.7), rescale back to original image size  
2. HardNet (2k, only for images not from google urban (based on image size))
3. ALIKE, 2k 
4. DKM (TTA: left-right, 500 features subset, then >0.7 probability), default
5. Superpoint+SuperGlue (only for google urban data, 8k, 0.3thr, image original size, 1.2x, filter by confidence)
6. Combine all keypoints.
7. Magsac++ 0.185, 0.25, depends on google urban or not, 100k iter

I had also a model I’ve uploaded the last minute and wanted to select, but didn’t))) - public 0.824, private **0.831**, it is with the fine-tuned DKM model.

****

**What maybe could have worked with more time investment:**

1. Keypoints + Line matching + Homography+ Fundamental from Homography. I found this paper and migrate it to python: [http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=7371221](url)  and tried running with just multiple homography matrices sequentially + ransac, as-is (didn’t work, need to test widely more general, maybe some bug). 
I think it's an idea to try in combination with line matches. Or even maybe with progressive-x it could have worked. But I didn’t have enough time for it.

2. Find similar dataset for validation. For sure that would made life much easier. For validation I was using IMC train dataset, subsampled to 2k with the same distribution, but it is not the same. 

3. DKM customization with re-training - I managed to fine-tune and get improvement on validation on approx. 0.6. I had an issue with the probability loss which I didn’t completely solve, so fine-tuning, but not to the scale I wanted. And some ideas I tried with epipolar loss, etc.  

4. LogPolar (re-training on MegaDepth, does improve a little bit, was trying to reincarnate it, but too slow)

5. QuadAttentionTree - based on simple resize didn't get great results, decided to leave it. 

****

**What didn’t work for me:**

1. Re-training NG-RANSAC 
2. ScaleNet ([https://github.com/axelbarroso/scalenet](url))
3. GOPAC
4. se-LoFRT
5. SGMNet
6. HardNet + AdaLAM

**References** 

1. LoFTR: [https://zju3dv.github.io/loftr/](url)
2. DKM: [https://github.com/Parskatt/DKM/ ](url)
3. ALIKE: [https://github.com/Shiaoming/ALIKE/ ](url)
4. SuperGlue: [https://github.com/magicleap/SuperGluePretrainedNetwork ](url)
