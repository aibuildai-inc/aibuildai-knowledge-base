# 2nd Place Solution, ensemble of old and new methods

Competition: image-matching-challenge-2022
Rank: #2
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/329317

Thanks to all the teams and especially the organizers' excellent work for hosting this competition, which helps to push the performance of image matching to the limit. It's my first kaggle competition but not the first time to participate in IMC, and I find it's always exciting to learn from other participants' brilliant solutions :)

## **Keypoints of my solution**

1.  A novel single model baseline achieving 0.838/0.833 in private/public LB
2. Ensemeble with other strong matchers further boosts baseline greatly
3. Normalized postional encoding for transformer-based matcher
4. No TTA/multi-resolution or post/pre processing is applied. 


## **Start with a good baseline**

My very initial motivation to participate in this competition is to benchmark a matching model we developed recently. Our model shares similar paradigm with LoFTR, using a transformer-style network to generate matches from raw image pairs without kpt detection. Since our paper is still under double-blind review, we might not be able to share more information publicly, but I'm sure the code should be available in near future :)

The model is submitted to IMC at a very early time and achieves an impressive score with 0.838/0.833 on private/public LB without any TTA. 

## **Ensemble with other methods**
Starting from the initial baseline, I ensemble more strong matchers into the solution, which always seems helpful. The results are reproted as follows.

| Matcher | Score(private/public) | Input Resolution |
| --- | --- | --- |
| Baseline | 0.838/0.833 | (1472,832) or (832,1472) | 
| LoFTR | 0.783/0.772 | (1400,792) or (792,1400)|
|SuperGlue+8k SuperPoint| 0.724/0.728| longest dim 1600|
|Baseline+QuadTree|0.854/0.848| --|
|Baseline+QuadTree+LoFTR|0.860/0.853|--|
|Baseline+QuadTree+LoFTR+SuperGlue|0.862/0.859|--|

Generally both LoFTR/Quadtree contrubutes largely to the final socre, SG boost the public score while the private score benefits less from it. For all transformer-based matcher, I keep two types of fixed resolution(chosen by aspect ratio) to maintain a roughly constant memory consumption. For all methods, I use *cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.2, 0.99999, 100000)* to get fundamental matrix.

It also needs to be clarified that I use a **normalized positional encoding** for baseline, loftr and quadtree, which seems always boost performance by 0.003-0.005(sometimes even more) w/o additional cost. I will discuss it in the next section.


## **Normalized Positional Encoding**

The original positional encoding of LoFTR is something like

**pe[0::4, :, :] = torch.sin(x_position * div_term)**, where the **x_position** is absolute pixel coordinate

A potential issue with this implementation is that when test resolution deviates from training resolution, the pe will use unseen coordinate for encoding, which harms the network's abillity to recognize locations.  To alleviate this, we apply a simple normalization as,

**pe[0::4, :, :] = torch.sin(x_position * (training_dim/test_dim) * div_term)**, where **training_dim/test_dim** is a normalizing factor to ensure all coordinate ranges within the training resolution. For example, train with 840 resolution and test with 1472 resolution, then the factor will be 840/1472=0.5706. Emprically I find this modification more or less benefit all transformer-based matcher by around 0.005.


## **Tried but didn't work**
1. Use semantic mask to filter out unmatchable objects (sky, cars , trees, pedestrians). This pre-processing is generally costly for me and doesn't deliever consistent improvements.

2. Homography pre-filtering. Use a homography hypothesis to filter out some matches and then estimate F, which is adopted by a last year's IMC winning team. Generally this pre-filtering is theoretically confusing for me and empirically doesn't give me gain. 

3. Replace LoFTR with SE2-LoFTR. At some point I guess there might be some mild rotation in the test set  (around 30-40 degree), so I try to replace LoFTR with SE2-LoFTR, but it turns out in-effective.

4. Using larger resolution for all methods, I have tried to enlarge the input resolution (like >1500), but the gain is limited and run-time largely increases.

5. Using AdaLAM to filter matches. There are some hyper-params for AdaLAM and maybe I didn't tune it well :(

## **Didn't try but may work**
1. TTA and multi-resolution input.
2. Learned outlier filters, like OANet(https://github.com/zjhthu/OANet), LMCNet (https://liuyuan-pal.github.io/LMCNet/) and CLNet (https://github.com/sailor-z/CLNet).
3. Two stage matching, cropping convisble regions/wrapping with an estimated homography at first and then matching again.
4. Ensemble SOTA descriptors, like ASLFeat(https://github.com/lzx551402/ASLFeat), ALIKE(https://github.com/Shiaoming/ALIKE) and PoSFeat(https://github.com/The-Learning-And-Vision-Atelier-LAVA/PoSFeat)
