# [9th place] Detailed report

Competition: image-matching-challenge-2022
Rank: #9
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/328796

First of all, I'd like to thank Kaggle staff and the organizers for such a fun and interesting competition, especially @oldufo and @eduardtrulls for their incredible responsiveness and for their great sample notebooks that shows us how to submit with LoFTR, DISK, and DKM. They introduced us to the new field of feature matching in the most gentle way possible and made our lives sooooo much easier!

Also, huge congratulations to @forcewithme's team with the 1st place! What you guys achieved, especially that jump from `0.836` to `0.863` just few days before the end, was amazing and inspired me to explore more creative solutions (although I failed to find anything haha). Can't wait to read the write-ups of top teams.

I joined the competition relatively late, only 19 days before the end and spent only few hours per day on it, so I did not expect a gold medal at all. I built my solution on top of the following notebooks: [IMC 2022-kornia : Score 0.725](https://www.kaggle.com/code/cbeaud/imc-2022-kornia-score-0-725) by @cbeaud, [SuperGlue baseline](https://www.kaggle.com/code/losveria/superglue-baseline) by @losveria, and [Public Baseline DKM - 0.667](https://www.kaggle.com/code/radac98/public-baseline-dkm-0-667) by @radac98. In this write-up, I'll try to describe in detail all tricks that I've used.

> Fun fact: I had a **0.851 private** run, that would put me on **5th** place, but for many reasons (I'll explain below) I did not even thought about choosing it for final submission.

-------------------------------------------------------------

# My solution

<!-- First thing that I noticed is that the competition dataset is quite small. There are a lot of image pairs, but they are not that much diverse (only 16 scenes), so clearly training or even fine-tuning descriptors, like SuperPoint or [the entire LoFTR stack](https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/320219), were not an option. Moreover, the [data description page](https://www.kaggle.com/competitions/image-matching-challenge-2022/data) clearly states that the hidden test data comes from a different source, so in my opinion the training data are provided only for local validation and experimentation purposes. -->

In this competition, post-processing is king! As @old-ufo mentioned in [this thread](https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/328976), the main goal of the organizers was to discover better post-processing techniques, since prior works in this area only focused on better modelling.

### Few words about MAGSAC++

I did not experimented with MAGSAC parameters at all. I kept my MAGSAC parameters as in this [public notebook](https://www.kaggle.com/code/cbeaud/imc-2022-kornia-score-0-725):
```
cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.1845, 0.999999, 220000)
```
Only in last submissions, I changed number of iterations to `250000` for more stable results, and just because I can afford that. I'm reporting my scores below in "public LB / private LB" format.


### Simple ensemble, TTA, and scaling: 0.824/0.833 baseline

A common bug in all public notebooks is that they don't [re-scale the resulting matched keypoints correctly back to original image size]((https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/318577#1774476). To ensemble multiple models and across TTAs, I just concatenated their keypoints together and fed everything to USAC_MAGSAC. After a few tries, here is what I added in my baseline ensemble:

- **Kornia LoFTR.** Input image is resized so that the longest side is 1200 pixels (it's the validation setting described in [the LoFTR paper](https://arxiv.org/abs/2104.00680)). TTAs: `[original, flip_lr]`.
- **SuperPoint + SuperGlue.** <!-- I used original weights by [Magic Leap][superglue_repo]. -->Input image is resized so that the longest side is 1600 pixels (it's the validation setting described in [the SuperGlue paper](https://arxiv.org/abs/1911.11763)). TTAs: `[original, flip_lr]`. To speed-up inference a bit, I also modified the code from [SuperGlue repository][superglue_repo] so that I can perform batched inference for SuperPoint and per-pair inference (with different keypoints number) for SuperGlue.
- **DKM**, proposed by @johanedstedt in [this thread](https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/321177). It seems to be a good addition to the ensemble, especially in harder samples when LoFTR and SuperPoint fails to detect many points. Unfortunately, this one can operate only on fixed image size. I found `640x480` resolution to be reasonably okay-ish. TTAs: `[original, flip_lr]`. I take only 100 points from each TTA of it though, which means DKM will have minimal effect when LoFTR and SuperGlue can detect a lot of matches, and it will take over when the other two fails to find matches.

Such simple baseline (with no hyperparameter search) got me **0.824/0.833** (public LB/private LB). Getting close to the gold range in just 1 week after I joined (and 12 days before the end of this competition), spending only few hours per day, was quite exciting. That's when I though "hmm, maybe I will have a chance here".


### DKM is weird. Pushing to 0.836/0.838

DKM is a bit more tricky to use. By default, it samples N pairs from ALL matched keypoint pairs (which is WxH pairs, where W, H is the size of input image, because it is a dense matcher) with sampling probability equal to `pair_confidence / sum_of_all_pair_confidences`. I modified it to use absolute confidence values, and for each TTA sample randomly 200 pairs among all pairs with absolute confidence larger than 0.8 (also an arbitrary chosen number). So, in the end, I'm sampling 600 pairs from DKM. I also changed the input resolution to 800x600. After such witchcraft manipulations, my score increased to **0.836/0.838** (public LB/private LB).

I have tried other sampling strategies: by relative confidence, just random on top of a threshold, with cycle-consistency check, etc. Their results in both local CV and public LB are not that good though.


### Pushing the limits with simple engineering: 0.842/0.845

It became clear that the bottleneck in my pipeline was MAGSAC (i.e. cv2.findFundamentalMat). I could not add more TTAs because I kept hitting time limits. No problem, **let's move it to a parallel thread!** (also, pre-fetch test samples)

Running MAGSAC in a different thread, in parallel with keypoints detection with LoFTR/SuperGlue/DKM allowed me to add some more TTAs: `[original, flip_lr]` for LoFTR, `[original, flip_lr, rotate(15), rotate(-15)]` for SuperGlue, and `[original, flip_lr, rotate(15)]` for DKM.

The pipeline now produces way too much matched keypoints. To lower the number of matched keypoints fed to MAGSAC, I set a confidence threshold of `0.3` for LoFTR matches and `0.4` for SuperGlue matches. I also increased the number of MAGSAC iterations to `250000`. Bam, 7 days before competition ends. instant increase to **0.842/0.845** (public LB/private LB).


### The right way to do TTA with SuperGlue: 0.848/0.847

It was 3 days before competition ends. I was pretty nervous about my position. So far, I performed TTA in a naive way. Schematically, it looks as follows:
```
tta1_img0 --> SuperPoint --> tta1_kpts0 --> | Super | --> tta1_matches0
tta1_img1 --> SuperPoint --> tta1_kpts1 --> | Glue  | --> tta1_matches1
...
ttaN_img0 --> SuperPoint --> ttaN_kpts0 --> | Super | --> ttaN_matches0
ttaN_img1 --> SuperPoint --> ttaN_kpts1 --> | Glue  | --> ttaN_matches1
```
Which is a total waste of resources. Why can't we mix-and-match, e.g. pairing `tta1` with `ttaN`. This way, we only have to execute SuperPoint (the slower network) N times, but we can perform SuperGlue as much as N^2 times and get even more matched keypoints.

```
1. Extract keypoints and descriptors: ttaK_image{0,1} --> SuperPoint --> ttaK_kpts{0,1}
2. Apply SuperGlue on various pairs of different TTAs
3. Convert the coordinates back (after SuperGlue, not before!!!)
```
My final choice of TTA pairs to apply SuperGlue on was chosen randomly: `[(original, original), (original, rotation(10)), (rotation(10), original), (original, rotation(-10)), (rotation(-10), original), (rotation(-10), rotation(10)), (flip_lr, flip_lr)]` (note that we cannot cross-match flipped and non-flipped images).

This simple idea boosted my score to **0.848/0.847** (public LB/private LB) and I used it for final submission (code: https://www.kaggle.com/code/chankhavu/loftr-superglue-dkm-with-inspiration?scriptVersionId=97150029).


### Multi-stage inference. The 0.851 submission that I ignored

I wanted to apply LoFTR on larger image sizes (i.e. with 1600 pixels on longer side), but Kaggle don't have enough GPU memory for that and it became too slow. So I tried a multi-stage approach:

```
Stage 1: inference on full image
  - LoFTR with no TTA
  - SuperGlue with enhancements as in 0.842/0.845 section
Preparation for stage 2:
  - Run a simple MAGSAC with only 1000 iterations to get inliers
  - Sample from each image a rectangle that contains all inliers
    (with min size of 200 and padding 20)
Stage 2: inference on Region-of-Interest
  - LoFTR with no TTA, scaled ROI to 2x (with limit on long side of 1200px)
  - Run DKM with above settings on 640x640 size
```

Here is the full inference code: https://www.kaggle.com/code/chankhavu/two-stage-loftr-superglue-dkm

I thought this solution is too unstable because it relies too heavily on the correct ROI, tends to ignore parts of co-visibility regions, and it showed only 0.002 improvements on public LB (over the 0.842/0.845 submission), so I decided not to try my luck with this one in final submission. Turns out that even without the improved TTAs in "0.848/0.847" submission, it outperformed that one on private LB. If I had 1 or 2 more days, I would definitely try to combine this idea with the improved TTAs for SuperGlue.

### What did not work?

- OpenGlue. I wanted to add SIFTAffnetHardnet to my pipeline, but it was too slow.
- DISK. I don't have a good keypoint matcher for it, and the naive matching algorithm (provided in the repo) is not good engouh.
- Semantic segmentation to filter out keypoints. I tried to add Mask2Former to filter out "people", "cars", "sky", etc. but quickly found out that it was not worth it. The score did not increase at all.

##### LoFTR keypoints correction with SuperPoint

In the last day, I also tried to correct LoFTR points with SuperPoint keypoints. The idea is that LoFTR is actually a region-matching (with each "point" corresponding to a whole region of 5x5 pixels or more), and is not really suitable for tasks that requires pixel-perfect keypoints. So I used this idea from IMC2021 lecture (link: https://www.youtube.com/watch?v=9cVV9m_b5Ys&ab_channel=KwangMooYi):



For each LoFTR keypoint, if there's a SuperPoint keypoint nearby (not more than 3 pixels) and it is closer to it than another LoFTR point, I'll "move" that LoFTR keypoint to the keypoint suggested by SuperPoint. The pseudocode is basically:

```
loftr_kpts_tree = spatial.cKDTree(loftr_kpts)
sp_kpts_tree = spatial.cKDTree(sp_kpts)

loftr2loftr_dist, loftr2loftr_idx = loftr_kpts_tree.query(loftr_kpts, k=2)
loftr2loftr_dist, loftr2loftr_idx = loftr2loftr_dist[:,-1], loftr2loftr_idx[:,-1]

loftr2sp_dist, loftr2sp_idx = sp_kpts_tree.query(loftr_kpts)
for i in range(len(loftr2sp_dist)):
    if loftr2sp_dist[i] < loftr2loftr_dist[i] / 2 and loftr2sp_dist[i] < 3:
        loftr_kpts[i] = sp_kpts[loftr2sp_idx[i]]
```
This idea miserably decreased my score by 0.01 points!! Code: https://www.kaggle.com/code/chankhavu/chasing-pixel-perfection?scriptVersionId=97293363

##### More "creative" TTA for SuperGlue

I think I misunderstood last year's winners (in the paper, the way I understood it is that they claimed to concatenate keypoints from different scales and TTAs before feeding to SuperGlue for matching). Inspired by that, I tried more sophisticated mix-and-match schemes, for example mixing 2 TTA groups:
```
For two TTA groups [ttaA, ttaB] and [ttaX, ttaY):
    1. Convert SuperPoint keypoints (i.e. ttaA_kpts0) back to original image coordinates
    2. Concatenate kpts0 = [ttaA_kpts0, ttaB_kpts0], same with kpts1
    3. Run SuperGlue on (kpts0, kpts1)
```
but this failed miserably (both my CV and score dropped 0.3 points with this one). I realized that you cannot feed affine-transformed keypoints to SuperGlue, because it will de-couple the keypoint position encodings from the orientation information baked into their descriptors, and produce poor results. Code: https://www.kaggle.com/code/chankhavu/loftr-superglue-dkm-with-inspiration?scriptVersionId=97151572



[superglue_repo]: https://github.com/magicleap/SuperGluePretrainedNetwork
