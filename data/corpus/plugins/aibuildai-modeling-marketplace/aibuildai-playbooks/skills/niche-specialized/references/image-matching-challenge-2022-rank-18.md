# 18th place - Ensemble of 3 matchers + bonus crazy ideas that didn't work

Competition: image-matching-challenge-2022
Rank: #18
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/328982

First of all, I and my teammate @radac98 want to deeply thank all organizers and colleagues that participated in this challenge. It was our first experience as kagglers, and we enjoyed it a lot! Very nice community, with a lot of knowledge sharing.
_____
# [Our solution]

Our best approach is actually very simple compared to several other complicated strategies we have tested during the competition. It is an ensemble of three matchers: Vanilla LoFTR + Vanilla DKM + Vanilla PDCNet_plus.
For each image pair, we do as follows:
1. For LoFTR, we perform a simple TTA rescaling of images for each pair (here, the numbers mean the size in pixels of the largest image dim for the first, and second image respectively):
LoFTR(1200, 1200)
LoFTR(1200, 600)
LoFTR(600, 1200)
2. For DKM, we randomly sample 9000 * **overlap** correspondences (*overlap* as estimated by its confidence map above a threshold of 0.05), and the sampling probability is weighted by its confidence.
3. For PDCNet, we randomly sample 6000 * **overlap** correspondences (*overlap* as estimated by its confidence map above a threshold of 0.2)

Concatenate all matches (with corrected scaling to the original image resolution) and use OpenCV's MAGSAC (15k iters, p=0.9999, px_thr = 0.25) and voilà!

We have chosen these parameters empirically based on the public LB score and it turned out to work quite nice for the private LB. 

*Little comment about the QuadTree Attention*: After reading some solutions after the competition ended, we (and several other people) were surprised that the QuadTree LoFTR alone performs very well on the LB. We simply overlooked it because the public kernel with QuadTree performed badly on the LB and its visual matches were quite bad. Maybe if we included it in our ensemble, we could have obtained better results with quite a simple modification.

______

# [Other matchers we have tried]
We also tested including other pretrained matchers and filtering methods in our Ensemble, which ended up not improving the results:
- ASLFeat
- Se-LOFTR
- SGMNet
- MatchFormer
- Patch2Pix
- SuperPoint + SuperGlue
- SIFT + HardNet
- SIFT + Key.Net kps  + HardNet desc
- Try to filter matches with AdaLAM

*In this case, we tried to add each method individually to our best ensemble. Maybe if we have tested more different combinations, they could have been useful in the resulting ensemble, but in the meantime we were testing more complicated solutions that ended up not working, which I'm describing below).

_________
# [Complicated solutions that ended up not working]
By far, the most complicated one we have tried was to try a two-view bundle adjustment, to do that, we had to decompose the estimated F matrix into K, R, t by using an optimization technique described in this paper: [Optimizing the Viewing Graph for Structure-from-Motion](https://openaccess.thecvf.com/content_iccv_2015/papers/Sweeney_Optimizing_the_Viewing_ICCV_2015_paper.pdf), where the F matrix can be decomposed to E = K'.T @ F @ K, and the essential matrix now has to satisfy the constraint in eqn (10) of the paper. We have used a scipy optimizer to optimize the focal lengths that minimizes that cost. I have tested this algorithm with images with known focal length, and surprisingly it was working quite well for pairs which the cost function was very low. Plus, with the BA, even if the value was a little off than optima,  it could converge in the optimization. After that, we than were able to obtain **K**, **R**, and **t**, by decomposing E and even obtain visually appealing 3D reconstructions. However, when attempting to bundle adjust the image pairs from the test data as a preliminary test, we observed that the estimated focal lengths were actually diverging from the pseudo-groundtruth of the test data during optimization. We suspect that this was happening because the two-view only setup + near planar structures causes the optimization to be ill-conditioned, maybe with more simultaneous views, it would be possible to obtain stable results (or maybe it was just a bug we didn't find?).  In the end, this solution was already quite complicated and we decided to focus on simpler ones.

Another thing that we have tried was to obtain sub-pix refined correspondences by using the two-view refinement stage from the work [Multi-View Optimization of Local Feature Geometry](https://github.com/mihaidusmanu/local-feature-refinement). It seemed to be working on small scale but it was slowish to run for thousands of matches, like 5 seconds + for thousands of matches which were impracticable. So we had to dramatically reduce the number of matches to a maximum of 800 to be able to run it in the time budget we had available for the submission. To only focus on very confident matches we even tried to first estimate a F matrix, get the inliers, reduce them to 800 and run the optimization on them to finally estimate a refined F, but apparently, limiting the matches was causing a big drop in the LB score.

We have also tried a similar strategy to [Guide Local Feature Matching by Overlap Estimation](https://arxiv.org/pdf/2202.09050.pdf), where we were estimating an overlap region with the ensemble and then tried to use LoFTR(the best model) to extract matches exactly on the estimated overlap region only, which would make the matcher only match parts of the image that were actually matchable, reducing the search space thus facilitating the matcher's job. This kind of worked in the beginning of the competition, when we were only using LoFTR + Local Features, but when we included DKM and PDCNet, it was actually making the LB score worse.

More things that we have tried but they failed:
-  We attempted to estimate a F matrix for each ensemble item individually, concatenated the inliers, and estimated a last F matrix from the inliers;
- We recursively estimated planes in the scene until no more planes were found, and tried to balance the number of matches on each plane -- the rationale behind this was that the F matrix estimation from a wide range of different planes would be more well-defined mathematically (similar idea to DegenSAC).

Last but not least, we thank the authors of all the amazing matching methods we have used in the experiments, releasing the source code to the community is essential to make progress in the field!
