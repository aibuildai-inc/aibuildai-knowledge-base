# 14th Place Solution

Competition: image-matching-challenge-2022
Rank: #14
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/329566

Congrats to all the winners, and thanks to organizers and all participants. This is a joint writeup with my wonderful teammate @yamash73 (big thanks to awesome team work).

**Overview**
Our final model is an ensemble of MatchFormer, QuadTreeAttention, SuperPoint+SuperGlue (2 scales) and Patch2Pix. Like the other solutions, we used only pre-trained weights for our solution.
Code: https://www.kaggle.com/code/yamash73/imc2022-14th-place-solution-private-lb-0-839/notebook
NOTE: we used the above notebook for both calculation of CV and submission. These two modes can be switched by changing the "mode" parameter in the CFG class.

**Number of matches from each model**
The maximum number of matches is set so that the number of matches from the 5 models is balanced, and the computation time is reduced. The number of matches from transformer-based methods (LoFTR/MatchFormer) is larger than that from SuperGlue in many cases.

**Ensemble**
We used the following 5 models for our final model. We used Patch2Pix for the final model, since LB/CV is slightly improved. However, the change in score is so small, and hence it is likely that this model is not required for the final model.
| # | model | resolution | Maximum number of matches|
| --- | --- | --- | --- |
| 1 | MatchFormer | 840 | 900 |
| 2 | QuadTreeAttention | 960 | 900 |
| 3 | SuperPoint + SuperGlue | 2000 | 450 |
| 4 | SuperPoint + SuperGlue | 1600 |450|
| 5 | Patch2Pix | 2048 | 400 |

**MAGSAC parameters**
We used cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.2, 0.99999, 18000) to compute the fundamental matrix like other teams. We decreased the maximum number of iterations to reduce the computation time.

**Trust CV...?**
We used two types of CV scores with fixed random seed.
- CV for 80 images (5 images for each scene): we used this CV to check whether our implementation is correct, and our approach has the potential to yield better results. If the CV was significantly low at this stage, the implementation was reviewed or that idea was not submitted.
- CV for 800 images (50 images for each scene): after calculation of CV for 80 images, we calculated this CV and used it to check if there is a correlation between CV and LB. Here we didn't consider the scope of covisibility, distance and rotation angle. So we could not see the strong correlation between LB/CV but the submissions with good LB scores had around the same CV scores (0.58~0.60 in our seed), so we were checking to see if CV could equal this score.

**Things that didn't work**
- Semantic segmentation to filter out keypoints/matches in specific class areas (sky, car, person, etc.). This never contributed to LB/CV scores...
- Refinement using ALIKE. This is based on IMC 2021 solution (LoFTR-SPP). We used ALIKE instead of SuperPoint but LB was not improved.
- Refinement using [this repository](https://github.com/mihaidusmanu/local-feature-refinement). The processing time is slow and the CV score was not good.
- Combinations of the other models. We tested PDCNet+, ASLFeat and COTR. However, the processing time is slow and LB/CV is not improved.
DEGENSAC. We used this for only one submission. The LB score was worse then.

**Things that may work**
- TTA (rotation/flip).
- Ensemble with DKM. We tested DKM using a public notebook and observed low LB score. But we should have tried the higher resolution...
- Retraining SuperGlue using Disk like IMC 2021 solution.
