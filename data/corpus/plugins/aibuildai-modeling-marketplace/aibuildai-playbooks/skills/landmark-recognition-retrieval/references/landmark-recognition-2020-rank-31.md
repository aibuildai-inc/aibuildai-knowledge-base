# 31st place solution - single DELG model.

Competition: landmark-recognition-2020
Rank: #31
Source: https://www.kaggle.com/c/landmark-recognition-2020/discussion/188268

31 place model:
I only used a single DELG model and with no extra training just the pre-trained weights with extra parameters tuning and post/pre-processing like e.g. below.
**RANSAC Parameters**

This was a new area for me, not much of experience, so started to read a lot, posted some finding in forum with topic The world of RANSAC https://www.kaggle.com/c/landmark-recognition-2020/discussion/180921

**RANSAC Parameters**
**MAX_INLIER_SCORE**

Here I tested some values and a value of 20 had best result in the beginning but saw in the papers that a value of 70 was used, I ended up with a value in between to be safe. Couldn’t see that this was a part of the original ransac implementation, nor finding any papers on the subject,  so didn’t gave it so much time. I also started using the DELG_SCORE_THRESHOLD_TENSOR and didn’t want to limit the data too much.

**RANSAC Parameters**
**MAX_REPROJECTION_ERROR, MAX_RANSAC_ITERATIONS, HOMOGRAPHY_CONFIDENCE**

I had the following research paper as a code reference on the subject. “Image Matching Across Wide Baselines: From Paper to Practice” from 17 aug  https://arxiv.org/pdf/2003.01587.pdf and also the Github https://github.com/ducha-aiki/pydegensac 
> “As usual, you may set pixel threshold for point to could as inlier, max_iters and confidence Besides this, there are following options:
laf_consistensy_coef. If > 0, it means that scale and orientation of the feature will be checked for consistency with the far-the-best found model. The threshold for checking px_th * laf_consistensy_coef. Becayse they are usually less precise that keypoint center, it is recommended toset laf_consistensy_coef > 1, e.g. 3.
error_type. The measure of the correspondence quality. Can be 'sampson', 'symm_sq_max', 'symm_max', 'symm_sq_sum', 'symm_sum'. For precise definition, please take a look to Chapter 4.2 Hartley and Zisserman "Multi View Geometry.pdf)"
symmetric_error_check If one should perform the additional check of the inliers with 'symm_max' error type"

I tested many of the other included PyRANSAC options Symmetric_error_check, Error_type , laf_consistensy_coef, but didn’t changed the score to the better, default parameters Pixel Threshold, Max_iters and Conf worked best for this problem.

From the paper I saw that for PyRANSAC a **MAX_REPROJECTION_ERROR** of 0.1-20 was used and they had the best threshold between 0.24-2.0, the baseline kernel had 4.0, for this Landmark problem I found with local validation that a value of 12 work best but used a value of 13 in the final model since it had a better Competition score.

For **max iterations** they wrote in the paper:

> We summarize the optimal hyperparameters – the maximum number of RANSAC iterations η and the ratio test threshold r – for each combination of methods. The number of RANSAC iterations Γ is set to 250k for PyRANSAC, 50k for DEGENSAC, and 10k for both GC-RANSAC and MAGSAC

And many public kernel used much higher iterations and they reported good results within that interval. In the DELG paper 1k was used for validation and same in the baseline. I could also see in my study of this parameter that the number of Iteration are related to number of outliner ratio, well described in this video https://youtu.be/5E5n7fhLHEM. My conclusion was that an iteration of 5k was enough to fit the inference limit and the problem, and I also finally used 5k after some interval testing.

In the same paper the **confidence level** was discussed,
> “All methods considered in this paper have three parameters in common: the confidence level in their estimates, τ ; the outlier (epipolar) threshold, η; and the maximum number of iterations, Γ. We find the confidence value to be the least sensitive, so we set it to τ = 0.999999”

I kept the baseline value of 0.99 as it had not much effect of the outcome, had already tweaked the kernel into outer space at this time.

**Max_distance**

I left the max_distance to 0.85 even though in some paper the value 0.8 are used and I could also see better result with higher value, so here we had an interval of choices, but finally left the default value, too much tweaking had already been done.

**Other notes:**

In baseline code I think removing the pre_verification_predictions part will make the inference quicker as it only use the post_verification_predictions part.

I tried the last days to cast with float16 and bigger batch size to speed up the inference, didn’t though time the inference, maybe it can help to fit more expensive settings or ensemble.

**Other models:**
I trained my own model as well but did not have time to train it better.

This was my first Landscape Recognition competition, so many new areas, ideas and valuable time trying to solve the problem.
A started to look at some of the previous competitions write-ups, research papers about the techniques and math’s used in the area, and also studied information from TF/DELG-sites, just to mark some for further reading and testing, it became a long list but a great driving license.

https://www.kaggle.com/c/landmark-retrieval-2019
https://www.kaggle.com/c/landmark-recognition-2019
https://www.kaggle.com/c/landmark-retrieval-2020

I could see that using EfficientNets would be good strategy but would take some time to finish.
First though I started with instead using the baseline and DELG-models, as stated in the papers a single model could compete in the top-3 segment, good track record to follow. I should also have started train the Efficientnets directly and not in the middle of the competition, but now this was the case this time, and using ifs and buts is not a delicate approach.

**Competition score and local validation ratio**

After looking at the past Landmark competitions I saw that it where a minimal to non-shake-ups, so fitting and rely on the leaderboard score should be save.

**EfficientNets**

First a quick recap of the training of my own model.
Starting kernel https://www.kaggle.com/ragnar123/efficientnetb3-data-pipeline-and-model, excellent work and all credit to @ragnar123 and I had a plan to follow this great winning solution https://arxiv.org/abs/2009.05132.
Trained a couple of EffNets 4-7, but with the limit of 40h/week on Kaggle TPU and 6h/day on Colab TPU, I should started the training much earlier in the competition to use them in final ensemble, Again - using ifs and buts is not delicate approach.
Best single model:
EffB7ns with GEM(3), ArcFace(0.1), No DropOut, 512Dim, SGDM, SyncBatchNormalization, weight average of 6 last epochs, no scheduler.

Switched from 384 to 512 before it converged when I saw it shouldn’t converge before competition deadline. With the colab limit of 6h I had to split the training data to half to train them in different epochs, to fit the limit, not the best situation, maybe I could have done a better solution.
Trained with random_flip_left_right and used TTA flip_left_right in Inference, increased the score.

Also used XLA and TensorFlow Auto Mixed Precision, the XLA part help to speed up the training but every attempt using mixed precision with TPU failed, could though start the training with 2x batch size but gave NAN loss after a while(?), even tried LossScaleOptimizer.
Ended up with Public 0.4924 Private 0.4721 – and not close to global minimum/maximum.

----------------------------

That's all ! 🙂 And it gave me my first silver medal! 🙂
With this competition it's also time celebrate one year with Kaggle🎉, and with almost different challenges/problems every time I hope some will return my 2nd year, facing the competitions even better the second time with some experience :)
