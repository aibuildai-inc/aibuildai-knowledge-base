# 21st place solution summary

Competition: image-matching-challenge-2022
Rank: #21
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/328896

Thanks again for a great competition and exciting race to the finish line about the lines! 🙂
Also extra thanks to the host that responded and interacted with professionalism, speed and full support during the competition. If we also had a host leaderboard this host would have been ranked high 😉

Thanks also the fellow competitors, this is not a solo performances show, it’s a community teamwork, and that is what we also have seen here, many contributions and top ideas.

-------------------------------------------------------------------------------

**Short summary of the solution as follows:**

The solution is to combine two public SOTA models, Loftr+ QuadTreeAttention and MatchFormer, tuned together in the same solution and using the information from each of the research papers as input as-well as the information in the training and/or testing codes, with some extra codes and parameter changes, merging them all together for this challenge. In this competition the results were better than using them separately, which is not a surprise as ensembles/ concatenating of different kind of models often leads to better results but with the cost of more time and hw consuming. But here we could take the advantage of the less GPU expansive and faster inference in the Loftr+Q framework.

**Training:**

No training, just pretrained models.

**Validation during creation and testing:**

Used the training data as validation during testing with the competition metric as validation score.

**Models/matchers used:**

Loftr + QuadTreeAttention
https://github.com/Tangshitao/QuadTreeAttention

MatchFormer
https://github.com/jamycheung/MatchFormer

**Specific solution details and parameter changes:**

Loftr + QuadTreeAttention with FeatureMatching

Outdoor Megadepth pretrained model.
config file = loftr_ds_quadtree.py
['match_coarse']['thr'] = 0.2255
['match_coarse']['border_rm'] = 1
['match_coarse']['skh_prefilter'] = True

Matchformer

Outdoor-large-LA model pretrained model
config file = megadepth_test_1500.py
['scens'] = 'outdoor'
['match_coarse']['thr'] = 0.180

**Image and fundamental matrix processing**

Image processing

Here I used the standard preprocessing in the standard Loftr+Q framework and used them for both models as what I could see worked well for both and even for other loftr models, and it had better results than a custom own codes for resizing, scaling etc, in other words Use the same as what the model is designed from.
I changed the standard values to mgdpt_df = 2 that worked best in this challenge.

Image dimension and scaling was one of key areas in the challenge. And looked like different models had different best and mandatory formats to work as it should do. Here I think I found a middle way for both. I did a classic try/except approaches, three of them. First used the max dim side of the second image, seemed working best, tested the original size with // 8*8 approach, if models failed the second try it added +64 to the size, and as the last try I just used dim 1088. All the sizes were test for best result and failproof.

Fundamental matrix processing

Then I filtered the results from the matchers with a confidence threshold 1e-5 as described in the paper/site. I even tried with and without threshold and instead sorted the top scores, e.g. use the top 1k, but the first approach was the better one here.
Next part was scaling, this is a very important section too, scale back the key points to original size and for this used the scaling from the saved scales in the output from the matchers.
After the scaling I just concatenated the key points before using the CV2 MAGSAC to create the Fundamental matrix. I had to go down to iteration of 40k to handle the timelimit.

That’s it! 🙂

----------------------------------------------

**What I also tried**

Sinkhorn

Used sinkhorn to the original loftr models as it gave better result than the dual_softmax match type. An idea was to use it as the third model, but it couldn’t fit the ensemble in terms of GPU and time limit, and it also used superglue which didn't match the prize rule criteria.

VSAC, org MAGSAC++, Degensac

Tried other algorithms like VSAC and Degensac but didn’t work better than CV2 Magsac (which is an implementation of the org Magsac++) this time, with my implementation of them. My plan with the own implementation of the org MAGSAC++ was also to combine different known algorithms and samplers like in this presentation http://cmp.felk.cvut.cz/cvpr2020-ransac-tutorial/presentations/RANSAC-CVPR20-Mishkin.pdf but didn't have the time. My implemention of the org MAGSAC++ didn't work better or the same as the CV2 one, more time needed here.
I have also shared the installation of these algorithms in the post https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/328877
