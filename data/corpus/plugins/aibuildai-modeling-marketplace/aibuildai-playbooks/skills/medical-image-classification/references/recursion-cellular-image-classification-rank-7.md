# 7th place solution summary

Competition: recursion-cellular-image-classification
Rank: #7
Source: https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110335

Thank you Recursion and Kaggle for organizing this unique and challenging competition! When I decided to participate in it, I was impressed by the quality of data and organization, [this beautiful site](https://www.rxrx.ai/) alone gave me a lot of motivation. GCP and TPU credits were also indispensable for me, hope to see more in the future. The leaks that were found were handled smoothly by Kaggle, specifically thanks to [Sohier Dane](https://www.kaggle.com/sohier). And credits to [Giulia Savorgnan](https://www.kaggle.com/giuliasavorgnan) for reporting the second leak. 

I was very fortunate to team up with Yuval on this one. His models scored and maintained top10 for very long time, and I was able to contribute meaningfully only much later in the competition. I still can't wrap my head around how it was possible for him to put so much work in this competition, work full-time at a day job, but also prepare for and run UTMB 171k ultra-marathon race at the end of August. He finished after 50k due to injury, but in my eyes it is already a super-human level of toughness. 

# Setup

All my models I trained with pytorch and TPUs, and used exactly all 600$ of the free GCP credit that I had. My setup I described [in this post](https://towardsdatascience.com/running-pytorch-on-tpu-a-bag-of-tricks-b6d0130bddd4). Despite being able to get overall good training speeds, the experience using TPUs with pytorch was rough. The main problem was that it hangs unexpectedly after a few hours of training. At times it was so annoying that I questioned myself if I lived this life correctly. Pytorch/XLA guys tried to help on the forum, but at the moment I think Pytorch/XLA is just not quite production ready (but I will still be happy to receive free TPUs for the next competition!).

# The model

From the beginning I started with resnet50 and didn’t have an opportunity to successfully try anything else. In hindsight, it was a good choice for this competition. Yuval has used other backbones, I believe he will describe his work in a separate post. 

- Resnet50 backbone
- ArcFace (m=0.2, s=30, 512 features), used “as is” from the beginning of training, no adjustments. A vanilla implementation [from the original article](https://arxiv.org/abs/1801.07698).
- 384x384 images cropped from original 512x512 with random shifts, flips and all angles rotations (albumentations package)
- Normalization of images per experiment and channel, with small randomization
- Not using well, plate and experiment meta info.
- 3 folds each containing full experiments
- Training for about 10 epochs together and then a separate model per cell type.
- Each site is treated as a separate sample
- Adam optimizer, decreasing learning rate
- HUVEC-05 is removed, as it is too different from HUVEC test experiments
- HUVEC-18 is moved to the train set

Besides this standard structure described above I had 3 special features in the model. I have not done any serious comparisons with and without them, but my feeling is that the first gives me a major boost, and second and third some additional smaller improvements (but a fair comparison vs regular pseudo-labeling is definitely missing).

1. **Training with test**. The model is trained on all train, test and control images together, predicting 1139 classes (1108 + 31 controls). For the test images I took their softmax output as a target for the log-loss. In this case it can be shown that the gradient of the features before the softmax is zero, that is, these samples have no effect. It makes sense intuitively, I take the predicted probability distribution as a target, - this is already the perfect prediction according to log-loss. But then I modify this target probabilities vector for each sample according to the rich structural information that we hold, - and this is quite unique for this competition. First, - [the plates leak](https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/102905), all components besides 277 are zeroed out. Second, - each set of 277 samples contains every sirna only once. This second one was not easy to enforce, but as a soft constraint in the loss it worked reasonably well. Additional note here, - validation set data was in the train as well, with hidden sirnas, to simulate similar setting to test. One more thing, - ArcFace didn’t work well here, so it was turned off for test/validation samples. And the last point, - I think this approach in general can be thought of as a dynamic type of pseudo-labelling.
2. **Mean normalization of ArcFace features**, kudos to Yuval who pioneered this idea. I save EMA of 512 features before ArcFace for each sample, calculate their mean per experiment and overall mean (vectors of length 512 again), and to each sample on the forward pass to these features I add the overall mean and subtract its experiment mean. This way the features entering ArcFace don’t have per-experiment bias and we improve this side of domain adaptation problem. Theoretically.
3. **Incremental hard pseudo-labelling (PL)**. The PL in point 1 above is a probabilistic PL, in the sense that the target is a probabilities vector of size 277, but not forced to be a specific value, anything is good as long as it is in the set. In this addition to the model at the end of the training I try to push all reasonably confident test and validation samples to lock into one sirna, which becomes its target. I can then zero this sirna for all other samples in the 277-set, which helps them to converge. Once locked, this sample also gets into ArcFace (remember, it is turned off for the test). In practice, judging by validation experiments, first 200 samples out of 277 can be easily locked this way without a single error for all cell types. They were confident anyway, so probably not a huge benefit, but helpful. 

# Test-time-augmentation (TTA) and ensemble

For each sample and each site I run 16 predefined transformations, aggregating those 16 by quantile 75, and then aggregating by gmean between the sites. I then aggregated also by gmean between the folds, and again by gmean between different runs (both me and Yuval had 2 runs). Guess what aggregation function we used to merge the results between me and Yuval’s probabilities? That’s right, gmean again. It just worked very well everywhere, despite me trying 7-8 competitors. We selected aggregation weights for all of this by CV and public LB, different per cell type.

# Mixed integer programming (MIP)

The output of the above procedure is a set of `18*4=72` matrices 277 by 277 where each row sums up to 1 (samples) but column sums range from 0.5 to 3 (sirnas). We want columns to sum up to one as well, as we know that each set contains every sirna only once. To achieve this we divided the matrices iteratively by sum of columns and then sum of rows several times (e.g. 10), to force it into desired [stochastic matrix structure](https://en.wikipedia.org/wiki/Stochastic_matrix). This procedure does reliably improve score. It doesn’t sound ideal, but we couldn’t do any better. 

And then the Hungarian algorithm, to get the solution. The name Hungarian algorithm was introduced to us by [Christopher Berner](https://www.kaggle.com/christopherberner) in [this great kernel](https://www.kaggle.com/christopherberner/hungarian-algorithm-to-optimize-sirna-prediction). But we solve this matching problem with pulp package with Cplex solver, with a mixed integer programming formulation so short that I can paste it here. Cplex solves each such matrix in a couple of seconds.

```
prob = LpProblem("Recursion",LpMaximize)    
p_vars = LpVariable.dicts("match",(range(L),range(L)),0,1,LpInteger)
    
# objective
prob += lpSum(lpSum(p_vars[d][i] * mat[d,i] for i in range(L)) for d in range(L))
    
# constraints
for d in range(L):
    prob += lpSum(p_vars[d][i] for i in range(L)) == 1, "OneSelected_%i"%d
for i in range(L):
    prob += lpSum(p_vars[d][i] for d in range(L)) &lt;= 1, "NoDuplication_%i"%i
```

# LB probing (1st place public explained)

The public set is comprised of experiments HUVEC-17, HEPG2-08, RPE-08, U2OS-04, as can be easily verified and initially [discussed here](https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/98075).

The LB probing idea stems from the observation that the above MIP formulation does not solve the problem with all the available information. Specifically, we know the scores of our own submissions, so we can decrease the feasible set with two constraints per submission:

```
for s in range(S):
    prob += lpSum(lpSum(p_vars[d][i] for i in range(L) if subs_sel[s,d,i])
        for d in range(D)) / D &gt;= vals[s], "SubLow_%i"%s
    prob += lpSum(lpSum(p_vars[d][i] for i in range(L) if subs_sel[s,d,i])
        for d in range(D)) / D &lt;= vals[s] + 0.001, "SubHigh_%i"%s
```

Note that here we need to solve for 16 sets, 4431 public samples simultaneously, because the constraints use all of them together. The LB probing procedure is simple, we generate a new solution with this formulation, it is a feasible solution in the sense that this solution can be the actual one given all the constraints. After submitting it we get some non-perfect score, which makes the found solution to be infeasible now, and we start from the beginning again.

This is the same approach that I used in LANL earthquake prediction competition, [this is a post about it](https://towardsdatascience.com/how-to-lb-probe-on-kaggle-c0aa21458bfe). It contains some interesting insights about the approach, and it is the same as in this case, so take a look if you are interested in more details.

Besides the constraints and the feasibility question, there is of course still the objective to maximize the likelihood of the matching, same as in initial MIP formulation. It helps to look for a solution in the right region, as this combinatoric problem is huge. Without good objective function which already solves most of the problem by itself (gives high public LB scores), it is hopeless to advance anywhere with this approach. For good and for bad, the problem here is easy enough for the approach to work. When we got to un-constrained score 0.979 (about 93 errors) the probing got traction and it solved the puzzle in 18 submissions from that. Of course building also on all 200+ submissions that we did before, both probing and regular ones, but getting a high score by itself is when it really clicked. 

Two comments on the LB probing
1. It would not be possible to get anything with the LB probing, if not for the plates leak. It just simplified the problem too much, so it worked. Otherwise it would have been searching in the infinite feasible set forever. Therefore I think that the original design of the competition was correct, - probing not possible.
2. We corrected 7 HEPG2, 8 RPE, and 45 U2OS predictions thanks to probing and added it all to the training. Note that this is very few. U2OS-04 is one of the hardest, while U2OS-05 is easy, and it is the only U2OS test experiment, so the benefit for it is small. Overall, my estimation is that it gave us private LB boost of 0.001 at the absolute maximum. But it was fun!

Finally, congratulations to the winners and kudos to all kagglers who participated in the kernels and forums discussions!

And I will drop this random but nice image from my notebook here, to remember what it was all about


