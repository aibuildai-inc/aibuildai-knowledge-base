# 4th Place Solution [no LB probing]

Competition: dont-overfit-ii
Rank: #4
Source: https://www.kaggle.com/c/dont-overfit-ii/discussion/91801#latest-532859

A journey to 4th place.

At first I would like to thank you all for I’ve learned a lot! I guess I’m standing now where Zach stood in the first Don’t Overfit contest. This was my first competition, and first serious contact with machine learning at all.

Unfortunately, I couldn’t rely on any metrics. I didn’t manage to get a reliable CV scoring throughout the competition. The only way to know if I was doing any better, was to check a Public LB score, so…
I figured I needed another ranking system, that will indicate me whether I’m going off or not. As Public LB scores was all I could rely on, I developed a simple tool that:
    * loads all previous models I submitted together with Public LB score,
    * loads a new model that I want predict a score for,
    * performs rank and MinMaxScale on all of those models 
    * calculates mse, mae, cos, dist, r2 and cor between a new model and all previous models I’ve submitted, searching for submitted model that was ‚closest’ to the one I’m predicting the score for.

This ranker improved over time, as I added more and more submissions to its database. 
While not very complex, it allowed me to reject non-promising ideas at an early stage, without losing tons of submissions. 


I believe many of you have already seen the kernel [Robust, Lasso, Patches with RFE &amp; GS](https://www.kaggle.com/featureblind/robust-lasso-patches-with-rfe-gs,) I will refer to it as a LassoKernel from now on. I originally posted it with heuristics yielding 0.868 on Public LB. The heuristics were carefully crafted, as touching any of the parameters would result in a significant drop of Public LB score. How did I come to that? When working on that kernel, at first I did not use any random seed and there was one lucky shot of 0.867 LB that I struggled to reproduce. I fed that one lucky shot into my ranking system described above and started a loop with different random seeds in search of a model as close to 0.867 lucky submission as I could get. That is how it’s gotten to this specific random seed that yields 0.868. This was my base model. Mind that it did not take advantage of any LB probing, as training was done on training samples only, but the process of developing it was strongly dependent on Public LB scores of all my previous submissions, so there definitely is a data leak.


Reading discussions and other kernels, I figured that many people favoured LogisticRegression. So I replaced kernels’ Lasso with LogisticRegression and did many experiments to get the best score I could, hoping to beat a result I got from Lasso. Results can be seen in my second kernel [ Robust, Patches with RFE &amp; GS and LogReg](https://www.kaggle.com/featureblind/robust-patches-with-rfe-gs-and-logreg) (LogregKernel).

The next step was to stack those models. Simple mean of submissions from the two above kernels yields a 0.872 Public LB (and mean rank 0.873 if I recall correctly). Each of the kernels is a mean of multiple models as well, so I decided to sacrifice a bunch of submissions to get the LB score of different components and some combinations of means or mean ranks of them. That’s how I got to know that eliminating last component of my Lasso kernel yielded 0.872 LB alone, and averaged with several components from my Logreg kernel, yielded a 0.875 (mean rank of 0.876)

LassoKernel = 0.872
LogregKernel = 0.868
BestSoFar = meanRank((LassoKernel, LogregKernel*), weights=[2,1]) = 0.876

*Precisely it was rank of some components from LogregKernel, but I have a difficulty retrospecting which ones, sorry!

Then there came a few weeks of stagnation, until @nightwolfbrooks kernel [Hyper parameter tuned SVM model](https://www.kaggle.com/nightwolfbrooks/hyper-parameter-tuned-svm-model) (NightwolfsSVMKernel ) came out. I think this kernel deserves much more attention than it seems to have gotten. It is very well crafted, and playing around with it I couldn’t improve it’s score, so I took the submission as is. It turned out that it introduced enough diversity into my BestSoFar, to get to the 0.881 Public LB.

NightwolfsSVMKernel = 0.861
TotalBest = meanRank((BestSoFar, NightwolfsSVMKernel), weights=[2,1]) = 0.881

As I mentioned above, I couldn’t rely on CV at all, so both GridSearchCV and RFECV in my Kernels were not reliable, and the overall performance of my kernels is a combination of pure chance and my determination to advance on Public LB. 
As to the noise introduced in my Lasso kernel - it only affects RFECV in some mysterious way, so that it chooses ‚optimal’ feature set. With noise removed (or with different random_state) other features get selected, and the LB score goes down. With noise introduced only to data fed to feature selector, but not to the GridSearchCV the 0.868 LB score still holds.

I find this a bit disappointing, but I hope you learn at least a bit from my approach.

Thank you all!
