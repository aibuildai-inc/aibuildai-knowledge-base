# 4th place solution - what's your best single model?

Competition: recognizing-faces-in-the-wild
Rank: #4
Source: https://www.kaggle.com/c/recognizing-faces-in-the-wild/discussion/104288#latest-603422

I used an excellent VGGFace Baseline kernel https://www.kaggle.com/hsinwenchang/vggface-baseline-197x197 by @hsinwenchang based on the code from @CVxTz with some improvements, which I will describe below:
- I discovered that concatenating (x1-x2)^2, x1^2-x2^2, and x1*x2 gives the best results.
- I did not setup an independent validation due to limited time I could devote to this competition so to avoid overfitting to the LB I averaged together 10 runs for each model, using different validation family (F00 ... F09) for each run. 
- Contrary to the discussions I saw on this forum I got best results for the original image size of 224x224 and not for 197x197.
- My best single model (averaged over 10 runs each having different validation family) gave LB score of  &gt;0.91 (it turns out that I did not make a separate submission for the best single model so I do not know exactly how good it was), where adding x1*x2 to (x1-x2)^2, x1^2-x2^2 gave me a boost of ~0.01. I wonder if anyone was able to get a better single model score?
- I have also run two additional cases of 197x197 and 251x251 image sizes and for each image size built another model, where I added an extra Dense layer (with the size of either 500 or 2000) before the last Dense layer, which did not really improve my best single model, but helped to improve the ensemble.
- Averaging of the 2 models with and without the extra Dense layer for the 224x224 image size, gave me the public LB score of 0.914 and private LB of 0.917.
- For the final ensemble, I averaged the 6 models (each averaged over 10 runs) and got a public score of 0.916 and a private score of 0.918.
- It turned out later that adding a model based on 336x336 image size to the ensemble would boost my private score to 0.920, but I did not see any improvement on the public LB when I tried it during the competition (the LB score remained at 0.916) and since the accuracy of the 336x336 model was pretty low, I decided to play it safe and did not use it in the final submission.
- Lastly, I agree with @mattemilio that adding more uncorrelated models, e.g. FaceNet, to the ensemble is likely to further boost the result, but unfortunately I did not have time to do this.

Thanks for sharing ideas and good luck in the future competitions!
