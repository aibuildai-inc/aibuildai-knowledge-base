# 24th Place Solution - Vesuvius Challenge

Competition: vesuvius-challenge-surface-detection
Rank: #24
Source: https://www.kaggle.com/c/vesuvius-challenge-surface-detection/writeups/24th-place-solution

Thank you to the organisers and other participants for providing such a challenge. I could not participate in previous Vesuvius Challenge competitions due to other commitments so it was amazing to finally have that change and learn so much about it, as well as 3D data in general. 


**A single model solution**

Before the test data got corrected, my strategy was to focus on precision rather than recall as I could not figure out how to make the topo score improve on LB.
One thing that helped a little to experiment faster was to use the weights from the nnUNet trained on the "old" dataset as pretrained weights to all my models (though it was a bit of a headache when dealing with folds and ensuring there is no data leakage). I had a nnUNet with Tversky loss (alpha=0.7, beta=0.3 to focus on precision) and 160x160x160 patches.
After the test set correction, I had to retrain my model using the Skeleton-Recall loss and it gave a +0.004 boost. It took more time than I expected as the threshold and the outputs were so different that it then behaved differently with the post-processing functions I had designed.
All models were first trained on a few folds to better assess hyperparameters and number of epochs, before training the final model on the full dataset.



**Post-processing**

I spent a lot of time designing methods that would remove incomplete sheets. These incomplete sheets were often the verso of a sheet, so there was no interest in keeping them. I removed any connected components below 5000 pixels unless the connected components where close to the edges or corners. I analysed the training data and it appeared that some sheets could be as small as 300-400 pixels when located in a corner and I defined threshold based on this.
I developed a method to detect whether a sheet was small due to it being in a corner (by touching both sides of a corner) or just incomplete. I also checked the main direction of the sheets to then remove incomplete sheets by looking at the size of their bounding boxes and if it goes to whole length of the image.
I used the hysteresis post-processing function shared on a public notebook and changed the z-radius from 3 to 2 and the xy-radius from 2 to 1.

| Methods | Performance Boost |
| --- | --- |
| Advanced object removal | +0.002 |
| Incomplete sheet filtering | +0.001 |
| Hysteresis radius 2-to-1 | +0.004 |






**What did not work (for me)**


- I tried ensembling models relatively early and it did not seem to work well. Though it seems to give a boost when looking at my private leaderboard scores and other solutions. Probably my biggest regret there as I had several models to ensemble at the end!
- To my surprise, lower step size with the nnUNet did not work and seemed to detect more verso sheets.
- Apply move conservative threshold close to areas where 2 sheets merge. The merges were detected by detecting bifurcations in 2D (on all 3 axes).
- Vary the thresholds based on the Betti numbers, I only started playing with this idea in the last 2-3 days.
- Hole filling in individual connected components or with 2D approaches.
- Distance transform thinning.
- 3D cascade nnUNet: I had actually trained a 3D cascade model but simply did not have time to create the inference pipeline, also because it was trained with my high-precision approach that did not work after the test set was fixed and the topo score was really taken into account.


**Conclusion**


It was a difficult competition due to the many changes and my teammate, Jirka, stopped contributing soon after the first dataset change. This was extremely unfortunate as I believe his notebooks had a great impact on the competition and I really wanted to work on this competition as a team. I am overall extremely happy with my final result and to see the wide variety in post-processing solutions and tricks used by everyone. Thank you for these crazy 3 (and a bonus half) months!
