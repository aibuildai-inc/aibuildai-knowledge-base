# 1st place solution - with code and official documentation

Competition: trackml-particle-identification
Rank: #1
Source: https://www.kaggle.com/c/trackml-particle-identification/discussion/63249

Hello everyone, thank you for a great competition! This was my first serious Kaggle competition, and I must say I'm impressed with how much fun the competition has been to me. I think the organizers have done a great job in making the scope of competition task large enough to be interesting, while not requiring much background knowledge from the field.

Edit: official documentation and code are now available at:
https://github.com/top-quarks/top-quarks/blob/master/top-quarks_documentation.pdf
https://github.com/top-quarks/top-quarks

I'm sorry, but I did not use much machine learning (only some logistic regression for candidate pruning), but rather classical mathematical modeling with statistics and 3d geometry. This, combined with the fact that I wrote everything in C++ with no dependencies, made the final code quite fast: about 8 minutes per event per cpu core for my final submission. So I believe my code could be a good starting point for the throughput phase.

**Now for my approach**

I divided my algorithm into several steps, and created a scoring metric after each step, so that I could easily tell at which step I could earn the most score. I also made load / score function after each step for rapid debugging and tuning.

There were 48 layers in the detector, each either an annulus or cylinder (approximately). I sorted these approximately so that each track would pass the layers in increasing order. I considered multiple hits of one particle on a single detector to be duplicate measurements, and only looked for a single hit per detector per track until step 4.

**1. Select promising pairs of hits.**

This was done by considering all pairs of hits on 50 pairs of adjacent layers that covered most of the tracks. These candidates were pruned heavily by a logistic regression model of several heuristics. Some of the heuristics were how far the line passing through the two hits passes from the origin, and the angle between the direction between hits and the direction given by the cells data for each of the hits.
This gave about 7 million candidate pairs covering about 99% of the score (meaning for tracks worth 0.99 had at least one pair on that track).

**2. Extend the pairs to triples**

This was done by extending the line passing through a pair, and looking where it hits the next adjacent detector layers using 3d geometry. I set the 10 closest hits to the intersection as triple candidates. Then I did another pass of pruning by logistic regression to get about 12 million candidate triples. In this step we had three points, so we could fit a helix through them, and we even had one degree of freedom left as a feature for the logistic regression. Other features were (the logarithm of) the radius of the helix, and again the deviation from the direction given by the cell data. The triples covered about 97% of the score (meaning for tracks worth 0.99 I had at least one triple on that track). And the remaining tracks were short, crooked (low momentum), and started far from the z axis.

**3. Extend triples to tracks**

We fitted a helix through the three hits, and extended it to the adjacent layers using 3d geometry. I always used the helix fitted by the 3 nearest hits on the track to the layer in question. Also here I added the closest hit to the intersection. The resulting (still about 12 million) tracks now contained about 60 million hits, and about 95% of the score (meaning if we optimally assigned tracks using the ground truth data, added all duplicate hits to each track, and ignored &gt;50% coverage constraints, we could get score 0.95).

**4. Add duplicate hits**

For each track we added the hits closest to it on each layer it passed through. I'm not exactly sure how, but now we covered about 96% of the score :) and I'm not complaining.

**5. Assign hits to tracks**

Until now all tracks had been processed completely separately, so they were massively overlapping. The goal here was to pick the best paths, and resolve any conflicts between them. My algorithm for this step was based on taking the "best" track (I will come back to the metric), removing all hits contained in it from all conflicting paths, and then repeating until there was nothing more to do. This was done efficiently using a data-structure based on a priority queue and dynamic updating of track scores.

The scoring metric to determine the "best" tracks was originally based on a random forest and distance from helixes, but I later found something much better. I didn't manage to model the perturbed helix noise. At least, I didn't feel like I had enough quantitative information to do this properly. This meant modeling the probabilities accurately as needed f.ex in a Kalman filter was infeasible. So instead of modeling the inliers (actual helix track), I modeled the probability of outliers (that we would find this track by chance). This was based on the assumption that we could model outliers by the density of hits on a layer, which I assumed was independent of the angle around the z-axis. This outlier density idea was also used for thresholding in all previous steps, so f.ex. saying "I want 0.1 outlier duplicates on average from each hit" for making the thresholding distance for duplicates.

The full algorithm gave the final score of about 0.92, using about 90% of the hits.

**More important considerations**

Of course there were several very important implementation details, note that the above explanation is a simplification down to the most important parts. A crucial technique considering performance, was that I used an acceleration data-structure to quickly access points to close to the helix intersection with a layer. This data-structure based on quad-trees was highly efficient, supported elliptic queries, and took into consideration imprefectness of the layers (they are not exactly annuluses and cylinders), and used polar coordinates to make the maths tractable. I also made a O(1) lookup for close to analytic outlier probability densities in any elliptic region on a detector. A crude model of the magnetic field strength as function of z position of the detector ( "1.002-z'*3e-2-z'^2*(0.55-0.3*(1-z'^2))", where z' = z/2750) gave a 0.003 score boost. On top of that there were a lot of parameters to tune, which were what gave me the last 0.01, and I'm sure there is more to gain if I had the patience.

**My takeaways from the competition:**

 - Kaggle has some really interesting competitions.
 - Loading bars are really cool! I used them everywhere :)
 - It's fun to submit to the leaderboard, even when it isn't strictly strategical considering winning chances.
 - Computational resources aren't everything. I got access to a supercomputer, but was unable to improve my score by increasing computational load.

Edit: I added @ersol to the team, as he had experience with cloud computing services. However, in practice I didn't need that, so he didn't end up helping me.
