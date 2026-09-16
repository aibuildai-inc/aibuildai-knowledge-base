# 1st place entry

Competition: challenges-in-representation-learning-the-black-box-learning-challenge
Rank: #1
Source: https://www.kaggle.com/c/challenges-in-representation-learning-the-black-box-learning-challenge/discussion/4717

<p>Earlier, I briefly described the first-place entry in the Blackbox challenge.This is the full description.</p>
<p><strong>My Approach</strong></p>
<p>My first 5 entries into this contest used sparse filtering(see below), feature selection and&nbsp;an svm (or a random forest in one instance). The highest scoring of those got 0.6384&nbsp;on the public board, and ended at 0.6300 on the private leaderboard. That entry
 used the&nbsp;best 60 features from a set of size 160. Throughout the competition, I used between a third&nbsp;and a half of the features developed, most often around 40%. The svm used throughout was libsvm.&nbsp;Most of my entries used a c-svc with a linear kernel. I also
 had some entries with an rbf kernel or nu-svc/linear kernel. For the most part, the linear kernel was fine, and the rbf kernel was not clearly better. That&nbsp;isn't surprising given that with most feature sets, I had a couple of hundred features&nbsp;for the 1000
 labeled examples. For the random forest submission, I used R's randomForest package.</p>
<p>The randomForest package in R can generate variable importance scores. It makes the&nbsp;scores by first training an RF classifier and then taking the out-of-bag data, permuting&nbsp;each field in turn, and reclassifying it. The decrease in OOB accuracy with the&nbsp;field
 permuted is the measure of the variable importance. I used this throughout to rank features by their importance, and select the more important ones. I used the variable importance rankings in a one-pass greedy manner. I experimented with panel-wise selection
 (cutting the first k% and then k% of the rest, etc.),&nbsp;but it didn't do any better, so I dropped it. Initially, I&nbsp;used randomForest just because it was already written and I wanted a quick test of&nbsp;the concept. Later, I tried a couple of wrapper methods, but
 couldn't find anything&nbsp;that worked better. Unfortunately this means that my system has to bounce data&nbsp;back and forth between Octave and R.</p>
<p>My best early feature set gets 0.60/0.59 when used whole on the public/private leaderboards&nbsp;(I didn't submit this during the competition), with a linear kernel svm.&nbsp;It got 0.6384 on the public board during competition after being cut to its best 60 features.&nbsp;I
 just re-submitted it for a 0.6480 using my current method, which trims rows out of the weight&nbsp;matrix rather than columns out of the features, resulting in a little bit different&nbsp;(often better) scaling. In any case, on sparse filter features, on this problem,&nbsp;using
 randomForest for feature selection was worth about 4-5%.</p>
<p>It is possible to get better than 60 with just the basic SF features &#43; svm system(no feature selection). The best result that I can find for a whole feature set&nbsp;is 62.4%. That set is size 320. There is a positive relationship between the number of features
 used and accuracy, but it is not huge.</p>
<p>After the fifth entry, most of what I turned in involved selecting&nbsp;features across different sets and combining them, then training a single classifier&nbsp;on the selected, aggregated features. Except for a few experiments, I did feature&nbsp;selection only within
 sets, and then combined the separately selected features into&nbsp;an aggregate set. The few experiments that I did involving selection across sets&nbsp;were very clearly worse (-2%).</p>
<p>I found that surprising. It seems to mean that having possibly redundant, but high-scoring features is good. It may have to do with regularization. If an identical feature is included twice in a classifier that uses regularization, the solver will give them
 equal weight, and their combined weight&nbsp;will be higher than if only one were present. That could improve performance if the redundant features were good. At this point, this is just speculation, though,&nbsp;I do not know how similar the top features out of different
 sets really are.</p>
<p>Some of these aggregate feature sets got pretty large. The largest set that I turned&nbsp;in had size 400. My other scored sets were size 274, 300 and 366. All of the highest scoring&nbsp;feature sets involved at least a little cherry-picking of 'good' feature sets,
 that is,&nbsp;they were aggregated from sets that I knew to be good when used whole. Also, for a few of them,&nbsp;I chose numbers of components that I thought would work well, rather than just taking, say, 40%.&nbsp;In the challenge code on bitbucket, there is a script
 for generating a bunch of weights and features&nbsp;and other scripts for training a model directly from such sets, selecting to a particular number&nbsp;of features. I have run those scripts a couple of times, and the best result I've found was&nbsp;69.16, so the hand-picking
 of good sets was worth around 1%.</p>
<p><br>
<strong>The high-scoring entry</strong></p>
<p>Although almost all of my submissions were single classifiers, the actual winning&nbsp;entry was a small ensemble of three previous submissions. It got 0.7022 on the&nbsp;private leaderboard. It wasn't my highest scoring entry on the public leaderboard.&nbsp;Each of its
 components was an aggregate of selected features as described above.&nbsp;I combined them with simple voting, using mode(), which returns the lowest value&nbsp;in the case of a tie. That seemed reasonable, since the classes are skewed&nbsp;toward low values. I was never
 committed to using an ensemble at all in this.&nbsp;Two other scored entries using single classifiers got 0.7018 and 0.7016.&nbsp;My highest scoring public leaderboard entries also used single svms for classifiers.</p>
<p><br>
<strong>Sparse Filtering.</strong></p>
<p>Sparse filtering is a relatively new method for unsupervised feature learning, introduced by Ngiam, et al in 2011. The paper on it is here:</p>
<p><a id="x_SparseFilterPaper" href="http://cs.stanford.edu/~jngiam/papers/NgiamKohChenBhaskarNg2011.pdf">Sparse Filtering Paper</a></p>
<p>It computes a set of features in unsupervised learning using an objective that is not reconstruction-based. instead it optimizes for characteristics of good features directly. Specifically, it seeks to ensure that features are about equally active overall,
 and that each example has a few, and only a few, significantly active features.</p>
<p>This was the first time that I have ever used, or even heard of, sparse filtering.&nbsp;Thanks to Foxtrot for pointing the method out. Obviously, the method performed well. Also, I found it to be pretty easy to get up and running. In the future,&nbsp;I will be pretty
 quick to reach for this method on semi-supervised tasks. Having said that, it has a few quirks:</p>
<ol>
<li>Different runs of the algorithm return different feature sets that, in general, have different classification performance. This difference can be significant. For instance, I ran 3, 100-feature sets together on the same data, stopping every few hundred
 iterations to train a linear kernel svm with the resulting features. The best one got 54.1 % after 800 iterations, the worst, 49.4%.
</li><li>The sparse filtering objective function is not a good guide to classification performance. Feature sets that have higher classification performance can also have higher loss under the sparse filtering objective.
</li><li>The classification performance of the same feature set can change for the better or for the worse during unsupervised sparse filter training. For instance, a feature set that i used in my first competition entries led to classifiers with an initial maximum
 of near 56% in accuracy (local 10-fold cv) at 800 iterations, then it went down to 55% and stayed there for hundreds of iterations, before ending at 57.6% at 2000 iterations. The next day, I tried training it longer and it headed back down.
</li><li>On the positive sode, feature sets that are good for classification are usually relatively good at any iteration count. That is, if you run a few sets at the same time, one that starts out as a good set will probably be good at higher iterations, too. In
 the test described in (1), the bad set produced a classifier that had 48% at 200 iterations; the good set was at 51% at the same point.
</li></ol>
<p>Some of these are just the usual unsupervised learning caveats. For instance, k-means will return different centroids in different runs of the algorithm. If the centroids are used as the basis for supervised learning features, the resulting classifiers will,
 in general, have different performance. Likewise, unsupervised learning objectives are generally not good guides to downstream performance. The bad point, out of those listed above, is the third. It means that we really don't know when to stop.</p>
<p><br>
<strong>What else I tried.</strong></p>
<p>Initially, I tried running svms and random forests on the raw data, the raw data&nbsp;mean-normalized and sigma-scaled, PCA with whitening, reduced to various numbers&nbsp;of components, and ICA. I used the fastICA package in R for the ICA.&nbsp;I used the training and
 test data to get the PCA basis,the whitening matrix,&nbsp;and to do ICA, but I didn't use any of the bulk data for that.&nbsp;Once when I did use some of the bulk data in a PCA basis, accuracy fell back a bit, which I attributed to a difference in distribution between
 the test and bulk&nbsp;data.&nbsp;With an rbf kernel, the svm with ICA or PCA whitened features topped out at around 47% in local cross-validation, which is probably around 50% on the board,&nbsp;but I didn't submit any of that. The random forest was a little lower. The
 best number of components that I found was around 35 for both.&nbsp;That is surprisingly low, but this data seems to have a lot of pretty strongly correlated features.</p>
<p>Later in the competition, after I had sparse filtering &#43; feature selection &#43; svm working pretty well, I spent most of my time optimizing that. I did try k-means features using some of the Cotes, Lee and Ng code, using triangle activation.&nbsp;They were better
 than PCA whitened or ICA features, but only by a few percent, and&nbsp;by that time, my current approach was in the mid-to-high 60's, so I abandoned it&nbsp;fairly quickly. I tried selecting and aggregating kmeans features, and that helped&nbsp;a bit, but not as much as
 with sparse filter features.&nbsp;I was unable to combine sparse filter features and kmeans features to any good effect.&nbsp;I also tried a voted ensemble of sparse filter features without feature selection,&nbsp;each with its own svm(that is, one SF feature set used whole,
 one svm, repeat).&nbsp;I only worked on that for a bit. It got to a bit over 64% on the board.&nbsp;At one point, I tried stacking sparse filter features, but I didn't get that to work well. Foxtrot says on his blog that he tried two layers and that he didn't find it
 to be any better, either. I also tried a couple of other activation functions. ReLU was bad, quadratic was better, but&nbsp;not as good as the soft absolute value function that is already used, so I didn't use either. Since initialization matters, I tried a few
 other initial distributions:&nbsp;Laplacian, randomly selected datapoints...there may have been one other. None of that helped.</p>
<p><strong>Links</strong></p>
<p>Ngiam's paper code, which I adapted in this challenge, is here:</p>
<p><a id="x_SparseFilteringCode" href="https://github.com/jngiam/sparseFiltering">Ngiam's Sparse Filtering Code</a></p>
<p>My challenge code is on my Bitbucket account here:</p>
<p><a id="x_bitbucket" href="https://bitbucket.org/dthal/blackbox-challenge-code">Challenge Code</a></p>
<p>Foxtrot's blog has some other results about sparse filtering:</p>
<p><a id="x_fastml" href="http://fastml.com/more-on-sparse-filtering-and-the-black-box-competition/">fastml</a></p>
<p></p>
<p>Finally, I'd like to thank Ian and the organizers for putting on a good challenge,&nbsp;Jiquan Ngiam and his coauthors for coming up with this interesting new method,&nbsp;and Foxtrot for bringing it to my attention.</p>
<p>Happy Hacking,</p>
<p>doubleshot</p>
