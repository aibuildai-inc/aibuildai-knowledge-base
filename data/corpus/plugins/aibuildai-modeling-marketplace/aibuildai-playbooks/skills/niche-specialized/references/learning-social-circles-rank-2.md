# Congratulations and Solutions sharing

Competition: learning-social-circles
Rank: #2
Source: https://www.kaggle.com/c/learning-social-circles/discussion/10798#57325

<p>First, many thanks to Julian and Kaggle for organizing this very interesting competition!</p>
<p>Here is a summary of the submission that ranked 2nd on the private LB:</p>

<p>I used a combination of two community detection methods available in python-igraph, namely Infomap and Walkrap, and used the provided features to weight the edges in&nbsp;each network.</p>
<p>The feature selection was based on running an Infomap on the 'feature-weighted' networks, where the weights are calculated using only one feature at each run.<br>I then chose seven of the features that had the best individual performance and run the following algorithm on each network:</p>
<ul>
<li>Use equal weights for the seven features.</li>
<li>Run a Weighted Infomap - keep only communities with at least seven&nbsp;nodes.</li>
<li>Run a Weighted Walktrap - keep only communities with at least seven nodes.</li>
<li>Take the intersection of each of the closest Infomap - Walktrap communities.</li>
</ul>
<p>Finally, if no communities are found after running the above algorithm, I fall back to the PageRank solution.</p>
