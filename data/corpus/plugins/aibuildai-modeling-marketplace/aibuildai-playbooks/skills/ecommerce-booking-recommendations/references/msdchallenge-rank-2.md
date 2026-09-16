# my solution to the MSD challenge..

Competition: msdchallenge
Rank: #2
Source: https://www.kaggle.com/c/msdchallenge/discussion/2413

<p>Hi everybody,</p>
<p>for the winning solution I basically adopted a item-based collaborative filtering approach with some &quot;crucial&quot; modification:
<br>
<br>
1) invented a new parametric similarity between songs (or users) which lead to 0.16665 on leaderboard</p>
<p>2) final calibration of the scores for ranking (0.17712 on leaderboard)</p>
<p>3) ranking aggregation with a user-similarity based predictor (roughly 0.178 on leaderboard)</p>
<p>As you can see, the first two were crucial for the high scoring!</p>
<p>You can find a quite exaustive description of the method in this paper:<br>
<br>
F. Aiolli, A Preliminary Study on a Recommender System for the Million Songs Dataset Challenge<br>
Preference Learning: Problems and Applications in AI (PL-12), ECAI-12 Workshop, Montpellier<br>
http://www.ke.tu-darmstadt.de/events/PL-12/papers/08-aiolli.pdf<br>
<br>
also available at my web: page http://www.math.unipd.it/~aiolli/paperi.html<br>
<br>
Unfortunately, the calibration step is not fully documented and it is not discussed in the paper above.I am just preparing a new paper which describes the whole method (see the code referred below to have a rough idea of this very simple method).<br>
<br>
I also published (a cleaned version of) the code I used for the winning submissions. It can also be used for validation. Hope it works!! There are three source files:
<br>
1) MSD_util.py, MSD utility functions <br>
2) MSD_rec.py, MSD implementation of the basic classes: Pred (predictor) and Reco (recommender)<br>
3) MSD_subm_rec.py, Example of a script for the computation of user recommendations</p>
<p>The code is not optimized and probably can be made far more efficient. I apologize for the code which is not commented appropriately. I hope it is not too criptic anyway. It might be easier to understand the code if you previously read the paper :)<br>
<br>
I am very busy in this period and not sure I can maintain and correct the code in the future. However, I would appreciate comments and suggestions. Also, I am very courious to hear about other people' solutions..
<br>
<br>
Cheers,<br>
-- Fabio.<br>
<br>
</p>
