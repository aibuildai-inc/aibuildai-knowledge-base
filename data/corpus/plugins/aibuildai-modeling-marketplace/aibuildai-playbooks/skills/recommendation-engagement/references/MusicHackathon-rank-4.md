# Code/approach sharing

Competition: MusicHackathon
Rank: #4
Source: https://www.kaggle.com/c/MusicHackathon/discussion/2242#12821

<p>[quote=Zstats;12734]</p>
<p>Hi, Steffen--</p>
<p>&nbsp;Since you are Dr. LibFM, would you be able to comment on how many factors (-dim parameter) you used and what -init_stdev parameter you used? I'm really intrigued by factorization machines, but I seem to have consistent bad luck when applying them... (Probably
 my choices of parameters are not very smart...) Do you have a particular method for deciding these parameters when you go into a new project, or do you do a formal or informal grid search for them?</p>
<p>[/quote]</p>
<p>&nbsp;</p>
<p>My best submission has dim=1,1,16 (i.e. k=16), and I use MCMC with -init_stdev 0.5. About selecting &quot;k&quot; and &quot;init_stdev&quot;:<br>
You can chose K and init_stdev by any holdout method (e.g. cross-validation). For K you can start with small values, and increase it (e.g. doubling it). I typically chose init_stdev first and keep it fixed as it mostly is quite stable for different K, features,
 etc.<br>
<br>
There are some general remarks about how to tune FM parameters in the article &quot;Factorization Machines with libFM&quot;: http://dl.acm.org/citation.cfm?doid=2168752.2168771 (You will be redirected to a free copy of this article if you follow the download link of
 &quot;Factorization Machines with libFM&quot; on http://cms.uni-konstanz.de/informatik/rendle/pub0/)<br>
<br>
Besides good choices for dim and init_stdev, for sure it is important to put good features/data in the model.</p>
