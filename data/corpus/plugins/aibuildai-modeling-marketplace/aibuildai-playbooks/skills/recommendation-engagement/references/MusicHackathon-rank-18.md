# Code/approach sharing

Competition: MusicHackathon
Rank: #18
Source: https://www.kaggle.com/c/MusicHackathon/discussion/2242#12619

<p>First, thanks to organizers,<strong>&nbsp;</strong>it makes a lot of fun to work in such short time line, without need to invest much time in competition!</p>
<p><strong>Feature genearation(very simple)</strong>:</p>
<p>1. Use (artist id, user id) from each train/test entry to get features from words.csv, and user id to get features from users.csv, then join it</p>
<p>2. Find for (artist_id, user_id) set of ratings in train.csv, and use it mean/max/min/median as feature for each train/test entry (removing rating of curent entry)</p>
<p>Make same, but agregating by artist_id, and user_id separately.</p>
<p><strong>Model</strong>: apply r gbm (gradient boosted trees) with some parameter tunning by hand.</p>
<p>Uploaded: main.py (feature genearation, alot of dumb/simple code), train_gbm.r (model building)</p>
