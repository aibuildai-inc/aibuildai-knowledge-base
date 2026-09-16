# Now that the competition is over - what everyone used?

Competition: yelp-recruiting
Rank: #15
Source: https://www.kaggle.com/c/yelp-recruiting/discussion/4980

<p>Congratulation to the anonymous winner and all the others! I’m really greedy to know &amp; learn what everyone used for this competition.. I personally really struggle to use anything which was related with text extraction and similar (NLP) .. I always got worst
 results! My final model end up using a Gradient Boosting on the logarithm of the #votes useful.</p>
<p>I’ve extracted about 20-30 extra features.. these following were pretty important</p>
<ul>
<li>review age ( discovered to be draft age ) </li><li># user votes useful / #reviews ( to get the average of useful votes x review – same x cool &amp; funny)
</li><li># reviews of the business&nbsp; / #check-ins (to get a coefficient to weight # of visits)
</li><li># user reviews / #check-ins (same as above but at user level) </li><li>Difference between user rating &amp; business average rating </li><li>A couple of features from clustering similar business together of various sizes (25 up to 100 clusters)
</li></ul>
<p>Didn’t really manage to get any useful information based on location (either clustering locations), just got my score worst. Same results clustering similar reviews based on bags of words.</p>
<p></p>
<p>I did try quite few things rather tha use just tree ensembles, but having the test set so un-replicable (due to post-date now available) made it a quite hard (and annoying) task.</p>
<p>Basically my model was based on user &amp; business ranking, rather than parse the review and understand if was useful or not! Anyone managed to do this?</p>
