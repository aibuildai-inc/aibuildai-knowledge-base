# Method Sharing

Competition: pkdd-15-taxi-trip-time-prediction-ii
Rank: #2
Source: https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii/discussion/14988#83302

<p>Here goes my approach.</p>
<p><br><strong>preprocessing &amp; feature generation</strong><br>From each training data, I created multiple training examples: each partial trajectory starting from the start location becomes a training example. For each training example, I then calculated the elapsed time until the last position, travel distance, the overall bearing, and the nearest taxi stand from the starting point.</p>
<p><strong>training</strong><br>I used gradient boosting (R gbm package) for training.</p>
<p><strong>local validation</strong><br>For local validation, I created two validation sets. Each consists of 5 cutoff points whose month, day of week, and time within day are same as the test dataset. I was able to pick the 2nd best result using the local validation result.</p>
