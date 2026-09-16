# Method sharing

Competition: pkdd-15-predict-taxi-service-trajectory-i
Rank: #3
Source: https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/discussion/15020#83298

<p>Here is my approach. I ended up with 3rd place on the private leaderboard.</p>
<p><br><strong>preprocessing</strong><br>I removed all training examples starting outside the box of latitude 41 to 42 and longitude -7 to -8.8.<br>I also calculated the nearest taxi stand from the starting position for each trip. Some trips consist of multiple legs, and the taxi doesn't move for a while between the legs. In such cases, I considered the last leg's starting point as the starting point.</p>
<p><strong>general strategy</strong><br>For each trip in the test set, I looked for similar trajectories in the training set and averaged the destination of those trips.</p>
<p><strong>similarity of the trips</strong><br>I counted all trips that<br>- start from locations with same nearest taxi stand,<br>- start within cutoff time - 3 hrs ~ + 1 hr (without considering month, week, day of week),<br>- pass within 150m of the last position of the test example<br>as similar trips.</p>
<p><strong>averaging</strong><br>I calculated the geometric median of the destination locations of the similar trips using Weiszfeld algorithm.</p>
<p><strong>postprocessing</strong><br>For trips whose duration is longer than 3,300 seconds, I set the destination as the starting location. It seems that for long trips, drivers often turn off the taximeter after the roundtrip.</p>
<p><strong>local validation</strong><br>For local validation, I created two validation sets. Each consists of 5 cutoff points whose month, day of week, and time within day are same as the test dataset. I was able to pick the best private leaderboard result for submission using the local validation result.</p>
