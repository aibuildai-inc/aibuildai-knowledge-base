# Congrats to the winners!vecorization

Competition: the-seeclickfix-311-challenge
Rank: #25
Source: https://www.kaggle.com/c/the-seeclickfix-311-challenge/discussion/5911

<p>Congrats James for getting the top score, and&nbsp;Tunguska for the win at the event! I'm looking forward to hearing your methods.&nbsp;<br><br>Some notes from my side:<br><br>1. Directly optimizing RMSLE was important in getting a competitive score on the leaderboard.&nbsp;</p>
<p>2. It was very easy to overfit the training data. I hadn't noticed that I was overfitting until the last hour of the competition, but early stopping seemed to be useful.&nbsp;<br><br>I'm curious what kind of features people used, personally I used:<br>- Individual TFIDF vectorization for summary and description text<br>- 1 / (1 + days from first 311 issue)<br>- One hot encoded information for tags, and source<br>- Binary indicator for each of the four regions from latitude and longitude<br><br>I used a linear model for the entire competition. But I suspect deep learning could be very powerful (although slow)<br><br>Looking forward to reading your insights.&nbsp;<br><br>EDIT:&nbsp;<br><br>Wow... that title got mangled - must have accidentally pressed the middle mouse button before creating the thread. Is there any way to edit the title?</p>
