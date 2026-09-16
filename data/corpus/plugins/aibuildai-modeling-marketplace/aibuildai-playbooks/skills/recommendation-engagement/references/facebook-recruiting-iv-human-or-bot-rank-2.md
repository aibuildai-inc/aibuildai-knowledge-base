# share your secret sauce

Competition: facebook-recruiting-iv-human-or-bot
Rank: #2
Source: https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot/discussion/14628

<p>Here's a little blurb about how I approached feature generation and cross-validation for this problem:</p>
<p>http://small-yellow-duck.github.io/auction.html</p>
<p>code: https://github.com/small-yellow-duck/facebook_auction</p>
<p>I used sklearn's RandomForestClassifier. The most useful features I identified were:</p>
<ul>
<li>the median time between a user's bid and that user's previous bid</li>
<li>the mean number of bids a user made per auction</li>
<li>the entropy for how many bids a user placed on each day of the week</li>
<li>the means of the per-auction URL entropy and IP entropy for each user</li>
<li>the maximum number of bids in a 20 min span</li>
<li>the total number of bids placed by the user</li>
<li>the average number of bids a user placed per referring URL</li>
<li>the number of bids placed by the user on each of the three weekdays in the data</li>
<li>the minimum and median times between a user's bid and the previous bid by another user in the same auction.</li>
<li>the fraction of IPs used by a bidder which were also used by another user which was a bot</li>
</ul>
<p>Cross-validation was a bit tricky: I did CV 100 times on different 80-20 splits. My mean CV score was almost the same (+/- 0.003) as my final leaderboard score.</p>

<p>edit to answer some further questions:</p>
<p>In order to convert the time scale to hours, I made a bids/unit time histogram and looked at the periodicity - bidding volumes cycled with a period of one day. There are pretty graphs and more details on my github page.</p>
<p>A concrete example of the entropy is the IP entropy: N!/(N_{IP1}! N{IP2}!... N{IPn}!). N is the total number of bids and N_{IPn} is the total number of bids placed from the nth IP. (Because the entropy is a large number, I used the log of the entropy.) The entropy is a measure of how both randomly distributed the bids are, how many IPs there are and how many bids there are. A bidder that places all their bids from the same IP has an entropy of N!/N! = 1.</p>
<p>I used the simple average of five RandomForestClassifiers (each with a different value for the initialization parameter, random_state) to generate the predictions.</p>

<p>Thanks to everyone else who shared their approaches! I thought that leaving out the bots with only one bid was a clever idea. And I was surprised to see that people got mileage out of the device type.</p>
