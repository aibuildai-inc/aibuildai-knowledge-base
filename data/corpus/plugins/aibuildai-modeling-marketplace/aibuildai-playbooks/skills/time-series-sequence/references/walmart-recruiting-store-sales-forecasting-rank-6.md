# Thank You and #2 rank model

Competition: walmart-recruiting-store-sales-forecasting
Rank: #6
Source: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8023#43949

<p>Congratulations to David Thaler, and thank you&nbsp;for sharing infos.</p>
<p>A big thanks to the Walmart and Kaggle staff who&nbsp;organized this competition!</p>
<p>---</p>
<p>I ended up 6th, using R, and working with a four dimension matrix:</p>
<p style="padding-left: 30px"><em>Sales[store, dept, year, week]</em></p>
<p>which was very handy to make all kinds of charts crossing dimensions.</p>
<p>I switched to days to be able to better synchronize the data.</p>
<p>The following chart of total aggregated sales per day and weighted errors gave me a lot of ideas:</p>
<p>[Aggregated sales per day]</p>
<p>I got most of my &quot;juice&quot; out of the following:</p>
<ul>
<li>using previous year sales for the same week (or day when using days), as the base</li>
<li>time synchronization / sales growth adjustment&nbsp;for ThanksGiving, Xmas, SuperBowl, Easter</li>
<li>total sales growth adjustment</li>
<li>sales growth adjustment per Store, and per Dept</li>
</ul>
<p>I used the leaderboard results to make most of the adjustments.</p>
<p>I plotted more than a hundred charts in total.</p>
<p>I spent 10 days trying to get something out of the markdown data; nothing came out of it, but I still wonder if it could be done by first optimizing [time sync + total growth + store growth + dept growth + (temperature/unemployment/etc)], and only then looking for markdown impacts.</p>
