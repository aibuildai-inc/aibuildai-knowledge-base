# Thank You and #2 rank model

Competition: walmart-recruiting-store-sales-forecasting
Rank: #8
Source: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8023#43848

<p><span style="line-height: 1.4">Mine was fairly simple too:</span></p>
<ul>
<li>Gave each week of the year a unique label so holidays lined up.</li>
<li>Created a linear regression model for each separate store/dept combination (so about 3600 simple models)</li>
<li>Used 3 features in regression: avg sales for that store/dept/week combo, Markdown4 (the only one I found to be useful), and sum(31 minus day-of-month) for each day in that week. &nbsp;This last feature was because days at the end of the month tended to have lower sales than at the beginning of the month - so the feature measured the number of &quot;beginning of month&quot; days and &quot;end of month&quot; days a week contained.</li>
<li>I had to make some adjustments for Christmas week because Christmas in the test set fell on Tuesday so that &quot;Christmas sales week&quot; only had 3 shopping days. &nbsp;So that got 3/7 Christmas sales and 4/7 week-after-Christmas sales.</li>
</ul>
