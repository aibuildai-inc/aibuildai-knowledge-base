# How did you do it?

Competition: acm-sf-chapter-hackathon-big
Rank: #6
Source: https://www.kaggle.com/c/acm-sf-chapter-hackathon-big/discussion/2778

<p><a href="http://www.kaggle.com/c/acm-sf-chapter-hackathon-small/forums/t/2777/how-did-you-do-it/14966">Same question as in the small dataset forum: How did you do it?</a></p>
<p>I basically used the same approach as for the small dataset, except:</p>
<ul>
<li>there was no category-specific normalization </li><li>kNN computed from similar queries within the same category </li><li>if there were less then 5 items from the kNN, extend the list with most popular items by category, most popular (global) items by query, and globally most popular items
</li></ul>
<p>My last submission was 0.57142.</p>
<p>Instead of cross-validation, I used a 10% split for validation, and sometimes a smaller subset for development. Overfitting was no problem, and the differences between validation results were good predictions for performance on the leaderboard.</p>
<p>The Python script for the final submission took 21.5 minutes to run on my laptop, so no real need for a cluster or cloud computing to tackle this problem ... well of course experiments would still run faster if you had several machines at your fingertips
 ...</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
