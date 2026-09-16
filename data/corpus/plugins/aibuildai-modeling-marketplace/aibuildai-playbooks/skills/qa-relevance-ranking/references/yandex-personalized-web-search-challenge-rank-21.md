# Share your approach?

Competition: yandex-personalized-web-search-challenge
Rank: #21
Source: https://www.kaggle.com/c/yandex-personalized-web-search-challenge/discussion/6811

<p>Here is mine:</p>
<p>I tried the Pointwise approach of ranking using association rules for the task. I found the user's past click history and past relevance results for the current query (across all users) to be most useful. I also found that associations with domains to be more useful that URLs.</p>
<p>To validate my models, I extracted 900k sessions from Day 26 of training data, and randomly sampled queries to be test queries, according to the selected criteria described.&nbsp;</p>
<p>My code is written in Python and ran on a 4-core desktop with 24GB of RAM. The actual prediction of test data takes less than 5 mins using pre-built models.</p>
<p>But as my models became more complex, even 24GB RAM wasn't enough. To make the test prediction task more manageable, I extracted the users and queries from the test set and only kept model data for these users and queries in memory.</p>
<p>After the 0.798 level, my progress stalled and everything I added to the model only reduced the score. I am not sure why that was...</p>
<p>Other feature/model ideas I tried:</p>
<ul>
<li><span style="line-height: 1.4">Associations of n-grams individual terms in queries with URLs and domains. Trigrams seemed to add most value.</span></li>
<li><span style="line-height: 1.4">List of top users for each query and using their relevance history.</span></li>
<li><span style="line-height: 1.4">Finding similar users based on past query history. This was slow and memory consuming, so I had to use a lot of shortcuts and tricks.</span></li>
<li><span style="line-height: 1.4">Associations of the queries made in the current session with URLs/domains. Didn't seem to add much value.</span></li>
<li><span style="line-height: 1.4">Associations of URLs clicked on in the current session with the URLs/domains. This seemed to help on my validation data, but performed poorly on the test data.</span></li>
<li><span style="line-height: 1.4">Prior probabilities of domains/URLs didn't seem to help.</span></li>
</ul>
<p>I am curious to know how other people approached this task. As the research suggests, did LambdaMART and other such models work the best?</p>
