# Solution to this competition

Competition: acm-sf-chapter-hackathon-big
Rank: #2
Source: https://www.kaggle.com/c/acm-sf-chapter-hackathon-big/discussion/2861

<p><span>Congrats to everyone in this competition. I notice that the rule&nbsp;<span>states that:</span></span></p>
<h3>WINNING SOLUTIONS MUST BE POSTED TO THE FORUMS</h3>
<p>And some teams have present their solutions in small version of this competition in foruma. I here present our solution to the big version of this competition in this topic.</p>
<p>We use naive bayes as our algorithm in this competition. The features we used is:</p>
<ol>
<li>query </li><li>time </li><li>user </li></ol>
<p><strong>A.</strong> We want to know the probability 𝑝(𝑖│𝑐) that user click sku 𝑖 in context 𝑐. We use naive bayes to predict this probability. Then&nbsp;select 5 item with highest predicted probability as prediction in context 𝑐. Here&nbsp;context 𝑐 is query&nbsp;&nbsp;𝑞.</p>
<p>The first context we used is query. By naive bayes, we have:</p>
<div style="padding-left:30px">𝑝(𝑖│𝑐)∝𝑝(𝑖)×∏𝑝(𝑤_𝑘 |𝑖)&nbsp;</div>
<div>here&nbsp;𝑝(𝑖) is prior and we use the frequency that item&nbsp;𝑖 appears in its category as prior.</div>
<div>∏𝑝(𝑤_𝑘 |𝑖)&nbsp;is the likelihood and&nbsp;𝑝(𝑤_𝑘 |𝑖) is the probability that word&nbsp;𝑤_𝑘 apprears when we see item&nbsp;𝑖.</div>
<div><strong>B.</strong> We use time information in our model. We d<span>ivided data into 12-day time periods based on click_time. Then used a smoothed frequency of items&nbsp;𝑖 appears in its category in its time period as prior in A.</span></div>
<div><span><strong>C.</strong> We also use a bigram model to improve naive bayes. The ∏𝑝(𝑤_𝑘 |𝑖)&nbsp;&nbsp;likelihood in A use words conditional probability. We generate a bigram model and use naive bayes model to fit it too. We generate bigram data for each query&nbsp;𝑞
 as follows:</span></div>
<div>
<ol>
<li>suppose use query “xbox call of duty” </li><li>rerank to “call duty xbox&quot; by&nbsp;alphabetic order ( with elimination of &quot;of&quot; as stopwords)
</li><li>bigram: [”call duty”, ”call xbox&quot;, “duty xbox“] </li><li>use naive bayes to fit it as the same as above. </li></ol>
</div>
<div>Then we use a linear combination to ensemble the unigram prediction and bigram prediction.</div>
<div><strong>D.</strong>&nbsp;We did some data processing. The first is query cleanning. In big version of this competition, we did 1,lemmatization;2,split words and numbers such as &quot;xbox360&quot; to &quot;xbox 360&quot;. we also did some words correction in small version of this
 competition.</div>
<div><strong>E. </strong>We rank the items that user have clicked by query&nbsp;&nbsp;𝑞 lower. Suppose we have prediction a,b,c,d,e when he query &nbsp;𝑞 and&nbsp;user &nbsp;<em>i
</em>clicked items a,c. We rerank prediction b,d,e,a,c for user <em>i</em>.</div>
<div><strong>F.</strong> We use python to implement all algorithms. The only 3rd lib need to install is nltk. We use it for&nbsp;lemmatization in data processing. The rest use puer python. The version of python is 2.7, as there is one function not supported in 2.6-
 version. &nbsp;For more information, Check readme.txt in the soucecode.&nbsp;Please download the second attached file. Because I find nothing can delete the first attached file in the system.<br>
&nbsp;</div>
