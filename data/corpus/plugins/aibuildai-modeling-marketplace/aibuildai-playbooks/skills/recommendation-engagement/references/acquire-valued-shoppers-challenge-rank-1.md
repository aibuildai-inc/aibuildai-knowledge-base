# We will post the winning solution here

Competition: acquire-valued-shoppers-challenge
Rank: #1
Source: https://www.kaggle.com/c/acquire-valued-shoppers-challenge/discussion/9756

<p>Hi everyone!</p>
<p>What a great feeling to finally get the 1st prize (as well as the Master's status that was our target in this one).</p>
<p>I would like to thank kaggle for the wonderful competitions, and my team mate Gert for his valiant effort and wit! Also congrats to the rest of the teams that played fairly (and thank you for not beating us as there was nothing to improve for quite some time!)</p>
<p>We will wait a couple of days before posting our solution in order to contact with Kaggle first (forgive us, it is the first time!)</p>
<p>Generally speaking, what was really important in this one was to find a way to cross validate(1st problem!) and retain features (or interactions of them ) and then again there was the big difference between the offers in the training and test set (2nd problem!).</p>
<p>For the first one we generally used a 1-vs-rest offers' approach to test the AUC&nbsp;and sometimes even derivatives of that. For the second (problem) we tried to maximize the with-in offers' auc (how well the offers score individually irrespective of the rest) and the total AUC&nbsp;(e.g. how the different offers blend together) as separate objectives.</p>
<p>We used 3 (conceptually) different approaches (and some other minor blends):</p>
<p>1. Train with similar offers <br>2. Train with whether the customer would have bought the product anyway<br>3. Assume that some features work for all offers in the same way (like: if you bought the product before, that increases the probability of becoming/staying a repeater)</p>
<p>More coming soon...</p>
