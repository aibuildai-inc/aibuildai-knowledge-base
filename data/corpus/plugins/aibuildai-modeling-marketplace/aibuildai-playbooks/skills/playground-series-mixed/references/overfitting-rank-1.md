# Lessons Learned

Competition: overfitting
Rank: #1
Source: https://www.kaggle.com/c/overfitting/discussion/601

<p>Hi everyone. Thanks Phil for holding such an interesting competition. Congratulations to all the winners who didn't overfit! It was the best contest I ever participated. I think everyone who participated in this competition has won. Thanks competitors who
 shared their methods in the forum.</p>
<p>I learned some lessons in this competition:</p>
<ol>
<li>I understand that the prize is not the only thing which competitors try for. </li><li>For the first time I used R. &nbsp; </li><li>I got more familiar with NN, SVM, LR, SVM-RFE (I used them as black boxes before)
</li><li>It was also a great feature selection challenge and most of the feature selection algorithms failed in case of overfitting . there is not a good general feature selection algorithm which can tackle such problems.
</li><li>I shouldn't trust cross validation on train set to avoid overfitting, for example I used an RFE method for feature selection and got an AUC of 0.997-0.999 on train set using 5-fold CV in several runs with different random seeds, but these variables didn't
 have AUC better than 0.9 on test set. </li><li>Tim showed us that sometimes with low number of submissions we can perform well if we think better.
</li><li>Good competitions teaches hidden knowledge which you can't even learn in college or read in papers.&nbsp; &nbsp;
</li></ol>
