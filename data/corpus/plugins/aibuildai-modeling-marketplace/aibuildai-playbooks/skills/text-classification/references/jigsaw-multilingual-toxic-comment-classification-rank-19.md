# 19th place solution

Competition: jigsaw-multilingual-toxic-comment-classification
Rank: #19
Source: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160927

Congratulations to @rafiko1 and @leecming for winning the first place, besides being in the top of the public LB for the longest time, and [Google Jigsaw](https://jigsaw.google.com/) for hosting the third edition of toxicity classification challenge.

We decided to divide our models into 3 parts. Number is listed alongside each model to denote what part they belong to.

## Models:

* XLM\_RoBERTa\_Base with different architectures/epochs/learning_rate/augmentation (I)
* XLM\_RoBERTa\_Large with different architectures/epochs/learning_rate/augmentation (I)
* XLM\_RoBERTa\_Large\_MLM\_training (I)
* Multilingual\_BERT (cased and uncased variants) (II)
* BERT (cased and uncased variants) (II)
* RNN: pooled Bidirectional GRU with fasttext aligned word vector embeddings on translated train (III)
* RNN: pooled Bidirectional LSTM + GRU + attention with fasttext + glove + paragram embedding averaged on translated test (III)
* WeakLearner: NB-SVM on translated train as well as translated test averaged (III)
* WeakLearner: SGD on translated train as well as translated test averaged (III)

Overall, there were a combination of about 50 models with the base models belonging to one type (parent) averaged. So, all varieties of XLM\_RoBERTa\_Base models were averaged and so on.

**First-Level Blending procedure:**

&gt; (I) -&gt; (XLM\_RoBERTa\_Base * 0.1)  + (XLM\_RoBERTa\_Large * 0.5) + (XLM_RoBERTa\_Large\_MLM\_training * 0.4)

&gt; (II) -&gt; (Multilingual\_BERT * 0.2) + (BERT  * 0.8)

&gt; (III) -&gt; (RNN * 0.6) + (WeakLearner * 0.4)

**Final Blending procedure:**

&gt; `toxic` = (I) * 0.8 + (II) * 0.1 + (III) * 0.1

## Post Processing: (0.0005 boost)

I think this is the major highlight of our approach. Just a day before the competition finish, @veryrobustperson discovered that most of our lower scoring submissions were over-predicting while compared to the higher scoring submissions. Based on this hypothesis, we tried to introduce a re-scaling factor for predictions &gt;0.8 as well as &lt;0.01 through the use of probabilistic random noise that introduces a small penalty. I am sure there was a lot to explore here, but due to shortage of submissions, and lack of time, we couldn't delve deeper into it and optimize further. For two different submissions that we tried, the score increased by about 0.0005, so we thought it would generalize well even on the private LB test data. Luckily, it did.  

## Things that didn't / couldn't make it to work:

* pseudo labelling by making &lt; 0.2 -&gt; 0 and &gt; 0.8 -&gt; 1.
* training a gradient boosting algorithm on the 1024 laser embeddings.
* hard-coding test probabilities to ground truth in validation as we found about 1000 samples overlapping based on cosine similarity of laser embeddings.
* changing the optimal threshold of toxic probabilities in `jigsaw-unintended-bias-train.csv` to a value ranging from 0.2 to 0.3, as 0.5 (default rounding) looked extremely non-toxic centric.
* using external datasets for hatespeech, toxic word list for 6 languages.
* label smoothing. 
* Wordbatch FM_FTRL.
* power averaging of base models.

Lastly, I thank my team-mates @veryrobustperson and @ipythonx for the wonderful discussions we've had and coming up with newer ideas from time to time. We almost made it to the gold zone :)

Hope you all had fun competing! 


cheers,
Nickil
