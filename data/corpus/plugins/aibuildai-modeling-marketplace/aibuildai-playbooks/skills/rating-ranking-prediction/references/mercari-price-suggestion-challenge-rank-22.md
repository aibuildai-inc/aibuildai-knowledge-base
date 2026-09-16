# 22nd Around the world solution

Competition: mercari-price-suggestion-challenge
Rank: #22
Source: https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50259

First of all I would like to congratulate the winners who were impressive all along in terms of sharing and scoring in such a constrained environment.

I would like to warmly thank @Kueipo who made this whole experience possible. He is very talented, with lots of energy. His knowledge and determination have been so precious to the team. He deserves the Kaggle master grade he just obtain with his 2nd silver medal.

Our solution is based on the following ;

1. Hashing the text data and not using any vocabulary based vectorizer. 
Dimension is 2 **24. The hash matrix is the addition of :

  - name ngram 1, 2
  - mix of category + name ngram 1, 2 &lt;= this is huge 
  - description (prefixed to seperate from words in name)
  - brand (prefixed to seperate from words in name)

2. Compute 5-fold OOF predictions from Ridge and LinearSVR

3. Train 2 LGBMs with different depths and regularization
Using name hashing vectorizer, a few numerical features and the above mentioned OOF predictions

The final prediction is a weighted merge of the LightGBM outputs.

To make it possible within a 1 hour time frame we extensively used the multiprocessing library with numerous workers, managers and queues. Multi threading package was used to fit Ridge and LinearSVR in parallel.

Eventhough team members were from UTC-6 to UTC+8 I think we managed to get everybody involved and most of us brought something to the mix. In particular I had a great pleasure working with Yifan, Mark and Rand. 

I let you guys add your own feeling on the competition and team work.
