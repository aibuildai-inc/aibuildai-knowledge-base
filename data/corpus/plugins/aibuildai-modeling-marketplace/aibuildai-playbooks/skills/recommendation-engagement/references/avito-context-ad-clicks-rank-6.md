# Congratulations!

Competition: avito-context-ad-clicks
Rank: #6
Source: https://www.kaggle.com/c/avito-context-ad-clicks/discussion/15606#87438

Congrats to the winners, and to everyone who participated for helping push the limits on this difficult dataset.  

Even though I'd been using python for the last few comps, I ended up coming back to C++ for this one due to the size of the data and the memory restrictions on my laptop. 

Best submission was a simple average of a custom sparse-input NN using FTRL, with one 10 node hidden layer, and an FFM (using the --on-disk flag) on similar set of features. Both scored about the same on the public LB, and the blend only improved by 0.0001

In terms of features, I ended up with 7 numeric, and 43 categorical (each hashed into a100k bucket), some of which were 2 and 3 way interactions. As others mentioned, the count of each type of Ad in the search instance was useful (premise being e.g. you might be more visually drawn to the highlighted ads when they are there), along with hash between them and their sum. Some numeric features included log(1+query_length), log(1+title_length), log(1+HistCTR*10000 ), number of times this user has seen this ad before, plus whether this user has already clicked on this ad before. 
Also, since the error rate was so different between previously seen users, and new users, I also found it helpful to include previous search count by user and/or previous context ad impressions by user (hoping the algo will bias accordingly)

A subset of the numeric features were also included as categorical features on instances when the count of the specific value exceeded 100)

The most frustrating aspect for me was how slow it was to iterate and try new ideas, so a lot of my time was spent re-engineering parts of the code to make the iteration time faster (e.g. using custom bitfields for the records to save as many bits of memory as possible; sorting by UserID+Time prior to feature construction to minimize cache misses and disk fetches; parsing then persisting data in binary format post-processing at various stages to save time re-parsing; etc. if anyone is interested in more details, hit me up)
