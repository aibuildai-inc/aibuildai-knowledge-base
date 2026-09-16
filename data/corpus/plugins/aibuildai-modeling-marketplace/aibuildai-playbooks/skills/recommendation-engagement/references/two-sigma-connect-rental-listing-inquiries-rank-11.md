# My Approach

Competition: two-sigma-connect-rental-listing-inquiries
Rank: #11
Source: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32116

There has not been much sharing yet, so I thought I should share my approach for this competition . Before that I would like to congratulate the people who finished within the top 10 and specifically plantsgo, Faron and Little boat for a strong finish!  Congrats to the organizers too, this was an engaging competition with lots of activity on the forums and some good notebook sharing and techniques displayed. 

Also, regarding the people who were hurt of me publishing the leak, I should mention that If I had not revealed the leak (and people had not found it) , I would have finished probably **#2nd**  so I was hurt in a similar way, - but it felt the right thing to do at that point .

About the features .On top of what everybody else has already said/shared ... 

Duplicates
----------

You will notice **that the data has A LOT of duplicate records** that differed only by price (and maybe features ) and were very close in time . So manager, building, dis address, street ,description , lat, lon - ALL the same. My speculation is that there was a bidding process where people kept changing the prices/features to make the properties more competitive (possibly appear on the top) . Some of my stronger features came from around this area. I found these groups of same values . I created different notions of groups . For example one with manager_building_address, another with manager_latitude_longitude and so on. I  sorted by time and **created leads-and lags of price, len(features), len(description) and difference in time**. I also created aggregated features and ranks within these groups ,like which was the **highest price, average price, rank, of each listing**. 

See an example:

![enter image description here][1]

missing value inference ...and corrections
-----------------------

Some properties have missing building_ids but you can easily infer it from the rest of the values (lat, lon, street address, display address) . I created a loop that would search for closeness (if not equality) of lan lon and addresses to fill in the gaps for most of the empty building ids. This gave a small boost onto my score. 

Also with the same logic there were building ids that were **wrongly** assigned a different building id as everything was the same apart from the building id! I found these and assigned them into the same groups.  I also did lots of work with address like replace ("street" with "st" , "st." with "st ", "west,east,north,south" to "w,e,n,s" and multiple such transformations to ensure same addresses are being regarded as the same. 


Binning of numerical variables 
----------

I grouped most of my numerical features into several categories based on equal population and made different models with one-hot encoding,likelihoods to counts on top of these features.

Interactions
-------------

Built up to 3-way interactions of all features binned - or not and created either one hot encoded, likelihoods or counts on these features

Features selection 
-------------

Mostly forwards and backward methods with cross validation. 

models
------

I generated around 100 models , but most of them automatically (with different transformation of the data)   and a couple of manual ones after the leak. Mostly Gbms and nns.  20% of these models came from the StackNet package, the rest were Xgboosts, Keras models, some lightgbms (although I discovered that late - very fast - very efficient), scikit-learn's rfs and ets. 

My best single model before the leak was 0.521x  (xgboost) on public LB and 0.507 after the leak. 

Stacking on the outputs of these 100 models on  equal weight of a gbm, nn and a linear model gave me my final score of 0.497X. I found later that the linear model over-fitted (quite a bit) .

You can find my submission attached to see what you get if you ensemble it :). 

I hope you find this useful and enjoyed playing with StackNet  . I take notes of all your concerns and bugs and will fix them , plus I will make certain the tool gets better and better over time :)

This was a nice competition (apart from the leak)  , with lots of sharing and much stuff to learn.   it would be nice to know what the top competitors have done too - so please share :) 


  [1]: https://kaggle2.blob.core.windows.net/forum-message-attachments/178000/6400/lead_and_lag.jpg
