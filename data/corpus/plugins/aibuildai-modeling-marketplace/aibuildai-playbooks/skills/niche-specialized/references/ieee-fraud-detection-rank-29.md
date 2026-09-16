# 29th place - Mini write up about automation and groupings

Competition: ieee-fraud-detection
Rank: #29
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111306

Well, that was a blast!

First i want to thank Kaggle of course for setting up the great competition, and&nbsp; my awesome teammate @psilogram ,It's been a real honor working with you, i learned a lot!
Thank you for deciding to work with me.

This has been an incredible competition.
I'll put emphasis on the non model processes I ran in order to support and enhance Silogram's models.&nbsp;There were so many iterations of my solution, I will only mention the last solution.&nbsp;Also - non of the things written here are facts, just my findings personal findings.
The main thing that guided me was Lynn's note :
`The logic of our labeling is define reported chargeback on the card as fraud transaction (isFraud=1) and transactions posterior to it with either user account, email address or billing address directly linked to these attributes as fraud too.&nbsp;&nbsp;`

After some EDA i found that once a customer is fraud once, most of the time all future transactions will also be fraud.&nbsp;My goal was to create groups of cards that if we find 1 of them as fraud, then we can mark them all as fraud, and also to retain the frauds we have in train set and mark these customers as frauds in the future.&nbsp;

We could use grouping by card, date-d1, addr1+2, dist1+2 to identify customers but there are lots of NAs, if only addr1 is different it could be a different customer, and when it's NA, we can't know who it is,&nbsp;So i took a different approach.&nbsp;

I'll start with the groupings that i used, and continue with how i came about finding those, all these were done using an iterative script .Within each of these grouping there are multiple sanity checks to consider for lost information due to missing data. for example - V307 is not&nbsp; simple cumsum of TranscationAmt, it's for last 30 days, if a card has more than 1 consecutive month of data, , sometimes V307 won't be equal to previous V307&nbsp;&nbsp;+ previous Transaction Amount, also if a customer had more than 30 days between transactions, V307 loses its signal and becomes 0.&nbsp;I'm not able to&nbsp; list all the checks and rounds, but i'll say the general guidelines.&nbsp;

**1) Customer\Card:**

Use&nbsp; unique_card1-6 and date-D1, identify next transaction&nbsp; by V307 =&nbsp; CumSum(TransactionAmt)&nbsp; in last 30 days (TransactionDT -&nbsp; 2592000), and either Date-D3 =date of previous transaction&nbsp; or&nbsp; V322= previous V322&nbsp;+1 (when D3 is NA)

**2 ) 1 type of User grouping.&nbsp;**

Group by id_19+id_20+id_31+DeviceInfo, next transaction identified by&nbsp;&nbsp;V264=&nbsp; CumSum(TransactionAmt)&nbsp; in last 30 days (TransactionDT -&nbsp; 2592000), V218= previous transaction V218+ 1, and shares same D1 or D4

**3 ) another type of User grouping.&nbsp;**
Group by id_19+id_20+id_31+DeviceInfo, next transaction identified by&nbsp;&nbsp;V203=&nbsp; CumSum(TransactionAmt)&nbsp; in last 30 days (TransactionDT -&nbsp; 2592000), V168 = previous transaction V168 + 1, and shares same D8


The "User Grouping" ones span across different cards
Some of them are relevant for specific ProductCDs.This allowed me to fill many NAs in the data, in addr, dist, card features.&nbsp;


My output for the ensemble was:
1) a 50 feature. 9492 lgbm that eventually wasn't used. It was craeted before the groupings

2) per each transaction,its card group id, user1 group ID and user2 group ID.&nbsp;&nbsp; Also i grouped together all groups that are related.&nbsp;Is we have group X with 8 transactions, and one of them is also part of group Y, create 1 grouping for all of them (as a new type).&nbsp;
&nbsp;
3) in order to retain customer frauds into the future, i joined all test transactions with all fraud transactions in train, and created features such as "is same addr", "is dist na", does past transaction's group has within it a matching addr", "do their groups share any transaction amounts " etc.&nbsp;I then trained a model to identify the probability that the transaction at hand is of the same customer as any of the frauds we know&nbsp; in the past.


Using these, Silogram was able to use additional features in the ensemble to enable the model to take transaction that it found as fraud, and to mark the transaction's entire group as higher fraud probability.&nbsp;

I'm sure there was a lot more to do, both Silogram and i had breakthroughs on the last day that allowed us to boost our score significantly, but not enough time to explore further.&nbsp;


I think now, after reading the other write ups - i might have took too conservative way to do the groupings, or picked logics that would generate the groups with not as good quality as others did.I did lots and lots of EDA, but probably i got myself too tired with lack of sleep, and sometimes i would just stare at the data while my brain melts.Might have been worth to revisit all of my assumptions from scratch with a new perspective from time to time.


...........&nbsp;
As a side note.&nbsp;I used a power Bi model in the start of competition to allow myself drag and drop and object&nbsp; interactions functionality in a dashboard.&nbsp;It was of great value because on the fly i could drag card, addr, D1, and have a measurement that that shows me all those that have both rows in train and test, have frauds, and that after 1st fraud, there are only frauds onwards. Clicking on 1 of the groups would bring up their transactions.&nbsp;This allows quick and intuitive hands on EDA that you code once, and then analyze on the fly from then on.&nbsp;

I do it in almost every competition and it's an awesome tool for immediate, interactive ,hands on interaction with the data&nbsp;
.........&nbsp;


Thank you all for reading,

Thank you for being such an awesome community.

You're all truly a unique group of people.
