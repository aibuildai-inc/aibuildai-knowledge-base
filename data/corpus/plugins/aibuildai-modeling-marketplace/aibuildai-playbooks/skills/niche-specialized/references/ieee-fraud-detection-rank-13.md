# 13th place solution

Competition: ieee-fraud-detection
Rank: #13
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111485

Thank you for all participants, Kaggle Admin, IEEE-CIS and Vesta corporation who realized this exciting competition. I would like to briefly share my 13th solution, which gave me my first solo gold medal.  

## 1. Validation Scheme
First I used 5 fold CV without shuffle. Then in later stage of the competition, I switched to simple time-split and used about last 100k records as val set. This is because 5 fold CV takes too much time. In addition to that, some of my important features generated based on time-split and 5 fold CV might have caused leakage.  

## 2. Reconstructing user\_id
When I closely examined df[df.isFraud==1], I found that there exists many clusters of very similar records. Because I experienced similar case in Home Credit Default Risk Competition, I could quickly understand what it means and what I should do. I generated wide variety of user\_ids from D features and card information by brute force attack approach. They covered wide spectrum from strict and deterministic one to a little ambiguous but robust to changing in physical address/e-mail address. This combination boosted my score greatly. Only strict version of user\_ids gave me ~0.9620 in PublicLB, but both of them resulted in &gt;0.9640.  

## 3. Feature Engineerings
I conducted various feature engineerings by using user\_ids. Out of lots of generated features, Seq2Dec and TargetEncoding on past records showed significant improvement in both of CV and LB. I think Seq2Dec is good method to encode individual transactions because it can prevent loss of order information and it can handle series with variable length. In addition, I added typical features such as count, share, length of the series of the user\_id, sinse\_first\_obs\_user\_id, to\_last\_obs\_user\_id etc.


Fig. Schematic illustration of Seq2Dec

## 4. Modeling and Blending
I mainly used LightGBM and added Regularized Greedy Forest as small portion of flavor. I added model diversity by seed averaging and bagging models with different n_folds for Seq2Dec/TargetEncoding.

## 5. Result
13th out of 6381 teams, Public: 0.965086, Private: 0.941338
