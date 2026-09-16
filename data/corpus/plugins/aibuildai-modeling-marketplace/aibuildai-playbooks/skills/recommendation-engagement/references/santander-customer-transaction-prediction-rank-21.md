# # 21 Magics

Competition: santander-customer-transaction-prediction
Rank: #21
Source: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88901#latest-512874

Thanks to everybody participated in this competition, it was really exciting. 
I want to briefly describe my magics.

1) Calculated for each column - is value duplicated? Then I calculated number of duplicated for each row and noticed 100k rows in test with 200duplicated. It was before "fake" kernel, so I thought it was private part of dataset. This was weird and I paid more attention to counts. I calculated "if value calculated based on TRAIN data" and then extrapolated this for test. This was boost for .901
2) Then I tried multiply raw values on 0/1 duplicated and filling 0 with column mean. This was about .907
3) 922+ appeared when I started calculating statistics on 300k rows. So there were following parts: raw columns, 0/1 duplicates, value_counts, MinMaxScaler(value_counts)*raw + MMS(vc)*(1-row_mean)
Actually MinMaxScaler(value_counts)*raw + MMS(vc)*(1-row_mean) was a big boost. It somehow shifts values to center and transforms the data. The hist looks like following (in attachments)
...this all was followed by TONNS of unsuccessful experiments...

Boost to 923-924 was in blending lgb+ctb and augmentating train part x2 and x3.
Maybe I'll write my story in more details and publish gist but not now - I need some sleep.
