# 10th solution and code

Competition: ieee-fraud-detection
Rank: #10
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/113159

Thanks my teammates for their efforts. 

### **Feature Engineering**

Find the openCardDay(day-D1) by EDA(card = card\_cols+addr\_cols+email\_cols, day=TransactionDT//86400):

There are different day-D1 between isFraud=0 and isFraud=1

- Encoding different uids by other columns---useful in old uids

- Count the number of transactions and types of product by uid in a period of time---useful in new uids

### **How to infer there is no shake up in this competition**

Although the number of old uid in test dataset decreased with time. But old uid with fraud=1 largely unchanged.



### **How to prevent overfitting**

Because the major improvement is old uid encoding. So more types of uid encoding, higher local cv. But not every uid in train dataset also in test dataset. The trick to prevent overfitting is setting uid which only in train and count &lt; k to NaN. Then you can add more types of uid and encoding.
My three models:



submission:[https://www.kaggle.com/daishu/10th-submission-part-daishu](https://www.kaggle.com/daishu/10th-submission-part-daishu)
code:[https://github.com/jxzly/Kaggle-IEEE-CIS-Fraud-Detection-2019](https://github.com/jxzly/Kaggle-IEEE-CIS-Fraud-Detection-2019)
