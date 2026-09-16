# 31th Place Solution | Using NLP for tabular data!

Competition: playground-series-s3e13
Rank: #31
Source: https://www.kaggle.com/c/playground-series-s3e13/discussion/405616

Hello everyone,

Please take a look at my notebook on Kaggle: https://www.kaggle.com/code/tolgayan/language-modeling-for-tabular-data

I used a unique approach to process tabular data by converting rows into text format. For instance, a row with 1s for sudden_fever, headache, nose_bleed columns would become "a person with the symptoms sudden fever, headache, nose bleed." I trained a BERT model using the emilyalsentzer/Bio_ClinicalBERT backbone model, which is well-suited for medical data.

A few observations:

- No preprocessing was done, just raw data, demonstrating the potential of the BERT approach.
- My best public LB score is 0.40728, with 0.39072 being the best in the Kaggle notebook due to random states.
- Due to the small size of the data, NLP models are prone to overfitting.
- Synthetic data created using external data could be beneficial.
- The base scores are promising, making it suitable for an ensemble architecture.

I am also curious about training a binary sentence-pair classification model where the first sentence is the row data in text format and the second sentence is the disease name. 

Please share your opinions about it. I am curious to see what could be done with this approach.

Thank you!
