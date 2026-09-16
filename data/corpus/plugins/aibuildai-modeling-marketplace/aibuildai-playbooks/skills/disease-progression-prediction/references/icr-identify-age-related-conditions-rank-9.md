# 9th Place Solution for the "ICR - Identifying Age-Related Conditions" Competition

Competition: icr-identify-age-related-conditions
Rank: #9
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/430906

Hello! This was unexpected, I'm really happy! 


# Context section
- Business context: https://www.kaggle.com/competitions/icr-identify-age-related-conditions
- Data context: https://www.kaggle.com/competitions/icr-identify-age-related-conditions/data

# Overview of the Approach 
I was very concerned about overfitting my model. With so many different opinions in the discussions and the difficulty of accurately gauging technique effectiveness due to the small dataset, I was unsure about the best approach to take. 

As a solution, I decided to integrate a mix of different models along with a variety of data preprocessing techniques.

# Details of the submission

### Part one:
- Data Preprocessing: 
 - Leveraged an over sampler to balance data distribution.
 - Use of Greeks' Epsilon with `Epsilon.max() + 1` for the test set.
 - Employing the SimpleImputer with the strategy set to 'constant'.
- Models:
 - Ensemble of two XGBClassifiers and two TabPFNClassifiers.
 - Employed 5-fold cross-validation, picking the best model.

### Part two:
- Data Preprocessing: 
 - No Greek's Epsilon.
 - Implemented feature scaling.
 - Used feature selection, opting for a subset of 40.
- Models: 
 - XGBClassifier and LGBMClassifier.
 - Employed 15-fold cross-validation, culminating in an ensemble mean.

### Conclusion
I found that the best score was from computing the mean of the first ensemble assigning a weight of 3 to it, while each of the other two models held a weight of 1.

### What did not work for me:
- postprocessing (obviously! 😉).
- Other models such as Tree Classifiers and Neural Networks...
- Using other fields (apart from Epsilon) in Greeks

# Sources:
- I drew inspiration from a publicly shared notebook for the first part:  https://www.kaggle.com/code/aikhmelnytskyy/public-krni-pdi-with-two-additional-models

I'd also like to thank everyone who actively participated in the discussion forum, I believe that this allowed me to learn a lot!
