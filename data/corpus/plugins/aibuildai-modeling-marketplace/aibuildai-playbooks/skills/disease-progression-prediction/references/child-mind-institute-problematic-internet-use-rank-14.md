# 14th place solution

Competition: child-mind-institute-problematic-internet-use
Rank: #14
Source: https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552517

Hello everyone! I am very happy to have won a gold medal and I would like to share my solution with you:
- This is a single classification model (yes! not a regression model)
- This solution was based on some sophisticated DNN that was the <a href="https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/430843">winning solution</a> of the <a href="https://www.kaggle.com/competitions/icr-identify-age-related-conditions">ICR competition</a>. I loved it when I saw it and I was waiting for a competition where I could apply it.
- Use <a href="https://scikit-learn.org/1.5/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer">IterativeImputer</a> class as a strategy for imputing the missing values,  which models each feature with missing values ​​based on other features and uses that estimate for imputation. I decided to use both the train and test data to perform the fit and trained the model online so that I could take the data from the private test.
- Take the enmo from the actigraphy data differentiating weekdays from weekends and grouping them by day and time slots.
- Train with ALL the data
- Use weights to balance the classes in the loss function
- Take the average of the predictions of the same model with 10 different seeds

Many thanks to the organizers of this competition and to everyone who participated by sharing their ideas and code !

<a href="https://www.kaggle.com/code/lauraromar/14th-place-solution?scriptVersionId=208026436">Here</a> I share the code of my solution.
