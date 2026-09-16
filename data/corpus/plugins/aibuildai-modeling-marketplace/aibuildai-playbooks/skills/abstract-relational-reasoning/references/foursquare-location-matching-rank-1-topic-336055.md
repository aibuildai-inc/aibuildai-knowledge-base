# 1st place solution

Competition: foursquare-location-matching
Rank: #1
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/336055

First of all, thanks to the host for an interesting competition and congratulations to all the winners!

While we learned many things from this competition, it is truly regrettable that there was leakage. It was very sad that we had to spend so much time looking for leaks in the last few days.

## Summary

We divided the generation of candidates and the determination of whether they were the same POI into four separate stages.
- 1st stage : Create candidates
- 2nd stage : Feature engineering and LightGBM
- 3rd stage :  xlm-roberta-large and mdeberta-v3-base and Catboost
- 4th stage :  Post Process and Predicting newly emerged candidates with xlm-roberta-large

Next, since there was the same data in train and test, we tried to link them and deleted the pairs that would have been incorrect.

## Cross validation strategy
- Local evaluation was done with 2 fold cross validation. In the training for submit, we used all of the 2 fold data.

## 1st stage
- For each id, we selected 100 candidates for each of the following two methods
   - Euclidean distance in latitude and longitude
   - Cos similarity of name embedding
         - model : bert-base-multilingual-uncased
         - knn : cuml
   - Generate features for candidates. Here, to save memory, only a few features were generated, such as jaro distance for names and jaro distance for categories.
   -  Using LightGBM, each of the above two patterns was predicted, leaving the top 20 candidates each, for a total of about 40 candidates. To reduce inference time, we used Forestinference.
  - Max IOU:  0.979

## 2nd stage
- We created about 120 features based on the following notebook.
   - https://www.kaggle.com/code/ryotayoshinobu/foursquare-lightgbm-baseline
   - Character similarity features such as Levenshtein distance and Jaro-winkler Distance.
   - Character similarity statistics (maximum, minimum, average) using id as key. 
Also, the ratio of those statistics.
   - Euclidean distance using latitude and longitude
   - Embedding of name with svd for dimensionality reduction
       - model : bert-base-multilingual-uncased
- LightGBM ( Inference : Forestinference)
   - CV : 0.875
- Threshold value : 0.01
   - The number of candidates was reduced to about 10%.

## 3rd stage
- We had suspected overlap between train and test, but had no direct evidence, so we created  train overfit models to increase LB and models to increase CV at the same time. After finding evidence of train and test overlap, we only needed to concentrate on improving the accuracy of the data that existed only in test, so we finally used models that had the highest CV and did not overfit train data(e.g. do not increase the number of epochs, use FGM, etc.)
   - catboost
       - The same features were used as in the 2nd
       - CV : 0.878
   - xlm-roberta-large
       - text : name + categories + address + city + state
       - Combine some features of 2nd ( about 70)
       - 3 epochs
    - mdeberta-v3-base
       - text : name + categories + address + city + state
       - Combine some features of 2nd and new features ( about 90 ) 
           - manhattan distance, harversinie distance, etc 
       - Trained with FGM and EMA
       -  4 epochs
       - CV : 0.907
- Ensemble : 2nd lgb * 0.01 + 3rd catboost * 0.32 + 3rd xlm-roberta-large * 0.29 + 3rd medebeta-v3-base * 0.38
    - CV : 0.911
- Threshold value : 0.5
## 4th stage
- Post procress
     - Compare the matches of two ids and merge the matches of two ids if the common id exceeds 50% from either side.
     - Prediction with xlm-roberta-large for the newly created pair above.
     - Threshold value : 0.02
     - CV : 0.9166

## Merge train (Leak)
- Tie train and test by name,lat,lon
- Add true positive
     - Create all pairs of true positive with POI as key (1)
- Remove false positive
    - Remove Pairs of id's tied with train data (2)
        - It was possible to determine whether the id associated with the train data was false positive or not
    - Remove pairs of id tied with training data and id not tied with training data(3)
- By using LB, we validated the effects of (1) add train-train tp, (2) remove train-train fp, (3) remove train-test fp, which are explained above.
   - We compared LB with submission at stage2.
       - Without merge train : 0.900
       - (1) : 0.943
       - (1) + (2) + (3) : 0.971
- Since we were not sure whether (3) would work for private data, we chose both (1) + (2) and (1) + (2) + (3) for the final submission.

## Private Score / Public Score
- Over fit version
   - mdeberta-v3-base(no dropout) 8epoch + xlm-roberta-large 8epoch
   - without train merge : 0.946 / 0.947
   - with train merge : 0.976 / 0.976
- Non over fit version1
  - as explained above
  - without train merge : 0.941 / 0.941
  - with train merge : 0.977 / 0.978
     - (1) + (2) + (3)
- Non over fit version2
   - Slightly changed model compared to non over fit version1
    - with train merge : 0.966 / 0.966
        - (1) + (2)

## About Leakage
For the past few weeks, we have been suspicious of leak for many reasons:
- There is a large gap between CV and LB. Max IOU is already high, so a gap of this magnitude is unnatural.
- LB score is higher with overfitted models. For example, no dropout or many epochs training. Training with many epochs leads to a worse CV score.
- As the CV increased, the CV/LB correlation disappeared. We had a big improvement of CV by training of NNs with FGM but LB was worse.
- The host explained that "we include a few example instances selected from the test set." and the two of the five sample records exist in training data.

In the few days just before the deadline, we focused on thinking about how to find and utilize leakage. And fortunately, we succeeded in discovering how to take advantage of leakage.

## Inference code
https://www.kaggle.com/code/takoihiraokazu/sub-ex73-74-75-ex104-115-90-101-merge-train3
