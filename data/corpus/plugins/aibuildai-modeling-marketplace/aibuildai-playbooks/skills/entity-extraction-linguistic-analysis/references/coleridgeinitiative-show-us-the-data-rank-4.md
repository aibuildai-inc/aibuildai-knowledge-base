# 4th place solution - LB probing, acronym detection, and NER

Competition: coleridgeinitiative-show-us-the-data
Rank: #4
Source: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/251457

Our solution is composed of 6 parts below.
- LB probing
- Acronym detection
- Acronym detection version 2
- String-matching with dataset-names from external data
- Dataset-name variation detection using NER
- String-matching with dataset-names from the train data

# 1. LB probing
The metric of this competition is F-score. Under this metric, assuming a current score is ***F***, a newly detected label can improve the score when the expected value that the label is true positive is greater than 0.8F. Therefore, it is important to estimate the private test score in order to determine the detection threshold. For example, if ***F*** is 0.6, the best threshold is 0.48 and if ***F*** is 0.4, the best threshold is 0.32. For this reason, it is very important to know the number of the training-data labels in the private test data, because it affects the private test score strongly.  
To tackle this problem, we did LB probing. In this competition, the public test data contains duplicates of the train data. Therefore, we can create a submission only with true positive labels and with no false positive labels by applying true positive labels of the train data to their duplicates. Thus, by setting the number of true positive labels to a value related to the hidden test data, we can get information about the hidden test data from the submission score. Using this strategy, we got rough estimates of values below (actual codes are [this](https://www.kaggle.com/osciiart/lb-probing-3), I shared in the competition period,  and [this](https://www.kaggle.com/osciiart/lb-probing-4)).   
- The number of the public test data: 923
- The number of the private test data: 7,695
- The number of labels in the public test data: 8,546
- The number of labels in the private test data: 62,671
- The number of detected labels in the test data by string-matching of train-data labels: 1,717  

From these results, we found that the public test score of train-data label string-matching is very high (0.530), but there are very few train-data labels in the test data (1,717). Therefore, at least 1,600 of the train-data labels in the test data might be in the public test data and very few might be in the private test data. Therefore, the public score of submission which discards the train-data labels from the prediction will correlate well with the private test score. The best submission can be obtained by finding the submission with the max score without train-data labels and adding train-data label string-matching to it.
**By this approach, we succussed to select the best-private-score submission out of our 201 submissions.** We knew we can survive the big shake of the private test LB, this was a very very big advantage for our team.

# 2. Acronym detection
Most datasets have acronyms (e.g., National Education Longitudinal Study → NELS). So, we did acronym detection to detect dataset-names that are not included in the train-data labels. The following procedure was used to extract them.
1. Make a list of words by splitting a text by space.
2. If a word in the list is surrounded by () and has uppercase characters and no lowercase characters, it is detected as an acronym candidate.
3. If the number of characters in the acronym candidate is less than the threshold, remove it.
4. Extract a few words before the acronym candidate from the text as a dataset-name candidate.
5. If the initial characters of each word in the dataset candidate can form the acronym candidate, detect them as a dataset-name/acronym pair. (The dataset candidate is allowed to have initial characters unrelated to the acronym candidate.)
6. Extract only those dataset-names that contain keywords (study, studies, data, survey, panel, census, cohort, longitudinal, or registry).
7. Exclude dataset-names that contain ban words (system, center, committee, etc.).
8. Apply the clean_text function.
9. Exclude dataset-name if Jaccard scores between the dataset-name and any train-data labels or acronym-detection labels are greater than or equal to 0.5.
10. Perform string-matching to the train and test data with the detected dataset-names and count the number of occurrences among the texts of each dataset-name. Extract only those dataset-names whose count is above the threshold, because It is more likely to be a dataset-name if it appears in a lot of texts.
11. Finally, perform string-matching using the extracted dataset-names. Only when a dataset name appears more than a threshold number of times in the text, it is detected as a label.  

The acronym itself is also detected as a dataset-name. String matching is performed on the dataset-name and the acronym. The acronym is detected as a label only when it and its long name appear more than a threshold number of times in the text.
By this acronym detection, we get a score of **0.418** on the public LB and **0.436** on the private LB. Each threshold was chosen based on the public LB score.

# 3. Acronym detection version 2
To obtain more dataset-names, we performed a more aggressive acronym detection. We extract words that contain uppercase characters and no lowercase characters from the texts as acronym candidates. We search a chunk of words that is valid as the full name of the acronym candidate among the entire text (the actual code is [this](https://www.kaggle.com/osciiart/210615-det-acronym-ver2/notebook?scriptVersionId=66042722)). This acronym detection does not improve the leaderboard score because it detects many false-positive labels. But we used the dataset-names detected by it for the NER model's training, which we describe in the later step.

# 4. String matching with dataset-names from external data
We used the external U.S. government’s dataset-names obtained from [this notebook](https://www.kaggle.com/mlconsult/100000-govt-datasets-api-json-to-df/). To reduce false-positive, we apply some processing below.
1. Ext Extract only those dataset-names that contain keywords (study, studies, etc.).
2. Apply the clean_text function.
3. Exclude dataset-name if Jaccard scores between the dataset-name and any train-data labels or acronym-detection labels are greater than or equal to 0.5.
4. Exclude dataset-name if the number of words it contains is less than the threshold.
5. Perform string-matching to the train and test data with the extracted dataset-names and count the number of occurrences among the texts of each dataset-name. Extract only those dataset-names whose count is above the threshold.
6. Finally, perform string-matching using the extracted dataset-names. 

This approach improved the score of the public LB score from **0.418** to **0.424** and the private LB score from **0.436** to **0.486**.

# 5. Dataset-name variation detection using named entity recognition (NER)
We attempted to train a NER model as a solution to this competition, using BERT or RoBERTa with the train data or the Rich Context competition data as training data. However, NER models never outperformed rule-based approaches. We think this is because a large number of true-positive labels are intentionally excluded from the provided train data. Therefore, the train data is incomplete as training data for machine learning. However, we found NER can be used to cover the weakness of string-matching. That is, NER is useful for detecting dataset-name variations that cannot be detected by string-matching. For example, National Education Longitudinal Study is sometimes quoted as National Educational Longitudinal Survey.  
We used spacy library to train a NER model. We used the train-data label, the acronym-detection label, the acronym-detection-version-2 labels, and the external U.S. government label for training.
We detect dataset-name candidates from the test data using the trained NER model. We calculated Jaccard scores between dataset-name candidates and any train-data labels or acronym-detection labels. We selected candidates with Jaccard scores greater than or equal to 0.5 as variations of the existing labels. This approach improved the score of the public LB score from **0.424** to **0.440** and the private LB score from **0.486** to **0.504**.

# 6. String matching with dataset-names from the train data
Finally, we applied basic string-matching using the train-data labels. We also used the acronyms of the train-data labels for string-matching. This approach improved the score of the public LB score from **0.440** to **0.614** and the private LB score from **0.504** to **0.513**.

# Codes
[Kaggle_Coleridge_4th_Solution](https://github.com/OsciiArt/Kaggle_Coleridge_4th_Solution) (main) 
[coleridge_ner](https://github.com/usuyama/coleridge_ner) (sub).  
Please refer to README.md of the main repository to reproduce the training and the prediction.
