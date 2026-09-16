# A few notes on #16 solution

Competition: foursquare-location-matching
Rank: #16
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/335860

First, thanks to Kaggle and Foursquare for providing a very interesting challenge. The leakage, though, is rather frustrating. This would have been an easy leak to avoid. I can't imagine how this slipped through. Looking for leaks is a normal part of Kaggle competitions, but when the instructions explicitly state that a particular leak does not exist, then it really shouldn't be there. 
     
Congrats to the winners, especially @psi for his late surge and winning solution (if we don't include explicit use of the leak). This was a tough competition that required a lot of ingenuity to do well.
      
My big take-aways from this competition are:
      
1. RAPIDS is a very mature and useful library. This was my first time using RAPIDS and I was really impressed with how much sklearn and pandas functionality can be run on the GPU with virtually no code changes. The speed-up was  impressive, espcially in cases where the non-GPU version uses only 1 CPU. Also, many thanks to @steubk for showing how to do inference using ForestInference (https://www.kaggle.com/competitions/foursquare-location-matching/discussion/328242).

2. Old-school nlp techniques such as text normalization, tfidf, keyword extraction, etc. are still relevant in isolated cases. In this competition, most of the text were names in multiple languages, which reduced the usefulness of pre-trained text models. 

3. As ML toolsets mature, ML is becoming more and more just a software engineering specialty. In this competition especially, setting up the problem correctly to process the data in chunks using minimal resources was just as important as the specific ML algorithms employed.

# Big question about this competition
     
Why are most of the top teams from Japan? Is there something about the data that gave teams from Japan an advantage?
     
Below is a brief outline of my solution (does not explicitly use the leak).
     
## Training Pipeline
    
1. Preprocess training data:
	a) Normalize text fields (lowercase, remove punctuation, normalize common spelling variations, convert to ASCII)
	b) Select frequent name keywords and encode as features
	c) Separate multiple categories and encode as single categories.

2. Randomly divide training set into two equal parts, grouped by point_of_interest, producing two sets -- train0 and train1.

3. Process the two training sets separately, as follows:

        a. Get term frequency vectors for text fields
	b. For each id, find 200 closest neighbors by lat/lon.
	c. Generate features in groups of 10 neighbors (e.g., first 1-10 closest neighbors, then 11-20, 21-30, etc.). Each id is paired with the 10 neighbors in the group to generate pairwise features. Altogether, about 50 features are generated, the most important of which are:
		1) Cosine similarities between text fields (especially normalized names, name_nonkey, and address)
		2) Distance rank (e.g., 2nd closest neighbor vs. 150th closest)
		3) Prev and Next distance (e.g., if we're processing the 5th closest neighbor, what are the distances of the 4th and 6th neighbors)
		4) Jaccard similarity between names
		5) Raw latitude/longitude
		6) Encoded categories
		7) number of matching name words in pair 
		8) length in characters of longest name
		9) Min and max ordinal values of characters in name (proxy for character set / language)
		10) Number of category matches
		11) Encoded country

4. Again, processing in groups of 10 neighbors, train shallow LGB models to predict pair matches.

5. Use the match predictions from #4 to select the most likely matches. The goal here is to reduce the candidate universe and get a more balanced training set without throwing out too many positives. For each group of 10 nearest neighbors, we train 2 shallow Lightgbm models -- one for train0 and one for train1. We use the model from one set to predict matches for the other set so that we get oof predictions for both sets. We save the 40 models (2 each for 20 groups of 10 neighbors) for inference.

6. Using the predictions from step 5, we set thresholds to filter out candidates that have almost no chance of being matches. I found that setting the threshold at about 30-1 (positives to negatives) worked well. It captured about 95% of the positives (the ones in the 200 nearest-neighbor pairs) while eliminating about 90% of the negatives. The threshold values were set using different percentiles in each group of 10. For example, in the neighbor group 1-10, we keep the top 15% of pairs, whereas in the group containing neighbors 190-199, we keep only the top 0.05% of pairs.

7. Use two additional heuristics to collect more candiates:
	a) Round latitude and longitude values and then merge by the rounded lat/lon and specific fields, such as name, address, phone, url, etc. Rounding the lat/lon values is equivalent to dividing the map into quadrants and assigning the same lat/lon values to all ids in the quadrant. Then if we merge by name, for example, we're finding all pairs sharing the same name within each quadrant.
	b) Merge by city and name.

8. Combine all candidates from steps 6-7. Recall that steps 3-7 are applied seprately to the two training sets generated in step #2, so in this step we also combine the candidates from those 2 sets. This gives us our master set of training candidates -- about 20m records, with a negative-positive ratio of about 15-1. Of the potential 1.9m positive pairs in the training set, we capture about 70%. 

9. Train Lightgbm and Xgboost models and save the models for inference.

10. Repeat steps 2-9 five times to get different lgb/xgb models.

## Inference pipeline

1. Apply steps 1-10 (excluding step #2) to the test set. For the modeling steps, we do inference only using the models generated from the train set.
2. Post process predictions to add two types of pairs that may be missing from the candidate set:
	a) inverses (e.g., if AB is a match, then BA should also be a match)
	b) graph traversal pairs (e.g., if AB and AC are pairs, then BC should also be a pair). For traversal pairs, we use a threshold of 0.65, meaning that subgraphs are formed only from pairs whose the probability of being a match is greater than 0.65.

## CV/LB Agreement
  
CV scoring was performed on a holdout set of 20% of the candidates, grouped by point_of_interest. From the beginning, the LB scores were slightly higher than the CV scores. By the end of the competition, they were considerably higher:
    
CV: 0.908 (single model)
LB: 0.935 (ensemble of 10 models)
    
I expected some increase due to the ensembling but this was more than expected. We now know that this is due to the leak. Essentially, the difference in CV/LB scores represents how much our models overfit. So, it's true, as some have written, that we all used the leak implicitly, but that's still a lot different from using it explicitly.
