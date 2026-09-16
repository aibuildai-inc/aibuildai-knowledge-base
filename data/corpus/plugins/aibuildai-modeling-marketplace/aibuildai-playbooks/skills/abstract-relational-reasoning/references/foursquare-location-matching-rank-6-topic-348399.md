# 6th place solution

Competition: foursquare-location-matching
Rank: #6
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/348399

First of all, thank you to everyone who organized the competition.
The competition was ended over a month ago, I share my solution.
This competition's task and data were interesting, but I regretted about the leakage.

# 1. Preprocessing

---

- Remove symbols that seem to be noise.
  - About zip and phone, leave only numbers.
- Apply pykakasi to JP and unidecode to all data.

# 2. Candidate Generation

---

- Generate 28 candidates per id, IoU=0.978.
- Dataset Creation Process
    - Split the data into two parts by poi.
    - Create candidates and features for each data.
    - Change seed and split the data multiple times.
    - Combine them to form a dataset.

#### Phase1
- Extract 6500 points for each point by coordinates, using kdtree.

#### Phase2
- Calculate the similarity with each point, and select top28 as a candidate by the following weighted sum.
    - square root of the coordinate distance
    - 1 - jarowinkler: name, categories
    - normalized levenshtein: name, categories
    - 1 - simpson: name splited by ' ', categories splited by ', '

# 3. Matching Prediction

---

#### Feature

- coordinate distance rank of kdtree
- language identification from name
    - using top3 labels and probs
- coordinate distance
    - haversine, chebyshev, euclidean, manhattan
- similarity of the categorical feature
    - LCS(Subsequence/Substring), levenshtein, jarowinkler
    - About categories splited by ', ', use similarity, calculated on a brute force basis, mean, min and max.
- string length of categorical features
- features about sets based on string and word
    - jaccard, dice, simpson
- count encoding
    - coordinate, name, address, url, phone
- tf-idf
- rank features about frequency of occurrence
    - country, categories, categories splited by ', ', categories splited by ' '(word)
    - Except country, labeling in both of ascending and descending order.
    - Labels less than 1% are to be removed.
- rank features for each candidate
    - Create a rank of between 28 candidates per id for each feature.

#### Model

- catboost, single model
- remove early stopping & increase iterations (overfit)
- train on all data

# 4. Postprocessing

---

- graph probability convolution

    - ex) Updating the prob of the connection from A to B (red line).
        - Extract A's candidates that have B as a candidate.
        - Limit to connections having 0.5 or more prob to A.
        - Calcurate a weighted average of the probs of the connection with B, using the probs of the connection with A as the weight.
        - AB’
             = (AC * CB + AD * DB + AA * AB) / (AC + AD + AA)
             = (0.7 * 0.7 + 0.9 * 0.1 + 1 * 0.6) / (0.7 + 0.9 + 1)
             = 0.45
- Add nodes that are connected with a prob of 0.5 or more at the adjacent nodes to the prediction.
    - Repeat twice.

# 5. Other Tips

---

- Build with dict and numpy basically.
- Batch processing of feature creation and prediction to avoid memory errors.
- Parallel processing of the candidate generation with 4 cpu at submission.
- Convert id dtype from str to int.
