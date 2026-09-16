# [Solution of 4th Place in Efficiency LB]

Competition: llm-detect-ai-generated-text
Rank: #47
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/471788

Firstly, I would like to thank Kaggle and THE LEARNING AGENCY LAB for hosting such an exciting competition. During this competition, I have been following the top solutions in the discussion area. These ideas are very cool, and I have learned a lot from them. I am deeply appreciative of the participants who generously shared their insights and observations.

My solution was simple, yet to my surprise, it secured the 4th place in the private Efficiency LB. From the perspective of the Public Efficiency LB, this achievement seemed almost impossible…

I would like to extend my thanks to @datafan07 for providing [DAIGT-V2](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset). I also acknowledge @mustafakeser4 for sharing a high-scoring [Bert model](https://www.kaggle.com/code/mustafakeser4/inference-detectai-distilroberta-0-927) and the source of the training data. Although I did not utilize the Bert model in my final submission, I incorporated the data as external training data. The results clearly indicate that this external data was the key factor behind this surprise.

#### **Solution Summary:**

I used multiple classifiers to classify the TFIDF features. Our code comes from

https://www.kaggle.com/code/batprem/llm-daigt-cv-0-9983-lb-0-960,

thanks to @batprem for sharing.

At the beginning of my participation in the competition, I observed that the leaders in the Public LB often also topped the Efficiency LB. This led me to hypothesize that utilizing TFIDF alone could simultaneously yield high scores and efficiency. Throughout the remainder of the competition, my focus was on enhancing the score as much as possible without compromising the algorithm’s efficiency.

Here is a summary of my solution:

1. I found that Cat was the primary cause of the algorithm’s slowdown. Consequently, I removed Cat while retaining lgb. This action reduced the algorithm’s runtime from approximately 2 hours to about 15 minutes.
2. I added ComplementNB and LinearSVC as base classifiers, which could slightly improve the public LB score.
3. I utilized DAIGT-V2 and the previously mentioned external training data. I ran the same feature extraction and classification algorithms on the two datasets separately and then ensemble them in a weight of 6:4. This resulted in a public score of 963 and a private score of 916. Besides, 75:25 yielded a public score of 964 and a private score of 906.
4. Taking inspiration from [the tricks for AUC metric](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/468150), I applied min-max normalization to each prediction result prior to integration, slightly improving both the public and private scores.

I have also tried to mix DAIGT-V2 and the external data for training instead of blending. This method produced a public score of 955 and a private score of 934. Due to the low public score, I did not select this submission as the final result. In my experiments of using different weights in the ensemble, I found that the more the weight is biased towards the external dataset, the higher the private score, but the lower the public score, so I only tried 6:4 and did not continue to try larger weights. I believe that if only external data is used for training, or if a more significant weight is set, it is possible to achieve 1st place in the Efficiency LB.

#### Code
https://www.kaggle.com/superfei/solution-of-4th-place-in-efficiency-lb
