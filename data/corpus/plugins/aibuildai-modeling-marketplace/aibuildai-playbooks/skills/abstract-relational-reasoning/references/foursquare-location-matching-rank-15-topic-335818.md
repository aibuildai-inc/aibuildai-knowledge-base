# 15th place solution

Competition: foursquare-location-matching
Rank: #15
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/335818

Although I was not able to win the gold medal this time, I would like to publish my solution.
Please forgive me if my English is not correct because I used an automatic translation.
My solution consists of Blocking, Matching, and Postprocessing.

- Blocking
	- KNN in L2 norm for a matrix combining the following three (performed with three weights, Lat/Lng : Name : Categories = 3 : 10 : 1, 100 : 10 : 1, 3000 : 10 : 1)
		- Latitude and longitude
		- Name (TFIDF)
			- After preprocessing by unidecode, vectorized by
			  - TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), sublinear_tf=True)
			- 'char_wb' was better than 'word' with higher tolerance for misspellings.
		- Categories (MDS)
			- Using the matching rate between categories in the train data as distance, I created 4-dimensional embeddings using [Multi Dimensional Scaling (MDS)](https://en.wikipedia.org/wiki/Multidimensional_scaling).
	- Exact matches for attributes (name, address, zip, phone (last 4 digits), url (domain name))
	- Merged the above and aligned the id/near_id pairs in one direction and removed duplicates.
	- 12 candidates per 1 id (about 7M candidates on test data), max IOU w/o postprocessing is about 0.983.
	- I also tried embedding based on deep learning, but it did not work well and took too much inference time, so I gave up on it.


- Matching
	- Distance and country tokens were added to the [Ditto](https://arxiv.org/abs/2004.00584) based input and trained mdeberta-v3-base using fine tuning. Example inputs are as follows.
	  - ' [D36] [RU] [COL] name [VAL] Аптека [COL] categories [VAL] Pharmacies [SEP]  [COL] name [VAL] Аптека [COL] categories [VAL] Pharmacies [COL] address [VAL] Ул  Лесная д  8А',
       ' [D29] [HU] [COL] name [VAL] Kazánház [COL] categories [VAL] Bars [SEP]  [COL] name [VAL] Cézár Ház B Épület [COL] categories [VAL] Residential Buildings (Apartments / Condos) [COL] city [VAL] Budapest [COL] zip [VAL] 1132',
       ' [D16] [JP] [COL] name [VAL] スシロー [COL] categories [VAL] Sushi Restaurants [COL] address [VAL] 寺崎北1 7 4 [COL] city [VAL] 佐倉市 [COL] state [VAL] 千葉県 [COL] phone [VAL] 5830 [COL] zip [VAL] 285-0819 [COL] url [VAL] akindo-sushiro [SEP]  [COL] name [VAL] 幕張イオン',
   - Distance tokens are represented as [D0]-[D49]. [Di] means that the haversine distance is in the 2i to 2(i+1) percentile of the entire train data.
   - The country token is simply the string of country.
   - Fine tuning is done in 4 epochs, all positive pairs are used, negative pairs are divided into 4 parts and each part is assigned to each epoch.
   - Trained by the entire train data, no validation. Single model, single fold.
   -  Google Colab (V100, high memory) was used, 40 hours per epoch (160 hours overall).
   - Learning rate is 2e-5. Batch size is 36.
   - Max token length is 128, which covers over 99% of train input sequences.
   - Adversarial Training (FGM) was used (eps = 0.1) .


- Postprocessing
  - I constructed my own algorithm based on the group average method.
      - The usual group average method is too time-consuming due to the computational complexity of O(N^2) where N is the length of the test data. But this algorithm works with O(NKC^2), where K is the number of candidates per id, and C is the average cluster size.
  - The rough descriptions of the algorithm are as follows.  
      - Use a priority queue to process the clusters in order from the edge with the highest value of predictions. If the prediction is known for all possible edges between clusters, and the value retrieved from the priority queue matches the average of the predictions for all edges, merge the clusters with Union-Find. If there are edges for which the prediction has not yet been determined, the information is temporarily stored in the queue for additional predictions. When the additional prediction queue is full to some extent, the above matching process is used to obtain the missing predictions for the missing edges, calculating the average of the predictions for all edges, and the information is again stored in the priority queue.
  - The score was expected to be improved by roughly 0.025 compared to the case where postprocessing was not used.
  - I initially used probability (0~1) as the value of prediction, but using the value of raw prediction before passing through sigmoid improved the score by about 0.002.
  - The loop ends when the number of performing merging reaches 35% of the length of the test data.
