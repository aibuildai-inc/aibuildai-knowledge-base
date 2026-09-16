# 38th place solution (NER + Heuristics)

Competition: coleridgeinitiative-show-us-the-data
Rank: #38
Source: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/249027

I am afraid my solution is way too simple compared to some of the cool models I've seen. But here goes:

I used Spacy's entity recogniser to filter all possible candidate datasets. Then I wrote a simple CFG which determines if those entities indeed denote a dataset. This part generates candidate datasets that are strict matches, thereby increasing the rate of false negatives. To mitigate this, I did a second pass on all texts with key terms in these candidate datasets to generate the final discovered dataset. That's the model in a nutshell. 

Few additional notes:
* This model is nearly unsupervised. 
* I did not use the train dataset for the actual dataset prediction. I only used it to determine what kind of CFG I need to write. 
* One very direct observation is, all datasets that are referred in these publications are camel-cased. If nothing else, you can start with these as your candidates and even exclude the named-entity task. (That gave me a private score of 0.29. The final model I submitted with Spacy gave 036).
* These are not free-form texts on which standard large-scale models have been trained. These are publications and they follow a certain standard. If you analyze the training dataset, you can automatically see that most of them are of the form `data collected from <some dataset>`, `Used <some report/study>`, and so on. The second pass in my model exists to capture those that don't follow these candidate phrasings.  
* I did not use Roberta for the same reason. See the first place solution post for a much more in-depth discussion on this. 
* As for the heuristics, you can just start with "data" and look for subtree (in the PCFG) matches that intersect with the matched entities.  I used a few more. Again, the input texts are almost free of grammatical errors and follow very similar patterns while quoting references and external datasets.  
* **Spacy 3.0's Transformer gave me much better initial candidates. But it kept throwing some submission errors.** So I used Spacy 2.5 since I didn't have much time to compete in this competition.
