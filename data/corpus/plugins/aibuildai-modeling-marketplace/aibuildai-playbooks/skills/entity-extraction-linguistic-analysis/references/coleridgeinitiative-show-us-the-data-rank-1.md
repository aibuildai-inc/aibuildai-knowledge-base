# 1st solution: Matching the <MASK>: Context Similarity via Deep Metric Learning and Beyond

Competition: coleridgeinitiative-show-us-the-data
Rank: #1
Source: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/248253

Thanks Kaggle for this exciting competition and our team (@suicaokhoailang). Congrats to me to become Kaggle Master and @suicaokhoailang to become Kaggle GM. Congratulate all winners -- we have learned a lot from this competition and all winners’ solutions !!. 



Here Is a summary of my solution:
- A shared Bert model for extract Context Embedding and Sequence Tokens Embedding.
- An ArcFace Loss for training Mask/NoMask Embedding.
- A BCE loss for training NER model to detect dataset citation in the input string.

Bellow is the most important question that we should answer for ourselves before deciding on an approach.

### Do we need a context?

Yes, a semantic context is very important for this kind of problem. As you can see, there are some solutions that don’t need semantic context and only needs dataset title. But those solutions are only suitable for this contest data since almost the dataset title includes keywords such as Study, Survey, Database, Dataset, … then the dataset title itself is good enough to classify if it’s a real dataset title or not. In this competition, we aim for a general solution that uses a semantic context to detect a dataset citation.


### How my model was trained?

1. An input text is randomly split into Support Set and Query Set.
2. In Support Set, we use a **MASK** token to replace all dataset titles in an input text.
3. Pass all input text from Support Set into a BertModel and extract **MASK**/**NoMASK** embedding.
4. In Query Set, we pass a raw input into a shared BertModel used for Support Set to extract a sequence tokens embedding.
5. **MASK** and **NoMask** embedding from Support/Query Set is passed into an ArcFace Metric Learning layer.
6. A **MASK** embedding from the Support Set contains only a semantic context feature is used to calculate l2 norm cosine similarity with every token feature from the input text in the Query Set.
7. A l2 norm cosine similarity will range from (-1, 1), we scale it into a range (-10, 10) then apply sigmoid activation before use it for training the NER model via BCE loss.


### How to use my trained model for the inference?

1. Pre-extract all **MASK** embedding from the given training set.
2. Random selection K **MASK** embedding then average them to obtain the semantic context vector and perform step (6), (7) from the above training process.
 

### Post-Processing

As @leecming said in his solution, a dataset is probably referenced by multiple documents so the more frequently they’re referenced, the more likely they’re a hit and not a spurious find. We use a threshold frequency of 2, which means if our model can detect that dataset title twice, we will get that dataset. We choose 2 because, in training data, there are many dataset titles mentioned only 1-2 times, we are afraid in the private dataset, such dataset titles will be many but we seem to be wrong, a higher threshold frequency gives better results on public and private leaderboards.

Here is a summary of my solution, there are a few string rules-based that were used but did not affect the final result too much. The final submission is based on an ensemble of allenai/scibert and allenai/biomed_roberta. The code is done on Tensorflow 2.4 with V100 16gb version in AWS EC2. 


**Inference Kernel**: https://www.kaggle.com/dathudeptrai/biomed-roberta-scibert-base

##### Thanks for reading !
