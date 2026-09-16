# 8th place solution [SBERT + LightGBM]

Competition: foursquare-location-matching
Rank: #8
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/335928

Hello.

Here is the solution that got us to 8th place. 
I apologize if the English is incorrect because I am using an automatic translation. 
Also, this is my first time posting in a discussion forum so I may not be able to summarize it well. If you have any questions, please comment.

### Overview

Like many other competitors, our method consists of three steps. 

1. Creation of candidate matching pairs. 
2. Binary classification to determine if the locations are identical or not. 
3. Post-processing

The basic process is copied from [@guoyonfan's baseline](https://www.kaggle.com/code/guoyonfan/binary-lgb-baseline-0-834). (Thank you for the great baseline.)

Below are some of the points we devised for each step

### Creation of candidate matching pairs

In addition to the distance-based kNearestNeighbor (distance kNN) used in the baseline, kNearestNeighbor (embedding kNN) using embedding created from fine-tuned SentenceBERT Added.

The k parameters used for the final submission are 10 for distance kNN and 20 for embedding kNN.
The ideal IoU simulated on the training data was 0.986.

The embedding kNN is very important and played a major role as a feature in the binary classification described below.

#### SentenceBERT embedding details

The embedding model was fine-tuned based on `sentence-transformers/all-MiniLM-L12-v2`. 

The key points that were important to get a good performance with the embedding kNN are
- Does not use multilingual model
    - In fact, we also use multilingual models to create features for the binary classification step, but the English-based models are by far the most important.
- Loss selection
- Two steps fine-tuning

##### does not use a multilingual model.
The input of the model is a concatenated string of name, address, city, state, country, category, and latlon. 
Since the base model is not a multilingual model, names and addresses are normalized by pykakasi for Japanese and by unidecode for other languages. 

The reason why we did not use the multilingual model is that the English-based model with normalized inputs simply performs better, but we believe that the reason is that the number of parameters in the multilingual model is very large and the amount of data is insufficient for fine-tuning. 
Usually, a large amount of training data is required for metric learning, but the number of bases available for this competition was only about 1.1 million, which is not enough data to obtain invariant expressions from data expressed in dozens of languages, let alone a single language. 

Another reason why we could not successfully train multilingual models may be that we had to reduce the size of the mini-batches that could be set during training due to the memory limitation of the GPU. 

The models were trained on Google Colab using a single GPU, and if more resources and time were available, the multilingual models might perform better. 

##### Loss Selection
For the loss function of fine-tuning, we used [Contrastive Loss](https://www.sbert.net/docs/package_reference/losses.html) implemented in sentence-transformer. library 

We chose this function because it has the best performance compared to TripletLoss and ConsineSimilarityLoss.

##### 2-step fine-tuning

To obtain better performance, two stages of fine-tuning were performed. 

In the first stage, fine-tuning is performed using the pair set created by distance kNN as training data. 
In the second stage, fine-tuning is performed on the training data, using the pair set created by embedding kNN using the model learned in the first stage, in addition to the distance kNN. 

We believe that the second stage of fine-tuning contributed as a hard negative to the training data, since the locations with similar names and addresses added by embedding KNN were added to the training data. 

Comparing the first and second stage models, the ideal IoU(recall) improved from 0.97 to 0.986. 

### Binary classification to determine if they are the same location

For the binary classification, we used 7 LightGBMs created by kFold. 
Below are the features that were important to us and our training efforts. 

#### Features that were effective for prediction

In addition to the features used in the baseline, we added the following

- cosine similarity of the embedding vectors created by the three types of SentenceBERT
    - 1st : multi-lingual model with name, address, latlon, etc. as input + fine-tuning with triplet loss
    - 2nd : same as embedding kNN
    - 3rd: multi-lingual model with only names as input + fine-tuning with contrastive loss
- Country and category label encoding
    - Label encoding of country and category associated with the id that is the key of the matching pair
    - To avoid over-learning, labels with high frequency in train.csv are selected, and the rest are replaced with "other".
- Land use characteristics by area
    - The following statistics were created using latlon rounded to two decimal places as the key
        - Number of POIs included
        - A 5-dimensional PCA was applied to the frequency distribution of the category labels.
    - The above statistic was combined with the latlon of one id of the matching pair
- Normalized name tfidf similarity
    - Normalized names in unidecode and pykakasi converted to character base tfidf vector and cosine similarity calculated

feature importance is as shown in the attached image.

#### Learning LightGBM

LightGBM was trained by optimizing logloss and applying early stopping. 
The training data were the same pairs created by distance kNN and embedding kNN. 

One of the major findings is that the larger the number of id and the number of pairs in the training data, the larger the stopping iteration of early stopping becomes and the better the score improves. 

Initially, the public LB was 0.908 after training with 550,000 id, but it improved to 0.928 with 700,000 id, and to 0.948 with 1.1 million id.

In the final submission, seven models were trained using 25 million pairs created from 1.1 million locations. 
The number of stop iterations for each model was about 1000. 
(Unfortunately, due to memory limitations, the kaggle environment did not work, so we used Google Colabe pro+ for training.)

Finally, the model became so large that inference on the CPU took too long, so we used Rapids to speed up the process when submitting. (The inference time was reduced by a factor of 40, which was very impressive.)

### Post-processing
Omitted because no changes have been made from the baseline.

### Conclusion
This was the first competition that I seriously worked on, and there were many points that required a lot of trial and error, and I learned a lot about how to learn the model and how to make the resources more efficient. 
I would like to thank the competition organizers and Kaggle, a great platform, for giving me this opportunity. 

Last but not least, thank you to my delightful teammates.
