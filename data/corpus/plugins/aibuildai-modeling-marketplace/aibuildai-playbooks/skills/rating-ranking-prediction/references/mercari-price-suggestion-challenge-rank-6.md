# 6th place solution!!

Competition: mercari-price-suggestion-challenge
Rank: #6
Source: https://www.kaggle.com/c/mercari-price-suggestion-challenge/discussion/50431

First of all! Thanks to everyone who participated here for making this a great learning experience. Special mentions to @Konstantin, @Pawel, @RDizzl @ololo @Muhammad @ThyKhueLy @Serigne and ofcourse my teammates @Ajay @huiqin

A big thanks to all my teammates for putting in lot of effort which helped us finish 6th!

Thanks to organizers and admins, I think they did a wonderful in handling their first kernel competition! 

Solution Summary
----------------


We had three models in our ensemble.

1. **NN with extra features** (Our main model which gets 0.400x on plb in 2500s):
(https://www.kaggle.com/tezdhar/nnet-v1001-attention-dbbd10)

 - text processing: We tried some manual text normalization, but yielded
   very little improvement, nevertheless we decided to keep it.

 -  Name and description fields were padded, followed by a embedding layer and average pooling, (we tried GRU with attention on name field
   and that yielded little improvement)

  - 2 gram features - Initially we started with CNN (filter sizes 2 and 3) to capture 2 and 3 grams but that resulted in longer running
   time, so I decided to encode name and description field into 2 gram  
   text and directly use embedding layer followed by average pooling.   
   The preprocessing time and running time for later was lower than CNN 
   model and resulted in almost similar score.

  - I explicitly added interaction features e.g. category*condition, category*shipping etc. This improved score by
   0.002-0.003 (May be I    could have added dot product layer for capturing interaction effects, but I got that idea    too late to implement)

  - We just concatenate all layers and pass through two sets of batch_norm and PRelu

  - Used different learning rate for each epoch with Adam optimizer

  - Optimized hyperparameters using gaussian process in scikit-optimize locally on validation set(5 splits with 5% data   
   concatenated)

  -  The next major thing was to use all 4 cores to run 4 NN's and bag them, we used multiprocessing to do this (we took the risk)
      
  - We used slightly modified targets in 2 NN's for added diversity, a standardized target and target with category means subtracted from    them.


2. **NN with conv1d**:

 - We started with [public kernel](https://www.kaggle.com/agrigorev/tensorflow-starter-conv1d-embeddings-0-442-lb) and tuned the architecture and hyperparameters using our learnings from above model


3.  **Fastetxt NN**
Same as 1st but without any text normalization and extra features. 

Both 2 and 3 ran very fast (combined under 20 minutes and scores 0.408x on plb). It can be found here:
https://www.kaggle.com/tezdhar/tf-conv1d-tf-fasttext

Final solution was a simple weighted average of above 3.

I also started with Keras based FM (based on script from qianqian) but was not able to give it time for it to be included in ensemble. It can be found here:
https://www.kaggle.com/tezdhar/kerasfm/code
In current state, it scores 0.421 on plb

Thanks for reading! :)
