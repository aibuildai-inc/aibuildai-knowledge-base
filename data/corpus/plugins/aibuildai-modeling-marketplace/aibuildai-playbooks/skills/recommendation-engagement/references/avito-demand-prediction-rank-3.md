# 3 place solution

Competition: avito-demand-prediction
Rank: #3
Source: https://www.kaggle.com/c/avito-demand-prediction/discussion/59885

Congratulations to the winners, especially **Dance with Ensemble** your score was amazing from start to finish. Also, congrats to the solo gold medallists, and especially to the solo kaggler who claimed it was **too hard to get a solo gold** and eventually snuck in at #13– we have been secretly rooting for you.

Here we will outline a brief explanation of our solution.

Ensemble/approach
-----------------

Our solution is the average of 2 ensembles. The first ensemble was trained using a standard 5-fold validation schema.

The second ensemble used a time-series schema. The test data begins a number of days past the final day included in the training data. We tried to replicate this in our internal validation. To do so, we trained on the first six days of training and used days ten through thirteen as validation. This meant days six through nine were excluded, mimicking the gap between train and test.Then, to generate predictions for the test data, we trained on the entire training set.  

In order to generate likelihoods of categorical features for this approach, we always applied a gap of 4 days. For example to estimate likelihoods for day four of the training data, we use the average of target for day zero. To estimate likelihoods for day five, we used likelihoods of (day0+day1)/2. We decided on a gap of four days for stability, as it gave similar CV and LB performance. 

Our best single models came from this approach. Our best single lgb clocked in at 0.2163, and our best single nn scored .2180.


LightGBM features
-----------------

 - Tf-idf on words  (2 grams on description, 1gram for the params and
   title)
 - Tf-idf on chars (5 grams)
 - Word2vec-based features on words  (this worked better for us than
   fastext)
 - Pretrained fastext features on words
 - Image quality features (from [here][1])
 - Vgg16 feature (from [here][2] 
   and [here][3] )
 - Vgg19 feature (similar to vgg16)
 - Resnet prediction (of object top 3)  features
 - Inception prediction (of object  top 3)  features
 - Xception prediction (of object  top 3)  features
 - Some binnings of numerical features (like price)
 - Some group-by user type of features (like average number of words,
   average number of days of Displaying ads)
 - Some text based counts (like upper counts, punctuation, counts of
   emojis)
 - Some location features based on latitude and  longtitude
 - Likelihood and counts on almost all 3-way interactions of categorical
   features (for kfold approach we excluded user_id interactions as they
   over fitted. For the time series approach, user_id likelihoods gave a
   good boost)

We primarily used **xentropy** for our objective, as probabilities are constrained between zero and one. 


Neural Networks
---------------

Our best nn used two stacked, bidirectional GRUs on the text with concatenated embeddings of fastext and those trained on our own with word2vec, along with some of the numerical features used in LightGBM. All categorical features (minus user id) had embedding layers of size 100. Our objective was binary cross-entropy with sigmoid output. 

A few small additions/key points:

 - There was a dense input for the vgg16 and vgg19 features
 - Text was stemmed based using nltk.
 - For text we combined description,title and params into one field,
   separated by a dividing character.


In addition to the above network, we ran a NN for each feature channel (image, text, character) with minimal categorical embeddings and likelihood features. When stacking with just these models, they scored ~.216 at the second level, which was comparatively weak to lightGBM. 

For these models, we primarily used each feature channel independently, than concatenated them with basic likelihood features and categorical embeddings before feeding them to dense layers for the output. While these models were individually weaker, they stacked well, and maximized the 2nd-level information from each feature channel.


Stacking
--------

Both approaches include mostly lightGBM and neural nets. In addition, simple ridge models were used to improve performance for some lightGBM models, in a similar manner to a couple of shared kernels.

To stack our 5-fold approach, we used the time schema as explained above (0-5, 10-14) to validate and optimize hyperparameters. We also did a bit of re-stacking, adding some counts and likelihoods spanning user-based interactions of categorical features and price. Our second level stack combined lightGBM, neural nets using 2 hidden layers, linear output, and mse objective, and sklearn’s ExtratTeesRegressor. This approach scored 0.2145.

For the time series approach, we found best parameters using days [10,11] for training and the remaining for validation. Restacking did not help here and we used lgb and nn of equal weight .  This scored 0.2140.

Finally, a blend (35% of time stack + 65% of 5-fold stack) gave our best public (0.2136) and private score.


  [1]: https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality
  [2]: http://%20https://www.kaggle.com/bguberfain/vgg16-train-features
  [3]: http://%20https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16
