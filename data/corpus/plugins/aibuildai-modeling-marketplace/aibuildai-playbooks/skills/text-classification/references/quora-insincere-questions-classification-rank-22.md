# 22nd Solution - 6 Models and POS Tagging

Competition: quora-insincere-questions-classification
Rank: #22
Source: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80514

Thanks to everyone who participated in the awesome kernels and discussions that happened during this competition as well as my brilliant teammates.  Always great to have people to bounce ideas off of. 

Here is the link to our final solution: 
https://www.kaggle.com/ryches/22nd-place-solution-6-models-pos-tagging

The guts of our solution was largely driven architected the same as the kernels we made public. 

 1. https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings
 2. https://www.kaggle.com/ryches/parts-of-speech-disambiguation-error-analysis
 3. https://www.kaggle.com/mihaskalic/lstm-is-all-you-need-well-maybe-embeddings-also
 4. https://www.kaggle.com/christofhenkel/inceptioncnn-with-flip
 5. https://www.kaggle.com/christofhenkel/keras-starter
 6. https://www.kaggle.com/ryches/parts-of-speech-disambiguation-error-analysis

I have written a relatively comprehensive description of our entire solution in the link above, but to give a summary:

In this competition we were able to train a total of 6 models for a total of 74 epochs. How did we fit so many epochs into our 2 hour limit? We filtered out the easy examples. @christofhenkel figured out by looking at the histogram of our predictions that within a few epochs our models had already confidently classified over 70 percent of our training samples. We trained a model really quickly in order to filter these easy questions. Once we threw those samples away we were able to train models just as accurately only using the 30 percent that remained. 

Now that we had this additional time we trained 5 models paired with different embeddings based on how they performed in our offline ensembling. Our hillclimbing found that the best combination with 5 models was:

* DPCNN with reversing and glove embeddings
* A bidirectional gru into an lstm with the glove embeddings. (this was very similar to what we used for the toxic comment challenge and was our strongest individual model here as well)
* a parrallel lstm and gru model w/glove embeddings
* parts of speech bidirectional lstm and gru model w/paragram embeddings
* parts of speech parallel lstm and gru model w/news embeddings

These choices actually seemed to make some sense given that we have a CNN model, our strongest LSTM/GRU models, use our strongest embedding 3 times and use POS tagging as an augmentor/differentiator to our weaker embeddings. 

The POS models ended up doing worse individually but when ensembled significantly boosted our score. Our second submission used 8 models in total and still got a worse score than our 6 models with two of them being POS. If we did not do the filtering trick then we would not have enough time to do the POS tagging as it is relatively slow. I have a more detailed write-up of the POS models in the parts of speech disambiguation kernel I shared.
