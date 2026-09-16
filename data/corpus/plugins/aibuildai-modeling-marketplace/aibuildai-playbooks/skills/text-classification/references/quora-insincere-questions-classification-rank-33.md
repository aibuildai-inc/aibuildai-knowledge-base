# 33rd place solution- FastText embedding

Competition: quora-insincere-questions-classification
Rank: #33
Source: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80577

Let me start by thanking everyone here who participated and contributed. We learned a lot of tips and tricks from community shared kernels and discussions. The competition was challenging in terms of finding correlated local validation, running the solution in 2 hours, and producing reproducible results, to name a few. We found a nearly correlated validation set before 1 week of the competition end. We used simple averaging where each trained NN model (Pytorch) was different either in terms of learning rate, pre-processing, embedding or architecture to maintain model diversity. One important thing we realised was: given small learning rate and large number of epochs, single FastText embedding based models were beating our glove+paragram models. So we focused on tuning FastText based models which could provide considerable score within 5-6 epochs. We also added Gaussian Noise to some models after embeddings to reduce the overdependence of RNN on specific keywords.

**Solution summary:**

 - **Runtime**: 6352.2 secs
 - **Preprocessing**: Cleaning special characters, number pre-processing, misspell cleaning (For some models changed preprocessing sequence to add diversity)
 - **Embedding**: GLoVe, FastText, Paragram embeddings 
 - **Neural Network architecture** (trained for 5-6 epochs with no fold): 
  - Stacked LSTM-GRU-128 hidden units, with GloVe+Paragram embedding
  - Stacked LSTM-GRU 60 hidden units with attention and capsule, and GloVe+Paragram embedding
  - Stacked LSTM-GRU-60 hidden units with GloVe+Paragram embedding
  - Stacked LSTM-GRU-60 hidden units with FastText embeddings
  - Stacked LSTM-GRU-80 hidden units with FastText embeddings and different preprocessing sequence 
 - **Blending**: Averaging prediction of each model with linear regression coefficients 

**Things that did not work**

 - We tried pseudo labelling in different ways, but it didn't provide major boost considering its running time for it, so we dropped it in the end.
 - We tried variety of preprocessing techniques to no avail. All of them tend to decrease the LB score with slight improvement in cv. Fearing overfitting pre-processing to training data with kept it to minimum.
 - One trick that we tried was weight saving and retraining. For example, we trained the model and saved its weights before the model reached optimum. Then for the next model we loaded the weights for LSTM and GRU and did not pass gradients through them. This forced the new parts of model like an extra CNN layers or linear units to cover up for this. This saved time as the new model reached optimum within 2 epochs. But this did not add considerable benefit to the ensemble. In my opinion, majority of the information pertaining text was captured by RNN units, leaving little information required to be captured by newly added layers. Do let me know your thoughts on this experiments and its results.

Special thanks to [Shujian][1], [Benjamin Minixhofer][2], [Dieter][3], [Ryches][4], [Bojan][5], to name a few, for great kernels and discussions!

This wouldn’t have been possible without awesome teammates, [Ashish][6] and [Rahul][7], who put in lot of effort and made this competition a great learning experience.

Thanks for reading, I am planning to release the code in few days after cleaning it. 	

Edit: Github repository: https://github.com/soham97/Quora-Insincere-Questions-Classification-Challenge-NLP


  [1]: http://www.kaggle.com/shujian
  [2]: http://www.kaggle.com/bminixhofer
  [3]: http://www.kaggle.com/christofhenkel
  [4]: http://www.kaggle.com/ryches
  [5]: http://www.kaggle.com/tunguz
  [6]: http://www.kaggle.com/ashish2123
  [7]: http://www.kaggle.com/rsrade
