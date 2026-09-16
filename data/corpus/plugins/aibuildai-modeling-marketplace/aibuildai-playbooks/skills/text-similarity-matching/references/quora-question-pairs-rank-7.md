# 7-th solution overview

Competition: quora-question-pairs
Rank: #7
Source: https://www.kaggle.com/c/quora-question-pairs/discussion/34697

Hi there!
The competition has ended, and I can finally get on with my life.
However, after studying the solutions that the top teams have kindly shared, I found out that my approach was considerably different in a number of ways. So, for the sake of learning, I decided to give a summary of the key parts.

**DNN features**
First of all, I really don't like feature engineering, particularly in NLP tasks. So from the very beginning I decided to devote most of my time (and computational resources) to building and training a diverse array of deep neural networks to do the boring stuff for me. The resulting features were very predictive, some exceeding even the 'magic' features by a large margin in terms of information gain. Here is a ranked list of networks I used to prepare the final submission:

1. Bidirectional LSTM enhanced with Recurrent Highway Layers (see [this paper][1]) and a custom attention mechanism. This network has a standard 'siamese' architecture, finely described in a [blog post][2] from Quora itself.  The whole difference came from the attention mechanism, which boosted the performance of said network significantly. The addition of RHN layers allowed to put more dropout on the recurrent connections of the LSTM, which turned out to be a very effective regularisation strategy. I used glove 840B vectors for the word embedding layer, which was frozen during training. After quite a lot of hyperparameter optimization, I found a configuration which yielded 89.1% accuracy on validation (with cv). More details on that approach will probably be released soon.
2. Basically the same as 1, but character-level. Somewhat surprisingly, this one was only 1% acc behind the word-level one, and it provided very good, de-correlated features. 
3. Same as 1, character-trigram level. Really nothing else to say there
4. 1-D convolutional net with [decomposable attention][3]. This one was blazing fast as compared to the first one (20 minutes vs 2 hours to convergence). I had three of those networks, with different kernel size sets (I used [3], [2,3], [2,3,5]).
5. Networks 1-4, but without the attention mechanism

I used 'contrastive loss' as an objective function for all networks, since it turned out to be somewhat better than binary crossentropy in the long run. My guess is that crossentropy does not behave very well when the labels are noisy, which was certainly the case. Minimal preprocessing was used, no stop words were removed because it harmed performance. The predictions of the networks (after sigmoid) were used as features for the XGB. To get out-of-fold predictions on the whole train set, 10-fold cv was used.

**Unsupervised NLP features**
Most of these came from publicly available kernels by Abhishek, Mephistopheles and the1owl. 
I also did compute some by myself, including

- WordNet similarity (based on a script i found on [github][4])

- Word mover distances, using glove840B word embeddings

- NER-based features. I got the tags with Stanford CoreNLP

- POS-based features. Same thing

Probably there was something else, I don't remember now.

**Magic features**
My use of magic features was pretty consistent with what I have read from the top teams.
In addition, I used another 'pseudo label' feature which was constructed as follows:

1. Using the best available model, predict the test set.

2. Build a sparse square matrix of size len(train+test) x len(train+test)

3. For all pairs with duplicate probability higher than threshold (say 0.3), put a '1' in two corresponding cells of said matrix

4. For each pair from train and test, compute cosine similarity between corresponding rows.

Interestingly, this feature turned out to be very helpful, and did not lead to overfitting.

**Post-processing**
Again, mostly similar to the published solutions. In addition, I did prediction clipping, which means that I set very confident predictions to 1e-5 and 1-1e-5, respectively.


----------


Overall, this was a great competition, which taught me a lot about the ways of NLP, DL, ensembling and a bunch of other stuff. For me, Kaggle worked brilliantly by providing motivation to dive deeper into all these areas, which I wouldn't have the energy to explore otherwise. 

Finally, many thanks to participants that shared their thoughts and approaches, some of which were very helpful. See you next time!

  [1]: https://arxiv.org/abs/1607.03474
  [2]: https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
  [3]: https://arxiv.org/abs/1606.01933
  [4]: https://github.com/sujitpal/nltk-examples/blob/master/src/semantic/short_sentence_similarity.py
