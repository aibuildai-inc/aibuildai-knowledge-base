# 15th place solution [0.21074]

Competition: gendered-pronoun-resolution
Rank: #15
Source: https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90366#latest-523380

My approach was quite straightforward for this competition I have used embeddings, extracted by BERT + some other features and fully connected NN to get the final predictions.

**Embeddings:**
I have used Bert large uncased (without any finetuning) and extracted embeddings from 9 layers – from -1 to -9. I have concatenated all of them and used as a 1024*9 = 9216-dimension vectors as final embeddings. Other sets of layers gave me worse results.

**Additional features:**
-	Pronoun offset
-	Pronoun token offset
-	A and B tokens offsets
-	A and B lengths
-	Distance between a/b and pronoun (in symbols and tokens)
-	Some features from this kernel https://www.kaggle.com/pheell/look-ma-no-embeddings

**Additional data:**
-	Manually corrected labels from here https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/81331

**Model:**
1.	All other (non-bert) features were passed through the linear layer:
Other_features (separate A+P and B+P)-&gt; bn -&gt; linear -&gt; relu -&gt; preprocessed other features
2.	Concatenated other and bert features were passed through the FCNN:
(A/B BERT emb + preprocessed other features) -&gt; linear(1800) -&gt; relu -&gt; dropout -&gt; bn -&gt; linear(500) -&gt; relu -&gt; dropout -&gt; linear(1) -&gt; A/B pred
3.	The final prediction = softmax(A pred, B pred, 0)

**Training:**
-	A have trained the model with Adam. Everything was highly regularized (~90% dropout, batch norm, weight decay ~6)
-	The model was trained on all GAP data using 10-fold cv. The final prediction was just a blend of 10 predictions.
