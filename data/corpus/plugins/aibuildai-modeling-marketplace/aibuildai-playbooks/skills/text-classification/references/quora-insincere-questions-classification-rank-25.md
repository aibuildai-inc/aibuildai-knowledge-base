# 25th place solution - unfreeze and tune embeddings!

Competition: quora-insincere-questions-classification
Rank: #25
Source: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80542

Hi all - I tackled this competition in R &amp; keras. Right after stage 1 docker-images got updated and I had a really bad feeling. I am relieved now that all worked out!

## Preprocessing:
I did some basic preprocessing (replacing common typos and separating special characters) – nothing special here (<a href="https://www.kaggle.com/springmanndaniel/preprocessing-in-r">link to preproc kernel</a>)

## Embeddings
I combined R and Python (Reticulate) to load and merge (GloVe + Para) pretrained embeddings. This way I could save some time. (<a href="https://www.kaggle.com/springmanndaniel/combine-r-and-python-to-load-embeddings">link to embedding kernel</a>)
Vocabulary was built on training-data only (196090 words).
All words that did not appear in GloVe/ Para were replaced by zeroes. Towards the end of each models (last epoch) training phase I turned the embedding layer to trainable (see **Boosting**). This way each model overfitted a little bit + the model created some representation for missing words. 

## Keras Model
I used a single model architecture and trained it on 6 folds. The final ensemble was a simple average of the six runs. 
The model was a mix of LSTM, Convolutions and fully connected Layers.  (<a href="https://www.kaggle.com/springmanndaniel/25th-place-solution?scriptVersionId=10255060">link to model kernel</a>)
- Epochs: **4**
- Learning-rate: **0.003, 0.003, 0.003, 0.001**
- Batch-size: **512x2** on epoch 1-3 and **512x1.5** on epoch 4
- Input sequence length: **60**



**CV SCORE: ~0.6977**  
**Fold: 1** Val F1 Score: **0.695** Val Loss: 0.0954 best thresh: 0.4 (unknown words = 51432)

**Fold: 2** Val F1 Score: **0.698** Val Loss: 0.0942 best thresh: 0.38 (unknown words = 7871)

**Fold: 3** Val F1 Score: **0.695** Val Loss: 0.0952 best thresh: 0.38 (unknown words = 20)

**Fold: 4** Val F1 Score: **0.695** Val Loss: 0.093 best thresh: 0.36 (unknown words = 20)

**Fold: 5** Val F1 Score: **0.701** Val Loss: 0.0917 best thresh: 0.41 (unknown words = 20)

**Fold: 6** Val F1 Score: **0.700** Val Loss: 0.0943 best thresh: 0.36 (unknown words = 20)


## Threshold
I calculated the threshold based on validation data. 

## Boost
What boosted my model most was unfreezing embeddings towards the end of each run and updating unknown words by their newly learned representations so that subsequent models  could utilize more words for training.
This helped because each of the 6 models overfitted slightly - which added more diversity to the final ensemble &amp; helped the model to deal with unknown words.

It looks like this:


Cheers
dan

_____
edit:
- fixed typos
- added keras network
