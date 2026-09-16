# 1st place solution - external data, teacher/student and sentence transformers

Competition: commonlitreadabilityprize
Rank: #1
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/257844

First of all, I would like to thank the competition organizers for this great competition. It was really interesting and I learned a lot. This was my first Kaggle competition and I enjoyed it very much.

Thanks also to the community. You were friendly, shared interesting ideas in discussions and notebooks and I was able to learn so much from you.

Special thanks to the following users for providing helpful bits and pieces that I used in my submission:

- Thanks @teeyee314 for https://www.kaggle.com/teeyee314/readability-url-scrape . I used this data and it saved me a lot of time and effort because it was so well prepared.

- Thanks to @maunish @rhtsingh @andretugan I used this notebook for pseudo-labeling https://www.kaggle.com/andretugan/commonlit-two-models and it greatly improved my score.

**TLDR Solution**
I made a large collection of external data. I used sentence bert (Reimers and Gurevych 2019 - https://github.com/UKPLab/sentence-transformers) to select for each excerpt in the train set the 5 most similar snippets in the external data collection. I then used a roberta-base (later on I used a roberta-base/-large ensemble) model trained on the original training data to label the selected data. I then filtered the data on standard error of each sample to keep roughly the same distribution as in the train set.

Then, I trained some different models on the pseudo-labeled data for 1-4 epochs. After training on the pseudo-labeled data, I trained each model on the original training set. Models used in my final submission were albert-xxlarge, deberta-large, roberta-large and electra-large. I used ridge regression to ensemble these models.


**Detailed solution**

**Background**

This is my first Kaggle competition and so far, my day job did not involve any practical work in data science or machine learning. I am working with data scientists in the field of NLP on a daily basis but in my role as Head of Product I do not code. When thinking about this competition, I knew from the start that I wouldn't be able to compete on sophisticated model design or extensive hyperparameter tuning. I decided to make smart use of external data instead.

I had read some papers which seemed useful in this competition and I decided to implement an approach similar to these papers.

The excellent work by Nils Reimers and the UKPLab was certainly a foundation in my solution. Their work on sentence embeddings allowed me to select data that was relevant to this competition. You can read up on sentence transformers in:

[Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

One of their papers that inspired my solution uses unlabeled external data and pseudo-labeling to improve sentence embedding performance. They use a small labeled dataset to train a cross encoder, retrieve pairs of unlabeled external data using BM-25 or Sentence BERT and label these pairs using the cross encoder trained on the gold dataset. They then proceed to train a Sentence BERT using the gold dataset and the pseudo-labeled data. See: [Nandan Thakur, Nils Reimers, Johannes Daxenberger, Iryna Gurevych. 2020. Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks](https://arxiv.org/abs/2010.08240)

Another paper that was very relevant to this competition is a paper published by Facebook AI and Stanford, where the authors use self-training to improve model performance on a wide range of NLU tasks. Here, a large external sentence bank is searched with sentence embeddings and the retrieved data is labeled by a model which was finetuned on the target data. They then use the pseudo-labeled data to train another model. See: [Jingfei Du, Edouard Grave, Beliz Gunel, Vishrav Chaudhary, Onur Celebi, Michael Auli, Ves Stoyanov, Alexis Conneau. 2020. Self-training Improves Pre-training for Natural Language Understanding](https://arxiv.org/abs/2010.02194)

I also tried to transfer the results from Google's noisy student paper to NLP. In the original paper, the authors apply their strategy to image classification, but I wanted to see if it could also be applicable in this competition. They train a model on a set of labeled images and then use this model to label unlabeled images. They then use this data to train an equally-sized or larger student model. Contrary to the solutions above, they do not stop after one round of pseudo-labeling. Instead, they use the resulting student model as the new teacher and label more data using this model. They also introduce noise in the form of data augmentation and various dropout techniques for the student. In total, they find that 3 rounds of teacher/student labeling yield the best results. I tried to run multiple rounds with my approach and I also tried to introduce noise in the form of backtranslation and word replacements (predicting MASK tokens), I also added dropout for my student. However, while multiple rounds of pseudo-labeling continuously improved my CV, my LB score got worse. I think I was overfitting to my evaluation folds because I was evaluating so often. See: [Qizhe Xie, Minh-Thang Luong, Eduard Hovy, Quoc V. Le. 2019. Self-training with Noisy Student improves ImageNet classification.](https://arxiv.org/abs/1911.04252)

**Training process**

*Basics:*

- I used 6-fold crossvalidation or bootstrapping for the models used in my final submission
- models trained were albert-xxlarge, deberta-large, roberta-large, electra-large and roberta-base

*External data selection:*

- I compiled a corpus of texts that seemed relevant to this competition (simplewiki, wikipedia, bookcorpus, ...)
- I made text snippets from external data that had roughly the same length as the data in the train set
- for each excerpt in the train set, I used this model https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2 to generate sentence embeddings and retrieve the five text snippets which had the highest cosine similarity to the original excerpt


*Pseudo-labeling:*

- I trained a roberta base model on the train set and used my best model to label the external data that I retrieved in the first step
- later on, I used the models from this notebook https://www.kaggle.com/andretugan/commonlit-two-models to do the pseudo-labeling (good improvements in CV and LB)
- I used standard error of each original excerpt to filter the pseudo-labeled external samples. Each external sample which had a pseudo-label score that deviated more from the original excerpt than it's standard error was removed from the external data selection


*Training:*

- first, I trained a single model just on the pseudo-labeled data
- I used low learning rates (7e-6 to 1e-5)
- depending on model size, I evaluated every 10 - 600 steps
- I evaluated on the whole train set
- the best model was saved

Then:
- I used the model from the previous step and trained 6 models on 6 folds of the original train set
- low learning rates
- evaluating every 10 steps
- I also trained a single albert-xxlarge on all of the training data without evaluation for 4 epochs (albert xxlarge was very stable for me). This single model got a public LB score of 0.459 and I believe it was very important in my winning submission
- I also trained some models using bootstrap sampling instead of crossvalidation

*Ensembling:*
- I got the out of fold predictions for each model that I trained
- I then made a new 6-fold splits of the oof-samples
- I used the new split to train 6 ridge regression models
- the final submission used 2 ridge regression ensembles of different models, a bootstrapped model and the single albert-xxlarge model
- ensembles and other models were aggregated using a weighted average (the weights for each ensemble were chosen by feel and public LB score)


**Bonus**

I also trained a completely different model using sentence transformers. For this model, I generated pairs of excerpts from the original data and the pseudo-labeled data. I then used the difference between the scores of each pair to generate labels for the difficulty distance between these pairs. I then trained a sentence embedding model using these pairs. For inference, I embedded both the train set and the test set. Then, I used the sentence embedding model to find for each test sample the 20 most similar (actually most similar in difficulty) samples from the train set. I then used the score of the selected train samples and the cosine similarity as features for another regressor (Ridge or BayesianRidge). My best model based on this sentence embedding approach scored 0.447 on public and private LB (sentence embeddings + other models). Using only sentence embeddings I got 0.456 on public LB and 0.463 on private LB.


**EDIT**
Here is the link to the code that I used for training: https://github.com/mathislucka/kaggle_clrp_1st_place_solution
