# 3rd place solution

Competition: foursquare-location-matching
Rank: #3
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/338112

Congrats to all competitors, and although it had a strange ending, I am confident that some techniques of this competition can be re-used in future.

### Summary

My solution is based on a two-stage approach utilizing NLP deep learning models. The first-stage model is a metric learning model with ArcFace loss trained on the original input data predicting the point-of-interest of a record and consequently putting similar records into a similar embedding space. The second stage additionally trains a bi-encoder NLP model based on the proposed candidates from the first stage. The final sub is a blend of both first- and second-stage models.

### First stage: ArcFace

In the first stage, I fit ArcFace model on most of the input columns separated by SEP tokens. I directly feed the longitude and latitude into the string but split it up every 3rd digit so that the tokenizers can work better with it. A sample record looks like follows:

`<s> Tesco HQ</s></s> Offices</s></s> 18 963 587</s></s> 47 450 562</s></s> 2040</s></s> Budaörs</s></s> Kinizsi út 1-3.</s>`

If a field was NAN, I just filled in an empty string, and no special tokens or other alterations were helpful.

It was still quite tricky to get ArcFace models to work, and some effort was necessary to get the training to stabilize and converge. I only rely on xlm-roberta-large models here. In the beginning I experimented with Triplet-like models, which were easier to get going, but ArcFace was clearly superior after making these models work.

I then calculate all-pairs similarity and cut-off below a certain threshold. All pairs above the threshold make it into the second stage and we also keep the predicted probability for blending.

### Second stage: Bi-Encoder

For training, I use in-sample results from the first stage as candidate pairs. I use various bi-encoder models to train for TP/FP of pairs. A record looks like follows:

`<s> Tesco HQ</s></s> Novo Nordisk</s></s> 18 963 587</s></s> 12 449 657</s></s> 47 450 562</s></s> 55 751 928</s></s> Offices</s></s> Offices</s></s> HU</s></s> DK</s></s> Pest</s></s> Bagsværd</s></s> Budaörs</s></s> Bagsværd</s></s> 2040</s></s> 2880</s></s> Kinizsi út 1-3.</s></s> Novo Allé</s></s></s></s></s></s></s></s> `

So pasting each column of both records next to each other and then training a bi-encoder model. I always added all TPs to the training data, and then increasingly added more FPs if time allowed.

### Cross-Validation

The test set in this competition contained ~600k records, and the whole training data contained approximately twice the data. Retrieval/Recognition type problems are always heavily dependent on the list of candidates to choose from, so naturally having the possibility of more unknown candidates will deteriorate the score. 

Hence, the most reasonable CV setup was to split out 600k records with unique points of interest as a validation set, and train models on the remaining data. When doing so I saw good correlation between CV and LB however there is one big issue with this approach, namely that you miss training models on half the data. Consequently, I decided around six weeks before competition deadline to only evaluate on public LB which from my point of view was the best set to evaluate my models due to the following reasons:

- There was enough time left for me to evaluate regularly on LB
- I saw good correlation before between CV and LB
- It allowed me to train all models directly on full data
- It included the full test data as candidates (all 100%)
- On CV, the score was stable on subsets, so any random 100k subset scored nearly the same as the full 600k set (keeping all 600k candidates of course).

Unfortunately, I only learned after competition ended, that the test data has a leak that was already discussed thoroughly in the forums. So I definitely biased my model choices towards that leak (e.g., longer training, more overfitting - which definitely works very well with my data and model setup). However, I still made sure that longer training would not hurt my CV score. Yet, I stopped trying to explicitly optimize my CV score, and in retrospective this is a bit unfortunate as I am fully convinced that my general approach could further be improved by working towards a different direction. 

### Blend and discussion

I blend both probabilities from first and second stage to get the final prediction and cut-off by a certain threshold. I applied some minor QE post processing, but not much was helping additionally.

I personally never suspected the leak, due to the dataset description and nature of the problem, so I also never actively searched for it. That said, I also noticed that longer training leads to better LB, now in retrospective I understand why. I was rather suspecting that the models learn / overfit the individual datasets (there are different synthetic data sources) which can help the matching tremendously as it reduces the possible candidates.
The [post-analysis](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/338035) revealed that my solution also scores very competitively on non-overlapping data which shows how well my pipeline can generalize to both scenarios. 

I still learned a lot in this competition and am very happy to have solved it with deep learning only. It is most fun to me personally, and I am sure this general approach will come in handy again in the future.
