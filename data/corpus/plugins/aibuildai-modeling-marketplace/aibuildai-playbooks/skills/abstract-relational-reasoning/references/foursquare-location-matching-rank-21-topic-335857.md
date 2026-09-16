# [21st place solution]

Competition: foursquare-location-matching
Rank: #21
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/335857

First of all thanks to Kaggle and the host for this competition. I enjoyed it quite a bit as this competition needed a lot of memory management (OOM is painful), NLP and GBDT related pipelines.

It was fun working with @locbaop who put in lots of hard work in this. 

Our solution is reasonably simple:

**Pre-process**
We preprocess the text by Pykakashi for JP text and unidecode+lower for other countries. 

**Blocking**
At the start I was mostly iterating using distance measure, and that kept us in 84-85 LB range. Then we tried TFIDF - 3 ngrams+char_analyzer and it immediately gave boost.

Most of our effort was spend on pretraining the SBERT models, in total we trained **5 SBERT models** and used the concat embeddings in KNN for blocking. We could get a MAX IOU of > 99 with concatenation of `paraphrase-xlm-r-multilingual-v1` & `msmarco-bert-base-dot-v5` embeddings. We train SBERT for 1 epoch on full data.

In total we created 20 neighs per ID.

**Binary Classification Features & Model**
LightGBM model with heavy regularization.

We considered name, address, category, and  another one based on ceil and floor of latitude/longitude, like below

`df['text'] = df['name'] + '[SEP]' + df['address'] + '[SEP]' + df['categories']+ '[SEP]'+ \
        lat_int_floor + '[SEP]' +long_int_floor + '[SEP]'+ lat_int_ceil + '[SEP]' +long_int_ceil`

☝️ This is also the text on which SBERT is trained. 

We used embedding similarity for all 5 SBERT models in train. Giving us local CV on 20% valid data ~94.3.

I also added counts as a last try and saw immediate boost to local logloss and CV improving to 94.5. Unfortunately I didn't have enough time to do full training and submit.

**Post Processing**
 Same as public notebooks.

**Inference**
We used inference forest from cuML to do infer, I think there is some accuracy drop due to lower casting (float64 to float32), but I was really impressed with cuML's abilities on GPU. Will explore this more.

**What did not work**
- Couldn't get mDebertaV3 to work, lots of NAN weights in model state dict (lol)
- Adding City, State for SBERT training
- XGB didn't work for us

Thats it, goes without saying we couldn't figure out the leak! 

Thanks,
