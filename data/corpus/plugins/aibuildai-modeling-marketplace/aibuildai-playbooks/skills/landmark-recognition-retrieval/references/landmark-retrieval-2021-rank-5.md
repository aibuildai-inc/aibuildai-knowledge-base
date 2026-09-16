# 5th Place Solution(Team: underfitting)

Competition: landmark-retrieval-2021
Rank: #5
Source: https://www.kaggle.com/c/landmark-retrieval-2021/discussion/275942

First of all, very much thanks to the host for holding Google Landmark Competition.
these competitions are very interesting and impressive for us.
Alsoo thanks to my teammate( @ks2019 )he contributes to our team for many work.

Our 5th solution summary is here

# Brief Summary
- Backbone + ArcFace, GLDv2 Training(4.1M)
- Post Process(WDA, bridged confidence etc....)

# Solution
## Models
Model architecture is backbone+ArcFace module+Average Pooling.
The backbone list is here.

- EfficientNet v2s 800
- EfficientNet v2m 732
- EfficientNet v2m 640
- EfficientNet v2l 720
- EfficientNet v2xl 640(training 512)

Training data is gldv2(4.1M).  in order to train large image size with high batch size, we used gradient accumulation and mixed-precision training,(above v2s)

We use Adam 15ep training with Cosine Annealing and extract 512 embedding vectors from each model. After we create concatenation vectors(2560).

our first result is KNN of GPU(cuml) method.
KNN in GPU is faster than CPU, it's a great performance.

Also, we use pre-compute embedding, it computes in local and uploads all.(it use post-process)

## Post-Process
We apply some post-process in our first result.
WDA, bridged-confidence, power KNN etc.... detail is here
https://www.kaggle.com/c/landmark-retrieval-2021/discussion/275955

## Hardware
TPU-v3 8core(GCP)

## Our code
https://www.kaggle.com/ks2019/landmark-retrieval-5th-place-inference-notebook
