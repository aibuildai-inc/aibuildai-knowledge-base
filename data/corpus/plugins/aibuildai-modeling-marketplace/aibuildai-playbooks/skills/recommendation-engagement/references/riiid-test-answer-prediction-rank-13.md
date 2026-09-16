# Public/Private 13th solution (team takoi + kurupical)

Competition: riiid-test-answer-prediction
Rank: #13
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209622

First of all, thanks to Hoon Pyo (Tim) Jeon and Kaggle team for such an interesting competition.
And congratulates to all the winning teams!

The following is the team takoi + kurupical solution.

<br>
<br>

# Team takoi + kurupical Overview


# validation
@tito's validation strategy. https://www.kaggle.com/its7171/cv-strategy

<br>
<br>
# **kurupical side**
I would like to have three kaggler to thank.
@takoi for inviting me to form a team. If it weren't for you, I couldn't reach this rank!
@limerobot for sharing DSB 3rd solution. I'm beginner in transformer for time-series data, so I learned a lot from your solution!
@wangsg for sharing notebook https://www.kaggle.com/wangsg/a-self-attentive-model-for-knowledge-tracing.!
I used this notebook as a baseline and finally get 0.809 CV for single transformer.

## model


## hyper parameters
* 20epochs
* AdamW(lr=1e-3, weight_decay=0.1)
* linear_with_warmup(lr=1e-3, warmup_epoch=2)

## worked for me
* baseline (SAKT, https://www.kaggle.com/wangsg/a-self-attentive-model-for-knowledge-tracing)
* use all data (this notebook use only last 100 history per user)
* embedding concat (not add) and Linear layer after cat embedding(@limerobot DSB2019 3rd solution) (+0.03) 
* Add min(timestamp_delta//1000, 300) (+0.02)
* Add "index that user answered same content_id at last" (+0.005)
* Transformer Encoder n_layers 2 -> 4 (+0.002)
* weight_decay 0.01 -> 0.1 (+0.002)
* LIT structure in EncoderLayer (+0.002)

## not worked for me
I did over 300 experiments, and only about 20 of them were successful.

* SAINT structure (Transformer Encoder/Decoder)
* Positional Encoding
* Consider timeseries
  * timedelta.cumsum() / timedelta.sum()
  * np.log10(timedelta.cumsum()).astype(int) as category feature and embedding
  etc...
* optimizer AdaBelief, LookAhead(Adam), RAdam
* more n_layers(4 => 6), more embedding_dimention (256 => 512)
* output only the end of the sequence
* large binning for elapsed_time/timedelta (500, 1000, etc...)
* treat elapsed_time and timedelta as continuous 

<br>
<br>

# **takoi side**
I made 1 LightGBM and 8 NN models. The model that combined Transformer and LSTM had the best CV. Here is architecture and brief description.
​
## Transformer + LSTM
​
</br>
​
.jpeg?generation=1610097396888437&alt=media)

### features
I used 17 features. 15 features were computed per user_id. 2 features were computed per content_id.
#### main features
 - sum of answered correctly
 - average of answered correctly
 - sum of answered correctly for tag_user_id
 - average of answered correctly for tag_user_id
 - lag time
 - lag time of same content_id
 - previous answered correctly for the same content_id
 - distance between the same content_id
 - average of answered correctly for each content_id
 - average of lag time for each content_id
​
</br>
​
## LightGBM
I used 97 features. The following are the main features.
- sum of answered correctly
- average of answered correctly
- sum of answered correctly for tag_user_id
- average of answered correctly for tag_user_id
- lag time
- lag time of same part
- lag time of same content_id
- previous answered correctly for the same content_id
- distance between the same content_id
- Word2Vec features of content_id
- decayed features (average of answered correctly)
- number of consecutive times with the same user answer
- flag if the user answer being answered in succession matches the correct answer
- average of answered correctly for each content_id
- average of lag time for each content_id
