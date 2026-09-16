# 1st Place Solution

Competition: predict-student-performance-from-game-play
Rank: #1
Source: https://www.kaggle.com/c/predict-student-performance-from-game-play/discussion/420217

Unbelievable to write this!

# Thanks!

As it is the usage, we first **thank the host and Kaggle**. These are special thanks because you and us have had a special link in this competition as we gave you more work by reporting data leaks. No doubt you tried to do your best. You are right to animate this community and to trust in it. You are part of it. Please take care of this community that is able to build so much together by sharing. As all of us you have made mistakes and we hope you will learn from them.

We also want to **thank all of you**, Kagglers. We love and are grateful to be part of our group/community. Thanks for sharing and for the collective learning experience.

# Context

- Business context: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/overview,
- Data context: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/data.

# Overview of the Approach

Our solution is essentially a blend of a XGBoost and a NN models. Both heavily rely on duration that appeared to be a powerful leverage. Time was aggregated in different ways and combined with counts for the GBDT while it is transformed via a custom TimeEmbedding block based on 1D convolutions that produce a representation combined with user event representations for the NN.
Robustness and efficiency founded our work. XGBoost models were validated on 10 bags of 5 folds and features incorporated only if the mean of the CV of these 10 bags was greater than the level of noise we quantified while we opted for a majority/consensus strategy to build the NN, i.e. validate choices only if 4 of 5 folds were improved. The 3rd place of the efficiency LB was achieved with a lightweight NN accelerated via TF Lite.

# Details of the submission

## Code

After publishing this write-up we decided to open our code: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/420332.
It is composed by several parts: [how to train the XGBoost models](https://www.kaggle.com/code/pdnartreb/pspfgp-1st-place-gbdt-training), how to [pretrain](https://www.kaggle.com/code/pdnartreb/pspfgp-1st-place-nn-pretraining) and [train](https://www.kaggle.com/code/pdnartreb/pspfgp-1st-place-nn-training) the NN models and the [inference notebook](https://www.kaggle.com/code/pdnartreb/pspfgp-1st-place-inference) used to win this competition.

## Data

Looking at the 1st data released showed that there aren't a lot of sessions so not a lot of sequences. Moreover these are long sequences. This is not ideal for a deep learning approach.  
Exploring the Field Day Lab research instructed that the Jo Wilder application was built to help learning to read and that way more than 11,500 learners had played this game.  
These 2 ideas led to search for a bigger dataset. In 1 Google search and 3 clicks we came up to the open data portal (https://fielddaylab.wisc.edu/opengamedata/) which contains a lot of sessions. 1 hour and 3 bash commands latter we knew that the train set was in part in the open data. So we took a week to **build a pipeline that extracts 98 % of the sessions of the train set perfectly and with minor errors for the last 2 %**. Our data are even better than the comp data because we knew before the host confirmation that for the sessions with 2 games the target was skewed (0 if wrong in 1 of the 2 games when we aim at predicting the responses for the 1st game). It seems that fixing these targets can bring a significant boost up to +0.002.

We took 1 more week to build a GBDT/XGBoost baseline that would have scored top 10 given the CV score, with the use of the supplemental data (~20,000 sessions) that gave +0.003/0.004 at that time. As we simulated the API locally (see after), we used some training sessions to infer and noticed that it scored 0.718. We were hoping that the LB sessions were not part of the open data portal but our 1st submission, LB 0.708, immediately showed to us that we had rebuilt about a half of the data and especially the targets in the public LB, because 0.708 = (0.698 + 0.718) / 2. The host and Kaggle have been immediately informed. You know what happened next (https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/415820).  
After the release of the LB data we measured that we perfectly rebuilt ~7000 sessions over the ~11,500 of the LB data.

We spent the first month exploring the data until we understood/knew it pretty well. For example we even reconstituted sessions for what might be schools (several games on 1 IP session), extracted every single session with at least 1 answer, ...

After the update we made a first submission that scored 0.72. This was shocking because this meant that some leaked data were remaining. A few days later we noticed that the open data was not totally similar with the state we found it 1 month before. A file was missing. So we returned to the host and Kaggle to give them more work.

**This process/work led us to perfectly understand the data model** (that changed since the 1st release of the game). This also allowed us to deeply understand the data itself.

Note that we only used the sessions for which we had responses to all questions of the 2 1st level groups. 1) This is more consistant with the sessions we want to predict (game from the beginning to the end) and 2) this approach preserves performance (vs all data) while reducing training time.

Our dataset is constituted by **37323 complete sessions (23562 comp + logs) in a total of 66376 sessions**.

The supplemental data (that we fully added 1 month ago) gave us consistently **CV +0.002**.

## Model

Our solution is mainly an ensemble of GBDT + NN models.

### Trust your validation

We think that **the main reason of the robustness of our solution is that we only relied on CV** for decision making. No choice had been made on LB.

Probing showed to us that the private test consists in the 1st 1450/1500 sessions served by the API. This is a small set. In our experiments 5,000 sessions is the minimum to guarantee a stable CV/LB alignment. A set less than 2,000 is very noisy so **robustness was the way to go**.

We **only added features that improved the CV for sure**. This is not easy to delete features that you believe in but this is needed as science is not a matter of belief. There are several ways to do so: for example monitor all folds in a CV (and accept only on majority or consensus), monitor several bags (composition of CV to not overfit validation), ...

For the GBDT approach, we mainly validated on the mean of 10 bags (we defined a bag as a composition of the folds). As we estimated the noise to be ~0.0003, only improvements greater than the noise have been considered. For the NN as we needed to iterate quicker we only used a single bag and only incorporated > 0.0003 overall improvements with at least 3 or 4 (over 5) folds improved.

### Metric

We experimented a lot on finding a threshold by question but found that this approach is less robust than a single threshold. We mainly used 0.625 as global threshold despite our highest LB scores that were obtained with a threshold per question.

### GBDT

We prototyped a baseline with **XGBoost because of the structured/tabular nature of the data**. The feature engineering process is interesting to understand what is predictive and to understand the causation, i.e. how the features or decision criteria that enable to predict correctly.

Generally speaking we followed 3 ways to build features: **business knowledge**, our **intuition** playing the game and a meticulous **exploration of the data**.
Business knowledge refers to using expert knowledge. Reading the papers of the researchers that built this game allow to understand the game beyond usage. For example, Jo Wilder has been built to improve the players reading skills. So this means that the text duration should be important. These are like killer features.

We exclusively made use of Polars because of the CPU constraints and to simply learn it.
Our features (663, 1993, 3734 for each level\_group) are mainly **durations and counts for different aggregations**: how much time in a level, in a room, reading a text, interacting in some way (event type), how many events in a level_group, how many events of each type, how many events of each type in a room or a level, ...  
We also built a few notebook dedicated features: how many type of events on the notebook in a level, ...  
Despite our efforts we weren't able to extract useful information from the coordinates, the only few features of this type had been mean and std for some events in the activities (journal interactions for example).

We considered that injecting targets predicted in the previous level groups was a compression of the signal, meaning a loss of information, so we used, for each session, **all interactions from the beginning of the game/session**. This led to a +0.002 at the time of this choice.

After the API needed to order the data, we noticed that **models trained both on original order and on index order** but validated on index order (inference order) improved our scores. This leads to more variety that was needed to **improve stability and robustness**. The same goes for the composition of the validation sets: usage of several bags (composition of validation sets) based on the comp data but also on the extracted data improved our scores. We detected late that increasing the number of folds from 5 to 10 could also be leveraged.

The code for GBDT allows to switch from XGBoost to LightGBM and CatBoost with a simple variable parameter but despite the good scores (~0.001 less than XGBoost), this did not bring to ensemble so we sticked to only XGBoost.

We experimented a lot around feature selection but were unable to build a stable strategy. So instead of a top-down approach consisting in deleting useless features, we adopted a bottom-up approach choosing carefully each group of features.

Our **XGBoost models score CV ~0.7025 +/-0.0003** and blending 5 of them (the only XGBoost we still have with correct score) scores **LB 0.704**.

### NN

After achieving a good score with gradient boosting and having understood well the data we focused on deep learning.

The **first attempt was with Transformers**. The 1st results were disappointed: CV 0.685 with 2 hours / fold (as far as we can remember). Transformers are very computationally intensive. Resources: https://arxiv.org/pdf/1912.09363.pdf, https://arxiv.org/pdf/2001.08317.pdf, https://arxiv.org/pdf/1711.03905.pdf, https://arxiv.org/pdf/1907.00235.pdf, ...

We then gave a try to **Conv1D**. In one day we had a very simple model that scored as Transformers but **10x faster** allowing to iterate quicker. So we pushed this approach and could seamlessly scaled it beyond our expectations.

Difficult to share the **tens or hundreds of experimentations** needed to achieve the final solution which is both based on a simple architecture and a slightly complex training pipeline.

#### Architecture roots

We browsed the literature based on the question: how to model time in deep learning?  
This research made us come to the idea of **time-aware events** (i.e. https://proceedings.mlr.press/v126/zhang20c/zhang20c.pdf) and back to **WaveNet** (https://arxiv.org/pdf/1609.03499.pdf) because it uses **Conv1D to model long sequences with considerations on causation**.  
Other papers also inspired us: https://arxiv.org/pdf/1703.04691.pdf build on top of WaveNet paper for time series, https://idus.us.es/bitstream/handle/11441/114701/Short-Term%20Load%20Forecasting%20Using%20Encoder-Decoder%20WaveNet.pdf?sequence=1&isAllowed=y also build on top of WaveNet.  
We also have to mention the excellent work that @abaojiang shared (https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/398565 and https://www.kaggle.com/code/abaojiang/lb-0-694-tconv-with-4-features-training-part). It inspired our research and maybe successfully biased it.

Let's focus on the model of our efficiency submission that is also one of our final ensemble and which performance is nearly the same as models with a few more features.

#### Feature representations



**5 features as inputs: duration, text\_fqid, room\_fqid, fqid, event\_name + name** (this is the event type from the original data model as far as we remember). Each of these information is encoded/embedded into a vector representation (d_model = 24) to be the merged. The 4 **categorical features feed a classical Embedding layer and the duration a TimeEmbedding** which is a custom block.

Developing the GBDT solution showed that the **duration** was crucial, so we put a crucial amount of time trying to model it greatly. The TimeEmbedding layer is a composition of 4x ConvBlock which is inspired by the Transformer main block: Conv1D -> skip connection -> layer norm -> dropout.

```
class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, n_blocks, d_model, dropout_rate):
        super(TimeEmbedding, self).__init__()
        self.conv_blocks = [ConvBlock(d_model, dropout_rate=dropout_rate) for _ in range(n_blocks)]
        
    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x
```

```
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout_rate):
        super(ConvBlock, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(d_model, kernel_size=5, padding='same', activation='gelu')
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        
    def call(self, inputs):
        x = self.conv1d(inputs)
        x = x + inputs
        x = self.layer_norm(x)
        outputs = self.dropout(x)
        return outputs
```



#### Time-aware events

As said, the goal of building these representations was to model time-aware events. We considered the **categorical features as events** because they represent the user interactions with business entities of the game. We then tried to incorporate duration to make them time-awared. Our main intuition showed to be the best. It is a **simple solution based on operation priority to represent that the duration should be associated to each event before associated them together**: duration * event_1 + duration * event_2 + ... which had been factorized to duration * (event_1 + event_2 + ...).

```
class ConvNet(tf.keras.Model):
    def __init__(self, input_dims, n_outputs, d_model, n_blocks=4, name=None):
        super(ConvNet, self).__init__(name=name)
        self.input_dims = input_dims
        self.n_outputs = n_outputs
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.event_embedding = tf.keras.layers.Embedding(input_dims['event_name_name'], d_model, mask_zero=True)
        self.room_embedding = tf.keras.layers.Embedding(input_dims['room_fqid'], d_model, mask_zero=True)
        self.text_embedding = tf.keras.layers.Embedding(input_dims['text'], d_model, mask_zero=True)
        self.fqid_embedding = tf.keras.layers.Embedding(input_dims['fqid'], d_model, mask_zero=True)
        self.duration_embedding = TimeEmbedding(n_blocks=n_blocks, d_model=d_model, dropout_rate=0.2)
        self.gap = tf.keras.layers.GlobalAveragePooling1D()
        
    def call(self, inputs):
        event = self.event_embedding(inputs['event_name_name'])
        room = self.room_embedding(inputs['room_fqid'])
        text = self.text_embedding(inputs['text'])
        fqid = self.fqid_embedding(inputs['fqid'])
        duration = self.duration_embedding(inputs['duration'])
        x = duration * (event + room + text + fqid)
        outputs = self.gap(x)
        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_dims': self.input_dims,
            'n_outputs': self.n_outputs,
            'd_model': self.d_model,
            'n_blocks': self.n_blocks,
            'name': self._name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
```


The 2 representations are equivalent: either you can think time-aware events as a combination of time and sub-events or as a combination of sub-events and time.

 
#### Training pipeline

The training pipeline is not totally straight forward.

@dongyk published great schematics that can be useful to illustrate what is explained bellow: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/420217#2332166.

##### 1st step (pre-training?)



The best approach for us consists in a kind of **backbone that represents the events of a level_group**.

This backbone is trained on all the data available for this level_group (i.e. on complete + incomplete sessions). It is associated with a temporary SimpleHead optimizing BCE loss.

```
class SimpleHead(tf.keras.Model):
    def __init__(self, n_units, n_outputs, name=None):
        super(SimpleHead, self).__init__(name=name)
        self.ffs = [tf.keras.layers.Dense(units, activation='gelu') for units in n_units]
        self.out = tf.keras.layers.Dense(n_outputs, activation='sigmoid')
        
    def call(self, inputs):
        x = inputs
        for ff in self.ffs:
            x = ff(x)
        outputs = self.out(x)
        return outputs
```

This approach allows to score **CV 0.70025 +/- 0.0005**.

##### 2nd step (training?)



**The weights of each of the 3 backbones (1 by level_group) are freezed** for the 2nd level of training to speedup training but also because it is more stable and efficient. These backbones can be thought as "embedders".

During this 2nd step, **all the submodels that composed the solution were trained on all complete sessions in an end-to-end setup**. The input data are 3 sequences of the 5 features, 1 for each of the 3 level groups. Each "embedders" outputs a 24 dim-vector representation. These outputs are the inputs of a head in which enters the representation of level\_group '0-4' to predict the 3 first questions and the concatenation of the previous and the current representations for level_groups '5-12' and '13-22' to make use of all information.

Proceeding like this allows to optimize the overall performance and to monitor it based on the F1 score that is the score of the competition. This means we optimized BCE with F1 score as a metric.

Our winning submission uses a simple **MLP head** but also a **skip head** (512 -> 512 -> 512 allow it for example). **MMoE** did not improve the simplest approaches.

This approach allows to score **CV 0.70175 +/- 0.0003** which is **comparable to the GBDT solution**.

### Inference

#### Build a simulator

Early in the competition we built a simulator of the API. Doing so we never experimented any submission error. Maybe trying to keep ideas and code as simple as possible was also key to debug easily.

#### Efficiency

We invested the efficiency part of the challenge for GBDT as well as NNs.  
Using **Treelite** for XGBoost allow us to divide by 2 the execution time.  
Our deep learning models were lights: **400,000 weights** for the end-to-end model which combines every parts/sub-models. Having already used **TF Lite** we knew it could be a game changer. Converting our models led to a significant boost in inference time without any performance loss (we do not remember exactly but we think it is at least **6x faster** on our local inference simulator).  
Beginning to explore pruning as well as hard quantization showed that the performance loss would be significant (which is OK in production but not in a competition) so we sticked to a simple TF Lite conversion.

We have not leveraged what seems to be a problem in the efficiency metric. As we identified the private test sessions to be the 1450/1500 first served by the API we tried to just predict the others to check which time was used (public for public and private for private). Doing so we gain a place but choose to not use this.

Our **efficiency submission is a NN that scores public LB 0.702 and private LB 0.699 in less than 5 minutes**.

#### Ensemble

We experimented a lot of ensembling alternatives. In the end we sticked to a simple average 50/50 GBDT/NN with:

  * 2 kinds of GBDT: trained on original order + trained on index order (validated on index order that is the inference case),
  * 3 kinds of NNs: trained on original order + trained on index order with 5 or all features.
  
As our models are lightweight we were able to build a hugh ensemble: **2 x 4 x 10 folds XGBoost + 3 x 4 x 5 folds NNs**. The bottleneck for us is the 8 Go RAM constraint.

The winning submission scores **CV 0.705, public LB 0.705 and private LB 0.705**.

## Conclusion

The main achievement of our work is that it is a good solution for the researchers, learners and children that can benefit of it and we hope it will contributes to progresses for a better learning experience. Up to you guys!

Thanks if you read until here!
If you have any question do not hesitate to ask. We will do our best to respond.

## Presentation to the host

A video presentation to the host has been recorded and can be available on demand. Feel free to ask via PM.

# Sources

Below are the main sources that we used. More sources can be found in section *Details of the submission* above.

- https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/420332,
- https://fielddaylab.wisc.edu/opengamedata/,
- https://arxiv.org/pdf/1609.03499.pdf,
- https://www.tensorflow.org/lite/guide
