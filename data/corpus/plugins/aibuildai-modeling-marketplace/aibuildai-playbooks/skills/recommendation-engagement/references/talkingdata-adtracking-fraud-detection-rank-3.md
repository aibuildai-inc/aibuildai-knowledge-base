# My brief summary,a mainly NN based solution(3th)

Competition: talkingdata-adtracking-fraud-detection
Rank: #3
Source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56262

Congrats to all the winners(['flowlight', 'komaki'].shuffle(),PPP is already in use) and all the kagglers who have worked hard and have learned a lot of from this competition.<br>

Thanks to TalkingData and Kaggle for such an interesting competition.<br>

Here is the summary of my solution,it is mainly based on NN models.<br>

It was very hard for me to choose among Landmark Competitions and TalkingData's,so I started them at the same time,when I found I can get decent NN models, I focus on this one,it is always interesting to solve a problem other than CV/NLP using NN models.<br>

My models mainly based on 23 features,by using these features,my single LGBM model scored 0.9817 on public LB,it is not a good one compared to other kagglers,this is the first time that I used LGBM in kaggle competition indeed,so there are a lot to learn from you!<br>

I designed NN models based on those 23 features,and prepared the features to fed them into network carefully[NA,out of vocabulary,log,scale],my NN model can reach 0.9820 on public LB.As we all know, the click delta is important,so I fed deltas of last 5 and next 5 click_times to the network and designed a model with RNN cell to find the patterns of the click series,my model can reach 0.9821 on public LB and 0.9830 on private LB.<br>

Then I designed different NN models to add diversities,they are very  simple,for example,adding some res-links to dense layers.I don’t know the single model performance of these 4 models because I judged them only by diversities.<br>

After have ensembled my NN models and LGBM models by weighted average,I can get 0.9827 on public LB and 0.9835 on private LB.<br>

Then,I predicted full set of train data on my n-fold models,it’s a little time consuming,but it’s a relatively small dataset for me when compared to other datasets I have met,I can train and predict a fold of my model in 2.x hours on a 1080i GPU.<br>

I trained second level NN models using predictions from the whole train and test dataset,and added some group by features based on IP,app-os-channel,my ensemble score improved to 0.9833 on public LB and 0.9840 on private LB,which is a huge improvement.<br>

This improvement happened at 30 hours before the competition end,I hope I could get such an improvement two days earlier,because I had no time to solve some limitations/weaknesses of my NN models,as I simulated on part of the data,I found NN models may led to a small drop(0.0003 also) on private dataset in some situation,which can explain my minor drop on private LB.So when you were talking about 0.9835 solution,I wanted to tell you don’t expect too much on my solution :) I had thought I will be very happy if I am still in top 5 when the private LB revealed<br>

Sorry for not so detailed,I feel quite sleepy now.<br>

------------------------------------Some Details------------------------------<br>
<b>The features:</b><br>
<pre>channel                                  1011
os                                        544
hour                                      472
app                                       468
ip_app_os_device_day_click_time_next_1     320
app_channel_os_mean_is_attributed         189
ip_app_mean_is_attributed                 124
ip_app_os_device_day_click_time_next_2     120
ip_os_device_count_click_id               113
ip_var_hour                                94
ip_day_hour_count_click_id                 91
ip_mean_is_attributed                      74
ip_count_click_id                          73
ip_app_os_device_day_click_time_lag1       67
app_mean_is_attributed                     67
ip_nunique_os_device                       65
ip_nunique_app                             63
ip_nunique_os                              51
ip_nunique_app_channel                     49
ip_os_device_mean_is_attributed            46
device                                     41
app_channel_os_count_click_id              37
ip_hour_mean_is_attributed                 21
</pre>

a simple GRU network:<br>
<pre>class GRU_V0a():
    def __init__(self, **kw):
        super(GRU_V0a, self).__init__(**kw)
	self.categorical=['app', 'device', 'os', 'channel', 'hour']
	self.continous=[col for col in features if col not in self.categorical]
        self.categorical_num = {
            'app': (769, 16),
            'device': (4228, 16),
            'os': (957, 16),
            'channel': (501, 8),
            'hour': (24, 8),
        }
    def build_model(self):
        categorial_inp = Input(shape=(len(self.categorical),))
        cat_embeds = []
        for idx, col in enumerate(self.categorical):
            x = Lambda(lambda x: x[:, idx,None])(categorial_inp)
            x = Embedding(self.categorical_num[col][0], self.categorical_num[col][1],input_length=1)(x)
            cat_embeds.append(x)
        embeds = concatenate(cat_embeds, axis=2)
        embeds = GaussianDropout(0.2)(embeds)
        continous_inp = Input(shape=(len(self.continous),))
        cx = Reshape([1,len(self.continous)])(continous_inp)
        x = concatenate([embeds, cx], axis=2)
        x = CuDNNGRU(128)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.20)(x)
        x = Dense(64)(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.20)(x)
        x = Dense(32)(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.05)(x)
        outp = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[categorial_inp, continous_inp], output=outp)
        print(model.summary())
        return model
</pre>
Thanks [@aharless][1] publish the code which I used as a reference,such as GaussianDropout.<br>
It's really simple? :)And the devil is in detail:<br>
To make NN model work,preprocess is very important.<br>
Fill-NA:Fill max value for delta feature.Fill mean for other feature.<br>
Log of delta features,part of count features when the values are large and all the nunique features.<br>
StandardScale<br>

It is not a tranditional RNN,the input_lenght=1 indeed,so I just used the function of CudnnGRU to get  pattern of clicks which can be expressed like the following:
<pre>zx=sigmoid(K.dot(Wz,X)
hx=tanh(K.dot(W,X)
h=zx*hx
</pre>

As we want to know the theory behind the stucture,and I have no time to prove it or write a paper,
we can have a look at the paper by Google:[Searching for Activation Functions][2] Swish:x · σ(βx), where σ(z) = (1 + exp(−z))−1,we can get some ideas to explain my setting.

As to private score estimation,it's always a interesting part of my competition :).<br>
There is not certain method to do so,I just bear in mind:<br>
Distribution variances lead to score variances.<br>
For example,in this competition,some category values in test-set are not in trainset,so I changed the same ratio of category value of validation set to values unseen in trainset.And the App19 is a very important app with high ratios of download and is imbalance in train and test-set.What's more,the ratio is differenct between the public and private set,so I tried to keep the ratio of my validatition as test set...We can also use a submission which I  believe it is stable as True label,and caculate AUC based on it,if the public and private score as close,then we can believe it too. 
<br>
Thanks for all the congrats to me ,I will upvote your comment and will not reply to everyone to save space of this page.
<br>


  [1]: https://www.kaggle.com/aharless/gpu-nn-validation-more-features/code
  [2]: http://Searching%20for%20Activation%20Functions
