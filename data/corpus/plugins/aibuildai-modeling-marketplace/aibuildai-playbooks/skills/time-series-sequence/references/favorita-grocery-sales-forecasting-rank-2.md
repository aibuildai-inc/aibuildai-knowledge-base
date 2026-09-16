# 2nd place solution overview

Competition: favorita-grocery-sales-forecasting
Rank: #2
Source: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47568

First, we want to thanks @sjv. We learned a lot from his prevous sharing.

Second, thanks to our team members to stay with me, discuss and work hard together until last minute. 

Also thanks to @CPMP, his success on WTF with CNN encouraged me to peruse breakthrough with deep learning model.

We won't introduce the basic model architecture for wavenet.  

We just post our understanding and our experience with this model. We think the model and the way to train model is naturally robust.

Feeding data with mini-batch with randomly sampled 128 sequence. Then randomly to choose the start of decode/target date. So, we could say somehow the model will see different data for each training iteration. because the total dataset is around 170000 (seq) x 365 days. My feeling is the wavenet trained in this way will be good to handle overfitting.

During the training, we set the learning rate as 0.0005 and with a learning decay.

In fact, the training loss is hard to get decrease after some iteration.

We take the approach for validation is step by step. We keep the last 16 days validation data. 

Trigger by WTF top solution,  We think the shifted unit_sales and onpromotion data could be used for capture better quarterly and yearly pattern. 

The way we feeding data gives us robust. but we think there is a little drawback. which is the prediction from model will be slightly 'overfitted' the last several hundreds or thousands of mini-batch data. So the result has a little fluctuation, not very stable. This experience also shown in Top solution in WTF. Also CPMP mentioned by different cv setting he got different performcen.

The best way we thought is averaging. In WTF top solution 30 models prediction averaging. We trained the model and after 5000 mini-batch, model will generate a prediction for every 2000 mini-batch. 

After certain iteration, choose 5 model average will give better result than single model. But different 5 or 10 model average output still have a little difference on performance from Local CV. 

So finally we choose EMA to average on prediction. The parameter used for EMA, is from LocalCV. 

New Item processing, it might only has minor impact, but because we are hanging on the edge of gold zone, so finally I suggest to do some work on new items. which is agreed by all team member to bet on it for gold medal.

After analysis of the data, we found the on the date 2015-07-01, 2015-09-01, 2015-10-01 and 2016-11-27, there are lots of items start to have unit_sales records. but the hard problem is that we do not know in test period most of new items on which day start to have unit_sale. and on which stores? By checking onpromotion information, all onpromotion with new items only on 08-30 and 08-31. by reviewing the data on 2015-10-01 or 2016-11-27, those 'old new items' is shown onpromotion on that day, not before that. So, we bet the most new items will start have unit_sales on 08-30 and 08-31. Before that there might be only a few new item was sold. We do not use model to predict to prevent overfitting, We only use some value to validate with loss calculation on those 'old new items'

We also used Max Observation value cap. etc.

Finally, thanks to our team member work together to get this very good result. 

Update, the new item prediction is not good. checked by late submission. the score got slight dropped less than 0.001 when I add new item prediction
