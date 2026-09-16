# 26th place solution

Competition: vsb-power-line-fault-detection
Rank: #26
Source: https://www.kaggle.com/c/vsb-power-line-fault-detection/discussion/85301#latest-497049

My solution based on kernel [VSB Competition : Base Neural Network](https://www.kaggle.com/tarunpaparaju/vsb-competition-attention-bilstm-with-features)  by [Tarun Sriranga Paparaju](https://www.kaggle.com/tarunpaparaju).  But I'm added some modification:
- I used denoising algorithm based on wavelet(bior1.3)
- I changed neural network architecture
```
def model_gru(input_shape, feat_shape, alpha):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    ifeat = Input(shape=(feat_shape[1],))
    feat = Dense(8)(ifeat)
    feat = BatchNormalization()(feat)
    feat = Activation('tanh')(feat)
    
    x = SpatialDropout1D(0.2)(inp)
    x = Bidirectional(CuDNNGRU(160, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(160, return_sequences=True))(x)
    
    attention = Attention(input_shape[1])(x)
    
    x = concatenate([attention, feat], axis=1)
    
    x = Dropout(0.8)(x)
    
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    
    x = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=[inp, ifeat], outputs=x)
    model.compile(loss=focal_loss(gamma=2, alpha=alpha), optimizer=Nadam(lr=0.003), 
metrics=[matthews_correlation])
    
    return model
```
-  For prevent overfitting I used strong dropout (SpatialDropout 0.2 before GRU and   dropout 0.8 before fully connected)
-  I used focal loss with gamma = 2 and alpha calculated based on data
```
alpha = np.sqrt((1-sum(train_y)/len(train_y))*0.8))
```
- Since the signal is periodic, I used a generator with a simple augmentation.
```
    def cyclic_shift(x, alpha=0.5):
        s = np.random.uniform(0, alpha)
        part = int(len(x)*s)
        x_ = x[:part, :]
        _x = x[-len(x)+part:, :]
        return np.concatenate([_x, x_], axis=0)
```
- Model was trained on StragtifiedKfolds(10 folds) used Nadam and CLR with triangle mode (clr 300 steps lr= (1e-4, 6*1e-6)) on  batch size 256 and 50 steps per epoch and early stoping with loss monitoring. 

Public score: 0.59499
Private score: 0.67159
