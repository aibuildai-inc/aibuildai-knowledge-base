# 1st place solution - 1DCNN combined with Transformer

Competition: asl-signs
Rank: #1
Source: https://www.kaggle.com/c/asl-signs/discussion/406684

First of all, I would like to express my gratitude to the Google for hosting this amazing competition. I have always been a big fan of the services, frameworks, and platforms provided by Google.(Colab, GCP, TensorFlow.. all of them are amazing). Without access to these offerings from Google, I wouldn't have been able to win this competition. I would also like to thank all the other participants who shared their ideas. In particular, I gained valuable insights from the ideas shared by @hengck23.

# TL;DR
My solution involved a combination of a 1D CNN and a Transformer, trained from scratch using all training data(competition data only), and used 4x seed ensemble for submission. and I initially started with PyTorch + GPU but later switched to TensorFlow + Colab TPU(tpuv2-8) to ensure compatibility with TensorFlow Lite.

# 1D CNN vs. Transformer?
My hypothesis was that in modeling sequential data, 
**if there is a strong inter-frame correlation, 1D CNNs would be more efficient than Transformers.**

As in my experiment, the pure 1D CNN easily outperformed the Transformer and as a result, I was able to achieve a public LB score of 0.80 using only the 1D CNN at the end. 

However, there still were roles for the Transformer, which could be used on top of the 1D CNN(we can view 1d cnn as some kind of trainable tokenizer).

# Model
```python
def get_model(max_len=64, dropout_step=0, dim=192):
    inp = tf.keras.Input((max_len,CHANNELS))
    x = tf.keras.layers.Masking(mask_value=PAD,input_shape=(max_len,CHANNELS))(inp)
    ksize = 17
    x = tf.keras.layers.Dense(dim, use_bias=False,name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95,name='stem_bn')(x)

    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(dim,expand=2)(x)

    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(dim,expand=2)(x)

    if dim == 384: #for the 4x sized model
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(dim,expand=2)(x)

        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(dim,expand=2)(x)

    x = tf.keras.layers.Dense(dim*2,activation=None,name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES,name='classifier')(x)
    return tf.keras.Model(inp, x)

```
Combining CNN and Transformer is a prevalent idea in recent state-of-the-art models(coatnet, conformer, Maxvit, nextvit…).  I started with an 192d 8-layer 1D CNN, then switched to a 192d (3+1)x2 conv-transformer structure, which yielded a +0.01 CV and LB improvement. 


The 1D CNN model employed depthwise convolution and causal padding. The Transformer used BatchNorm + Swish instead of the typical LayerNorm + GELU, due to slightly(negligible) lighter inference with the same accuracy.
single model has around 1.85M parameters.

# Masking
**Handling variable-length input correctly was very crucial for ensuring train-test consistency and efficient inference** , as we do not necessarily have to pad the short videos. During training, I used a max_len=384 with padding and truncation, while for inference, I only applied truncation. This approach provided sufficient inference speed and allowed the use of reasonably large models. To accurately  apply masking to the 1D CNN, I used causal padding to maintain the mask index. In TensorFlow, masking can be easily implemented using tf.keras.layers.Masking at the beginning of the model. Plus, It is essential to ensure that masking is accurately applied to operations like batch normalization and global average pooling which can be affected by masking.

# Regularization
1. Drop Path(stochastic depth, p=0.2)
2. high rate of Dropout (p=0.8)
3. AWP(Adversarial Weight Perturbation, with lambda = 0.2)
###### 
As we need to train the model from scratch, regularization technics played a significant role. I used drop_path(0.2, applied after each block), dropout(0.8, applied after GAP) and AWP(Adversarial Weight Perturbation, lambda=0.2) AWP and the dropout applied after epoch 15. All these methods were very crucial for preventing overfitting when training with long epochs(>300). All three methods had a significant impact on both CV and leaderboard scores, and removing any one of them led to noticeable performance drops.

# Preprocessing
```python
class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_len=MAX_LEN, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    def call(self, inputs):
        if tf.rank(inputs) == 3:
            x = inputs[None,...]
        else:
            x = inputs
        
        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1,2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5,x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2) #N,T,P,C
        std = tf_nan_std(x, center=mean, axis=[1,2], keepdims=True)
        
        x = (x - mean)/std

        if self.max_len is not None:
            x = x[:,:self.max_len]
        length = tf.shape(x)[1]
        x = x[...,:2]

        dx = tf.cond(tf.shape(x)[1]>1,lambda:tf.pad(x[:,1:] - x[:,:-1], [[0,0],[0,1],[0,0],[0,0]]),lambda:tf.zeros_like(x))

        dx2 = tf.cond(tf.shape(x)[1]>2,lambda:tf.pad(x[:,2:] - x[:,:-2], [[0,0],[0,2],[0,0],[0,0]]),lambda:tf.zeros_like(x))

        x = tf.concat([
            tf.reshape(x, (-1,length,2*len(self.point_landmarks))),
            tf.reshape(dx, (-1,length,2*len(self.point_landmarks))),
            tf.reshape(dx2, (-1,length,2*len(self.point_landmarks))),
        ], axis = -1)
        
        x = tf.where(tf.math.is_nan(x),tf.constant(0.,x.dtype),x)
        
        return x
```
I used left-right hand, eye, nose, and lips landmarks. For normalization, I used the 17th landmark located in the nose as a reference point, since it is usually located close to the center ([0.5, 0.5]). I used motion feature of lag1 x[:1] - x[1:], and lag2 x[:2] - x[2:](lag > 2 did not help much).

#Augmentation

- temporal augmentation
1. Random resample (0.5x ~ 1.5x to original length)
2. Random masking

- Spatial augmentation
1. hflip
2. Random Affine(Scale, shift, rotate, shear)
3. Random Cutout

#Training

- Epoch = 400
- Lr = 5e-4 * num_replicas = 4e-3
- Schedule = CosineDecay with no warmup
- Optimizer = RAdam with Lookahead(better than AdamW with optimal parameters)
- Loss = CCE with label smoothing=0.1 or just plain CCE
###### 
final single model result
CV(participant split 5fold): 0.80
public LB: 0.80
private LB: 0.88
###### 
Training takes around 4 hours with colab TPUv2-8.

Single model CV was around 0.80 with participant split(5fold) at the end. I ensemble 4 different seed(with some minor differences in training configurations) for the final model and got LB 0.81. By the way, I could see slightly worse score when I submitted 4x size(384d, 16layers) single model with the same settings. I think it can achieve same or better score with better configurations.

#Tried but not worked
- GCNs
- More Complex augmentations: augmentation based on angle and the distance between the landmarks, grid distortion on temporal, spatial axis, etc.
- CutMix, MixUp: Not worked. Main problem was how to define new label with two different length of inputs. I could not find the correct way to implement it.
- Knowledge Distillation: I tried to use single 4x sized model and distill it with 4x seed 4x sized model, but I could not manage it to work due to lack of time.
###### 
I'm always amazed by the fact that we did try each other’s methods, but we came up with different result. I also gave the 2D-CNN(similar to @kolyaforrat 's brilliant solution) and pure transformer approaches a try, but due to their initial weak performance and my confidence in my own hypothesis, I didn't dig any deeper. Seeing other teams succeed with the ideas I had trouble with, through their skill and hard work, has been truly inspiring. I've learned a lot through this competition, and I'm grateful for the experience. Thank you all!!! :)


Edit: I made my code public, check https://www.kaggle.com/competitions/asl-signs/discussion/406978
