# 7th place solution

Competition: g2net-gravitational-wave-detection
Rank: #7
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275349

Preprocess:
30Hz highpass filtering
Normalized by the absolute means of individual observations
 
Augmentation:
Roll +/-500 points (p=0.5)
Scale 0.85-1.15 (p=0.5)
Multiply -1 (p=0.5)
 
# All implemented using TF trained on TPU 
#The bellow frontend is used by all models. dim=128

input = L.Input(shape=(3, 4096)) # 3 observations
x = tf.reshape(input,(-1,4096,1)) # 3 observations folded into batch
x = wavenet(x, dim, dilations=12, kernel_size=5)
x = L.Dense(dim//4)(x)
x = wavenet(x, dim, dilations=12, kernel_size=5)
x = L.Dense(dim//4)(x)
x = L.Dense(dim)(x)
x = tf.reshape(x, (-1,3,4096,dim))
x = tf.transpose(x,(0,2,3,1))
x = L.BatchNormalization()(x)
x = L.Activation('gelu')(x)
        
Model1:
Efficient B3 (size=128*128), CV 87.79-88.04
Model2:
Wavenet + GRU, CV 87.80-88.05
Model3:
Wavenet, CV 87.83-88.07
