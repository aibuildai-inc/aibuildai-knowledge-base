# 🥈27th Place Solution🥈

Competition: asl-fingerspelling
Rank: #27
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434388

# TLDR
This competition is very similar to speech recognition. Therefore, I attempted new ideas focusing on techniques used in speech recognition. Among them, the most effective ones were **conformer architecture**,  **interCTC technique**, **strong augmentation** and **nomasking**.
<br>

# Data Processing
Out of the provided 543 landmarks, I utilized 42 hand data, 33 pose data, and 40 lip data. Input features were constructed by concatenating xy coordinates (drop z) and motion (xy[1:] - xy[:-1]).

 * preprocess

        def pre_process0(x):
          x = x[:args.max_frame]
          x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
          n_frames = tf.shape(x)[0]

          lhand = tf.transpose(tf.reshape(x[:, 0:63], (n_frames, 3, args.n_hand_landmarks)), (0, 2, 1))
          rhand = tf.transpose(tf.reshape(x[:, 63:126], (n_frames, 3, args.n_hand_landmarks)), (0, 2, 1))
          pose = tf.transpose(tf.reshape(x[:, 126:225], (n_frames, 3, args.n_pose_landmarks)), (0, 2, 1))
          face = tf.transpose(tf.reshape(x[:, 225:345], (n_frames, 3, args.n_face_landmarks)), (0, 2, 1))

          x = tf.concat([
              lhand,
              rhand,
              pose,
              face
              ], axis=1)

          x = x[:, :, :2]
          return x

        def decode_fn(record_bytes, augment=False):
          schema = {
              'coordinates': tf.io.VarLenFeature(tf.float32),
              'phrase': tf.io.VarLenFeature(tf.int64)
          }
          x = tf.io.parse_single_example(record_bytes, schema)

          coordinates = tf.reshape(tf.sparse.to_dense(x["coordinates"]), (-1, args.input_dim))
          phrase = tf.sparse.to_dense(x["phrase"])

          if augment:
            coordinates, phrase = augment_fn(coordinates, phrase)

          dx = tf.cond(tf.shape(coordinates)[0]>1,lambda:tf.pad(coordinates[1:] - coordinates[:-1], [[0,1],[0,0]]),lambda:tf.zeros_like(coordinates))
          coordinates = tf.concat([coordinates, dx], axis=-1)

          return coordinates, phrase

<br>

# Augmentation
Three types of augmentations were applied, each of which significantly influenced CV.

* flip hand(CV improvement : ~0.003)
        
        # flip hand
        def flip_hand(video):
          video = tf.reshape(video, shape=(-1, args.n_landmarks, 2))
          hands = video[:, :int(2 * args.n_hand_landmarks)]
          other = video[:, int(2 * args.n_hand_landmarks):]

          lhand = hands[:, :args.n_hand_landmarks]
          rhand = hands[:, args.n_hand_landmarks:]

          lhand_x, rhand_x = lhand[:, :, 0], rhand[:, :, 0]
          lhand_x = tf.negative(lhand_x) + 2 * tf.reduce_mean(lhand_x, axis=1, keepdims=True)
          rhand_x = tf.negative(rhand_x) + 2 * tf.reduce_mean(rhand_x, axis=1, keepdims=True)

          lhand = tf.concat([tf.expand_dims(lhand_x, axis=-1), lhand[:, :, 1:]], axis=-1)
          rhand = tf.concat([tf.expand_dims(rhand_x, axis=-1), rhand[:, :, 1:]], axis=-1)

          flipped_hands = tf.concat([rhand, lhand, other], axis=1)
          flipped_hands = tf.reshape(flipped_hands, shape=(-1, args.input_dim))
          return flipped_hands

* flip video and phrase(CV improvement : ~0.005)
          
        # flip video and phrase
        def reverse_frames(x, y):
          x = x[::-1]
          y = y[::-1]
          return x, y

* concat 2 videos and 2 phrases(CV improvement : ~0.01)
        
        # concat 2 videos and 2 phrases
        def cat_augment(inputs, inputs2):
          x, y = inputs
          x2, y2 = inputs2

          x_shape = tf.shape(x)
          x2_shape = tf.shape(x2)

          should_concat = tf.random.uniform(()) < 0.5

          x_condition = should_concat & (x_shape[0] + x2_shape[0] < args.max_frame)

          x = tf.cond(x_condition, lambda: tf.concat([x, x2], axis=0), lambda: x)
          y = tf.cond(x_condition, lambda: tf.concat([y, y2], axis=0), lambda: y)

          return x, y

<br>

# Model
The model was constructed utilizing the architectures proposed in Conformer.  At first, I employed the Conformer code from https://github.com/TensorSpeech/TensorFlowASR. However, by making specific modifications to the code utilized by @hoyso48 , I successfully implemented the Conformer architecture. This implementation led to an enhancement of 0.008 in the LB score. This improvement is likely due to the removal of residual blocks between Conformer blocks, allowing for the integration of more layers and resulting in improved robustness against overfitting.  Furthermore, the incorporation of the interCTC([paper](https://arxiv.org/abs/2102.03216)) contributed to an LB score(~0.005).


Furthermore, as the model's output was computed without masking and a maximum length of 384 was employed for the CTC loss "input-length' variable, using a larger kernel size(=31) effectively conveyed frame information across the 384-length range, resulting in improved performance.

* The overall structure of the model is as follows.

        def get_model(max_len=384, dim=160,ksize=31,drop_rate=0.1, num_layers=16):
          NUM_CLASSES=63
          PAD=0
          inp = tf.keras.Input((None,2*230))
          x = inp
          x = tf.keras.layers.Dense(dim, use_bias=False,name='stem_conv')(x)
          x = tf.keras.layers.BatchNormalization(momentum=0.95,name='stem_bn')(x)

          xs = []
          for i in range(num_layers):
              x = TransformerBlock(dim,expand=2)(x)
              x = Conv1DBlock(dim,ksize,drop_rate=drop_rate)(x)
              xs.append(x)

          classifier = tf.keras.layers.Dense(NUM_CLASSES,name='classifier')

          x1 = tf.keras.layers.Dropout(0.2)(xs[-8])
          x1 = classifier(x1)

          x2 = tf.keras.layers.Dropout(0.2)(xs[-1])
          x2 = classifier(x2)
          return x1, x2
<br>

# Training

* Scheduler : lr_warmup_cosine_decay 
* Warmup Ratio : 0.1 
* Optimizer : AdamW 
* Weight Decay : 0.01
* Epoch : 300(last 100 epoch for only train data(not suppl))
* Learning Rate : 1e-3 
* Loss Function : CTCLoss

<br>

# Didn't Work
* masking 
* adding distance features
* mlm pretraining 
<br>

# Code
https://github.com/siwooyong/Google-American-Sign-Language-Fingerspelling-Recognition
<br>
