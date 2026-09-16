# [18th place solution] Transformer + bpp conv + relative position bias

Competition: stanford-ribonanza-rna-folding
Rank: #18
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460383

Thanks to Kaggle and the hosts for organizing this competition. This competition was a good experience for me.

**Features**

I only used RNA sequence and bpp matrix provided by host. I tried other features, but it didn't have much effect.

**Model**

> 
        x1 = rna_sequence
        x2 = bpp_matrix
        x = tf.keras.layers.Embedding(num_vocab, hidden_dim, mask_zero=True)(x1)
        x2 = tf.expand_dims(x2, axis=1)
        x2 = tf.keras.layers.Conv2D(16, 16, activation='gelu', padding='same', data_format='channels_last')(x2)
        pos = RelativePositionBias(16, 32, 128)(x, x)
        for i in range(12):
            x = rel_transformer_block(hidden_dim, hidden_dim*4, head=16, drop_rate=0.2, dtype=dtype)([x, x2, pos])
        x = tf.keras.layers.Dense(2)(x)

**Training**

- Epoch : 80
- Batch size : 128
- learning rate : 1e-3, with Cosine Decay
- Optimizer : Ranger
- loss : mae loss
- only single model

**Mystery**

When I added the conv2d layer to my model, the learning speed slowed down noticeably. Does anyone know why?
