# Silver - LB 0.770 - Two Lines of Code!

Competition: asl-fingerspelling
Rank: #42
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434353

Thanks Kaggle and Google for hosting a fun ASL competition.

I joined a few days ago, so I didn't have time to build my own model. Instead, I began with the best public notebook [here][2] (version 17) and attempted to improve it. After making a few changes, I boosted the LB from 0.700 to an amazing 0.770! and obtained Silver medal !!

# Change Two Lines of Code
The best public notebook is Rohith Ingilela's awesome public notebook [here][1] which was improved by Saidineshpola [here][2]. Version 17 of Saidineshpola's notebook achieves CV = 0.689 (scroll to bottom of version 17) and LB = 0.697.

That notebook uses TF Records made by Rohith Ingilela [here][3]. An easy trick to boost the performance of the public notebook is create TF Records which keep frames where hands are missing. The TF Records made by linked notebook removes all frames without hands. Instead we can use `"output two"` below and keep 50% of the frames with missing hands below to boost CV and LB. Updated notebook published [here][4].



Here is preprocess code to use when making TF Records and during inference:

    hand = tf.concat([rhand, lhand], axis=1)
    hand = tf.where(tf.math.is_nan(hand), 0.0, hand)
    mask = tf.math.not_equal(tf.reduce_sum(hand, axis=[1, 2]), 0.0)
    alternating_tensor = tf.math.equal( tf.cumsum(
        tf.ones_like( tf.reduce_sum(hand, axis=[1, 2]) ))%2, 1.0 )
    mask = tf.math.logical_or(mask, alternating_tensor)

# CTC (Connectionist Temporal Classification) Loss:
In this competition, I learned about CTC loss. This loss is amazing! It allows us to create a model with variable length input and predict variable length output in one step. So we can do things like seq2seq without waiting for a sequential decoder to decode each step. Instead we predict the entire output at once super fast! Giving the model some of the frames with hands missing helps identify duplicates and transistions:



# Time Augmentation LB +0.004!
The public notebook trains for 50 epochs, I found that training for more epochs continues to boost CV and LB score! My final submission trains for 200 epochs. Furthermore, we can add augmentation (i.e. regularization) which helps the model train longer, prevent overfitting, and generalize better.

Below we see the histogram of train data Number of Frames divided by Character Length of Target Phrase. From this plot, we see that the ratio varies a lot. Some videos have a different recorded frame rate than others and some participants sign faster than others.



What this means is that the model has a hard time transfer learning one frame rate to learn about another frame rate. We can help the model by using time augmentation. For each input sequence, we can randomly shrink frame length by 50% or enlarge 150%. This will add lots of new train data and help the model learn about different frame rates. Here is augmentation code:

    if tf.random.uniform(shape=(), minval=0, maxval=1)<0.2:
        new_height = tf.math.round( tf.random.uniform(
            shape=(), minval = tf.cast(tf.shape(lip)[0],tf.float32) / 2.0, 
            maxval = tf.cast(tf.shape(lip)[0],tf.float32) * 1.5) )
        for x in [lip, rhand, lhand, rpose, lpose]:
            x = tf.image.resize(x, (new_height, tf.shape(x)[1]) )

# Post Processing LB +0.004!
Sometimes the model doesn't make a good prediction. The shortest train data is length 3. When the model predicts length 2 or less, we know it is a bad prediction. Therefore we can replace bad predictions with the best constant length prediction. Anokas found the best constant length prediction [here][5]. We add this to our TF Lite mode with the following code:

    x = tf.cond(tf.shape(x)[0] < 3, lambda: tf.constant(
        [17, 0, 32, 12, 36, 0, 12, 32, 49, 46, 36], tf.int64), lambda: tf.identity(x))

# Solution Code
I published a Kaggle notebook [here][4] demonstrating the above 3 changes. The first four bullet points below achieve `LB = 0.763`, then time augmentation boost `+0.004` and PP boost `+0.004`. Note that the second bullet point (about batch size) just makes things faster but doesn't change the CV nor LB score. The other bullet points are the key:

* Change 2 lines of code to keep 50% missing hand frames
* Change batch size from 32 to 128 and learning rate from 1e-3 to 4e-3
* Change FRAME_LEN from 128 to 216
* Increase epochs 50 to 200
* Add time augmenation
* Add post process for preds less than 3 chars

[1]: https://www.kaggle.com/code/irohith/aslfr-ctc-based-on-prev-comp-1st-place
[2]: https://www.kaggle.com/code/saidineshpola/aslfr-ctc-based-on-prev-comp-1st-place?scriptVersionId=139048726
[3]: https://www.kaggle.com/code/irohith/aslfr-preprocess-dataset-tfrecords-mean-std
[4]: https://www.kaggle.com/code/cdeotte/2-lines-of-code-change-lb-0-760
[5]: https://www.kaggle.com/code/anokas/static-greedy-baseline-0-157-lb
