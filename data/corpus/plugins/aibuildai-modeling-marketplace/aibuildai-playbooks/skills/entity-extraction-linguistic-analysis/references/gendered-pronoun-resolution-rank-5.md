# 5th Placed Solution

Competition: gendered-pronoun-resolution
Rank: #5
Source: https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90668#latest-526599

##Summary

My model was based on fine-tuning BERT-large with rudimentary output layer and no additional statistical features. The code in `run_classifier.py()` provided with BERT was modified to achieve this. I will post the [full code here](https://github.com/kenkrige/BERT-Fine-tune-for-GAP) once I've neatened it up, but for now an overview of the main elements.

For the input layer, I sorted the 3 char offsets and stored the corresponding permutation of P, A, B and split the text into 4 segments at these offsets. Each segment was Wordpiece tokenized and the token segments were truncated to match the required sequence length. The choice of truncating algorithim made a big difference to the results. The token offsets were calculated as the cumulative sum of tokens in each segment. The token offsets were sorted back into P, A, B order and used to make binary mask “features” for the positions of P, A, B.

The output from BERT was masked to retrieve the 1024 dimensional vectors for P, A, B. Three new “probability embeddings” were computed as (element-wise multiplication):

1. PA 
2. PB
3. AB – PP 

The idea of the first two was to represent the similarity between P&amp;A or P&amp;B as abstract vector embeddings. For ‘Neither” the abstract embedding in 3 is supposed to represent the extent to which A&amp;B are similar to each other, but differ from P.  These embeddings (1024 dim vectors) were then reduced to a scalar probability by a trainable (1024, 1) tensor. The same tensor was used for the probs of A and B to prevent the model learning from whether “True” or “False” appeared first, as my own pseudo-data (not used in the end) was somewhat unbalanced in this regard. A seperate trainable tensor was applied to (AB – PP).

This architecture on a 64 token sequence gave a stage 1 score (corrected gap data) of 0.30 to 0.38 (single model), which ensembled to 0.275 by simple averaging of the predictions from 6 models. I assume the reason for the range of results has to do with the architecture being a bit unstable with respect to random initialisation.

##Input

Within `run_classifier.py()`, changes were made to the input methods to suite the GAP data. A detailed explanation is done via [this](https://www.kaggle.com/kenkrige/bert-example-prep) short kernel, which illustrates how a single example is prepared for the input layer. Pay particular attention to the way text was truncated to fit the chosen maximum token sequence. Changes to this algorithm made a big difference to results. I think that doing this truncating at a sentence level might bring further gains.

##Hidden Layers

Vanilla BERT-large.

##Output

Unfortunately, the 1024 dimensional vectors make it impractical to do a similar demonstration of the output. Instead I will give a short explanation of some of the code.

Firstly, the output was extracted and masked to get the P, A, B embeddings.

```
all_out = model.get_sequence_output()
P = tf.boolean_mask(all_out, P_mask)
A = tf.boolean_mask(all_out, A_mask)
B = tf.boolean_mask(all_out, B_mask)
```

This worked fine on a GPU, but `tf.boolean_mask()` is not implemented for TPU, which I stepped up to for BERT-large. It took a good few hours to figure out a messy workaround, but it did the job.

The probability embeddings were then calculated as explained above.

```
PA = tf.multiply(P, A)
PB = tf.multiply(P, B)
PP = tf.multiply(P, P)
AB = tf.multiply(A, B)
N = tf.subtract(PP, AB)
```

And finally reduced to probabilities by trainable tensors and a softmax.

```
AB_weights = tf.get_variable(
      "AB_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

N_weights = tf.get_variable(
      "N_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

A_out = tf.matmul(PA, AB_weights, transpose_b=True)
B_out = tf.matmul(PB, AB_weights, transpose_b=True)
N_out = tf.matmul(N, N_weights, transpose_b=True)

output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

logits = tf.concat([A_out, B_out, N_out], axis=1)
logits = tf.nn.bias_add(logits, output_bias)
probabilities = tf.nn.softmax(logits, axis=-1)
log_probs = tf.nn.log_softmax(logits, axis=-1)
```
That was about it. The rest of the code was pretty routine or already included in `run_classifier.py`.
