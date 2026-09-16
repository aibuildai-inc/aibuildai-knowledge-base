# [4th Place Solution] Conformer Encoder-Decoder Ensemble with beam search and edit_dist optimization

Competition: asl-fingerspelling
Rank: #4
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434983

Thanks to Kaggle and the organizer for running this competition! It was a quite unique challenge and even after three months of optimizing it feels like there are still so many things to improve, which is quite special in my opinion.

## TLDR

My solution is an ensemble of two encoder-decoder models. The encoder is a 12 layer adapted conformer and the decoder is a two layer regular transformer. I added various augmentations and training techniques to align the training objective with the edit_distance competition metric. For decoding I implemented a (cached) beam search for TFLite.

## Challenge and plan

The goal of the competition was to translate sign language spelling from videos that were preprocessed with human pose recognition. Submissions were made as TFLite models with a limited OPs set and evaluated using the Levenshtein Edit Distance.

These choices had a couple of implications for modelling:

- "honest" predictions are often not edit-distance-optimal, esp. when the model recognizes no characters in a phrase,
  the honest prediction "" achieves a score of 0.0 while "2 a-e -aroe" scores 0.16,
  see [Anoka's Static Greedy Baseline](https://www.kaggle.com/code/anokas/static-greedy-baseline-0-157-lb)
- TFLite models in this competition only allowed a very restrictive ops-set, which meant for example that all the native
  implementations of beam_search and similar algorithms were not supported
- there were time and size limitations to the model (40MB) causing where to "spend" your parameters becoming a major
  design decision
- the datasets contained quite a few samples where most or all data was missing

With these things in mind, I assumed "making things up" would be a significant part of good predictions and
encoder-decoder architectures seemed naturally aligned to this. Additionally having a decoder abstracts away one of the time dimensions which made developing downstream algorithms like beam search or ensembling easier. My early tests also suggested encoder-decoders to work slightly better than CTC, so I went with that architecture.

## Data

My model used 214 inputs: 21 LHand, 21 RHand, 25 Pose, 1 Nose and 40 Lips points, each using x- and y-coordinates. Data was normalized, NaNs zero-filled and the deltas to t-1 and t-2 values were used as additional Features. During training a maximum of 500 frames was used with longer sequences being resized.

I applied quite a few data augmentations:

- Flip left-right
- Resample along the time dimension
- Scale / Translate / Rotate
- Mask up to 60% of all frames (worked better than masking sequences)
- Spatial cutout (similar to [Hoyso48's 1st place solution](https://www.kaggle.com/competitions/asl-signs/discussion/406684))   <br>


All of these made a significant impact.

On top of that I transformed the tokens too. I used the fact that when replacing a single token in a phrase with a random token, the ground truth is still an edit-distance-optimal target. This change gave a quite nice boost of +0.008 (using smaller models). I also added single token deletions and insertions but they had a minor impact (if at all).

I split the data five-fold and most of my experiments used only one fold to train with smaller models due to compute restrictions.


## Model

The base of my model was a deep [Conformer](https://arxiv.org/pdf/2005.08100.pdf) encoder followed by a two-layer transformer decoder. The encoder used twelve layers with dimension of 144. The MHSA had four heads with dim-per-head of 64 and the Convolution used a kernel of size 65.
Like the original formulation the model used two macaron-style feed forwards with an expansion factor of four. I made some additional small changes, like changing the position of the BatchNorm and adding DropPath to each Submodule of the Conformer. The model used a drop rate of 0.1 almost everywhere, only before the final classifier it used 0.3. Instead of causal padding I used same padding and explicitly zeroed-out the padded parts.  
 
On the decoder side I tested many different configurations but ended up using a very slim, two layer transformer
decoder. It used four attention heads with dim-per-head of 32 and a feed forward with expansion factor of only two. This was the smallest configuration that I could train without significant performance drop off. Using a small decoder was important since the autoregressive decoding is very performance intensive.

## Training

A full training run for a single twelve layer encoder model took around two days on my local 3090. To experiment with different architectures, augmentations etc., I only trained shallower models on ~20% of the data for most of the competition.<br>

The final training used a cross entropy loss and RAdam optimizer (but AdamW with warmup worked pretty much the same) with a peak lr of 1e-3 and cosine schedule. Weight decay of 2e-6 and label smoothing 0.2 (very minor effect) were used for regularization in combination with light gaussian weight noise (had similar effect as AWP in my test, but lower overhead). I trained for 300 epochs, the first 100 of which used the supplemental data.<br>

I used minimum word error rate training after the model finished training. Where character-based edit distance is used as "word error rate". The method starts with a converged model and uses beam search to generate say the top four predictions. It then calculates each prediction's edit distance and uses this as a weight for the model's predicted probabilities. See for example [Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models](https://arxiv.org/abs/1712.01818). The method's results are unstable even after optimizing it quite a bit. However, short training runs of 1-5 epochs gave very considerable gains in early testing. Unfortunately on the final large, ensembled model it was a rather modest improvement of 0.001-0.002.

## Beam search and inference-time optimizations

Using an ensemble of two models, it was easy to reach the 40MB model size limit. To max out the run time dimension too, I implemented a beam search algorithm that is compatible with the restricted TFLite ops set of this competition. Using it with cached autoregressive decoding allowed me to use beam sizes of five to six (with six sometimes failing the 5h limit). This resulted in + 0.005 on the final ensemble (and even more on earlier, weaker models). The implementation was a bit tricky as there are a few edge cases like having to reorder the decoding caches when beams are changed etc. To prevent the early termination problem when decoding with beam search I used a linear length penalty of 0.15.<br>

On top of this I realized my model achieved an edit distance of 0.0 on low information samples (e.g. < 50 frames and < 5 frames with any hand showing). But we knew that a greedy prediction of e.g. "2 a-e -aroe" gets a score of 0.16. Since most of these low information samples seem entirely corrupted, I simply replace the model's predictions on these with a constant prediction. I used " a-e -are", which slightly different from the greedy one mentioned before as I optimized it towards shorter, low information sequences.<br>
In the end, adding this one line:

```x = tf.cond(num_frames < 50 and num_hand_frames <= 3,
    lambda: tf.constant([[59, 0, 32, 12, 36, 0, 12, 32, 49, 36, 60]]),
    lambda: tf.identity(x))```

gave an improvement of +0.005 across the board (local eval, private and public LB for all models). Which is as much as the whole beam search ...

## What worked and didn't

- Beam search gave a decent +0.005 improvement
- Replacing the model's prediction on corrupt data samples with a constant default prediction gave +0.005
- Deeper models worked better than wider ones
- MWER-training gave a small improvement (+0.001 - +0.002) - however, this was with beam search k=5, with greedy decoding gains were larger (+0.005 in local eval)
- Replacing a single input token with a random one was a decent augmentation
- CTC didn't help as an auxiliary loss
- masking decoder input did not help (when random token replacement was used)
- z-coordinates did not help

As always: really looking forward to reading everyone's solutions. Let me know if there are any questions. Code coming _soon_.
