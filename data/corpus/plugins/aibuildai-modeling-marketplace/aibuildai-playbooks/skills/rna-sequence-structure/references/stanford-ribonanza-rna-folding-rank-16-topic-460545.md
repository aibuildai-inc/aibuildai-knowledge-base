# 16th Place Solution

Competition: stanford-ribonanza-rna-folding
Rank: #16
Source: https://www.kaggle.com/c/stanford-ribonanza-rna-folding/discussion/460545

First off, I wanted to thank the competition hosts for putting together this competition as well as the community for all of the discussion.

# Summary
- Transformer architecture
    -  Separate models for 2A3 and DMS
- Used Pr(base paired) derived from BPPM
- Dynamic position bias

# Approach

## Data:
The input features were:
- Sequence
- Pr(base paired) (i.e. sum along one axis of the BPPM) from 4 RNA structure packages

I was hesitant to include the BPPM for a few reasons that I thought would result in hindering the generalized performance for longer sequences:
1. Base-base pairing probability is overly specific ‘wrong’ information
2. Lack of consideration of pseudoknots
3. Dilution of individual probabilities for longer test sequences compared to the shorter train sequences

My compromise was to consider the global probability of pairing for each base. [This paper from the Das lab](https://www.nature.com/articles/s41592-022-01605-0) assessed how Pr(paired) as calculated by various RNA structure packages compared to chemical mapping assay results. They found CONTRAfold 2, ViennaRNA 2 at 60 degrees C, and RNAsoft with BLStar parameters to be the most effective approaches. Therefore, I generated Pr(paired) vectors for all sequences from these three methods as well as Eternafold to use as features.

As an aside, I used Arnie to run these packages and spent more time than I care to admit getting it setup for multiprocessing, although overall I think it ended up saving time. I did try to expand to additional methods, however, I pretty quickly abandoned it due to bugs / run time constraints.

Earlier on in the competition, I did also try various structure features generated from ViennaRNA (MFE, centroid, ensemble). Compared to sequence alone, I was able to get improved performance, however, it added no benefit once the Pr(base paired) feature was included.

## Architecture:
My approach ended up being a single transformer architecture separately trained to predict 2A3 or DMS.

**Seq**
* 4 token embedding -> 512 output dimension

**Pr(base paired)**
* Linear layer 4 input dimension -> 512 output dimension

**Transformer**
* 18 layers, 512 dimension, 8 heads
* Postnorm
* Dynamics position bias
* Intra-attention Gating on Values
* Input: Seq + Pr(base paired)

This was my first time working with a transformer architecture. Luckily, I found the library [x-transformers](https://github.com/lucidrains/x-transformers) by @lucidrains and used that to try out / learn about numerous variations of transformers.

**What helped:**
*Dynamic Position Bias*
One of my first goals was to understand how to generalize a transformer to longer sequences. Initially I found the ALiBi paper and thought that would be a useful approach. However, based on the x-transformers documentation it appeared as if dynamic position bias would be more likely to be effective in this context. I did not attempt any direct comparisons between ALiBi (or other relative position biases) and dynamic position bias since I did not trust performance differences on length extrapolation from the train set alone. Dynamic position bias did show the largest bump of any modification on the public vs. private sets (public - 0.15400 vs 0.15002; private - 0.18476 vs 0.14976). Note: this result was before the change to using Pr(base paired) data.

*Intra-attention Gating on Values*
This is an attention variant from Alphafold2 that gates the aggregated values with the input. This was one of the feature variations available on x-transformers that appeared to have a small improvement on performance. Note: I added this one a while back and just kept it in, so it is entirely possible it adds nothing of value in the final model.

*Postnorm*
One of the first changes I assessed with the transformer was to switch from the generally more preferred prenorm to the original postnorm approach. This resulted in improvements to convergence rate and performance. I rechecked this multiple times over the course of the competition as the architecture and input data approaches shift and consistently got the same type of result. I also attempted the various other norm approaches found in x-transformers, but none resulted in any noticeable improvement.

*18 Layers*
18 layers seemed to be the sweet spot of performance and training time. The main depths I tested were 12, 18, 24. 12 vs 18 showed a small performance improvement on the leader boards (public - 0.14383 vs 0.14330; private - 0.14595 vs 0.14517). 18 vs 24 I did not notice any difference in the CV score, so I did not submit any 24 layer models.

*Sum embeds*
My approach to integrating the embedded sequence and the Pr(base paired) after passing through a linear layer (both of dimension 512) was to simply sum them elementwise. I tried other approaches, but they all appeared to have negative impacts to convergence and/or performance.

## Training:

I used the splits used in @iafoss starter transformer notebook and all of my experimental work was done using fold 0. Because I ended up using individual models for 2A3 and DMS prediction, the set of sequences satisfying SN >= 1.0 and reads >= 100 was higher for each group since there was a subset of sequences where only one of 2A3/DMS satisfied the constraint.

Additionally, I incorporated the react_error values as a means of weighting the loss so that I could use more of the lower quality sequences with greater confidence (luckily this happened right after the time the issue with react_error was identified and corrected). The react_error values were strictly greater than 0 and were usually fairly low (I believe the median on at least the high quality samples was ~0.125). I decided to try weighting the loss of each base using an exponential function:

Weighting v1:
exp(-react_error)

Weighting v2:
exp(-(react_error^2))

This would result in a weight value between 0 and 1 for each base in the sequence. While I don’t have a good mathematical rationale for this approach and I am sure there may be better options, it did give a noticeable boost to CV performance (~0.0005 - 0.001), including when using only the higher quality sequences.

My approach to training ended up being somewhat erratic for the final model.

- 2A3 model
    - fit_one_cycle(epochs=256, lr=5e-4, pctstart=0.01)
    - SN >= 1.0, reads >= 100
    - exp(-(react_error^2)) weighting
    - Early stopping at epoch 45
- 2A3 model part 2
    - Fine tune 2A3 model, lr=5e-5
    - SN >= 0.2, reads >= 75
    - exp(-react_error) weighting
- DMS model
    - Fine tune 2A3 model part 2, lr=5e-5
    - SN >= 1.0, reads >= 100
    - exp(-(react_error^2)) weighting
- DMS model part 2
    - Fine tune DMS model, lr=5e-5
    - SN >= 0.2, reads >= 75
    - exp(-react_error) weighting

As I mentioned above, I used fold 0 of my CV splits for all experimentation. Unfortunately, I lost time testing out other architectural approaches in the final 2 weeks and only settled on coming back to the split models due to the slight performance improvement within the last day, so I was unable to run the full set of CV folds. However, in a last minute effort to at least capture some of the fold 0 validation data in my model I ended up running 1 epoch on this data for my second final submission, which ended up giving me a nice little last minute boost on the leaderboard (public: 0.14330 vs 0.14199; private: 0.14517 vs 0.14458).
