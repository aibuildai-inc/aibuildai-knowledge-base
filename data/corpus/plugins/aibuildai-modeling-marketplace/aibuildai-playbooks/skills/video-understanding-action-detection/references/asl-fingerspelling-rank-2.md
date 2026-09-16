# 2nd place solution - Test & Compare ASR algorithms

Competition: asl-fingerspelling
Rank: #2
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434588

Once again, thanks to Kaggle, Google and other organizers for hosting this exciting competition. I felt this one as well as the previous competition, is well-organized so that we can try various ideas and could learn a lot from each other. Hope this kinds of competition open in kaggle more often.. :)

##TLDR
My overall solution is almost identical to the previous competition. It's mainly just the adoption of ASR(Automatic Speech Recognition) algorithms (joint CTC + Attention) on top of the previous 1st place model. The details of the model, preprocessing, and training are largely similar to the previous competition, so please refer to the [previous competition solution](https://www.kaggle.com/competitions/asl-signs/discussion/406684) for more details on model/training/augmentations/etc.

## ASR Algorithms
I had a deep interest in ASR, and after recognizing its correlation with the competition, I studied and experimented with various algorithms. While I couldn't find anything particularly superior to the widely used vanilla CTC among the numerous ASR algorithms, I learned a lot through various experiments, and I will mainly share those insights.

As I began studying ASR for this competition, I discovered from reading papers that current NN ASR algorithm mainly consists of CTC, Attention-based, and Transducer. I implemented all three (though there are more diverse algorithms out there). In conclusion, from a baseline performance perspective, all three algorithms were quite similar and each have pros & cons. Here are the insights I gathered from implementing each algorithm:

- CTC:
    - During greedy decoding, the prediction step = O(1). Therefore, when using a Single model, GreedyCTC is the most efficient. (this is probably why many solutions use a large single model with CTC).
    - With beam search, prediction step = O(L) (where L = encoder output length). Depending on the efficiency of the algorithm, the computational overhead isn't very large, so the overhead by introducing beam search isn't very significant. However, the performance improvement from using beam search is minimal (+0.003), making it not very tempting(as its tflite-convertible implementation is not quite trivial) compared to Attention.
    - As discussed, there's no guarantee of alignment between model prediction timesteps, so simple average ensembling can't be applied.

- Attention-based:
    - Uses an autoregressive approach, so even with greedy decoding, prediction step = O(N) (where N = decoder output length).
    -  If the decoder is an RNN or Transformer, stateful inference(i.e. previous key, value caching with Transformer) can be used to reduce the complexity. In actual implementation with Transformer, it reduced the inference time on the CPU by about 20~30%.
    - Beam search is possible, but unlike CTC, it requires introducing appropriate heuristics to penalize the output length. Although several attempts were made, none worked. 
    - It's easier to apply ensembling with Attention. Simply take the average of each model at each prediction step, and it works well (ensembling three models gave +0.009).
    - More room for score improvement than CTC(ex more decoder layers with augmentations).

- Transducer:
Prediction steps = O(L) (where L = encoder output length). Due to the most prediction steps in a greedy manner, it's not very efficient. I didn't consider it further as optimization was harder and performance was slightly lower compared to CTC or Attention. However, it has the advantage of real-time recognition in streaming mode if the encoder is causal (which wasn't relevant for this competition).

Rough single model inference time on test dataset with kaggle kernel is as follows.

> CTC greedy(~40mins) < AttentionGreedy=CTCBeamsearch(~1h20mins) <=  CTCAttentionJointGreedy(~1h30mins)

##CTC-Attention Joint Training & Decoding
As both CTC and Attention showed similar performance, I tried to find a method to utilize both techniques.  I primarily referred to the following two papers:

*Joint CTC-Attention Based End-To-End Speech Recognition Using Multi-Task Learning, Kim et al. 2017.*
[https://arxiv.org/pdf/1609.06773.pdf](url)
*Joint CTC/attention decoding for end-to-end speech recognition, Hori et al. 2017.*
[https://aclanthology.org/P17-1048.pdf](url)

Joint CTC-Attention Training is, as the name implies, adding both a CTC decoder (single GRU layer) and an attention decoder (single Transformer decoder layer) to one encoder for multitask learning. The loss weight is set to CTC=0.25 and attention=0.75. But the Joint training itself did not bring noticeable performance boost.

By using the CTC prefix score, it is possible to calculate the probability of an arbitrary output hypothesis "h" without depending on the output timestep of the CTC output. In other words, **implementing the CTC prefix score computation allows for ensembling outputs not only between CTC models but also between CTC and Attention models**. CTC-attention joint decoding with CTC weight=0.3 showed a performance improvement of +0.007~8 without much impact on inference time. It was especially challenging for me to implement it accurately, efficiently, and without any issues to be compatible with tflite.





##Preprocessing
Similar to the previous competition solution, but simpler. Every landmark and xyz was used and standardized. Flip left-handed signer(rather than augmentation). MAX_LEN=768 was used. Any hand-crafted feature was not significant, likely due to the absence of complex relations between movements in frames.

##Model
Encoder is the same as the previous competition (Stacked Conv1DBlock + TransformerBlock) but increased in size (expand ratio 2->4 in Conv1DBlock) and depth (8 layers -> 17 layers). Single model has ~6.5M parameters. I applied padding='same'(rather than 'causal') and output stride=2 (requires slightly more logic for handling masking). In my case, mixing in Transformer blocks wasn't as effective as in the previous competition. Perhaps global features were less crucial in this comp. Additionally, I added one BN to input of the Conv1DBlock for more training stability(especially with awp + more epochs).

CTC decoder used a single GRU layer followed by one FC layer. Attention Decoder used a single-layer Transformer decoder. Introducing augmentation to the Decoder input and adding up to 4 Decoder layers improves the performance of the attention decoder (up to +0.004). However, considering the number of parameters and inference speed, I considered it inefficient and thus used a single-layer decoder.


##Augmentations
* Random resample (0.5x ~ 1.5x to original length)
* Random Affine
* Random Cutout
* Random token replacement on decoder input(prob=0.2)

What I overlooked this time is that augmentations which showed no performance improvement or even degraded performance in shorter epochs might actually help improve performance in longer epochs. I experienced a similar phenomenon in the previous competition, but it seemed more pronounced in this one. By selecting augmentations solely based on the results from a 60epoch experiment, I think I missed out on many potentially beneficial augmentations when testing the 400epoch training in the final week.


##Training
Epoch = 400
bs = 16 * num_replicas = 128
Lr = 5e-4 * num_replicas = 4e-3
AWP = 0.2 starts at 0.1 * Epoch
Schedule = CosineDecay with warmup ratio 0.1
Optimizer = AdamW (slightly better than RAdam with Lookahead)
Loss = CTC(weight=0.25) + CCE with label smoothing=0.1~0.25(weight=0.75)

Training takes around 14 hours with colab TPUv2-8(as colab TPU runtime recently reduced to 3~4 hours, needed 4 consecutive sessions to complete training).
Longer Epoch always gave better CV(5fold split by id) and LB but got no time to try over 400 epochs.

##LB history
|  | public LB | private LB | 
| --- | --- | --- |
| prevcompsinglemodel + CTC (or Attention) | 0.76 | 0.74 |
| + deeper and wider model, add pose | 0.79 | 0.78 |
| + 3 seed ensemble with Attention | 0.80 | 0.79 |
| + ctc attention joint decoding | 0.81 | 0.80 | 
| + use all landmarks, longer epoch | 0.82 | 0.81 |


Seeing solutions from other kagglers, not only the top solutions but also the public notebooks is always inspiring. Thanks to kagglers who shared their insights and ideas as always. And big congrats to @christofhenkel and @darraghdog for winning this highly competitive competition!

Training/Inference code: https://www.kaggle.com/competitions/asl-fingerspelling/discussion/436873
