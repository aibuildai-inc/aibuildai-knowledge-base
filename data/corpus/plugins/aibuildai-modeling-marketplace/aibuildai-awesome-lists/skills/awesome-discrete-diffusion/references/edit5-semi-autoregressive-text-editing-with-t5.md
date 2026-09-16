---
arxiv_id: "2205.12209"
title: "EdiT5: Semi-Autoregressive Text Editing with T5 Warm-Start"
year: 2022
source: arxiv2md
---

## Abstract

Abstract We present EdiT5 1 1 1 Code and pre-trained models https://edit5.page.link/code – a novel semi-autoregressive text-editing model designed to combine the strengths of non-autoregressive text-editing and autoregressive decoding. EdiT5 is faster during inference than conventional sequence-to-sequence (seq2seq) models, while being capable of modeling flexible input-output transformations. This is achieved by decomposing the generation process into three sub-tasks: (1) tagging to decide on the subset of input tokens to be preserved in the output, (2) re-ordering to define their order in the output text, and (3) insertion to infill the missing tokens that are not present in the input. The tagging and re-ordering steps, which are responsible for generating the largest portion of the output, are non-autoregressive, while the insertion step uses an autoregressive decoder. Depending on the task, EdiT5 on average requires significantly fewer autoregressive steps, demonstrating speedups of up to 25x when compared to seq2seq models. Quality-wise, EdiT5 is initialized with a pre-trained T5 checkpoint yielding comparable performance to T5 in high-resource settings when evaluated on three NLG tasks: Sentence Fusion, Grammatical Error Correction, and Decontextualization while clearly outperforming T5 in low-resource settings.

## 1 Introduction

Figure: Figure 1: EdiT5 transforms the input text A long user query into the output The user query is very long by first generating a sequence of edit tags D K K K (where K stands for keeping and D for deleting the input token), re-ordering the input tokens with the pointer network, and infilling missing tokens into the source sequence with an autoregressive decoder which jointly predicts the text spans (The and is very) and the position where to insert them (pos0 and pos2). The blue arrow shows how the token pos2 is predicted conditioned on the prefix <s> pos0 The generated thus far. The dotted arrow lines depict the encoder-decoder cross attention over the re-ordered input tokens and edit tags.
Refer to caption: /html/2205.12209/assets/x1.png

Pre-trained seq2seq models such as T5 , BART , and MASS have established strong baselines for the majority of text-to-text transduction tasks. A recent trend to massively scale up model sizes, e.g., all the way up to 540B params , as well as the sizes of pretraining corpora, has further pushed the state-of-the-art without signs of reaching a plateau. From a practical point of view, running inference with such models is prohibitively expensive for most applications, which motivates the work on finding efficient recipes for model distillation, e.g., and choosing a model architecture that can provide a better trade-off between performance on a given task and inference speed. A typical choice is to distill a large language model into a smaller seq2seq model, e.g., Transformer . In this paper we propose a novel model architecture EdiT5which blends ideas from a seq2seq T5  and text-editing to provide faster inference without sacrificing on task performance.

Seq2seq-based models output text token-by-token from scratch, allowing them to model any kind of input-output relationship. However, for many real-world tasks this degree of generality is unnecessary, especially for monolingual tasks where the input and output texts have relatively high degrees of overlap. In such cases a natural approach is to cast conditional text generation as a text-editing task, where the model learns to construct target texts by applying a set of edit operations to the inputs . Typically the set of edit operations is defined ahead of time , which on the one hand limits the flexibility of the model to reconstruct arbitrary output texts from the inputs, but on the other, leads to latency improvements as the limited set of allowed operations significantly reduces the output vocabulary of the decoder. In this paper, we propose an approach which is both fast at inference time and flexible, able to model arbitrary rewrites.

#### Faster inference.

A common method for achieving low latency in serving models is to reduce their size, thus reducing their computational cost. Doing so naively, however, often leads to inferior model quality, and much work has gone into finding better methods for model size reduction, such as distillation .

Regardless of model size, one of the major contributors to the total inference time for seq2seq models is the decoder, which generates the output sequence step-by-step. EdiT5 also relies on an autoregressive decoder, but generates the majority of the output sequence with its tagging and pointing networks, and as such the decoder makes far fewer steps.

#### Flexible text-editing.

Recent text-editing approaches, e.g., , are not as powerful as general purpose seq2seq approaches when it comes to modeling arbitrary input-output text transductions. EdiT5 supports open-vocabulary generation by relying on an autoregressive decoder. In the extreme case, where there is no overlap between the source and the target texts, it reduces to a vanilla seq2seq model generating the entire output from scratch. However, when the input and output overlap, it can benefit from the tagging and pointer networks to reconstruct the bulk of the output text that is further infilled (refined) by the autoregressive decoder.

#### Warm start.

Training a high-precision text generation model typically requires large amounts of high-quality supervised data. Self-supervised techniques based on text in-filling  have been shown to provide a crucial advantage over non-pre-trained models especially in low-resource settings. Hence, we design EdiT5 to be able to benefit from already existing pre-trained language models (specifically T5), where the final model is directly fine-tuned on the downstream task.

EdiT5 decomposes the generation task into three steps: tagging, pointing and insertion
(see Fig. [1](#S1.F1)). The tagger and pointer networks decide which source tokens to preserve and in which order they should appear in the output, thus allowing for arbitrary word dropping and reordering. The tagger is implemented using a non-autoregressive feedforward network, and pointing is implemented using a novel non-autoregressive pointing mechanism combined with sinkhorn layers . The insertion network inserts/infills words which are present in the target sequence but do not appear in the source sequence. The network is implemented using an autoregressive transformer decoder, which attends to the tagged, reordered source sequence. The decoder predicts both the locations of where the token spans should be infilled, as well as the spans themselves.

We evaluate EdiT5 on three distinct text generation tasks: Sentence Fusion, Grammatical Error Correction (GEC), and Decontextualization, comparing to recent text-editing approaches and T5. Each task is unique in the editing operations required and the amount of training data available, which helps to better quantify the value of modeling decisions we have integrated into EdiT5.

Additionally, we explore the impact of training data size and model size on EdiT5. Finally we quantify the latency of EdiT5, providing a detailed analysis and comparison to T5.

## 2 Model description

The model architecture of EdiT5 resembles a vanilla Transformer  composed of an encoder and a decoder. EdiT5 decomposes the generation of a text $\mathbf{y}$ from an input $\mathbf{x}$ into three parts: predicting a sequence of edit tags $\mathbf{y}^{t}$ (indicating whether a token from the input should be copied to the output), a permutation of the input tokens $\mathbf{\pi}$ (indicating the order that copied tokens should appear in in the output), and a sequence of tokens $\mathbf{y}^{d}$ (indicating additional tokens that should be in the output, and where in the permuted input they should be inserted). $\mathbf{y}^{t}$ and $\mathbf{\pi}$ are modeled by the encoder, and $\mathbf{y}^{d}$ by the decoder.

There are multiple ways to choose the triple ($\mathbf{y}^{t}$, $\mathbf{\pi}$, $\mathbf{y}^{d}$) for a given ($\mathbf{x}$, $\mathbf{y}$) pair. During dataset creation we choose a single such triple for each training pair (see section [2.1](#S2.SS1.SSS0.Px6) for details), in which case the probability of $\mathbf{y}$ can be expressed as:

$$ $\displaystyle P(\mathbf{y}|\mathbf{x}):=$ $\displaystyle\left(\prod^{|\mathbf{y}^{d}|}_{i}P(\mathbf{y}^{d}_{i}|\mathbf{y}^{d}_{<i},\mathbf{y}^{t},\mathbf{\pi},\mathbf{x})\right)$ $\displaystyle*P(\mathbf{\pi}|\mathbf{y}^{t},\mathbf{x})*P(\mathbf{y}^{t}|\mathbf{x})$ (1) $$

During inference, we first greedily set $\mathbf{y}^{t}$ to maximize the third term, then $\mathbf{\pi}$ to maximize the second term, and finally $\mathbf{y}^{d}$ to maximize the first term. The output text $\mathbf{y}$ is realized by applying the tags $\mathbf{y}^{t}$ and permutation $\mathbf{\pi}$ to the input sequence $\mathbf{x}$ and then inserting the tokens $\mathbf{y}^{d}$.

### 2.1 Text-editing encoder

The EdiT5 encoder consists of three steps:
encoding, tagging, and pointing.

#### Encoder.

The source sentence ${\mathbf{x}}$ is first encoded using $N$ transformer layers into the hidden representations $\mathbf{h}$.

#### Tagging.

The tag sequence $\mathbf{y}^{t}$ is constructed as follows: source tokens that must be copied are assigned the KEEP tag, tokens not present in the output are marked by the DELETE tag.
Tags are predicted by applying a single transformer layer followed by a classification layer to the output of the encoder $\mathbf{h}$, which is trained using cross-entropy:

$$ $\mathcal{L}_{tagging}=-\sum_{j}^{|\mathbf{x}|}\log P({y}^{t}_{j}|f_{t}(\mathbf{h})_{j})$ (2) $$

where $\mathbf{y}^{t}$ are the gold tags, $j$ is the index of the source token, and $f_{t}$ is a transformer layer followed by a classification layer. During inference we use *argmax* to determine the tags, whereas during training we use the gold tags. The encoder hidden state is then updated to take these tags into account:

$$ $\mathbf{h}^{t}_{j}=f_{te}([\mathbf{h}_{j};TE(\mathbf{y}^{t}_{j})])$ (3) $$

Where $TE$ is a tag embedding layer, whose output is concatenated to the original hidden representation of the source sequence, before a feed-forward layer $f_{te}$ is applied.

#### Pointing.

In many tasks it is helpful for the model to be able to rearrange the kept input tokens. For example, we can grammatically correct the sentence *Who you are?* to *Who are you?* purely by reordering tokens from the input. In EdiT5 this is made possible thanks to its pointing mechanism. In contrast, in text editing approaches such as , correcting this sentence involves first deleting the words *you are* and then recreating them in the right order.

Given a sequence $\mathbf{x}$ and the predicted tags $\mathbf{y}^{t}$, the re-ordering model generates a permutation $\mathbf{\pi}$. Our implementation is based on a pointer network , where an attention mechanism points to the next token. We follow which, unlike previous approaches where a decoder state attends over an encoder sequence, applies intra-attention, where source tokens attend to all other source tokens. As such the output of this model is a series of predicted pointers, where each source token predicts the token that comes after it. $\mathbf{\pi}$ can easily be constructed by daisy-chaining these predicted pointers together, as seen in Fig. [2](#S2.F2).
We calculate attention using key-query attention, where we include an additional transformer layer prior to the key network:

$$ $\alpha_{m,j}=f^{q}(\mathbf{h}^{t})_{m}\boldsymbol{\cdot}f^{k}(\mathbf{h}^{t})_{j}$ (4) $$

Where $\alpha_{m,j}$ is the unnormalized attention, $f^{q}$ is the query network, a single feed-forward layer, and $f^{k}$ is the key network, a transformer layer followed by a single feedfoward layer.

Unlike , we ensure a valid permutation is formed, i.e. no token is pointed to twice, by using sinkhorn layers , which normalizes over both the rows and the columns of the intra-pointer attention $\alpha$. Sinkhorn layers are defined as:

$$ $\displaystyle S^{0}$ $\displaystyle=\exp(\alpha)$ (5) $\displaystyle S^{i}$ $\displaystyle=T_{c}(T_{r}(S^{i-1}(\alpha))$ (6) $$

where $T^{j,m}_{c}(X)=\frac{X_{j,m}}{\sum_{l}X_{l,m}}$ is the column normalization operator and $T^{j,m}_{r}(X)=\frac{X_{j,m}}{\sum_{l}X_{j,l}}$
is the row normalization operator.

The loss for the pointing network is defined as:

$$ $\mathcal{L}_{pointing}=CE(\pi|S(\alpha))$ (7) $$

Where CE is the cross-entropy loss. During inference we use argmax to determine $\pi$.

We use additional positional embeddings to update the hidden states with their new position (offset from 0). For example if *Who you are?* was reordered into *Who are you?*, the position information would be updated as 0Who 2you 1are 3?.

$$ $\mathbf{h}^{p}_{j}=(\mathbf{h}^{t}_{j}+\mathbf{PE}(\pi_{j}))$ (8) $$

where $PE$ are learnt absolute positional embeddings . These additional positional embeddings are masked out for those source words which do not appear in the target sequence. Finally we apply a transformer encoder layer to $\mathbf{h}^{p}$ forming the final encoded representation of the sequence $\mathbf{h}^{f}$. $\mathbf{h}^{f}$ captures the edits as well as the original sequence $\mathbf{x}$, and the decoder attends to this representation.

Figure: Figure 2: Pointing mechanism to transform “a long user query" into “user query long".

#### Decoder.

We use a standard transformer decoder, which is tasked with inserting tokens which are in the output sequence but don’t appear within the input sequence. EdiT5 takes advantage of the pre-training of a T5 model, where T5 was pre-trained to infill missing spans. When pre-training T5 uses special tokens *⟨pos_i⟩* to indicate where missing spans should be inserted, as demonstrated in Figure 3. EdiT5 re-purposes these special tokens, using them to indicate at which position new tokens should be infilled. I.e. *⟨pos_1⟩*, indicates that the tokens should be inserted after the first token. As such the decoder first decodes a special position token and then decodes the inserted tokens which should appear after this token. For example to insert *the cat* after the first token, the decoder generates: *⟨pos_1⟩ the cat*. The decoder is trained with a standard cross-entropy loss:

$$ $\mathcal{L}_{insertion}=-\sum_{i}^{|\mathbf{y}^{d}|}\log P(y^{d}_{i}|\mathbf{y}^{d}_{<i},h^{f})$ (9) $$

Where $i$ is the decoder index, and ${h}^{f}$ is the encoder output. The loss for the entire model is defined as the sum of the three individual losses:

$$ $\mathcal{L}=\lambda_{1}\mathcal{L}_{tagging}+\lambda_{2}\mathcal{L}_{pointing}+\lambda_{3}\mathcal{L}_{insertion}$ (10) $$

where $\lambda_{1}$, $\lambda_{2}$ and $\lambda_{3}$ are hyper-parameters determining the relative importance of tagging, pointing and insertion losses in the final loss.

#### Pre-training.

Figure: Figure 3: Example pre-training noise for T5 and EdiT5. K and D indicate keep and delete tags resspectivly, and [0] indicates *pos0*.

While we initialize EdiT5 from T5 base, T5 was pre-trained with 12 decoder layers, and for EdiT5 we use a single decoder layer. To account for this change in the decoder layers, we perform additional pre-training. We use a pre-training objective which combines a T5 style span insertion task, with a generic text-editing denoising task, as used in BART . A source sentence is corrupted by dropping, swapping and adding spans (an example can be seen in Figure [3](#S2.F3)), and we task our model to reconstruct the original sentence. By introducing noise we are able to train the tagger to detect incorrect spans, and the pointer to reorder the sentence. The decoder then behaves like the T5 pre-training objective inserting the content of missing spans. Unlike BART’s pre-training, our approach is computationally cheap, as we do not decode the entire sequence when training, instead just decoding the missing spans.

#### Dataset construction.

When constructing the training dataset, there are many possible combinations of $\mathbf{y}^{t}$, $\mathbf{\pi}$ and $\mathbf{y}^{d}$ which could produce $\mathbf{y}$. For instance, all source tokens could be deleted and the decoder could then produce all the target tokens. However to minimize latency, we wish to make the number of inserted tokens (i.e. the number of decoder steps) as small as possible, and maximize the number of kept tokens.

To produce alignments from a target sequence to a source sequence, we iterate left-to-right through characters in the target sequence, trying to find spans of target characters which appear in the sequence of source tokens, as described in Algorithm [1](#algorithm1) (see Appendix [A](#A1)). Each source token can only be aligned to a single target span. Those target spans that can’t be aligned are instead inserted after the closest previous aligned source token. In cases where there are multiple possible alignments, e.g. the same token appears multiple times in the source, we align the target character span to produce the longest contiguous span of source tokens aligned with the target, i.e. where source tokens appear one-after-another in the target sequence. To find the longest contiguous span we compare the contiguous overlap between source and target for each possible alignment.

## 3 Experiments

We evaluate EdiT5 on three distinct text-editing tasks: Sentence Fusion, Grammatical Error Correction, and Decontextualization. In addition to reporting previously published results for each task, we also compare to Felix , a recent non-autoregressive text-editing model, and a strong pre-trained T5 baseline implemented in the T5X framework .

#### Modeling.

For EdiT5 we initialize with a T5 base model with a 12-layer Transformer encoder, and single-layer Transformer decoder. Our code is based on the Tensorflow Model Garden’s  TF2 version of T5. After initializing with the T5 checkpoint, we further pre-train on the denoising objective (see Section [2.1](#S2.SS1.SSS0.Px5)) using the C4 corpus , training for 100k steps.

For all experiments EdiT5 is trained using AdamW , additionally the learning rate was decayed using the validation set, and exact match is used for checkpoint selection. Tokenization is based on T5’s SentencePiece vocabulary , with a vocabulary size of 32k. We, however, modify the vocabulary, removing tokens which have punctuation as a suffix, and replacing them with additional span insertion special token, giving EdiT5 512 span insertion special token. Unless otherwise stated, we use an input sequence length of 128. We performed minimal hyper-parameter selection, which is discussed in the Appendix.

#### Task Analysis.

The chosen tasks cover a diverse set of edit operations and a wide range of dataset sizes, varying from under 11 thousand data points to over 4.5 million. Table [1](#S3.T1) provides dataset statistics including: the size, input sequence length, output sequence length for seq2seq models, the output sequence length for EdiT5, and the translation error rate (TER) between the source and target sentences. We use TER to highlight unique properties of each task.

From Table [1](#S3.T1) we see that for all tasks EdiT5 requires significantly fewer decoder steps than a seq2seq model, which results in significant latency savings. We also see that decontextualization has the longest input and output sequences, where the maximum input length of decontextualization is 512 tokens. Decontextualization has the highest TER, with the major contribution being deletion, which is due to the input sequence consisting of a paragraph, whereas the output is a single sentence. In contrast GEC, has the shortest input and output sequence, with the majority of the dataset consisting of a single input and a single output sentence. GEC has the lowest TER, however it has the highest insertion TER. Sentence fusion consists of two sentences being rewritten into a single sentence, and has a middling TER and sequence lengths. It also has the fewest substitutions.

**Table 1: Statistics across tasks: size of the dataset (Size), source length in tokens ($L_{\text{src}}$), target length in tokens ($L_{\text{tgt}}$), EdiT5 insertion tokens (E5-Ins), and TER scores, including number of insertions (Ins), deletions (Del), substitutions (Sub), and shifts (Shft). Token counts are measured using a sentencepiece tokenizer and averaged over the development set.**
| Dataset | Size | $L_{\text{src}}$ | $L_{\text{tgt}}$ | E5-Ins | TER | Ins | Del | Sub | Shft |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sentence fusion | 4.5M | 42.5 | 41.1 | 5.8 | 10.92 | 2.49 | 04.91 | 3.75 | 0.62 |  |
| GEC | 2.3M | 24.3 | 24.7 | 4.6 | 09.72 | 2.99 | 01.19 | 5.05 | 0.49 |  |
| Decontextualization | 11K | 193.9 | 49.1 | 7.2 | 84.80 | 0.28 | 90.64 | 6.43 | 2.65 |  |

### 3.1 Sentence Fusion

Sentence Fusion is the task of fusing independent sentences into a coherent output sentence(s) . It requires operations such as inferring the appropriate discourse connective, pronominalization, reordering the text to introduce relative clauses, and changing the order of the input sentences.

#### Data.

We use the “balanced Wikipedia” portion of the DiscoFuse dataset  and also study the impact of training data size by creating four additional smaller subsets of DiscoFuse consisting of: 450,000 (10%), 45,000 (1%), 4,500 (0.1%) and 450 (0.01%) data points.

#### Setup.

Following , we report Exact match, which is the percentage of exactly correctly predicted fusions. In addition to the T5 baseline and the text-editing baselines LaserTagger , Felix , and Seq2Edits , an autoregressive text-editing model, we also report state-of-the-art seq2seq models ROBERTASHARE , based on ROBERTA large, and AugBERT , based on BERT base. Additionally, we measure the impact of our pre-training (Section 2.1) initializing EdiT5 with a T5 checkpoint, without additional pre-training.

#### Results.

**Table 2: Sentence fusion results (Exact Match, lower-cased) under various data conditions, latency (ms), and number of parameters.**
|  | #Params | 100% | 10% | 1% | 0.1% | 0.01% | latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LaserTagger | 110M | 53.80 | 47.31 | 38.46 | 25.74 | 12.32 | - |
| Felix | 220M | 61.31 | 52.85 | 45.45 | 36.87 | 16.96 | 1.8 |
| Seq2Edits | 279M | 61.71 | - | - | - | - | - |
| EdiT5 | 141M | 64.95 | 59.26 | 52.09 | 43.83 | 28.64 | 2.2 |
| - pre-training | 141M | 65.16 | 59.27 | 50.39 | 34.18 | 1.90 | 2.2 |
| T5 base | 220M | 65.52 | 59.75 | 50.75 | 33.84 | 10.75 | 52.7 |
| ROBERTA | 380M | 66.6 | - | - | - | - | - |
| AugBERT | 157M | 65.0 | - | - | - | - | - |

From the top section in Table [2](#S3.T2) we first observe that EdiT5 strongly outperforms other text-editing methods. Next it performs comparably to T5 in high-resource settings (100% and 10%), where it’s just 0.5 points lower in exact match than T5, whilst achieving a latency that is 25 times faster, and using fewer parameters. The current SOTA, ROBERTASHARE, which outperforms EdiT5 by 1.5 points, is based on the ROBERTA large checkpoint which overall has more parameters and a larger encoder. In low-resource settings, EdiT5 clearly outperforms T5 by up to 18 points (0.01%, i.e. 450 training examples).

The results in Table [2](#S3.T2) additionally demonstrate that the significant improvements of EdiT5 over Felix in high/medium-resource settings do not stem from EdiT5 pre-training. With 450 datapoints, pre-training is critical since there’s a larger mismatch between EdiT5 and T5 checkpoints than there is between Felix and BERT checkpoints. We additionally ablated the impact of sinkhorn layers, and found that under the 100% data condition there was a modest decrease in performance (0.5 exact match points).

### 3.2 Decontextualization

Sentence decontextualization task was introduced by . The goal is to rewrite an input sentence to make it stand-alone without the original context.

#### Data.

We use the train, dev and test data from , where sentences were selected from Wikipedia passages. Human annotators were asked to rewrite them, if possible, to be interpretable and grammatical without the context. We compare against T5 base, T5 xxl, Felix, and a copy baseline. All models use a sequence length of 512.

#### Metrics.

Following , we report exact match, exact match when a sentence needs to be rewritten and SARI F1 (deletion and addition) on unigrams .

#### Analysis.

Results in Table [3](#S3.T3) show that EdiT5 achieves a higher exact match scores, and SARI delete score when compared to T5 base, with a significant drop in latency and using fewer parameters. T5 base achieves significantly higher SARI add, suggesting its better at inserting new tokens, which is unsurprising as EdiT5 is primarily focused on copying the source sequence. Both T5 and EdiT5 achieve significantly higher numbers than Felix. EdiT5 and T5 base, however, still achieve a significantly lower score than the T5 xxl, which can be explained by the difference in model size.

**Table 3: Decontextualization results, including exact match (*EM*, exact match on those sentences which need rewriting *EMc*, SARI *ADD*, SARI *DEL*ete, latency (ms), and number of parameters. * indicates scores were calculated by running the models provided by on the test set.**
|  | #Params | EM | EMc | ADD | DEL | latency |
| --- | --- | --- | --- | --- | --- | --- |
| Repeat | - | 36 | 0 | 0 | 0 | - |
| T5 xxl | 11B | 52 | 32 | 43 | 47 | - |
| Felix | 220M | 32 | 10 | 28 | 32 | 4 |
| EdiT5 | 141M | 48 | 23 | 31 | 41 | 3.8 |
| T5 base* | 220M | 40 | 21 | 36 | 40 | 75 |

### 3.3 Grammatical Error Correction

GEC requires systems to identify and fix grammatical errors in a given input text.

#### Data.

We evaluate on the standard GEC test set BEA , and use BEA-DEV for checkpoint selection. For pre-training we use an artificial GEC dataset C4_200M of 200M sentences . We then fine-tune on cLang-8 , a distilled version of the Lang-8 learners corpus .

#### Setup.

We report *ERRANT* F0.5 scores for BEA. We report additional gT5/gFelix baseline numbers from , where T5/Felix models were trained only on cLang-8. For pre-training we sampled 0.2% examples from the training set to use as a development set, and train till convergence as measured on this development set.

We additionally measure the impact that model size has on quality and latency, training T5 and EdiT5 small, base, and large models. To make the latency comparison fairer, we also train single-decoder-layer variants of the T5 models we call T5 Slim. To further ensure a fair latency comparison between EdiT5 and T5 we use the same framework for both models. Additionally, we do not perform EdiT5 specific pre-training.

#### Results.

From Table [4](#S3.T4), we see that all models outperform their equivalent gT5/gFelix models, which is not surprising as the latter models were trained on less data. A surprising result is that the T5 slim variants achieve comparable scores to the full T5 models while having significantly lower latency. Comparing EdiT5 against T5 models, we see up to $\sim$1 point differences in F0.5 scores between models of the same size (small/base/large), however EdiT5 produces speed ups between 10x and 25x.

In Figure [4](#S3.F4), we study the latency–quality trade-offs of T5, T5 slim, and EdiT5 models. We omit Felix from this analysis, because Felix achieves a significantly lower score. We focus on the 95 percentile latency, as it is often the case that users require that a model returns a result within a fixed latency budget. We see that EdiT5 drops less than 0.25 F0.5 points comparing across model sizes, whilst being significantly faster. Additionally for a given latency budget of 5ms, no full T5 model would fit, and only the T5 slim small would fit, whereas both EdiT5 small and base fit. Comparing EdiT5 base against T5 slim small, we see that EdiT5 scores 3 F0.5 points higher, whilst being faster. For any latency budget under 20ms, EdiT5 is quicker and offer better results than T5 and T5 slim. For latency budgets above 20ms, T5 slim large scores slightly (<0.25 F0.5) higher than EdiT5, and if latency is not a factor then gT5 xxl should be used.

**Table 4: GEC F0.5 results for gT5, gFelix, T5, T5 slim, Felix, and EdiT5; number of parameters; mean, mode and 95 percentile latencies (in milliseconds); we also present speed up, the ratio of 95 percentile latency to T5 base.**
| Model | #Params | F0.5 | Mean | Median | 95% | Speed Up |
| --- | --- | --- | --- | --- | --- | --- |
| gT5 small | 76M | 65.01 | - | - | - | - |
| gT5 base | 248M | 69.39 | - | - | - | - |
| gT5 large | 783M | 72.06 | - | - | - | - |
| gT5 xxl | 11B | 75.88 | - | - | - | - |
| gFelix base | 220M | 59.05 | - | - | - | - |
| T5 small | 76M | 69.79 | 10.5 | 9.2 | 21.0 | 3.5x |
| T5 base | 248M | 72.39 | 35.5 | 31.2 | 74.1 | 1.0x |
| T5 large | 783M | 73.43 | 92.4 | 81.3 | 184.8 | 0.4x |
| T5 slim small | 55M | 68.50 | 2.6 | 2.3 | 5.1 | 14.5x |
| T5 slim base | 144M | 71.78 | 4.7 | 4.3 | 8.7 | 8.5x |
| T5 slim large | 391M | 73.18 | 11.1 | 10.1 | 20.0 | 3.7x |
| Felix base | 220M | 63.50 | 1.8 | 1.8 | 1.8 | 41.2x |
| EdiT5 small | 50M | 68.40 | 0.9 | 0.8 | 1.3 | 57.0x |
| EdiT5 base | 141M | 71.58 | 1.8 | 1.6 | 2.5 | 29.6x |
| EdiT5 large | 391M | 72.93 | 4.1 | 3.9 | 6.6 | 11.2x |

Figure: Figure 4: Mean and 95% percentile latency for T5, T5 slim and EdiT5 across model sizes on BEA.
Refer to caption: /html/2205.12209/assets/x2.png

## 4 Latency analysis

The tasks on which EdiT5 outperforms seq2seq models in latency are those that have overlap between sources and targets, but it’s unclear how much overlap is required for EdiT5 to produce latency savings. To answer this question, we split EdiT5 base, T5 base and T5 slim base into components whose latencies we measure separately and compare. Details on how latencies are measured can be found in the Appendix [C](#A3).

A seq2seq model decomposes into two parts: the encoder (we include the input embedding here, so we refer to this as encoder* below), and the decoder. EdiT5 has both of these parts, but also includes a third part (which we call its overhead), comprising of pointer realization and additional transformer layers. To make our analysis simpler and more task-agnostic, we make two simplifying assumptions. First, we assume the worst-case that no tokens are deleted by EdiT5 and there are no padding tokens in the input(^2^22The pointer realization runs for exactly input-length steps.), in practice this is not the case, and provides significant latency savings for EdiT5. Second, we assume that decoder latency is linear in the number of decoder steps(^3^33This ignores decoder self-attention, but is justified when the number of decoder steps is small.). Both of these assumptions benefit the latency of seq2seq models more than EdiT5.

#### Results.

In Table [5](#S4.T5) we present latencies of encoder*, worst-case EdiT5 overhead and the per-step latency of a decoder under various input-length conditions. We see the overhead added by EdiT5 even in the worst-case is small.

From these results we can derive a simple rule for when EdiT5 will provide a net latency benefit. Compared to T5 slim base(^4^44The overhead is smaller than 1 step of T5 base.), EdiT5 base must save on average 4 decoder steps with an input length of 128, and 7 steps with an input length of 512.

Finally, collating the results in Table [5](#S4.T5) with the number of decoder steps performed by EdiT5 and T5 in Table [1](#S3.T1), we see that whereas in T5 the decoder latency dominates the latency of encoder*, in EdiT5 this is no longer the case. For instance for GEC, at 24.7 decoder steps on average required to construct the output, T5 slim spends 3.7x more time in its decoder than in encoder*. EdiT5 however spends less time in its decoder than in encoder*, as such the encoder* is now the latency bottleneck.

**Table 5: Mean latencies (in milliseconds, $\pm$ 0.01ms) measured for the components of EdiT5 and T5 models for various input lengths. EdiT5 overhead is normally input dependent, but we estimate worst-case latency.**
| Component | Len. 128 | Len. 512 |
| --- | --- | --- |
| Encoder* | 0.98 | 2.65 |
| Worst-case EdiT5 overhead | 0.49 | 1.16 |
| 1 layer decoder per-step | 0.15 | 0.17 |
| 12 layer decoder per-step | 1.26 | 1.47 |

## 5 Related work

T5 is a pre-trained, Transformer-based encoder-decoder model which has become a general-purpose tool for a variety of sequence-transduction tasks, establishing many new state-of-the-art results . However, two considerable challenges hindering the productionizing of T5-based models are the high latency caused by autoregressive decoding and the need for having a relatively large number of training examples despite the fact that pre-training makes T5 more sample efficient. Recently, it has been found that the sample efficiency problem can be mitigated by performing in-context few-shot learning, but this typically requires scaling up the model size even further , increasing the latency.

To reduce latency, a number of non-autoregressive (NAT) seq2seq methods have been proposed for neural machine translation  but a quality gap compared to autoregressive methods still exists. To decrease the gap, it is common to run the NAT methods iteratively, which, however, limits the inference speed advantage over autoregressive methods . In contrast, we show that for tasks where inputs and outputs overlap, we can maintain an order-of-magnitude speed-up without compromising on the model quality by treating the problem as a text-editing task and producing the output in a single pass.

A number of text-editing models have been proposed as a faster and more sample efficient alternative to seq2seq models like T5 . Another recently proposed approach to speed up the inference time of Transformer models is called aggressive decoding .

Closest to our work, show that adding pointing mechanism for reordering and a separate insertion model allow their text-editing model, Felix, to produce an arbitrary output in a flexible manner. Felix is a non-autoregressive model which first predicts the tokens to keep, their order, and the locations at which to insert new tokens.
Then it runs a separate model based on a BERT masked language model for inserting new tokens. In contrast, EdiT5 employs a single, end-to-end model which has an autoregressive insertion component. This enables more accurate insertions, while keeping the latency low, given that most of the tokens can be copied from the source non-autoregressively. Other text-editing models that employ autoregressive insertion include EditNTS , the text-normalization model by , Seq2Edits , ESC  and LEWIS . However, unlike EdiT5, these models perform also the edit operation prediction autoregressively, making them potentially slower at inference time.

## 6 Conclusions

In this paper we have proposed EdiT5 a low latency solution to text generation, that achieves comparable or betters results, across three distinct tasks, to a strong T5 baseline whilst achieving inference latencies that are up to 25x quicker than the baseline model.

In the future we wish to explore the following ideas: 1) The impact of distillation for EdiT5. Distillation has previously been shown to be particularly advantageous to non-autoregressive models. 2) Exploring the impact that quantization has on both latency and quality. 3) Applying EdiT5 to additional languages. EdiT5 makes no language specific assumptions and we plan to apply it to languages other than English.

## Limitations

A limitation of EdiT5, and text-editing models in general, is the assumption of overlapping text between the input and output sequences. For instance, in machine translation the overlap between source and target is minimal to none. As such EdiT5 would decode the entire target sequence, thus offering no latency saving.

An additional limitation is that all of our experiments were done on English tasks.
It is unclear how EdiT5’s pointing mechanism would behave with languages which have a less strict word-order, such as Czech.

Finally, we have measured latency only on V4 TPUs, and thus it is unclear how the performance would behave on different graphics cards or on CPUs. As such to determine if EdiT5 offers a good trade-off between quality and latency, one must measure latency on the target device.

## Acknowledgement

We thank Sebastian Krause, Sascha Rothe, and Hongkun Yu for useful
discussions, suggestions and feedback. We also thank Shankar Kumar and Felix Stahlberg for providing feedback on an earlier draft of the paper.

## Appendix A Alignment Algorithm

Figure: Algorithm 1 EdiT5 Alignment

## Appendix B Training Details

All models were trained on 4x4 or 8x8 TPUs, all EdiT5 models completed training (including EdiT5 pre-training) in under a day. T5 large pre-training large took 2 days to complete and was done using a 4x4 TPU.

### B.1 Hyper-Parameters Selection

For T5 we compared the T5 1.0 and T5 1.1 version using the base model on the validation sets and found that T5 1.1 performed better, as such used T5 1.1. For EdiT5 we used the BEA dev set, finding that T5 1.0 base performed better than T5 1.1 and selected 1.0 for all experiments.

For T5 we used the recommend fine-tuning settings, including using the adafactor optimizer , with a learning rate of 0.001. For EdiT5 we used AdamW with default settings and the default learning rate of 3e-4.

#### DiscoFuse.

For both EdiT5 and T5 we experimented with 3 different batch sizes 128, 256, 1024. For 100% and 10%, there was not a noticeable difference in the DEV set exact match performance, so we chose 1024 as it converged the quickest. For 1% and lower, we found that a batch size of 128 performed the best on the dev set.

#### Decontextualization.

For EdiT5 we experimented with the batch size 128, 256, 1024 and found that 256 offered the best exact match and used this. We also slightly modified the pre-processing code, bracketing the target sequence with [CLS] and [SEP], which helped the alignment code.

#### GEC.

For both EdiT5 and T5 we used the T5 recommended number of tokens per batch of: batch size = 512, maximum sequence length = 128. We however note that T5 used the inverse: batch size = 128, maximum sequence length = 512. For T5 and EdiT5 we disabled learning rate warmup when fine-tuning on cLang-8. Two additional hyperparameters were set for EdiT5, during pre-training on C4_200M, we noted that EdiT5 train set performance was lower than T5, as such we disabled dropout on the additional EdiT5 specific transformer layers. We additionally used the dev set to set the values of lambda for equation 10. We experimented with tagging/pointing $\lambda$ being 1, 2, 10, or equal to the number of tokens. Where $\lambda$ equal to the number of tokens produced the best results.

## Appendix C Latency measurement

To report latency for a model, we run inference on a Cloud TPU V4 chip with batch size 1 and report the time spent in computations on the device. This approach ignores some practical contributors to latency, such as memory transfers between the host and device, but we found it also reduced noise significantly, while focusing on the main performance differences between EdiT5, T5 and T5 slim (the amount of computation they each perform). To further minimize spurious latency differences, both EdiT5 and the baseline models are based on the same T5 implementation, found in TensorFlow Model Garden .