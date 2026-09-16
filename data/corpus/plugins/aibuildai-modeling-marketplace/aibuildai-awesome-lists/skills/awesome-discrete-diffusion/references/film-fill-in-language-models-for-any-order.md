---
arxiv_id: "2310.09930"
title: "FiLM: Fill-in Language Models for Any-Order Generation"
year: 2023
source: arxiv2md
---

## Abstract

Abstract Language models have become the backbone of today’s AI systems.
However, their predominant left-to-right generation limits the use of bidirectional context, which is essential for tasks that involve filling text in the middle.
We propose the F ill- i n L anguage M odel (FiLM), a new language modeling approach that allows for flexible generation at any position without adhering to a specific generation order.
Its training extends the masked language modeling objective by adopting varying mask probabilities sampled from the Beta distribution to enhance the generative capabilities of FiLM.
During inference, FiLM can seamlessly insert missing phrases, sentences, or paragraphs, ensuring that the outputs are fluent and are coherent with the surrounding context.
In both automatic and human evaluations, FiLM outperforms existing infilling methods that rely on left-to-right language models trained on rearranged text segments.
FiLM is easy to implement and can be either trained from scratch
or fine-tuned from a left-to-right language model.
Notably, as the model size grows, FiLM’s perplexity approaches that of strong left-to-right language models of similar sizes, indicating FiLM’s scalability and potential as a large language model. 1 1 1 Our code is available at https://github.com/shentianxiao/FiLM

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable success in open-ended text generation and a variety of natural language understanding and reasoning tasks .
The next word prediction objective, inherent in the training of these models, has positioned them as predominantly Causal Language Models (CLMs). However, this confines their generation order to left-to-right, constraining their versatility and applicability, particularly in tasks that require filling in the middle .

We present the Fill-in Language Model (FiLM), designed for flexible sequence generation in any desired order.
As depicted in Fig. [1](#S1.F1), FiLM exhibits the capability to fill in text segments at any specified position, taking into account both the preceding and subsequent context.
This distinctive attribute opens up avenues for a myriad of applications, including but not limited to, assisting in text editing and revision, automating template filling, and facilitating code completion.

Training FiLM takes insights from both Masked Language Models (MLMs)  and text diffusion models .
Unlike MLMs that are trained with a fixed mask ratio, FiLM adopts a strategy inspired by diffusion models that utilize varying noise levels (§[3.1](#S3.SS1)).
For each training sequence, a mask probability is drawn from the Beta distribution, and each token is masked with this probability.
The model then learns to predict the masked tokens based on the surrounding context. This adaptive masking strategy significantly enhances FiLM’s generative capacity (§[4.1](#S4.SS1)).

At decoding time, FiLM has the flexibility to start with either a sequence entirely of masks or a partially complete text interspersed with masks. It progressively replaces one mask with a predicted token at each step. We explore various decoding orders for FiLM (§[3.2](#S3.SS2)), and our analysis shows that, besides proceeding from the leftmost mask to the right, selecting the mask position with the minimum entropy is also an effective strategy (§[4.1](#S4.SS1)).

FiLM can be either trained from scratch or fine-tuned from off-the-shelf MLMs or CLMs.
In this paper, we experiment with the latter setting, which avoids the expensive pretraining stage and is appealing in practice.
We develop a method to evaluate the perplexity of any-order language models,
enabling a direct comparison between FiLM and CLM (§[3.3](#S3.SS3)).
When fine-tuned from GPT2-xl, FiLM yields perplexity of $14.03$ and $20.32$ on the WikiText-103 and One Billion Word datasets, respectively.
While these numbers lag behind those by CLM fine-tuned from GPT2-xl on the same data ($11.29$ and $16.46$), we observe a diminishing disparity with an increasing model size.
Specifically, as the pretrained model scales up from GPT2-small to GPT2-xl, the gap narrows from $5.85$ to $2.74$ and from $7.96$ to $3.86$ on the respective datasets.
This trend indicates FiLM’s promising potential with further scaling up, positing it as a viable alternative in the realm of LLMs (§[4.2](#S4.SS2)).

FiLM excels in filling text in the middle and substantially outperforms previous state-of-the-art infilling methods that employ a CLM trained on rearranged data .
In particular, in our human evaluation for story completion,
FiLM is favored in 48% of the cases over a specially trained CLM that is four times larger, while the latter is preferred in only 21%, with the remaining cases resulting in ties.

Figure: Figure 1: Flexible sequence infilling by FiLM-1.6B. The given context is in black, and the text generated by the model is in color.

## 2 Related work

FiLM can be viewed as an extension of MLMs, specifically tailored to enhance generative capabilities.
While MLMs utilizing bidirectional context have shown exceptional performance in language understanding tasks , their application in generation has been limited.
A line of work represented by T5  and BART  employs an MLM-style encoder but remains dependent on a CLM decoder.
Another approach attempts to transform MLMs into generators by interpreting them as Markov random fields or energy-based models and using Markov-chain Monte-Carlo (MCMC) sampling algorithms for decoding .
Despite the intricacy of these techniques, the generative performance of such models still lags significantly behind that of CLM.
Distinctively, FiLM is a decoder-only model that leverages bidirectional context and achieves superior text infilling performance.

Previous studies have investigated conditional MLMs for non-autoregressive machine translation , with the goal of speeding up generation by decoding multiple tokens per step .
FiLM also possesses the ability to simultaneously fill in multiple masks in each iteration.
Nonetheless, in the absence of the anchoring provided by source sentences in translation tasks, tokens often exhibit strong interdependence. Making independent predictions in such scenarios could compromise the quality of the generated text. Recognizing this challenge, this paper adopts a sequential decoding approach, filling in one mask at a time conditioned on previous predictions.
The exploration of strategies to accelerate generation with FiLM is a promising direction for future research .

Apart from FiLM, several other any-order language models possess the ability to generate text in a non left-to-right order, notably including XLNet , the General Language Model (GLM; ), and the Blank Language Model (BLM; ). XLNet employs a permutation language modeling objective for training, while GLM adopts an autoregressive blank infilling objective. However, these models have primarily been developed for language understanding tasks and generating text from scratch, with a limited focus on text infilling performance.
BLM is explicitly designed for filling in blanks, but it faces the challenge of markedly higher perplexity compared to CLM, which is a significant disadvantage.
FiLM achieves competitive perplexity with CLM, demonstrating its potential as a versatile tool for text generation.

## 3 Fill-in language model (FiLM)

Figure: Figure 2: Training and decoding of FiLM. During training, the mask probability $p$ is sampled according to a noise schedule, and then each token is independently replaced with [MASK] with probability $p$; FiLM is trained to predict the original tokens at the masked positions. At decoding time, the masks are sequentially filled in, each conditioned on the given context and previous predictions.
Refer to caption: /html/2310.09930/assets/x1.png

FiLM uses a special [MASK] token to indicate positions to be filled.
It can operate in two modes: (1) generate text from scratch by populating a sequence consisting entirely of masks;
(2) start from partial text and fill in the masked positions.
In this section, we dive in to the training of FiLM, and how to decode from it. Additionally, we extend the established perplexity evaluation for any-order language models, enabling a direct comparison between FiLM and Causal Language Models (CLMs).

### 3.1 Training

Given a training sequence $x$ consisting of tokens $(x_{1},\dots,x_{n})$, we first sample the mask probability $p$ according to a noise schedule, then independently mask each token $x_{i}$ with probability $p$.
Let $\tilde{x}=(\tilde{x}_{1},\dots,\tilde{x}_{n})$ denote the resulting masked sequence.
FiLM takes $\tilde{x}$ as input and is trained to predict the original tokens in $x$ at the masked positions of $\tilde{x}$, as illustrated in Fig. [2](#S3.F2) (Left).
The training process is designed so that FiLM leverages both the left and right contextual information available for each mask position when making predictions.

An intuitive initial choice for the noise schedule is sampling $p$ from the uniform distribution $U[0,1]$ . This approach ensures that FiLM learns to generate text from sequences with varying numbers of masks, which is crucial for generation from scratch. However, assigning equal weights across different mask probabilities can be suboptimal, as a high mask ratio leaves the model with insufficient information for predictions, while a low mask ratio oversimplifies the task.

To address this, we turn to the beta distribution, defined over the interval $[0,1]$ and characterized by two shape parameters, $\alpha$ and $\beta$.
Different values of $\alpha,\beta$ allow for skewing towards different values of mask probabilities and thereby avoiding extreme values.
Specifically, considering that the mode of Beta$(\alpha,\beta)$ is $\frac{\alpha-1}{\alpha+\beta-2}$ for $\alpha,\beta>1$, we maintain a constant sum of $\alpha+\beta$ (set to $5$ based on favorable results from our experiments), and adjust them to produce modes between $0.1$ and $0.9$ at intervals of $0.1$. The distributions obtained are depicted in Fig. [3](#S3.F3) (Left).
We empirically compare the performance of FiLM trained using different noise schedules: with $p$ sampled from the uniform distribution or from the beta distribution with varying modes, to determine the most effective strategy.

Note that when $p$ is fixed, we recover the Masked Language Modeling (MLM) objective. For instance, BERT employs a mask ratio of $0.15$ . Although training with a fixed mask ratio can be effective for representation learning, as we shall see in §[4.1](#S4.SS1), it significantly hinders the model’s capacity to generate text from scratch.

Figure: Figure 3: Left: Illustration of the Beta distribution with varying modes. Right: Perplexity of FiLM when trained with different Beta distributions, each value depicted as the deviation from the baseline perplexity achieve by the uniform distribution.
Refer to caption: /html/2310.09930/assets/figs/beta_dist.png

### 3.2 Decoding

At decoding time, given a sequence $\tilde{x}$ of incomplete text that contains masks, FiLM fills in one mask at each step, conditioning on the provided context and previous predictions.
This process is iterated until no masks remain.
Fig. [2](#S3.F2) (Right) illustrates this procedure.

When there are multiple masks in $\tilde{x}$, FiLM needs to determine which mask to fill in first.
Two straightforward strategies are: (1) making a random selection, which reflects FiLM’s training process;
(2) generating in a unidirectional manner, either from the leftmost mask to the right or vice versa. This mirrors a CLM, but with the additional conditioning on the subsequent context.

In addition, we explore two adaptive strategies based on the probability distribution predicted by the model for each mask:
(3) selecting the mask position with the minimum entropy, indicating the model’s highest certainty;
(4) selecting the position with the maximum entropy, where the model is least certain.
The min-entropy strategy acts as a heuristic to search for an “easy-first” decoding order, whereas max-entropy pursues a “hard-first” order.
Note that the generation order in these approaches is not predetermined but is established step by step during the model’s decoding process.

Upon determining the decoding order, conventional decoding algorithms of CLM—such as sampling, greedy decoding, and beam search—are equally compatible with FiLM.
In our experiments, we use sampling and evaluate the performance of FiLM under different decoding orders.
We will observe in §[4.1](#S4.SS1) that left-to-right and min-entropy decoding strategies consistently perform well, whereas random and max-entropy decoding strategies are less effective.

### 3.3 Perplexity

To calculate the perplexity of FiLM, we need to compute the probability it assigns to a sequence.
Although marginalizing over all possible sequence generation orders is intractable, evaluating FiLM with a specific decoding order is feasible.
Our method of computing perplexity can be used to evaluate other any-order language models as well.

Specifically, we first compute $p_{\text{len}}(n)$ by calculating the frequency of sequence length $n$ in the training data(^2^22We apply add-one smoothing to $p_{\text{len}}(n)$ to avoid assigning a probability of zero to unseen sequence lengths.).
Subsequently, we can determine the log-probability of generating a sequence $x=(x_{1},\dots,x_{n})$ with a decoding order $\sigma$, where $\sigma$ is an $n$-permutation.
Note that $\sigma$ can be deterministic, such as left-to-right and right-to-left; random; or adaptive, like the min-entropy and max-entropy strategies discussed in the previous subsection.

Let $\theta$ represent the model parameters. We define $p_{\theta}(x_{\sigma_{t}}|x_{\sigma_{1}},\dots,x_{\sigma_{t-1}},n)$ as the probability that FiLM predicts $x_{\sigma_{t}}$ given an $n$-length sequence with $x_{\sigma_{1}},\dots,x_{\sigma_{t-1}}$ filled.
For instance, consider $n=4$, $\sigma=(3,1,4,2)$, and $t=3$, $p_{\theta}(x_{4}|x_{3},x_{1},4)$ is the probability of predicting $x_{4}$ at the last mask position in the sequence $(x_{1},\texttt{[MASK]},x_{3},\texttt{[MASK]})$. We have:

$$ $\log p_{\theta}(x;\sigma)=\log p_{\text{len}}(n)+\sum_{t=1}^{n}\log p_{\theta}(x_{\sigma_{t}}|x_{\sigma_{1}},\dots,x_{\sigma_{t-1}},n)$ (1) $$

The perplexity is then computed as $\exp\left(-\frac{1}{n+1}\log p_{\theta}(x;\sigma)\right)$.
We divide by $n+1$ to ensure comparability with CLM,
which appends an [EOS] token to $(x_{1},\dots,x_{n})$ to signify the end of generation, resulting in a total sequence length of $n+1$.
In contrast, FiLM determines the sequence length through $p_{\text{len}}(n)$ and then fills in an $n$-length sequence.
This calculated perplexity serves as a metric not only for comparison with CLM but also for evaluating the effectiveness of various training and decoding strategies of FiLM.

## 4 Experiments

In this section, we first empirically analyze FiLM’s various training and decoding strategies, as discussed in §[3.1](#S3.SS1) and §[3.2](#S3.SS2), to identify the optimal configuration (§[4.1](#S4.SS1)).
Subsequently, we compare FiLM with CLM on language modeling (§[4.2](#S4.SS2)).
Lastly, we test FiLM for text infilling (§[4.3](#S4.SS3)) and story completion (§[4.4](#S4.SS4)) and design evaluation protocols to compare it with previous infilling methods in terms of fluency, coherence, and logical consistency with the surrounding context.

#### Datasets

We conduct experiments on three datasets: WikiText-103 (WT-103; ), One Billion Word (1BW; ), and ROCStories .
WikiText-103 is a collection of Wikipedia articles with $103$M words in total. Each article has several thousand words, which we chunk into windows of $512$ tokens.
One Billion Word is a sentence-level dataset, with an average length of $28.5$ tokens and a total of $1$B words.
ROCStories consists of five-sentence commonsense stories, each averaging $51.4$ tokens in length, totaling $5$M words.

#### Experimental setup

We evaluate the performance of FiLM when fine-tuned from both an MLM and a CLM.
For this investigation, we employ two pretrained models: RoBERTa , representing MLMs, and GPT2 , representing CLMs.
Note that the causal masking should be deactivated when fine-tuning FiLM from a CLM.

RoBERTa is available in two sizes: base ($124$M parameters) and large ($355$M), while GPT2 is offered in four sizes: small ($124$M), medium ($355$M), large ($774$M), and xl ($1558$M).
The models are trained using the Adam optimizer with a learning rate of $2\mathrm{e}-5$ and a batch size of $20$K tokens.
Training is conducted for $500$K steps on WikiText-103 and One Billion Word datasets, and for $50$K steps on ROCStories.
Utilizing automatic mixed precision (AMP), our largest model based on GPT2-xl takes about one week to train using two $80$G A100 GPUs.

### 4.1 Analysis of FiLM

In the following analysis, we compare the perplexity of FiLM (§[3.3](#S3.SS3)) under various training noise schedules (§[3.1](#S3.SS1)) and decoding orders (§[3.2](#S3.SS2)) to determine the most effective strategy. We use pretrained models RoBERTa-base and GPT2-small here for efficiency, and report results on the validation sets of WikiText-103 and One Billion Word.

We first examine the impact of employing different noise schedules for training FiLM.
We use the left-to-right decoding here, with an in-depth investigation into decoding orders to follow.
Table [1](#S4.T1) shows that a fixed noise schedule $\delta(0.15)$ leads to substantially higher perplexity on both datasets, highlighting the necessity for a variable mask probability $p$ in order to enhance generative capacity.
To elucidate the effects of training with the Beta distribution featuring varying modes, we plot the difference in perplexity compared to $U[0,1]$ in Fig. [3](#S3.F3) (Right).
The findings corroborate our hypothesis that overly small or large values for the mode of $p$ are suboptimal.
The lowest perplexity is achieved at Beta$(2.5,2.5)$ with mode $0.5$,
marking an improvement of approximately $0.4$ over that of $U[0,1]$.
In light of these results, we adopt the Beta$(2.5,2.5)$ noise schedule in subsequent experiments.

Next, we investigate the effects of decoding from FiLM in different orders. The results are presented in Table [2](#S4.T2).
While FiLM is trained to predict a random subset of words, decoding from left to right substantially outperforms decoding in a random order.
This finding is consistent with the inherent sequential nature of language.
Intriguingly, when fine-tuned from an order-agnostic MLM, FiLM demonstrates near-optimal performance with right-to-left decoding, even attaining the lowest perplexity on One Billion Word.
However, when fine-tuned from a left-to-right CLM, the efficacy of right-to-left decoding significantly degrades, lagging behind even random decoding.
This illustrates the resistance in altering a CLM’s behavior from left-to-right to right-to-left.

The min-entropy order consistently emerges as the second-best strategy, while the max-entropy order proves to be the least effective.
Fig. [4](#S4.F4) showcases the decoding process for both min-entropy and max-entropy orders.
The min-entropy strategy generates text in a segmented manner, sequentially predicting cohesive phrases such as “thank you” and “for your service”, deferring the more uncertain name after “Mr.” to the final stage.
In contrast, the max-entropy strategy opts for a “challenging” order, selecting distant positions at each step.

Given the simplicity and superior performance of left-to-right decoding, irrespective of whether FiLM is fine-tuned from an MLM or a CLM, we select this order for FiLM in subsequent experiments.

**Table 1: Perplexity of FiLM under different training noise schedules.**
| Dataset | Pretrained model | $\delta(0.15)$ | $U[0,1]$ | Beta$(2.5,2.5)$ |
| --- | --- | --- | --- | --- |
| WT-103 | RoBERTa-base | 26.76 | 19.02 | 18.59 |
| 1BW | RoBERTa-base | 34.68 | 30.56 | 30.15 |

**Table 2: Perplexity of FiLM using different decoding orders. The bold numbers highlight the best perplexity, while the underlined ones denote the second best.**
| Dataset | Pretrained model | Random | L2R | R2L | Min-Ent | Max-Ent |
| --- | --- | --- | --- | --- | --- | --- |
| WT-103 | RoBERTa-base | 19.95 | 18.59 | 19.04 | 18.70 | 21.38 |
| GPT2-small | 28.42 | 21.68 | 29.34 | 23.47 | 30.68 |  |
| 1BW | RoBERTa-base | 31.85 | 30.15 | 29.92 | 30.07 | 33.16 |
| GPT2-small | 41.52 | 32.55 | 42.73 | 34.59 | 44.04 |  |

Figure: Figure 4: An illustration of FiLM decoded using min-entropy and max-entropy orders. The mask position selected to be filled in at each step is highlighted in the green color.

### 4.2 Language modeling

In this subsection, we evaluate the performance of FiLM on language modeling and compare it with CLM. Both FiLM and CLM are fine-tuned on WikiText-103 and One Billion Word, using pretrained models of varying sizes. The resulting perplexities are plotted in Fig. [5](#S4.F5).

The choice of the pretrained model has a significant influence on the performance of FiLM. When fine-tuned from RoBERTa, which incorporates bidirectional context during pretraining, FiLM demonstrates superior performance compared to when it is fine-tuned from GPT2, which leverages only unidirectional context during pretraining.
We hypothesize that employing FiLM directly as the pretraining objective might unlock further enhancements in model performance.

While FiLM exhibits higher perplexity than CLM when generating text from scratch, this gap diminishes as the model size increases. As the pretrained model scales from $124$M GPT2-small to $1558$M GPT2-xl, the difference in perplexity decreases from $5.85$ to $2.74$ on WikiText-103 and from $7.96$ to $3.86$ on One Billion Word.
This narrowing gap suggests that FiLM could benefit from further scaling and holds considerable potential as an alternative LLM.

Figure: Figure 5: Perplexity of FiLM and CLM on WikiText-103 (Left) and One Billion Word (Right).
Refer to caption: /html/2310.09930/assets/figs/ppl.png

### 4.3 Text infilling

We evaluate FiLM’s text infilling performance using the WikiText-103 and One Billion Word datasets. For a given sequence $x$ of length $n$, we first sample the number of spans $m$ from $1$ to $5$. Subsequently, we draw $2m$ numbers from $1$ to $n$ without replacement and sort them to get $a_{1},\dots,a_{2m}$ as the endpoints of each span. Tokens in $x$ located between $[a_{2i-1},a_{2i})$ ($i=1,\dots,m$) are masked, and the model is tasked with filling in these spans.
Given that WikiText-103 comprises long documents, the infilling tasks on this dataset involve composing multiple sentences. In contrast, the One Billion Word dataset consists of individual sentences, and the infilling tasks are primarily at the phrase level.

Previous state-of-the-art methods for infilling have predominantly relied on training CLMs on rearranged data .
In these methods, random spans of text are replaced with special sentinel tokens and moved to the end of the sequence. Then a CLM is trained to generate text in this modified order.
For instance, the manipulated sequence for the example in Fig. [2](#S3.F2) would appear as “They [MASK:0] good [MASK:1] cream [FILL:0] have really [FILL:1] ice”.
At test time, the tokens generated after the given context “They [MASK:0] good [MASK:1] cream [FILL:0]” are re-integrated into the corresponding mask positions.
This approach is referred to as causal masking (CM).
We fine-tune both FiLM and CM from GPT2-xl and use top-p sampling with a threshold of $0.95$ and a temperature of $0.8$ for decoding.

To evaluate the model outputs, we compute the ROUGE scores  against the original text to measure their overlap.
Specifically, ROUGE-1, ROUGE-2, and ROUGE-L measure the overlap of unigrams, bigrams, and the longest common subsequence between the generated and reference texts, respectively.
Since there may be valid infillings different from the original, we also employ GPT4 for evaluation .
We present the outputs generated by FiLM and CM to GPT4 in a random order, and instruct it to determine which option is more grammatically fluent and coherent with the surrounding context. When neither option is more fitting than the other, GPT4 is directed to declare a tie. This GPT4 evaluation is conducted on $500$ examples from the test set.

As shown in Fig. [6](#S4.F6), FiLM demonstrates a significant advantage over CM, improving the average ROUGE score by $10.12$ on WikiText-103 and $7.35$ and One Billion Word.
Moreover, GPT4 prefers the outputs by FiLM over CM with margins of $8.4\%$ and $5.4\%$ on the respective datasets.
Fig. [7](#S4.F7) displays example infillings, where GPT4 accurately identifies repetitions and inconsistencies generated by CM.
Despite CM’s attempt to consider subsequent context by artificially altering the text order,
it still introduces redundancy by inserting “stress” between “Depression,” and “and stress”, and commits a logical error by adding
“farming, and a world leader in” between “Switzerland is the source of Europe’s biggest rivers, supporting agriculture and” and “nuclear power stations”.
In contrast, FiLM generates apt fillings such as “loneliness” and “the construction of new”, resulting in coherent sentences.
Additional examples are available in Fig. [10](#A0.F10) and Fig. [11](#A0.F11) in the Appendix.

Figure: Figure 6: Text infilling results on WikiText-103 and One Billion Word. The left table presents the ROUGE scores, and the right figure illustrates the comparative evaluation by GPT4.
Refer to caption: /html/2310.09930/assets/figs/compare_infill.png

Figure: Figure 7: Text infilling examples from One Billion Word, evaluated by GPT4. The provided context is in black, and the model generated text is in color. The number before each model denotes the option number presented to GPT4 for evaluation.

### 4.4 Commonsense story completion

In this set of experiments, we assess FiLM’s proficiency in logically completing commonsense stories using the ROCStories dataset.
For each story, we randomly remove one of the five sentences and ask the model to fill it in.
In addition to ROUGE scores and GPT4 evaluation, we conduct human evaluation to determine which model output best preserves the story’s logical flow and coherence. We present $100$ examples to human judges and collect two labels for each.
Under the three categories “Option 1 is better”, “Option 2 is better”, and “Tie”, GPT4 and human evaluators reach consensus $49.5\%$ of the time, while the agreement between humans is $62\%$. Table [3](#A0.T3) in the Appendix provides further details of the agreement analysis.

Due to the relatively small size of the ROCStories dataset, we find that FiLM, when fine-tuned from the bidirectional RoBERTa-large, achieves a lower loss ($2.63$) compared to when fine-tuned from the unidirectional GPT2-xl ($2.90$), despite the former having only a quarter of the parameters of the latter.
Therefore, we choose RoBERTa-large as the base model for FiLM here.
We compare FiLM against the causal masking model (CM) fine-tuned from GPT2-xl, using top-p sampling with a threshold of $0.95$ and a temperature of $0.2$ for decoding.

The results depicted in Fig. [8](#S4.F8) indicate that FiLM outperforms CM by an average of $4.6$ ROUGE score points. Moreover, FiLM is preferred by both GPT4 and human evaluators, with preference margins of $19\%$ and $27\%$, respectively. Fig. [9](#S4.F9) showcases several story completions. While CM’s generated sentence aligns with the prior context, it struggles to link appropriately with the sentences that follow. In contrast, FiLM’s outputs exhibit seamless integration, like mentioning “started to itch” considering the subsequent allergy, and adding “But he didn’t like to eat it until it was ready” before “So he decided this time he would sneak a piece before dinner”, thereby preserving the story’s logical flow. For more examples illustrating their qualitative difference, please refer to Fig. [12](#A0.F12) in the Appendix.

Figure: Figure 8: Story completion results on ROCStories. The left table presents the ROUGE scores, and the right figure illustrates the comparative evaluation by both GPT4 and human judges.
Refer to caption: /html/2310.09930/assets/figs/compare_story.png

Figure: Figure 9: Story completion examples from ROCStories, evaluated by GPT4. The provided context is in black, and the model generated text is in color. The number before each model denotes the option number presented to GPT4 for evaluation.

## 5 Conclusion

In this paper, we have introduced FiLM, a novel language modeling approach capable of flexibly generating output sequences in any order.
Analogous to MLMs, FiLM is trained to predict masked tokens conditioning on the surrounding context from both sides.
Its masking probabilities are randomly drawn from a Beta distribution, allowing FiLM to generate sequences from scratch.

We have devised a method to compute the perplexity of any-order language models, including FiLM, to facilitate direct comparisons with CLM.
In language modeling, FiLM’s perplexity performance is competitive, nearing that of comparable CLMs as the models scale up in size.
This suggests FiLM’s promising potential as a new bidirectional LLM.
FiLM excels in text infilling and story completion tasks,
where it outperforms strong baselines in terms of both automatic and human evaluations.