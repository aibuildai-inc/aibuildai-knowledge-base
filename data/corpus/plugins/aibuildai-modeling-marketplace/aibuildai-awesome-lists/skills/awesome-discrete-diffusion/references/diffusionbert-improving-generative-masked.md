---
arxiv_id: "2211.15029"
title: "DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models"
year: 2023
source: arxiv2md
---

## Abstract

Abstract We present DiffusionBERT, a new generative masked language model based on discrete diffusion models.
Diffusion models and many pre-trained language models have a shared training objective, i.e., denoising , making it possible to combine the two powerful models and enjoy the best of both worlds.
On the one hand, diffusion models offer a promising training strategy that helps improve the generation quality.
On the other hand, pre-trained denoising language models (e.g., BERT) can be used as a good initialization that accelerates convergence.
We explore training BERT to learn the reverse process of a discrete diffusion process with an absorbing state and elucidate several designs to improve it.
First, we propose a new noise schedule for the forward diffusion process that controls the degree of noise added at each step based on the information of each token.
Second, we investigate several designs of incorporating the time step into BERT.
Experiments on unconditional text generation demonstrate that DiffusionBERT achieves significant improvement over existing diffusion models for text (e.g., D3PM and Diffusion-LM) and previous generative masked language models in terms of perplexity and BLEU score. 1 1 1 Our code is publicly available at https://github.com/Hzfinfdu/Diffusion-BERT

## 1 Introduction

Diffusion models  have recently emerged as a new class of state-of-the-art generative models, achieving high-quality synthesis results on image data . Though these models captured widespread attention from not only the research community but also the public, applying diffusion models to text data is still challenging and under-explored due to the discrete nature of the text. A few prior works that explored using diffusion models on text data can be divided into two lines. The first is to extend diffusion models to discrete state spaces . The second is to perform the diffusion process and its reverse process in the continuous domain and bridge the continuous and the discrete domain through embedding and rounding . However, none of these works leveraged pre-trained language models (PLMs, ), which are an unmissable treasure in the NLP community.

This work, to our knowledge, is the first attempt to combine diffusion models with PLMs. Such a combination is built upon a shared training objective between diffusion models and PLMs, i.e., denoising. Diffusion models consist of a forward process (data to noise) and a reverse process (noise to data). In the forward process, a small amount of noise is gradually added to the data. Then, a neural network ($p_{\theta}$ in Figure [1](#S0.F1)) is employed to learn the reverse process step by step, i.e., learn to denoise. Such a denoising neural network is naturally related to a wide class of PLMs that are pre-trained with denoising objectives such as BERT  and BART . Hence, pre-trained denoising language models can serve as a good start point to learn the reverse diffusion process. On the other hand, diffusion models also offer a promising training strategy for generative PLMs.
In contrast to commonly used generative PLMs (e.g., GPT ) that relies on an autoregressive factorization of the joint probability, diffusion models provide another way of factorization along the dimension of time and therefore allow the model to be not necessarily autoregressive. Thus, diffusion models can be combined with a variety of PLMs that may not be pre-trained for generation.

In the discrete domain, the forward diffusion process can be implemented by a chain of transition matrices that gradually corrupt the clean text. As shown in Figure [1](#S0.F1), the clean text "Hello world !" is gradually corrupted into "[MASK] [MASK] [MASK]" during the diffusion process. In this work, we explore using pre-trained denoising language models (e.g., BERT) to learn the reverse diffusion process and demonstrate their advantages in accelerating convergence and improving generation quality. Further, we propose a new noise schedule of the forward process based on the principle of distributing the corrupted information uniformly across the forward process. The noise schedule, called spindle schedule, generates noise for $\mathbf{x}_{t}$ conditioned not only on $\mathbf{x}_{t-1}$ but also on $\mathbf{x}_{0}$, making the forward process non-Markovian without changing the original training objective. Note that the denoising model takes as input $\mathbf{x}_{t}$ and time step $t$ to predict $\mathbf{x}_{t-1}$, where $t$ is unseen during the pre-training of language models so we investigate several ways of incorporating the time step into PLMs. As a result, we find that the best result is achieved by throwing away the time information, which we call time-agnostic decoding (TAD).

Experimental results on unconditional text generation demonstrate the benefit of combining diffusion models with PLMs: the proposed DiffusionBERT significantly improves the generation quality over existing diffusion models for text generation (e.g., D3PM  and Diffusion-LM ) and previous generative masked language models (e.g., BERT-Mouth ).
The effectiveness of the proposed spindle schedule and time-agnostic decoding is confirmed by ablation studies. In a nutshell, DiffusionBERT enjoys the best of both worlds.

**Table 1: Examples generated by three generative masked language models showing the difference of noise schedule and generation quality.**
| BERT-Mouth | $t=0$ | [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] |
| --- | --- | --- |
| $t=8$ | [MASK] of [MASK] five [MASK] remain [MASK] in [MASK] . |  |
| $t=16$ | two of [MASK] five structures remain [MASK] this location . |  |
| $t=24$ | five of [MASK] the windows remain at this location . |  |
| $t=32$ | most of even the windows stand still this day . |  |
| D3PM | $t=0$ | [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] |
| $t=8$ | [MASK] [MASK] [MASK] [MASK] been [MASK] [MASK] [MASK] [MASK] . |  |
| $t=16$ | [MASK] [MASK] [MASK] [MASK] been [MASK] [MASK] the [MASK] . |  |
| $t=24$ | [MASK] [MASK] [MASK] also been [MASK] by the [MASK] . |  |
| $t=32$ | the man has also been arrested by the police . |  |
| DiffusionBERT | $t=0$ | [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] |
| $t=8$ | [MASK] , [MASK] [MASK] [MASK] [MASK] [MASK] that [MASK] . |  |
| $t=16$ | today , [MASK] will be [MASK] [MASK] that [MASK] . |  |
| $t=24$ | today , [MASK] will be remembered for that mistake . |  |
| $t=32$ | today , he will be remembered for that mistake . |  |

## 2 Background

### 2.1 Diffusion Models

Diffusion models  are a class of latent variable models that are originally designed for continuous domains. A diffusion model is consisting of a forward diffusion process and a reverse diffusion process. Given a sample $\mathbf{x}_{0}\sim q(\mathbf{x}_{0})$, a Markov chain of latent variables $\mathbf{x}_{1},\cdots,\mathbf{x}_{T}$ are produced in the forward process by progressively adding a small amount of Gaussian noise to the sample:

$$ $q(\mathbf{x}_{t}|\mathbf{x}_{t-1})=\mathcal{N}(\mathbf{x}_{t};\sqrt{1-\beta_{t}}\mathbf{x}_{t-1},\beta_{t}\mathbf{I}),$ (1) $$

where $\{\beta_{t}\in(0,1)\}_{t=1}^{T}$ is a noise schedule controlling the step size of adding noise. Eventually $\mathbf{x}_{T}$ becomes an isotropic Gaussian distribution. If $\beta_{t}$ is small enough, the reverse process $q(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ is also a Gaussian, which is learned by a parameterized model

$$ $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t},t)=\mathcal{N}(\mathbf{x}_{t-1};\mu_{\theta}(\mathbf{x}_{t},t),\Sigma_{\theta}(\mathbf{x}_{t},t)),$ (2) $$

where $\mu_{\theta}(\cdot)$ and $\Sigma_{\theta}(\cdot)$ can be implemented by a U-Net or a Transformer. When conditioning also on $\mathbf{x}_{0}$, $q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})$ has a closed form so we can manage to minimize the variational lower bound to optimize $\log p_{\theta}(\mathbf{x}_{0})$:

$$ $\displaystyle\mathcal{L}_{\text{vlb}}=\mathbb{E}_{q}[D_{\text{KL}}(q(\mathbf{x}_{T}|\mathbf{x}_{0})\parallel p_{\theta}(\mathbf{x}_{T}))]$ $\displaystyle+\mathbb{E}_{q}[\sum_{t=2}^{T}D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})\parallel p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t},t))]$ $\displaystyle-\log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1}),$ (3) $$

where $\mathbb{E}_{q}(\cdot)$ denotes the expectation over the joint distribution $q(\mathbf{x}_{0:T})$.

### 2.2 Diffusion Models in Discrete Domain

For discrete domains, each element of $\mathbf{x}_{t}$ is a discrete random variables with $K$ categories. For text data, $K=|V|$ is the size of the vocabulary. Denote $\mathbf{x}_{t}$ as a stack of one-hot vectors, the process of adding noise can be written as

$$ $q(\mathbf{x}_{t}|\mathbf{x}_{t-1})=\texttt{Cat}(\mathbf{x}_{t};\mathbf{p}=\mathbf{x}_{t-1}\mathbf{Q}_{t}),$ (4) $$

where $\texttt{Cat}(\cdot)$ is a category distribution and $\mathbf{Q}_{t}$ is a transition matrix that is applied to each token in the sequence independently: $[\mathbf{Q}_{t}]_{i,j}=q(x_{t}=j|x_{t-1}=i)$. It is easy to obtain that

$$ $\displaystyle q(\mathbf{x}_{t-1}|$ $\displaystyle\mathbf{x}_{t},\mathbf{x}_{0})=\frac{q(\mathbf{x}_{t}|\mathbf{x}_{t-1},\mathbf{x}_{0})q(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})}$ $\displaystyle=\texttt{Cat}$ $\displaystyle\left(\mathbf{x}_{t-1};\mathbf{p}=\frac{\mathbf{x}_{t}\mathbf{Q}_{t}^{\top}\odot\mathbf{x}_{0}\overline{\mathbf{Q}}_{t-1}}{\mathbf{x}_{0}\overline{\mathbf{Q}}_{t}\mathbf{x}_{t}^{\top}}\right),$ (5) $$

where $\overline{\mathbf{Q}}_{t}=\mathbf{Q}_{1}\mathbf{Q}_{2}\cdots\mathbf{Q}_{t}$. Note that $\odot$ is element-wise multiplication and the division is row-wise.

With $q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})$ at hand, according to Eq. ([2.1](#S2.Ex1)), we can use a parameterized model $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t},t)$ to learn the reverse diffusion process.

## 3 DiffusionBERT

In contrast to recently proposed diffusion models for text, e.g., Diffusion-LM  and DiffuSeq , which are based on continuous diffusion models, we instead explore discrete diffusion models to integrate PLMs as the backbone. We first introduce a specific instance of discrete diffusion models , which considers a transition matrix with an absorbing state for the sake of using PLMs (§ [3.1](#S3.SS1)). Secondly, we introduce a new noise schedule of the forward diffusion process, called spindle schedule, which is based on the principle of distributing the corrupted information uniformly across the forward process (§ [3.2](#S3.SS2)). Then, we investigate several alternatives of incorporating the time step into PLMs for predicting $\mathbf{x}_{t-1}$ given $\mathbf{x}_{t}$ and $t$ (§ [3.3](#S3.SS3)).

### 3.1 Diffusion Models with a Discrete Absorbing State

To be combined with pre-trained denoising language models, we incorporate an absorbing state, e.g., [MASK] for BERT, in the Markov process. In particular, each token in the sequence either stays the same or transitions to [MASK] with some probability. Formally, each entry of the transition matrix at step $t$ is as follows,

$$ $\displaystyle[\mathbf{Q}_{t}]_{i,j}=\begin{cases}1&\text{if}\ i=j=\texttt{[M]},\\ \beta_{t}&\text{if}\ j=\texttt{[M]},i\neq\texttt{[M]},\\ 1-\beta_{t}&\text{if}\ i=j\neq\texttt{[M]},\end{cases}$ (6) $$

where [M] is the abbreviation of [MASK]. Such a Markov process converges to a stationary distribution $q(\mathbf{x}_{T})$, which places all probability mass on a sequence with all [MASK] tokens.

The $t$-step marginal $q(\mathbf{x}_{t}^{i}|\mathbf{x}_{0}^{i})$ can be easily obtained in a closed form,

$$ $\displaystyle q(\mathbf{x}_{t}^{i}|\mathbf{x}_{0}^{i})=\begin{cases}\overline{\alpha}_{t}&\text{if}\ \mathbf{x}_{t}^{i}=\mathbf{x}_{0}^{i},\\ 1-\overline{\alpha}_{t}&\text{if}\ \mathbf{x}_{t}^{i}=\texttt{[M]},\end{cases}$ (7) $$

where $\overline{\alpha}_{t}=\prod_{i=1}^{t}(1-\beta_{i})$, $\mathbf{x}_{t}^{i}$ denotes the $i$-th token in the sequence at step $t$. Combining with Eq. ([2.1](#S2.Ex1)) and ([2.2](#S2.Ex3)), we can derive a training objective to optimize $p_{\theta}(\mathbf{x_{t-1}|\mathbf{x}_{t}},t)$ and generate a sample by performing the reverse diffusion process:

$$ $p_{\theta}(\mathbf{x}_{0:T})=p(\mathbf{x}_{T})\prod_{t=1}^{T}p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t},t).$ (8) $$

### 3.2 Spindle Noise Schedule

The noise schedule in the continuous domain, such as the linear schedule  and the cosine schedule , has shown to be important to the performance of diffusion models.

In contrast to the continuous domain where the noise can be easily controlled by the variance of the Gaussian, (1) it is less obvious how to control the degree of noise added at each step in the discrete domain. For the discrete domain, the noise schedule $\beta_{t}=(T-t+1)^{-1}$ has been explored for the case of the uniform transition matrix  and the absorbing-state transition matrix . However, (2) such a schedule assumes all tokens carry the same amount of information and does not consider the linguistic difference among the tokens in a sequence. Besides, (3) it violates the easy-first-generation nature of denoising language models. That is, the model tends to generate tokens that are most frequently appearing (and is least surprising) in the training corpus to achieve a higher likelihood. As the context becomes richer, more details come up in the sequence.

To address the above issues, we consider a noise schedule that (1) measures the added noise at each step by the corrupted information and encourage the corrupted information to be uniformly distributed across the diffusion steps. Since the information is measured independently for each token, (2) different tokens in a sequence are assigned different probabilities of transitioning to the [MASK] token. Moreover, inspired by the easy-first-generation phenomenon, (3) we put the tokens in a sequence in descending order of their information and divide them into $T$ buckets. Each bucket is ensured to contain the same amount of information. That is, we mask the most informative tokens at the start of the forward process and mask the least informative tokens at the end of the forward process such that the learnable reverse process follows an easy-first generative behavior.

In particular, distributing corrupted information uniformly across the forward steps can be formally described by

$$ $1-\frac{t}{T}=\frac{\sum_{i=1}^{n}H(\mathbf{x}_{t}^{i})}{\sum_{i=1}^{n}H(\mathbf{x}_{0}^{i})}=\frac{\sum_{i=1}^{n}\overline{\alpha}_{t}^{i}H(\mathbf{x}_{0}^{i})}{\sum_{i=1}^{n}H(\mathbf{x}_{0}^{i})},$ (9) $$

where $H$ denotes the entropy, which measures the amount of information of a random variable, $\mathbf{x}^{i}$ denotes the $i$-th token in the sequence and $n$ denotes the length of the sequence. According to Eq. ([7](#S3.E7)), $\overline{\alpha}_{t}^{i}=\prod_{j=1}^{t}(1-\beta_{j}^{i})$ denotes the probability that the $i$-th token remains the same at step $t$, i.e., $\mathbf{x}_{t}^{i}=\mathbf{x}_{0}^{i}$.
We expect that $\overline{\alpha}_{t}^{i}>\overline{\alpha}_{t}^{j}$ if $H(\mathbf{x}_{t}^{i})<H(\mathbf{x}_{t}^{j})$ such that easy (low-information) tokens emerges earlier than hard (high-information) tokens during the reverse process.

Considering these aforementioned properties, we construct $\overline{\alpha}_{t}^{i}$ as follows,

$$ $\overline{\alpha}_{t}^{i}=1-\frac{t}{T}-S(t)\cdot\tilde{H}(\mathbf{x}_{0}^{i}),$ (10) $$

$$ $S(t)=\lambda\sin\frac{t\pi}{T},$ (11) $$

$$ $\tilde{H}(\mathbf{x}_{0}^{i})=1-\frac{\sum_{j=1}^{n}H({\mathbf{x}_{0}^{j}})}{nH({\mathbf{x}_{0}^{i}})},$ (12) $$

where $S(t)$ is introduced to control the effect of the informativeness at time step $t$. It is designed to be sinusoidal to ensure $S(0)=S(T)=0$ such that $\mathbf{x}_{t}$ can retain all (zero) information when $t=0$ ($t=T$). The effect of $S(t)$ is controlled by a hyperparameter $\lambda$. When $\lambda=0$, the noise schedule is degraded to $\beta_{t}=(T-t+1)^{-1}$ as in . Figure [2](#S3.F2) shows how $\overline{\alpha}$ progresses during the forward process. The schedule is named as spindle due to the shape of the probability curves.

In our proposed schedule, the transition probability at time step $t$ depends not only on the current state but also on the original text, making the forward diffusion process non-Markovian. Nevertheless, as revealed by Eq. ([2.2](#S2.Ex3)), this does not change the original training objective.

Figure: Figure 2: Each token in a sequence has a specific noise schedule depending on how much information is lost when they are masked. For instance, in the sentence "Bella is sitting over there.", "Bella" is the most informative word. Thus it is encouraged to be masked at the early stage so that our model learns to recover it in the last place.
Refer to caption: /html/2211.15029/assets/src/wf_example.jpg

### 3.3 The Design Space of Feeding Time Steps

Typically, a diffusion model takes as input a noised sample and the time step to predict the denoised sample during the reverse process, i.e., $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t},t)$. However, $t$ is an additional variable that is unseen during the pre-training of language models and therefore it is less trivial how to feed the time information into the PLMs. Here we explore three design choices of feeding time steps.

#### Layer-wise Time Embedding

A straightforward choice is to include the time step as the same way as positional encoding, i.e., using the Transformer sinusoidal embedding or a learnable MLP in each Transformer layer. Note that this way is commonly adopted in previous work .

#### Prefix Time Embedding

Prompting language models by prepending trainable soft tokens to the input sequence has shown promising results recently . Hence, we also explore including a time step token embedding $\mathbf{v}(t)$ as a prefix of the input token embeddings $\langle\mathbf{v}(\mathbf{x}_{t}^{1}),\mathbf{v}(\mathbf{x}_{t}^{2}),\cdots,\mathbf{v}(\mathbf{x}_{t}^{n})\rangle$. In particular, the time step token is inserted in between the [CLS] token and the input sequence. These added time step token embeddings are trained along with the PLM.

#### Time-Agnostic Decoding

Another alternative is not to explicitly incorporate the time step $t$ because it can be implied by the noised sample $\mathbf{x}_{t}$.
In contrast to the image data, it is easier to implicitly infer the diffusion time step by counting the number of corrupted tokens (i.e., [MASK]) in the noised sequence. In this way, the PLM has to perform iterative decoding while being ignorant of the current time step, i.e., $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})$.

## 4 Experiments

**Table 2: Main results on LM1B. The methods proposed in this work are marked with wavy lines. The best results are in bold and the second best results are underlined. LTE: layer-wise time embedding. PTE: prefix time embedding. TAD: time-agnostic decoding.**
| Method | Pretrained | Schedule | Time Step | PPL $\downarrow$ | BLEU $\uparrow$ | Self-BLEU $\downarrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| D3PM | ✗ | $(T-t+1)^{-1}$ | LTE | 82.34 | 0.3897 | 0.2347 |
| TAD | 125.15 | 0.3390 | 0.2720 |  |  |  |
| Spindle | LTE | 77.50 | 0.4241 | 0.2288 |  |  |
| Diffusion-LM | ✗ | Cosine | LTE | 118.62 | 0.3553 | 0.2668 |
| ✓ | Cosine | LTE | 132.12 | 0.3562 | 0.2798 |  |
| BERT-Mouth | ✓ | - | - | 142.89 | 0.2867 | 0.1240 |
| DiffusionBERT | ✓ | $(T-t+1)^{-1}$ | LTE | 92.53 | 0.3995 | 0.2118 |
| PTE | 79.95 | 0.3886 | 0.2156 |  |  |  |
| TAD | 78.76 | 0.4213 | 0.2116 |  |  |  |
| Spindle | TAD | 63.78 | 0.4358 | 0.2151 |  |  |

### 4.1 Experimental Setup

We mainly focus on unconditional text generation in complex scenarios where the training data covers a wide range of topics and is composed of a large vocabulary. Experiments are conducted on the One Billion Word dataset (LM1B) . LM1B is a language corpus with about 30 million sentences and a vocabulary of about 793k. We use the standard train-test split and take 1% of the training set for validation. All text data are lower-cased to align with the settings of .

Our DiffusionBERT is based on Bert-Base-Uncased with about 110M parameters. We train DiffusionBERT using the AdamW optimizer  for 1.9 million steps with learning rate of 3e-6, dropout probability of 0.1, batch size of 32. For the first 10K steps, we use a linear warmup schedule starting from learning rate of 1e-8. All experiments are conducted on NVIDIA A100 Tensor Core GPUs. We use 4 GPUs for training and a single GPU for sampling.

### 4.2 Baselines

We conduct comparison on unconditional text generation against several non-autoregressive (NAR) baselines: D3PM , Diffusion-LM , and BERT-Mouth .(^2^22Another strong baseline of NAR text generation is SUNDAE but unfortunately there is no public implementation available. We will include comparison with SUNDAE in later versions by directly using the results reported in the original paper and use the same settings to train DiffusionBERT for fair comparison.)

#### D3PM

D3PM is a general framework of discrete diffusion models. We implement an instance of D3PM with the absorbing state and a layer-wise time embedding. Both DiffusionBERT and D3PM are implemented with a sequence length $n=128$ and diffusion steps $T=2048$. During inference, we perform decoding with 16 time steps in each iteration. The total inference cost is 128 iterations, which is smaller than that chosen in existing diffusion or diffusion-like models for unconditional generation . This has no impact on our conclusions since increasing the diffusion step does not bring substantial improvement.

#### Diffusion-LM

Diffusion-LM learns an embedding to map discrete text into the continuous space where it performs Gaussian diffusion process. A rounding step is required to map the continuous embeddings into discrete texts. We re-implemented Diffusion-LM with the model architecture of BERT and diffusion steps $T=2000$. Since the performance drop of Diffusion-LM is bigger than D3PM and DiffusionBERT when we sample less steps during generation, we do not skip steps so the number of inference is about 4 times that of DiffusionBERT and the exact generation time comparison is reported in § [4.5](#S4.SS5).

#### BERT-Mouth

BERT-Mouth samples text from BERT via order-agnostic autoregressive masked language modeling. Starting from a sequence of [MASK], BERT samples one token at each time step in random order. Another option is decoding from left to right, like autoregressive models. In our preliminary experiments, we find that random position sampling performs better. We continue pretraining BERT on LM1B to adapt BERT to downstream training corpus.

### 4.3 Main Results

Our main results are included in Table [2](#S4.T2). We choose BLEU-4 as the metric for generation quality and diversity. For each method, we sample 1K text for evaluating BLEU score and another 1K for self-BLEU. Note that with different sampling strategy, the BLEU/self-BLEU results may vary. For fair comparison, the sentences sampled by D3PM and DiffusionBERT have a fixed length $n=64$ and are sampled by a top-$K$ filter where $K=30$. Diffusion-LM and BERT-Mouth are trained and sampled following their original implementation. Overall, DiffusionBERT achieves the best generation quality and diversity trade-off among the considered NAR methods. Besides, the perplexity of DiffusionBERT with the spindle noise schedule is substantially higher. Evidence of lower bound is used as a proxy of the perplexities of DiffusionBERT and D3PM since the exact likelihood of diffusion models is intractable.

#### DiffusionBERT vs. Other Generative BERT Models

We compare DiffusionBERT  with another representative generative masked language model, BERT-Mouth .
Experimental results show that DiffusionBERT achieves better performance in terms of the perplexity and the BLEU score.
We attribute the superior performance of DiffusionBERT to its one-time sampling of all tokens, which helps DiffusionBERT generate more coherent text, especially in a long range. Although such decoding may face the problem of multimodality , inappropriate phrases can be fixed in the upcoming diffusion steps. The probabilistic modeling offers more flexibility in that generated tokens with low probability are more likely to be masked and re-sampled. In BERT-Mouth, however, the tokens are fixed once sampled. also proposed to continue masking and predicting tokens after the whole sequence is complete, revising the sentence for higher quality. But such randomness in the selection and replacement of tokens results in low inference speed.

Figure: Figure 3: BLEU scores on the LM1B test set. Left is better, lower is better. For GPT and Transformer decoder, we control quality-variation with sampling temperature. D3PM and DiffusionBERT are controlled by truncation sampling hyperparameter $K$.
Refer to caption: /html/2211.15029/assets/src/div_qua_tradeoff.jpg

#### Discrete vs. Continuous Diffusion Models

We then focus on the comparison of discrete and continuous diffusion models for text generation.
To achieve this, we mainly compare DiffusionBERT with recently proposed Diffusion-LM, which is based on continuous diffusion models.
As a result, despite of its outstanding controlling ability, we show that the texts generated by Diffusion-LM have a lower quality than DiffusionBERT.
Though both DiffusionBERT and Diffusion-LM adopt the same configuration of Transformer, it is worth noting that the superior performance of DiffusionBERT may be contributed by not only the discrete diffusion models but also the use of pre-trained models.
To disentangle the effect of pre-training and discrete/continuous diffusion models, we also explore initializing Diffusion-LM with BERT.
As shown in Table [2](#S4.T2), training Diffusion-LM from BERT initialization performs even worse than training from scratch.
We conjecture that the continuous nature of Diffusion-LM is not compatible with the initialization from BERT since the embedding learned by BERT may not be suitable for the Gaussian diffusion process.
In contrast, the comparison of D3PM and DiffusionBERT shows that DiffusionBERT benefits much from the BERT initialization due to its discrete diffusion process.

Figure: Figure 4: Curve of validation ELBO during training.
Refer to caption: /html/2211.15029/assets/src/Convergence.jpg

#### Effect of Time Step

In terms of both likelihood and generation quality, the layer-wise time embedding (LTE) lags far behind the other two time step designs for DiffusionBERT while time-agnostic decoding (TAD) achieves the best result. By contrast, D3PM without time step embedding performs significantly worse. In a nutshell, simplifying time step design has positive effect on DiffusionBERT but is quite harmful for D3PM. This suggests that initializing $p_{\theta}$ with PLMs enables DiffusionBERT to perform generation without explicitly providing time information yet achieving better generation results.
The resemblance between BERT pre-training objective and absorbing diffusion models makes it easier for DiffusionBERT to generalize to noisier scenarios while a Transformer encoder trained from scratch needs a specific time-aware module to model the reverse process.

#### Effect of the Spindle Noise Schedule

We try our proposed spindle noise schedule on both DiffusionBERT and D3PM. The perplexity is improved by 18% and 19% for D3PM and DiffusionBERT, respectively. Besides, D3PM with the spindle schedule outperforms that with the standard $(T-t+1)^{-1}$ schedule in generation quality. The same trend holds for DiffusionBERT but with a smaller margin.

### 4.4 Quality-Diversity Trade-off

As shown in Figure [3](#S4.F3), DiffusionBERT exhibits comparable generation ability with a Transformer decoder trained from scratch and pushes the Pareto front of NAR generation quality/diversity trade-off by a large margin. However, it still falls behind pretrained AR models of the same size.

**Table 3: Comparison of inference time and perplexity among baselines and DiffusionBERT.**
| Method | Steps | Inference Time (secs) | PPL |
| --- | --- | --- | --- |
| DiffusionBERT | 2 | 0.66 | 313.57 |
| 8 | 1.39 | 91.01 |  |
| 16 | 1.80 | 75.66 |  |
| 64 | 4.25 | 65.83 |  |
| 128 | 7.53 | 63.78 |  |
| 512 | 27.48 | 54.63 |  |
| Diffusion-LM | 2000 | 83.67 | 112.12 |
| BERT-Mouth | 64 | 2.18 | 142.89 |
| 512 | 14.39 | 86.78 |  |
| GPT | 64 | 1.55 | 38.7 |

### 4.5 Efficiency of Training and Generation

One important feature of DiffusionBERT is that with time-agnostic decoding, all parameters are initialized by pretrained models. Consequently, DiffusionBERT includes fewer parameters and is free from adapting new parameters, improving training and decoding efficiency.

#### Faster Convergence

DiffusionBERT converges remarkably faster than D3PM. Figure [4](#S4.F4) demonstrates the curve of validation ELBO in the training process. Even if the training budget is cut to 30% (i.e. 0.5 million steps), DiffusionBERT is still able to match the performance reported in Table [2](#S4.T2).

#### Sampling Speed

With the $x_{0}$-parameterization proposed in and , DiffusionBERT is able to perform inference with any given budget by controlling the step size in the reverse process. We also control the sampling time of BERT-Mouth by adjusting the max iteration count of its mask-predict process. We list the decoding speed and the corresponding perplexity on the LM1B test set in Table [3](#S4.T3). Overall, DiffusionBERT exhibits competitive performance even when it reaches comparable speed to GPT and outperforms BERT-Mouth in efficiency-performance tradeoff.

## 5 Related Work

### 5.1 BERT for Text Generation

It has been shown by that the transfer-learning ability of BERT does not only helps to achieve impressive results in natural language understanding but also benefits sequential sampling for text generation. However, its bi-directionality nature holds BERT from matching the decoder-only counterparts in modeling text from left to right.

### 5.2 Diffusion Models for Text

This work lies in the line of diffusion models, a latent variable generative framework proposed by . It has been architecturally improved by and has gained broad attention for its impressive generation ability in continuous domain (e. g. image and audio) .
Despite their great success and state-of-the-art sample quality in the above domains, diffusion models for text still struggle to match autoregressive models in various generation tasks. Since the Gaussian noise proposed in cannot be directly applied to discrete data, they also introduced a discrete forward process with a Bernoulli transition kernel. made a step forward from Bernoulli to categorical distributions. A more general family of discrete diffusion processes was introduced in , including absorbing kernels and combinations of absorbing and uniform transition kernels.
models text in the continuous embedding space, which is closer to the settings in earlier works of diffusion models and shows impressive performance in classifier-controlled text generation. While the decoding and convergence speed are substantially slower and the generated text lacks coherence. Moreover, in scenarios where the vocabulary is large, the k-nearest-neighbor algorithm used in decoding holds up decoding even more severely.

### 5.3 Non-Autoregressive Text Generation

Absorbing discrete diffusion models resembles conditional masked language models (CMLMs) in that both methods predict the whole sequence simultaneously and follows a construct-destruct pattern to iteratively refine the generated text. The main difference lies in the training objective: DiffusionBERT models a stochastic process and drives BERT to learn a group of distributions to gradually recover training data while CMLMs forces the neural network to deterministically recover the whole sequence in every iteration, thus it fails to explicitly model the denoising process. proposed to approach the problem of non-autoregressive text modeling via unrolling the generation path to prepare the model for the partially corrupted sequences it will encounter during generation, which resembles the idea of diffusion models for unconditional text generation. Non-autoregressive models are also considered in translation but implemented in various ways, e.g., insertion/deletion and iterative sequence alignment .

## 6 Conclusion

This work aims to approach the problem of unconditional text generation for non-autoregressive models. To achieve this, we combine pretrained language models with absorbing-state discrete diffusion models for text. The training procedure of our proposed DiffusionBERT includes two main deviations from current discrete diffusion models, i.e., a new family of time step designs and the spindle noise schedule. The novel spindle noise assigns a schedule for each token according to its frequency in the training corpus. Experimental results demonstrate the success of DiffusionBERT in terms of perplexity. It also pushes the Pareto front of quality-variance tradeoff of NAR methods by a large margin, comparable to a Transformer decoder trained from scratch.