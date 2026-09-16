---
arxiv_id: "2210.17432"
title: "SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control"
year: 2023
source: arxiv2md
---

## Abstract

Abstract Despite the growing success of diffusion models in continuous-valued domains (e.g., images), similar efforts for discrete domains such as text have yet to match the performance of autoregressive language models. In this work, we present Ssd-LM —a diffusion-based language model with two key design choices. First, Ssd-LM is semi-autoregressive , iteratively generating blocks of text, allowing for flexible output length at decoding time while enabling local bidirectional context updates. Second, it is simplex-based , performing diffusion on the natural vocabulary space rather than a learned latent space, allowing us to incorporate classifier guidance and modular control using off-the-shelf classifiers without any adaptation. We evaluate Ssd-LM on unconstrained text generation benchmarks, and show that it matches or outperforms strong autoregressive GPT-2 models across standard quality and diversity metrics, while vastly outperforming diffusion-based baselines. On controlled text generation, Ssd-LM also outperforms competitive baselines, with an extra advantage in modularity. 1 1 1 Our code and models can be found at https://github.com/xhan77/ssd-lm .

## 1 Introduction

Diffusion models , trained to iteratively refine noised inputs, have recently emerged as powerful tools for generative modeling in several continuous-valued domains such as images , audio , video , among others. Attempts to adapt them for discrete domains such as text data, however, have only had limited success: prior work have shown to be promising on specialized cases and small datasets ,
but diffusion models for text still underperform (and thus are not widely adopted) compared to autoregressive language models (AR-LMs) which remain the state-of-the-art general purpose text generators .

Despite potential advantages of diffusion models for text, there are two key challenges. First, diffusion models generate text non-autoregressively, i.e., they generate (and update) the entire sequence simultaneously rather than token by token left-to-right. Although this property
is useful in practice since each output token is informed by a broader bi-directional context , it requires pre-defining an output sequence length. This limits the flexibility and applicability of trained models.
On the other hand, non-autoregressive training with long sequences is expensive and difficult to optimize.
In this work, we propose a *semi-autoregressive* solution which strikes a balance between length flexibility and the ability to alter previously generated tokens.

A major advantage of diffusion models over the current standard of autoregressive LMs is their post-hoc controllability using guidance from auxiliary models such as style classifiers . However, controllability is hard to achieve without compromises in modularity in diffusion-based LMs for text. To enable diffusion generation into discrete text rather than continuous modalities, prior approaches have employed different approximations, e.g., training with embeddings, character, or byte-level methods . In contrast, existing mainstream LMs and the guidance classifiers they derive often operate at a sub-word level with sub-word representations trained jointly with the language model . Subsequently, changing the input representations to characters or embeddings requires developing guidance models from scratch, which can be expensive or infeasible in many cases. In this work, we propose a *simplex-based* solution which enables the diffusion over discrete texts while maintaining the advantages of diffusion models with plug-and-control guidance models.

In sum, to enable diffusion-based LMs for text we present Ssd-LM (§[3](#S3)), addressing the above two challenges. Ssd-LM is trained to generate text semi-autoregressively—generating blocks of tokens left-to-right with bidirectional context within the block—which offers the benefits of both AR-LMs and diffusion models. It supports training with and generating variable-length sequences. At the same time, it allows refinement within the token block, in contrast to token-level autoregressive decoding where previously generated tokens cannot be modified at all.
Ssd-LM uses the same tokenization as popular AR-LMs, representing discrete text via a distribution (or simplex) defined over the vocabulary and is trained to reconstruct texts from noisy versions of the distributions.
Due to its underlying representation, our method also offers an easy and modular way of guided (controlled) generation using off-the-shelf text classifiers under the minimal assumption of shared tokenizer.

Our evaluation experiments show, for the first time,
that a diffusion-based LM matches or outperforms strong AR-LMs on standard text generation benchmarks (§[4](#S4)).
We evaluate Ssd-LM on two tasks: (1) unconstrained prompt-based generation substantially outperforming existing diffusion LM approaches and performing on par with or outperforming strong autoregressive LM GPT-2  on both quality and diversity (§[4.2](#S4.SS2)); and (2) controlled text generation with guidance from off-the-shelf classifiers (no post-hoc training/adaptation) outperforming competitive controlled text generation baselines (§[4.3](#S4.SS3)).

## 2 Background

### 2.1 Diffusion model

Since their inception as image generators, diffusion models (and their cousins score-based models ) have been widely adopted as high-quality generative models for multiple data modalities. Here, we briefly describe a simplified view of a canonical method, denoising diffusion probabilistic models which we adapt in this work for text generation. We assume a given dataset $\mathcal{D}=\{{}^{1}\boldsymbol{x}_{0},\ldots,{}^{N}\boldsymbol{x}_{0}\}$ of continuous valued items ${}^{i}\boldsymbol{x}_{0}$ (e.g., pixel values of an image) henceforth referred to as $\boldsymbol{x}_{0}$ for simplicity.

#### Training

Training a diffusion model first involves adding a series of Gaussian noise to the original data $\boldsymbol{x}_{0}$, through $T$ timesteps:

$$ $\displaystyle\boldsymbol{x}_{t}$ $\displaystyle=\sqrt{\bar{\alpha}_{t}}\boldsymbol{x}_{0}+\sqrt{1-\bar{\alpha}_{ t}}\boldsymbol{\epsilon}_{t}$ (1) $$

where $t\in(1,T)$ and $\boldsymbol{\epsilon}_{t}\sim\mathcal{N}(\boldsymbol{0},\mathbf{I})$.
$\bar{\alpha}_{t}=\prod_{t^{\prime}=1}^{t}\alpha_{t^{\prime}}$, where $\alpha_{t^{\prime}}$ follow a predefined schedule such that $\bar{\alpha}_{t}\to 0$ as $t\to T$. This process is called *forward diffusion*.
A diffusion model (parameterized by $\theta$) is trained to reverse this forward process by predicting the added noise $\boldsymbol{\epsilon}_{t}$ given $\boldsymbol{x}_{t}$ with the following loss:

$$ $\displaystyle\mathcal{L}(\theta)=\mathbb{E}_{t\sim\mathcal{U}(1,T)}\lVert \epsilon_{\theta}(\boldsymbol{x}_{t},t)-\boldsymbol{\epsilon}_{t}\rVert^{2}$ (2) $$

#### Inference

To get an output from this model, we sample $\boldsymbol{x}_{T}\sim\mathcal{N}(\boldsymbol{0},\mathbf{I})$ and iteratively reconstruct a sample $\boldsymbol{x}_{0}$ by going back in time,

$$ $\displaystyle\boldsymbol{x}_{t-1}$ $\displaystyle=\frac{1}{\sqrt{\alpha_{t}}}(\boldsymbol{x}_{t}-\frac{1-\alpha_{t }}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon_{\theta}(\boldsymbol{x}_{t},t))$ (3) $$

for $t=T,\ldots,1$.(^2^22We omit an additional noise term $z$ here for simplicity, which is present in DDPM but not in another variant DDIM .)
The key obstacle in using vanilla diffusion models directly as text generators is that language consists of discrete tokens, i.e., a non-continuous $\boldsymbol{x}_{0}$ to which a continuous valued Gaussian noise cannot be added. We propose a straightforward and effective solution by treating tokens as continuous valued simplexes over the vocabulary .
Other existing methods addressing this problem are discussed in §[5](#S5).

### 2.2 Autoregressive LM

An autoregressive LM
model optimizes for the likelihood of a sequence of tokens $w^{0},\ldots,w^{L-1}$.

$$ $\displaystyle p_{\theta}(\boldsymbol{w}^{0:L})=\prod_{c=0}^{L-1}p_{\theta}(w^{ c}\mid\boldsymbol{w}^{<c})$ (4) $$

To decode from AR-LMs, one can provide a context $\boldsymbol{w}^{<c}$ and decode the next token $w^{c}$ iteratively by predicting $p_{\theta}(w^{c}\mid\boldsymbol{w}^{<c})$ and sampling from it to get the discrete token . Prior work has shown that these decoding approaches (and by extension the LMs themselves) are prone to degrade when generating long sequences and often devolve into repeating subsequences . In addition, such LMs do not provide a natural way to incorporate sequence-level control as tokens are generated one at a time without the ability to modify previously generated tokens .
In this work, we present a method to train a semi-autoregressive LM that decodes blocks of $B$ tokens at a time,
alleviating said issues with the support of diffusion models. Existing literature addressing the two issues individually are discussed in §[5](#S5).

## 3 Ssd-LM

We introduce Ssd-LM—Semi-autoregressive Simplex-based Diffusion Language Model— adapting key components from both autoregressive LM and vanilla diffusion models. Conceptually, Ssd-LM uses diffusion model to decode $\boldsymbol{w}^{c:c+B}$, a block of tokens of length $B$, given a Gaussian noise and a context $\boldsymbol{w}^{<c}$ of length $c$.
We show an intuitive diagram and pseudo-code for the training and decoding algorithm of Ssd-LM in [Figure 1](#S3.F1), [Figure 2](#S3.F2), and [Figure 3](#S3.F3).

Figure: Figure 1: Training Ssd-LM (a snapshot at context size $c=2$, block size $B=3$). Horizontal axis represents the order of tokens. Vertical axis represents the diffusion timesteps. Shade means observable variables. Square means discrete vocabulary, while circle means continuous logits. Red components are inputs to the learning model $\theta$.
Refer to caption: extracted/2210.17432v2/resources/reindex_train_fig.png

Figure: Figure 2: Decoding from Ssd-LM (continuing [Figure 1](#S3.F1)). Red components are inputs to the learned model $\theta$. Dash means intermediate variables.
Refer to caption: extracted/2210.17432v2/resources/reindex_decode_fig.png

Figure: Algorithm 1 Training

### 3.1 Training

#### Continuous data representation

To build a continuous representation for discrete tokens, we adopt an *almost-one-hot* simplex representation over the model’s vocabulary $V$. We define a simple operation $\operatorname{logits-generation}(.)$ to map a token $w$ to $\tilde{\boldsymbol{w}}\in\{-K,+K\}^{|V|}$ as follows.

$$ $\displaystyle\tilde{w}_{(i)}=\begin{cases}+K\text{ when }w=V_{(i)}\\ -K\text{ when }w\neq V_{(i)}\end{cases}$ (5) $$

where $i$ is the index of the vocabulary. We call $\tilde{\boldsymbol{w}}$ the logits for token $w$, and $\operatorname{softmax}(\tilde{\boldsymbol{w}})$ gives a probability simplex over the vocabulary $V$, with a probability mass concentrated on the token $w$. There is no learnable parameter in this mapping.

#### Forward diffusion

Following , we add a time-dependent Gaussian noise to the logits.

$$ $\displaystyle\tilde{\boldsymbol{w}}_{0}^{c:c+B}$ $\displaystyle=\operatorname{logits-generation}(\boldsymbol{w}^{c:c+B})$ (6) $\displaystyle\tilde{\boldsymbol{w}}_{t}^{c:c+B}$ $\displaystyle=\sqrt{\bar{\alpha}_{t}}\tilde{\boldsymbol{w}}_{0}^{c:c+B}+\sqrt{ 1-\bar{\alpha}_{t}}\boldsymbol{\epsilon}_{t}$ (7) $$

where $t\in(1,T)$, $\boldsymbol{\epsilon}_{t}\sim\mathcal{N}(\boldsymbol{0},K^{2}\mathbf{I})$, and $\bar{\alpha}_{t}\to 0$ as $t\to T$. At the final step $T$, $\operatorname{softmax}(\tilde{\boldsymbol{w}}_{T}^{c:c+B})$ are fully noisy simplexes over $V$, with a
logit-normal distribution .

#### Loss function

In [Equation 2](#S2.E2), a diffusion model is trained to predict the added noise from the noisy representations. Since the forward diffusion process can be computed in a single step ([Equation 1](#S2.E1)), the notion here is equivalent to predicting the original data representation . Our objective follows the same intuition but estimates a likelihood instead of the L2 distance while conditioning on additional context:(^3^33L2 distance did not work in our pilot study potentially due to the intrinsically skewed simplex representation.)

$$ $\displaystyle\mathcal{L}(\theta)$ $\displaystyle=\mathbb{E}[-\log p_{\theta}(\boldsymbol{w}^{c:c+B}\mid\tilde{ \boldsymbol{w}}_{t}^{c:c+B},\boldsymbol{w}^{<c})]$ (8) $\displaystyle=\mathbb{E}\left[\sum_{j=c}^{c+B-1}-\log p_{\theta}(w^{j}\mid \tilde{\boldsymbol{w}}_{t}^{c:c+B},\boldsymbol{w}^{<c})\right]$ (9) $$

$\mathbb{E}[\cdot]$ is a shorthand for $\mathbb{E}_{c\sim\mathcal{U}(1,L-B),t\sim\mathcal{U}(1,T)}[\cdot]$.
The architecture for $\theta$ throughout this work is a bi-directional Transformer encoder . Specifically, the input to the model is a concatenation of the context $\boldsymbol{w}^{<c}$ and a sequence of noisy vocabulary simplexes $\operatorname{softmax}(\tilde{\boldsymbol{w}}_{t}^{c:c+B})$ of length $B$. The target output is the original tokens $\boldsymbol{w}^{c:c+B}$ at positions $c$ to $c+B$.

One minimal modification made to the Transformer model is that in addition to the conventional embedding lookup for $\boldsymbol{w}^{<c}$, we modify the embedding layer to take as input a distribution over the vocabulary, $\operatorname{softmax}(\tilde{\boldsymbol{w}}_{t}^{c:c+B})$, and compute the embedding vector as a weighted sum of the embedding table. A timestep embedding is also added before the first Transformer block to inform the model of the current timestep.(^4^44More specifically, we have word embeddings for the context, $\operatorname{Emb}_{\text{ctx}}(\boldsymbol{w}^{<c})$, and for the noisy diffusion representations, $W_{\text{diff}}[\operatorname{softmax}(\tilde{\boldsymbol{w}}_{t}^{c:c+B})]$. The timestep embedding is added to the diffusion word embeddings, $W_{\text{time}}(t/T)$. It is similar to positional embeddings, just not varying across sequence positions. We fold it in $\theta$ for notation simplicity.)

In §[A](#A1), we present another interpretation of the training objective as an intuitive contrastive loss.

### 3.2 Decoding

#### Logits projection

Similar to continuous-valued diffusion models, sampling from Ssd-LM involves reverse diffusion from $t=T,\ldots,1$ starting with a Gaussian noise.
At any timestep $t$, our model $\theta$ takes as input noised logits $\tilde{\boldsymbol{w}}_{t}^{c:c+B}$ and estimates the probability distribution of the original tokens in data by first predicting the logits:

$$ $\displaystyle\boldsymbol{w}_{\text{logits},t}^{c:c+B}=\operatorname{logits}_{ \theta}(\boldsymbol{w}^{c:c+B}\mid\tilde{\boldsymbol{w}}_{t}^{c:c+B}, \boldsymbol{w}^{<c})$ (10) $$

which are then converted to a distribution via softmax. To feed this output to the next step of reverse diffusion, $t-1$, we define a $\operatorname{logits-projection}$ operation to build a predicted data representation close to the initial data representation (almost-one-hot mapping; [Eq. 5](#S3.E5)).
We consider three projection operations.

- •
*Greedy:* creates an almost-one-hot logit centered at the highest probability token.(^5^55This shares a similar intuition as a greedy clamping trick in the embedding-based diffusion in .)
$\displaystyle\hat{w}_{(i)}\text{=}\begin{cases}+K\text{ if $i$=}\operatorname{
argmax}(\boldsymbol{w}_{\text{logits}})\\
-K\text{ otherwise}\end{cases}$
(11)
- •
*Sampling:* creates an almost-one-hot logit centered around a token sampled from the output distribution using top-$p$ sampling . $p$ is a hyperparameter.
$\displaystyle\hat{w}_{(i)}\text{=}\begin{cases}+K\text{ if $i$=}\text{top-}p
\text{-sample}(\boldsymbol{w}_{\text{logits}})\\
-K\text{ otherwise}\end{cases}$
(12)
- •
*Multi-hot:* creates an almost-one-hot logit centered around *all* tokens in the top-$p$ nucleus.
$\displaystyle\hat{w}_{(i)}\text{=}\begin{cases}+K\text{ if $i\in$ }\text{top-}
p\text{-all}(\boldsymbol{w}_{\text{logits}})\\
-K\text{ otherwise}\end{cases}$
(13)

#### Decoding iteration

Starting from pure noise $\tilde{\boldsymbol{w}}_{T}^{c:c+B}\sim\mathcal{N}(\boldsymbol{0},K^{2}\mathbf{
I})$, in each decoding timestep we compute:

$$ $\displaystyle\hat{\boldsymbol{w}}^{c:c+B}_{t}=\operatorname{logits-projection} (\boldsymbol{w}_{\text{logits},t}^{c:c+B})$ (14) $\displaystyle\tilde{\boldsymbol{w}}_{t-1}^{c:c+B}=\sqrt{\bar{\alpha}_{t-1}} \hat{\boldsymbol{w}}^{c:c+B}_{t}+\sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{z}$ (15) $$

for $t=T,\ldots,1$ and $\boldsymbol{z}\sim\mathcal{N}(\boldsymbol{0},K^{2}\mathbf{I})$.

At $t=1$, the final $B$-token block is computed simply as $\operatorname{argmax}\tilde{\boldsymbol{w}}_{0}^{c:c+B}$. To generate the next block, we concatenate the generated block to the previous context to create a new context of length $c+B$ and follow the reverse-diffusion process again as described above. This process can be repeated until the maximum desired length is reached.(^6^66Alternatively, one can also terminate the process if certain special end-of-sequence tokens have been generated.)

It is worth noting that our proposed decoding algorithm is novel and different from the DDPM decoding ([Equation 3](#S2.E3)). The DDPM decoding is designed for diffusion in a continuous space and failed to generate sensible outputs in our preliminary experiments based on simplexes. In §[B](#A2), we draw a theoretical connection between our decoding algorithm and DDPM decoding, and also highlight the intuitive difference between the two.

#### Highly-modular control

A useful property of continuous diffusion models that naturally arises from their definition is the ability to guide the generated samples to have user-defined attributes at test time. This can be done using gradients from auxiliary models such as classifiers , e.g., guiding the output of an LM to be of a positive sentiment using a sentiment classifier.
There is a vibrant community of developers on platforms such as HuggingFace where many such text classifiers are publicly available.
The underlying data representation of Ssd-LM is based on vocabulary simplexes. Hence, as long as a classifier shares the same tokenizer as the LM, it can be used for control in an off-the-shelf manner without modifications. This is in contrast to prior work in diffusion language models that do not support such classifiers due to differences in their input representation space and require retraining the classifiers from scratch.
This ability makes Ssd-LM highly modular for controlled text generation and offers key benefits: (1) Training accurate classifiers for many tasks requires huge amounts of data where retraining them can be quite expensive, and (2) this approach allows control from classifiers that are open to use but have been trained on closed source data.

To guide Ssd-LM to generate texts with a target attribute $y$ via a standalone attribute model $f_{\phi}(\cdot)$, we update $\boldsymbol{w}_{\text{logits},t}^{c:c+B}$ ([Eq. 10](#S3.E10)) at each timestep $t$ to the form below, drifting according to the gradients from the attribute classifier.

$$ $\displaystyle\boldsymbol{w}_{\text{logits},t}^{c:c+B}+\lambda\nabla_{ \boldsymbol{w}_{\text{logits},t}^{c:c+B}}f_{\phi}(y\mid\boldsymbol{w}_{\text{ logits},t}^{c:c+B},\boldsymbol{w}^{<c})$ (16) $$

where $\lambda$ is a hyperparameter balancing the weight of control. The parameters of the standalone attribute model $\phi$ are frozen. We make a trivial modification to the embedding computation as in §[3.1](#S3.SS1), to allow the classifier to take as input a simplex.

### 3.3 Additional details

#### Forward diffusion coefficient α ¯ t subscript ¯ 𝛼 𝑡 \bar{\alpha}_{t} over¯ start_ARG italic_α end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT

We follow for a cosine schedule of $\bar{\alpha}_{t}$:

$$ $\displaystyle\bar{\alpha}_{t}=\frac{r(t)}{r(0)},~{}r(t)=\cos(\frac{t/T+s}{1+s} \cdot\frac{\pi}{2})^{2}$ (17) $$

where $s$ is small offset set to 1e-4 in our work and $\alpha_{t}=\frac{\bar{\alpha}_{t}}{\bar{\alpha}_{t-1}}$.

#### Fewer timesteps T 𝑇 T italic_T in decoding

Decoding from diffusion models requires a series of timesteps ($T$) which can be computationally expensive if $T$ is large. Following , we consider using a smaller value of $T$ at test time to improve decoding speed.
In this work, we primarily experiment with $T_{\text{decode}}=\frac{T_{\text{train}}}{2}$ and $T_{\text{decode}}=\frac{T_{\text{train}}}{5}$.

#### Flexible decoding block size B 𝐵 B italic_B

Our Ssd-LM is trained with a fixed token block size $B_{\text{train}}$. However, the decoding algorithm has a freedom to use a different $B_{\text{decode}}$. In our experiments, we consider both scenarios of $B_{\text{train}}=B_{\text{decode}}$ and $B_{\text{train}}\neq B_{\text{decode}}$. Nevertheless, we leave for future work a more detailed analysis of the impact of the difference between $B_{\text{train}}$ and $B_{\text{decode}}$ on model performance.

## 4 Experiments

### 4.1 Ssd-LM pretraining setup

#### Model architecture

We use a bidirectional Transformer encoder RoBERTa-large (0.4B, comparable size to GPT2-medium) as Ssd-LM’s underlying architecture.(^7^77We initialize the model with RoBERTa’s weights as well. We observe in our initial exploration that it helps the training loss converge faster than a randomly initialized model. However, given enough computational resources, we conjecture that a randomly initialized model will offer similar performance.) Note that RoBERTa uses a general BPE tokenization , same as a variety of LMs such as GPT-2 , GPT-3 , OPT , etc. Any attribute classifier using the same tokenization strategy can be used to control Ssd-LM in a highly modular way.

#### Pretraining data, constants, and resource

We train Ssd-LM on the same data as GPT2 to make fair comparisons possible: OpenWebText which contains 9B tokens.
Following , we consider this data as one contiguous sequence of tokens and break it into sequences of length 200 (same as the maximum sequence length our model accepts). We randomly sample 99% of these sequences for pretraining while leaving the rest as held out for evaluation. We use the following model hyperparameters:(^8^88Future work can do a search given more resources.)

$$ $L=200,B_{\text{train}}=25,T_{\text{train}}=5000,K=5$ $$

We use an aggregated batch size of 6,144 and a learning rate of 1e-4 with an AdamW optimizer . We trained Ssd-LM for 100K steps, which took about 6 days on 32 Nvidia V100 GPUs.

#### Pretraining loss

Canonical training-time perplexity of LMs is not compatible with diffusion LMs due to the difference in the inputs to the models ([Equation 4](#S2.E4) and [Eq. 9](#S3.E9)).
Our pretraining loss is a per-token negative log-likelihood (NLL) that depends on the specific noise schedule being used. Ssd-LM gets an average NLL of 3.87 at the end of pretraining. We show a pretraining loss curve in the appendix (§[D](#A4)).

### 4.2 Unconstrained text generation

**Table 1: Unconstrained generation evaluation of Ssd-LM and GPT-2 models at length 50. For GPT-2 models, the results are averaged across 5 random seeds, and we show the best sampling parameter configuration. For our Ssd-LM, we show the top-3 configurations. All configurations are ranked based on MAUVE, with original parameters from . The perplexity (PPL) is measured by GPT-Neo-1.3B.(^9^99 MAUVE, Dist-1/2/3, and Rep are in percentage. PPL is obtained through a micro average following .)**
| (Length 50) | MAUVE $\uparrow$ | PPL $\xrightarrow[\text{gold}]{}$ | $|\Delta_{\log\text{PPL}}|$ $\downarrow$ | Dist-1 $\uparrow$ | Dist-2 $\uparrow$ | Dist-3 $\uparrow$ | Zipf $\xrightarrow[\text{gold}]{}$ | Rep $\downarrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gold continuation | 100.00 | 17.75 | 0.00 | 88.62 | 95.88 | 93.71 | 0.88 | 0.10 |
| GPT2-medium (Best config) |  |  |  |  |  |  |  |  |
| Top-$p$=0.95 | 96.57   $\pm$ 0.40 | 12.72   $\pm$ 0.07 | 0.33 | 66.31   $\pm$ 0.11 | 91.77   $\pm$ 0.03 | 92.75   $\pm$ 0.06 | 1.01 | 0.26   $\pm$ 0.04 |
| GPT2-large (Best config) |  |  |  |  |  |  |  |  |
| Top-$p$=0.95 | 96.41   $\pm$ 0.78 | 10.57   $\pm$ 0.05 | 0.51 | 64.91   $\pm$ 0.13 | 90.88   $\pm$ 0.06 | 92.38   $\pm$ 0.05 | 1.01 | 0.41   $\pm$ 0.06 |
| GPT2-xl (Best config) |  |  |  |  |  |  |  |  |
| Typical-$\tau$=0.95 | 97.03   $\pm$ 0.50 | 10.33   $\pm$ 0.04 | 0.54 | 64.87   $\pm$ 0.15 | 90.69   $\pm$ 0.07 | 92.16   $\pm$ 0.05 | 1.01 | 0.37   $\pm$ 0.04 |
| Ssd-LM-“medium” (Top-3) |  |  |  |  |  |  |  |  |
| Sampling $p$=0.99, $T$=1000 | 97.89 | 30.68 | 0.54 | 68.99 | 92.60 | 92.94 | 1.01 | 0.16 |
| Sampling $p$=0.95, $T$=1000 | 96.64 | 27.34 | 0.43 | 67.75 | 92.16 | 92.91 | 1.01 | 0.16 |
| Sampling $p$=0.9, $T$=2500 | 96.46 | 20.56 | 0.14 | 66.61 | 91.46 | 92.56 | 1.05 | 0.26 |

#### Setup

First, we benchmark Ssd-LM with autoregressive LMs trained on the same data (GPT2) on text generation quality. We randomly sample 1000 sequences from the held-out OpenWebText test data, extract their prefixes as prompts (context), and generate continuations from the LMs.
We consider three setups: with prompt lengths 25, 50 and 100 with respective output lengths as 25, 50 and 100 tokens. In each setup, we sample 5 continuations for each input context, thus comparing the quality of 5,000 generations from baseline GPT-2 models and our Ssd-LM.

We compare Ssd-LM with GPT2-medium, large and xl models (containing 0.4B, 0.8B and 1.6B parameters respectively) as baselines. For reference, our model size is comparable to GPT2-medium. We experiment with two popular decoding strategies for the baseline GPT-2 models with canonical parameters: nucleus sampling with a top-$p$ of 0.9 and 0.95, and typical sampling with a typical-$\tau$ of 0.2 and 0.95.

For Ssd-LM, we consider three logits projection strategies, sampling and multi-hot with $\text{top-}p\in\{0.0,0.1,0.2,0.5,0.7,0.9,0.95,0.99\}$, and greedy (which is functionally equivalent to the sampling with top-$p$=0). We use a test block size ($B_{\text{decode}}$) of 25. When generating samples of length 50 or 100, we semi-autoregressively sample in blocks of 25 and feed them as additional context to generate the next block as described in §[3.2](#S3.SS2).

We evaluate the generated continuations on two axes: quality and diversity. As automatic quality metrics, we report perplexity measured by a separate, larger language model  . Prior works, however, have shown that low perplexity of generated text is not necessarily an indication of high quality but of degenerate behavior  and have proposed closeness to the perplexity of human-written text as a better evaluation. Hence, we also report the difference of log perplexity between the generated text and human-written continuations ($|\Delta_{\log\text{PPL}}|$). For diversity evaluation, we report Zipf’s coefficient (Zipf) and average distinct $n$-grams in the output samples . In addition, we also report the repetition rate , measuring the proportion of output samples that end in repeating phrases. Finally, we report MAUVE  which evaluates both quality and diversity together by approximating information divergence between generated samples and human-written continuations (from the OpenWebText held-out set).

#### Results

[footnote 9](#footnote9) summarizes our main results on the 50-token prompt and output setup. We report the numbers for the best performing three settings for logits projection and decoding steps $T$ in Ssd-LM. We report the best setting for the baselines. The results for other generation lengths have a similar trend and can be found in the appendix (§[D](#A4)).

We find that Ssd-LM, though being smaller in size, outperforms larger GPT-2 models on the unified metric MAUVE. On diversity, Ssd-LM outperforms GPT-2 in Dist-$n$ while achieving lower repetition rates. On perplexity, the results are slightly mixed. We observe a trade-off between MAUVE and perplexity for different settings we considered,
indicating that further tuning of the hyperparameters may be required.
However, one of our best performing settings (sampling top-$p$=0.9, $T$=2500) still achieves the closest perplexity to the gold continuation.

In §[D](#A4), we show the influence of different logits projection strategies and the associated parameters on the output text quality in [Figure 4](#A4.F4). We also show qualitative examples of the generations by Ssd-LM in [Table 8](#A4.T8) and a trajectory of intermediate states during the decoding process in [Table 9](#A4.T9).

**Table 2: Unconstrained generation results of Ssd-LM and Diffusion-LM on ROCStories with 50 prompt tokens and 50 output tokens. We report the MAUVE score between the gold continuation and model generations. We also show the perplexity (PPL) of model generations measured by GPT-Neo-1.3B.(^10^1010Due to a lowercase tokenization of ROCStories, we use BERT-base-uncased as MAUVE’s embedding model here.)**
| (ROCStories) | MAUVE | PPL |
| --- | --- | --- |
| Gold continuation | 100.00 | 18.57 |
| Diffusion-LM | 46.11 | 35.96 |
| Ssd-LM | 87.22 | 22.91 |

#### Comparison with Li et al. ( 2022 )

A prior work to us, propose Diffusion-LM, an embedding-based diffusion model trained on two small toy datasets, E2E and ROCStories .
In this subsection, we make a diversion to compare the embedding-based Diffusion-LM with our semi-autoregressive, simplex-based Ssd-LM.
Following , we train a Diffusion-LM on ROCStories with a default embedding size of 128, 0.1B parameters under a BERT-base structure,(^11^1111We train two versions of Diffusion-LM, with and without BERT’s encoder weights as an initialization. The default no-initialization setup as in works reasonably, while the other degenerates. Details can be found in §[C](#A3).) and a sequence length of 100.
For a fair comparison, *only within this subsection* we train a Ssd-LM with ROCStories sequences of 100 tokens, a decoding block size of 25, and a BERT-base initialization. Further details of the setup can be found in §[C](#A3).

On 2,700 held-out ROCStories sequences, we use the first 50 tokens of each sequence as a prompt and have the model generate the next 50.
In [footnote 10](#footnote10), we show the MAUVE score and perplexity of both models. We observe a substantially higher MAUVE score and lower perplexity with Ssd-LM.

### 4.3 Controlled text generation

#### Setup

To evaluate Ssd-LM’s ability for highly-modular control, we consider the task of sentiment controlled generation where given a prompt, the goal is to generate a continuation with a positive (or negative) polarity.
We use a set of 15 short prompts as in and generate 20 samples per prompt per sentiment category, making the total number of generated samples to be 600. Following , we generate samples with 3 different output lengths: 12, 20 and 50.
For guidance, we simply import a popular sentiment classifier(^12^1212[https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)) from HuggingFace trained with Twitter sentiment data with over 58M training examples . This model serves as $f_{\phi}(\cdot)$ as shown in [Eq. 16](#S3.E16).
In addition to quality and diversity of the generated samples, we also evaluate them on control (that is measuring if the generated output is actually positive or negative in polarity). For this, we use an *external* sentiment classifier trained on a different dataset. Specifically, we use a classifier trained with Yelp reviews(^13^1313[https://huggingface.co/textattack/bert-base-uncased-yelp-polarity](https://huggingface.co/textattack/bert-base-uncased-yelp-polarity)) following the evaluation setup in the baselines we consider.

Again, we consider the sampling and multi-hot decoding strategies with $\text{top-}p\in\{0.2,0.5,0.9\}$, $T_{\text{decode}}\in\{1000,2500,5000\}$, and the multiplier for control $\lambda\in\{0,100,500,2000\}$. For the generation of 12/20/50 tokens, we use $B_{\text{decode}}$=12/20/25 and apply the decoding algorithm for $m$=1/1/2 iterations respectively.

**Table 3: Controlled text generation results of Ssd-LM and baselines at length 50. We report the external classifier’s accuracy (C-Ext.) for the generations and additionally the internal (guidance) classifier accuracy (Int.) if available. The perplexity (PPL) is computed with GPT2-xl. MuCoLa is the version using two discriminators. $\mathbb{CM}$ stands for customized language model, $\mathbb{CC}$ stands for customized classifier, and $\mathbb{HMC}$ stands for highly-modular classifier (in an order of increasing modularity). The best of all results are boldfaced, and the best of $\mathbb{HMC}$ results are italicized.(^14^1414 PPL is obtained through a macro average following .)**
| (Length 50) | C-Ext.(Int.) | PPL | Dist-1/2/3 |
| --- | --- | --- | --- |
| DAPT${}^{\mathbb{CM}}$ | 79.8 | 57.2 | 61/92/94 |
| PPLM${}^{\mathbb{CC}}$ | 60.7 (73.6) | 29.0 | - |
| FUDGE${}^{\mathbb{CC}}$ | 59.1 | 8.4 | 47/83/92 |
| GeDi${}^{\mathbb{CM}}$ | 99.2 | 107.3 | 71/93/92 |
| DExperts${}^{\mathbb{CM}}$ | 94.8 | 37.1 | 56/90/92 |
| MuCoLa${}^{\mathbb{CC}}$ | 86.0 | 27.8 | 52/76/80 |
| M&M LM${}^{\mathbb{HMC}}$ | 68.6 (93.8) | 122.3 | - |
| Ssd-LM${}^{\mathbb{HMC}}$ | 94.1 (99.0) | 23.1 | 46/84/92 |

#### Results

We show the quality of the controlled generations from three perspectives: target attribute via the external classifier accuracy, fluency via perplexity, and diversity via the distinctiveness measures. In [footnote 14](#footnote14), we show the experimental results for output length 50. The results at length 12 and 20 have a similar trend and can be found in the appendix (§[D](#A4)).

Among the baseline methods, DAPT , GeDi , and DExperts require training customized language models aware of the desired attributes (denoted as CM in [Table 7](#A4.T7)). PPLM , FUDGE , and MuCoLa require training a customized attribute classifier (CC). While our proposed method Ssd-LM and M&M LM can directly import mainstream existing attribute classifiers from platforms like HuggingFace and are thus highly modular (HMC). We show the baseline results as reported in and .

Ssd-LM shows strong controllability while possessing great modularity. Ssd-LM outperforms M&M LM, the other HMC method by a large margin. Even when comparing with the CC and CM methods, our method achieves a good balance in control, fluency, and diversity.

In §[D](#A4), we show the impact of the control weight $\lambda$ and top-$p$ on the attribute accuracy and perplexity in [Figure 5](#A4.F5). We also show qualitative examples of the controlled generations by Ssd-LM in [Table 8](#A4.T8).

## 5 Related work

#### Diffusion models

Diffusion models have demonstrated impressive performance in popular continuous-valued domains such as images , audio , video  and recently also been adopted for 3D-shapes, protein structures, and more . Since they are based on adding Gaussian noise, these approaches are not straightforward to apply to discrete valued domains like text.
propose diffusing in the discrete space using categorical distributions which are modified using transition matrices. However, these methods do not straightforwardly support control and yield worse results than comparable autoregressive models.
propose to represent each token as a continuous embedding and apply diffusion in the embedding space.
They train the LM to generate a fixed length sequence whereas Ssd-LM allows flexibility in the generated sequence length by generating block-wise. Further, their LM is trained with specialized datasets and not evaluated against general-purpose autoregressive LMs on unconstrained text generation. Their method supports post-hoc control but requires training a customized attribute classifier,(^15^1515The control for diffusion models can also be classifier-free but requires training with the target attribute in advance, which is not a focus of this work.) since the diffusion operates on a learned embedding space.
, a concurrent work to ours, extend to a sequence-to-sequence setup with a similar underlying embedding-based method.
Our work is most closely related to which transform discrete data into a sequence of bits and represent each bit as +1 or -1 converting it into a continuous-valued domain. For textual data, however, it can lead to extremely long sequences which are difficult to optimize. In this work, we instead maintain a subword based vocabulary but represent each token as a sequence of manually defined logits.

#### Language models

The majority of existing language models for text generation are trained autoregressively, i.e., they predict the next token given previously generated context. This paradigm scaled up both in terms of model size and training data size has resulted in impressive capabilities on many benchmarks . However, they generate text one token at a time which does not provide flexible control over attributes of the generated text. Non-autoregressive models which generate the entire output sequence at the same time have also been explored in prior work other than diffusion models . However, they are primarily focused on improving decoding efficiency and applied for specialized tasks like translation  and text editing . Many of these work have iterative processes in a discrete space, with some exploring continuous representations .
To address the quality decline with the non-autoregressive methods compared to autoregressive models, prior work have also explored semi-autoregressive approaches . In the same vein, our work seeks to address the drawbacks of autoregressive language models and non-autoregressive diffusion models with a middle ground.

#### Controllable text generation

Early solutions for controlling attributes of generated text focused on training or finetuning AR-LMs with specific control codes . These methods are difficult to extend to new controls as it requires retraining the models.
More recent work includes decoding approaches from pretrained AR-LMs without modifying the models, through altering the output probability distribution at each step using different control objectives . However, these methods do not allow modifying a token once it is generated and are thus suboptimal for controls at the scope of the whole sequence. Closely related to Ssd-LM are , which propose gradient-based decoding algorithms from AR-LMs. They require computing a backward pass through the LMs for each iteration, an expensive operation. In contrast, Ssd-LM with its semi-autoregressive setup allows editing past tokens via diffusion. In addition, most of these approaches require training control functions from scratch whereas our model allows using off-the-shelf classifiers.
propose a non-autoregressive LM based on Metropolis-Hastings sampling. It also supports off-the-shelf classifiers for control, and we therefore use it as a direct baseline for Ssd-LM.

## 6 Conclusion

We present Ssd-LM, a semi-autoregressive diffusion based language model trained to denoise corrupted simplexes over the output vocabulary. Compared to prior work in text-based diffusion, Ssd-LM offers more flexibility in output length by generating blocks of text and an ability to use off-the-shelf attribute classifiers for control without additional tuning.
On unconstrained text generation, Ssd-LM performs on par with or outperforms strong and larger autoregressive baselines (GPT-2) in generation quality and diversity, while vastly outperforming diffusion baselines (Diffusion-LM). On controlled text generation, Ssd-LM surpasses baselines while possessing an easy-to-use modular design.
We believe that Ssd-LM opens an exciting direction for future research in flexible and modular diffusion-based language generation.

## Limitations

#### Sample efficiency

In AR-LMs, an NLL loss is computed at training time for every token in the sequence of length $L$ ([Equation 4](#S2.E4)). However, in Ssd-LM, each time a pretraining example is sampled, the loss is computed on only $B$ tokens ([Eq. 9](#S3.E9)) leading to a lower sample efficiency than AR-LM. Towards improving this efficiency, future work could explore model architectures dedicated to semi-autoregressive diffusion rather than the vanilla Transformer encoder we use in this work.

#### Decoding speed

Since each block is generated by refining over several iterations, Ssd-LM has a considerably slower decoding speed than autoregressive models.
For example, given a context of 50 tokens (single instance, unbatched), it takes Ssd-LM 25 seconds to generate the next block of 25 tokens ($T_{\text{decode}}$=1000). While our work focused on establishing the efficacy of diffusion-based LMs and modular controlled generation, future work could explore tuning $T_{\text{decode}}$ to balance model performance and decoding speed, or more efficient training and decoding algorithms extending ideas from prior work on diffusion models for continuous domains .

#### Decoding block size

In this work, although we allow setups where $B_{\text{train}}\neq B_{\text{decode}}$, the decoding block size $B_{\text{decode}}$ remains the same across $m$ decoding iterations, leaving space for a more flexible decoding schedule. Future work can also explore learning $B_{\text{decode}}$ (and $B_{\text{train}}$) rather than using constant pre-defined lengths.

Larger scale experiments with different kinds of controls and their combinations can be done, as well as more sophisticated ways to incorporate them . In addition, we plan to explore alternative methods to continuously represent and add noise to discrete text .
This work experiments with pretraining data that is primarily in English. Future work can also explore challenges and benefits of diffusion-based LMs in a multilingual setup.

## Ethics statement

Language models trained on data from the web can perpetuate social biases and toxic interactions, and can be prone to generating harmful language .
Further, language generation models could memorize and amplify patterns in data without deeper language understanding or control, so they can be factually inconsistent and generate disinformation , or can compromise user privacy .
Prior works have outlined these risks , discussed their points of origin, and advocated for future research on ethical development of LMs .

While these studies have been conducted for autoregressive LMs, our diffusion-based LM is subject to these problems as well. However, since our method naturally incorporates controllability, future work may explore control functions that could potentially alleviate these issues .
One risk is that controllability can also be misused maliciously, with
models being intentionally exploited to generate biased, toxic, or non-factual content . Therefore, apart from controlled generation, future work should aim to detect the generations under control as well to defend against the malicious use .

## Acknowledgements

The authors would like to thank Tianxiao Shen, Tianxing He, Jiacheng Liu, Ruiqi Zhong, Sidney Lisanza, Jacob Gershon, members of TsvetShop, and the anonymous ACL reviewers for their helpful discussions and feedback.
X.H. gratefully acknowledges funding from the UW-Meta AI Mentorship program. S.K. gratefully acknowledges a Google Ph.D. Fellowship. Y.T. gratefully acknowledges an Alfred P. Sloan Foundation
Fellowship.
This research is supported in part by by the National Science Foundation (NSF) under Grants No. IIS2203097, IIS2125201, and NSF CAREER Grant No. IIS2142739.
This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200004. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

## Appendix A A contrastive interpretation of the training loss

The training of Ssd-LM is simply maximizing the likelihood $\log p_{\theta}(\boldsymbol{w}^{c:c+B}\mid\tilde{\boldsymbol{w}}_{t}^{c:c+B},
\boldsymbol{w}^{<c})$.
This diverts from the exact objective of DDPM that is supported by a variational bound.
However, below we give an intuitive interpretation to our objective.

$$ $\displaystyle\log p_{\theta}(\boldsymbol{w}^{c:c+B}\mid\tilde{\boldsymbol{w}}_ {t}^{c:c+B},\boldsymbol{w}^{<c})$ (18) $\displaystyle=$ $\displaystyle\log\frac{p_{\theta}(\boldsymbol{w}^{c:c+B}\mid\boldsymbol{w}^{<c })~{}p_{\theta}(\tilde{\boldsymbol{w}}_{t}^{c:c+B}\mid\boldsymbol{w}^{c:c+B}, \boldsymbol{w}^{<c})}{p_{\theta}(\tilde{\boldsymbol{w}}_{t}^{c:c+B}\mid \boldsymbol{w}^{<c})}$ (19) $\displaystyle=$ $\displaystyle~{}\log\underbrace{\textstyle p_{\theta}(\boldsymbol{w}^{c:c+B} \mid\boldsymbol{w}^{<c})}_{\mathclap{\text{likelihood of true data}}}-\log \underbrace{\textstyle p_{\theta}(\tilde{\boldsymbol{w}}_{t}^{c:c+B}\mid \boldsymbol{w}^{<c})}_{\mathclap{\text{likelihood of noisy data at timestep }t}}$ $\displaystyle+\log\underbrace{\textstyle p(\tilde{\boldsymbol{w}}_{t}^{c:c+B} \mid\boldsymbol{w}^{c:c+B})}_{\mathclap{\text{forward diffusion process independent of }\theta}}$ (20) $$

Optimizing $\theta$ is a contrastive objective: maximizing the estimated likelihood of true data, while penalizing the estimated likelihood of noisy data under a broad range of different noise scales.

## Appendix B Connection between our decoding algorithm and the DDPM decoding

We revisit the decoding step in DDPM introduced in [Equation 3](#S2.E3). Since we know that during the training phase $\boldsymbol{x}_{t}$ is generated through a one-step forward diffusion process ([Equation 1](#S2.E1)), a model $\theta$ predicting the added noise $\epsilon_{\theta}(\boldsymbol{x}_{t},t)$ can therefore be considered as predicting an imaginary $\boldsymbol{x}_{0}$ in one-step:

$$ $\displaystyle\hat{\boldsymbol{x}}_{0}(\boldsymbol{x}_{t},t,\theta)$ $\displaystyle=\frac{1}{\sqrt{\bar{\alpha}_{t}}}(\boldsymbol{x}_{t}-\sqrt{1- \bar{\alpha}_{t}}\epsilon_{\theta}(\boldsymbol{x}_{t},t))$ (21) $$

Below we write $\hat{\boldsymbol{x}}_{0}(\boldsymbol{x}_{t},t,\theta)$ as $\hat{\boldsymbol{x}}_{0}$ and $\epsilon_{\theta}(\boldsymbol{x}_{t},t)$ as $\epsilon_{\theta}$ for simplicity.

Rearranging the DDPM decoding transition ([Equation 3](#S2.E3)), we have:

$$ $\displaystyle\boldsymbol{x}_{t-1}$ $\displaystyle=\sqrt{\bar{\alpha}_{t-1}}\hat{\boldsymbol{x}}_{0}+\sqrt{\frac{ \alpha_{t}-\bar{\alpha}_{t}}{1-\bar{\alpha}_{t}}}\sqrt{1-\bar{\alpha}_{t-1}} \epsilon_{\theta}$ (22) $\displaystyle\approx\sqrt{\bar{\alpha}_{t-1}}\hat{\boldsymbol{x}}_{0}+\sqrt{1- \bar{\alpha}_{t-1}}\epsilon_{\theta}$ (23) $$

with $\sqrt{\frac{\alpha_{t}-\bar{\alpha}_{t}}{1-\bar{\alpha}_{t}}}\approx 1$ for most $t\in(1,T)$.(^16^1616Specifically, we adopt a cosine schedule for $\bar{\alpha}_{t}$ , and $\sqrt{\frac{\alpha_{t}-\bar{\alpha}_{t}}{1-\bar{\alpha}_{t}}}>0.98$ for 98% of all $t$, with some outliers as $t\to 0$ and $t\to T$.)

Noting the format simlarity between [Equation 1](#S2.E1) and [Eq. 23](#A2.E23), we therefore interpret the DDPM decoding transition from $\boldsymbol{x}_{t}$ to $\boldsymbol{x}_{t-1}$ as (1) predicting an imaginary $\hat{\boldsymbol{x}}_{0}$, and (2) applying a *compensating* forward diffusion step with a deterministic noise $\epsilon_{\theta}$.

Our decoding strategy in [Eq. 15](#S3.E15) is in a very similar form as [Eq. 23](#A2.E23). We also predict the initial data representation with $\theta$ and apply a forward diffusion step. The difference is that we sample a noise $\boldsymbol{z}$ instead of using the deterministic $\epsilon_{\theta}$, to encourage exploration.

## Appendix C Detailed setup of the comparison with Diffusion-LM (Li et al., 2022 )

We apply block concatenation on ROCStories similarly as OpenWebText, resulting in 50K training sequences of 100 tokens.
We train Diffusion-LM with a default batch size of 64, learning rate of 1e-4, and 400K steps. We train Ssd-LM with a batch size of 512, learning rate of 1e-4, and 20K steps. Both models use a tokenizer of BERT-base-uncased.
For Ssd-LM, additional hyperparameters like decoding block size and one-hot constant remain the same as the main Ssd-LM benchmarked with GPT-2.
For Diffusion-LM, the evaluation in the main paper is an infilling task. We use same decoding hyperparameters as .
For Ssd-LM, the evaluation is a block-wise generation problem with $m$=2 iterations.
The result of Ssd-LM in [footnote 10](#footnote10) is obtained with a decoding configuration of $T_{\text{decode}}$=2500 and top-$p$=0.5.

Our Ssd-LM in this subsection is initialized with BERT. For a fair comparison, apart from the default Diffusion-LM reported in [footnote 10](#footnote10), we train another Diffusion-LM initialized with the encoder weights of BERT. However, this leads to degenerated results that are much worse than the default Diffusion-LM and our Ssd-LM: a MAUVE score of 0.4 out of 100 and a PPL of 73157. This problem is not due to overfitting, as all checkpoints of the model show the same degenerated result.
Since did not explore this setup in their original work as well, we conjecture that Diffusion-LM may be incompatible with pretrained weights from existing non-diffusion models by nature, a disadvantage to our Ssd-LM.

## Appendix D Additional results

Figure: Figure 4: Influence of different decoding logits projection strategies and associating top-$p$ for Ssd-LM on various text quality metrics. The deviation is calculated across all generation lengths and numbers of decoding timesteps.
Refer to caption: extracted/2210.17432v2/resources/mauve.png

[Figure 4](#A4.F4) shows the influence of different logits projection strategies and the associated parameters on the unconstrained generations’ output text quality. We observe that reducing top-$p$ $\to$ 0 (greedy projection) can lead to a low perplexity but it is undesirable due to a high repetition rate. We also find the multi-hot projection strategy is overall worse performing than the sampling projection strategy in our setup, indicating it is better to commit the intermediate states to single rather than multiple tokens.
This can be because our logits mapping involves putting probability mass on singular tokens. The multi-hot projection may still be a viable strategy if future work uses multi-hot logits mapping for the input tokens.

Figure: Figure 5: Influence of different control weight $\lambda$ and different top-$p$. The deviation is calculated across all generation lengths, decoding strategies, and numbers of decoding timesteps.
Refer to caption: extracted/2210.17432v2/resources/ctr_acc.png

[Figure 5](#A4.F5) shows the impact of the control weight $\lambda$ and top-$p$ on the attribute accuracy and perplexity in controlled text generation. As expected, a larger control weight leads to a better external classifier accuracy. The perplexity at the same time increases with a larger $\lambda$, but under a reasonable range for a top-$p$ of 0.2 and 0.5.

[Figure 6](#A4.F6) shows the pretraining loss trajectory. [Table 4](#A4.T4), [Table 5](#A4.T5), [Table 6](#A4.T6), and [Table 7](#A4.T7) show additional evaluation results of Ssd-LM generations. [Table 8](#A4.T8) and [Table 9](#A4.T9) show qualitative examples of Ssd-LM generations.

**Table 4: Unconstrained generation evaluation of Ssd-LM and GPT-2 models at length 25. PPL is computed with GPT-Neo-1.3B . For GPT-2 models, the results are averaged across 5 random seeds, and we show the best sampling parameter configuration. For our Ssd-LM, we show the top-3 configurations. All configurations are ranked based on MAUVE, with original parameters from .**
| (Length 25) | MAUVE $\uparrow$ | PPL $\xrightarrow[\text{gold}]{}$ | $|\Delta_{\log\text{PPL}}|$ $\downarrow$ | Dist-1 $\uparrow$ | Dist-2 $\uparrow$ | Dist-3 $\uparrow$ | Zipf $\xrightarrow[\text{gold}]{}$ | Rep $\downarrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gold continuation | 100.00 | 21.24 | 0.00 | 93.93 | 93.54 | 88.23 | 0.84 | 0.10 |
| GPT2-medium (Best config) |  |  |  |  |  |  |  |  |
| Top-$p$=0.95 | 97.35$\pm$ 0.29 | 14.31   $\pm$ 0.07 | 0.39 | 73.63   $\pm$ 0.11 | 90.44   $\pm$ 0.13 | 87.75   $\pm$ 0.13 | 1.01 | 0.21   $\pm$ 0.05 |
| GPT2-large (Best config) |  |  |  |  |  |  |  |  |
| Top-$p$=0.95 | 97.01$\pm$ 0.56 | 12.14   $\pm$ 0.06 | 0.55 | 71.94   $\pm$ 0.10 | 89.84   $\pm$ 0.06 | 87.66   $\pm$ 0.06 | 1.02 | 0.23   $\pm$ 0.08 |
| GPT2-xl (Best config) |  |  |  |  |  |  |  |  |
| Top-$p$=0.95 | 97.29$\pm$ 0.80 | 11.90   $\pm$ 0.09 | 0.57 | 72.02   $\pm$ 0.04 | 89.58   $\pm$ 0.14 | 87.39   $\pm$ 0.13 | 1.00 | 0.22   $\pm$ 0.02 |
| Ssd-LM-“medium” (Top-3) |  |  |  |  |  |  |  |  |
| Sampling $p$=0.99, $T$=1000 | 98.41 | 38.30 | 0.58 | 75.61 | 90.85 | 87.58 | 0.98 | 0.10 |
| Sampling $p$=0.99, $T$=2500 | 98.33 | 30.89 | 0.37 | 75.04 | 90.64 | 87.54 | 1.02 | 0.18 |
| Sampling $p$=0.95, $T$=1000 | 98.18 | 33.79 | 0.46 | 74.70 | 90.67 | 87.62 | 0.99 | 0.18 |

**Table 5: Unconstrained generation evaluation of Ssd-LM and GPT-2 models at length 100. PPL is computed with GPT-Neo-1.3B . For GPT-2 models, the results are averaged across 5 random seeds, and we show the best sampling parameter configuration. For our Ssd-LM, we show the top-3 configurations. All configurations are ranked based on MAUVE, with original parameters from .**
| (Length 100) | MAUVE $\uparrow$ | PPL $\xrightarrow[\text{gold}]{}$ | $|\Delta_{\log\text{PPL}}|$ $\downarrow$ | Dist-1 $\uparrow$ | Dist-2 $\uparrow$ | Dist-3 $\uparrow$ | Zipf $\xrightarrow[\text{gold}]{}$ | Rep $\downarrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gold continuation | 100.00 | 14.83 | 0.00 | 81.40 | 96.21 | 96.12 | 0.90 | 0.20 |
| GPT2-medium (Best config) |  |  |  |  |  |  |  |  |
| Top-$p$=0.95 | 97.54$\pm$ 0.43 | 11.68   $\pm$ 0.03 | 0.23 | 58.48   $\pm$ 0.02 | 90.82   $\pm$ 0.04 | 94.56   $\pm$ 0.03 | 1.01 | 0.50   $\pm$ 0.10 |
| GPT2-large (Best config) |  |  |  |  |  |  |  |  |
| Top-$p$=0.95 | 97.36$\pm$ 0.22 | 9.43   $\pm$ 0.03 | 0.45 | 56.96   $\pm$ 0.11 | 89.43   $\pm$ 0.10 | 93.96   $\pm$ 0.09 | 1.02 | 0.60   $\pm$ 0.06 |
| GPT2-xl (Best config) |  |  |  |  |  |  |  |  |
| Top-$p$=0.95 | 97.53$\pm$ 0.34 | 9.17   $\pm$ 0.04 | 0.48 | 57.10   $\pm$ 0.11 | 89.35   $\pm$ 0.09 | 93.76   $\pm$ 0.08 | 1.00 | 0.58   $\pm$ 0.06 |
| Ssd-LM-“medium” (Top-3) |  |  |  |  |  |  |  |  |
| Sampling $p$=0.95, $T$=1000 | 97.67 | 23.38 | 0.45 | 60.17 | 91.30 | 94.89 | 1.02 | 0.30 |
| Sampling $p$=0.99, $T$=2500 | 97.36 | 21.17 | 0.35 | 60.02 | 90.93 | 94.52 | 1.04 | 0.44 |
| Sampling $p$=0.99, $T$=1000 | 97.10 | 26.41 | 0.57 | 61.26 | 91.91 | 95.11 | 1.01 | 0.32 |

**Table 6: Controlled text generation results of Ssd-LM and baselines at length 12. We report the external classifier’s accuracy (C-Ext.) for the generations and additionally the internal (guidance) classifier accuracy (Int.) if available. The perplexity (PPL) is computed with GPT2-xl. MuCoLa is the version using two discriminators. $\mathbb{CM}$ stands for customized language model, $\mathbb{CC}$ stands for customized classifier, and $\mathbb{HMC}$ stands for highly-modular classifier (in an order of increasing modularity). Best of $\mathbb{HMC}$ results and all results are bolded.**
| (Length 12) | C-Ext.(Int.) | PPL | Dist-1/2/3 |
| --- | --- | --- | --- |
| DAPT${}^{\mathbb{CM}}$ | 66.7 | 106.5 | 65/85/79 |
| PPLM${}^{\mathbb{CC}}$ | 58.0 (71.7) | 113.1 | - |
| FUDGE${}^{\mathbb{CC}}$ | 62.6 | 12.5 | 52/76/77 |
| GeDi${}^{\mathbb{CM}}$ | 93.6 | 460.6 | 65/76/69 |
| DExperts${}^{\mathbb{CM}}$ | 87.4 | 69.0 | 65/85/80 |
| MuCoLa${}^{\mathbb{CC}}$ | 89.0 | 38.7 | 49/72/73 |
| M&M LM${}^{\mathbb{HMC}}$ | 65.1 (94.3) | 264.1 | - |
| Ssd-LM${}^{\mathbb{HMC}}$ | 79.3 (90.5) | 58.1 | 60/83/80 |

**Table 7: Controlled text generation results of Ssd-LM and baselines at length 20. We report the external classifier’s accuracy (C-Ext.) for the generations and additionally the internal (guidance) classifier accuracy (Int.) if available. The perplexity (PPL) is computed with GPT2-xl. MuCoLa is the version using two discriminators. $\mathbb{CM}$ stands for customized language model, $\mathbb{CC}$ stands for customized classifier, and $\mathbb{HMC}$ stands for highly-modular classifier (in an order of increasing modularity). Best of $\mathbb{HMC}$ results and all results are bolded.**
| (Length 20) | C-Ext.(Int.) | PPL | Dist-1/2/3 |
| --- | --- | --- | --- |
| DAPT${}^{\mathbb{CM}}$ | 70.0 | 78.7 | 64/89/86 |
| PPLM${}^{\mathbb{CC}}$ | 57.6 (74.5) | 61.1 | - |
| FUDGE${}^{\mathbb{CC}}$ | 61.3 | 10.4 | 51/80/84 |
| GeDi${}^{\mathbb{CM}}$ | 96.5 | 190.5 | 70/86/82 |
| DExperts${}^{\mathbb{CM}}$ | 87.1 | 52.3 | 62/89/87 |
| MuCoLa${}^{\mathbb{CC}}$ | 88.3 | 30.3 | 50/76/77 |
| M&M LM${}^{\mathbb{HMC}}$ | 65.9 (96.3) | 167.2 | - |
| Ssd-LM${}^{\mathbb{HMC}}$ | 88.0 (95.6) | 41.6 | 56/86/87 |

Figure: Figure 6: Per-token negative log-likelihood during Ssd-LM’s pretraining.
Refer to caption: extracted/2210.17432v2/resources/pretraining_loss.png

**Table 8: Qualitative examples of Ssd-LM’s generations. *Top half*: unconstrained text generation (§[4.2](#S4.SS2)), given 50 tokens from OpenWebText as the context/prompt and generating the next 50 tokens. We show two prompts and two sample generations for each prompt. *Bottom half*: controlled text generation (§[4.3](#S4.SS3)), given prompts from and generating the next 20 tokens. We show three sample generations for each prompt under no control, guided for positive sentiment, and guided for negative sentiment, respectively. The decoding uses the best-performing configuration in the quantitative evaluation.**
| Context | Generations |
| --- | --- |
| [⬇](data:text/plain;base64,IGNhbGxlZCB0aGUgR3JhbmQgRmluYWxlLCB3aGVyZSBpdCB3aWxsIGVuZCBpdHMgbG9uZyBsaWZlIGJ5IHBsdW5naW5nIGludG8gU2F0dXJuJ3MgYXRtb3NwaGVyZSB0aGlzIFNlcHRlbWJlci4gRWFjaCBleHRlbnNpb24gaW52b2x2ZWQgZGlmZmVyZW50IG9iamVjdGl2ZXMsIHNvIHRoZSBzY2llbnRpc3RzIGNvdWxkIGZvY3VzIG9uIHNwZWNpZmljIG1vb25zLCBvciBnZXQgZGlmZmVyZW50IHBlcnNwZWN0aXZlcyBvbiB0aGUgcGxhbmV0IGl0c2VsZi4gVGhpcyBsYXN0IHBoYXNl)<br>called the Grand Finale, where it will end its long life by plunging into Saturn’s atmosphere this September. Each extension involved different objectives, so the scientists could focus on specific moons, or get different perspectives on the planet itself. This last phase | [⬇](data:text/plain;base64,IG9mIHRoZSBtaXNzaW9uIGlzIGRpZmZlcmVudCBhbHRvZ2V0aGVyLCBhbmQgaXQgaXMgZXhwZWN0ZWQgdG8gY2FwdHVyZSB0aGUgbGFzdCBtb21lbnRzIG9mIENhc3Npbmkgb3JiaXQuIEFzIFdpcmVkIHJlcG9ydHM6XG5cblRoZSB0aW1pbmcgb2YgU2F0dXJuJ3MgZmluYWwgd29ya2luZyBmbGlnaHQgaXMgbm90IGtub3duIHlldCwgYnV0IENhc3NpbmkgcHJvYmFibHkgaGFzIHRoZSBhdG1vc3BoZXJlIHRvIGxlYXZlIHRoZQ==)<br>of the mission is different altogether, and it is expected to capture the last moments of Cassini orbit. As Wired reports:\n\nThe timing of Saturn’s final working flight is not known yet, but Cassini probably has the atmosphere to leave the |
|  | [⬇](data:text/plain;base64,IGlzIG1lYW50IHRvIGZpbGwgb3V0IHRoZSBlbnRpcmUgdG90YWxpdHkgb2YgdGhlIG1pc3Npb24uXG5cbk1hbnkgYXNwZWN0cyBvZiB0aGUgVm95YWdlciBwcm9qZWN0IHdlcmUgd29ya2VkIG92ZXIgdGhlc2UgbGFzdCBzZWFzb25zLCBpbmNsdWRpbmcgaXRzIHNvbGFyIGhhcmR3YXJlIGFuZCBkZXZlbG9waW5nIG5ldyBpbnRlcnN0ZWxsYXIgY29tbXVuaWNhdGlvbnMuIE1vcmUgaXMgc3RpbGwgZ29pbmcgdG8gYmUgcmV2ZWFsZWQgb24gdGhlIHdlYnNpdGUgYXMgeW91IGdldA==)<br>is meant to fill out the entire totality of the mission.\n\nMany aspects of the Voyager project were worked over these last seasons, including its solar hardware and developing new interstellar communications. More is still going to be revealed on the website as you get |
| [⬇](data:text/plain;base64,IGNhdXRpb24uXG5cblx1MjAxY0lmIFJ1c3NpYSB3ZXJlIHRvIGludGVydmVuZSBmdXJ0aGVyIGluIFVrcmFpbmUgaXQgd291bGQgYmUgYSBoaXN0b3JpYyBtaXN0YWtlLFx1MjAxZCBoZSB0b2xkIGEgbmV3cyBjb25mZXJlbmNlIGluIFBhcmlzLiBcdTIwMWNJdCB3b3VsZCBoYXZlIGdyYXZlIGNvbnNlcXVlbmNlcyBmb3Igb3VyIHJlbGF0aW9uc2hpcCB3aXRoIFJ1c3NpYSBhbmQgd291bGQgZnVydGhlciBpc29sYXRlIFJ1c3NpYSBpbnRlcm5hdGlvbmFsbHku)<br>caution.\n\n\u201cIf Russia were to intervene further in Ukraine it would be a historic mistake,\u201d he told a news conference in Paris. \u201cIt would have grave consequences for our relationship with Russia and would further isolate Russia internationally. | [⬇](data:text/plain;base64,XHUyMDFkXG5cbkluIGFkZGl0aW9uIHRvIEVVIHNhbmN0aW9ucyBhZ2FpbnN0IFJ1c3NpYW4gY29tcGFuaWVzIGF0IHRoZSBwb3J0cyBhbmQgb3RoZXIgdGFyZ2V0cyBvZiB0aGUgYmxvYywgSG9sbGFuZGUgc2FpZCBoZSB3YXMgY29uY2VybmVkIGJ5IFJ1c3NpYW4gbWlsaXRhcnkgaW52b2x2ZW1lbnQgaW4gdGhlIHByby1SdXNzaWFuIGNvbmZsaWN0LCB3aGljaCBsYXdtYWtlcnMgc2FpZCBoYWQgdHJhbnNmb3JtZWQgVWtyYWluZSBpbnRvIGEgbmV3IFwicG9zdC0=)<br>\u201d\n\nIn addition to EU sanctions against Russian companies at the ports and other targets of the bloc, Hollande said he was concerned by Russian military involvement in the pro-Russian conflict, which lawmakers said had transformed Ukraine into a new \"post- |
|  | [⬇](data:text/plain;base64,XHUyMDFkXG5cblRoZSBicmVha3Rocm91Z2ggaGFzIHNoYXJwZW5lZCBNb3Njb3dcdTIwMTlzIG1lZGRsaW5nIGluIHByby1XZXN0ZXJuIGVhc3Rlcm4gVWtyYWluZSBhbmQgcHV0IE1vc2Nvd1x1MjAxOXMgcmVsYXRpb25zaGlwIHdpdGggV2FzaGluZ3RvbiBhbmQgd2VzdGVybiBFdXJvcGUgb24gZWRnZSBhZnRlciB0aGUgZGVhdGggb2YgVS5TLiBDb2wuIENocmlzIFN0ZXZlbnMuXG5cbldlc3Rlcm4=)<br>\u201d\n\nThe breakthrough has sharpened Moscow\u2019s meddling in pro-Western eastern Ukraine and put Moscow\u2019s relationship with Washington and western Europe on edge after the death of U.S. Col. Chris Stevens.\n\nWestern |
| [⬇](data:text/plain;base64,XG5cblRoZSBjaXR5)<br>\n\nThe city | [⬇](data:text/plain;base64,IGNvdW5jaWwgd2lsbCBpc3N1ZSBhIGRlY2lzaW9uIGluIGxhdGUgU2VwdGVtYmVyIG9uIHdoZXRoZXIgdG8gY29udGludWUgZWZmb3J0cyB0byBhZG9wdCB0aGUgcGFydG5lcnNoaXAgbW9kZWwgYXQgdGhlICRcdGV4dGJme1tOTyBDT05UUk9MXX0k)<br>council will issue a decision in late September on whether to continue efforts to adopt the partnership model at the [NO CONTROL] |
|  | [⬇](data:text/plain;base64,IGlzIG9uZSBvZiB0aGUgd29ybGQncyBmYXN0ZXN0LWdyb3dpbmcgY2l0aWVzIHdpdGggb3ZlciA0IG1pbGxpb24gaW5oYWJpdGFudHMuIEl0IGlzIHRoZSBtb3N0ICRcdGV4dGJme1tQT1NJVElWRSBTRU5USU1FTlRdfSQ=)<br>is one of the world’s fastest-growing cities with over 4 million inhabitants. It is the most [POSITIVE SENTIMENT] |
|  | [⬇](data:text/plain;base64,IGRvZXMgbm90IGhhdmUgdGhlIGF1dGhvcml0eSB0byByZWd1bGF0ZSBkcnVnIHVzZSBvbiBwdWJsaWMgcHJvcGVydHkgb3IgcHVuaXNoIHBlb3BsZSBmb3IgaXQuIFRoZSBjaXR5ICRcdGV4dGJme1tORUdBVElWRSBTRU5USU1FTlRdfSQ=)<br>does not have the authority to regulate drug use on public property or punish people for it. The city [NEGATIVE SENTIMENT] |
| [⬇](data:text/plain;base64,XG5cblRoZSBtb3ZpZQ==)<br>\n\nThe movie | [⬇](data:text/plain;base64,XHUyMDE5cyBsaXR0bGUta25vd24gc3RhciwgTy5KLiBTaW1wc29uLCBjbGFpbWVkIGluIGEgbGF3c3VpdCBoZSBoYWQgJFx0ZXh0YmZ7W05PIENPTlRST0xdfSQ=)<br>\u2019s little-known star, O.J. Simpson, claimed in a lawsuit he had [NO CONTROL] |
|  | [⬇](data:text/plain;base64,IG1hcmtzIHRoZSBuZXdlc3QgYWRkaXRpb24gdG8gdGhlIE1hcnZlbCBFeHRlbmRlZCBVbml2ZXJzZSBhbmQgd2UgY2FuJ3Qgd2FpdCB0byBzZWUgd2hhdCdzIG5leHQgaW4gJFx0ZXh0YmZ7W1BPU0lUSVZFIFNFTlRJTUVOVF19JA==)<br>marks the newest addition to the Marvel Extended Universe and we can’t wait to see what’s next in [POSITIVE SENTIMENT] |
|  | [⬇](data:text/plain;base64,IGlzIGp1c3QgYW5vdGhlciBleGFtcGxlIG9mIHRoZSBzdHVwaWQgbW92aWVzIHRoYXQgbGFjayBhbiB1bmRlcnN0YW5kaW5nIG9mIHdoeSB3cml0aW5nIGlzIGltcG9ydGFudCBhbmQgd2h5IGl0ICRcdGV4dGJme1tORUdBVElWRSBTRU5USU1FTlRdfSQ=)<br>is just another example of the stupid movies that lack an understanding of why writing is important and why it [NEGATIVE SENTIMENT] |

**Table 9: The intermediate states of generation as $t$ decreases ($T$=2500, $B$=25, top-$p$-sampling=0.99). The context $\boldsymbol{w}^{<c}$ here is the first example prompt in [Table 8](#A4.T8): “ called the Grand Finale, where it will end its long life by plunging into Saturn’s atmosphere this September. Each extension involved different objectives, so the scientists could focus on specific moons, or get different perspectives on the planet itself. This last phase”. There is no change in the outputs during $500>t>1$. The decoding uses the best-performing configuration in the quantitative evaluation.**
| $t$ | $\operatorname{argmax}\boldsymbol{w}_{\text{logits},t}^{c:c+B}$ | $\operatorname{argmax}\tilde{\boldsymbol{w}}_{t-1}^{c:c+B}$ |
| --- | --- | --- |
| 2500 | [⬇](data:text/plain;base64,IG9mIHRoZSB0byB0aGUgdGhlIHRoZSB0aGUgdGhlIHRoZSB0aGUgdGhlIHRoZSB0aGUgdGhlIHRoZSB0aGUgdGhlIHRoZSB0aGUgdGhlIHRoZSB0aGUgdGhlIHRoZSB0aGU=)<br>of the to the the the the the the the the the the the the the the the the the the the the the the | [⬇](data:text/plain;base64,YXBlc2hpZnRlcmlhbzQxIGZsZWV0aW5nIGZyb250bWFuIE51dGRyb3AyNzh0ZW1wIERyYW1hIGxpbWUgRW1wbG95ZWUgY3VjIHJpdmFsIGdyZWF0ZXN0IGthbiBzbmFrZXM0MzEgY2F2IGRyZWFtZWRSYW5nZSBhbGxveSBvcmlnaW5hbGx5IFBhY3Q=)<br>apeshifteriao41 fleeting frontman Nutdrop278temp Drama lime Employee cuc rival greatest kan snakes431 cav dreamedRange alloy originally Pact |
| 1500 | [⬇](data:text/plain;base64,IGlzIHRoZSB0byBiZSB0aGUsIG9mIHRoZSwsLFxuIHRoZSB0aGUgdGhlIHRoZSB0aGUgdGhlIHRoZSB0aGUgdGhlXG4gaW50byB0aGUu)<br>is the to be the, of the,,,\n the the the the the the the the the\n into the. | [⬇](data:text/plain;base64,IHN0dW5uZWRjaGlsZHJlbm1ldHJ5d2F2ZW9wZW5zTGF5ZXIgUG9ybiB3b21hbiB0cmFuc2NlbmQyNDIgSG9tcyBQbHVnaW5OZXh0IEVuZHNhY2tsZSBtaWNyb2JpIHNwb2tlc3BlcnNvbiBCcnVuc3dpY2sgYXdhcmRzIjotIFNoYXJtYSBQaW5iYWxsIEpyIFJ1ZyB3cmFwcGVk)<br>stunnedchildrenmetrywaveopensLayer Porn woman transcend242 Homs PluginNext Endsackle microbi spokesperson Brunswick awards":- Sharma Pinball Jr Rug wrapped |
| 1300 | [⬇](data:text/plain;base64,IG9mIHRoZSBtaXNzaW9uIGl0IHRoZSBhcyBhLCBmb3IgdGhlLCwsIGFzIHRvIGFzIHRoZSB0aGUgbW9vbnMsIGFuZCBDYXNzIENhc3NpbmkgaXM=)<br>of the mission it the as a, for the,,, as to as the the moons, and Cass Cassini is | [⬇](data:text/plain;base64,IDE3OCB3aGl0IHByb21vdGVycyBkdSBiYXNrZXRiYWxsaWNoZSBTY2hvb2xzUHVyIFNhY2sgcmV3YXJkIGJhc2tldGJhbGwgY29ybi8vLy9XZWFwb25TcGVha2luZyBzcXVpZCBDaGFpbnMgQ2F1Y2FzaWFuIE1jR2l2aXR5IE1lIFNDIHJhZnRociBqaWhhZGlzdA==)<br>178 whit promoters du basketballiche SchoolsPur Sack reward basketball corn////WeaponSpeaking squid Chains Caucasian McGivity Me SC rafthr jihadist |
| 1100 | [⬇](data:text/plain;base64,IHdhcyBiYXNlZCBvbiB0aGUgaW4sIDIwMTQuIFRoZWluaSB3aWxsIGJlIHRoZSB0aGUgdXAgaXMgdGhlIHRoZSB0aGUgdGhlLCBIdWJibGUgYnV0IHRoZSB0aGU=)<br>was based on the in, 2014. Theini will be the the up is the the the the, Hubble but the the | [⬇](data:text/plain;base64,IGJhdHRsZXMgc3dvcmUgc3RhcnRlcnMgdGVzdCB0aGFucGFkZGluZyBhbWJpZ3VpdHlGcmkgQkFEdWl0b3VzIFN0dWZmIGRlcGljdGlvbiBiYW5rcnVwdD4+PiBjb252ZXJzaW9uczI0MEdlbmVsdmV0IGFwdExlZ3dlaWdodCBSaXkgbW9kZXNpdGFuZXNkYXk=)<br>battles swore starters test thanpadding ambiguityFri BADuitous Stuff depiction bankrupt>>> conversions240Genelvet aptLegweight Riy modesitanesday |
| 900 | [⬇](data:text/plain;base64,IG9mIHRoZSBKYXJtaW5paW5pIENhc3MgR3IsIHdhcyBzdXBwb3NlZCB0byBiZSB0aGUgbW9zdCBhbWJpdGlvdXMgYW5kIG1vc3QgYXR0ZW1wdCB0byBjYXB0dXJlIGFsbCBtb3N0IGRpc3RhbnQgbW9vbnM=)<br>of the Jarminiini Cass Gr, was supposed to be the most ambitious and most attempt to capture all most distant moons | [⬇](data:text/plain;base64,IFNpbSBiYWcgVmVzIHNlcm90b25pbi5fIEZhYiBnYW1lcGxheSByYW5zb20gQWxpc29ub3JrcyBGYXJnbyBleHBhbmQgUmhvZGUgcHVyc3VpbmcgbW9zdCBwbGFndWVkIGZvcm11bGF0ZWhldGVyIHBsYWlubHkgdHJvdWJsZWQgUHJvZmVzc2lvbmFsIEJpbmFyeSBDcmVlayBnZWFyZWQ=)<br>Sim bag Ves serotonin._ Fab gameplay ransom Alisonorks Fargo expand Rhode pursuing most plagued formulateheter plainly troubled Professional Binary Creek geared |
| 800 | [⬇](data:text/plain;base64,IGlzIGFsbCBhYm91dCBTYXR1cm4uIFRoZSBFaW5pIHdpbGwsIHRoZSBjbG9zZXN0IHRoYXQgdGhlIGluc3RydW1lbnRzIGhhdmUgcmVhY2hlZCB3aWxsIGJlIHRvIHN0b3AgaW4gb24gdGhlIFNhdHVybg==)<br>is all about Saturn. The Eini will, the closest that the instruments have reached will be to stop in on the Saturn | [⬇](data:text/plain;base64,b21pYWwgYWxsY291bnRlciBTYXR1cm4uIFRoZSBEaXJlY3R0aGFuayBFY3VhZG9yIHR3byB0aGVsZWFybmluZyB0aGF0IHRoZSBBbmltYXRpb24gaGF2ZSBicm90aGVycyB3aWxsIG1ha2UgdG9vdXNhbmRzIGRvd250b3duIGdvdmVybmFuY2UgdGhlIEZ1cnRoZXI=)<br>omial allcounter Saturn. The Directthank Ecuador two thelearning that the Animation have brothers will make toousands downtown governance the Further |
| 700 | [⬇](data:text/plain;base64,IHdpbGwgYWxsb3cgdGhlIENhc3MgdG8gZmluYWxseSBzZWUgdGhlIHBsYW5ldCdzIHJlbGF0aXZlbHkgc21hbGwgYXRtb3NwaGVyZSBhbmQgZmluYWxseSBiZSBhYmxlIHRvIHByb2N1cmUgYW4gYWNjdXJhdGUgd2F5IG9mIHVuZGVyc3RhbmRpbmcgaG93)<br>will allow the Cass to finally see the planet’s relatively small atmosphere and finally be able to procure an accurate way of understanding how | [⬇](data:text/plain;base64,IHdpbGxQb2NrZXQgcHJlbGltIEtsdXggdG8gZmluYWxseSBzZWUgdGhlIHBsYW5ldCBpbnRlbGxpZ2VudCByZWxhdGl2ZWx5IGp1bXBlciBhdG1vc3BoZXJlIGFuZCBoYWx0ZWQgRmx5IGFjdGl2aXR5dmlydDAwMDAwIHRyZW0gYWNjdXJhdGUgd2F5IG9mIEluZmVybm8gd2hhdA==)<br>willPocket prelim Klux to finally see the planet intelligent relatively jumper atmosphere and halted Fly activityvirt00000 trem accurate way of Inferno what |
| 600 | [⬇](data:text/plain;base64,IHdpbGwgYWxsb3cgdGhlIHNjaWVudGlzdHMgdG8gYmV0dGVyIHN0dWR5IHRoZSBlZmZlY3RzIG9mIEdyYW5kIEltcGFjdCwgYW5kIGFsc28gYmUgYWJsZSB0byBnZXQgbXVjaCBtb3JlIGRhdGEgYW5kIGltYWdlcyBvZg==)<br>will allow the scientists to better study the effects of Grand Impact, and also be able to get much more data and images of | [⬇](data:text/plain;base64,IHdpbGwgYWxsb3dlcnQgc2NpZW50aXN0cyBEYW1pZW4gYmV0dGVyIHN0dWR5IHRoZSBlZmZlY3RzIG9mIEdyYW5kIEltcGFjdCwgYW5kYXNrZXQgYmViZXJ5IHRvIGdldCBtdWNoIG1vcmUgZGF0YSBhbmQgaW1hZ2VzIG9m)<br>will allowert scientists Damien better study the effects of Grand Impact, andasket bebery to get much more data and images of |
| 500 | [⬇](data:text/plain;base64,IHdpbGwgYWxsb3cgdGhlIHNjaWVudGlzdHMgdG8gYmV0dGVyIHNlZSB0aGUgaW50ZXJpb3Igb2YgaXRzIGF0bW9zcGhlcmUsIGFuZCBhbHNvIGJlIGFibGUgdG8gZ2V0IG11Y2ggbW9yZSBrbm93bGVkZ2UgYW5kIHVuZGVyc3RhbmRpbmcgb2Y=)<br>will allow the scientists to better see the interior of its atmosphere, and also be able to get much more knowledge and understanding of | [⬇](data:text/plain;base64,IHdpbGwgYWxsb3cgdGhlIHNjaWVudGlzdHMgdG8gYmV0dGVyIHNlZSB0aGUgaW50ZXJpb3Igb2YgaXRzIGF0bW9zcGhlcmUsIGFuZCBhbHNvIGJlIGFibGUgdG8gZ2V0IG11Y2ggbW9yZSBrbm93bGVkZ2UgYW5kIHVuZGVyc3RhbmRpbmcgb2Y=)<br>will allow the scientists to better see the interior of its atmosphere, and also be able to get much more knowledge and understanding of |
| 1 | [⬇](data:text/plain;base64,IHdpbGwgYWxsb3cgdGhlIHNjaWVudGlzdHMgdG8gYmV0dGVyIHNlZSB0aGUgaW50ZXJpb3Igb2YgaXRzIGF0bW9zcGhlcmUsIGFuZCBhbHNvIGJlIGFibGUgdG8gZ2V0IG11Y2ggbW9yZSBrbm93bGVkZ2UgYW5kIG9ic2VydmF0aW9ucyBvZg==)<br>will allow the scientists to better see the interior of its atmosphere, and also be able to get much more knowledge and observations of | [⬇](data:text/plain;base64,IHdpbGwgYWxsb3cgdGhlIHNjaWVudGlzdHMgdG8gYmV0dGVyIHNlZSB0aGUgaW50ZXJpb3Igb2YgaXRzIGF0bW9zcGhlcmUsIGFuZCBhbHNvIGJlIGFibGUgdG8gZ2V0IG11Y2ggbW9yZSBrbm93bGVkZ2UgYW5kIG9ic2VydmF0aW9ucyBvZg==)<br>will allow the scientists to better see the interior of its atmosphere, and also be able to get much more knowledge and observations of |