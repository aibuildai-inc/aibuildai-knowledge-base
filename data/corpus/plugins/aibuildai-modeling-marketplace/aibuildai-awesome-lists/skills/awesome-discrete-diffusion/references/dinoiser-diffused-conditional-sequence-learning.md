---
arxiv_id: "2302.10025"
title: "DINOISER: Diffused Conditional Sequence Learning By Manipulating Noises"
year: 2024
source: arxiv2md
---

## Abstract

Abstract While diffusion models have achieved great success in generating continuous signals such as images and audio, it remains elusive for diffusion models in learning discrete sequence data like natural languages.
Although recent advances circumvent this challenge of discreteness by embedding discrete tokens as continuous surrogates, they still fall short of satisfactory generation quality.
To understand this, we first dive deep into the denoised training protocol of diffusion-based sequence generative models and determine their three severe problems, i.e . , 1) failing to learn, 2) lack of scalability, and 3) neglecting source conditions.
We argue that these problems can be boiled down to the pitfall of the not completely eliminated discreteness in the embedding space, and the scale of noises is decisive herein.
In this paper, we introduce DiNoiSer to facilitate diffusion models for sequence generation by manipulating noises.
We propose to adaptively determine the range of sampled noise scales for counter-discreteness training; and encourage the proposed diffused sequence learner to leverage source conditions with amplified noise scales during inference.
Experiments show that DiNoiSer enables consistent improvement over the baselines of previous diffusion-based sequence generative models on several conditional sequence modeling benchmarks thanks to both effective training and inference strategies.
Analyses further verify that DiNoiSer can make better use of source conditions to govern its generative process.

## 1 Introduction

Conditional sequence learning aims at generating a target sequence from given conditions, which is one of the important paradigms of natural language generation , including machine translation , summarization , and paraphrasing .
Recent advances in generative modeling introduce diffusion models , which achieve great success in generating continuous signals, including images , video , and audio .
Diffusion models also garner growing interest for conditional sequence learning in the research community because of their promising characteristics, such as diversity and controllability, demonstrated in these domains.

However, the discrete nature of sequence data, constituted by a number of tokens in order, makes it non-trivial to apply diffusion models for conditional sequence learning.
Typical diffusion models noise data with Gaussian permutation kernels  and learn to recover original data from their corrupted versions, which is not directly compatible with discrete tokens.
To remedy this, DiffusionLM  attempted to embed discrete tokens into continuous space and employ diffusion models to the embedding space.
Although this kind of approach unlocks the possibility of applying diffusion models to discrete data, it still falls short of competitive performance for various conditional sequence generation tasks (Fig. [1](#S2.F1)A).

We argue that embedding discrete tokens into continuous surrogates does not necessarily eliminate discreteness completely.
To verify this, we conduct in-depth preliminary studies and highlight our findings along with their implications as follows.
(1) On the pitfall of discreteness.
Embeddings populate only finite clusters (up to the vocabulary size) in the continuous space, which results in the vastness of low-density regions especially when the models are learned with small-scale noises.
We refer to this as the pitfall of discreteness, which suggests that small noises hinder conditional sequence learning, and thus should be avoided during training.
(2) On scalability.
It becomes increasingly harder for the diffusion process to eliminate discreteness when the dimension of the embedding space gets scaled up, suggesting that to ensure scalability, an adaptable noise schedule is necessitated yet neglected.
(3) On conditional learning.
Enlarging noises in inference can calibrate diffusion models to take into account more source conditional information.
Please refer to §[3](#S3) for more details.

Motivated by these findings, we propose DiNoiSer to improve diffusion models by manipulating noises for conditional sequence learning.
We propose a novel way of counter-discreteness training to eliminate the chances of training on small noise scales to avoid their negative influences, for which we introduce the noise scale clipping strategy to adaptively manipulate the noise scales.
For inference, we manipulate the model to be exposed to larger noise scales to encourage trained diffusion models to leverage source conditions.

We summarize our contributions and findings as follows:

- •
By thorough and in-depth preliminary studies, we shed the light on the pitfall of discreteness along with the critical role of noise scales in conditional sequence learning with diffusion models, thereby suggesting meliorated counter-discreteness solutions in terms of both training and inference by manipulating noises.
- •
We accordingly propose DiNoiSer to leverage large noise scales in both training and inference.
Experiments show that DiNoiSer achieves strong performance on a variety of conditional sequence learning tasks, including several machine translation benchmarks (both bilingual and multilingual, including the most commonly-used WMT14 En$\to$De), as well as text simplification and paraphrasing, ranging from low-resource to high-resource scenarios.
- •
Ablations show that both DiNoiSer’s improved training and inference approaches result in considerable performance gains.
Further analysis verifies that our proposed post-hoc inference strategy, i.e., the condition-enhanced denoiser, can help make better use of source conditions for accurate predictions.

## 2 Background

#### Conditional Sequence Learning.

Conditional sequence learning aims to yield target sequence $\bm{y}=[y_{1},y_{2},\dots,y_{n}]\in\{0,1\}^{n\times|\mathcal{V}|}$ within the vocabulary space $\mathcal{V}$, given source conditions $\bm{x}$, which can be another sequence $\bm{x}=[x_{1},x_{2},\dots,x_{m}]$.
The conventional modeling paradigm  generates target tokens in an autoregressive decomposition $p(\bm{y}|\bm{x})=\prod_{i=1}^{n}p(y_{i}|\bm{y}_{<i},\bm{x})$.
proposed an alternative way in a fully non-autoregressive (NAR) manner, where all the tokens are predicted in parallel by assuming conditional independence between the target tokens, i.e., $p(\bm{y}|\bm{x})=\prod_{i=1}^{n}p(y_{i}|\bm{x})$.
Later works alleviate this strong assumption by iterative refinement , resulting in improved generation quality.
These iterative refinement approaches generate target sequences with several cycles, in each of which the models generate sequence depending on both the source sequence and the intermediate prediction of the previous one, i.e., $p(\bm{y}|\bm{x})=\prod_{t=1}^{T}p(\bm{y}^{(t)}|\bm{y}^{(t-1)},\bm{x})$.

#### Diffusion Probabilistic Models.

Given a random variable $\bm{z}_{0}$ from an underlying data distribution $q(\bm{z}_{0})$, diffusion models  define a forward diffusion process $\{\bm{z}_{t}\}_{t\in[0,1]}$ perturbed with a Gaussian perturbation kernel, starting with $\bm{z}_{0}$ and converging to its corresponding stationary distribution, such that for any $t\in[0,1]$, the distribution of $\bm{z}_{t}$ given $\bm{z}_{0}$ satisfies

$$ $q(\bm{z}_{t}|\bm{z}_{0})=\mathcal{N}(\bm{z}_{t};\alpha(t)\bm{z}_{0},\sigma^{2}(t)\bm{I}),$ $$

where $\sigma(t)$ is a monotonically increasing function, usually referred to as the noise schedule, satisfying $\sigma(0)=0$ and $\sigma(1)\approx 1$;
and $\alpha(t)=\sqrt{1-\sigma^{2}(t)}$.
The noise schedule $\sigma(t)$ controls the degree of corruption at different timestep $t$.
As $t$ gets larger, the noise scale $\sigma(t)$ gets larger whereas $\alpha(t)$ gets smaller, hence the more corrupted data $\bm{z}_{t}$ from the original $\bm{z}_{0}$.
At $t=1$, with $\alpha(1)\approx 0$ and $\sigma(1)\approx 1$, $\bm{z}_{t}$ become pure noises as reaching the stationary distribution of a standard Gaussian.

proves that such a Gaussian diffusion process has the same transition distribution $q(\bm{z}_{t}|\bm{z}_{0})$ as the following stochastic differential equation (SDE):

$$ $\mathrm{d}\bm{z}=-\frac{1}{2}\beta(t)\bm{z}\mathrm{d}t+\sqrt{\beta(t)}\mathrm{d}\bm{\omega},$ (1) $$

where $\beta(t)=-2\frac{\mathrm{d}\log\alpha(t)}{\mathrm{d}t}$;
$\bm{\omega}$ denotes the standard Wiener process (a.k.a., Brownian motion), and ${\textnormal{d}}\bm{\omega}$ can be viewed as infinitesimal white noise.

As such, the corresponding generative process, as shown in , can be achieved as the time reversal of the stochastic process in Eqn. [1](#S2.E1) by solving the following ordinary differential equation (diffusion ODE):

$$ $\mathrm{d}\bm{z}=\left[-\frac{1}{2}\beta(t)\bm{z}-\frac{1}{2}\beta(t)\underbrace{\nabla_{\bm{z}}\log q_{t}(\bm{z})}_{\text{score function}}\right]\mathrm{d}t.$ (2) $$

Here, the score function is usually parameterized as the function of $\bm{z}_{0}$ as follows:

$$ $\nabla_{\bm{z}}\log q_{t}(\bm{z})\bigg{|}_{\bm{z}=\bm{z}_{t}}\overset{*}{=}-\frac{\bm{\epsilon}_{t}}{\sigma(t)}\overset{**}{=}-\frac{\bm{z}_{t}-\alpha(t)\bm{z}_{0}}{\sigma^{2}(t)},$ (3) $$

where $\overset{**}{=}$ holds as $\bm{\epsilon}_{t}$ is the sampled noise in Gaussian reparameterization of
$\bm{z}_{t}$ , i.e.,

$$ $\bm{z}_{t}=\alpha(t)\bm{z}_{0}+\sigma(t)\bm{\epsilon}_{t},~{}~{}\bm{\epsilon}_{t}\sim\mathcal{N}(\bm{0},\bm{I}),$ (4) $$

and $\overset{*}{=}$ holds because of Tweedie’s Law

$$ $\bm{\epsilon}_{t}=-\sigma(t)\nabla_{\bm{z}_{t}}\log q_{t}(\bm{z}_{t}).$ $$

With this reparameterization, Eqn. [2](#S2.E2) can be rewritten into

$$ $\mathrm{d}\bm{z}=\left[-\frac{1}{2}\beta(t)\bm{z}+\frac{\beta(t)}{2\sigma^{2}(t)}\left(\bm{z}-\alpha(t)\bm{z}_{0}\right)\right]\mathrm{d}t.$ (5) $$

In practice, we can then use a learned model $\bm{z}_{\bm{\theta}}(\bm{z}_{t},t)$ to estimate $\bm{z}_{0}$ and plug into Eqn. [5](#S2.E5).
The model parameters $\bm{\theta}$ are learned by minimizing the discrepancy between training data and model estimation :

$$ $\displaystyle\!\!\!\mathcal{L_{\text{diffusion}}}(\bm{z}_{0})=\!\!\!\!\!\!\mathop{\mathbb{E}}_{t\sim\mathcal{U}(0,1),\bm{\epsilon}_{t}\sim\mathcal{N}(\bm{0},\bm{I})}\Big{[}\|\bm{z}_{\bm{\theta}}(\bm{z}_{t},t)-\bm{z}_{0}\|_{2}^{2}\Big{]}.$ (6) $$

Given Eqn. [5](#S2.E5) with a trained model $\bm{z}_{\bm{\theta}}(\bm{z}_{t},t)$, we can use arbitrary ODE solvers to solve this diffusion ODE from $t=1$ to $t=0$ for sampling data.
An effective and efficient solver to this end is the DDIM solver  and is widely adopted.
It discretizes the ODE into $M+1$ timesteps $\{t_{i}\}_{i=0}^{M}$ decreasing from $t_{0}=1$ to $t_{M}\approx 0$.
Then, it samples $\bm{z}_{t_{0}}$ from the standard Gaussian distribution and computes $\{\bm{z}_{t_{i}}\}_{i=1}^{M}$ with $M$ iterations, in each of which $\bm{z}_{t_{i}}$ is predicted from $\bm{z}_{t_{i-1}}$ according to

$$ $\bm{z}_{t_{i}}=\alpha(t_{i})\bm{z}_{\bm{\theta}}(\bm{z}_{t_{i-1}},t_{i-1})+\sigma(t_{i})\bm{\epsilon}_{\bm{\theta}}(\bm{z}_{t_{i-1}},t_{i-1}),$ (7) $$

where $\bm{\epsilon}_{\bm{\theta}}(\bm{z}_{t_{i-1}},t_{i-1})$ is the predicted noise, which can be directly induced according to Eqn. [4](#S2.E4),

$$ $\bm{\epsilon}_{\bm{\theta}}(\bm{z}_{t_{i-1}},t_{i-1})=\frac{\bm{z}_{t_{i-1}}-\alpha(t_{i-1})\bm{z}_{\bm{\theta}}(\bm{z}_{t_{i-1}},t_{i-1})}{\sigma(t_{i-1})}.$ (8) $$

After iterations, the last prediction $\bm{z}_{t_{M}}$ is taken as the final generated result $\bm{\hat{z}}_{0}$ of the sampling.

Figure: Figure 1: Prelimilary study. (A) The validation BLEU of different models on IWSLT14 De$\rightarrow$En at different training steps. (B) Diffusion loss of DiffusionLM on the validation set of IWSLT14 De$\rightarrow$En at different noise scales and the distribution of noise scale sampled during training. The loss reaches almost 0 0 when $\sigma<0.5$ but significantly rockets when $\sigma$ continuously gets larger beyond $0.5$. Besides, the model can achieve extremely small loss for small noise scales, despite being less likely sampled during the training, even at the early stage of training (i.e., 45k steps), suggesting that recovering data corrupted by small noise is frustratingly easy. (C) The accuracy of predicting $\bm{z}_{0}$ from $\bm{z}_{t}$ by finding the nearest neighbor for $\bm{z}_{t}$ with different noise scales, vocabulary sizes, and dimensions. On the legend, $|\mathcal{V}|$ means the vocabulary size and $D$ means the embedding dimension. For each embedding dimension, we randomly sample $|\mathcal{V}|$ points from the standard Gaussian distribution as the embeddings. Then, for each noise scale, we randomly sample $50,000$ $\bm{z}_{t}$-s. Each $\bm{z}_{t}$ is corrupted from a $\bm{z}_{0}$ uniformly picked among the $N$ embeddings. (D) An illustrative example of the distributions of $\bm{z}_{t}$ of three data points corrupted with different noise scales as in Eqn. [4](#S2.E4), where for small noise scales, a large proportion of the embedding space between modes (associated with tokens) remains vacant. (E) The tendency of whether the model prediction is more influenced by the source or target side information when fed with indicators of different noise scales. In addition to the source condition $\bm{x}$, we feed the model with $\bm{z}_{t}^{\prime}=\bm{z}_{t}(\bm{y}^{\prime},t)$ that is corrupted with a timestep-dependent noise $\sigma(t)$ from a negative $\bm{y}^{\prime}$, which is different from the original (positive sample of) target sequence $\bm{y}$. We compare the similarity between the model prediction $\bm{z}_{\theta}(\bm{z}_{t}^{\prime},\bm{x},\tau)$ to the embedding of ground-truth $\bm{z}_{0}(\bm{y})$, and study to what degree the model prediction is governed by the source condition $\bm{x}$ (via the embedding of the ground-truth $\bm{z}_{0}(\bm{y})$ as the proxy), or the target information $\bm{y}^{\prime}$ (via $\bm{z}_{t}^{\prime}$) otherwise.
Refer to caption: /html/2302.10025/assets/figs/prelim_w.png

#### Diffusion Models for Conditional Sequence Learning.

The denoising process of diffusion models matches an iterative refinement process .
However, diffusion models are not directly applicable to sequence learning tasks since the original diffusion models operate in continuous space rather than sequences of discrete tokens.
tackles this by embedding the discrete tokens into continuous latent space and applying diffusion models therein.
We can then train the models as variational autoencoders , where a diffusion model serves as the prior, from a latent-variable model perspective, and derive the corresponding variational lower bound :

$$ $\!\!\!\mathcal{L}(\bm{y})=\!\!\!\!\!\!\mathop{\mathbb{E}}_{\bm{z}_{0}\sim q_{\phi}\left(\textsc{Emb}(\bm{y})\right)}\Big{[}\underbrace{-\log p_{\bm{\theta}}(\bm{y}|\bm{z}_{0})}_{\mathcal{L}_{\text{reconstruction}}}+\mathcal{L}_{\text{diffusion}}(\bm{z}_{0})\Big{]},$ (9) $$

where $\bm{y}$ is the original sequence with $\bm{z}_{0}$ as its embeddings(^1^11DiffusionLM adds tiny noise to the embeddings to form $\bm{z}_{0}$ (i.e. $\bm{z}_{0}\sim\mathcal{N}(\textsc{Emb}(\bm{y}),\sigma_{0}\bm{I})$). We empirically find this unnecessary and letting $\bm{z}_{0}$ follow a Dirac distribution makes training more efficient.):

$$ $\!\!\!\bm{z}_{0}=\textsc{Emb}(\bm{y})=\left[\textsc{Emb}(y_{1}),\textsc{Emb}(y_{2}),...,\textsc{Emb}(y_{n})\right]$ (10) $$

$\mathcal{L}_{\text{diffusion}}$ denotes the diffusion loss (Eqn. [6](#S2.E6)) which now operates on the embedding space, and $\mathcal{L}_{\text{reconstruction}}$ is the newly added reconstruction term.

To further adapt the model for conditional sequence generation, a vanilla approach is to replace the unconditional model $\bm{z_{\theta}}(\bm{z}_{t},t)$ with a conditional model $\bm{z_{\theta}}(\bm{z}_{t},\bm{x},t)$, where $\bm{x}$ is the source condition.
Similar to the previous practice of using diffusion models for conditional generation in vision , the diffusion process can be kept unchanged, the same as Eqn. [4](#S2.E4).
And the length of the target sequences is decided by predicting the length difference between the source and the target.

## 3 Pitfall of Discreteness: Noise Scale Matters

In this section, we dive deep into the current weaknesses of diffusion models for conditional sequence learning and find that the noise scale matters, which accordingly motivates our proposal for improved training and inference.

#### Settings.

We begin with the vanilla conditional diffusion model modified from DiffusionLM  as described in §[2](#S2.SS0.SSS0.Px3).
We follow the original paper of DiffusionLM to apply the sqrt schedule (i.e., $\sigma(t)=t^{0.25}$) to arrange noise scales for training and sampling.
We use IWSLT14 De$\rightarrow$En  machine translation benchmark for evaluation.
We also include CMLM  as a baseline for comparison, which is a strong conditional sequence generative model that generates sequence by iterative refinement similar to diffusion models but in discrete tokens space.

#### Observations.

We summarize our findings as follows:

- O1.
DiffusionLM still falls short of conditional sequence learning.
Fig. [1](#S2.F1)(A) shows the validation performance of the two models at different training steps, in which the performance of DiffusionLM still lags behind CMLM by a large margin, even taking many more steps before convergence.
This shows that the performance and training efficiency of the vanilla diffusion-based sequence learner remain unsatisfactory.
- O2.
Diffusion losses at small noise scales are unexpectedly small.
DiffusionLM uniformly samples timesteps hence the corresponding noise scales during training.
As shown in Fig. [1](#S2.F1)(B), we find that the magnitudes of diffusion losses approach almost zero for small noise scales, indicating that it is quite trivial to recover the corrupted embeddings under such circumstances.
We conjecture that, combined with the illustrated example in Fig. [1](#S2.F1)(D), this is because there remain highly discrete modes for the embedding density such that any corrupted embedding is very likely to lie in a region with a small radius around the original token embedding.
As a consequence, the more the modes of embeddings separate from each other the smaller the diffusion loss, which adheres to the following observation.
- O3.
It becomes increasingly harder for the diffusion process to eliminate discreteness while the dimension of the embedding space scales up.
Fig. [1](#S2.F1)(C) shows a surprisingly high accuracy of recovering corrupted embeddings that can be easily achieved by simply seeking the nearest neighbor when embedding dimensions enlarge, even at considerably large noise scales.
This reveals that scaling embedding space leads to more severe discreteness, namely a curse of dimensionality.
- O4.
On condition learning: larger noise scales calibrate diffusion models in taking into account more source conditional information during inference.
We have seen that recovering embeddings corrupted with small noise scales is easy (O2), and if modes distribute separately, even finding the nearest neighbor is enough (O3).
In Fig. [1](#S2.F1)(C), as the noise scale decreases, the prediction accuracy by finding the nearest neighbor increases and achieves almost 100% under a threshold, which can be learned trivially even with little to no source conditions.
This results in the hallucination as shown in Tab. [1](#S3.T1), which is an unexpected consequence for conditional sequence generative models to yield output loyal to the input condition.
To mitigate this, we quantitatively study the influence of noise scales on conditional reliance.
As shown in Fig [1](#S2.F1)(E), we find that as the noise scales are larger, the model can predict more faithfully to source conditions.

**Table 1: Illustration of hallucinations of vanilla DiffusionLM on IWSLT14 De$\rightarrow$En translation task, along with DiNoiSer’s predictions, where the vanilla DiffusionLM generates inexplicable expressions that are irrelevant to the source condition whose meaning dramatically differs from the groud-truth target.**
| Source | Mit welchen worten würden sie ban beschreiben? |
| --- | --- |
| Reference | What are the words you would use to describe ban? |
| DiffusionLM | In which words would you save ban? |
| DiNoiSer | In which words would you describe ban? |

#### Concluding remarks.

We summarize conclusions from the aforementioned observations along with suggestions for more plausible diffused conditional sequence learning:

- C1.
We should not train on too small noise scales to circumvent the pitfall of discreteness.
Both O2 and O4 show the negative influences of small noise scales on training that it leads to a not smooth embedding space with vast regions of low density between modes associated with tokens (O2).
These regions can inevitably be sampled during inference as the result of maximum likelihood estimation(^2^22Consider that a token $\alpha$ is translated into token $A$ or $a$ with 50% each. Without extra information, the optimal prediction for translating $\alpha$ is the center of the embedding of $A$ and $a$. This is because minimizing its training objective (i.e., $\frac{1}{2}\|\bm{z}_{\bm{\theta}}-\bm{z}_{A}(0)\|_{2}^{2}+\frac{1}{2}\|\bm{z}_{\bm{\theta}}-\bm{z}_{a}(0)\|_{2}^{2}$) results in $\bm{z}_{\bm{\theta}}=\frac{\bm{z}_{A}(0)+\bm{z}_{a}(0)}{2}$. The prediction exactly falls in the blank area that lies between embeddings.), thereby giving rise to error accumulation.
Besides, it also impedes conditional learning (O4).
To remedy this for the counter-discreteness purpose, a probably simple way is to eliminate the chance of training with small noise scales.
- C2.
We need to determine the noise schedule according to the dimensionality of the embedding space.
Fitting more complex datasets usually requires larger embedding dimensions.
O3 indicates the criterion to distinguish large and small noise scales depends on the embeddings hence the complexity of the datasets.
However, existing methods employ a fixed noise schedule for all embedding dimensions, which lacks scalability.
This, therefore, demands a task-specific noise schedule to accommodate diverse datasets.
- C3.
We could manipulate the model to be exposed to larger noise scales so as to potentially better leverage source conditions.
O4 suggests that the more corrupted the embeddings, the more difficult for the model to recover, thereby necessitating more reliance on source conditions.
As a result, we could probably encourage trained diffusion models to care more about source conditions for free by post-hoc manipulating the noise to large ones.

## 4 DiNoiSer

Provided the observations and postulates we discussed in §[3](#S3), we accordingly propose DiNoiSer, a simple yet effective method that improves diffusion models by manipulating noises for conditional sequence learning.
In a nutshell, we propose to eliminate the chance of training diffused sequence learners with small-scale noises so as to circumvent the aforementioned pitfall of discreteness in embedding space.
As for sampling, we propose a new effective sampler to amplify the impact of source conditions on the model prediction, where indicators of large noise scales are always fed into the model.
We now dive deep into the details of the proposed DiNoiSer.

Figure: Figure 2: (A) Illustration of the proposed noise scale clipping, where we showcase the Gaussian distributions ${\bm{z}_{t}}/{\alpha(t)}=\mathcal{N}(\textsc{Emb}(y_{i}),\tilde{\sigma}^{2}(t)\bm{I})$, of three corrupted embeddings perturbed with $\sigma(t)$ at timestep $t$ for a token $y_{i}$ and its two (spatially) nearest neighbors $y_{j}$ and $y_{k}$ in the embedding space, respectively. To remedy the pitfall of discreteness, we propose to ensure a sufficiently large minimum “overlap” between corrupted embeddings. As shown in this example, such a goal of counter-discreteness is achieved by bounding the standard deviation $\tilde{\sigma}(t)$ of $\textsc{Emb}(y_{i})$ by $\delta_{ij}$ the “distance” to its nearest neighbor (i.e., $y_{j}$). (B) Comparison between sqrt noise schedule and our noise schedule $\sigma(t)=t$ manipulated with the proposed noise scale clipping. The adaptive clipping threshold $\sigma_{\min}$ at several training steps is presented as it varies during training. The clipping threshold $\sigma_{\min}$ starts with a large number at around $0.7$ and stays steady later maintaining at around $0.5$. The sqrt noise schedule, on the other hand, increases rapidly at small timesteps, exceeding our clipping threshold at the start of training (i.e., $\sigma(t\approx 0.2)$) and later stage of training steps (i.e., $\sigma(t\approx 0.1)$), respectively.
Refer to caption: /html/2302.10025/assets/figs/noise_clipping.png

### 4.1 Noise Scale Clipping: Counter-Discreteness Training with Manipulated Noises

Recall that C1 and C2 in §[3](#S3) demonstrate that small noises can barely help “discrete” embeddings populate the entire continuous space, and also undermine conditional learning.
A simple yet effective way to mitigate this is to encourage training diffusion models with sufficiently large noise scales.
As such, we propose noise scale clipping, where we bound the minimum noise scale $\sigma_{\min}$ for training such that only timesteps satisfying $\sigma(t)\geq\sigma_{\min}$ could be sampled, which is decided adaptively as the model learn progresses.

To start with, we can eliminate the scaling effect of $\alpha(t)$ in the forward diffusion process for the embedding of each token $y_{i}$ by rewriting Eqn. [10](#S2.E10) into(^3^33Here we let $\bm{z}_{t}$ and $\bm{z}_{0}$ denote $\bm{z}_{t}(y_{i})$ and $\bm{z}_{0}(y_{i})$ for brevity.):

$$ $\displaystyle\frac{\bm{z}_{t}}{\alpha(t)}=\bm{z}_{0}$ $\displaystyle+\frac{\sigma(t)}{\alpha(t)}\bm{\epsilon}_{t}\Rightarrow\frac{\bm{z}_{t}}{\alpha(t)}\sim\mathcal{N}\biggl{(}\bm{z}_{0},\frac{\sigma^{2}(t)}{1-\sigma^{2}(t)}\bm{I}\biggr{)}$ (11) $\displaystyle\Rightarrow\frac{\bm{z}_{t}}{\alpha(t)}\sim\mathcal{N}\biggl{(}\textsc{Emb}(y_{i}),\tilde{\sigma}^{2}(t)\bm{I}\biggr{)}$ $$

As illustrated in Fig. [2](#S4.F2)(A), there, intuitively, should exist a sufficiently large number $\delta$ measuring the minimum “overlap” between the distributions of two corrupted embeddings under the Gaussian perturbation kernel with a standard deviation of $\tilde{\sigma}(t)=\frac{\sigma(t)}{\sqrt{1-\sigma^{2}(t)}}$.
To this end, we let $\delta^{2}$ be the minimum amount of variation of added noise, defined as the average squared L2-distances between the embeddings and their nearest neighbor, normalized by the dimension of embeddings (according to C2 in §[3](#S3)):

$$ $\displaystyle\delta^{2}$ $\displaystyle=\frac{1}{{|\mathcal{V}|\cdot D}}\sum_{i=1}^{|\mathcal{V}|}\min_{1\leq j\not=i\leq|\mathcal{V}|}\delta^{2}_{ij}$ (12) $\displaystyle=\frac{1}{{|\mathcal{V}|\cdot D}}\sum_{i=1}^{|\mathcal{V}|}\min_{1\leq j\not=i\leq|\mathcal{V}|}\|\textsc{Emb}(y_{i})-\textsc{Emb}(y_{j})\|_{2}^{2}.$ $$

We now define the noise scale clipping as follows:

###### Definition 4.1 (The noise scale clipping) .

Let $\mathcal{V}$ be the target vocabulary with corresponding embeddings in $D$-dimensional space $\forall y_{i}\in\mathcal{V}:\textsc{Emb}(y_{i})\in\mathbb{R}^{D}$, the noise scale ciipping is performed if the noise scale $\sigma(t)$ at timestep $t$ satisfies the following condition:

$$ $\small\tilde{\sigma}^{2}(t)=\frac{\sigma^{2}(t)}{1-\sigma^{2}(t)}\geq\delta^{2}~{}\Rightarrow~{}\frac{\sigma^{2}_{\min}}{1-\sigma^{2}_{\min}}=\delta^{2},$ (13) $$

the clipping threshold $\sigma_{\min}$ is whereby derived when the equality in Eqn. [13](#S4.E13) holds, such that

$$ $\small\!\!\!\!\sigma_{\min}\!=\!\Biggl{(}\frac{{|\mathcal{V}|}\cdot D}{\sum_{i=1}^{|\mathcal{V}|}\mathop{\min}\limits_{1\leq j\not=i\leq|\mathcal{V}|}\|\textsc{Emb}(y_{i})-\textsc{Emb}(y_{j})\|_{2}^{2}}+1\!\Biggr{)}^{\!\!-\frac{1}{2}}\!\!\!\!\!\!,$ (14) $$

obtained by substituting Eqn. [12](#S4.E12) into the R.H.S of Eqn. [13](#S4.E13).

As illustrated in Fig. [2](#S4.F2)(B), the clipping threshold $\sigma_{\min}$ is estimated dynamically during training, depending on how properly the model learns the embeddings up to the minimum pair-wise distances within the vocabulary.
In each training step, we first estimate the clipping threshold $\sigma_{\min}$ with Eqn. [14](#S4.E14), then sample timesteps among $t$ that satisfies $\sigma(t)>\sigma_{\min}$.
In practice, one can first estimate the noise scale threshold $\sigma_{\min}$ and then turn it into the timestep threshold $t_{\min}=\sigma^{-1}(\sigma_{\min})$ in general.
In this work, we select $\sigma(t)=t$ as the noise scheduler to simplify this procedure(^4^44This can be done since the effects of different noise schedules are theoretically interchangeable up to different weight factors under the simplified training objective we adopted (see Appendix [A](#A1)). We also provide empirical comparisons between different schedules in Tab. [4](#S5.T4). This allows us to more conveniently manipulate noise scales.).

Figure: Algorithm 1 Training with noise scale clipping

As a result, the updated diffusion loss with an enlarged minimum timestep threshold (thus an increased minimum noise scale) in the final training objective (modified from Eqn. [9](#S2.E9)) now becomes:

$$ $\displaystyle\mathcal{L}^{\prime}_{\text{diffusion}}(\bm{y})=\!\!\!\!\!\!\mathop{\mathbb{E}}_{t\sim\mathcal{U}(t_{\min},1),\bm{\epsilon}_{t}\sim\mathcal{N}(\bm{0},\bm{I})}\Big{[}\|\bm{z_{\theta}}(\bm{z}_{t},\bm{x},t)-\bm{z}_{0}\|_{2}^{2}\Big{]}.$ $$

We provide pseudocodes regarding how to manipulate noises in training as such in Alg. [1](#alg1).

Figure: Figure 3: A synthesis experiment similar to Fig. [1](#S2.F1)(E), where the model is asked to predict with current timestep $\tau=t$ and an alternative larger timestep $\tau=0.995$, respectively. We compare the MSE between the model prediction $\bm{z}_{\bm{\theta}}(\bm{z}_{t}^{\prime},\bm{x},\tau)$ to the embedding of ground-truth $\bm{z}_{0}(\bm{y})$ (top) and negative sample $\bm{z}_{0}(\bm{y}^{\prime})$ (bottom) respectively, and study to which target the model prediction assimilates, the original $\bm{y}$ or the negative one $\bm{y}^{\prime}$, hence should most likely be governed by the source or the target information.
Refer to caption: /html/2302.10025/assets/figs/tendency_Helvetica_x.png

###### Definition 4.1 (The noise scale clipping) .

### 4.2 CeDi : Sampling with Manipulated Noises for Enhanced Condition Awareness

Based on C3 in §[O4.](#S3.I1.ix4), we suppose the model relies more on the source conditions when indicated by large noise scales.
This implies that we may make the model more faithful to source conditions by feeding indicators of large noise scales to the model.
Fig. [3](#S4.F3) shows a synthesis experiment similar to Fig. [1](#S2.F1)(E), wherein the predictions using a large timestep $0.995$ (namely, a larger noise scale) are closer to the embedding of the original target $\bm{y}$, while more distant to that of the misleading $\bm{y}^{\prime}$, reiterating that the model relies more on the source condition $\bm{x}$ when exposed to large noise indicator due to manipulation in inference.

Inspired by the aforementioned observations, we propose a condition-enhanced denoiser (CeDi) for sampling.
CeDi always feeds a large $t$ to the model $\bm{z_{\theta}}$ to encourage the model to make use of the source condition.
In practice, we largely follow the framework of DDIM solver  but pick two sets of timesteps.
In the first set $\{t_{i}\}_{i=0}^{M}$, timesteps decrease uniformly from $t_{0}=1$ to $t_{M}\approx 0$ as normal.
As for the other set $\{\tau_{i}\}_{i=0}^{M}$, $\tau_{i}$s decrease uniformly from $\tau_{0}=1$ to a large time $\tau_{M}\gg 0$(^5^55Empirically, we find that $\tau_{M}$ satisfying $\sigma(\tau_{M})=0.99$ (i.e., $\tau_{M}=0.99$ for $\sigma(t)=t$ and $\tau_{M}=0.9606$ for ’s sqrt schedule $\sigma(t)=t^{0.25}$) generally works well.).
When making predictions, we assign timesteps from the second set to the model.
By replacing corresponding timesteps in the framework of DDIM (Eqn. [7](#S2.E7) and Eqn. [8](#S2.E8)) with $\tau_{i}$s, we generate our predictions by iteratively computing

$$ $\hat{\bm{z}}_{t_{i}}=\alpha(t_{i})\bm{z}_{\bm{\theta}}(\hat{\bm{z}}_{t_{i-1}},\bm{x},{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\tau_{i-1}})+\sigma(t_{i})\bm{\epsilon}_{\bm{\theta}}(\hat{\bm{z}}_{t_{i-1}},\bm{x},{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\tau_{i-1}}),$ $$

where the predicted noise is also updated as

$$ $\bm{\epsilon}_{\bm{\theta}}(\hat{\bm{z}}_{t_{i-1}},\bm{x},{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\tau_{i-1}})=\frac{\hat{\bm{z}}_{t_{i-1}}-\alpha({\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\tau_{i-1}})\bm{z}_{\bm{\theta}}(\hat{\bm{z}}_{t_{i-1}},\bm{x},{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\tau_{i-1}})}{\sigma({\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\tau_{i-1}})}.$ $$

We also demonstrate how CeDi works in Alg. [2](#alg2).

Figure: Algorithm 2 Sampling with CeDi for enhanced condition awareness

## 5 Experiment

We conduct experiments to verify the effectiveness of our proposed method and study its characteristics.

### 5.1 Experimental Setup

Tasks and Datasets. We mainly experiment on machine translation, a well-established benchmark task for conditional sequence learning. We consider IWSLT14 De$\leftrightarrow$En (160K pairs), WMT14 En$\leftrightarrow$De (4.0M pairs), and WMT14 En$\leftrightarrow$Ro (610K pairs), six machine translation tasks with variant sizes of training data.
Additionally, we experiment on two of the datasets introduced by DiffuSeq , including Wiki  for text simplification and QQP(^6^66[https://www.kaggle.com/c/quora-question-pairs](https://www.kaggle.com/c/quora-question-pairs)) for paraphrasing.

**Table 2: Comparison in SacreBLEU on machine translation tasks. “LB”: the size of the length beam search. “MBR”: the number of candidates for each length beam to apply Minimum Bayes-Risk decoding. “KD”: results are obtained with knowledge distillation . Provided that KD is common and effective practice in non-autoregressive (NAR) machine translation, though not the focus of this study, we also provide further experiments with KD in Appendix [B](#A2) for reference. The best NAR results without KD are in bold and the second best ones are underlined. The results of CDCD are quoted from , and the results of Difformer, DiffuSeq, and SeqDiffuSeq are from . ${\dagger}$: how CMLM originally selects candidates with different lengths differs from the MBR decoding we used for diffusion models, and we thus include its results with MBR decoding for fair comparisons. ${\ddagger}$: results are presented in tokenized BLEU as reported in , and we encourage readers to check the original papers for more details.**
| Methods | IWSLT14 | WMT14 | WMT16 |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| De$\rightarrow$En | En$\rightarrow$De | De$\rightarrow$En | En$\rightarrow$De | Ro$\rightarrow$En | En$\rightarrow$Ro |  |
| Transformer   (AR, beam $=5$) | 33.61 | 28.30 | 30.55 | 26.85 | 33.08 | 32.86 |
| CMLM   (NAR, LB $=5$) | 29.41 | 24.33 | 28.71 | 23.22 | 31.13 | 31.26 |
| CMLM   (NAR, LB $=5$, MBR=1^†) | 29.32 | 24.34 | 28.43 | 23.09 | 31.07 | 30.92 |
| DiffusionLM   (LB $=5$, MBR $=1$) | 26.61 | 20.29 | 17.31 | 15.33 | 28.61 | 27.01 |
| DiffusionLM   (LB $=5$, MBR $=10$) | 29.11 | 22.91 | 19.69 | 17.41 | 30.17 | 29.39 |
| CDCD   (MBR $=10$) | - | - | 25.40 | 19.70 | - | - |
| CDCD   (MBR $=100$) | - | - | 26.00 | 20.00 | - | - |
| Difformer      (LB$\times$MBR $=20$) | - | - | - | 23.80 | - | - |
| DiffuSeq^‡        (KD, LB$\times$MBR $=10$) | - | - | - | 15.37 | - | 25.45 |
| SeqDiffuSeq^‡   (KD, LB$\times$MBR $=10$) | - | - | - | 17.14 | - | 26.17 |
| DiNoiSer  (LB $=5$, MBR $=1$) | 31.29 | 25.55 | 28.83 | 24.25 | 31.14 | 30.93 |
| DiNoiSer  (LB $=5$, MBR $=10$) | 31.61 | 25.70 | 29.05 | 24.26 | 31.22 | 31.08 |
| DiNoiSer  (LB $=10$, MBR $=5$) | 31.44 | 26.14 | 29.01 | 24.62 | 31.24 | 31.03 |
| DiNoiSer  (KD, LB $=10$, MBR $=5$) | - | - | 30.30 | 25.88 | 33.13 | 32.84 |

Baselines. We include three groups of baselines for machine translation:
(1) The autoregressive Transformer ;
(2) The CMLM , an iterative-based non-autoregressive model for conditional sequence learning.
(3) Previous diffusion-based sequence generative models, including the vanilla design that simply extends the original DiffusionLM  with an additinal condition encoder, and the other recently proposed improved methods CDCD , DiffuSeq , SeqDiffuSeq  and Difformer .
For text simplification and paraphrasing, we compare our method with DiffuSeq .

Metrics. We primarily report SacreBLEU(^7^77The signature is nrefs:1|case:mixed|eff:no| tok:intl|smooth:exp|version:2.3.1 if the target language is German, and nrefs:1|case:mixed|eff:no| tok:13a|smooth:exp|version:2.3.1 for others.)  for machine translation, following CDCD .
We also report tokenized BLEU  in Appendix [B](#A2) for reference.
For text simplification and paraphrasing, we follow DiffuSeq to employ sentence-level BLEU under the tokenizer of BERT-BASE-UNCASED as the evaluation metric.

Implementation Details.
All implementations are based on Transforme-base  for all datasets except IWSLT14.
For IWSLT14, we use a smaller architecture that has 4 attention heads and 1024-dimensional feedforward layers.
The embedding dimension for the diffusion model is 16 on IWSLT14 and 64 on the others.
In the implementation of our method, we follow recent advances and apply self-conditioning techniques . Besides, following previous practice in non-autoregressive machine translation, we train our model both with and without knowledge distillation .

During inference, for machine translation, we apply beam search in the autoregressive Transformer with beam size 5.
Correspondingly, we use length beam 5 in the non-autoregressive models, except for CDCD and DiffuSeq since they vary the target lengths by predicting paddings instead of length predictions.
For text simplification and paraphrasing, we report results with various length beams as length prediction on these tasks is more challenging and less studied.
For all the diffusion-based methods, we follow previous work  and apply Minimum Bayes-Risk (MBR) decoding .
For both DiffusionLM and our model, we perform sampling with 20 steps.

We implement DiffusionLM and DiNoiSer upon fairseq , and also train Transformer and CMLM baselines using fairseq.
For data preprocessing, we follow the instruction in fairseq for IWSLT14(^8^88[https://github.com/facebookresearch/fairseq/tree/main/examples/translation](https://github.com/facebookresearch/fairseq/tree/main/examples/translation)) and use the preprocessed data by  for WMT14 and WMT16(^9^99[https://github.com/shawnkx/Fully-NAT](https://github.com/shawnkx/Fully-NAT)).
For Wiki and QQP, we use preprocessed data provided by DiffuSeq(^10^1010[https://github.com/Shark-NLP/DiffuSeq](https://github.com/Shark-NLP/DiffuSeq)) and tokenized them with byte-pair encoding .
The training batch size is 128K for WMT14/WMT16, and 32K for the others.
We empirically find checkpoint averaging unnecessary for our method and have not applied it in all our implementations.

### 5.2 Main Results

The results of machine translation and the other two tasks are in Tab. [2](#S5.T2) and Tab. [3](#S5.T3), respectively.

Overall performance.
Our DiNoiSer demonstrates effectiveness on all selected conditional sequence learning tasks,
which we summarize into the following three aspects:

- •
DiNoiSer achieves state-of-the-art results among diffusion-based models on one of the representative conditional sequence generation tasks, i.e., machine translation, where DiNoiSer outperforms the vanilla design of DiffusionLM, as well as the previous strongest approaches such as CDCD  and Difformer  by a large margin (Tab. [2](#S5.T2)).
For DiffuSeq and SeqDiffuSeq, although their reported tokenized BLEUs are not strictly comparable to our SacreBLEU results due to the difference in tokenizers, our performance is far above them by over 4 BLEU score and even more if we involve knowledge distillation, which supports our superiority over them.
- •
DiNoiSer demonstrates strong competitiveness in conditional sequence learning.
It even surpasses CMLM  on almost all the experimented machine translation datasets (Tab. [2](#S5.T2)).
Provided that CMLM is one of the leading approaches among NAR sequence learners, the performance DiNoiSer achieves can be considered quite competitive.
- •
DiNoiSer is generic to various conditional sequence learning tasks. Results on Tab. [3](#S5.T3) shows that DiNoiSer also works well in tasks other than machine translation, surpassing previously proposed DiffuSeq.

**Table 3: Sentence-level BLEU of our method and DiffuSeq on Wiki (text simplification) and QQP (paraphrasing). “NFE”: number of function evaluations, measuring the total number of forward passes to the model for each prediction. The results of DiffuSeq are quoted from .**
| Methods | Steps | LB | MBR | NFE | Wiki | QQP |
| --- | --- | --- | --- | --- | --- | --- |
| DiffuSeq | 2000 | - | 10 | 20000 | 36.22 | 24.13 |
| DiNoiSer | 20 | 10 | 1 | 200 | 35.36 | 26.07 |
| DiNoiSer | 20 | 20 | 1 | 400 | 36.94 | 25.42 |
| DiNoiSer | 20 | 20 | 5 | 2000 | 36.88 | 25.57 |

In addition to the overall performance, DiNoiSer also demonstrates several nice properties over the baselines.
We elaborate on them as follows:

Scalability.
As shown in Tab. [2](#S5.T2), DiffusionLM seems more challenging to accommodate larger datasets like WMT14 than smaller ones (e.g., IWSLT14).
This verifies the curse of scalability problem discussed in §[3](#S3).
In contrast, DiNoiSer surpasses CMLM on almost all large- and small-scale scenarios, which indicates that DiNoiSer indeed greatly improves the scalability of diffusion-based sequence learners.
This advantage of DiNoiSer could help facilitate further research and open up the possibilities of large-scale real-world applications of diffused sequence generative models.

Sampling efficiency.
Given the sizes of length beam search (LB) and MBR decoding shown in Tab. [2](#S5.T2), DiNoiSer surpasses or closely approaches CMLM even when MBR=$1$, while the vanilla DiffusionLM heavily relies on a large number of candidates used for MBR decoding.
Besides, DiNoiSer necessitates much fewer NFEs to achieve strong performance, e.g., only 20 steps, resulting in only 1% to 10% computational consumption and latency compared to previous works .
This manifests that DiNoiSer is more accurate yet efficient compared to previous diffusion-based sequence learning models.

**Table 4: Ablation Study on WMT14 En$\rightarrow$De. All the results are in SacreBLEU scores and the length beam sizes are 5.**
| Training Settings | DDIM (MBR $=1$) | CeDi (MBR $=1$) | DDIM (MBR $=10$) | CeDi (MBR $=10$) |
| --- | --- | --- | --- | --- |
| Ours [final] | 19.23 | 24.25 | 22.12 | 24.26 |
| w/o self-conditioning | 20.37 | 23.03 | 22.58 | 23.14 |
| w/o noise scale clipping | 7.95 | 21.16 | 11.51 | 21.40 |
| w/o self-conditioning, w/o noise scale clipping | 11.30 | 20.86 | 14.84 | 21.47 |
| w/ sqrt noise schedule | 20.11 | 24.13 | 22.83 | 24.07 |
| w/ sqrt noise schedule, w/o noise scale clipping | 16.68 | 23.22 | 20.46 | 23.40 |

### 5.3 Effect of Sizes of Length Beam and MBR

DiNoiSer can leverage both length beam search and MBR decoding to produce diverse candidates for selection.
The results on all of the evaluated datasets (Tab. [2](#S5.T2) and Tab. [3](#S5.T3)) demonstrate that the method can gain its performance by properly adjusting the two hyperparameters for sampling.
In particular, we search on various combinations of length beams and MBRs and evaluate the corresponding performance of DiNoiSer on the validation set of WMT14 En$\rightarrow$De, shown in Fig. [4](#S5.F4).
The model performance rises first and then drops down as the length beam increases.
And for each length beam, we can further boost the performance of DiNoiSer with MBR $>1$, suggesting that the effects of the two factors are complementary.

Using both length beam search and MBR decoding also brings benefits to DiNoiSer over those only involving one of them.
Compared to CMLM which decodes deterministically, DiNoiSer is able to sample multiple sentences for each length beam, providing more diverse candidates.
Compared to CDCD, which predicts paddings to generate sentences of various lengths and whose sampling efficiency is restricted by maximum target length, DiNoiSer’s use of length beams allows more fine-grained control of the computational budget.

### 5.4 Effect of Noise Scale Clipping for Training

Figure: Figure 4: SacreBLEU on the validation set of WMT14 En$\rightarrow$De with different length beams and MBR sizes.
Refer to caption: /html/2302.10025/assets/figs/beam_mbr.png

We compare models trained with different settings in Tab. [4](#S5.T4) to study the effect of our training strategy.
We find that the proposed noise scale clipping consistently helps improve the model performance.
Replacing the noise schedule in DiNoiSer (i.e., $\sigma(t)=t)$ with the sqrt schedule proposed by  has negligible influence on the final performance.
This is expected since we only set the noise schedule as $\sigma(t)=t$ for convenience.
The difference is that the improvement by noise clipping is relatively smaller when using the sqrt schedule.
This is because the noise scale of the sqrt schedule increases quickly in small timesteps (Fig. [2](#S4.F2)B). When $t=0.2$, $\sigma_{sqrt}(t)\approx 0.67$, which is close to our initial clipping threshold.
This suggests the previous success of the sqrt schedule may also be partly explained as training more on large-scale noises.

### 5.5 Effect of Condition Enhancement for Sampling

We compare the performance between different denoisers, i.e., DDIM and the proposed CeDi, in Tab. [4](#S5.T4).
In a nutshell, CeDi impressively outperforms DDIM, especially for small MBR candidate sizes.
We also notice that DDIM performs particularly unsatisfactorily when the model is trained without noise scale clipping.
However, for these models, CeDi can still produce a relatively good performance (over 20.50 for MBR=1) that even surpasses well-designed CDCD (20.00 for MBR=100).
What’s more, we highlight two critical characteristics of CeDi as follows:

#### CeDi indeed better leverage source conditions for inference.

Recall that we propose the CeDi with the purpose of encouraging the model to make better use of source conditions for prediction (§[4.2](#S4.SS2)).
To investigate whether the denoiser achieves this, we apply Layer-wise Relevant Propagation  to measure the relative contribution of the source condition to the model prediction.
As shown in Fig. [5](#S5.F5)(B), we compare the source contribution of our CeDi and the DDIM solver along the sampling iterations.
CeDi maintains a high source contribution, while the source contribution of CeDi is unsatisfactory in the first few steps, which demonstrates that sampling with our CeDi does leverage the source condition more.
Correspondingly, as shown in Fig. [5](#S5.F5)(A), the prediction accuracy of CeDi increases steadily, while the performance of DDIM fails to improve at the beginning of the iterations, suggesting correlations between higher source contribution and higher performance improvement.
Among all iterations, the first few steps establish the foundation for the overall performance.
Although DDIM improves its performance in later iterations, it still falls behind our CeDi.
This suggests the effectiveness of increasing the source contribution, especially at the beginning of the sampling process.

Figure: Figure 5: The difference between our CeDi and the DDIM solver over steps. (A) The prediction accuracy at each step, measured with SacreBLEU. (B) The proportion of source contribution to the prediction at each step.
Refer to caption: /html/2302.10025/assets/figs/cedi.png

#### CeDi can handle sequence generation from complex and multiple conditions.

To further show the strength of CeDi in capturing source conditions, we explore more complex conditional sequence learning scenarios.
We simulate this under two multilingual translation settings, i.e. many-to-one and one-to-many translation.
In the many-to-one scenario, a unified model needs to translate source sentences in multiple different source languages to English counterparts, requiring the model to handle complicated source conditions.
On the other hand, the one-to-many setting simulates a multi-conditional scenario, requiring the model to recognize the target language as another crucial condition to capture the target distribution.
To this end, we construct a dataset by combining four language pairs of
IWSLT14 translation benchmark, i.e., En$\leftrightarrow$De, En$\leftrightarrow$Ro, En$\leftrightarrow$Nl, and En$\leftrightarrow$Pt-br.
In the one-to-many translation, we append language tokens to the source sequences to incorporate the target language as a condition.
We also include a baseline in which the models are trained separately for each language pair for comparison.

As shown in Tab. [5](#S5.T5), DiNoiSer works well in multilingual settings, showing its strong capability in modeling conditions, i.e. a complex multimodal condition (source sentences of four languages in many-to-one), and multiple conditions (source sentence of English as well as identity of target languages).
Particularly, CeDi shows huge advantages over DDIM in the multilingual setting of one-to-many translation.
In this case, the language accuracy of DDIM is much lower than that of CeDi, suggesting DDIM has trouble capturing the given condition, namely the language identity in this one-to-many scenario.
In contrast, DiNoiSer augmented with CeDi yields satisfactory predictions with high language accuracy, exhibiting superiority in working with multiple conditions.

**Table 5: Results of multilingual machine translation ({De,Ro,Pt-br,Nl}$\leftrightarrow$En). “Bilingual”: integrated results of separate models of every language pair. “Multiling.”: results from a unified multilingual model. We employ langdetect to infer the language of generated sequences for computing the language accuracy.**
| Settings | Methods | SacreBLEU (Lang Acc %) |  |
| --- | --- | --- | --- |
| Bilingual | Multiling. |  |  |
| many-to-one<br>{De,Ro,Pt-br,Nl}<br>$\to$En | CMLM | 33.85 | 35.23 |
| DiNoiSer (DDIM, MBR=1) | 32.40 | 33.43 |  |
| DiNoiSer (DDIM, MBR=10) | 33.73 | 35.48 |  |
| DiNoiSer (MBR=1) | 34.57 | 35.26 |  |
| DiNoiSer (MBR=10) | 34.74 | 35.66 |  |
| one-to-many<br>En$\to$<br>{De,Ro,Pt-br,Nl} | CMLM | 28.10 | 30.55 (72.40) |
| DiNoiSer (DDIM, MBR=1) | 27.44 | 17.95 (67.58) |  |
| DiNoiSer (DDIM, MBR=10) | 28.54 | 18.57 (67.50) |  |
| DiNoiSer (MBR=1) | 28.72 | 30.52 (72.81) |  |
| DiNoiSer (MBR=10) | 28.81 | 30.67 (72.88) |  |

## 6 Related Work

### 6.1 Non-autoregressive Sequence Generative Models

Non-autoregressive sequence learning (NAR) was first proposed by  as an alternative to its autoregressive counterpart.
It generates target tokens in parallel, either fully NAR  or up to a mild number of iterations , liberating sequence modeling from the constraint of a predefined order .
With recent efforts, NAR shows great potential in the applications of various domains, including language , speech , proteins , and molecules .
Different from more commonly-used autoregressive (AR) models , NAR models assume conditional independence among the output tokens.
Such an assumption risks ignoring the target dependencies  and leads to the multi-modality problem .
As a result, the vanilla fully NAR model has inferior generation quality.
Some of the later improvements alleviate the strong assumption by reformulating NAR formulation under iterative refinement , which iteratively takes as input the previously generated sequence, which serves as an intermediate random variable, to produce the tokens of its refined or denoised version in parallel until convergence or the budget of maximum iterations run out.
Some recent advances herein follow the idea of discrete diffusion and formalize iterative refinement as Markov processes .
Although both are named after diffusion models, these works operate on discrete state space, whereas our focus, continuous diffusion models accommodate the continuous (embedding) space of discrete tokens.

### 6.2 Diffusion Models for Sequence Learning

Continuous diffusion models  gained first success in generating high-quality images.
Recently, successfully adapted them to sequence learning and proposed the DiffusionLM, the first diffusion-based sequence generative model with a special focus on controllable text generation.
Later improvements to the diffusion-based sequence generative models are mainly categorized threefold.
The first line includes novel components for diffusion modeling, such as the partial diffusion process proposed by , self-conditioning techniques introduced by , and the adaptive noise schedule of .
The second line applies diffusion models to the latent space of specific pretrained language models .
And the third tries to incorporate conventional practice in discrete token prediction. For instance, and incorporate the cross-entropy objectives in training.
For the application of diffusion-based models for sequence learning, previous work found their advantages in controllable generation , and generating diverse sequences .
demonstrates that diffusion-based sequence generative models can benefit from large-scale self-supervised pretraining.
While almost all these works mainly focus on the training phrase of diffusion-based sequence generative models, our study emphasizes both training and inference.

## 7 Conclusion

In this paper, we shed light on the crucial role of noise scales in conditional sequence learning with diffusion models and determine that small noise scales impede diffusion models in both learning and inference.
Motivated by our findings, we propose DiNoiSer to leverage large noise scales in both training and inference.
DiNoiSer makes training samples cover the whole sample space more efficiently and also enables the model to better utilize source conditions for prediction, thereby leading to considerable performance improvements.
In this work, we demonstrate the potential diffusion models for conditional sequence learning.
We expect our study could help facilitate further research on diffusion models to empower various applications of conditional sequence learning.

## Appendix A Relationship Between Different Noise Schedules and Time Samplers

Generally, the training objective of diffusion models can be expressed as

$$ $\mathbb{E}_{t\sim r(t),\bm{\epsilon}\sim\mathcal{N}(\bm{0},\bm{I})}[w(t)\|\bm{z}_{\bm{\theta}}-\bm{z}(0)\|_{2}^{2}],$ $$

where $\bm{z}_{\bm{\theta}}$ is the model prediction $\bm{z}_{\bm{\theta}}(\sqrt{1-\sigma^{2}(t)}\bm{z}(0)+\sigma(t)\bm{\epsilon},t)$ for short.

The above expectation over timesteps can be rewritten into the expectation of noise scales as follows.

$$ $\displaystyle\mathbb{E}_{t\sim r(t),\bm{\epsilon}}[w(t)\|\bm{z}_{\bm{\theta}}-\bm{z}(0)\|_{2}^{2}]$ $\displaystyle=$ $\displaystyle\mathbb{E}_{\bm{\epsilon}}[\int_{0}^{1} r(t)w(t)\|\bm{z}_{\bm{\theta}}-\bm{z}(0)\|_{2}^{2}\mathrm{d}t]$ $\displaystyle=$ $\displaystyle\mathbb{E}_{\bm{\epsilon}}[\int_{0}^{1}\hat{r}(\sigma)\hat{w}(\sigma)\|\bm{z}_{\bm{\theta}}-\bm{z}(0)\|_{2}^{2}]\frac{\mathrm{d}t}{\mathrm{d}\sigma}\mathrm{d}\sigma$ $\displaystyle=$ $\displaystyle\mathbb{E}_{\sigma\sim U(0,1),\bm{\epsilon}}[w^{\prime}(\sigma)\|\bm{z}_{\bm{\theta}}-\bm{z}(0)\|_{2}^{2}],$ $$

where $w^{\prime}(\sigma)=w(\sigma^{-1})r(\sigma^{-1})\frac{\mathrm{d}t}{\mathrm{d}\sigma}$. Therefore, training with different noise schedules and different time samplers is interchangeable by applying different weighting functions.

## Appendix B More Results on Machine Translation

In order to provide references for further study and comparisons, we report more results on machine translation.

#### Knowledge distillation (KD).

A common practice to improve the performance of non-autoregressive machine translation is knowledge distillation .
We report the performance of our method trained on distilled data of WMT14 and WMT16 on Tab. [6](#A2.T6).
The result shows that the performance gap between our method and the autoregressive transformer is small when knowledge distillation is used.
This suggests that our method is able to achieve performance that satisfies the need of applications.

**Table 6: Model performances on machine translation with knowledge distillation. The results of transformer are from raw data, while DiNoiSer is trained on distilled data. The performances are measured with SacreBLEU.**
|  | WMT14 | WMT16 |  |  |
| --- | --- | --- | --- | --- |
| Methods | De$\rightarrow$En | En$\rightarrow$De | Ro$\rightarrow$En | En$\rightarrow$Ro |
| Transformer | 30.55 | 26.85 | 33.08 | 32.86 |
| DiNoiSer (LB=5, MBR=1) | 30.13 | 25.70 | 32.96 | 32.58 |
| DiNoiSer (LB=5, MBR=10) | 30.12 | 25.90 | 33.04 | 32.57 |
| DiNoiSer (LB=10, MBR=5) | 30.30 | 25.88 | 33.13 | 32.84 |

#### Evaluation with tokenized BLEU.

Some of the previous studies in machine translation reported tokenized BLEU, despite inconsistent tokenizers (other than the standard Moses tokenizer) they might use.
To help conveniently compare DiNoiSer to them, we also report the performance of DiNoiSer with tokenized BLEU in Tab. [12](#footnote12).

**Table 7: Tokenized BLEU of our method on machine translation datasets. We use the moses tokenizer(^12^1212[https://github.com/alvations/sacremoses](https://github.com/alvations/sacremoses)) for all the texts. “LB”: the size of length beam. “MBR”: the number of candidates for each length beam to apply Minimum Bayes-Risk decoding. +KD means the results are obtained with knowledge distillation.**
|  | IWSLT14 | WMT14 | WMT16 |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Methods | De$\rightarrow$En | En$\rightarrow$De | De$\rightarrow$En | En$\rightarrow$De | Ro$\rightarrow$En | En$\rightarrow$Ro |
| DiNoiSer (LB=5, MBR=1) | 32.23 | 25.54 | 29.35 | 24.43 | 31.21 | 31.18 |
| DiNoiSer (LB=5, MBR=10) | 32.48 | 25.68 | 29.53 | 24.45 | 31.39 | 31.29 |
| DiNoiSer (LB=10, MBR=5) | 32.25 | 25.99 | 29.40 | 24.48 | 31.50 | 31.27 |
| DiNoiSer + KD (LB=5, MBR=1) | - | - | 30.64 | 26.08 | 33.21 | 32.57 |
| DiNoiSer + KD (LB=5, MBR=10) | - | - | 30.62 | 26.29 | 33.29 | 32.59 |
| DiNoiSer + KD (LB=10, MBR=5) | - | - | 30.76 | 26.04 | 33.40 | 32.89 |