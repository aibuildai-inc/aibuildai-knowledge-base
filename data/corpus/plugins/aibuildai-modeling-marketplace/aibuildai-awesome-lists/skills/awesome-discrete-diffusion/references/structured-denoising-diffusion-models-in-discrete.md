---
arxiv_id: "2107.03006"
title: "Structured Denoising Diffusion Models in Discrete State-Spaces"
year: 2021
source: arxiv2md
---

## Abstract

Abstract Denoising diffusion probabilistic models (DDPMs) [ 19 ] have shown impressive results on image and waveform generation in continuous state spaces. Here, we introduce Discrete Denoising Diffusion Probabilistic Models (D3PMs), diffusion-like generative models for discrete data that generalize the multinomial diffusion model of Hoogeboom et al. [ 20 ] , by going beyond corruption processes with uniform transition probabilities. This includes corruption with transition matrices that mimic Gaussian kernels in continuous space, matrices based on nearest neighbors in embedding space, and matrices that introduce absorbing states. The third allows us to draw a connection between diffusion models and autoregressive and mask-based generative models. We show that the choice of transition matrix is an important design decision that leads to improved results in image and text domains. We also introduce a new loss function that combines the variational lower bound with an auxiliary cross entropy loss. For text, this model class achieves strong results on character-level text generation while scaling to large vocabularies on LM1B. On the image dataset CIFAR-10, our models approach the sample quality and exceed the log-likelihood of the continuous-space DDPM model.

## 1 Introduction

Generative modeling is a core problem in machine learning, useful both for benchmarking our ability to capture statistics of natural datasets and for downstream applications that require generating high-dimensional data like images, text, and speech waveforms.
There has been a great deal of progress with the development of methods like GANs , VAEs , large autoregressive neural network models , normalizing flows , and others, each with their own tradeoffs in terms of sample quality, sampling speed, log-likelihoods, and training stability.

Recently, diffusion models have emerged as a compelling alternative for image  and audio  generation, achieving comparable sample quality to GANs and log-likelihoods comparable to autoregressive models with fewer inference steps.
A diffusion model is a parameterized Markov chain trained to reverse a predefined forward process, which is a stochastic process constructed to gradually corrupt training data into pure noise.
Diffusion models are trained using a stable objective closely related to both maximum likelihood and score matching , and they admit faster sampling than autoregressive models by using parallel iterative refinement .

Although diffusion models have been proposed in both discrete and continuous state spaces , most recent work has focused on Gaussian diffusion processes that operate in continuous state spaces (e.g. for real-valued image and waveform data). Diffusion models with discrete state spaces have been explored for text and image segmentation domains , but they have not yet been demonstrated as a competitive model class for large scale text or image generation.

Figure: Figure 1: D3PM forward and (learned) reverse process applied to a quantized swiss roll. Each dot represents a 2D categorical variable. Top: samples from the uniform, discretized Gaussian, and absorbing state D3PM model forward processes, along with corresponding transition matrices $\bm{Q}$. Bottom: samples from a learned discretized Gaussian reverse process.
Refer to caption: /html/2107.03006/assets/figures/final.png

Our aim in this work is to improve and extend discrete diffusion models by using a more structured categorical corruption process to shape data generation, as illustrated in Figure [1](#S1.F1). Our models do not require relaxing or embedding discrete data (including images) into continuous spaces, and can embed structure or domain knowledge into the transition matrices used by the forward process. We achieve significantly improved results by taking advantage of this flexibility. We develop structured corruption processes appropriate for text data, using similarity between tokens to enable gradual corruption and denoising. Expanding further, we also explore corruption processes that insert [MASK] tokens, which let us draw parallels to autoregressive and mask-based generative models. Finally, we study discrete diffusion models for quantized images, taking inspiration from the locality exploited by continuous diffusion models. This leads to a particular choice of discrete corruption process that diffuses preferentially to more similar states and leads to much better results in the image domain.

Overall, we make a number of technical and conceptual contributions.
Beyond designing several new structured diffusion models, we introduce a new auxiliary loss which stabilizes training of D3PMs and a family of noise schedules based on mutual information that lead to improved performance. We strongly outperform various non-autoregressive baselines for text generation on character-level text generation, and successfully scale discrete diffusion models to large vocabularies and long sequence lengths. We also achieve strong results on the image dataset CIFAR-10, approaching or exceeding the Gaussian diffusion model from on log-likelihoods and sample quality.

## 2 Background: diffusion models

Diffusion models are latent variable generative models characterized by a forward and a reverse Markov process. The forward process $q(\bm{x}_{1:T}|\bm{x}_{0})=\prod_{t=1}^{T}q(\bm{x}_{t}|\bm{x}_{t-1})$ corrupts the data $\bm{x}_{0}\sim q(\bm{x}_{0})$ into a sequence of increasingly noisy latent variables $\bm{x}_{1:T}=\bm{x}_{1},\bm{x}_{2},...,\bm{x}_{T}$. The learned reverse Markov process $p_{\theta}(\bm{x}_{0:T})=p(\bm{x}_{T})\prod_{t=1}^{T}p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})$ gradually denoises the latent variables towards the data distribution. For example, for continuous data, the forward process typically adds Gaussian noise, which the reverse process learns to remove.

In order to optimize the generative model $p_{\theta}(\bm{x}_{0})$ to fit the data distribution $q(\bm{x}_{0})$, we typically optimize a variational upper bound on the negative log-likelihood:

$$ $\displaystyle L_{\mathrm{vb}}=\mathbb{E}_{q(\bm{x}_{0})}\bigg{[}$ $\displaystyle\underbrace{D_{\mathrm{KL}}[q(\bm{x}_{T}|\bm{x}_{0})||p(\bm{x}_{T})]}_{L_{T}}+\sum_{t=2}^{T}\underbrace{\mathbb{E}_{q(\bm{x}_{t}|\bm{x}_{0})}\big{[}D_{\mathrm{KL}}[q(\bm{x}_{t-1}|\bm{x}_{t},\bm{x}_{0})||p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})]\big{]}}_{L_{t-1}}$ $\displaystyle\underbrace{-\mathbb{E}_{q(\bm{x}_{1}|\bm{x}_{0})}[\log p_{\theta}(\bm{x}_{0}|\bm{x}_{1})]}_{L_{0}}\bigg{]}.$ (1) $$

When the number of time steps $T$ goes to infinity, both the forward process and the reverse process share the same functional form , allowing the use of a learned reverse process from the same class of distributions as that of the forward process.
Furthermore, for several choices of the forward process the distribution $q(\bm{x}_{t}|\bm{x}_{0})$ converges to a stationary distribution $\pi(\bm{x})$ in the limit $t\rightarrow\infty$ independent of the value of $\bm{x}_{0}$.
When the number of time steps $T$ is large enough and
we choose $\pi(\bm{x})$ as the prior $p(\bm{x}_{T})$,
we can guarantee that the
$L_{T}$ term in ([1](#S2.E1)) will approach zero regardless of the data distribution $q(\bm{x}_{0})$.
(Alternatively, one can use a learned prior $p_{\theta}(\bm{x}_{T})$.)

While $q(\bm{x}_{t}|\bm{x}_{t-1})$ can in theory be arbitrary, efficient training of $p_{\theta}$ is possible when $q(\bm{x}_{t}|\bm{x}_{t-1})$:

- 1.
Permits efficient sampling of $\bm{x}_{t}$ from $q(\bm{x}_{t}|\bm{x}_{0})$ for an arbitrary time $t$, allowing us to randomly sample timesteps and optimize each $L_{t-1}$ term individually with stochastic gradient descent,
- 2.
Has a tractable expression for the forward process posterior $q(\bm{x}_{t-1}|\bm{x}_{t},\bm{x}_{0})$, which allows us to compute the KL divergences present in the $L_{t-1}$ term of ([1](#S2.E1)).

The majority of recent work in continuous spaces defines the forward and reverse distributions as $q(\bm{x}_{t}|\bm{x}_{t-1})=\mathcal{N}\left(\bm{x}_{t}|\sqrt{1-\beta_{t}}\bm{x}_{t-1},\beta_{t}\bm{I}\right)$ and $p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})=\mathcal{N}\left(\bm{x}_{t-1}|\bm{\mu}_{\theta}(\bm{x}_{t},t),\bm{\Sigma}_{\theta}(\bm{x}_{t},t)\right)$, respectively.
The aforementioned properties hold in the case of these Gaussian diffusion models:
the forward process $q(\bm{x}_{t}|\bm{x}_{0})$ converges to a stationary distribution, motivating the choice $p(\bm{x}_{T})=\mathcal{N}\left(\bm{x}_{T}|\bm{0},\bm{I}\right)$, and both $q(\bm{x}_{t}|\bm{x}_{0})$ and $q(\bm{x}_{t-1}|\bm{x}_{t},\bm{x}_{0})$ are tractable Gaussian distributions for which the KL divergence can be computed analytically.

## 3 Diffusion models for discrete state spaces

Diffusion models with discrete state spaces were first introduced by , who considered a diffusion process over binary random variables.
extended
the model class to categorical random variables with transition matrices characterized by uniform transition probabilities. In their supplementary material, also derived this extension, although no experiments were performed with this model class.
Here, we briefly describe a more general framework for diffusion with categorical random variables which includes these models as special cases.

For scalar discrete random variables with $K$ categories $x_{t},x_{t-1}\in{1,...,K}$
the forward transition probabilities can be represented by matrices: $[\bm{Q}_{t}]_{ij}=q(x_{t}=j|x_{t-1}=i)$.
Denoting the one-hot version of $x$ with the row vector $\bm{x}$, we can write

$$ $\displaystyle q(\bm{x}_{t}|\bm{x}_{t-1})=\mathrm{Cat}(\bm{x}_{t};\bm{p}=\bm{x}_{t-1}\bm{Q}_{t}),$ (2) $$

where $\mathrm{Cat}(\bm{x};\bm{p})$ is a categorical distribution over the one-hot row vector $\bm{x}$ with probabilities given by the row vector $\bm{p}$, and $\bm{x}_{t-1}\bm{Q}_{t}$ is to be understood as a row vector-matrix product.
We assume that $\bm{Q}_{t}$ is applied to each pixel of an image or each token in a sequence independently, and that $q$ factorizes over these higher dimensions as well; we thus write $q(\bm{x}_{t}|\bm{x}_{t-1})$ in terms of a single element.
Starting from $\bm{x}_{0}$, we obtain the following $t$-step marginal and posterior at time $t-1$:

$$ $\displaystyle q(\bm{x}_{t}|\bm{x}_{0})=\mathrm{Cat}\left(\bm{x}_{t};\bm{p}=\bm{x}_{0}\overline{\bm{Q}}_{t}\right),\quad\text{with}\quad\overline{\bm{Q}}_{t}=\bm{Q}_{1}\bm{Q}_{2}\ldots\bm{Q}_{t}$ $\displaystyle q(\bm{x}_{t-1}|\bm{x}_{t},\bm{x}_{0})=\frac{q(\bm{x}_{t}|\bm{x}_{t-1},\bm{x}_{0})q(\bm{x}_{t-1}|\bm{x}_{0})}{q(\bm{x}_{t}|\bm{x}_{0})}=\mathrm{Cat}\left(\bm{x}_{t-1};\bm{p}=\frac{\bm{x}_{t}\bm{Q}_{t}^{\top}\odot\bm{x}_{0}\overline{\bm{Q}}_{t-1}}{\bm{x}_{0}\overline{\bm{Q}}_{t}\bm{x}_{t}^{\top}}\right).$ (3) $$

Note that due to the Markov property of the forward process $q(\bm{x}_{t}|\bm{x}_{t-1},\bm{x}_{0})=q(\bm{x}_{t}|\bm{x}_{t-1})$.
Assuming that the reverse process $p_{\theta}(\bm{x}_{t}|\bm{x}_{t-1})$ is also factorized as conditionally independent over the image or sequence elements, the KL divergence between $q$ and $p_{\theta}$ can be computed by simply summing over all possible values of each random variable; we thus satisfy criteria 1 and 2 discussed in Section [2](#S2).
Depending on $\bm{Q}_{t}$, the cumulative products $\overline{\bm{Q}}_{t}$ can often be computed in closed form, or simply precomputed for all $t$.
However, for large $K$ and large $T$ this may be prohibitive. In Appendix [A.4](#A1.SS4) we discuss how to ensure $\overline{\bm{Q}}_{t}$ can still be computed efficiently in this case, allowing the framework to scale to a larger number of categories.

In the next section we discuss the choice of the Markov transition matrices $\bm{Q}_{t}$ and corresponding stationary distributions. From here on, we refer to the general class of diffusion models with discrete state spaces as Discrete Denoising Diffusion Probabilistic Models (D3PMs).

### 3.1 Choice of Markov transition matrices for the forward process

An advantage of the D3PM framework described above is the ability to control the data corruption and denoising process by choosing $\bm{Q}_{t}$, in notable contrast to continuous diffusion, for which only additive Gaussian noise has received significant attention.
Besides the constraint that the rows of $\bm{Q}_{t}$ must sum to one to conserve probability mass, the only other constraint in choosing $\bm{Q}_{t}$ is that the rows of $\overline{\bm{Q}}_{t}=\bm{Q}_{1}\bm{Q}_{2}\ldots\bm{Q}_{t}$ must converge to a known stationary distribution(^1^11If a stationary distribution is not known, we can introduce a learned prior $p_{\theta}(\bm{x}_{T})$; we note that this is equivalent to extending the forward process by appending a rank-one matrix $\bm{Q}_{T+1}$ that ignores $\bm{x}_{T}$ and produces a deterministic $\bm{x}_{T+1}$, then learning the reverse step $p_{\theta}(\bm{x}_{T}|\bm{x}_{T+1})=p_{\theta}(\bm{x}_{T})$.)
when $t$ becomes large, which can be guaranteed while imposing minimal restrictions on $\bm{Q}_{t}$ (see Appendix [A.1](#A1.SS1)).

We argue that for most real-world discrete data, including images and text,
it makes sense to add domain-dependent structure to the transition matrices $\bm{Q}_{t}$ as a way of controlling the forward corruption process and the learnable reverse denoising process.
Below we briefly discuss the uniform transition matrices that have been studied in prior work , along with a set of structured transition matrices we have explored for our image and text dataset experiments; see Appendix [A.2](#A1.SS2) for more details on each matrix type. We also note that this set is not exhaustive, and many other transition matrices could also be used within the D3PM framework.

Uniform (Appendix [A.2.1](#A1.SS2.SSS1)). considered a simple $2\times 2$ transition matrix for binary random variables. later extended this to categorical variables, proposing a transition matrix $\bm{Q}_{t}=(1-\beta_{t})\bm{I}+\beta_{t}/K\;\mathbbm{1}\mathbbm{1}^{T}$ with $\beta_{t}\in[0,1]$. Since this transition matrix is doubly stochastic with strictly positive entries, the stationary distribution is uniform.
Because the transition probability to any other state is uniform, in this paper we equivalently refer to this discrete diffusion instance as D3PM-uniform.

Absorbing state (Appendix [A.2.2](#A1.SS2.SSS2)). Motivated by the success of BERT and recent work on Conditional Masked Language Models (CMLMs) in text, we consider a transition matrix with an absorbing state (called [MASK]), such that each token either stays the same or transitions to [MASK] with some probability $\beta_{t}$. This does not impose particular relationships between categories, similar to uniform diffusion, but still allows corrupted tokens to be distinguished from original ones. Moreover, the stationary distribution is not uniform but has all the mass on the [MASK] token. For images, we reuse the grey pixel as the [MASK] absorbing token.

Discretized Gaussian (Appendix [A.2.3](#A1.SS2.SSS3)). Instead of transitioning uniformly to any other state, for ordinal data we propose imitating a continuous space diffusion model by using a discretized, truncated Gaussian distribution. We choose a normalization such that the transition matrix is doubly stochastic, leading to a uniform stationary distribution. This transition matrix will transition between more similar states with higher probability, and is well suited for quantized ordinal data such as images.

Token embedding distance (Appendix [A.2.4](#A1.SS2.SSS4)). Textual data does not have ordinal structure, but there may still be interesting semantic relationships. For instance, in a character level vocabulary vowels may be more similar to each other than they are to consonants. As a demonstration of the generality of the D3PM framework, we explore using similarity in an embedding space to guide the forward process, and construct a doubly-stochastic transition matrix that transitions more frequently between tokens that have similar embeddings while maintaining a uniform stationary distribution.

For uniform and absorbing-state diffusion, the cumulative products $\overline{\bm{Q}}_{t}$ can be computed in closed form (see Appendix [A.4.1](#A1.SS4.SSS1)); the remainder can be precomputed.

### 3.2 Noise schedules

We consider several different options for the noise schedule of the forward process. For discretized Gaussian diffusion, we explore linearly increasing the variance of the Gaussian before discretizing it. (Note that a linear schedule for $\bm{Q}_{t}$ leads to a nonlinear amount of cumulative noise in $\overline{\bm{Q}}_{t}$.)
For uniform diffusion we use the cosine schedule which sets the cumulative probability of a transition to a cosine function, as introduced by and adapted by .
For a general set of transition matrices $\bm{Q}_{t}$ (such as the one based on token embeddings), previously proposed schedules may not be directly applicable. We consider linearly interpolating the mutual information between $\bm{x}_{t}$ and $\bm{x}_{0}$ to zero, i.e. $I(\bm{x}_{t};\bm{x}_{0})\approx(1-\frac{t}{T})\,H(\bm{x}_{0})$. Interestingly, for the specific case of absorbing-state D3PMs, this schedule reduces to exactly the $(T-t+1)^{-1}$ schedule proposed by for a Bernoulli diffusion process. See Appendix [A.7](#A1.SS7) for more details.

### 3.3 Parameterization of the reverse process

While it is possible to directly predict the logits of $p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})$ using a neural network $\mathrm{nn}_{\theta}(\bm{x}_{t})$, we follow and and focus on using a neural network
$\mathrm{nn}_{\theta}(\bm{x}_{t})$ to predict the logits of a distribution $\widetilde{p}_{\theta}(\widetilde{\bm{x}}_{0}|\bm{x}_{t})$, which we combine with $q(\bm{x}_{t-1}|\bm{x}_{t},\bm{x}_{0})$ and a summation over one-hot representations of $\bm{x}_{0}$ to obtain the following parameterization

$$ $\displaystyle p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})\propto\sum_{\widetilde{\bm{x}}_{0}}q(\bm{x}_{t-1},\bm{x}_{t}|\widetilde{\bm{x}}_{0})\widetilde{p}_{\theta}(\widetilde{\bm{x}}_{0}|\bm{x}_{t}).$ (4) $$

We note that under this $\bm{x}_{0}$-parameterization the KL divergence $D_{\mathrm{KL}}[q(\bm{x}_{t-1}|\bm{x}_{t},\bm{x}_{0})||p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})]$ will be zero if $\widetilde{p}_{\theta}(\widetilde{\bm{x}}_{0}|\bm{x}_{t})$ places all of its probability mass on the original value $\bm{x}_{0}$.
The decomposition of $q(\bm{x}_{t-1}|\bm{x}_{t},\bm{x}_{0})$ in ([3](#S3.E3)) also provides us with a motivation for this parameterization. According to ([3](#S3.E3)), in a given state $\bm{x}_{t}$, the optimal reverse process only takes into account transitions to states for which $q(\bm{x}_{t}|\bm{x}_{t-1})$ is non-zero. Therefore, the sparsity pattern of $\bm{Q}_{t}$ determines the sparsity pattern of the ideal reverse transition probabilities in $p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})$. The parameterization in ([4](#S3.E4)) automatically ensures that the learned reverse probability distribution $p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})$ has the correct sparsity pattern dictated by the choice of the Markov transition matrix $\bm{Q}_{t}$. This parameterization also lets us perform inference with $k$ steps at a time, by predicting $p_{\theta}(\bm{x}_{t-k}|\bm{x}_{t})=\sum q(\bm{x}_{t-k},\bm{x}_{t}|\widetilde{\bm{x}}_{0})\widetilde{p_{\theta}}(\widetilde{\bm{x}}_{0}|\bm{x}_{t})$.

Finally, when modeling ordinal discrete data, instead of predicting the logits of $\widetilde{p}_{\theta}(\widetilde{\bm{x}}_{0}|\bm{x}_{t})$ directly with the output of a neural net, another option is to model the probabilities with a truncated discretized logistic distribution (see Appendix [A.8](#A1.SS8)). This provides an extra ordinal inductive bias to the reverse model and boosts FID and log-likelihood scores for images.

### 3.4 Loss function

While the original diffusion models introduced by were optimized with the negative variational lower bound $L_{\mathrm{vb}}$ of ([1](#S2.E1)), more recent diffusion models are optimized with different objectives.
For instance, derive a simplified loss function ($L_{\mathrm{simple}}$) that reweights the negative variational bound, and explore a hybrid loss $L_{\mathrm{hybrid}}=L_{\mathrm{simple}}+\lambda L_{\mathrm{vb}}$ (using one term to learn the predicted mean and the other to learn predicted variance).
Inspired by this recent work, we introduce an auxiliary denoising objective for the $\bm{x}_{0}$-parameterization of the reverse process, which encourages good predictions of the data $\bm{x}_{0}$ at each time step.
We combine this with the negative variational lower bound, yielding the following alternative loss function:

$$ $\displaystyle L_{\lambda}=$ $\displaystyle L_{\mathrm{vb}}+\lambda\;\mathbb{E}_{q(\bm{x}_{0})}\mathbb{E}_{q(\bm{x}_{t}|\bm{x}_{0})}[-\log\widetilde{p}_{\theta}(\bm{x}_{0}|\bm{x}_{t})].$ (5) $$

Note that the auxiliary loss coincides with the cross entropy term $L_{0}$ in ([1](#S2.E1)) at $t=1$. Furthermore, due to the $\bm{x}_{0}$-parameterization of $p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})$, both the auxiliary loss term and $D_{\mathrm{KL}}[q(\bm{x}_{t-1}|\bm{x}_{t},\bm{x}_{0})||p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})]$ in $L_{\mathrm{vb}}$ are minimized exactly when $\widetilde{p}_{\theta}(\widetilde{\bm{x}}_{0}|\bm{x}_{t})$ has all its mass on the datapoint $\bm{x}_{0}$.
We find that training with this loss leads to improved quality of image samples.

## 4 Connection to existing probabilistic models for text

In this section we expand on interesting connections between the D3PM framework and several existing probabilistic and language modeling approaches.

BERT is a one-step diffusion model: One possible D3PM transition matrix is a combination of a uniform transition matrix and an absorbing state at the [MASK] token
(i.e. $\bm{Q}=\alpha\mathbbm{1}e_{m}^{T}+\beta\mathbbm{1}\mathbbm{1}^{T}/K+(1-\alpha-\beta)I$, where $e_{m}$ is a one-hot vector on the [MASK] token).
For a one-step diffusion process in which $q(\bm{x}_{1}|\bm{x}_{0})$ replaces 10% of tokens with [MASK] and 5% uniformly at random, this leads precisely to the BERT denoising objective, i.e. $L_{vb}-L_{T}=-\mathbb{E}_{q(\bm{x}_{1}|\bm{x}_{0})}[\log p_{\theta}(\bm{x}_{0}|\bm{x}_{1})]=L_{BERT}$,
since $L_{T}$ is a constant independent of $\theta$ (assuming a fixed prior).

Autoregressive models are (discrete) diffusion models: Consider a diffusion process that deterministically masks tokens one-by-one in a sequence of length $N=T$:
$q(\left[\bm{x}_{t}\right]_{i}\mid\bm{x}_{0})=[\bm{x}_{0}]_{i}\text{ if }i<N-t\text{ else [MASK] }$. This is a deterministic forward process, so $q(\bm{x}_{t-1}|\bm{x}_{t},\bm{x}_{0})$ is a delta distribution on the $\bm{x}_{t}$ sequence with one fewer mask: $q(\left[\bm{x}_{t-1}\right]_{i}|\bm{x}_{t},\bm{x}_{0})=\delta_{[\bm{x}_{t}]_{i}}\text{ if }i\neq T-t\text{ else }\delta_{[\bm{x}_{0}]_{i}}$. While this process is not applied independently to each token, it can
be recast as an independently-applied diffusion process on the product space
$[0...N]\times\mathcal{V}$, where each token is tagged with its position in the sequence, $\mathcal{V}$ is the vocabulary, and $\bm{Q}$ is an $N\times|\mathcal{V}|\times N\times|\mathcal{V}|$ sparse matrix.

Because all tokens except the one at position $i=T-t$ have deterministic posteriors, the KL divergence $D_{KL}(q([\bm{x}_{t-1}]_{j}|\bm{x}_{t},\bm{x}_{0})\mid\mid p_{\theta}([\bm{x}_{t-1}]_{j}|\bm{x}_{t}))$ is zero for all other positions. The only token for which this is not true is the token at position $i$, for which $D_{KL}(q([\bm{x}_{t-1}]_{i}|\bm{x}_{t},\bm{x}_{0})\mid\mid p_{\theta}([\bm{x}_{t-1}]_{i}|\bm{x}_{t}))=-\log p_{\theta}([\bm{x}_{0}]_{i}|\bm{x}_{t})$, the standard cross entropy loss for an autoregressive model.

(Generative) Masked Language-Models (MLMs) are diffusion models: Generative Masked Language Models (, ) are generative models that generate text from a sequence of [MASK] tokens. They are usually trained by sampling a sequence $\bm{x}_{0}$, masking $k$ tokens according to some schedule, and learning to predict the masked tokens given context. It turns out that a D3PM absorbing ([MASK]) model trained on the usual ELBO objective with
the $\bm{x}_{0}$-parameterization from [3.3](#S3.SS3) reduces to a reweighted version of this MLM objective (see Appendix [A.3](#A1.SS3) for a detailed derivation).

## 5 Text generation

For text, we experiment with generation on two datasets: text8 , a character-level dataset extracted from English-language Wikipedia, and the One Billion Word dataset (LM1B) , a large dataset of shuffled English-language sentences. For both, we train a D3PM uniform model based on the work by (D3PM uniform) and a model that masks tokens (D3PM absorbing). We also consider a model that transitions uniformly to nearest neighbors in a token embedding space (D3PM NN). We follow and use $T=1000$ timesteps, although we are also able to evaluate on fewer due to the parameterization in Section [3.3](#S3.SS3).

### 5.1 Character-level generation on text8

**Table 1: Quantitative results on text8. NLL is reported on the entire test set. Sample times are for generating a single example of length 256. Results are reported on two seeds. All models are standard 12-layer transformers unless otherwise noted. ${}^{\dagger}$Transformer XL is a 24-layer transformer, using a 784 context window. ${}^{\ddagger}$Results reported by by running code from official repository.**
| Model | Model steps | NLL (bits/char) ($\downarrow$) | Sample time (s) ($\downarrow$) |
| --- | --- | --- | --- |
| Discrete Flow ($8\times 3$ layers) | - | $1.23$ | $0.16$ |
| Argmax Coupling Flow | - | $1.80$ | $0.40\pm 0.03$ |
| IAF / SCF ${}^{\ddagger}$ | - | $1.88$ | $0.04\pm 0.0004$ |
| Multinomial Diffusion (D3PM uniform) | $1000$ | $\leq 1.72$ | $26.6\pm 2.2$ |
| D3PM uniform (ours) | $1000$ | $\leq 1.61\pm 0.02$ | $3.6\pm 0.4$ |
| D3PM NN ($L_{\mathrm{vb}}$) (ours) | $1000$ | $\leq 1.59\pm 0.03$ | $3.1474\pm 0.0002$ |
| D3PM mask ($L_{\lambda=0.01}$) (ours) | $1000$ | $\leq 1.45\pm 0.02$ | $3.4\pm 0.3$ |
| D3PM uniform (ours) | $256$ | $\leq 1.68\pm 0.01$ | $0.5801\pm 0.0001$ |
| D3PM NN ($L_{\mathrm{vb}}$) (ours) | $256$ | $\leq 1.64\pm 0.02$ | $0.813\pm 0.002$ |
| D3PM absorbing ($L_{\lambda=0.01}$) (ours) | $256$ | $\leq 1.47\pm 0.03$ | $0.598\pm 0.002$ |
| Transformer decoder (ours) | $256$ | $1.23$ | $0.3570\pm 0.0002$ |
| Transformer decoder | $256$ | $1.18$ | - |
| Transformer XL ${}^{\dagger}$ | $256$ | $1.08$ | - |
| D3PM uniform (ours) | $20$ | $\leq 1.79\pm 0.03$ | $0.0771\pm 0.0005$ |
| D3PM NN ($L_{\mathrm{vb}}$) (ours) | $20$ | $\leq 1.75\pm 0.02$ | $0.1110\pm 0.0001$ |
| D3PM absorbing ($L_{\lambda=0.01}$) (ours) | $20$ | $\leq 1.56\pm 0.04$ | $0.0785\pm 0.0003$ |

text8 is a character-level text dataset consisting of a small vocabulary of 27 tokens: the letters ‘a’-‘z’ and the ‘_’ whitespace token. We follow the convention of training and evaluating text8 in chunks of length 256 without any preprocessing . For nearest-neighbor D3PM, our nearest neighbor graph in character-space is shown in Appendix [B.2.1](#A2.SS2.SSS1). D3PM uniform models were trained with a cosine schedule from (ablations in Appendix [B.2.1](#A2.SS2.SSS1)), while D3PM absorbing and D3PM NN models were trained with a mutual information schedule.

Table [1](#S5.T1) shows that for D3PM, the D3PM absorbing model performed the best, exceeding the uniform and NN diffusion models. We were able to improve upon the baseline result of with hyperparameter tuning, and our uniform and NN results outperformed results from across all inference steps, down to as few as 20. We found that $L_{\lambda=0.01}$ worked best for D3PM absorbing, while $L_{\mathrm{vb}}$ was better for D3PM uniform. Our model outperforms all non-autoregressive baselines except one, the Discrete Flow model
(for which unfortunately no open-source implementations exist), and is also faster than all but one method, the IAF/SCF model . It is also nearly 20x faster than an autoregressive transformer of the same size. We also include a plot of inference time as a function of iterations in Appendix [B.2.1](#A2.SS2.SSS1).
D3PM with the mask absorbing token was by far the best performing model, which lends credibility to the use of masks in denoising auto-encoders. Nearest-neighbor diffusion only narrowly improves upon a D3PM-uniform model: this was a surprising negative result for us, suggesting that not all notions of structure are meaningful.

### 5.2 Text generation on LM1B

Text generation for large-scale text datasets and large vocabularies with discrete diffusion models has not been previously demonstrated. We include results from LM1B as a proof of concept, showing that these models can indeed scale (as discussed in Appendix [A.4](#A1.SS4)), and that the D3PM absorbing model continues to excel. All models were trained and evaluated on packed sequences of length $128$, using a sentencepiece(^2^22[https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)) vocabulary of size $8192$.

Figure: Figure 2: Left: perplexity v.s. sampling iterations for LM1B. Right: Using a trained D3PM absorbing model for LM1B to (top) generate new sentences and (bottom) reconstruct corrupted examples.
Refer to caption: /html/2107.03006/assets/figures/perplexity3.png

Table [2](#S5.T2) contains results from experiments on LM1B. Overall, mask diffusion (D3PM absorbing) does relatively well, approaching the performance of a comparable autoregressive model of the same size, and scaling to far fewer steps, while uniform diffusion performs significantly worse. We find, surprisingly, that the D3PM NN model performs worse than the uniform model in terms of log likelihoods (although it demonstrates unique qualitative behavior). This suggests that word embedding similarity may not be a meaningful kind of locality in a diffusion process. We found the the $L_{\lambda=0.01}$ loss worked best for the mask absorbing model, but reduced performance for the other models. We note the surprising scaling in perplexity in Figure [2](#S5.F2), achieving strong results with as few as 10 inference steps. We also show samples from our model and completions from corrupted samples.

**Table 2: Quantitative results on LM1B. Perplexity reported on the test set. Results are reported on two seeds. All models have context window length 128 and 12 layers unless otherwise noted. ${}^{\dagger}$Transformer XL is a 24 layer transformer. ${}^{\ddagger}$rounded for readability, see Appendix [B.2.2](#A2.SS2.SSS2).**
| Metric: | Perplexity ($\downarrow$) |  | Sample time${}^{\ddagger}$ (s) ($\downarrow$) |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| inference steps: | $1000$ | $128$ | $64$ |  | $1000$ | $128$ | $64$ |
| D3PM uniform | 137.9 $\pm$ 2.1 | 139.2 $\pm$ 1.2 | 145.0 $\pm$ 1.2 |  | 1.82 | 0.21 | 0.08 |
| D3PM NN | 149.5 $\pm$ 1.3 | 158.6 $\pm$ 2.2 | 160.4 $\pm$ 1.2 |  | 21.29 | 6.69 | 5.88 |
| D3PM absorbing | 76.9 $\pm$ 2.3 | 80.1 $\pm$ 1.2 | 83.6 $\pm$ 6.1 |  | 1.90 | 0.19 | 0.10 |
| Transformer (ours) | - | 43.6 | - |  | - | 0.26 | - |
| Transformer XL ${}^{\dagger}$ | - | 21.8 | - |  | - | - | - |

## 6 Image generation

**Table 3: Inception scores (IS), Frechet Inception Distance (FID) and negative log-likehood (NLL) on the image dataset CIFAR-10. The NLL is reported on the test set in bits per dimension. We report our results as averages with standard deviations, obtained by training five models with different seeds.**
| Model | IS ($\uparrow$) | FID ($\downarrow$) | NLL ($\downarrow$) |
| --- | --- | --- | --- |
| Sparse Transformer |  |  | 2.80 |
| NCSN | $8.87\pm 0.12$ | 25.32 |  |
| NCSNv2 | $8.40\pm 0.07$ | 10.87 |  |
| StyleGAN2 + ADA | $9.74\pm 0.05$ | 3.26 |  |
| Diffusion (original), $L_{\mathrm{vb}}$ |  |  | $\leq 5.40$ |
| DDPM $L_{\mathrm{vb}}$ | $7.67\pm 0.13$ | $13.51$ | $\leq 3.70$ |
| DDPM $L_{\mathrm{simple}}$ | $9.46\pm 0.11$ | $3.17$ | $\leq 3.75$ |
| Improved DDPM $L_{\mathrm{vb}}$ |  | $11.47$ | $\leq 2.94$ |
| Improved DDPM $L_{\mathrm{simple}}$ |  | $2.90$ | $\leq 3.37$ |
| DDPM++ cont |  | $2.92$ | $2.99$ |
| NCSN++ cont. | $9.89$ | $2.20$ |  |
| D3PM uniform $L_{\mathrm{vb}}$ | $5.99\pm 0.14$ | $51.27\pm 2.15$ | $\leq 5.08\pm 0.02$ |
| D3PM absorbing $L_{\mathrm{vb}}$ | $6.26\pm 0.10$ | $41.28\pm 0.65$ | $\leq 4.83\pm 0.02$ |
| D3PM absorbing $L_{\lambda=0.001}$ | $6.78\pm 0.08$ | $30.97\pm 0.64$ | $\leq 4.40\pm 0.02$ |
| D3PM Gauss $L_{\mathrm{vb}}$ | $7.75\pm 0.13$ | $15.30\pm 0.55$ | $\leq 3.966\pm 0.005$ |
| D3PM Gauss $L_{\lambda=0.001}$ | $8.54\pm 0.12$ | $8.34\pm 0.10$ | $\leq 3.975\pm 0.006$ |
| D3PM Gauss + logistic $L_{\lambda=0.001}$ | $8.56\pm 0.10$ | $7.34\pm 0.19$ | $\leq 3.435\pm 0.007$ |

We evaluate the performance of several D3PM models on the task of unconditional image generation with the dataset CIFAR-10 . We follow and use $T=1000$ timesteps for all models and verify that for all models the forward process converges to the stationary distribution within $T$ steps, yielding a value of at most $L_{T}\approx 10^{-5}$ bits per dimension. We train three versions of D3PM with different transition matrices: doubly stochastic matrices with uniform transition probabilities (D3PM uniform) , transition matrices with an absorbing state located at R, G and B values of 128 (D3PM absorbing) and doubly stochastic discretized Gaussian transition matrices (D3PM Gauss). For the D3PM uniform model we experimented with a linear $\beta_{t}$ schedule as well as the cosine schedule as proposed in , with the cosine schedule producing the best results. For D3PM absorbing we use the schedule $\beta_{t}=(T-t+1)^{-1}$ as also proposed in , which corresponds to increasing the probability of being in the absorbing state linearly over time.
For D3PM Gauss we use the same linear schedule as in . See Appendix [B.1](#A2.SS1) for more details on the experimental setup.

Figure: Figure 3: Left: progressive sampling at $t=1000,900,800,...,0$ for D3PM absorbing (top) and D3PM Gauss + logistic (bottom), trained with $L_{\lambda}$ loss on CIFAR-10. These samples were cherry picked. Right: (non cherry picked) samples from the D3PM Gauss + logistic model.
Refer to caption: /html/2107.03006/assets/figures/progressive_samples_mask_hybrid.png

Table [3](#S6.T3) shows that for D3PM models trained with the $L_{\mathrm{vb}}$ objective, D3PM Gauss performs better than D3PM absorbing and uniform on all metrics: Inception score (IS), Frechet Inception Distance (FID) and negative log-likelihood (NLL). The IS score of the uniform and absorbing D3PM models are comparable, while the FID score and NLL of the D3PM absorbing model are slightly better. We trained both D3PM absorbing and D3PM Gauss with the alternative loss function $L_{\lambda}$ of ([5](#S3.E5)), and we found $\lambda=0.001$ to work best.
We have also experimented with larger values of $\lambda$ and a model trained only with the auxiliary denoising term in ([5](#S3.E5)). Although this led to a more rapid increase in performance early on in training, the NLL leveled off at higher values for larger $\lambda$ and the FID even started increasing again.
The results show that the models trained with $L_{\lambda}$ perform significantly better than their counterparts trained with $L_{\mathrm{vb}}$.
One explanation for this boost in performance is that the cross entropy term leads to gradient noise that varies less with the time step $t$, which is in contrast to the large change in magnitude of the $L_{t-1}$ terms in $L_{\mathrm{vb}}$ for smaller $t$, as demonstrated by .
Finally, we achieve our best results by combining D3PM Gauss trained on $L_{\lambda}$ with a truncated logistic parameterization of the reverse process distribution $p_{\theta}(\widetilde{\bm{x}}_{0}|\bm{x}_{t})$ (D3PM Gauss + logistic). Figure [3](#S6.F3) shows samples from our best model (D3PM Gauss + logistic), as well as the D3PM absorbing model.

## 7 Related Work

Diffusion generative models were first proposed by and have gained renewed attention recently due to strong results on image and waveform generation . Recent works have proposed improvements for diffusion model training, including importance sampling of the ELBO, better noise schedules and implicit diffusion models . Several works have also drawn connections to score matching , leading to improved sampling algorithms in the continuous-time limit .

While most works have considered continuous diffusion models, discrete diffusion-like models were described in and applied to text generation and image segmentation data in . Some works have dealt with discrete data by embedding it in continuous space and leveraging Gaussian diffusion, but have not applied this to text. also considered generation of discrete structured objects using a diffusion-like Markov corruption process.

For text, denoising autoencoders have a long history both in representation learning and more recently as generative models . These closely resemble our absorbing state diffusion variants for a particular schedule and transition matrix (see Section [4](#S4)), although our framing allows us to compute log-likelihoods and experiment with alternative transition matrices. Other works have considered non-autoregressive translation and speech transcription via insertion and deletion , masking , and iteratively-refined sequence alignments .

## 8 Discussion

We have presented D3PMs, a class of models that improves diffusion models for discrete data by defining new kinds of discrete corruption processes.
We achieve strong empirical results relative to previous work on discrete diffusion models, even surpassing performance of continuous diffusion models in terms of log-likelihoods for image generation.
While these results are promising, one limitation is that—like much other work on non-autoregressive generative models—our models are still inferior to strong autoregressive models like Transformer XL for text generation, and continuous diffusion models still yield stronger results on image quality.
We expect that D3PMs can benefit further from the rapid development of continuous diffusion models . For example, further research in alternative losses for D3PM’s can take inspiration from the reweighted $L_{\mathrm{simple}}$ objective used in , or the resampled variational bound in . Furthermore, D3PM’s might benefit from increasing the number of timesteps and a more optimized noise schedule, as discussed in .
Another limitation comes from the choice of evaluation metrics that we use (and that are standard for evaluation of generative models).
Inception score and Frechet Inception Distance are based on neural networks that have been trained on a particular distribution of data, which is not representative for all use-cases, and focusing on average quality metrics may not accurately reflect performance across the wide diversity of settings where these generative models may be applied.
This creates a risk of negative social impacts where advances disproportionately favor a subset of the population.
Going forward, we are excited about the space of possibilities that arise within the D3PM framework.
We have found successes in leveraging the flexibility that comes from defining discrete corruption processes for discrete data, but we believe that there are many more possibilities that make use of richer forms of structure to define even more powerful discrete diffusion models.

## Acknowledgments and Disclosure of Funding

We would like to thank Hugo Larochelle for providing high-level feedback during the project, and Ben Poole for reviewing a draft version of this manuscript. We would also like to thank Julia Kreutzer and Xavier Garcia for helpful conversations about language experiments. We, the authors, declare to have no competing interests. The research conducted for this paper was entirely supported by Google.

## Appendix A Additional details regarding D3PMs

### A.1 Doubly-stochastic matrices

As discussed in Section [3.1](#S3.SS1), there are two constraints on $\bm{Q}_{t}$ that allow it to be used within a D3PM: the rows of $\bm{Q}_{t}$ must sum to one to conserve probability mass, and the rows of $\overline{\bm{Q}}_{t}=\bm{Q}_{1}\bm{Q}_{2}\ldots\bm{Q}_{t}$ must converge to a known stationary distribution as $t$ becomes large. Technically, it is also possible to use a learned prior $p_{\theta}(\bm{x}_{T})$, but assuming this is still modeled under a conditional independence assumption, $q(\bm{x}_{T}|\bm{x}_{0})$ must still be close to a stationary distribution for the $L_{T}$ loss term to be small.

One way to ensure that this occurs is to chose $\bm{Q}_{t}$ as increasing powers of a doubly stochastic base matrix $\bm{Q}$ (rows and columns sum to 1) with strictly positive entries. This is enough to ensure that $\bm{Q}$ is is irreducible and aperiodic and that product $\overline{\bm{Q}}_{t}$ converges as $t\rightarrow\infty$ to a uniform distribution over all states.
To show this, consider $\pi_{i}=1/K$ for $i=1,...,K$, and $\sum_{i=1}^{K}\bm{Q}_{i,:}=\bm{1}$ and $\sum_{j=1}^{K}\bm{Q}_{:,j}=\bm{1}$, then $[\bm{Q}\bm{\pi}]_{i}=\sum_{j=1}^{K}\bm{Q}_{i,j}\pi_{j}=1/K\sum_{j=1}^{K}\bm{Q}_{i,j}=1/K=\pi_{i}$, thus the uniform distribution is an eigenvector of the transition matrix with eigenvalue 1. Convergence to this distribution follows from the Perron-Frobenius theorem for positive square matrices.

More generally, a similar argument shows that even for $\bm{Q}_{t}$ that are not powers of the same base matrix, as long as each $\bm{Q}_{t}$ is doubly stochastic, irreducible, and aperiodic, the uniform distribution is the only possible stationary distribution, and as long as the second largest eigenvalue of $\bm{Q}_{t}$ is bounded below, the cumulative product $\overline{\bm{Q}}_{t}$ will converge to the uniform distribution. In practice, we choose $\bm{Q}_{t}$ to add more noise as $t$ increases, which ensures that $\overline{\bm{Q}}_{T}$ is very close to reaching a uniform stationary distribution.

### A.2 More details on possible choices of Markov transition matrices

#### A.2.1 Uniform diffusion

The transition matrix described by for the binary case, and extended by , to the categorical case, can be represented using the following $K\times K$ transition matrix

$$ $\displaystyle\left[\bm{Q}_{t}\right]_{ij}=\begin{cases}1-\frac{K-1}{K}\beta_{t}\quad&\text{if}\quad i=j\\ \frac{1}{K}\beta_{t}\quad&\text{if}\quad i\neq j\end{cases},$ (6) $$

This transition matrix can also be written as $(1-\beta_{t})I+\beta_{t}\mathbbm{1}\mathbbm{1}^{T}/K$, where $\mathbbm{1}$ is a column vector of all ones.

#### A.2.2 Diffusion with an absorbing state

For our diffusion models with an absorbing state $m$, we use the following matrix:

$$ $\displaystyle\left[\bm{Q}_{t}\right]_{ij}=\begin{cases}1\quad&\text{if}\quad i=j=m\\ 1-\beta_{t}\quad&\text{if}\quad i=j\neq m\\ \beta_{t}\quad&\text{if}\quad j=m,i\neq m\\ \end{cases}$ (7) $$

The transition matrix can also be written as $(1-\beta_{t})I+\beta_{t}\mathbbm{1}e^{T}_{m}$, where $e_{m}$ is a vector with a one on the absorbing state $m$ and zeros elsewhere.
Since $m$ is an absorbing state, the corruption process converges not to a uniform distribution but to the point-mass distribution on $m$.

For text generation, we let $m$ be the [MASK] token at index $K-1$; this leads to a BERT-like training objective, which masks tokens according to some schedule and learns to denoise them iteratively (see Section [4](#S4)). For image generation, we set $m$ to the gray RGB pixel $(128,128,128)$ at index $K//2$.

#### A.2.3 Discretized Gaussian transition matrices

For our D3PM models applied to ordinal data, inspired by continuous-space diffusion models, we use the following $K\times K$ matrix:

$$ $\displaystyle\left[\bm{Q}_{t}\right]_{ij}=\begin{cases}\frac{\exp\left(-\frac{4|i-j|^{2}}{(K-1)^{2}\beta_{t}}\right)}{\sum_{n=-(K-1)}^{K-1}\exp\left(-\frac{4n^{2}}{(K-1)^{2}\beta_{t}}\right)}\quad&\text{if}\quad i\neq j\[15.00002pt] 1-\sum_{l=0,l\neq i}^{K-1}[\bm{Q}_{t}]_{il}\quad&\text{if}\quad i=j\\ \end{cases}$ (8) $$

Normalization is ensured by assigning the diagonal values to one minus the sum of each row (not including the diagonal entry). Note that due to the normalization of the off-diagonal values over the range $\{-K+1,...,K-1\}$ the sum of each row excluding the diagonal entry is always smaller than 1. The result yields an irreducible doubly stochastic matrix and a forward process with a uniform stationary distribution. Similar to the continuous Gaussian diffusion model, the parameters $\beta_{t}$ influence the variance of the forward process distributions.

#### A.2.4 Structured diffusion in text: using word-embedding distance to introduce locality

For text, we construct a $k$-nearest neighbor adjacency matrix

$$ $[\mathbf{G}]_{ij}=1\text{ if }w_{i}\text{ is a k-nearest neighbor of }w_{j}\text{ else }0$ $$

constructed from a pre-trained embedding space over the vocabulary. Then we consider a symmetrized adjacency matrix of the form $\mathbf{A}=(\mathbf{G}+\mathbf{G}^{T})/(2k)$ where $k$ is the number of nearest neighbors of each node,
and finally construct a doubly stochastic rate matrix with

$$ $\displaystyle\left[\bm{R}\right]_{ij}=\begin{cases}-\sum_{l\neq i}A_{il}\quad&\text{if}\quad i=j\\ A_{ij}\quad&\text{ otherwise }\\ \end{cases}$ (9) $$

Our final transition matrix is constructed as a matrix exponential of this rate matrix:

$$ $\mathbf{Q}_{t}=\exp(\alpha_{t}\mathbf{R})=\sum_{n=0}^{\infty}\frac{\alpha_{t}^{n}}{n!}\bm{R}^{n}$ $$

Since $\bm{R}$ is symmetric and sums to zero along each row, $\mathbf{Q}_{t}$ is doubly stochastic, which ensures we have a uniform stationary distribution (as long as $G$ is connected). Increasing $\alpha_{t}$ over time allows us to add more noise for larger values of $t$.

Assuming word embeddings are some metric for syntactic or semantic similarity, this results in a corruption process that gradually moves away from the ground-truth sentence, swapping words with nearest-neighbors in embedding space. For character level modeling, this is a graph over characters, which more often transitions for instance from vowels to other vowels than from vowels to consonants. For words, this could transition between semantically similar words.

Figure: Figure 4: Two examples of noise schedules transforming text data. The top is a BERT-like absorbing + uniform diffusion which replaces tokens with [MASK] tokens (and occasionally with any other token, in black). The bottom is nearest-neighbor diffusion in embedding space. At left represents a possible column in the transition matrix.
Refer to caption: /html/2107.03006/assets/figures/text_diffusion.png

Figure: Figure 5: The character-level symmetrized 5-NN graph.
Refer to caption: /html/2107.03006/assets/figures/char_nn.png

For example, in Figure [4](#A1.F4), we construct the forward process to diffuse from "dog" to "cat" or "cow", which are nearby in embedding space, but not to more distant words. We can either bootstrap this process by updating the transition matrix $\bm{Q}$ dynamically during training, or use pretrained embeddings; we use pretrained embeddings for all of our experiments.

#### A.2.5 Band-diagonal transitions

A class of transition matrices that introduce local, ordinal inductive biases for structured data are band-diagonal transition matrices which only allow the corruption process to transition locally between states and biases the reverse process towards local iterative refinement. For example, in images, this can be used to allow transitions only between adjacent pixel values.

$$ $\displaystyle\left[\bm{Q}_{t}\right]_{ij}=\begin{cases}\frac{1}{K}\beta_{t}\quad&\text{if}\quad 0<|i-j|\leq v\\ 1-\sum_{l\neq i}Q_{il}\quad&\text{if}\quad i=j\\ \end{cases}$ (10) $$

where $v$ is the number of nonzero off-diagonal elements of $\bm{Q}$ above (and below) the main diagonal. Note that this is a doubly stochastic matrix, so the stationary distribution is uniform. We do not use these in our experiments.

#### A.2.6 Combinations of absorbing diffusion and other diffusion

A few ablations in Appendix [B.2.1](#A2.SS2.SSS1) consider transition matrices that combine absorbing-state or nearest-neighbor and uniform D3PM models. For instance, an absorbing-uniform transition matrix can be constructed $\bm{Q}=\alpha\mathbbm{1}e_{m}^{T}+\beta\mathbbm{1}\mathbbm{1}^{T}/K+(1-\alpha-\beta)I$, where $e_{m}$ is a one-hot vector on the [MASK] token.

### A.3 Generative Masked Language Models are Diffusion Models

Generative Masked Language Models are generative models that generate text from a sequence of [MASK] tokens. These are usually trained by sampling a sequence $\bm{x}_{0}$, masking tokens according to some schedule, and learning to predict the masked tokens given context. The actual masking procedure can either be done independently, i.e. by masking each token with probability $p=k/T$, like , or by sampling exactly $k$ tokens. The usual objective is(^3^33Sometimes the loss is un-normalized or normalized by the full sequence length.):

$$ $\min-\mathbb{E}_{q(\bm{x}_{0})}\left[\mathbb{E}_{k\in[1...|\bm{x}_{0}|]}\left[\frac{1}{k}\ \mathbb{E}_{\bm{x}_{k}\text{with $k$ masked tokens}}\left[\sum_{i\text{ with }[\bm{x}_{k}]_{i}=m}\log p_{\theta}([\bm{x}_{0}]_{i}|\bm{x}_{k})\right]\right]\right]$ (11) $$

where we first sample a datapoint $\bm{x}_{0}$, sample a number of tokens to mask $k$ (either uniformly or according to some schedule), then mask that many tokens at random and compute a cross entropy loss over those masked tokens. We claim that this training objective is a (reweighted) absorbing-state D3PM objective with a particular noise schedule and the $\bm{x}_{0}$-parameterization from [3.3](#S3.SS3) (and indeed, that any absorbing-state D3PM model with [MASK] as the absorbing state will be a reweighted version of this loss with different weights assigned to different numbers of masked tokens $k$).

Consider a D3PM with a schedule that masks tokens with probability $\beta_{t}$. The reverse process predicts $\widetilde{p}_{\theta}(\widetilde{\bm{x}_{0}}|\bm{x}_{t})$, then uses the forward process to compute $p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})\propto\sum q(\bm{x}_{t-1},\bm{x}_{t}|\widetilde{\bm{x}_{0}})\widetilde{p}_{\theta}(\widetilde{\bm{x}}_{0}|\bm{x}_{t})$. In the particular case of absorbing-state diffusion, for each masked token $[\bm{x}_{t}]_{i}=m$ in $\bm{x}_{t}$, we thus have

$$ $p_{\theta}([\bm{x}_{t-1}]_{i}|\bm{x}_{t})\propto\begin{cases}[\beta_{t}\prod_{s<t}(1-\beta_{s})]\widetilde{p}_{\theta}([\widetilde{\bm{x}}_{0}]_{i}=[\bm{x}_{0}]_{i}|\bm{x}_{t})&\text{for }[\bm{x}_{t-1}]_{i}=[\bm{x}_{0}]_{i}\neq m\\ 1-\prod_{s\leq t}(1-\beta_{s})&\text{for }[\bm{x}_{t-1}]_{i}=m\end{cases}$ $$

We note that for each unmasked token $[\bm{x}_{t}]_{i}=[\bm{x}_{0}]_{i}$, the KL-divergence is zero since unmasked tokens cannot make any other type of transition other than becoming masked. Also, the term in the KL divergence due to the probability of mask transitions is a constant, since mask transitions are independent of the model parameters $\theta$.
Our $L_{t}$ term is then

$$ $D_{\mathrm{KL}}[q(\bm{x}_{t-1}|\bm{x}_{t},\bm{x}_{0})||p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})]=-\left[\beta_{t}\prod_{s<t}(1-\beta_{s})\right]\sum_{i\text{ with }[\bm{x}_{t}]_{i}=m}\log\widetilde{p}_{\theta}([\bm{x}_{0}]_{i}|\bm{x}_{t})+C$ $$

where $C$ is independent of $\theta$ and the sum is taken over the masked tokens in $\bm{x}_{t}$.
For example, if we use $\beta(t)=1/(T-t+1)$ from , $\beta_{t}\prod_{i=0}^{t-1}(1-\beta_{i})=1/T$ and $1-\prod_{i=0}^{t}(1-\beta_{i})=(t-1)/T$, so $q([\bm{x}_{t-1}]_{i}=[\bm{x}_{0}]_{i}|[\bm{x}_{t}]_{i}=m,\bm{x}_{0})=1/t$ for non-mask tokens and we can simplify our $L_{t}$ objective to

$$ $D_{\mathrm{KL}}[q(\bm{x}_{t-1}|\bm{x}_{t},\bm{x}_{0})||p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})]=-\left[\frac{1}{t}\sum_{i\text{ with }[\bm{x}_{t}]_{i}=m}\log\widetilde{p}_{\theta}([\bm{x}_{0}]_{i}|\bm{x}_{t})\right]+C$ $$

where $\bm{x}_{t}$ masks tokens independently and uniformly with probability $t/T$. The $L_{T}$ term in our ELBO is 0 for the $1/(T-t+1)$ schedule, so the full objective (up to a constant) reduces to

$$ $\displaystyle\mathbb{E}_{q(\bm{x}_{0})}\Bigg{[}-\sum_{t=2}^{T}\frac{1}{t}\mathbb{E}_{q(\bm{x}_{t}|\bm{x}_{0})}\big{[}\sum_{i\text{ with }[\bm{x}_{t}]_{i}=m}\log p_{\theta}([\bm{x}_{0}]_{i}|\bm{x}_{t})]\big{]}$ $\displaystyle-\mathbb{E}_{q(\bm{x}_{1}|\bm{x}_{0})}[\sum_{i\text{ with }[\bm{x}_{1}]_{i}=m}\log p_{\theta}([\bm{x}_{0}]_{i}|\bm{x}_{1})]\Bigg{]}$ $\displaystyle=-\mathbb{E}_{q(\bm{x}_{0})}\left[\sum_{t=1}^{T}\frac{1}{t}\ \mathbb{E}_{q(\bm{x}_{t}|\bm{x}_{0})}\big{[}\sum_{i\text{ with }[\bm{x}_{t}]_{i}=m}\log p_{\theta}([\bm{x}_{0}]_{i}|\bm{x}_{t})]\big{]}\right]$ (12) $$

Note that while this looks very similar to Equation [11](#A1.E11) (with each term reweighted by $1/t$, the expected number of masked tokens) it is not exactly identical since masking is computed independently per-token position (instead of choosing exactly $k$ tokens to mask). This is an entirely practical way to do masking (and indeed some methods implement it this way).

Furthermore, since the masking probability varies linearly as $1-\prod(1-\beta_{t})=t/T$, this is very close to uniformly sampling the number of masked tokens $k$, but $k$ is actually drawn from a mixture of binomial distributions, i.e.

$$ $\displaystyle=-\mathbb{E}_{q(\bm{x}_{0})}\left[\mathbb{E}_{k\in[1...|X|]}\left[\mathbb{E}_{\bm{x}_{k}\text{with $k$ masked tokens}}\left[\alpha(k)\sum_{i\text{ with }[\bm{x}_{k}]_{i}=m}\log p_{\theta}([\bm{x}_{0}]_{i}|\bm{x}_{k})]\right]\right]\right]$ (13) $$

$$ $\displaystyle\alpha(k)=q(\bm{x}_{t}\text{ has $k$ masked tokens}|\bm{x}_{0}\text{ has $n$ tokens})=\frac{1}{T}\sum_{t=1}^{T}{n\choose k}\left(\frac{t}{T}\right)^{n-1}\left(1-\frac{t}{T}\right)^{n-k}$ (14) $$

which is very close to uniform weight over terms, but slightly downweights terms near 0 0 and $T$. By upweighting terms near the boundary, you could in theory make this exactly uniform and thus exactly recover Equation [11](#A1.E11). For instance, for 50 categories, absorbing-state diffusion produces the weighting shown in Figure [6](#A1.F6).

Figure: Figure 6: Plot of the probabilities of having $k$ tokens masked out of a length-50 sequence under a D3PM absorbing schedule with $T=50$ steps, which is very similar to the uniform weighting used by .
Refer to caption: /html/2107.03006/assets/figures/mask_weighting.png

### A.4 Scaling to a large number of categories

When the number of categories $K$ is large, it can quickly become impractical to store all of the transition matrices $\bm{Q}_{t}$ in memory, as the memory usage grows like $O(K^{2}T)$. And even if there is an algorithm to compute individual step matrices $\bm{Q}_{t}$ on demand, it may or may not be possible to do the same for the cumulative products $\overline{\bm{Q}}_{t}$. We propose two approaches to scaling D3PMs to large numbers of categories that ensure cumulative products are efficient: using low-rank corruption and using matrix exponentials.

#### A.4.1 Low-rank corruption

In the low-rank case, we consider structuring our transition matrices as

$$ $\displaystyle\bm{Q}_{t}$ $\displaystyle=\beta_{t}\bm{A}_{t}+(1-\beta_{t})\bm{I},$ (15) $$

where each $\bm{A}_{t}$ is a diagonalizable low-rank matrix with the same nonzero eigenvectors. In particular, recall that both absorbing-state diffusion and uniform diffusion
have this form: for uniform diffusion, $\bm{A}^{\text{uniform}}_{t}=\mathbbm{1}\mathbbm{1}^{T}/K$, and for absorbing-state diffusion $\bm{A}^{\text{abs}}_{t}=\mathbbm{1}\bm{e}_{m}^{T}$ where $\bm{e}_{m}$ is a one-hot vector on the absorbing state. Since products of $\bm{A}_{t}$’s are also low rank, the cumulative products $\overline{\bm{Q}}_{t}$ can be efficiently precomputed and stored using a much smaller amount of memory $O(r^{2}T)$ where $r=\text{rank}(\bm{A}_{t})$.

As an illustrative example, we describe in more detail how to efficiently represent uniform and absorbing-state transition matrices using the low-rank structure.

To compute products of uniform transition matrices (i.e. $\prod_{i}(1-\beta_{i})I+\beta_{i}\mathbbm{1}\mathbbm{1}^{T}/K$),
we can take advantage of the useful fact that products of matrices of the form $\alpha I+\beta\mathbbm{1}\mathbbm{1}^{T}$ also have this same form: $I^{2}=I$ and $\left(\beta\mathbbm{1}\mathbbm{1}^{T}\right)^{2}=\beta^{2}K\mathbbm{1}\mathbbm{1}^{T}$. We can thus treat this as a formal polynomial in one variable $X=(\mathbbm{1}\mathbbm{1}^{T}/K)$. Then products can be computed as $\prod_{i}\left[(1-\beta_{i})+\beta_{i}X\right]$ over the quotient ring $\mathbb{R}[X]/(X^{2}-X)$, since $X^{2}=X$. Functionally, this means you can instantiate a polynomial $(1-\beta_{i})+\beta_{i}X$ and repeatedly perform ordinary polynomial multiplication over $\mathbb{R}[X]$ for the $t<T$ timesteps. After each multiplication, the higher-order terms are reduced by $X^{2}=X$, leaving a polynomial of degree 1 where the $X$ term has coefficient given by the sum of all higher-order terms. This can be computed with the convenient np.polynomial module.

Similarly, the transition matrices for D3PM absorbing can be computed in closed form. Fundamentally, in each step, we transition to a [MASK] token with probability $\beta_{t}$ and stay the same with probability $1-\beta_{t}$. Since the [MASK] state is absorbing, after $t$ steps, the only operative quantity is the probability of not yet having transitioned to the [MASK] state, given by $\widetilde{\alpha_{t}}=\prod_{i=0}^{t}(1-\beta_{i})$. Hence for D3PM absorbing, $\overline{\bm{Q}}=\tilde{\alpha_{t}}I+(1-\widetilde{\alpha_{t}})\mathbbm{1}e^{T}_{m}$ where $e_{m}$ is a one-hot vector on the [MASK] token.

#### A.4.2 Matrix exponentials

In the matrix exponential case, we specify our transition matrices as

$$ $\displaystyle\bm{Q}_{t}$ $\displaystyle=\exp(\alpha_{t}\bm{R})=\sum_{n=0}^{\infty}\frac{\alpha_{t}^{n}}{n!}\bm{R}^{n},$ $\displaystyle\overline{\bm{Q}}_{t}$ $\displaystyle=\textstyle\exp\left(\left(\sum_{s\leq t}\alpha_{s}\right)\bm{R}\right),$ (16) $$

where $\bm{R}$ is a *transition rate matrix* and $\exp$ denotes the matrix exponential operation; the similar form for $\bm{Q}_{t}$ and $\overline{\bm{Q}}_{t}$ is a consequence of the “exponential of sums” property for commuting matrices.
For efficiency, we further assume that each of the $\alpha_{t}$ is an integer multiple $n_{t}\alpha_{\star}$ of some common factor $\alpha_{\star}$, and precompute matrices $\exp(2^{k}\alpha_{\star}\bm{R})$ for $0\leq k\leq\log_{2}(\overline{\alpha}_{T}/\alpha_{\star})$, where $\overline{\alpha}_{T}=\sum_{t<T}\alpha_{t}$, taking space $O(K^{2}\log(\overline{\alpha}_{T}/\alpha_{\star}))$. Then, to compute matrix-vector products with $\bm{Q}_{t}$ or $\overline{\bm{Q}}_{t}$, we can iteratively take products with a subset of these precomputed matrices based on the digits of a binary expansion of the desired multiple $n_{t}$ in time $O(K^{2}\log(\overline{\alpha}_{T}/\alpha_{\star}))$.(^4^44This is closely related to the well-known “exponentiation-by-squaring” technique.)

As long as $\bm{R}$ has non-positive off-diagonal entries and sums to zero along each row, the matrix exponential produces a valid transition matrix $\bm{Q}_{t}$; convergence to a specific stationary distribution can also be ensured by controlling the eigenvectors. In particular, if every column also sums to zero, the resulting $\bm{Q}_{t}$ will be doubly stochastic and will thus have a uniform stationary distribution.

We note that this parameterization can be viewed as a discretization of a continuous-time discrete-space Markov processes; we describe this connection in more detail in the following section.

### A.5 Continuous-time Markov process transition rates

Following , we define a continuous-time discrete-space Markov process as a collection of random variables $\{\bm{x}_{t}\}_{t>0}$ parameterized by $t\in\mathbb{R}^{+}$ and characterized by a Markov property ($\bm{x}_{t}\perp\bm{x}_{s}\mid\bm{x}_{\tau}$ if $t<\tau<s$), a transition probability matrix $\Pi(t)\in\mathbb{R}^{N\times N}$ where $N$ is the cardinality of $\bm{x}_{t}$, and a set of transition rates $\bm{\gamma}_{i}(t)$.

A conceptual way to understand these processes is to imagine a continuous Poisson process occurring in each state $i$ at rate $\bm{\gamma}_{i}(t)$ determining when a transition between states occurs. When a transition occurs (at time $t$), a Markov transition occurs between states $i$ and $j$ with probability $\Pi_{ij}(t)$. Many common stochastic processes fall into this family, including Poisson processes. Like in the case of stochastic differential equations (), we can derive a set of Kolomogorov equations (or Fokker-Planck equations in the continuous-state space case) that determine the marginal probability $\partial q_{ij}(\tau,t)$ of ending up in state $j$ at time $t$ having started in state $i$ at time $s$. The general form of the Kolmogorov forward equations is

$$ $\frac{\partial q_{ij}(\tau,t)}{\partial t}=-\bm{\gamma}_{k}(t)q_{i}(\tau,t)+\sum_{j}\bm{\gamma}_{j}(t)\Pi_{kj}(t)q_{ik}(t)$ $$

Now we can state and prove a theorem connecting continuous time Markov processes and matrix exponentials.

###### Theorem 1 .

Let $\{\bm{x}_{t}\}_{t\geq 0}$ be a discrete-space, continuous-time Markov process with (possibly time-dependent) transition probability matrix $\Pi(t)$ and transition rates $\bm{\gamma}_{i}(t)$. Then for a particle with an initial distribution $q(\bm{x}_{s})$ at time $s$, the probability of ending in state $j$ at time $t$ is

$$ $q(\bm{x}_{t}|\bm{x}_{s})=\exp\left(\int_{s}^{t}\operatorname{diag}(\bm{\gamma}(\tau))(\Pi(\tau)-I)\,d\tau\right)q(\bm{x}_{s})$ $$

where $\exp$ is the matrix exponential and we view $q(\bm{x}_{t})$ and $\bm{\gamma}(t)$ as vectors in $\mathbb{R}^{N}$.

###### Proof (sketch).

From the Kolmogorov equations for continuous-time Markov processes, we have the ODE

$$ $\frac{\partial q(\bm{x}_{t}|\bm{x}_{s})}{\partial t}=\operatorname{diag}(\bm{\gamma}(t))(\Pi(t)-I)q(\bm{x}_{t}|\bm{x}_{s})$ $$

where $\Pi(t)$ is the transition probability matrix. Solving this as a first-order ODE using integrating factors yields the desired equation.
∎

We note that, if $\Pi(t)=\Pi$ is independent of $t$ and $\bm{\gamma}(s)=\gamma(s)\mathbf{r}$ for some scalar function $\gamma:\mathbb{R}\to\mathbb{R}$ and vector $\mathbf{r}\in\mathbb{R}^{N}$, this simplifies to exactly our matrix exponential parameterization with

$$ $\mathbf{R}=\operatorname{diag}(\mathbf{r})(\Pi-I).$ $$

where we set

$$ $\alpha_{t}=\int_{t-1}^{t}\gamma(t)\,dt.$ $$

In other words, the $\alpha_{t}$ parameters in Equation [16](#A1.E16) correspond to a discretization of the cumulative transition rate of a continuous-time process.

###### Theorem 1 .

###### Proof (sketch).

### A.6 Continuous-limit of schedule from Sohl-Dickstein et al. [ 43 ]

Consider for example the schedule described by for Bernoulli variables $\beta_{t}=1/(T-t+1)$, i.e. the Bernoulli variable would stay the same with probability $1-\beta_{t}=(T-t)/(T-t+1)$ and transition with probability $\beta_{t}$. In this section, we show that a D3PM absorbing or D3PM uniform process with this schedule is exactly a discretization of a continuous-time jump process of the form described in Theorem [1](#Thmtheorem1).

We start by observing that both absorbing-state and uniform D3PM transition matrices can be expressed equivalently as matrix exponentials. In the uniform case, we have

$$ $Q_{t}=\exp(\alpha_{t}\mathbf{R}_{\text{unif}})=\exp\left(\alpha_{t}\left(\frac{1}{K}\mathbbm{1}\mathbbm{1}^{T}-I\right)\right)=\exp(-\alpha_{t})I+(1-\exp(-\alpha_{t}))\frac{1}{K}\mathbbm{1}\mathbbm{1}^{T},$ $$

and in the absorbing case we have

$$ $Q_{t}=\exp(\alpha_{t}\mathbf{R}_{\text{abs}})=\exp\left(\alpha_{t}\left(\mathbbm{1}\mathbf{e}_{m}^{T}-I\right)\right)=\exp(-\alpha_{t})I+(1-\exp(-\alpha_{t}))\mathbbm{1}\mathbf{e}_{m}^{T}.$ $$

In either case, by setting this equal to the explicit forms in Appendix [A.2](#A1.SS2), we obtain the relationship

$$ $\beta_{t}=1-\exp(-\alpha_{t})$ $$

where $\beta_{t}$ is defined as in Appendix [A.2](#A1.SS2), and $\alpha_{t}$ is the matrix exponential coefficient as used in the previous section.
Using the correspondence discussed in the previous section, we also know

$$ $\alpha_{t}=\int_{t-1}^{t}\gamma(s)\,ds$ $$

for the continuous-time transition rate function $\gamma(s)$. Defining $\beta_{t}=1/(T-t+1)$, we have

$$ $1-\beta_{t}=1-\frac{1}{(T-t+1)}=\frac{T-t}{T-t+1}=\exp\left(-\int_{t-1}^{t}\gamma(\tau)d\tau\right)$ $$

Denoting the anti-derivative $\int\gamma(t)=F(t)$, we have $\log(T-t)-\log(T-t+1)=-F(t)+F(t-1)$, so we can deduce $F(t)=-\log(T-t)$ (up to a constant offset). Taking a derivative then yields $\gamma(t)=1/(T-t)$, which has the same form as the original schedule but is now interpreted as a continuously-varying rate function instead of a probability (and is also shifted by 1 unit in time). Intuitively, we can interpret this as a schedule which assigns uniform probability of a transition occurring over the remaining time, but instead of dividing it between $T-t+1$ discrete steps, we divide it across a continuous interval of size $T-t$. We note that using larger values of $T$ is equivalent to performing a finer discretization on a scaled version of this continuous-time process.

### A.7 Mutual-information-based noise schedule

An important part of designing the forward process for a diffusion process is to specify the *noise schedule*: how much noise is added at each step $t$ such that after $T$ steps the process has (approximately) reached the stationary distribution of the transition matrix. Previous work on continuous-state diffusion models has focused on controlling the variance of the continuous noise added at each step, but in a discrete state space it is less obvious how to measure or control the level of noise added.

For uniform or absorbing-state transition matrices, once a single transition occurs, all information about the original data point is lost. In this case, the schedule introduced by is a natural choice, since it is designed to make this first transition for $t/T$ of the elements by time $t$.
However, when the transition matrix imposes additional structure on the transitions, such as for our token-embedding based transition matrix, it is not sufficient to perturb $t/T$ of the elements by time $t$, since the value at time $t$ may be highly correlated with the value at time $t-1$ even after a transition occurs; we thus explore using mutual information to quantify how much noise has been added.
Here we describe the mutual-information-based schedules in more detail. We focus on transition matrices that are parameterized as matrix exponentials, i.e. they have the form

$$ $\displaystyle\bm{Q}_{t}$ $\displaystyle=\exp(\alpha_{t}\bm{R})=\sum_{n=0}^{\infty}\frac{\alpha_{t}^{n}}{n!}\bm{R}^{n},$ $\displaystyle\overline{\bm{Q}}_{t}$ $\displaystyle=\textstyle\exp\left(\left(\sum_{s\leq t}\alpha_{s}\right)\bm{R}\right)=\textstyle\exp\left(\bar{\alpha}_{t}\bm{R}\right).$ $$

Inspired by the schedule introduced by , we consider setting our $\alpha_{t}$ such that $\frac{t}{T}$ of the information about $p(\bm{x}_{0})$ has been lost by time $t$. Our goal is to find exponents such that

$$ $\frac{t}{T}=1-\frac{I(\bm{x}_{t};\bm{x}_{0})}{H(\bm{x}_{0})}=\frac{H(\bm{x}_{0},\bm{x}_{t})-H(\bm{x}_{t})}{H(\bm{x}_{0})}=\frac{\sum_{\bm{x}_{0},\bm{x}_{t}}p(\bm{x}_{0})q(\bm{x}_{t}|\bm{x}_{0})\log\frac{q(\bm{x}_{t}|\bm{x}_{0})}{\sum_{\bm{x}_{0}^{\prime}}p(\bm{x}_{0}^{\prime})q(\bm{x}_{t}|\bm{x}_{0}^{\prime})}}{\sum_{\bm{x}_{0}}p(\bm{x}_{0})\log p(\bm{x}_{0})}$ (17) $$

where $H$ denotes the entropy of a random variable, and $p(\bm{x}_{0})$ denotes the distribution of a randomly chosen token in the data.

In practice, we estimate $p(\bm{x}_{0})$ by computing empirical frequencies over the training set, and compute the value of the right-hand side of [17](#A1.E17) for transition matrices $\exp(\bar{\alpha}\bm{R})$ with 256 geometrically-spaced exponents $\bar{\alpha}$ distributed in a large range (linear on a log scale between 1e-4 and 1e5). We then interpolate using a monotonic cubic spline to find the particular exponents $\bar{\alpha}_{t}$ that ensure the above property holds approximately, and round them so that they are all multiples of a common factor $\alpha_{\star}$ to ensure efficiency (as described in Appendix [A.4](#A1.SS4)). Finally, we set $\bm{Q}_{t}=\exp((\bar{\alpha}_{t}-\bar{\alpha}_{t-1})\bm{R})$.

It turns out that, for the specific case of absorbing-state diffusion with a [MASK] token, the mutual information schedule reduces to exactly the $(T-t+1)^{-1}$ schedule proposed by . To see this, let $m_{t}$ be the probability that a given value from time 0 has been replaced with [MASK] at time $t$. We note then that

$$ $\displaystyle H(\bm{x}_{t})$ $\displaystyle=\sum_{\bm{x}_{0}}(1-m_{t})p(\bm{x}_{0})\log\left((1-m_{t})p(\bm{x}_{0})\right)+m_{t}\log m_{t}$ $\displaystyle=(1-m_{t})\sum_{\bm{x}_{0}}p(\bm{x}_{0})\log p(\bm{x}_{0})+(1-m_{t})\log(1-m_{t})+m_{t}\log m_{t}$ $$

where we have used the fact that a mask token has zero probability under the data distribution. We also have the joint entropy

$$ $\displaystyle H(\bm{x}_{0},\bm{x}_{t})=\sum_{\bm{x}_{0}}p(\bm{x}_{0})\log p(\bm{x}_{0})+m_{t}\log m_{t}+(1-m_{t})\log(1-m_{t}).$ $$

We can then calculate

$$ $\displaystyle 1-\frac{I(\bm{x}_{t};\bm{x}_{0})}{H(\bm{x}_{0})}$ $\displaystyle=\frac{H(\bm{x}_{0},\bm{x}_{t})-H(\bm{x}_{t})}{H(\bm{x}_{0})}$ $\displaystyle=\frac{\sum_{\bm{x}_{0}}p(\bm{x}_{0})\log p(\bm{x}_{0})+m_{t}\log m_{t}+(1-m_{t})\log(1-m_{t})}{\sum_{\bm{x}_{0}}p(\bm{x}_{0})\log p(\bm{x}_{0})}$ $\displaystyle\qquad\qquad-\frac{(1-m)\sum_{\bm{x}_{0}}p(\bm{x}_{0})\log p(\bm{x}_{0})+(1-m_{t})\log(1-m_{t})+m_{t}\log m_{t}}{\sum_{\bm{x}_{0}}p(\bm{x}_{0})\log p(\bm{x}_{0})}$ $\displaystyle=\frac{m_{t}\sum_{\bm{x}_{0}}p(\bm{x}_{0})\log p(\bm{x}_{0})}{\sum_{\bm{x}_{0}}p(\bm{x}_{0})\log p(\bm{x}_{0})}=m_{t}.$ $$

It follows that the mutual information schedule for masks is one that ensures $m_{t}=q(\bm{x}_{t}=\text{[MASK]}|\bm{x}_{0})=\frac{t}{T}$. But this is exactly the $(T-t+1)^{-1}$ schedule. To see this, let $\beta_{t}$ be the probability that a non-mask token becomes a mask token at time $t$, and note that $m_{t}=1-\prod_{s=1}^{t}(1-\beta_{s})$. Thus,

$$ $\displaystyle\beta_{t}=1-\frac{1-m_{t}}{1-m_{t-1}}=1-\frac{1-\frac{t}{T}}{1-\frac{t-1}{T}}=1-\frac{T-t}{T-t+1}=\frac{(T-t+1)-(T-t)}{T-t+1}=\frac{1}{T-t+1}$ $$

as desired.

Interestingly, although the $(T-t+1)^{-1}$ schedule was designed for the case of a uniform transition matrix (an used for this purpose by and ), the $(T-t+1)^{-1}$ schedule is NOT in general identical to the mutual information schedule in that setting. We leave further investigation of these schedules to future work.

### A.8 Parameterizing the reverse process with a discretized truncated logistic distribution

For ordinal data such as images, we can instill an ordinal inductive bias
in the logits of $\widetilde{p}_{\theta}(\widetilde{\bm{x}}_{0}|\bm{x}_{t})$ by modeling them using a discretization of a distribution on real-valued numbers. In this paper we choose the underlying continuous distribution to be a truncated logistic distribution. The code below shows how we compute the logits for $\widetilde{p}_{\theta}(\widetilde{\bm{x}}_{0}|\bm{x}_{t})$, given a location/mean and a log scale that were predicted by a neural network $\mathrm{nn}_{\theta}$.

## Appendix B Experiments

### B.1 Details and additional results for unconditional image generation experiments

We follow the same training and evaluation setup as used by . For completeness we repeat these settings here. The model architecture is based on the backbone of a PixelCNN++ architecture: a U-Net based on a Wide ResNet with weight normalization layers replaced by group normalization layers . The model has four feature map resolutions and two convolutional residual blocks for each resolution level. At the $16\times 16$ resolution level a self-attention block is placed between the convolutional blocks . The time step $t$ is included in the neural net through a Transformer sinusoidal position embedding in each residual block.
Furthermore, we use the same hyperparameters and augmentation settings as in without tuning them: the dropout rate is set to 0.1; we use a learning rate of $2\times 10^{-4}$ with the Adam optimizer with standard settings, a batch size of 128; for evaluation we use an exponential moving average (EMA) for the model parameters with a decay factor of $0.9999$; and finally, we use random horizontal flips as augmentation during training.

We built our implementation of D3PMs for images based on
a re-implementation of the DDPM model in JAX and Flax , with the same settings as those mentioned above. This re-implementation has been verified to produce similar results as those reported in .
For the D3PM models for which the logits of $\widetilde{p}_{\theta}(\widetilde{\bm{x}}_{0}|\bm{x}_{t})=\mathrm{Cat}(\widetilde{\bm{x}}_{0}|\bm{p}_{\theta})$ are modeled directly as the output of a neural network, we model them as $\mathrm{logits}=\mathrm{nn}_{\theta}(\mathrm{normalize}(\bm{x}^{\mathrm{int}}_{t}))+\bm{x}^{\mathrm{one-hot}}_{t}$, where $\bm{x}^{\mathrm{int}}_{t}$ and $\bm{x}^{\mathrm{one-hot}}_{t}$ denote integer and one-hot representations of $\bm{x}_{t}$ respectively. The function $\mathrm{normalize}(\bm{x}^{\mathrm{int}}_{t})$ maps the integer values $\{0,...,K-1\}$ to the interval $[-1,1]$. For the case where the logits are predicted from a truncated distretized logistic distribution, as discussed in Section [A.8](#A1.SS8), the neural network outputs a log scale $\log\bm{s}$ and the mean $\bm{\mu}$ of the underlying logistic distribution: $[\log\bm{s},\bm{\mu}^{\prime}]=\mathrm{nn}_{\theta}(\mathrm{normalize}(\bm{x}^{\mathrm{int}}_{t}))$, $\bm{\mu}=\tanh(\mathrm{normalize}(\bm{x}^{\mathrm{int}}_{t})+\bm{\mu}^{\prime})$.
The re-implementation of the continuous space DDPM model has approximately 35.7M parameters, which is the same number of parameters as that of the CIFAR-10 model that we loaded from the officially released checkpoint by the authors of .(^5^55Code and checkpoints for the DDPM models from are available at [https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion).)
Our D3PM models that output logits directly have around 36.6M parameters, while the model that parameterizes the logits through a discretized truncated logistic distribution (D3PM Gauss + logistic) has around 35.7M parameters.

We trained all our models for 1.5M steps on TPUv2 accelerators with a $4\times 4$ topology.
Our Inception and FID scores were computed on 50000 samples with the Inception-v3 model . We have included averages and standard deviations over models trained with 5 different seeds.

Figure: Figure 7: Samples from the D3PM uniform model trained with $L_{\mathrm{vb}}$ (top), the D3PM absorb model trained with $L_{\lambda=0.001}$ (middle), and the D3PM Gauss + logistic model trained with $L_{\lambda=0.001}$ (bottom). These samples were not cherry picked.
Refer to caption: /html/2107.03006/assets/figures/cifar10_samples_uniform_row.png

##### Noise schedule settings

For the D3PM Gauss models with discretized Gaussian transition matrices as described in Appendix [A.2.3](#A1.SS2.SSS3), we use the same linear schedule for the $\beta_{t}$’s as in : $\beta_{t}$ is linearly increased from $1\times 10^{-4}$ to $0.02$. We did not explore any other noise schedules for D3PM Gauss models.
For the D3PM uniform model (see Section [A.2.1](#A1.SS2.SSS1)) we experimented with a linear schedule for $\beta_{t}$ (linearly increasing from $0.02$ to $1$) and the cosine schedule as suggested by . Table [4](#A2.T4) shows that the D3PM uniform model with a cosine schedule produces much better results than the same model with a linear $\beta_{t}$ schedule. For the D3PM absorbing model (see Section [A.2.2](#A1.SS2.SSS2)) the absorbing state is the gray pixel, corresponding to the RGB values (128, 128, 128). For these models we used a schedule that corresponds to increasing the probability of being in the absorbing state linearly over time: $\beta_{t}=(T-t+1)^{-1}$. This schedule was also proposed in for diffusion with binary random variables, which has a uniform stationary distribution as opposed to the stationary distribution with all the mass on the absorbing state.

##### Samples

Additional samples from the D3PM uniform model trained on $L_{\mathrm{vb}}$, the D3PM absorb model trained on $L_{\lambda=0.001}$, and the D3PM Gauss + logistic model trained on $L_{\lambda=0.001}$ can be bound in Figure [7](#A2.F7).

**Table 4: Quantitative results on the image dataset CIFAR-10 for D3PM uniform models trained with $L_{\mathrm{vb}}$. The cosine noise schedule for the uniform D3PM model was suggested by . The linear schedule corresponds to linearly increasing $\beta_{t}$ from $0.02$ to $1$. Results displayed for models trained with 3 (linear) and 4 (cosine) seeds.**
| Model | $\beta_{t}$ schedule | IS ($\uparrow$) | FID ($\downarrow$) | NLL ($\downarrow$) |
| --- | --- | --- | --- | --- |
| D3PM uniform | linear | $4.44\pm 0.05$ | $79.86\pm 1.64$ | $\leq 4.99\pm 0.03$ |
| D3PM uniform | cosine | $5.99\pm 0.14$ | $51.27\pm 2.15$ | $\leq 5.08\pm 0.02$ |

### B.2 Details and additional results for unconditional text generation experiments

Our experiments using text8 and LM1B were performed with a standard transformer encoder following the T5 architecture with 12 layers and 70 million parameters (12 heads, mlp dim 3072, qkv dim 768). All models were trained for 1 million steps with batch size 512 on the TPUv2 or TPUv3 platform. Our code is implemented in JAX and Flax . For our experiments, we used learning rate $5\times 10^{-4}$ with a 10000 step learning rate warmup and inverse sqrt decay. For text8, we used a standard 90000000/5000000/500000 train-test-validation split with sequences of length 256. For LM1B, we used the standard test-train split from TFDS with 30,301,028 examples in the training set and 306,688 in the test set. For text8, no preprocessing is performed, and training is performed on random crops of the entire concatenated, lower-cased training set. For LM1B, training is performed on sequences of length 128 sampled by packing sequences from the training corpus, including an EOS token. Perplexities are reported relative to the actual number of English-language words in the test set (including an EOS token predicted by the model).

Our autoregressive transformer baseline was a standard transformer decoder with the same basic architecture (but including causal masking, as is standard for autoregressive models) with the same number of parameters.

Table [5](#A2.T5) contains additional comparisons of hybrid losses. We found that the hybrid loss $L_{\lambda=0.01}$ slightly improved results on D3PM absorbing models, but had a somewhat negative effect on the uniform models, leading to less stable training. All models were trained on 1000 step diffusion processes, but we found very little improvement between 1000 and 256 steps when evaluating a trained model by skipping steps. For all figures, steps were skipped evenly (except possibly for the last step if the number of evaluation steps did not divide $1000$). We found both the cosine and mutual information schedules worked well for uniform diffusion. We used the cosine variant introduced by , i.e.

$$ $\displaystyle f(t)=\cos\left(\frac{t/T+s}{1+s}+\frac{\pi}{2}\right)\qquad\beta(t)=1-\frac{f(t+1)}{f(t)}$ (18) $$

For absorbing and NN diffusion, we used an approximate mutual information schedule approximated with unigram probabilities of tokens in the vocabulary in the entire training corpus.

Figure [8](#A2.F8) shows scaling of bits/dim on text8 for 3 D3PM models with the number of inference steps. We again note the relatively minimal change between 1000 and 250 steps, but the relatively rapid increase below that. Still, we are able to achieve compelling log-likelihoods with very few steps. Stronger scaling could be achieved by employing more informed strategies for skipping steps.

#### B.2.1 Additional tables and figures for text8

**Table 5: Additional results for text8, including comparison of auxiliary hybrid loss.**
| Model | Model steps | NLL (bits/char) ($\downarrow$) |
| --- | --- | --- |
| D3PM uniform (ours) ($L_{\lambda=0.01}$) | $1000$ | $\leq 1.91$ |
| D3PM uniform (ours) ($L_{\mathrm{vb}}$) | $1000$ | $\leq 1.61$ |
| D3PM absorbing ($L_{\lambda=0.01}$) (ours) | $1000$ | $\leq 1.44$ |
| D3PM absorbing ($L_{\mathrm{vb}}$) (ours) | $1000$ | $\leq 1.47$ |
| D3PM absorbing + NN ($L_{\lambda=0.01}$) (ours) | $1000$ | $\leq 1.53$ |
| D3PM uniform (ours) | $50$ | $\leq 1.7$ |
| D3PM NN ($L_{\mathrm{vb}}$) (ours) | $50$ | $\leq 1.62$ |
| D3PM absorbing ($L_{\lambda=0.01}$) (ours) | $50$ | $\leq 1.53$ |

**Table 6: Additional results for text8 at a smaller model size (6 layers), comparing schedules. All at 1000 steps.**
| Model | Schedule | NLL (bits/char) ($\downarrow$) |
| --- | --- | --- |
| D3PM uniform | ($1/(T-t+1)$ schedule) | $\leq 2.37$ |
| D3PM uniform | cosine | $\leq 1.73$ |
| D3PM uniform | mutual info | $\leq 1.74$ |

Figure: Figure 8: Scaling of text8 bits/dim with inference steps. “mask” denotes D3PM absorbing.
Refer to caption: /html/2107.03006/assets/figures/text8_elbo_scaling.png

Figure: Figure 9: Inference time for a D3PM absorbing model (‘mask’) on text8 in seconds as a function of iterations, compared to an autoregressive model.
Refer to caption: /html/2107.03006/assets/figures/inference-time.png

#### B.2.2 Additional tables and figures for LM1B

**Table 7: Sample times for LM1B. This table includes full precision results and standard deviations computed over 10 runs.**
| Metric: | Sample time (s) ($\downarrow$) |  |  |
| --- | --- | --- | --- |
| inference steps: | $1000$ | $128$ | $64$ |
| D3PM uniform | 1.8161 $\pm$ 0.0002 | 0.2120 $\pm$ 0.0005 | 0.0831 $\pm$ 0.0002 |
| D3PM NN | 21.29 $\pm$ 0.03 | 6.6861 $\pm$ 0.0009 | 5.8786 $\pm$ 0.0008 |
| D3PM absorbing | 1.9049 $\pm$ 0.0005 | 0.1983 $\pm$ 0.0003 | 0.1017 $\pm$ 0.0002 |
| Transformer | - | 0.26 $\pm$ 0.03 | - |

### B.3 Additional uncurated generation examples from various models

Figure: Figure 10: Using an absorbing-state D3PM model (trained on LM1B with 128 denoising steps) to complete test-set examples at different noise levels. We corrupt the example using $q(\bm{x}_{t}|\bm{x}_{0})$, then iteratively sample from $p_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})$ to reconstruct. Mask token shown as “[M]”.