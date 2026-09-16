---
arxiv_id: "2310.16834"
title: "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution"
year: 2024
source: arxiv2md
---

## Abstract

Abstract Despite their groundbreaking performance for many generative modeling tasks, diffusion models have fallen short on discrete data domains such as natural language. Crucially, standard diffusion models rely on the well-established theory of score matching, but efforts to generalize this to discrete structures have not yielded the same empirical gains. In this work, we bridge this gap by proposing score entropy, a novel loss that naturally extends score matching to discrete spaces, integrates seamlessly to build discrete diffusion models, and significantly boosts performance. Experimentally, we test our Score Entropy Discrete Diffusion models (SEDD) on standard language modeling tasks. For comparable model sizes, SEDD beats existing language diffusion paradigms (reducing perplexity by 25 25 25 25 - 75 75 75 75 %) and is competitive with autoregressive models, in particular outperforming GPT-2. Furthermore, compared to autoregressive mdoels, SEDD generates faithful text without requiring distribution annealing techniques like temperature scaling (around 6 6 6 6 - 8 × 8\times 8 × better generative perplexity than un-annealed GPT-2), can trade compute and quality (similar quality with 32 × 32\times 32 × fewer network evaluations), and enables controllable infilling (matching nucleus sampling quality while enabling other strategies besides left to right prompting).

## 1 Introduction

Many recent advances in deep learning have centered around generative modeling. Here, a model learns how to generate novel samples from unstructured data. With the powerful capabilities of modern neural networks, these “generative AI” systems have developed unparalleled capabilities, such as creating images given only text and answering complex questions .

The crucial part for any deep generative model is the probabilistic modeling technique. For discrete data such as natural language, autoregressive modeling –arguably the simplest modeling type since it derives from the probabilistic chain rule–has remained the only competitive method for decades. Although modern autoregressive transformers have produced stunning results , there are limits. For example, the sequential sampling of tokens is slow, hard to control, and often degrades without distribution annealing techniques like nucleus sampling .

To alleviate these issues, researchers have sought alternative approaches to generating text data. In particular, inspired by their success in the image domain, many works have extended diffusion models to language domains . Yet, despite considerable effort, no such approach yet rivals autoregressive modeling, as they are not competitive on likelihoods, are slower to sample from, and do not generate comparable samples without resorting to heavy annealing and empirical alterations.

In our work, we challenge the longstanding dominance of autoregressive models by introducing Score Entropy Discrete Diffusion models (SEDD). SEDD parameterizes a reverse discrete diffusion process using the ratios of the data distribution. These are learned using score entropy, a novel loss that is analogous to score matching for standard diffusion models and results in several empirical benefits(^†^†We open source our code at [github.com/louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)):

- 1.
On core language modeling tasks, SEDD outperforms all existing language diffusion models by large margins and is competitive with autoregressive models of the same size (beating GPT-2 on its zero-shot perplexity tasks ).
- 2.
SEDD generates high quality unconditional samples and enables one to naturally trade off compute for quality. When measuring the generative perplexity (given by large models) of unconditional and un-annealed samples from similarly sized models, SEDD beats GPT-2 by $6$-$8\times$ and can match performance using $32\times$ fewer function evaluations.
- 3.
By directly parameterizing probability ratios, SEDD is highly controllable. In particular, one can prompt SEDD from arbitrary positions without specialized training. For both standard (left to right) and infilling, SEDD outperforms language diffusion models and is comparable with autoregressive models with nucleus sampling (as measured by MAUVE score ).

## 2 Preliminaries

### 2.1 Discrete Diffusion Processes

We will be modeling probability distributions over a finite support $\mathcal{X}=\{1,\dots,N\}$. As the support is discrete, note that our probability distributions can be represented by probability mass vectors $p\in\mathbb{R}^{N}$ that are positive and sum to $1$. To define a discrete diffusion process, we evolve a family of distributions $p_{t}\in\mathbb{R}^{N}$ according to the a continuous time Markov process given by a linear ordinary differential equation :

$$ $\frac{dp_{t}}{dt}=Q_{t}p_{t}\quad p_{0}\approx p_{\rm data}$ (1) $$

Here, $Q_{t}$ are the diffusion matrices $\mathbb{R}^{N\times N}$ and have non-negative non-diagonal entries and columns which sum to zero (so that the rate $\frac{dp_{t}}{dt}$ sums to 0 0, meaning $p_{t}$ does not gain or lose total mass). Generally, $Q_{t}$ are simple (e.g. a simple scalar factor $Q_{t}=\sigma(t)Q$) so $p_{t}$ approaches a limiting distribution $p_{\rm base}$ as $t\to\infty$.

One can simulate this process by taking small $\Delta t$ Euler steps and randomly sampling the resulting transitions. In particular, the samples are defined by transition densities which come from the columns of $Q_{t}$:

$$ $p(x_{t+\Delta t}=y|x_{t}=x)=\delta_{xy}+Q_{t}(y,x)\Delta t+O(\Delta t^{2})$ (2) $$

Finally, this process has a well known reversal given by another diffusion matrix $\overline{Q}_{t}$:

$$ $\frac{dp_{T-t}}{dt}=\overline{Q}_{T-t}p_{T-t}\quad\overline{Q}_{t}(y,x)=\frac{ p_{t}(y)}{p_{t}(x)}Q_{t}(x,y)\\ \overline{Q}_{t}(x,x)=-\sum_{y\neq x}\overline{Q}_{t}(y,x)$ (3) $$

This reverse process is analogous to the time reversal for typical diffusion processes on $\mathbb{R}^{n}$, with the ratios $\frac{p_{t}(y)}{p_{t}(x)}$ (which are collectively known as the concrete score ) generalizing the typical score function $\nabla_{x}\log p_{t}$ (^1^11The gradient operator for discrete structures is (up to some scaling) defined for pairs $x\neq y$ by $\nabla f(xy):=f(y)-f(x)$. The score function would generalize to the normalized gradients $\frac{\nabla p(xy)}{p(x)}=\frac{p(y)}{p(x)}-1$.)

### 2.2 Discrete Diffusion Models

The goal of a discrete diffusion model is to construct the aforementioned reverse process by learning the ratios $\frac{p_{t}(y)}{p_{t}(x)}$. Unlike the continuous diffusion case, which has settled around (up to minor scaling variations) the theoretical framework given by score matching , there currently exist many competing methods for learning discrete diffusion models. In particular, these tend to produce mixed empirical results, which spurs the need for a reexamination.

Mean Prediction. Instead of directly parameterizing the ratios $\frac{p_{t}(y)}{p_{t}(x)}$, instead follow a strategy of to learn the reverse density $p_{0|t}$. This actually recovers the ratios $\frac{p_{t}(y)}{p_{t}(x)}$ in a roundabout way (as shown in our Theorem 4.2), but comes with several drawbacks. First, learning $p_{0|t}$ is inherently harder since it is a density (as opposed to a general value). Furthermore, the objective breaks down in continuous time and must be approximated . As a result, this framework largely underperforms empirically.

Ratio Matching. Originally introduced in and augmented in , ratio matching learns the marginal probabilities of each dimension with maximum likelihood training. However, the resulting setup departs from standard score matching and requires specialized and expensive network architectures . As such, this tends to perform worse than mean prediction.

Concrete Score Matching. generalizes the standard Fisher divergence in score matching, learning $s_{\theta}(x,t)\approx\begin{bmatrix}\frac{p_{t}(y)}{p_{t}(x)}\end{bmatrix}_{y
\neq x}$ with concrete score matching:

$$ $\mathcal{L}_{\rm CSM}=\frac{1}{2}\mathbb{E}_{x\sim p_{t}}\left[\sum_{y\neq x} \left(s_{\theta}(x_{t},t)_{y}-\frac{p_{t}(y)}{p_{t}(x)}\right)^{2}\right]$ (4) $$

Unfortunately, the $\ell^{2}$ loss is incompatible with the fact that $\frac{p_{t}(y)}{p_{t}(x)}$ must be positive. In particular, this does not sufficiently penalize negative or zero values, leading to divergent behavior. Although theoretically promising, Concrete Score Matching struggles (as seen in Appendix D).

## 3 Score Entropy Discrete Diffusion Models

In this section, we introduce score entropy. Similar to concrete score matching, we learn the collected concrete score $s_{\theta}(x,t)\approx\begin{bmatrix}\frac{p_{t}(y)}{p_{t}(x)}\end{bmatrix}_{y
\neq x}$ ($s_{\theta}:\mathcal{X}\times\mathbb{R}\to\mathbb{R}^{|\mathcal{X}}|$). We design the score entropy loss to incorporate the fact that these ratios are positive and evolve under a discrete diffusion.

###### Definition 3.1 .

The score entropy $\mathcal{L}_{\rm SE}$ for a distribution $p$, weights $w_{xy}\geq 0$ and a score network $s_{\theta}(x)_{y}$ is

$$ $\mathbb{E}_{x\sim p}\left[\sum_{y\neq x}w_{xy}\left(s_{\theta}(x)_{y}-\frac{p( y)}{p(x)}\log s_{\theta}(x)_{y}+K\left(\frac{p(y)}{p(x)}\right)\right)\right]$ (5) $$

where $K(a)=a(\log a-1)$ is a normalizing constant function that ensures that $\mathcal{L}_{\rm SE}\geq 0$.

###### Remark .

Instead of building off of Fisher divergences, score entropy builds off of the Bregman divergence $D_{F}\left(s(x)_{y},\frac{p(y)}{p(x)}\right)$ when $F=-\log$ is the convex function. As such, score entropy is non-negative, symmetric, and convex. It also generalizes standard cross entropy to general positive values (instead of simplex-valued probabilities), inspiring the name. The weights $w_{xy}$ are used primarily when combining score entropy with diffusion models.

While this expression is more complex than the standard score matching variants, it satisfies several desiderata for a discrete diffusion training objective:

###### Definition 3.1 .

###### Remark .

### 3.1 Score Entropy Properties

First, score entropy is a suitable loss function that recovers the ground truth concrete score.

###### Proposition 3.2 (Consistency of Score Entropy) .

Suppose $p$ is fully supported and $w_{xy}>0$. As the number of samples and model capacity approaches $\infty$, the optimal $\theta^{*}$ that minimizes Equation 5 satisfies $s_{\theta^{*}}(x)_{y}=\frac{p(y)}{p(x)}$ for all pairs $x,y$ Furthermore, $\mathcal{L}_{\rm SE}$ will be 0 0 at $\theta^{*}$.

Second, score entropy directly improves upon concrete score matching by rescaling problematic gradients. For the weights $w_{xy}=1$, $\nabla_{s_{\theta}(x)_{y}}\mathcal{L}_{\rm SE}=\frac{1}{s_{\theta}(x)_{y}}
\nabla_{s_{\theta}(x)_{y}}\mathcal{L}_{\rm CSM}$, so the gradient signals for each pair $(x,y)$ are scaled by a factor of $s_{\theta}(x)_{y}$ as a normalization component. As such, this forms a natural log-barrier which keeps our $s_{\theta}\geq 0$.

Third, similar to concrete score matching, score entropy can be made computationally tractable by removing the unknown $\frac{p(y)}{p(x)}$ term. There are two alternative forms, the first of which is analogous to the implicit score matching loss :

###### Proposition 3.3 (Implicit Score Entropy) .

$\mathcal{L}_{\rm SE}$ is equal up to a constant independent of $\theta$ to the implicit score entropy

$$ $\mathcal{L}_{\rm ISE}=\mathbb{E}_{x\sim p}\left[\sum_{y\neq x}w_{xy}s_{\theta} (x)_{y}-w_{yx}\log s_{\theta}(y)_{x}\right]$ (6) $$

Unfortunately, a Monte Carlo estimate would require sampling an $x$ and evaluating $s_{\theta}(y)_{x}$ for all other $y$. For high dimensions, this is intractable, which means we have to sample $y$ uniformly, but this introduces additional variance analogous to that introduced by the Hutchinson trace estimator for sliced score matching . As a result, implicit score entropy is impractical for large-scale tasks. Instead, we work a denoising score matching loss variant of score entropy:

###### Theorem 3.4 (Denoising Score Entropy) .

Suppose $p$ is a perturbation of a base density $p_{0}$ by a transition kernel $p(\cdot|\cdot)$, ie $p(x)=\sum_{x_{0}}p(x|x_{0})p_{0}(x_{0})$. The score entropy $\mathcal{L}_{\rm SE}$ is equivalent (up to a constant independent of $\theta$) to the denoising score entropy $\mathcal{L}_{\rm DSE}$ is

$$ $\underset{\begin{subarray}{c}x_{0}\sim p_{0}\\ x\sim p(\cdot|x_{0})\end{subarray}}{\mathbb{E}}\left[\sum_{y\neq x}w_{xy}\left (s_{\theta}(x)_{y}-\frac{p(y|x_{0})}{p(x|x_{0})}\log s_{\theta}(x)_{y}\right) \right]\\$ (7) $$

$\mathcal{L}_{\rm DSE}$ is scalable since Monte Carlo sampling only requires the evaluation of one $s_{\theta}(x)$, which gives us all $s_{\theta}(x)_{y}$, and the variance introduced by $x_{0}$ is manageable. Additionally, it is particularly appealing for discrete diffusion since the intermediate $p_{t}$ are all perturbations of the base density $p_{0}$ (resulting from Equations 1, 2), enabling us to train with $\mathcal{L}_{\rm DSE}$ using the diffusion transition densities $p_{t|0}(\cdot|x_{0})$ (which we can make tractable).

###### Proposition 3.2 (Consistency of Score Entropy) .

###### Proposition 3.3 (Implicit Score Entropy) .

###### Theorem 3.4 (Denoising Score Entropy) .

### 3.2 Likelihood Bound For Score Entropy Discrete Diffusion

Fourth, the score entropy can be used to define an ELBO for likelihood-based training and evaluation.

###### Definition 3.5 .

For our time dependent score network $s_{\theta}(\cdot,t)$, the parameterized reverse matrix is $\overline{Q}_{t}^{\theta}(y,x)=\begin{cases}s_{\theta}(x,t)_{y}Q_{t}(x,y)&x
\neq y\\
-\sum_{z\neq x}\overline{Q}_{t}^{\theta}(z,y)&x=y\end{cases}$ found by replacing the ground truth scores in Equation 3. Our parameterized densities $p_{t}^{\theta}$ thus satisfy the following differential equation:

$$ $\frac{dp_{T-t}^{\theta}}{dt}=\overline{Q}_{T-t}^{\theta}p_{T-t}^{\theta}\quad p _{T}^{\theta}=p_{\rm base}\approx p_{T}$ (8) $$

The log likelihood of data points can be bounded using an ELBO based off of Dynkin’s formula , which was derived for discrete diffusion models in . Interestingly, this takes the form of our denoising score entropy loss weighted by the forward diffusion:

###### Theorem 3.6 (Likelihood Training and Evaluation) .

For the diffusion and forward probabilities defined above,

$$ $-\log p_{0}^{\theta}(x_{0})\leq\mathcal{L}_{\rm DWDSE}(x_{0})+D_{KL}(p_{T|0}( \cdot|x_{0})\parallel p_{\rm base})$ (9) $$

where $\mathcal{L}_{\rm DWDSE}(x_{0})$ is the diffusion weighted denoising score entropy for data point $x_{0}$

$$ $\int_{0}^{T}\mathbb{E}_{x_{t}\sim p_{t|0}(\cdot|x_{0})}\sum_{y\neq x_{t}}Q_{t} (x_{t},y)\Bigg{(}s_{\theta}(x_{t},t)_{y}-\\ \frac{p_{t|0}(y|x_{0})}{p_{t|0}(x_{t}|x_{0})}\log s_{\theta}(x_{t},t)_{y}+K \left(\frac{p_{t|0}(y|x_{0})}{p_{t|0}(x_{t}|x_{0})}\right)\Bigg{)}dt$ (10) $$

Crucially, this result allows us to directly models based on their likelihood values (and the related perplexity scores), the core metric for language modeling tasks. In particular, we can train and evaluate an upper bound.

###### Remark .

The DWDSE (and the implicit version) can be derived from the general framework of assuming a concrete score parameterization. In particular, the implicit version coincides with the likelihood loss introduced in .

###### Definition 3.5 .

###### Theorem 3.6 (Likelihood Training and Evaluation) .

###### Remark .

### 3.3 Practical Implementation

Fifth, score entropy can be scaled to high dimensional tasks.

In practice, our state factorizes into sequences $\mathcal{X}=\{1,\dots,n\}^{d}$ to form sequences $\mathbf{x}=x^{1}\dots x^{d}$ (e.g. sequences of tokens or image pixel values). As a general $Q_{t}$ would be of exponential size, we instead choose a sparse structured matrix that perturbs tokens independently with a matrix $Q_{t}^{\rm tok}$. In particular, the nonzero entries of $Q_{t}$ are given by

$$ $Q_{t}(x^{1}\dots x^{i}\dots x^{d},x^{1}\dots\widehat{x}^{i}\dots x^{d})=Q_{t}^ {\rm tok}(x^{i},\widehat{x}^{i})$ (11) $$

Since $\mathcal{L}_{\rm DWDSE}$ weights the loss by $Q_{t}(x,y)$, this token level transition $Q_{t}$ renders most ratios irrelevant. In particular, we only need to model all ratios between sequences with Hamming distnace $1$, so we can build our score network $s_{\theta}(\cdot,t):\{1,\dots,n\}^{d}\to\mathbb{R}^{d\times n}$ as a seq-to-seq map:

$$ $(s_{\theta}(x^{1}\dots x^{i}\dots x^{d},t))_{i,\widehat{x}^{i}}\approx\frac{p_ {t}(x^{1}\dots\widehat{x}^{i}\dots x^{d})}{p_{t}(x^{1}\dots x^{i}\dots x^{d})}$ (12) $$

To fully compute $\mathcal{L}_{\rm DWDSE}$, we just need to calculate the forward transition $p_{t|0}^{\rm seq}(\cdot|\cdot)$. Luckily, this decomposes as each token is perturbed independently:

$$ $p_{t|0}^{\rm seq}(\mathbf{\widehat{x}}|\mathbf{x})=\prod_{i=1}^{d}p_{t|0}^{\rm tok }(\widehat{x}^{i}|x^{i})$ (13) $$

For each $p_{t|0}^{\rm tok}(\cdot|\cdot)$, we employ the previously discussed strategy and set $Q_{t}^{\rm tok}=\sigma(t)Q^{\rm tok}$ for a noise level $\sigma$ and a fixed transition $Q^{\rm tok}$. This avoids numerical integration as, if we define $\overline{\sigma}(t)$ as the cumulative noise $\int_{0}^{t}\sigma(s)ds$, we have:

$$ $\displaystyle p_{t|0}^{\rm tok}(\cdot|x)=x\text{-th column of }\exp\left( \overline{\sigma}(t)Q^{\rm tok}\right)$ (14) $$

There are some practical consequences that render most $Q^{\rm tok}$ unusable for large scale experiments (e.g. for GPT-2 tasks, $n=50257$). In particular, one is not able to store all edge weights $Q_{\rm tok}(i,j)$ since this takes around $20$ GB of GPU memory and is extremely slow to access. Furthermore, one must be able to compute the columns $\exp(\overline{\sigma}(t)\cdot Q^{\rm tok})$ to get the transition ratios, but this must avoid matrix-matrix multiplication again can’t be stored in memory.

To sidestep these issues, we follow prior work and use two standard matrices with special structures. They arise, respectively, from considering a fully connected graph structure and from introducing a MASK absorbing state (similar to the BERT language modeling paradigm ):

$$ $\displaystyle Q^{\rm uniform}=\begin{bmatrix}1-N&1&\cdots&1\\ 1&1-N&\cdots&1\\ \vdots&\vdots&\ddots&\vdots\\ 1&1&\cdots&1-N\end{bmatrix}$ (15) $\displaystyle Q^{\rm absorb}=\begin{bmatrix}-1&0&\cdots&0&0\\ 0&-1&\cdots&0&0\\ \vdots&\vdots&\ddots&\vdots&\vdots\\ 0&0&\cdots&-1&0\\ 1&1&\cdots&1&0\end{bmatrix}$ (16) $$

With such a structured $Q$, one can quickly and cheaply compute all values in $\mathcal{L}_{\rm DWDSE}$. As such, our training iteration is about as fast and uses a similar amount of memory as standard autoregressive training. In particular, our training algorithm is given in Algorithm 1.

## 4 Simulating Reverse Diffusion with Concrete Scores

Given our scores $s_{\theta}$, we now derive various strategies for simulating a path $\mathbf{x}_{t}=x_{t}^{1}x_{t}^{2}\dots x_{t}^{d}\sim p_{t}$ of the reverse diffusion process. Notably, the additional information that we gain from $s_{\theta}$ being an approximate ratio of $p_{t}$ can be used to enhance the sampling process.

### 4.1 Time-Reversal Strategies

To simulate the diffusion in Definition 3.5, one may be tempted to use the Euler strategy from Equation 2. However, as noted in , this is inefficient because the structure of $Q_{t}^{\rm seq}$ only allows one position to be modified per step. Instead, a natural alternative has been to use $\tau$-leaping , which performs an Euler step at each position simultaneously. In particular, given a sequence $\mathbf{x}_{t}$, we construct $\mathbf{x}_{t-\Delta t}$ by sampling each token $x_{t-\Delta t}^{i}$ (independently) from the corresponding probability

$$ $\delta_{x_{t}^{i}}(x_{t-\Delta t}^{i})+\Delta tQ_{t}^{\rm tok}(x_{t}^{i},x_{t- \Delta t}^{i})s_{\theta}(\mathbf{x}_{t},t)_{i,x_{t-\Delta t}^{i}}$ (17) $$

While $\tau$-leaping is a viable simulation strategy, it is agnostic to fact that our $s_{\theta}$ approximates the true concrete score. In particular, knowing all $\frac{p_{t}(y)}{p_{t}(x)}$ enables optimal denoising, analogous to Tweedie’s theorem :

###### Theorem 4.1 (Discrete Tweedie’s Theorem) .

Suppose that $p_{t}$ follows the diffusion ODE $dp_{t}=Qp_{t}$. Then the true denoiser is given by

$$ $p_{0|t}(x_{0}|x_{t})=\left(\exp(-tQ)\begin{bmatrix}\frac{p_{t}(i))}{p_{t}(x_{t })}\end{bmatrix}_{i=1}^{N}\right)_{x_{0}}\exp(tQ)(x_{t},x_{0})$ (18) $$

Unfortunately, we do not know all of the ratios (only ratios between Hamming distance 1 sequences). However, we can use this intuition to build a Tweedie denoiser analogue of $\tau$-leaping. In particular, we replace the token transition probabilities (for $x_{t-\Delta t}^{i}$) with the values

$$ $\displaystyle\big{(}\exp(-\sigma_{t}^{\Delta t}Q)s_{\theta}(\mathbf{x}_{t},t)_ {i}\big{)}_{x_{t-\Delta t}^{i}}\exp(\sigma_{t}^{\Delta t}Q)(x_{t}^{i},x_{t- \Delta t}^{i})$ (19) $\displaystyle\text{where }\sigma_{t}^{\Delta t}=(\overline{\sigma}(t)- \overline{\sigma}(t-\Delta t))$ (20) $$

This generalizes the theorem but enforces the tau-leaping independence condition and, in fact, is optimal:

###### Theorem 4.2 (Tweedie $\tau$ -leaping) .

Let $p_{t-\Delta t|t}^{\rm tweedie}(\mathbf{x}_{t-\Delta t}|\mathbf{x}_{t})$ be the probability of the token update rule defined by Equation 19. Assuming $s_{\theta}$ is learned perfectly, this minimizes the KL divergence with the true reverse $p_{t-\Delta t|t}(\mathbf{x}_{t-\Delta t}|\mathbf{x}_{t})$ for all $\tau$-leaping strategies (i.e. token transitions are applied independently and simultaneously).

These simulation algorithms are unified in Algorithm 2.

###### Theorem 4.1 (Discrete Tweedie’s Theorem) .

###### Theorem 4.2 (Tweedie τ 𝜏 \tau italic_τ -leaping) .

### 4.2 Arbitrary Prompting and Infilling

Our concrete score can also be used to enable greater control over the generative process. This is due to the fact that we are modeling a function of the probability, allowing us to include conditional information through Bayes’ rule. In particular, we consider the infilling problem

$$ $p_{t}(\mathbf{x}^{\Omega}|\mathbf{x}^{\overline{\Omega}}=\mathbf{y})\quad \Omega\text{ unfilled indices}\quad\overline{\Omega}\text{ filled}$ (21) $$

As an example, a standard autoregressive conditional generation would have $\overline{\Omega}=\{1,2,\dots,c\}$ and $\Omega=\{c+1,c+2,\dots,d\}$. By Bayes’ rule, the conditional scores can be recovered exactly from the unconditional score.

$$ $\frac{p_{t}(\mathbf{x}^{\Omega}=\mathbf{z}^{\prime}|\mathbf{x}^{\overline{ \Omega}}=\mathbf{y})}{p_{t}(\mathbf{x}^{\Omega}=\mathbf{z}|\mathbf{x}^{ \overline{\Omega}}=\mathbf{y})}=\frac{p_{t}(\mathbf{x}=\mathbf{z}^{\prime} \oplus_{\Omega}\mathbf{y})}{p_{t}(\mathbf{x}=\mathbf{z}\oplus_{\Omega}\mathbf{ y})}$ (22) $$

where $\oplus_{\Omega}$ is concatenation along $\Omega$ and $\overline{\Omega}$. Since the unconditional and conditional scores coincide, we can use our $s_{\theta}$ (learned unconditionally) for conditional sampling (given arbitrary $\overline{\Omega}$). For a $\tau$-leaping update rule (Equation 17 or 19), one would only modify by changing the values at $\Omega$. An explicit pseudocode of this is given in Algorithm 3.

## 5 Experiments

We now empirically validate that our score entropy discrete diffusion (SEDD) model on a variety of language modeling tasks. We measure both perplexity (i.e. likelihood estimation capabilities) as well as generation quality, finding that our method performs quite well in both aspects.

### 5.1 Model and Training Setup

**Table 1: Zero-shot unconditional perplexity ($\downarrow$) on a variety of datasets. For a fixed size, the best perplexity is bolded. Our SEDD model with absorbing transition beats GPT-2 on a majority of the tasks and entirely outperforms prior language diffusion models .**
| Size | Model | LAMBADA | WikiText2 | PTB | WikiText103 | 1BW |
| --- | --- | --- | --- | --- | --- | --- |
| Small | GPT-2 | 45.04 | 42.43 | 138.43 | 41.60 | 75.20 |
|  | SEDD Absorb | $\leq$50.92 | $\leq$41.84 | $\leq$114.24 | $\leq$40.62 | $\leq$79.29 |
|  | SEDD Uniform | $\leq$65.40 | $\leq$50.27 | $\leq$140.12 | $\leq$49.60 | $\leq$101.37 |
|  | D3PM | $\leq$93.47 | $\leq$77.28 | $\leq$200.82 | $\leq$75.16 | $\leq$138.92 |
|  | PLAID | $\leq$57.28 | $\leq$51.80 | $\leq$142.60 | $\leq$50.86 | $\leq$91.12 |
| Medium | GPT-2 | 35.66 | 31.80 | 123.14 | 31.39 | 55.72 |
|  | SEDD Absorb | $\leq$42.77 | $\leq$31.04 | $\leq$87.12 | $\leq$29.98 | $\leq$61.19 |
|  | SEDD Uniform | $\leq$51.28 | $\leq$38.93 | $\leq$102.28 | $\leq$36.81 | $\leq$79.12 |

Our core model is based on the diffusion transformer architecture , which incorporates time conditioning into a standard encoder-only transformer architecture , although we make some minor modifications such as employing rotary positional encoding .

We construct SEDD Absorb and SEDD Uniform, which correspond to the matrices $Q^{\rm uniform}$ and $Q^{\rm absorb}$ respectively. We tested a geometric noise schedule (that interpolates between $10^{-5}$ and $20$), as well as a log-linear noise schedule (the number of changed tokens for total noise $\overline{\sigma}(t)$ is approximately $td$ for both transitions), which helps SEDD Absorb for perplexities. Outside of this, we did not systemically explore noise schedules or alternative loss weightings, although these could likely improve generation quality.

When training, we employ sentence packing to create uniform length blocks to feed to our model, which is done typically for language modeling tasks. The only exception to this rule is our experiment on text8, which randomly samples contiguous subsequences to match prior work (although we found that this did not substantially change results). We also matched architecture hyperparameters with prior work (including number of layers, hidden dimension, attention heads, etc…), although our models have slightly more parameters ($\approx 5-10\%$) than a typical transformer due to time conditioning. We also use the same tokenizers as prior work (which otherwise could be a source of artifacts) as well as the same data splits.

### 5.2 Language Modeling Comparison

We begin by evaluating our model on core language modeling (effectively likelihood-based modeling) on three common datasets across a variety of scales.

#### 5.2.1 Text 8 Dataset

We compare on the text8 dataset, a small, character level language modeling task. We follow for network hyperparameters and dataset splits and compare with methods that employ a similar model size.

We report bits per character (BPC) in Table 2. SEDD outperforms other non-autoregressive models and is only beaten by an autoregressive transformer and the discrete flow (which incorporates an autoregressive base distribution) . Furthermore, SEDD substantially improves upon D3PM , despite both being built from the same discrete diffusion principles.

#### 5.2.2 One Billion Words Dataset

**Table 2: Bits Per Character on text8. Our SEDD models achieve second-best overall result (best for non-autoregressive), only being beaten out by the autoregressive model and a discrete flow (which uses an autoregressive model as a backbone) by a small margin. SEDD also substantially improves upon prior the discrete diffusion model D3PM .**
| Type | Method | BPC ($\downarrow$) |
| --- | --- | --- |
| Autoregressive Backbone | IAF/SCF | 1.88 |
|  | AR Argmax Flow | 1.39 |
|  | Discrete Flow | 1.23 |
|  | Autoregressive | 1.23 |
| Non-autoregressive | Mult. Diffusion | $\leq$ 1.72 |
|  | MAC | $\leq$ 1.40 |
|  | BFN | $\leq$ 1.41 |
|  | D3PM Uniform | $\leq$ 1.61 |
|  | D3PM Absorb | $\leq$ 1.45 |
| Ours (NAR) | SEDD Uniform | $\leq$ 1.47 |
|  | SEDD Absorb | $\leq$ 1.39 |

We also test SEDD on One Billion Words, a more medium sized and real world dataset. We follow for the tokenization, training, and model size configurations. In particular, our baselines are all around the size of GPT-2 small. Following , we compare primarily against other language diffusion models, although we also train a standard autoregressive transformer as a benchmark.

We report perplexity values in Table 3. Our SEDD model outperforms all other diffusion language modeling schemes by $50$-$75\%$ lower perplexity (in particular D3PM). Furthermore, SEDD is within $1$ perplexity of the autoregressive model, likely matching since we only report an upper bound.

**Table 3: Test perplexities on the One Billion Words Dataset. The autoregressive result is an exact likelihood, while the diffusion results are upper bounds. SEDD beats all other discrete diffusion models (by at least $2\times$) while matching the autoregressive baseline.**
| Type | Method | Perplexity ($\downarrow$) |
| --- | --- | --- |
| Autoregressive | Transformer | 31.98 |
| Diffusion | D3PM Absorb | $\leq$ 77.50 |
|  | Diffusion-LM | $\leq$ 118.62 |
|  | BERT-Mouth | $\leq$ 142.89 |
|  | DiffusionBert | $\leq$ 63.78 |
| Ours (Diffusion) | SEDD Uniform | $\leq$ 40.25 |
|  | SEDD Absorb | $\leq$ 32.79 |

Figure: (a) Generative Perplexity $(\downarrow)$ vs. Sampling Iterations.
Refer to caption: extracted/5649957/imgs/img_perplexity.png

#### 5.2.3 GPT-2 Zero Shot Tasks

Finally, we compare SEDD against GPT-2 . We train on OpenWebText as the original WebText dataset has not been made available (this is typical practice and does not meaningfully affect results in practice) and test on the LAMBADA, WikiText2, PTB, WikiText103, and One Billion Words datasets (which were all of the GPT-2 zero-shot tasks that measured perplexity). We recompute baseline likelihoods for all datasets except 1BW, where we encountered unexpected behavior with the public implementations. Our likelihood computation changes from the original setting since we evaluate unconditionally (i.e. without a sliding window), and this results in higher values than originally reported.

Our results are reported in Table 1. Our SEDD Absorb beats GPT-2 on a majority of the zero-shot tasks across both sizes. To the best of our knowledge, this is the first time where a non-autoregressive language model has matched a modern, reasonably sized, and well-known autoregressive model for perplexities. We also compare against the most competitive continuous and discrete diffusion baselines, seeing a large improvement over both.

### 5.3 Language Generation Comparison

With our trained models, we compare against prior work in terms of generation quality. In particular, we compare GPT-2 with our SEDD Absorb on a variety of scales. Results for SEDD Uniform are given in Appendix D.

#### 5.3.1 Unconditional Generation

We first compare the quality of unconditional samples between GPT-2 and SEDD. As most language metrics are meant for comparing conditional generations , we instead measure the generative perplexity of sampled sequences (using a GPT-2 large model for evaluation). This is a simple and common metric but can easily be “hacked” by simple distribution annealing methods. So, we compare analytically sampled generations (i.e. no temperature scaling).

For SEDD, we simulate using 32 to 2048 steps, which approximates the learned distribution with minimal error for a large number of steps (the sequences are length 1024). Our results (both the measured generative perplexity and some samples) are shown in Figure 1. SEDD matches GPT-2 quality using 32$\times$ fewer network evaluations and outperforms by $6$-$8\times$ when using the full 2048 steps. Furthermore, SEDD forms a predictable log-log linear pareto frontier between the number of sampling steps and generative perplexity. However, each network evaluation is different due to the KV-cache, which introduces a cost benefit tradeoff that we discuss more in Section 6.

**Table 4: Conditionally Generated Text. Prompt tokens are given in blue. Our model is able to generate meaningful text with prompt tokens in the front, the end, the middle, or even split up. Additional samples are given in Appendix D.3.**
| A bow and arrow is a traditional weapon that enables an attacker to attack targets at a range within a meter or maybe two meters. They have a range far longer than a human can walk, and they can be fired … |
| --- |
| $\dots$ skydiving is a fun sport that makes me feel incredibly silly. I think I may’ve spent too much, but it could’ve been amazing! While sky diving gives us exercise and fun, scuba diving is an act of physical fitness, … |
| $\dots$ no one expected the results to much better than last year’s one-sided endorsement. Nearly 90 percent of the results were surveyed as ”independent,” an promising result for school children across the country. |
| $\dots$ results show that Donald Trump and Hillary Clinton are in 38 states combined with less than 1% of the national vote. In a way, it’s Trump and Hillary Clinton who will work overtime to get people to vote this $\dots$ |

#### 5.3.2 Infilling Conditional Generation

Finally, we showcase SEDD’s ability for conditional generation. We generate samples conditioned on a fixed amount of input text (from the WebText dataset) and compare their MAUVE scores . For SEDD, we consider two prompting strategies: standard generation given the beginning and infilling using the beginning and end, although obviously more sampling strategies exist (and several are visualized in Table 4).

We compare against GPT-2 and SSD-LM , a competitive language diffusion model built for this task (all models are medium sized). Interestingly, a critical component for both baselines is distribution annealing: nucleus sampling for autoregressive modeling (which clips the token probability) and thresholding for diffusion (which constrains generation to disallow paths in low probability spaces). As introducing similar annealing methods for SEDD is out of scope for this paper, we compare against both the annealed and un-annealed baselines samples.

Our results are given in Table 5. SEDD is highly competitive with the best configuration for both baselines, in fact beating both when using standard prompting. This is rather notable since SEDD does not use distribution annealing and does not explicitly encode left to right prompting as an architectural inductive bias (while GPT-2 and SSD-LM were trained explicitly for autoregressive-like generation).

**Table 5: Evaluation of conditionally generated text. SEDD with standard prompting beats both GPT-2 and SSD-LM. SEDD also offers more flexibility (enabling infilling generation with comparable performance) and does not require distribution annealing techniques for good generation.**
| Method | Annealing | Mauve ($\uparrow$) |
| --- | --- | --- |
| GPT-2 | Nucleus-0.95 | 0.955 |
|  | None | 0.802 |
| SSD-LM | Logit Threshold-0.95 | 0.919 |
|  | None | 0.312 |
| SEDD Standard | None | 0.957 |
| SEDD Infill | None | 0.942 |

## 6 Related Work

Continuous Diffusion Models for Text Data. Initially proposed by , continuous language diffusion models embed tokens in a latent space, learn a diffusion model there, and take the nearest neighbor to dequantize. While initial versions struggled, these models have achieved significant results by iterating on several empirical components. For example, prior works improve downstream performance with alternative loss functions (moving away from likelihood-based score matching) and explicitly encoding conditional information (e.g. inputting an infilling mask) . Additionally, distribution annealing methods like thresholding and classifier-free guidance can further improve generation quality, although recent work has shown that methods like self-conditioning and designing a less sparse embedding space (e.g. based on bits) can obviate the need for such methods. Finally, showed that, with many surgical changes to the training paradigm, it is possible for language diffusion models to begin approaching autoregressive performance for likelihoods.

Discrete Diffusion Models. Most discrete diffusion works follow the framework set out by D3PM which mimics “mean prediction” . These discrete diffusion methods are largely applied to fields other than language (e.g. images), likely due to empirical challenges. Despite this, some works have shown strong performance on language, particularly for seq-to-seq tasks and more efficient generation . Notably, from these works discrete diffusion has tended to be advantageous over continuous diffusion in reducing network evaluations.

SEDD vs Prior Work. SEDD is a discrete diffusion model that focuses on score matching, the crucial ingredient for continuous diffusions . Many such works also focus on reversing a discrete diffusion process , so score entropy is naturally related with prior training objectives. However, SEDD focuses on a principled, scalable, and performant objective (namely denoising score entropy), filling in shortcomings found in previous works. In particular, prior methods train either with the equivalent of implicit score entropy (which is intractable and high variance) or propose alternate losses that suffer from other issues. These critical differences enable large improvements for language tasks, where prior discrete diffusion models have conspicuously struggled on.

Furthermore, SEDD achieves better results (for both perplexity and generation) than even continuous diffusion models (without resorting to empirically driven heuristics). This is desirable since discrete data should necessitate a novel approach. Future work could adapt empirical designs from continuous diffusion, further improving performance.

Finally, SEDD challenges autoregressive models, achieving competitive perplexities (beating GPT-2) and generation quality (beating nucleus sampling). While there is still a large gap with modern large language models, we believe that future work can bridge this using SEDD as a backbone.

SEDD vs Autoregressive Sampling Iterations. SEDD and autoregressive models have significantly different sampling procedures due to the introduction of the KV-cache for standard decoder-only transformer models. In particular, this complicates the inference code (as each network pass changes from being a standard full batch forward) and trades off speed with memory. For example, for our (known) unoptimized codebase and the existing huggingface transformers library , we observed that SEDD matches autoregressive inference time when using around 100 steps but can increase the batch size by roughly $4-6$ times by removing the KV-cache memory. Future work will likely decrease the steps required for optimal generation (similar to existing work in standard diffusion ) which can improve this tradeoff.

## 7 Conclusion

We have introduced score entropy discrete diffusion (SEDD) models, a discrete diffusion model that is parameterized by the concrete score and can be trained efficiently with our novel score entropy loss. SEDD beats previous language diffusion models and rivals autoregressive models for both perplexity and quality. We hope that future work can build off our framework to defines alternatives to the modern autoregressive language modeling paradigm.

## Impact Statement

This paper proposes work that advances the field of natural language generation. Outside of existing ethical questions for this area (e.g. bias, toxicity, fake content), our approach does not present any specific danger as the core work is largely theoretical and not at the scale to pose a specific problem.

## Acknowledgements

This project was supported by NSF (#1651565), ARO (W911NF-21-1-0125), ONR (N00014-23-1-2159), CZ Biohub, a Stanford HAI GCP grant. AL is supported by a NSF Graduate Research Fellowship.

## Appendix A Proof of Main Results

###### Proof of Prop 3.2 .

Given infinite samples, the loss becomes equivalent to minimizing

$$ $\min_{\theta}\sum_{x,y\neq x}p(x)w_{xy}\left(s_{\theta}(x)_{y}-\frac{p(y)}{p(x )}\log s_{\theta}(x)_{y}\right)$ (23) $$

where we have removed constants not depending on $\theta$. This is minimized when

$$ $s_{\theta}(x)_{y}-\frac{p(y)}{p(x)}\log s_{\theta}(x)_{y}$ (24) $$

is minimized for all $x,y$. Taking a derivative with respect to $s$ and setting to 0 0, we see that this occurs when $s_{\theta}(x)_{y}=\frac{p(y)}{p(x)}$, which can be easily checked to be optimal as the function is convex as a function of $s$. One can check that the loss is 0 0 at the minimum.
∎

###### Proof of Prop 3.3 .

The trick is the categorical equivalent of the divergence theorem. In particular, we have

$$ $\displaystyle\mathbb{E}_{x\sim p}\sum_{y\neq x}\frac{p(y)}{p(x)}f(x,y)$ $\displaystyle=\sum_{x,y:x\neq y}\frac{p(y)}{p(x)}p(x)f(x,y)$ $\displaystyle=\sum_{x,y:x\neq y}p(y)f(x,y)$ $\displaystyle=\mathbb{E}_{y\sim p}\sum_{x\neq y}f(x,y)$ $\displaystyle=\mathbb{E}_{x\sim p}\sum_{y\neq x}f(y,x)$ $$

for abitrary $f$. By setting $f(x,y)=w_{xy}\log s_{\theta}(x)_{y}$, we get that

$$ $\displaystyle\mathbb{E}_{x\sim p}\left[\sum_{y\neq x}w_{xy}\left(s_{\theta}(x) _{y}-\frac{p(y)}{p(x)}\log s_{\theta}(x)_{y}+K\left(\frac{p(y)}{p(x)}\right) \right)\right]$ $\displaystyle=\mathbb{E}_{x\sim p}\left[\sum_{y\neq x}w_{xy}s_{\theta}(x)_{y}- w_{yx}\log s_{\theta}(y)_{x}+w_{xy}K\left(\frac{p(y)}{p(x)}\right)\right]$ $$

which is the desired equivalent (as the last term does not depend on $\theta$).
∎

###### Proof of Thm 3.4 .

This is similar to the same denoising variant for concrete score matching. We just need to show that the $\log s_{\theta}(x_{t})_{y}\frac{p_{t}(y)}{p_{t}(x)}$ marginalizes out, since everything else does not change or is a constant.

$$ $\displaystyle\mathbb{E}_{x\sim p}\sum_{y\neq x}f(x,y)\frac{p(y)}{p(x)}$ $\displaystyle=\sum_{y\neq x}f(x,y)p_{t}(y)$ $\displaystyle=\sum_{y\neq x}\sum_{x_{0}}f(x_{t},y)p(y|x_{0})p_{0}(x_{0})$ $\displaystyle=\mathbb{E}_{x_{0}\sim p_{0}}\sum_{y\neq x}f(x,y)\frac{p(y|x_{0}) }{p(x|x_{0})}p(x|x_{0})$ $\displaystyle=\mathbb{E}_{x_{0}\sim p_{0},x\sim p(\cdot|x_{0})}\sum_{y\neq x}f (x,y)\frac{p(y|x_{0})}{p(x|x_{0})}$ $$

Applying this to our loss when $f(x,y)=w_{xy}\log s_{\theta}(x)_{y}$ gives us

$$ $\displaystyle\mathbb{E}_{x\sim p}\left[\sum_{y\neq x}w_{xy}\left(s_{\theta}(x) _{y}-\frac{p(y)}{p(x)}\log s_{\theta}(x)_{y}+K\left(\frac{p(y)}{p(x)}\right) \right)\right]$ $\displaystyle=\mathbb{E}_{x\sim p}\left[\sum_{y\neq x}w_{xy}\left(s_{\theta}(x )_{y}+K\left(\frac{p(y)}{p(x)}\right)\right)\right]-\mathbb{E}_{x_{0}\sim p_{0 },x\sim p(\cdot|x_{0})}\left[\sum_{y\neq x}\frac{p(y|x_{0})}{p(x|x_{0})}w_{xy} \log s_{\theta}(x)_{y}\right]$ $\displaystyle=\mathbb{E}_{x_{0}\sim p_{0},x\sim p(\cdot|x_{0})}\left[w_{xy} \left(s_{\theta}(x)_{y}\frac{p(y|x_{0})}{p(x|x_{0})}\log s_{\theta}(x)_{y}+K \left(\frac{p(y)}{p(x)}\right)\right)\right]$ $$

∎

###### Proof of Thm 3.6 .

The full bound is given by

$$ $-\log p_{0}^{\theta}(x_{0})\leq\mathcal{L}_{\rm DWDSE}(x_{0})+D_{\rm KL}(p_{T| 0}(\cdot|x_{0})\parallel\pi)$ (25) $$

where $\mathcal{L}_{\rm DWDSE}$ is given by

$$ $\int_{0}^{T}\mathbb{E}_{x_{t}\sim p_{t|0}(\cdot|x_{0})}\sum_{y\neq x_{t}}Q_{t} (x_{t},y)\left(s_{\theta}(x_{t},t)_{y}-\frac{p_{t|0}(y|x_{0})}{p_{t|0}(x_{t}|x _{0})}\log s_{\theta}(x,t)_{y}+K\left(\frac{p_{t|0}(y|x_{0})}{p_{t|0}(x_{t}|x_ {0})}\right)\right)dt$ $$

Effectively, $\mathcal{L}_{\rm DWSDE}$ is the path measure KL divergence , and the proof follows similarly. In particular, we have that, by the data processing inequality

$$ $-\log p_{0}^{\theta}(x_{0})=D_{\rm KL}(\delta_{x_{0}}\parallel p_{0}^{\theta}) \leq D_{\rm KL}(\mathbb{P}_{x_{0}}\parallel\mathbb{P}^{\theta})$ (26) $$

where $\mathbb{P}_{x_{0}}$ is the path measure for the reverse of the noising process applied to $\delta_{x_{0}}$ and $\mathbb{P}^{\theta}$ is the learned reverse process. Generally, we can replace $\delta_{x_{0}}$ with a more general data distribution $p_{\rm data}$, with the computation remaining the same. We have,

$$ $D_{\rm KL}(\mathbb{P}_{x_{0}}\parallel\mathbb{P}^{\theta})\leq\mathbb{E}_{x_{T }\sim p_{T|0}(\cdot|x_{0})}\left[D_{\rm KL}(\mathbb{P}_{x_{0}}(\cdot|x_{T}) \parallel\mathbb{P}^{\theta}(\cdot|x_{T}))\right]+D_{\rm KL}(p_{T|0}(\cdot|x_{ 0})\parallel\pi)$ (27) $$

We analyze the term $\mathbb{E}_{x_{T}}D_{\rm KL}(\mathbb{P}_{x_{0}}(\cdot|x_{T})\parallel\mathbb{P
}^{\theta}(\cdot|x_{T}))$, which we can compute by Dynkin’s formula , which, similar to Girsanov’s Theorem for standard SDEs , allows one to compute the change in measure. In particular, by applying Theorem 7.1 of with degenerate SDE coefficients, we find the expectation to be given explicitly by

$$ $\displaystyle\int_{0}^{T}\mathbb{E}_{x_{t}\sim p_{t|0}(\cdot|x_{0})}$ $\displaystyle\sum_{y\neq x_{t}}\overline{Q}_{t}^{\theta}(y,x_{t})-Q_{t}(y,x_{t })\log(\overline{Q}_{t}^{\theta}(x_{t},y))$ (28) $\displaystyle+Q_{t}(y,x_{t})\log Q_{t}(y,x_{t})+Q_{t}(x_{t},y)K\left(\frac{p_{ t|0}(y|x_{0})}{p_{t|0}(x_{t}|x_{0})}\right)dt$ (29) $$

Since our reverse rate matrices $\overline{Q}_{t}^{\theta}$ are parameterized with $s_{\theta}$, we can simplify the above to

$$ $\int_{0}^{T}\mathbb{E}_{x_{t}\sim p_{t|0}(\cdot|x_{0})}\sum_{y\neq x_{t}}Q_{t} (x_{t},y)\left(s_{\theta}(x_{t},t)_{y}+K\left(\frac{p_{t|0}(y|x_{0})}{p_{t|0}( x_{t}|x_{0})}\right)\right)-Q_{t}(y,x_{t})\log s_{\theta}(y,t)_{x_{t}}dt$ (30) $$

To finalize, we simply note that the summation over $Q(y,x_{t})\log(s_{\theta}(y,t)_{x_{t}})$ can be simplified with the (reverse of) the trick used for proving 3.3.

$$ $\displaystyle\mathbb{E}_{x_{t}\sim p_{t|0}(\cdot|x_{0})}\sum_{y\neq x_{t}}Q(y, x_{t})\log s_{\theta}(y)_{x_{t}}$ $\displaystyle=\sum_{x_{t},y\neq x_{t}}p_{t|0}(x_{t}|x_{0})Q(y,x_{t})\log s_{ \theta}(y)_{x_{t}}$ (31) $\displaystyle=\mathbb{E}_{y\sim p_{t|0}(\cdot|x_{0})}\frac{p_{t|0}(x_{t}|x_{0} )}{p_{t|0}(y|x_{0})}Q(y,x_{t})\log s_{\theta}(y)_{x_{t}}$ (32) $\displaystyle=\mathbb{E}_{x_{t}\sim p_{t|0}(\cdot|x_{0})}\frac{p_{t|0}(y|x_{0} )}{p_{t|0}(x_{t}|x_{0})}Q(x_{t},y)\log s_{\theta}(x_{t})_{y}$ (33) $$

where the last line is just a permutation of the notation of $x_{t}$ and $y$. As such, we get the desired loss

$$ $\int_{0}^{T}\mathbb{E}_{x_{t}\sim p_{t|0}(\cdot|x_{0})}\sum_{y\neq x_{t}}Q_{t} (x_{t},y)\left(s_{\theta}(x_{t},t)_{y}-\frac{p_{t|0}(y|x_{0})}{p_{t|0}(x_{t}|x _{0})}\log s_{\theta}(x,t)_{y}+K\left(\frac{p_{t|0}(y|x_{0})}{p_{t|0}(x_{t}|x_ {0})}\right)\right)dt$ $$

∎

###### Proof of Thm 4.1 .

This can be shown by Bayes’ rule:

$$ $p_{0|t}(x_{0}|x_{t})=\frac{p_{t|0}(x_{t}|x_{0})p_{0}(x_{0})}{p_{t}(x_{t})}=p_{ t|0}(x_{t}|x_{0})\frac{p_{0}(x_{0})}{p_{t}(x_{t})}$ (34) $$

We have $p_{0}=\exp(-\sigma Q)p_{t}$ and $p_{t|0}(x_{t}|x_{0})=\exp(\sigma Q)_{x_{t},x_{0}}$, so the theorem follows.
∎

###### Proof of Thm 4.2 .

Using our factorization assumption we get that

$$ $\displaystyle D_{\rm KL}\left(p_{t-\Delta t|t}(\mathbf{x}_{t-\Delta t}|\mathbf {x}_{t})\parallel p_{t-\Delta t|t}^{\theta}(\mathbf{x}_{t-\Delta t}|\mathbf{x} _{t})\right)$ (35) $\displaystyle=-\sum_{i=1}^{d}\mathbb{E}_{\mathbf{x}_{t-\Delta t}\sim p_{t- \Delta t|t}(\mathbf{x}_{t-\Delta t}|\mathbf{x}_{t})}\left[\log p_{t-\Delta t|t }^{\theta}(x_{t-\Delta t}^{i}|\mathbf{x}_{t})\right]+C$ (36) $$

where $C$ is a constant independent of $\theta$. We simply need to minimize the following cross entropy loss for each $i$

$$ $-\mathbb{E}_{\mathbf{x}_{t-\Delta t}\sim p_{t-\Delta t|t}(\mathbf{x}_{t-\Delta t }|\mathbf{x}_{t})\left[\log p_{t-\Delta t|t}^{\theta}(x_{t-\Delta t}^{i}| \mathbf{x}_{t})\right]}$ (37) $$

Our $\tau$-leaping condition implies that our transition assumes no change in other dimensions, so in particular $p_{t-\Delta t}^{i}(x_{t-\Delta t}^{i}|\mathbf{x}_{t})=p_{t-\Delta t|t}^{\theta
}(x_{t}^{1}\dots x_{t-\Delta t}^{i}\dots x_{t}^{d}|\mathbf{x}_{t})$. By the standard properties of cross entropy, this is minimized when $p_{t-\Delta t|t}^{\theta}(x_{t}^{1}\dots x_{t-\Delta t}^{i}\dots x_{t}^{d}|
\mathbf{x}_{t})=p_{t-\Delta t|t}(\mathbf{x}_{t-\Delta t}|\mathbf{x}_{t})$. This equality follows directly from Thm 4.1.
∎

###### Proof of Prop 3.2 .

###### Proof of Prop 3.3 .

###### Proof of Thm 3.4 .

###### Proof of Thm 3.6 .

###### Proof of Thm 4.1 .

###### Proof of Thm 4.2 .

## Appendix B Algorithms for Training and Inference

Figure: Algorithm 1 Score Entropy Training Loop (Multiple Dimensions)

Figure: Algorithm 2 Score Entropy Sampling (Unconditional)

Figure: Algorithm 3 Score Entropy Sampling (Conditional)

## Appendix C Additional Experimental Details

### C.1 Diffusion Details

The geometric noise distribution is $\overline{\sigma}(t)=\sigma_{\rm min}^{1-t}\sigma_{\rm max}^{t}$. The log linear noise schedule is $\overline{\sigma}(t)=-\log(1-(1-\epsilon t))$ for some small epsilon for numerical stability as $t\to 1$, commonly $10^{-3}$ or $10^{-4}$. These noise schedules were chosen such that the prior loss $D_{\rm KL}(p_{T|0}(\cdot x_{0})\parallel\pi)$ and the approximation of $p_{\rm data}$ with $p_{\rm\overline{\sigma}(0)}$ are negligible. We typically scale the uniform transition matrix down by $\frac{1}{N}$ and take $p_{\rm base}$ to be uniform. For the absorbing state, we take $p_{\rm base}$ to be the MASK state with some leakage of probability to a random non-MASK state (to avoid $\inf$ KL divergence, although this is negligible and is not used for generation in practice).

### C.2 Model Details

Our model train with flash attention with fused kernels wherever applicable. We also use the adaLN-zero time information network of with $128$ hidden dimension. Following previous work, we parameterize the network with the total noise level instead of the time $t$. We also found it easier to postprocess the output of our network to form $s_{\theta}$, rather than outputting it directly. Concretely, we exponentiate (which maintains positivity) to be beneficial to avoid numerical errors and also found that scaling by $e^{\overline{\sigma}}-1$ helps for absorbing diffusion.

SEDD models have the same hidden dimensions, number of blocks, and number of heads as their corresponding GPT-2 models. However, SEDD models also use a separate word embedding matrix and output matrix. In total, SEDD small and SEDD medium have around 90M parameters and 320M non embedding parameters respectively (compared to GPT-2 small 86M and GPT-2 medium 304M non-embedding parameters respectively).

### C.3 Training Details

All models were trained with a batch size of 512 and trained with a learning rate of $3\times 10^{-4}$. We clip our gradient norm to 1 and have a linear warmup schedule for the first 2000 iterations. We also use a 0.9999 EMA.

We trained on nodes of 8 A100 80GB or 16 A100 40GB GPUs, using gradient accumulation when our batch size did not fit into memory (as is the case for SEDD medium).

### C.4 Hyperparameter Search

We did not do a hyperparameter or achitecture search. Our hyperparameters were chosen for convenience purposes (e.g. the architecture was taken from DDiT , but we use rotary embeddings since they come included in previous work ) or were naturally lifted from previous training recipes (e.g. the ubiquitous $3\times 10^{-4}$ learning rate, $0.9999$ EMA).

### C.5 Baseline Details (for Likelihood-based Training and Evaluation)

#### C.5.1 Text8

The baselines are taken from , with many coming from . In particular, they are IAF/SCF , the Autoregressive Argmax Flow , and the discrete flow for autoregressive models. The non-autoregressive baselines are, in order, Multinomial Diffusion , MAC , Bayesian Flow Networks , and D3PM .

#### C.5.2 One Billion Words Perplexity

The baselines are taken from . They are D3PM , Diffusion-LM , BERT-mouth , and DiffusionBert .

#### C.5.3 GPT-2

The only two non GPT-2 baselines are PLAID and D3PM (with Absorbing Transition) . We retrain both models (as they have not been trained with our exact specifications) to compare against small models. We reuse our model architecture and match hyperparameters (i.e. model size, training specifications).

### C.6 Likelihood Evaluation Details

We randomly sample with $1000$ timesteps to Monte Carlo estimate our likelihoods. We use invertible tokenizers, as is customary for GPT-2 experiments. We report results on the test set for all datasets besides WikiText02, where we report on the train set since WikiText02 and WikiText103 share the same test set.

### C.7 Unconditional Generation Details

We generate using the Tweedie denoiser, which performed slightly better than the Euler sampling (typically by 1-4 perplexity points). We generated $1000$ samples for all models.

### C.8 Conditional Generation Details

We follow and generate $5$ samples for each ground truth sample before calculating MAUVE. Note that this implies that we compare $5000$ generated samples and $1000$ ground truth samples. We sample by conditioning on $50$ tokens and generating a new $50$. For autoregressive-type sampling, this means we take the first $50$ tokens. For SEDD with infilling, this means we clamp all input text sizes to a max of $100$ tokens and condition on the first and last $25$ tokens.

## Appendix D Additional Experimental Results

### D.1 Ablation of Concrete Score Matching

We also ablated the concrete score matching objective from for the GPT-2 scale experiments. This was done by simply replacing the score entropy term with the corresponding $\ell^{2}$ based loss (in particular keeping the scaling by $Q_{t}(x,y)$). In general, we found that this did not train well, resulting in $3-4\times$ higher likelihood loss, which corresponds to 10,000$\times$ higher perplexity. Similarly,

Figure: Figure 2: Generative Perplexity for SEDD Uniform.
Refer to caption: extracted/5649957/imgs/img_perplexity_large_uniform.png

### D.2 Further Evaluation of Generative Perplexity

We further evaluate our generative perplexity for uniform models as well as different sampling schemes (analytic sampling based on Tweedie’s vs Euler sampling based off of reverse diffusion). Results are shown in Figure 2. Generally, we find that uniform does not produce the same linear tradeoff curve as absorbing (most likely due to a bottleneck in generation quality). Futhermore, analytic generally outperforms Euler sampling, and this is a major factor for the uniform model.

We also generated on our trained baselines , finding both performed substantially worse than our SEDD Absorb baseline but slightly better than our SEDD Uniform.

### D.3 Additional Samples

Continued on next page.

Figure: Figure 3: GPT-2 Small Analytic Sampling. Unconditional

Figure: Figure 4: SEDD-Uniform Small. Unconditional

Figure: Figure 5: SEDD-Absorbing Small. Unconditional

Figure: Figure 6: GPT-2 Medium Analytic Sampling. Unconditional.

Figure: Figure 7: SEDD-Uniform Medium. Unconditional

Figure: Figure 8: SEDD-Absorbing Medium. Unconditional

Figure: Figure 9: SEDD-Absorbing Small. Conditional in blue.

Figure: Figure 10: SEDD-Absorbing Small. Conditional in blue.

Figure: Figure 11: SEDD-Absorbing Small. Conditional in blue.

Figure: Figure 12: SEDD-Absorbing Medium. Conditional in blue.

Figure: Figure 13: SEDD-Absorbing Medium. Conditional in blue.

Figure: Figure 14: SEDD-Absorbing Medium. Conditional in blue.