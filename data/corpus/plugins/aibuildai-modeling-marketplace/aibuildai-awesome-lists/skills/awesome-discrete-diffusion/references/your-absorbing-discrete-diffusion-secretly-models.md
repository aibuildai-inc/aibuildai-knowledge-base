---
arxiv_id: "2406.03736"
title: "Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data"
year: 2024
source: arxiv2md
---

## Abstract

Abstract Discrete diffusion models with absorbing processes have shown promise in language modeling. The key quantities to be estimated are the ratios between the marginal probabilities of two transitive states at all timesteps, called the concrete score. In this paper, we reveal that the concrete score in absorbing diffusion can be expressed as conditional probabilities of clean data, multiplied by a time-dependent scalar in an analytic form. Motivated by this finding, we propose reparameterized absorbing discrete diffusion (RADD), a dedicated diffusion model without time-condition that characterizes the time-independent conditional probabilities. Besides its simplicity, RADD can reduce the number of function evaluations (NFEs) by caching the output of the time-independent network when the noisy sample remains unchanged in a sampling interval, which enables sampling acceleration.
Built upon the new perspective of conditional distributions, we further unify absorbing discrete diffusion and any-order autoregressive models (AO-ARMs), showing that the upper bound on the negative log-likelihood for the diffusion model can be
interpreted as an expected negative log-likelihood for AO-ARMs. Further, our RADD models achieve SOTA performance among diffusion models on 5 zero-shot language modeling benchmarks (measured by perplexity) at the GPT-2 scale.
Our code is available at https://github.com/ML-GSAI/RADD .

## 1 Introduction

Auto-regressive models  have dominated the area of language modeling for many years. In particular, such models significantly benefit from large-scale transformers  and training data and have achieved remarkable progress . From a probabilistic perspective, the sequential sampling process of auto-regressive models is inefficient and limits the reasoning ability in nonsequential orders . Intrinsically, this is
because such models characterize the joint distribution by the chain rule of probability, motivating research on developing other types of generative models for text.

Diffusion models  generate data in a coarse-to-fine manner efficiently  and all dimensions simultaneously, providing an appealing alternative to auto-regressive models. Among other efforts  (see [Section˜5](#S5) for a comprehensive discussion), score entropy discrete diffusion (SEDD)  has shown promise in text generation. In particular, SEDD has achieved comparable results to auto-regressive models on 5 zero-shot language modeling benchmarks at the GPT-2 scale. Meanwhile, SEDD can reduce the number of function evaluations (NFEs) in sampling and fulfill text conditioned on prompts at different positions.

Technically, SEDD employs a discrete-state (absorbing) Markov process that adds noises to data by randomly replacing a token with a mask token $[\textbf{M}]$ and then learns a reverse process to denoise from an entirely masked sentence. The key quantities to be estimated are the ratios between the marginal probabilities of two transitive states at all timesteps, called the concrete score. SEDD also proposes a “scaling trick” (see details in [Section˜3](#S3)) that scales the output of the score estimation by a factor. The trick has been proven effective in practice yet not fully understood in theory .

One of our main contributions is to reveal that the concrete score in absorbing diffusion can be expressed as conditional probabilities of clean data, multiplied by a time-dependent scalar in an analytic form (see Theorem [1](#Thmtheorem1)).
Our finding theoretically explains the benefits of the scaling trick as a reparameterization for better optimization. Motivated by the finding, we propose reparameterized absorbing discrete diffusion (RADD), a dedicated diffusion model that characterizes the time-independent conditional probabilities by removing the time conditions from the score estimation (see [Fig.˜1](#S3.F1)). Besides its simplicity, RADD can significantly reduce the NFEs by caching the output of the time-independent network when the noisy sample remains unchanged during a sampling interval.

Built upon the new understanding of the concrete score, we further unify absorbing discrete diffusion and any-order autoregressive models (AO-ARMs) , demonstrating that their training objectives are equivalent (see Theorem [2](#Thmtheorem2)). To establish the theory, we first rewrite the original training objective for absorbing discrete diffusion into a simpler form (named $t$-denoising cross-entropy, $t$-DCE). Then, we apply a change of variable from the time $t$ to the probability that a single-dimensional token is masked at time $t$ in the forward process. By integrating the probability variable analytically, we show its equivalence to the training objectives for AO-ARMs. These theoretical findings offer a fresh perspective that the upper bound on the negative log-likelihood of an absorbing discrete diffusion can be interpreted as the expected negative log-likelihood for corresponding AO-ARMs. Furthermore, they provide alternative objective functions for training and likelihood evaluation.

Empirically,
the RADD model converges faster while achieving similar performance to the strongest baseline, i.e., SEDD . Moreover,
we trained our RADD models on different objective functions,
achieving state-of-the-art performance among diffusion models on five zero-shot language modeling benchmarks (measured by perplexity) at the GPT-2 scale. This empirical evidence validates our theoretical findings.

In summary, this paper has several contributions:

- •
Deeper understanding of discrete diffusion: Both the factorization form of the concrete score and unified training objective for absorbing discrete diffusion and AO-ARMs reveal important yet overlooked theoretical properties of absorbing discrete diffusion, which explain the mysterious scaling trick, provide practice guidance, and may inspire future work.
- •
Simpler parameterization: By removing the time conditions, we reparameterize the model to focus on a time-independent conditional probability, simplifying the existing model.
- •
Efficient sampling: Leveraging the reparameterized form, RADD can use a caching strategy to improve the sampling speed and achieve faster convergence.
- •
Enhanced zero-shot language modeling performance:
Our architectural simplifications and optimized training loss lead to superior results.
On five zero-shot language modeling benchmarks, RADD achieves state-of-the-art performance among discrete diffusion models (measured by perplexity) at the GPT-2 scale.

## 2 Background

In this section, we provide an overview of continuous-time discrete diffusion models ([Section˜2.1](#S2.SS1)) and any-order autoregressive models ([Section˜2.2](#S2.SS2)), with a more detailed discussion available in [Appendix˜G](#A7). For a complete list of notations and definitions, please refer to [Appendix˜A](#A1).

### 2.1 Continuous time discrete diffusion model

#### Single dimension

Let $x$ denote a single dimensional sample with possible values in $\mathcal{X}=\{1,\ldots,N\}$.
A continuous-time discrete Markov chain at time $t$ is characterized by a transition rate matrix ${\bm{Q}}_{t}$ as follows

$$ $p_{t+\Delta t|t}(\hat{x}|x)=\begin{cases}{\bm{Q}}_{t}(x,\hat{x})\Delta t+o(\Delta t),&\hat{x}\neq x,\\ 1+{\bm{Q}}_{t}(x,x)\Delta t+o(\Delta t),&\hat{x}=x,\end{cases}$ (2.1) $$

where ${\bm{Q}}_{t}(x,\hat{x})$ is the $(x,\hat{x})$ element of transition rate matrix ${\bm{Q}}_{t}$, denoting the transition rate from state $x$ to state $\hat{x}$ at time $t$.
Equivalently, ${\bm{Q}}_{t}(x,\hat{x})$ is defined as

$$ ${\bm{Q}}_{t}(x,\hat{x})=\begin{cases}\lim_{\Delta t\rightarrow 0}\frac{p_{t+\Delta t|t}(\hat{x}|x)}{\Delta t},&\hat{x}\neq x,\\ \lim_{\Delta t\rightarrow 0}\frac{p_{t+\Delta t|t}(x|x)-1}{\Delta t},&\hat{x}=x.\end{cases}$ (2.2) $$

Given the above definition, denote ${\bm{P}}_{t|s}(x,\hat{x}):=p_{t|s}(\hat{x}|x)$. The following Kolmogorov’s forward equation holds :

$$ $\frac{d}{dt}{\bm{P}}_{t|s}={\bm{P}}_{t|s}{\bm{Q}}_{t}.$ (2.3) $$

In practice , ${\bm{Q}}_{t}$ is parameterized as $\sigma(t){\bm{Q}}$, where $\sigma(t)$ is a scalar function representing the noise schedule and ${\bm{Q}}$ is a constant matrix. In this case, the solution to [Eq.˜2.3](#S2.E3) can be solved analytically as ${\bm{P}}_{t|s}=\exp\left((\bar{\sigma}(t)-\bar{\sigma}(s)){\bm{Q}}\right)$,
where $\bar{\sigma}(t)=\int_{0}^{t}\sigma(s)ds$ and $\exp$ is the matrix exponential. Therefore, we can directly sample $x_{t}$ from $x_{s}$ in one step for any $t>s$.

Further, ${\bm{Q}}$ is often designed to diffuse towards a uniform distribution or an absorbing state $[\textbf{M}]$. Recent work  suggests that the absorbing matrix achieves better empirical performance. Besides, as detailed in Section [3](#S3), the specific structure of the absorbing matrix can be leveraged to improve performance and accelerate sampling. Therefore, we focus on the absorbing matrix as follows:

$$ $\displaystyle{\bm{Q}}^{\text{absorb}}=\left[\begin{array}[]{ccccc}-1&0&\cdots&0&1\\ 0&-1&\cdots&0&1\\ \vdots&\vdots&\ddots&\vdots&\vdots\\ 0&0&\cdots&-1&1\\ 0&0&\cdots&0&0\end{array}\right].$ (2.9) $$

The time reversal of the forward process is characterized by a reverse transition rate matrix $\tilde{{\bm{Q}}}_{t}$ , whose element from state $x_{t}$ to state $\hat{x}_{t}$ is given by

$$ $\tilde{{\bm{Q}}}_{t}(x_{t},\hat{x}_{t})=\begin{cases}\frac{p_{t}(\hat{x}_{t})}{p_{t}(x_{t})}{\bm{Q}}_{t}(\hat{x}_{t},x_{t}),&\hat{x_{t}}\neq x_{t},\\ -\sum_{k\neq x_{t}}\tilde{{\bm{Q}}}_{t}(x_{t},k),&\hat{x}_{t}=x_{t}.\end{cases}$ (2.10) $$

Simulating the reverse process requires learning the reverse transition rate $\tilde{{\bm{Q}}}_{t}(x_{t},\hat{x}_{t})$. As ${\bm{Q}}_{t}(\hat{x}_{t},x_{t})$ is known, it is sufficient to estimate the concrete score $\frac{p_{t}(\hat{x}_{t})}{p_{t}(x_{t})}$ by a score network $s_{\theta}(x_{t},t)\approx[\frac{p_{t}(\hat{x}_{t})}{p_{t}(x_{t})}]_{\hat{x}_{t}\in\mathcal{X}}$ . Denoising score entropy (DSE)  is an effective objective to train the score network

$$ $\int_{0}^{T}\mathbb{E}_{x_{t}\sim p_{t|0}\left(x_{t}\mid x_{0}\right)}\sum_{{\hat{x}_{t}}\neq x_{t}}{\bm{Q}}_{t}\left(\hat{x}_{t},x_{t}\right)\left(s_{\theta}\left(x_{t},t\right)_{\hat{x}_{t}}-\frac{p_{t\mid 0}\left({\hat{x}_{t}}\mid x_{0}\right)}{p_{t\mid 0}\left(x_{t}\mid x_{0}\right)}\log s_{\theta}\left(x_{t},t\right)_{\hat{x}_{t}}+C\right)dt,$ (2.11) $$

where the optimize irrelevant constant $C=K\left(\frac{p_{t\mid 0}\left({\hat{x}_{t}}\mid x_{0}\right)}{p_{t\mid 0}\left(x_{t}\mid x_{0}\right)}\right)$ and $K(a):=a\log a-a$. In particular, the DSE loss in [Eq.˜2.11](#S2.E11) is an upper bound of the negative log-likelihood with an unknown gap. Nevertheless, existing work  still employs it for training and likelihood evaluation.

After training, sampling can be understood as discretizing the following reverse process

$$ $\frac{d}{ds}{\bm{P}}_{s|t}={\bm{P}}_{s|t}\tilde{{\bm{Q}}}_{s},$ (2.12) $$

where $ds$ is an infinitesimal negative timestep and the concrete score is replaced by the score network. Existing samplers include the Euler method and Tweedie $\tau$ -leaping, as detailed in [Appendix˜D](#A4).

#### Multi-dimension

In a state space of length $d$ like $\mathcal{X}^{d}=\{1,\ldots,n\}^{d}$, we denote the sample as a sequence of one-dimensional data, i.e.,${\bm{x}}=x^{1}\ldots x^{d}$. The transition matrix ${\bm{Q}}_{t}\in\mathbb{R}^{n^{d}\times n^{d}}$ has an exponential number of possible states, making it expensive to reverse. To alleviate this issue, existing work  assumes independence between dimensions and each dimension is a one-dimensional diffusion process with the same transition rate matrix ${\bm{Q}}^{\text{tok}}_{t}\in\mathbb{R}^{n\times n}$.

Under the independent assumption, ${\bm{Q}}_{t}$ assigns zero values  for all sequences with a Hamming distance larger than 1. Therefore, it is sufficient to model the concrete score between sequences that differ by a Hamming distance of 1, such as $\hat{{\bm{x}}}_{t}=x_{t}^{1}\ldots\widehat{x}_{t}^{i}\ldots x_{t}^{d}$ given ${\bm{x}}_{t}=x_{t}^{1}\cdots x_{t}^{d}$. Therefore, the score network ${\bm{s}}_{\theta}(\cdot,t):\{1,\ldots,n\}^{d}\rightarrow\mathbb{R}^{d\times n}$ is defined as

$$ ${\bm{s}}_{\theta}\left({\bm{x}}_{t},t\right)_{\hat{{\bm{x}}}_{t}}={\bm{s}}_{\theta}\left(x_{t}^{1}\ldots x_{t}^{i}\ldots x_{t}^{d},t\right)[i,\widehat{x}_{t}^{i}]\approx\frac{p_{t}\left(x_{t}^{1}\ldots\widehat{x}_{t}^{i}\ldots x_{t}^{d}\right)}{p_{t}\left(x_{t}^{1}\ldots x_{t}^{i}\ldots x_{t}^{d}\right)},$ (2.13) $$

which leads to the following expression to estimate the reverse transition rate matrix $\tilde{{\bm{Q}}}_{t}$:

$$ $\displaystyle\tilde{{\bm{Q}}}_{t}\left(x_{t}^{1}\ldots x_{t}^{i}\ldots x_{t}^{d},x_{t}^{1}\ldots\widehat{x}_{t}^{i}\ldots x_{t}^{d}\right)$ $\displaystyle={\bm{Q}}^{\text{tok}}_{t}\left(\widehat{x}_{t}^{i},x_{t}^{i}\right)\frac{p_{t}\left(x_{t}^{1}\ldots\widehat{x}_{t}^{i}\ldots x_{t}^{d}\right)}{p_{t}\left(x_{t}^{1}\ldots x_{t}^{i}\ldots x_{t}^{d}\right)}$ (2.14) $\displaystyle\approx{\bm{Q}}^{\text{tok}}_{t}\left(\widehat{x}_{t}^{i},x_{t}^{i}\right){\bm{s}}_{\theta}\left(x_{t}^{1}\ldots x_{t}^{i}\ldots x_{t}^{d},t\right)[i,\widehat{x}_{t}^{i}].$ (2.15) $$

Existing samplers assume that each dimension is independent within a small interval $\Delta t$ and update each dimension in parallel for efficiency .

### 2.2 Any-order autoregressive models

Any-order autoregressive models (AO-ARMs)  model the joint distribution autoregressively for all possible orders $\pi$ of the $d$ variables.
Formally, they factorize the joint distribution as $\prod_{k=1}^{d}p(x^{\pi(k)}|x^{\pi(<k)})$.
To learn such a distribution, an AO-ARM utilizes a weight-sharing neural network to model all univariate conditionals and employs mask tokens to represent absent variables.
During training, the expected negative log-likelihood over the uniform distribution of all orders $U_{\pi}$ is minimized:

$$ $\displaystyle\mathcal{L}_{AO}({\bm{x}}_{0})$ $\displaystyle=\mathbb{E}_{\pi\sim U_{\pi}}\sum_{l=1}^{d}-\log q_{\theta}(x_{0}^{\pi(l)}|x_{0}^{\pi(<l)};\pi).$ (2.16) $$

## 3 Reparameterized absorbing discrete diffusion

In [Section˜3.1](#S3.SS1), we reveal that the concrete score of absorbing discrete diffusion can be reparameterized as conditional distributions of clean data, which enables efficient sampling by caching the output of time-independent network (see [Section˜3.2](#S3.SS2)). In [Section˜3.3](#S3.SS3), we unify the training objective of absorbing discrete diffusion and AO-ARMs.

### 3.1 Reparameterizing the concrete score as conditional distributions of clean data

A key observation is that only the transition from the masked token to an unmasked token is valid in the reverse process of an absorbing discrete diffusion. In particular, according to the definition of the transition matrix of the absorbing process (see [Eq.˜2.9](#S2.E9)), we have ${\bm{Q}}^{\text{absorb}}(\hat{x}_{t}^{i},x_{t}^{i})=0$ for any unmasked $x_{t}^{i}\neq[\textbf{M}]$ and $\hat{x}_{t}^{i}\neq x_{t}^{i}$.
Therefore, the corresponding element in the transition matrix of the reverse process $\tilde{{\bm{Q}}}_{t}$ (see [Eq.˜2.10](#S2.E10)) equals zero. Namely,

$$ $\displaystyle\tilde{{\bm{Q}}}_{t}\left(x_{t}^{1}\ldots x_{t}^{i}\ldots x_{t}^{d},x_{t}^{1}\ldots\widehat{x}_{t}^{i}\ldots x_{t}^{d}\right)=\sigma(t){\bm{Q}}^{\text{absorb}}\left(\widehat{x}_{t}^{i},x_{t}^{i}\right)\frac{p_{t}\left(x_{t}^{1}\ldots\widehat{x}_{t}^{i}\ldots x_{t}^{d}\right)}{p_{t}\left(x_{t}^{1}\ldots x_{t}^{i}\ldots x_{t}^{d}\right)}=0,$ (3.1) $$

for any unmasked state $x_{t}^{i}\neq[\textbf{M}]$ and $\hat{x}_{t}^{i}\neq x_{t}^{i}$ and it is unnecessary to model the corresponding concrete score $\frac{p_{t}\left(x_{t}^{1}\ldots\widehat{x}_{t}^{i}\ldots x_{t}^{d}\right)}{p_{t}\left(x_{t}^{1}\ldots x_{t}^{i}\ldots x_{t}^{d}\right)}$.
Also, note that the concrete score always takes the value of one if $\hat{x}_{t}^{i}=x_{t}^{i}$. Therefore, we only need to characterize the concrete score for $x_{t}^{i}=[\textbf{M}]$ and $\hat{x}_{t}^{i}\neq[\textbf{M}]$.

Interestingly, in this case, we discover that the concrete score has a simple analytic form w.r.t. to the conditional distributions of clean data, as summarized in the following [Theorem˜1](#Thmtheorem1).

###### Theorem 1 .

(Analytic concrete score in absorbing diffusion, proof in [Appendix˜B](#A2)) For ${\bm{x}}_{t}=x_{t}^{1}\ldots x_{t}^{i}\ldots x_{t}^{d}$ and $\hat{{\bm{x}}}_{t}=x_{t}^{1}\ldots\widehat{x}^{i}_{t}\ldots x_{t}^{d}$, if $x_{t}^{i}=[\textbf{M}]$ and $\hat{x}_{t}^{i}\neq[\textbf{M}]$, the concrete score at time $t$ can be expressed as a time-independent conditional distribution at time zero multiplied by an analytic time-dependent term:

$$ $\underbrace{\frac{p_{t}\left(x_{t}^{1}\ldots\widehat{x}_{t}^{i}\ldots x_{t}^{d}\right)}{p_{t}\left(x_{t}^{1}\ldots x_{t}^{i}\ldots x_{t}^{d}\right)}}_{\displaystyle\text{concrete score}}=\underbrace{{\color[rgb]{1,.5,0}\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}}\vphantom{\frac{p_{t}\left(x_{t}^{1}\ldots\widehat{x}_{t}^{i}\ldots x_{t}^{d}\right)}{p_{t}\left(x_{t}^{1}\ldots x_{t}^{i}\ldots x_{t}^{d}\right)}}}_{\displaystyle\text{scalar}}~~\underbrace{{\color[rgb]{0,0,1}p_{0}(\hat{x}_{t}^{i}|{\bm{x}}_{t}^{\textrm{UM}})}\vphantom{\frac{p_{t}\left(x_{t}^{1}\ldots\widehat{x}_{t}^{i}\ldots x_{t}^{d}\right)}{p_{t}\left(x_{t}^{1}\ldots x_{t}^{i}\ldots x_{t}^{d}\right)}}}_{\begin{subarray}{c}\displaystyle\text{clean data}\\ \displaystyle\text{distribution}\end{subarray}}$ $$

where ${\bm{x}}_{t}^{\textrm{UM}}$ is the vector consists of all unmasked tokens of ${\bm{x}}_{t}$.

One immediate implication of Theorem [1](#Thmtheorem1) is to theoretically explain the benefit of the “scaling trick” in existing work  (see Appendix C.2 therein), which significantly improves the practical performance of discrete diffusion (see [Table˜1](#S4.T1)) but has not been fully understood before. In particular, the scaling trick divides the output of the score network by a factor. Equivalently, it reparameterizes ${\bm{s}}_{\theta}({\bm{x}}_{t},t)$ as follows:

$$ ${\bm{s}}_{\theta}({\bm{x}}_{t},t)={\color[rgb]{1,.5,0}\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}}\tilde{{\bm{s}}}_{\theta}({\bm{x}}_{t},t),$ (3.2) $$

where $\tilde{{\bm{s}}}_{\theta}({\bm{x}}_{t},t)$ is the output of the reparameterized score network and the scaling factor coincides with the time-dependent term in Theorem [1](#Thmtheorem1). In the original parameterization, the score network $s_{\theta}$ must model the whole time-dependent concrete score. In contrast, with the scaling trick, the reparameterized score $\tilde{{\bm{s}}}_{\theta}({\bm{x}}_{t},t)$ can focus on capturing the clean data distribution $p_{0}(\hat{x}^{i}|{\bm{x}}_{t}^{\textrm{UM}})$ and simplifies learning, according to Theorem [1](#Thmtheorem1).

Further, Theorem [1](#Thmtheorem1) suggests that the reparameterized score is essentially a conditional probability on clean data, which is time-independent. Motivated by this, we propose reparameterized absorbing discrete diffusion (RADD), which employs a time-independent network ${\bm{c}}_{\theta}({\bm{x}}_{t})$ that defines a model distribution $q_{\theta}$ by corresponding conditional distributions to approximate data distribution $p_{0}$ directly:

$$ ${\bm{c}}_{\theta}({\bm{x}}_{t})[i,\hat{x}_{t}^{i}]=q_{\theta}(\hat{x}_{t}^{i}|{\bm{x}}_{t}^{\textrm{UM}})\approx p_{0}(\hat{x}_{t}^{i}|{\bm{x}}_{t}^{\textrm{UM}}).$ (3.3) $$

In practice, we make a minimal modification of the score network in SEDD  for simplicity and fairness as shown in [Fig.˜1](#S3.F1). Specifically, we remove the time-conditioning input, reducing the architecture to a form similar to the standard GPT model, with softmax as the final activation. Further details can be found in [Section˜J.1](#A10.SS1).

Our reparameterization approach, which removes $t$, applies to both score and mean parameterizations, as detailed in [Appendix˜E](#A5). This not only simplifies the training target but also enables a more efficient sampling process than SEDD , as presented below.

Figure: Figure 1: Reparameterized network architecture vs. SEDD (DiT). Our network simplifies the original DiT by removing time conditions and outputs the conditional distributions on clean data, similar to the standard Transformer. For example, with a vocabulary of {R, A, D}, only the probabilities in the columns corresponding to the [M] token are meaningful (highlighted in color), since the network is designed to learn to denoise the masked input. The remaining output, shown in grey, will not be utilized.
Refer to caption: 2406.03736v4/x1.png

###### Theorem 1 .

### 3.2 Efficient samplers to reduce NFEs by caching the output of RADD

In the reverse process of an absorbing discrete diffusion, once a token transitions from $[\textbf{M}]$ to an unmasked token, it remains unchanged. Consequently, for a sequence consisting of $d$ tokens, there will be at most $d$ intervals during the sampling process where changes occur, regardless of the number of sampling steps $D$. In the remaining steps, the sequence remains unchanged across all $d$ dimensions.
This property allows us to cache ${\bm{c}}_{\theta}(x_{t})$ to avoid the need to reevaluate the time-independent ${\bm{c}}_{\theta}$ when ${\bm{x}}_{t}$ is unchanged in the previous step (see [Appendix˜I](#A9) for the pseudo-code). However, since SEDD is conditioned on time, it does not support this caching strategy for reducing NFEs.

The NFEs with the caching strategy is a random variable. To quantify it, we calculate the expected NFEs (E-NFEs) in analytic form, conditioned on the sampling method, time steps, and noise schedule. For instance, using the Tweedie $\tau$-leaping method with a log-linear noise schedule , the E-NFEs can be expressed by sampling steps $n$ and generating length $l$
(^1^11A similar analysis was presented in for their discrete-time sampler, although it is based on different assumptions and applicable to different scenarios. A detailed comparison is provided in [Appendix F](#A6).)
(proof in [Section˜D.5](#A4.SS5)):

$$ $\text{E-NFEs}(n)=n(1-(1-\frac{1}{n})^{l}),$ (3.4) $$

In [Fig.˜2(a)](#S3.F2.sf1), we plot the curve of [Eq.˜3.4](#S3.E4) in blue, which aligns well with our experiments (red stars). This demonstrates that our method theoretically reduces E-NFEs, particularly at larger sampling steps. This reduction is also supported by [Fig.˜2(b)](#S3.F2.sf2), where RADD shows faster convergence trends.

Furthermore, based on [Theorem˜1](#Thmtheorem1), simplified forms of the reverse process for both Euler method and Tweedie $\tau$ -leaping method can be derived, which leads to corresponding analytic forms of E-NFEs given time steps and noise schedule.
We also prove that these two sampling methods are equivalent under a log-linear noise schedule for absorbing discrete diffusion (see [Appendix˜D](#A4) for more details).

### 3.3 Unifying absorbing discrete diffusion and any-order autoregressive model

Building upon Theorem [1](#Thmtheorem1), we further prove the equivalence between absorbing discrete diffusion and any-order
autoregressive models introduced in [Section˜2.2](#S2.SS2), as presented in the following theorem.

###### Theorem 2 .

The absorbing discrete diffusion objective of [Eq.˜2.11](#S2.E11) is equivalent to any-order autoregressive objective of [Eq.˜2.16](#S2.E16) when the final total noise level $\bar{\sigma}(T)\rightarrow+\infty$.

The proof of [Theorem˜2](#Thmtheorem2) consists of three key steps, which introduce three different yet equivalent loss functions. Below we briefly present the key ideas and defer the proof in [Appendix˜C](#A3).

In the first step, by removing the terms ${\bm{s}}_{\theta}\left({\bm{x}}_{t},t\right)_{\hat{{\bm{x}}}_{t}}$ and $K\left(\frac{p_{t\mid 0}\left({\hat{{\bm{x}}}_{t}}\mid{\bm{x}}_{0}\right)}{p_{t\mid 0}\left({\bm{x}}_{t}\mid{\bm{x}}_{0}\right)}\right)$ in [Eq.˜2.11](#S2.E11), we can define
a simpler loss $\mathcal{L}_{t\textrm{-DCE}}^{T}$ called $t$-denoising cross-entropy loss (abbr. $t\textrm{-DCE}$), which is equivalent to DSE loss. In the multi-dimensional case, it has the form:

$$ $\mathcal{L}_{t\textrm{-DCE}}^{T}({\bm{x}}_{0})=\int_{0}^{T}\mathbb{E}_{{\bm{x}}_{t}\sim p_{t|0}({\bm{x}}_{t}|{\bm{x}}_{0})}\left[\sum_{x_{t}^{i}=[\textbf{M}]}-\frac{\sigma(t)e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}\log\left(\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}q_{\theta}(x_{0}^{i}|{\bm{x}}_{t}^{\textrm{UM}})\right)\right]dt$ (3.5) $$

We emphasize that [Eq.˜3.5](#S3.E5) holds in a nonparametric setting because RADD can be interpreted as a model distribution $q_{\theta}$ representing the conditional distribution of clean data, which approximates the true distribution $p_{0}$, as proven in [Theorem˜1](#Thmtheorem1).

In the second step, inspired by  , we change the variable from $t$ to $\lambda(t)=1-e^{-\bar{\sigma}(t)}$, which represents the probability of a token being masked in $[0,t]$ during the forward process. Thus, $\mathcal{L}^{T}_{t\textrm{-DCE}}({\bm{x}}_{0})$ can be rewritten as an integral of $\lambda$, defined as $\lambda$-denoising cross-entropy loss (abbr. $\lambda\textrm{-DCE}$):

$$ $\mathcal{L}_{\lambda\textrm{-DCE}}({\bm{x}}_{0}):=\int_{0}^{1}\frac{1}{\lambda}\mathbb{E}_{{\bm{x}}_{\lambda}\sim p_{\lambda}({\bm{x}}_{\lambda}|{\bm{x}}_{0})}\left[\sum_{x_{\lambda}^{i}=[\textbf{M}]}-\log q_{\theta}({\bm{x}}_{0}^{i}|{\bm{x}}_{\lambda}^{\textrm{UM}})\right]d\lambda,$ (3.6) $$

where $p_{\lambda}({\bm{x}}_{\lambda}|{\bm{x}}_{0})$ is the joint distribution induced by masking each dimension in ${\bm{x}}_{0}$ independently with a probability $\lambda$.

Finally, we prove that $\lambda\textrm{-DCE}$ loss in [Eq.˜3.6](#S3.E6) can be integrated analytically and rewritten as $\mathcal{L}_{AO}$ in [Eq.˜2.16](#S2.E16). We summarize our proof procedure by the equivalence between these losses:

$$ $\mathcal{L}_{\textrm{DSE}}^{T}({\bm{x}}_{0})\overset{\textrm{Appendix}~\ref{subsec:DSE/DCE}}{\iff}\mathcal{L}_{t\textrm{-DCE}}^{T}({\bm{x}}_{0})\overset{\textrm{Appendix}~\ref{subsec:DCEloss_lambda}}{\iff}\mathcal{L}_{\lambda\textrm{-DCE}}({\bm{x}}_{0})\overset{\textrm{Appendix}~\ref{subsec:DCEloss_k}}{\iff}\mathcal{L}_{AO}({\bm{x}}_{0}).$ (3.7) $$

A direct benefit from [Theorem˜2](#Thmtheorem2) is that we can use an absorbing discrete diffusion model to sample like AO-ARM and vice versa. For training and likelihood evaluation, the four losses in [Eq.˜3.7](#S3.E7) can also be used (see [Appendix˜I](#A9) for pseudo-code). To efficiently estimate the four losses using Monte Carlo methods, we can replace the sum or integral with an expectation. Take [Eq.˜3.6](#S3.E6) for example, it can be rewritten as the following form of expectation on $\lambda$:

$$ $\mathcal{L}_{\lambda\textrm{-DCE}}({\bm{x}}_{0})=\mathbb{E}_{\lambda\sim U([0,1])}\frac{1}{\lambda}\mathbb{E}_{{\bm{x}}_{\lambda}\sim p_{\lambda}({\bm{x}}_{\lambda}|{\bm{x}}_{0})}\left[\sum_{x_{\lambda}^{i}=[\textbf{M}]}-\log q_{\theta}({\bm{x}}_{0}^{i}|{\bm{x}}_{\lambda}^{\textrm{UM}})\right].$ (3.8) $$

Additionally, [Theorem˜2](#Thmtheorem2) provides a new perspective on the DSE loss. While it has been traditionally viewed as an upper bound on the negative log-likelihood for the diffusion model, it can also be interpreted as an expected negative log-likelihood over factorial numbers of orderings for AO-ARM by [Eq.˜2.16](#S2.E16). As discussed in , for different orders $\pi$, $q_{\theta}(x_{0};\pi)$ will be inconsistent in general. Despite this inconsistency, it can be viewed as an ensemble of multiple autoregressive models with different orders, potentially more robust than fixed-order models.

We mention that establish the equivalence between ARMs and the ELBO of absorbing diffusion models. In comparison, built upon our [Theorem˜1](#Thmtheorem1), we extend existing work by unifying four loss functions in [Eq.˜3.7](#S3.E7), deepening the understanding of absorbing discrete diffusion. We provide a detailed discussion in [Appendix˜H](#A8) and a systematic empirical study in [Section˜4.3](#S4.SS3).

###### Theorem 2 .

## 4 Experiments

We present the experimental setups in [Section˜4.1](#S4.SS1).
We then evaluate the performance of accelerated generation in [Section˜4.2](#S4.SS2) and zero-shot perplexity on various language datasets in [Section˜4.3](#S4.SS3).

### 4.1 Settings

Below, we briefly present the experimental settings. For more details, please see [Appendix˜J](#A10).

#### Model.

We use RADD model ${\bm{c}}_{\theta}$ reparameterzied as described in [Section˜3.1](#S3.SS1). Compared with SEDD models, RADD models have fewer parameters because the time-condition has been eliminated. We trained our RADD model ${\bm{c}}_{\theta}$ using denoising score entropy, $t$-denoising cross-entropy, $\lambda$-denoising cross-entropy and any-order autoregressive loss, abbreviated as RADD-DSE, RADD-$t\textrm{-DCE}$, RADD-$\lambda\textrm{-DCE}$ and RADD-AO, since all these models share the same architecture as described in [Fig.˜1](#S3.F1). For the SEDD small and medium model, we employed their pre-trained model. When performing text generation tasks, we used RADD-$\lambda\textrm{-DCE}$ medium models.

#### Data.

Following SEDD, we trained on the OpenWebText  dataset and tested on the LAMBADA, WikiText2, PTB, WikiText103, and One Billion Words datasets . For data splits and data processing, we adopted the same settings and techniques as SEDD, which involves packing sentences to generate uniform-length blocks as model input.

#### Training setup.

We used a log-linear noise schedule where the expectation of the number of changed tokens at time $t$ is linear with $t$.
Following SEDD, we report results for RADD trained over 400K iterations in [Tables˜1](#S4.T1) and [2](#S4.T2).
To further analyze the convergence and performance trends, we extended the training of RADD-small to 1,000K iterations, detailed in [Table˜7](#A10.T7) of the appendix.

#### Metric.

Following , we conduct experiments on unconditional generation and language modeling tasks.
For language modeling tasks, we report the perplexity calculated on the dataset with different models.
For generation, we assess sample quality using perplexity (PPL) on unconditional samples measured by an additional larger language model (i.e., GPT-2 large), and sample diversity through unigram entropy .

### 4.2 Efficient sampling

As shown in Fig.1 of , SEDD surpasses AR in terms of sampling speed. Therefore, we compare the sample quality between SEDD and our RADD model measured by perplexity. As shown in [Fig.˜2(b)](#S3.F2.sf2), RADD with the caching strategy is more efficient than SEDD. This improvement is expected because the NFEs is limited by the generating sequence length. We further conducted batch size ablation and compare the running time and unigram entropy as detailed in [Section˜J.4](#A10.SS4).

As discussed in [Section˜3.3](#S3.SS3), we can also use RADD as an any-order autoregressive model to generate samples in different orders,
as detailed in [Section˜J.4](#A10.SS4). We present more sampling details in [Section˜J.3](#A10.SS3). and the generated samples in [Section˜K.1](#A11.SS1).

### 4.3 Improved zero-shot perplexity on language modeling

Following SEDD, we present zero-shot perplexities on the LAMBADA, WikiText2, PTB, WikiText103, and 1 Billion Words datasets  in [Tables˜1](#S4.T1) and [2](#S4.T2)
and compare the zero-shot perplexity of our model with other baseline models . Perplexities of RADD models are calculated based on their corresponding loss (e.g., RADD-$\lambda\textrm{-DCE}$ on $\mathcal{L}_{\lambda\textrm{-DCE}})$, which is valid for likelihood estimation as discussed in [Section˜3.3](#S3.SS3).

Firstly, we conduct an ablation study of the scaling trick in the middle of the [Tables˜1](#S4.T1) and [2](#S4.T2). For the absorbing diffusion, the perplexity of the scaled version of SEDD outperforms its unscaled version, which matches our theoretical discovery in [Theorem˜1](#Thmtheorem1). Secondly, under the same DSE loss and similar parameter counts, we observed that the RADD-DSE model without time-conditioning outperforms the SEDD-Scale model with time-conditioning. This ablation validates our analysis in [Section˜3.1](#S3.SS1), indicating that time-conditioning is unnecessary for absorbing discrete diffusion models.
Additionally, while RADD models trained with four equivalent loss functions achieve similar performance, minor discrepancies persist. These differences stem from variations in gradient estimation on finite data, leading models to converge at distinct local optima despite the theoretical equivalence of their objectives on expectation.
Overall, all RADD losses outperform SEDD on average across the five datasets, validating our analysis in [Sections˜3.1](#S3.SS1) and [3.3](#S3.SS3).

**Table 1: Zero-shot language modeling perplexity ($\downarrow$) on five datasets using small models. "SEDD-Unscale" and "SEDD-Scale" refer to the unscaled and scaled versions of the absorbing models, respectively. All SEDD and RADD models are trained for 400k iterations. Results for other diffusion models are based on the upper bound from . For RADD models, the results are calculated based on the corresponding loss.**
| Method | LAMBADA | WikiText2 | PTB | WikiText103 | 1BW |
| --- | --- | --- | --- | --- | --- |
| GPT-2 | 45.04 | 42.43 | 138.43 | 41.60 | 75.20 |
| D3PM | 93.47 | 77.28 | 200.82 | 75.16 | 138.92 |
| PLAID | 57.28 | 51.80 | 142.60 | 50.86 | 91.12 |
| SEDD-Uniform | 65.40 | 50.27 | 140.12 | 49.60 | 101.37 |
| SEDD-Unscale | 52.21 | 44.75 | 130.49 | 43.14 | 80.70 |
| SEDD-Scale | 50.92 | 41.84 | 114.24 | 40.62 | 79.29 |
| RADD-DSE | 49.57 | 38.83 | 111.74 | 37.46 | 72.35 |
| RADD-$t\textrm{-DCE}$ | 50.56 | 39.02 | 109.03 | 36.38 | 72.60 |
| RADD-$\lambda\textrm{-DCE}$ | 51.70 | 39.98 | 107.85 | 37.98 | 72.99 |
| RADD-AO | 50.27 | 38.26 | 110.38 | 35.90 | 74.28 |

**Table 2: Zero-shot language modeling perplexity ($\downarrow$) on five datasets using medium models. "SEDD-Unscale" and "SEDD-Scale" refer to the unscaled and scaled versions of the absorbing models, respectively. All SEDD and RADD models are trained for 400k iterations.**
| Method | LAMBADA | WikiText2 | PTB | WikiText103 | 1BW |
| --- | --- | --- | --- | --- | --- |
| GPT-2 | 35.66 | 31.80 | 123.14 | 31.39 | 55.72 |
| SEDD-Unscale | 44.60 | 34.85 | 93.26 | 32.97 | 67.91 |
| SEDD-Scale | 42.77 | 31.04 | 87.12 | 29.98 | 61.19 |
| RADD-DSE | 42.30 | 29.17 | 75.16 | 28.03 | 57.45 |
| RADD-$t\textrm{-DCE}$ | 43.24 | 30.19 | 78.77 | 29.36 | 57.95 |
| RADD-$\lambda\textrm{-DCE}$ | 44.10 | 30.60 | 82.08 | 29.29 | 60.32 |
| RADD-AO | 41.96 | 29.96 | 79.06 | 28.51 | 57.07 |

## 5 Related work

#### Continouous-state diffusion models for text generation.

Several works have been proposed to apply continuous diffusion to text . use an embedding layer to map discrete tokens to a latent space and learn a continuous-state diffusion on it. Bit Diffusion  learns a continuous diffusion model to generate binary bits of discrete tokens. However, transforming between these continuous representations and discrete tokens by thresholding may lose information. Bayesian Flow Network  achieves competitive log-likelihood on character-level language modeling tasks and is proven equivalent to continuous stochastic differential equations trained by denoising score matching . Such models underperform auto-regressive models on standard text generation tasks.

#### Discrete-state diffusion models for text generation.

Several discrete-state diffusion models have been proposed . D3PM  proposed a diffusion framework based on any probability transition matrix and trained with a lower bound of log-likelihood. DiffusionBERT  utilizes a pre-trained BERT  as an initialization of diffusion. Furthermore,  generalizes the framework to continuous time by introducing a rate matrix. It is difficult to apply the score matching in such models because the gradient of the data distribution is undefined. Several works try to generalize the score matching on discrete data .  introduce the concrete score and the denoising concrete score matching loss. Furthermore, SEDD bridges the discrete state diffusion and the concrete score by introducing a denoising score entropy loss . By incorporating an absorbing process, SEDD achieves competitive performance with the auto-regressive models, especially, GPT-2. Motivated by the success of absorbing discrete diffusion, RADD is specifically designed for this class of models. Due to fundamental differences in score formulations, it can not directly apply to other models like multinomial diffusion. proposed discrete flow models. proposed a discrete non-Markov diffusion model to accelerate sampling, which has some connections to our cache-based acceleration in [Section˜3.2](#S3.SS2). A detailed comparison can be found in [Appendix˜F](#A6).

#### Concurrent works

We mention that and independently conducted related studies on absorbing discrete diffusion. We provide a detailed discussion here.

derived a weighted integral of cross-entropy loss in their Eq.(5) similar to our $t\textrm{-DCE}$ loss in [Eq.˜3.5](#S3.E5). Besides, their Proposition 1, which connects the score parameterization and the mean parameterization(^2^22Our conclusions are based on score parameterization but can be extended to mean prediction parameterization (please see [Appendix E](#A5)).), also resembles our [Theorem˜1](#Thmtheorem1). In comparison, we simplified the conditional expectation term (related to $t$) in Proposition 1  to a time-independent conditional probability at time zero. Motivated by the finding, we proposed a simpler parameterization that enables fast sampling. It is worth noting that the hyperparameters they selected significantly contribute to the model’s performance, which also applies to our RADD models (see [Section˜J.2](#A10.SS2) for details). In addition, proposed a generalized masked diffusion model allowing state-dependent masking schedules.

derive the same cross-entropy losses with .
Despite lacking a theoretical foundation, they conducted time-conditioning ablation which shows that time-conditioning has minimal impact on perplexity. Notably, their method for removing time conditioning retained the same network structure (e.g., keeping adaptive layer normalization) while setting the time input to zero uniformly. In contrast, our approach removes the network structure related to time inputs entirely, eliminating the need for time input and thereby simplifying the network design. They also proposed a caching strategy to accelerate sampling. While this coincides with our work in [Section˜3.2](#S3.SS2), we present a complete theoretical analysis of E-NFEs to quantify the acceleration efficiency.

Our unique contribution lies in the decomposition of the concrete score and time-independent parameterization, serveing as the foundation for subsequent contributions in [Sections˜3.2](#S3.SS2) and [3.3](#S3.SS3).

## 6 Conclusion

We introduce RADD, a dedicated discrete diffusion model that characterizes the time-independent conditional probabilities, built upon a new factorization form of the concrete score. RADD is more efficient by reducing the NFEs with a cache strategy while maintaining comparable performance to strong baselines. Additionally, we demonstrated the unification of training objectives for absorbing discrete diffusion and AO-ARMs. On five zero-shot language modeling benchmarks, our RADD models achieve state-of-the-art performance at the GPT-2 scale.

#### Limitaition.

Our model has been trained and evaluated primarily on the GPT-2 scale. For broader applicability, it is essential to explore the effects of scaling on the performance , which is left as future work. The success of diffusion transformers on images  and videos  suggests that diffusion models can be scaled up by incorporating transformers.

Another limitation is that our model can only generate full-length outputs, unlike auto-regressive models that can produce variable-length outputs. This restricts the flexibility of our model in certain applications. We leave the investigation on this issue as future work.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (No. 92470118); the Beijing Nova Program (No. 20220484044); Beijing Natural Science Foundation (No. L247030).

We thank Aaron Lou for his prompt and detailed responses to our inquiries, which greatly assisted our research. We also thank Zebin You for his support in setting up the coding environment.

Ethics statement. For the current theoretical and experimental scope of this paper, we have not found any direct social impacts. However, considering future developments, the paper potentially contributes to the next-generation large language models. In this context, this work could significantly reduce the inference cost of language models but may also lead to hallucinations, amplify biases and discrimination in the data, and pose risks of misuse. As with other generative models, addressing these issues requires further advancements in the field.

Reproducibility statement
We have open-sourced our code in [https://github.com/ML-GSAI/RADD](https://github.com/ML-GSAI/RADD). For detailed instructions on environment setup and running scripts, please refer to the README.md file. Comprehensive explanations and proofs of our theoretical claims can be found in [Appendices˜B](#A2) and [C](#A3).

## Appendix A Detailed notations and definitions

We introduce the notations used throughout the paper. Let lower, boldface lower and upper case letters represent scalers (e.g., $a$), vectors (e.g., ${\bm{a}}$), and matrices (e.g., ${\bm{A}}$), respectively. For a vector ${\bm{a}}$, $a^{i}$ denotes its $i$-th element. For a matrix ${\bm{A}}$, ${\bm{A}}(i,j)$ denotes $(i,j)$-th element. For a vector function ${\bm{f}}$, ${\bm{f}}({\bm{x}})_{i}$ denotes the $i$-th element of ${\bm{f}}({\bm{x}})$. Constants and random variables are not distinguished in the notation if there is no confusion. We represent the distributions of the forward and reverse processes by $p$ and $q_{\theta}$ respectively. The transition probability from time $s$ to time $t$ is denoted by $p_{t|s}(\cdot|\cdot)$, and the probability at time $t$ is denoted by $p_{t}(\cdot)$. Complete notations and definitions are listed below:

- •
$x$, $\hat{x}$: Scalar variables representing states in a model.
- •
$\mathcal{X}$: A one-dimensional sample space $\{1,\cdots,N\}$.
- •
${\bm{Q}}_{t}$: The transition rate matrix at time $t$.
- •
$p$: The probability of the forward process defined by the transition rate matrix ${\bm{Q}}_{t}$.
- •
$q_{\theta}$: The probability of the reverse process defined by model ${\bm{c}}_{\theta}$.
- •
$p_{t|s}(\hat{x}|x)$: The transition probability from state $x$ to state $\hat{x}$ from time $s$ to time $t$.
- •
$p_{t}(x)$: The probability of $x$ at time $t$.
- •
${\bm{P}}_{t|s}$: The transition probability matrix from time $s$ to time $t$.
- •
$\sigma(t)$: The noise schedule function.
- •
$\tilde{{\bm{Q}}}_{t}$: The reverse transition rate matrix at time $t$.
- •
${\bm{s}}_{\theta}({\bm{x}}_{t},t)_{{\hat{{\bm{x}}}_{t}}}$: The corresponding element of ${\bm{s}}_{\theta}({\bm{x}}_{t},t)$,
which approximates
$\frac{p_{t}({\hat{{\bm{x}}}_{t}})}{p_{t}({\bm{x}})}$.
- •
$[\textbf{M}]$: A special mask token in the absorbing process.
- •
$\mathcal{X}^{d}$: A multi-dimensional sample space $\{1,\cdots,N\}^{d}$.
- •
${\bm{x}}_{t}$: A multi-dimensional vector.
- •
$x_{t}^{i}$: The $i$-th element of ${\bm{x}}_{t}$.
- •
$p_{s|t}^{i}(\cdot|{\bm{x}}_{t})$: The probability on dimension $i$ from time $s$ to time $t$ conditioned on full vector ${\bm{x}}_{t}$.
- •
$p_{s|t}^{\text{tweedie}}(\cdot|\cdot)$: The transition probability from time $s$ to time $t$ under Tweedie $\tau$-leaping method.
- •
$p_{s|t}^{\text{euler}}(\cdot|\cdot)$: The transition probability from time $s$ to time $t$ under the Euler method.
- •
${\bm{Q}}^{\text{tok}}_{t}$: Transition rate matrix for each dimension of ${\bm{x}}_{t}$.
- •
${\bm{x}}^{\textrm{UM}}$: vector consists of all unmasked tokens of ${\bm{x}}$.
- •
${\bm{x}}^{a:b}$: The elements of ${\bm{x}}$ with indices ranging from $a$ to $b$.
- •
${\bm{c}}_{\theta}({\bm{x}}_{t})$: A network that characterizes the time-independent conditional probabilities in reparameterized absorbing discrete diffusion (RADD).
- •
$d$: Total sequence length or dimension of ${\bm{x}}$.
- •
$l$: Generating sequence length.
- •
$\pi$: one permutation, $\pi(l)$ denotes the $l$-th element of permutation $\pi$, $\pi(<l)$ denotes the elements of permutation $\pi$ with indices less than $l$.
- •
$U(\cdot)$: Uniform distribution.
- •
$p_{\lambda}(\cdot|{\bm{x}}_{0})$: The joint distribution induced by masking each dimension in ${\bm{x}}_{0}$ independently with a probability $\lambda$.
- •
Cat: Categorical distribution.
- •
NFEs: Number of function evaluations.
- •
E-NFEs: Expected number of function evaluations.

## Appendix B Proof of Theorem 1

In this section, we provide a detailed proof of [Theorem˜1](#Thmtheorem1), which is carried out in three key steps. The core idea of the proof involves leveraging the properties of a continuous-time Markov chain with an absorbing state, where the forward diffusion process is independent across different dimensions. This independence simplifies the analysis of both the conditional and joint distributions.

First, we derive the analytic form of the conditional distribution, as stated in [Lemma˜1](#Thmlemma1). This can be derived directly from [Eq.˜2.3](#S2.E3), but for a better understanding, we provide a more intuitive proof for ${\bm{Q}}_{t}=\sigma(t){\bm{Q}}^{\text{absorb}}$. Second, we extend this analysis to multiple dimensions to obtain the joint distribution, as formalized in [Proposition˜1](#Thmproposition1). Finally, by simply dividing the joint distributions derived in the second step, we decouple the concrete score, thereby completing the proof of [Theorem˜1](#Thmtheorem1).

###### Lemma 1 .

(Analytic conditional distribution in absorbing diffusion)
Suppose $\{X_{t}\}$ is a continuous time Markov chain with transition rate matrix ${\bm{Q}}_{t}=\sigma(t){\bm{Q}}^{\text{absorb}}$, given the value $x_{0}$ at time zero , the conditional distribution $p_{t|0}(x_{t}|x_{0})$ has the following analytic form:

$$ $p_{t|0}(x_{t}|x_{0})=\begin{cases}e^{-\bar{\sigma}(t)},&x_{t}=x_{0},\\ 1-e^{-\bar{\sigma}(t)},&x_{t}=[\textbf{M}],\\ 0,&x_{t}\neq[\textbf{M}]\ \text{and}\ x_{t}\neq x_{0}.\end{cases}$ (B.1) $$

###### Proof.

Given the initial value $x_{0}\in\mathcal{X}=\{1,\cdots,N\}$, we have

$$ $x_{t}=\begin{cases}x_{0},&t<T_{h},\\ [\textbf{M}],&t\geq T_{h}.\end{cases}$ (B.2) $$

Here, $T_{h}$ represents the holding time before $x_{0}$ transitions to the absorbing state [M].

Based on the definition of the ${\bm{Q}}_{t}$ in [Eq.˜2.2](#S2.E2) and ${\bm{Q}}^{\text{absorb}}$, the probability of $x_{0}$ remaining the same after a small time increment $\Delta t$ is

$$ $p_{t+\Delta t|t}(x_{0}|x_{0})=1+\sigma(t){\bm{Q}}^{\text{absorb}}(x_{0},x_{0})\Delta t+o(\Delta t).$ (B.3) $$

Partitioning the interval $[0,t]$ into $\{s_{k}\}_{k=0}^{n}$ and utilizing the memoryless property of continuous-time Markov chains, we can express the probability of $x_{0}$ remaining the same from time 0 to $t$ as a product of probabilities over these small intervals. This gives us:

$$ $\displaystyle p_{t|0}(x_{0}|x_{0})$ $\displaystyle=\prod_{k=1}^{n}p_{s_{k}|s_{k-1}}(x_{0}|x_{0})$ (B.4) $\displaystyle=\prod_{k=1}^{n}\left(1+\sigma(t_{k-1}){\bm{Q}}^{\text{absorb}}(x_{0},x_{0})(s_{k}-s_{k-1})+o(s_{k}-s_{k-1})\right)$ (B.5) $\displaystyle=\exp\left(\sum_{k=1}^{n}\ln\left(1+\sigma(t_{k-1}){\bm{Q}}^{\text{absorb}}(x_{0},x_{0})(s_{k}-s_{k-1})+o\left(s_{k}-s_{k-1}\right)\right)\right)$ (B.6) $\displaystyle=\exp\left(\sum_{k=1}^{n}\sigma(t_{k-1}){\bm{Q}}^{\text{absorb}}(x_{0},x_{0})(s_{k}-s_{k-1})+o(s_{k}-s_{k-1})\right).$ (B.7) $$

Let $\max(s_{k}-s_{k-1})\to 0$ , the Riemann sum in [Eq.˜B.7](#A2.E7) equals the following continuous integral:

$$ $p_{t|0}(x_{0}|x_{0})=\exp\left(\int_{0}^{t}\sigma(s){\bm{Q}}^{\text{absorb}}(x_{0},x_{0})ds\right)=\exp\left({\bm{Q}}^{\text{absorb}}(x_{0},x_{0})\bar{\sigma}(t)\right).$ (B.8) $$

By [Eq.˜2.9](#S2.E9), ${\bm{Q}}^{\text{absorb}}(x_{0},x_{0})=-1$, we have

$$ $\displaystyle p_{t|0}(x_{0}|x_{0})=P(T_{h}>t)=e^{-\bar{\sigma}(t)}$ (B.9) $\displaystyle p_{t|0}([\textbf{M}]|x_{0})=P(T_{h}\leq t)=1-e^{-\bar{\sigma}(t)}$ (B.10) $\displaystyle p_{t|0}(k|x_{0})=0\quad\text{if}\ k\neq[\textbf{M}]\ \text{and}\ k\neq x_{0}.$ (B.11) $$

Similarly, given value $x_{s}$ at time $s<t$, the conditional distribution can be expressed as

$$ $p_{t|s}(x_{t}|x_{s})=\begin{cases}e^{-\left(\bar{\sigma}(t)-\bar{\sigma}(s)\right)},&x_{t}=x_{s},\\ 1-e^{-\left(\bar{\sigma}(t)-\bar{\sigma}(s)\right)},&x_{t}=[\textbf{M}],\\ 0,&x_{t}\neq[\textbf{M}]\ \text{and}\ x_{t}\neq x_{s}.\end{cases}$ (B.12) $$

∎

###### Proposition 1 .

(Analytic joint distribution in absorbing diffusion)

Suppose $\{X_{t}\}$ is a continuous time Markov chain with transition rate matrix ${\bm{Q}}_{t}=\sigma(t){\bm{Q}}^{\text{absorb}}$. For ${\bm{x}}_{t}=x_{t}^{1}\cdots x_{t}^{d}$ with $d_{1}$ components as $[\textbf{M}]$ and $d_{2}=d-d_{1}$ components as unmasked tokens, $p_{t}({\bm{x}}_{t})$ can be expressed as

$$ $p_{t}({\bm{x}}_{t})=[1-e^{-\bar{\sigma}(t)}]^{d_{1}}[e^{-\bar{\sigma}(t)}]^{d_{2}}p_{0}({\bm{x}}_{t}^{\textrm{UM}}),$ (B.13) $$

where ${\bm{x}}_{t}^{\textrm{UM}}$ is the vector consists of all unmasked tokens of ${\bm{x}}_{t}$.

[Proposition˜1](#Thmproposition1) shows that the joint distribution $p_{t}({\bm{x}}_{t})$ can be expressed as the multiplication of two terms. One is an analytic term only depending on time, the other is a $d_{2}$ dimensions joint distribution of clean data $p_{0}({\bm{x}}_{t}^{\textrm{UM}})$ independent of time.

###### Proof.

Without loss of generality, let’s assume that the preceding $d_{1}$ terms of ${\bm{x}}$ are all [M], and the remaining $d_{2}$ terms are unmasked tokens. That is, ${\bm{x}}_{t}=[\textbf{M}]\cdots[\textbf{M}]x_{t}^{d_{1}+1}\cdots x_{t}^{d}$, and here $x^{k}$ is an unmasked token in $\mathcal{X}$.

Using the law of total probability and Lemma [1](#Thmlemma1), along with the assumption of independence between different dimensions of the diffusion process, we can express the joint distribution $p_{t}([\textbf{M}]\cdots[\textbf{M}]x_{t}^{d_{1}+1}\cdots x_{t}^{d})$ as a sum over all possible initial states ${\bm{x}}_{0}\in\mathcal{X}^{d}$:

$$ $\displaystyle p_{t}([\textbf{M}]\cdots[\textbf{M}]x_{t}^{d_{1}+1}\cdots x_{t}^{d})=$ $\displaystyle\sum_{{\bm{x}}_{0}\in\mathcal{X}^{d}}p_{t|0}([\textbf{M}]\cdots[\textbf{M}]x_{t}^{d_{1}+1}\cdots x_{t}^{d}|{\bm{x}}_{0})p_{0}({\bm{x}}_{0})$ $\displaystyle=$ $\displaystyle\sum_{x_{0}^{1}\in\mathcal{X},\cdots,x_{0}^{d}\in\mathcal{X}}p_{t|0}([\textbf{M}]\cdots[\textbf{M}]x_{t}^{d_{1}+1}\cdots x_{t}^{d}|x_{0}^{1}\cdots x_{0}^{d})p_{0}(x_{0}^{1}\cdots x_{0}^{d})$ $\displaystyle=$ $\displaystyle\sum_{x_{0}^{1}\in\mathcal{X},\cdots,x_{0}^{d}\in\mathcal{X}}\prod_{k=1}^{d_{1}}p_{t|0}^{k}([\textbf{M}]|x_{0}^{k})\prod_{k=d_{1}+1}^{d}p_{t|0}^{k}(x_{t}^{k}|x_{0}^{k})p_{0}(x_{0}^{1}\cdots x_{0}^{d}).$ $$

Substituting the analytic forms of $p_{t|0}^{k}([\textbf{M}]|x_{0}^{k})$ and $p_{t|0}^{k}(x_{t}^{k}|x_{0}^{k})$ from Lemma [1](#Thmlemma1), above equations can be further simplified as follows:

$$ $\displaystyle\sum_{x_{0}^{1}\in\mathcal{X},\cdots,x_{0}^{d}\in\mathcal{X}}\prod_{k=1}^{d_{1}}p_{t|0}^{k}([\textbf{M}]|x_{0}^{k})\prod_{k=d_{1}+1}^{d}p_{t|0}^{k}(x_{t}^{k}|x_{0}^{k})p_{0}(x_{0}^{1}\cdots x_{0}^{d})$ $\displaystyle=$ $\displaystyle\sum_{x_{0}^{1}\in\mathcal{X},\cdots,x_{0}^{d_{1}}\in\mathcal{X}}\prod_{k=1}^{d_{1}}p_{t|0}^{k}([\textbf{M}]|x_{0}^{k})[e^{-\bar{\sigma}(t)}]^{d_{2}}p_{0}(x_{0}^{1}\cdots x_{0}^{d_{1}}x_{t}^{d_{1}+1}\cdots x_{t}^{d})$ $\displaystyle=$ $\displaystyle\sum_{x_{0}^{1}\in\mathcal{X},\cdots,x_{0}^{d_{1}}\in\mathcal{X}}[1-e^{-\bar{\sigma}(t)}]^{d_{1}}[e^{-\bar{\sigma}(t)}]^{d_{2}}p_{0}(x_{0}^{1}\cdots x_{0}^{d_{1}}x_{t}^{d_{1}+1}\cdots x_{t}^{d})$ $\displaystyle=$ $\displaystyle[1-e^{-\bar{\sigma}(t)}]^{d_{1}}[e^{-\bar{\sigma}(t)}]^{d_{2}}\sum_{x_{0}^{1}\in\mathcal{X},\cdots,x_{0}^{d_{1}}\in\mathcal{X}}p_{0}(x_{0}^{1}\cdots x_{0}^{d_{1}}x_{t}^{d_{1}+1}\cdots x_{t}^{d})$ $\displaystyle=$ $\displaystyle[1-e^{-\bar{\sigma}(t)}]^{d_{1}}[e^{-\bar{\sigma}(t)}]^{d_{2}}p_{0}(x_{t}^{d_{1}+1}\cdots x_{t}^{d}).$ $$

By noting that $p_{0}(x_{t}^{d_{1}+1}\cdots x_{t}^{d})=p_{0}({\bm{x}}_{t}^{\textrm{UM}})$, in the general case, we have

$$ $p_{t}({\bm{x}}_{t})=[1-e^{-\bar{\sigma}(t)}]^{d_{1}}[e^{-\bar{\sigma}(t)}]^{d_{2}}p_{0}({\bm{x}}_{t}^{\textrm{UM}}),$ $$

which demonstrates that the likelihood of the noisy data ${\bm{x}}_{t}$ at time $t$ equals the likelihood of the unmasked part ${\bm{x}}_{t}^{\textrm{UM}}$ at time 0 multiplied by an analytic time-dependent term.
∎

See [1](#Thmtheorem1)

###### Proof.

According to [Proposition˜1](#Thmproposition1), if $x_{t}^{i}=[\textbf{M}]$ and $\hat{x}_{t}^{i}\neq[\textbf{M}]$, $\hat{{\bm{x}}}_{t}^{\textrm{UM}}=({\bm{x}}_{t}^{\textrm{UM}},\hat{x}_{t}^{i})$,

$$ $\displaystyle\frac{p_{t}(\hat{{\bm{x}}}_{t})}{p_{t}({\bm{x}}_{t})}=$ $\displaystyle\frac{[1-e^{-\bar{\sigma}(t)}]^{d_{1}-1}[e^{-\bar{\sigma}(t)}]^{d_{2}+1}p_{0}(\hat{{\bm{x}}}_{t}^{\textrm{UM}})}{[1-e^{-\bar{\sigma}(t)}]^{d_{1}}[e^{-\bar{\sigma}(t)}]^{d_{2}}p_{0}({\bm{x}}_{t}^{\textrm{UM}})}$ $\displaystyle=$ $\displaystyle\frac{[1-e^{-\bar{\sigma}(t)}]^{d_{1}-1}[e^{-\bar{\sigma}(t)}]^{d_{2}+1}p_{0}({\bm{x}}_{t}^{\textrm{UM}},\hat{x}_{t}^{i})}{[1-e^{-\bar{\sigma}(t)}]^{d_{1}}[e^{-\bar{\sigma}(t)}]^{d_{2}}p_{0}({\bm{x}}_{t}^{\textrm{UM}})}$ $\displaystyle=$ $\displaystyle\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}p_{0}(\hat{x}_{t}^{i}|{\bm{x}}_{t}^{\textrm{UM}}).$ $$

∎

###### Lemma 1 .

###### Proof.

###### Proposition 1 .

###### Proof.

###### Proof.

## Appendix C Proof of Theorem 2

See [2](#Thmtheorem2)

Here, the infinity final total noise level guarantees that all tokens will be finally masked with probability one ($1-e^{-\bar{\sigma}(T)}$). Below we present the detailed proof in three steps.

### C.1 Equivalence between DSE loss and t-DCE loss

For a given noisy input ${\bm{x}}_{t}$, as established in [Section˜3.1](#S3.SS1), ${\hat{{\bm{x}}}_{t}}$ is valid only when it contains exactly one more unmasked token than ${\bm{x}}_{t}$.
In this case, the transition probability ${\bm{Q}}_{t}\left({\hat{{\bm{x}}}_{t}},{\bm{x}}_{t}\right)$ equals $\sigma(t)$.
Replace ${\bm{s}}_{\theta}({\bm{x}}_{t})$ with $\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}{\bm{c}}_{\theta}({\bm{x}}_{t})$, we can express the DSE loss in the multi-dimensional case as follows:

$$ $\displaystyle\mathcal{L}_{\textrm{DSE}}^{T}({\bm{x}}_{0})=\int_{0}^{T}\mathbb{E}_{{\bm{x}}_{t}\sim p_{t|0}({\bm{x}}_{t}|{\bm{x}}_{0})}\left[\sum_{x_{t}^{i}=[\textbf{M}],j\neq[\textbf{M}]}\sigma(t)\left(\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}{\bm{c}}_{\theta}({\bm{x}}_{t})[i,j]\right.\right.$ $\displaystyle\left.\left.-\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}\mathbb{I}(x_{0}^{i}=j)\log\left(\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}{\bm{c}}_{\theta}({\bm{x}}_{t})[i,j]\right)+K\left(\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}I(x_{0}^{i}=j)\right)\right)\right]dt$ $\displaystyle=\int_{0}^{T}\mathbb{E}_{{\bm{x}}_{t}\sim p_{t|0}({\bm{x}}_{t}|{\bm{x}}_{0})}\left[\sum_{x_{t}^{i}=[\textbf{M}],j\neq[\textbf{M}]}\sigma(t)\left(\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}{\bm{c}}_{\theta}({\bm{x}}_{t})[i,j]\right)\right]dt$ $\displaystyle+\int_{0}^{T}\mathbb{E}_{{\bm{x}}_{t}\sim p_{t|0}({\bm{x}}_{t}|{\bm{x}}_{0})}\left[\sum_{x_{t}^{i}=[\textbf{M}],j\neq[\textbf{M}]}-\frac{\sigma(t)e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}\mathbb{I}(x_{0}^{i}=j)\log\left(\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}{\bm{c}}_{\theta}({\bm{x}}_{t})[i,j]\right)\right]dt$ $\displaystyle+\int_{0}^{T}\mathbb{E}_{{\bm{x}}_{t}\sim p_{t|0}({\bm{x}}_{t}|{\bm{x}}_{0})}\left[\sum_{x_{t}^{i}=[\textbf{M}],j\neq[\textbf{M}]}\sigma(t)K\left(\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}I(x_{0}^{i}=j)\right)\right]dt.$ $$

We analyze each term in the above equation separately. The first term simplifies due to the property $\sum_{j\neq[\textbf{M}]}{\bm{c}}_{\theta}({\bm{x}}_{t})[i,j]=1$:

$$ $\int_{0}^{T}\mathbb{E}_{{\bm{x}}_{t}\sim p_{t|0}({\bm{x}}_{t}|{\bm{x}}_{0})}\left[\sum_{x_{t}^{i}=[\textbf{M}]}\sigma(t)\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}\right]dt.$ (C.1) $$

The third term can be simplified by substituting $K(a)=a\log a-a$ and using $0\log 0=0$:

$$ $\int_{0}^{T}\mathbb{E}_{{\bm{x}}_{t}\sim p_{t|0}({\bm{x}}_{t}|{\bm{x}}_{0})}\left[\sum_{x_{t}^{i}=[\textbf{M}]}\sigma(t)\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}\left(\log\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}-1\right)\right]dt.$ (C.2) $$

Combining the first and third terms:

$$ $\displaystyle\int_{0}^{T}\mathbb{E}_{{\bm{x}}_{t}\sim p_{t|0}({\bm{x}}_{t}|{\bm{x}}_{0})}\left[\sum_{x_{t}^{i}=[\textbf{M}]}\sigma(t)\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}\left(\log\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}\right)\right]dt$ (C.3) $\displaystyle=$ $\displaystyle\int_{0}^{T}d(1-e^{-\bar{\sigma}(t)})\sigma(t)\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}\left(\log\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}\right)dt$ (C.4) $\displaystyle=$ $\displaystyle d\int_{0}^{T}\sigma(t)e^{-\bar{\sigma}(t)}\log\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}dt.$ (C.5) $$

Introducing a new variable $\lambda(t)=1-e^{-\bar{\sigma}(t)}$, which represents the probability of a token being masked from 0 to $t$ in the forward process. As $\bar{\sigma}(t)=\int_{0}^{t}\sigma(\tau)d\tau$ and $\bar{\sigma}(T)=\infty$, we have $\lambda(0)=0$, $\lambda(T)=1$ and $d\lambda=\sigma(t)e^{-\bar{\sigma}(t)}dt$. Obviously, $\lambda(t)$ is invertible, which allows us to perform a change of variables from $t$ to $\lambda$ and simplifies [Eq.˜C.5](#A3.E5) to

$$ $\displaystyle d\int_{0}^{1}\log\frac{1-\lambda}{\lambda}d\lambda=-d\left(\lambda\log\lambda+(1-\lambda)\log(1-\lambda)\right)|_{0}^{1}=0.$ (C.6) $$

Here we used

$$ $\lim_{\lambda\to 0}\lambda\log\lambda=\lim_{\lambda\to 1}(1-\lambda)\log(1-\lambda)=0.$ $$

Thus, the DSE loss reduces to the second term, which we define as the $t$-denoising cross-entropy loss ($t\textrm{-DCE}$):

$$ $\displaystyle\mathcal{L}_{t\textrm{-DCE}}^{T}({\bm{x}}_{0})$ $\displaystyle=\int_{0}^{T}\mathbb{E}_{{\bm{x}}_{t}\sim p_{t|0}({\bm{x}}_{t}|{\bm{x}}_{0})}\left[\sum_{\begin{subarray}{c}x_{t}^{i}=[\textbf{M}]\\ j\neq[\textbf{M}]\end{subarray}}-\frac{\sigma(t)e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}\mathbb{I}(x_{0}^{i}=j)\log\left(\frac{e^{-\bar{\sigma}(t)}{\bm{c}}_{\theta}({\bm{x}}_{t})[i,j]}{1-e^{-\bar{\sigma}(t)}}\right)\right]dt$ $\displaystyle=\int_{0}^{T}\mathbb{E}_{{\bm{x}}_{t}\sim p_{t|0}({\bm{x}}_{t}|{\bm{x}}_{0})}\left[\sum_{x_{t}^{i}=[\textbf{M}]}-\frac{\sigma(t)e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}\log\left(\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}{\bm{c}}_{\theta}({\bm{x}}_{t})[i,x_{0}^{i}]\right)\right]dt$ $\displaystyle=\int_{0}^{T}\mathbb{E}_{{\bm{x}}_{t}\sim p_{t|0}({\bm{x}}_{t}|{\bm{x}}_{0})}\left[\sum_{x_{t}^{i}=[\textbf{M}]}-\frac{\sigma(t)e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}\log\left(\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}q_{\theta}(x_{0}^{i}|{\bm{x}}_{t}^{\textrm{UM}})\right)\right]dt.$ $$

### C.2 Equivalence between t-DCE loss and lambda-DCE loss

Starting from the $t\textrm{-DCE}$ loss in [Eq.˜3.5](#S3.E5), we can perform a change of variable from $t$ to $\lambda(t)=1-e^{-\bar{\sigma}(t)}$, as demonstrated in [Section˜C.1](#A3.SS1). This allows us to rewrite the $t\textrm{-DCE}$ loss integral in terms of $\lambda$:

$$ $\displaystyle\int_{0}^{1}\frac{1}{\lambda}\mathbb{E}_{{\bm{x}}_{\lambda}\sim p_{\lambda}({\bm{x}}_{\lambda}|{\bm{x}}_{0})}\left[\sum_{x_{\lambda}^{i}=[\textbf{M}]}-\log\left(\frac{1-\lambda}{\lambda}q_{\theta}(x_{0}^{i}|{\bm{x}}_{\lambda}^{\textrm{UM}})\right)\right]d\lambda$ $\displaystyle=$ $\displaystyle\int_{0}^{1}\frac{1}{\lambda}\mathbb{E}_{{\bm{x}}_{\lambda}\sim p_{\lambda}({\bm{x}}_{\lambda}|{\bm{x}}_{0})}\left[\sum_{x_{\lambda}^{i}=[\textbf{M}]}-\log q_{\theta}(x_{0}^{i}|{\bm{x}}_{\lambda}^{\textrm{UM}})\right]d\lambda$ $\displaystyle-$ $\displaystyle\int_{0}^{1}\frac{1}{\lambda}\mathbb{E}_{{\bm{x}}_{\lambda}\sim p_{\lambda}({\bm{x}}_{\lambda}|{\bm{x}}_{0})}\left[\sum_{x_{\lambda}^{i}=[\textbf{M}]}\log\frac{1-\lambda}{\lambda}\right]d\lambda.$ $$

Given the independence of the forward process and [Lemma˜1](#Thmlemma1), the original probability $p_{t|0}({\bm{x}}_{t}|{\bm{x}}_{0})$ can be factorized as $\prod_{i=1}^{d}p_{t|0}^{i}(x_{t}^{i}|x_{0}^{i})$, where

$$ $p_{t|0}^{i}(x_{t}^{i}|x_{0}^{i})=\begin{cases}1-e^{-\bar{\sigma}(t)},&x_{t}^{i}=[\textbf{M}],\\ e^{-\bar{\sigma}(t)},&x_{t}^{i}=x_{0}^{i},\\ 0,&\text{else}.\end{cases}$ (C.7) $$

Therefore, the induced probability $p_{\lambda}({\bm{x}}_{\lambda}|{\bm{x}}_{0})=\prod_{i=1}^{d}p_{\lambda}^{i}(x_{\lambda}^{i}|x_{0}^{i})$
where

$$ $p_{\lambda}^{i}(x_{\lambda}^{i}|x_{0}^{i})=\begin{cases}\lambda,&x_{\lambda}^{i}=[\textbf{M}],\\ 1-\lambda,&x_{\lambda}^{i}=x_{0}^{i},\\ 0,&\text{else}.\end{cases}$ (C.8) $$

Next, consider the second term. Similar to [Eq.˜C.3](#A3.E3), we can prove that it equals zero:

$$ $\int_{0}^{1}\frac{1}{\lambda}\mathbb{E}_{{\bm{x}}_{\lambda}\sim p_{\lambda}({\bm{x}}_{\lambda}|{\bm{x}}_{0})}\left[\sum_{x_{\lambda}^{i}=[\textbf{M}]}\log(\frac{1-\lambda}{\lambda})\right]=0.$ (C.9) $$

Therefore, $t\textrm{-DCE}$ loss is equivalent to the first term, defined as $\lambda$-denoising cross-entropy ($\lambda\textrm{-DCE}$):

$$ $\mathcal{L}_{\lambda\textrm{-DCE}}^{T}({\bm{x}}_{0})=\int_{0}^{1}\frac{1}{\lambda}\mathbb{E}_{{\bm{x}}_{\lambda}\sim p_{\lambda}({\bm{x}}_{\lambda}|{\bm{x}}_{0})}\left[\sum_{x_{\lambda}^{i}=[\textbf{M}]}-\log q_{\theta}(x_{0}^{i}|{\bm{x}}_{\lambda}^{\textrm{UM}})\right]d\lambda.$ (C.10) $$

### C.3 Equivalence between lambda-DCE loss and any-order autoregressive loss

Based on $\lambda\textrm{-DCE}$ loss in [Eq.˜3.6](#S3.E6), we first define the sample space and analytically express the expectation term.

Given ${\bm{x}}_{0}$, we define the sample space of ${\bm{x}}_{\lambda}$ as $\tilde{\mathcal{X}}({\bm{x}}_{0}):=\{x_{0}^{1},[\textbf{M}]\}\times\cdots\{x_{0}^{d},[\textbf{M}]\}$ and $\tilde{\mathcal{X}}_{k}({\bm{x}}_{0}):=\{\tilde{{\bm{x}}}:\tilde{{\bm{x}}}\in\tilde{\mathcal{X}}({\bm{x}}_{0})\land\tilde{{\bm{x}}}\ \text{has exact k dimensions with values }[\textbf{M}]\}$ . It follows that $|\tilde{\mathcal{X}}({\bm{x}}_{0})|=2^{d}$ and $|\tilde{\mathcal{X}}_{k}({\bm{x}}_{0})|=\binom{d}{k}$. Therefore, the sample space $\tilde{\mathcal{X}}({\bm{x}}_{0})$ can be decoupled by the number of masked tokens $k$ in $\tilde{{\bm{x}}}$:

$$ $\displaystyle\int_{0}^{1}\frac{1}{\lambda}\mathbb{E}_{{\bm{x}}_{\lambda}\sim p_{\lambda}({\bm{x}}_{\lambda}|{\bm{x}}_{0})}\left[\sum_{\tilde{x}^{i}=[\textbf{M}]}-\log q_{\theta}(x_{0}^{i}|{\bm{x}}_{\lambda}^{\textrm{UM}})\right]d\lambda$ (C.11) $\displaystyle=$ $\displaystyle\int_{0}^{1}\frac{1}{\lambda}\sum_{\tilde{x}\in\tilde{\mathcal{X}}({\bm{x}}_{0})}p_{\lambda}(\tilde{{\bm{x}}}|{\bm{x}}_{0})\left[\sum_{\tilde{x}^{i}=[\textbf{M}]}-\log q_{\theta}(x_{0}^{i}|\tilde{{\bm{x}}}^{\textrm{UM}})\right]d\lambda$ (C.12) $\displaystyle=$ $\displaystyle\int_{0}^{1}\frac{1}{\lambda}\sum_{k=0}^{d}\sum_{\tilde{{\bm{x}}}\in\tilde{\mathcal{X}}_{k}({\bm{x}}_{0})}\lambda^{k}(1-\lambda)^{d-k}\left[\sum_{\tilde{x}^{i}=[\textbf{M}]}-\log q_{\theta}(x_{0}^{i}|\tilde{{\bm{x}}}^{\textrm{UM}})\right]d\lambda$ (C.13) $\displaystyle=$ $\displaystyle\int_{0}^{1}\frac{1}{\lambda}\sum_{k=1}^{d}\sum_{\tilde{{\bm{x}}}\in\tilde{\mathcal{X}}_{k}({\bm{x}}_{0})}\lambda^{k}(1-\lambda)^{d-k}\left[\sum_{\tilde{x}^{i}=[\textbf{M}]}-\log q_{\theta}(x_{0}^{i}|\tilde{{\bm{x}}}^{\textrm{UM}})\right]d\lambda.$ (C.14) $$

The last equation holds because there are no masked tokens when $k=0$, and the inner sum is zero.

From [Eq.˜C.14](#A3.E14), by rearranging the order of summation and integration, we can analytically evaluate the integral $\int_{0}^{1}\lambda^{k-1}(1-\lambda)^{d-k}d\lambda$ using the Beta function, which eliminates $\lambda$:

$$ $\displaystyle\lx@cref{creftype~refnum}{eq:lambda_k_mix}=$ $\displaystyle\sum_{k=1}^{d}\int_{0}^{1}\lambda^{k-1}(1-\lambda)^{d-k}d\lambda\sum_{\tilde{{\bm{x}}}\in\tilde{\mathcal{X}}_{k}({\bm{x}}_{0})}\left[\sum_{\tilde{x}^{i}=[\textbf{M}]}-\log q_{\theta}(x_{0}^{i}|\tilde{{\bm{x}}}^{\textrm{UM}})\right]$ (C.15) $\displaystyle=$ $\displaystyle\sum_{k=1}^{d}\frac{(k-1)!(d-k)!}{d!}\sum_{\tilde{{\bm{x}}}\in\tilde{\mathcal{X}}_{k}({\bm{x}}_{0})}\left[\sum_{\tilde{x}^{i}=[\textbf{M}]}-\log q_{\theta}(x_{0}^{i}|\tilde{{\bm{x}}}^{\textrm{UM}})\right]$ (C.16) $\displaystyle=$ $\displaystyle\sum_{k=1}^{d}\frac{1}{kC_{d}^{k}}\sum_{\tilde{{\bm{x}}}\in\tilde{\mathcal{X}}_{k}({\bm{x}}_{0})}\left[\sum_{\tilde{x}^{i}=[\textbf{M}]}-\log q_{\theta}(x_{0}^{i}|\tilde{{\bm{x}}}^{\textrm{UM}})\right].$ (C.17) $$

[Eq.˜C.17](#A3.E17) can be reformulated in terms of an expectation over a uniform distribution $U(\tilde{\mathcal{X}}_{k}({\bm{x}}_{0}))$ as follows:

$$ $\displaystyle\sum_{k=1}^{d}\frac{1}{k}\mathbb{E}_{\tilde{{\bm{x}}}\sim U(\tilde{\mathcal{X}}_{k}({\bm{x}}_{0}))}\left[\sum_{\tilde{x}^{i}=[\textbf{M}]}\left(-\log(q_{\theta}(x_{0}^{i}|\tilde{{\bm{x}}}^{\textrm{UM}}))\right)\right].$ (C.18) $$

Let $\pi$ be one permutation of the integers $1,\cdots,d$, and $U_{\pi}$ represent the uniform distribution of all orders.
We note that [Eq.˜C.18](#A3.E18) is equivalent to the following term from the perspective of any-order autoregressive model:

$$ $\displaystyle\sum_{k=1}^{d}\frac{1}{k}\mathbb{E}_{\pi\sim U_{\pi}}\sum_{r=d-k+1}^{d}-\log q_{\theta}(x_{0}^{\pi(r)}|x_{0}^{\pi(<d-k+1)};\pi).$ (C.19) $$

Here, $x_{0}^{\pi(<l)}$ denotes the sequence of the first $l-1$ elements in the permutation $\pi$. Given a fixed $k$, the term $x_{0}^{\pi(<d-k+1)}$ can be interpreted as the unmasked part of the noisy data $\tilde{{\bm{x}}}^{\textrm{UM}}$. For $r=d-k+1,\cdots,d$, $x_{0}^{\pi(r)}$ corresponds to the $k$ items of the masked part. Since both $\pi$ and $\tilde{{\bm{x}}}$ are both uniformly sampled, [Eq.˜C.18](#A3.E18) and [Eq.˜C.19](#A3.E19) are equivalent.

Further, we can make a simple subscription transformation by letting $l=d-k+1$ and change the summation to Monte Carlo estimation on [Eq.˜C.19](#A3.E19) :

$$ $\displaystyle d\cdot\mathbb{E}_{l\sim U(1,\cdots,d)}\frac{1}{d-l+1}\mathbb{E}_{\pi\sim U_{\pi}}\sum_{r=l}^{d}-\log q_{\theta}(x_{0}^{\pi(r)}|x_{0}^{\pi(<l)};\pi).$ (C.20) $$

In , it was proved that [Eq.˜C.20](#A3.E20) is mathematically equivalent to [Eq.˜2.16](#S2.E16). Actually, [Eq.˜C.20](#A3.E20) is widely used as a training objective for any-order autoregressive models for efficient parallel optimization.

This concludes our proof of [Theorem˜2](#Thmtheorem2).

## Appendix D Sampling methods

In this section, we first derived the exact reverse distribution for absorbing discrete diffusion in [Section˜D.1](#A4.SS1). This derivation led to simplified forms of the Tweedie $\tau$-leaping and Euler methods, detailed in [Section˜D.2](#A4.SS2) and [Section˜D.3](#A4.SS3), respectively. In [Section˜D.4](#A4.SS4), we proved the equivalence of these two methods under a log-linear noise schedule. Finally, in [Section˜D.5](#A4.SS5), we discussed the expected number of function evaluations (E-NFEs) for these methods.

### D.1 Exact reverse distribution in absorbing discrete diffusion

###### Lemma 2 .

(Analytic reverse distribution in absorbing diffusion)
Suppose $\{X_{t}\}$ is a continuous time Markov chain with transition rate matrix ${\bm{Q}}_{t}=\sigma(t){\bm{Q}}^{\text{absorb}}$. For ${\bm{x}}_{t}=x_{t}^{1}\cdots x_{t}^{d}$ with $d_{1}$ masked tokens and $d_{2}=d-d_{1}$ unmasked tokens, and ${\bm{x}}_{s}=x_{s}^{1}\cdots x_{s}^{d}$ with $d_{1}-\Delta d$ masked tokens and $d_{2}+\Delta d$ unmasked tokens, the reverse distribution is given by:

$$ $p_{s|t}({\bm{x}}_{s}|{\bm{x}}_{t})=\begin{cases}\left[\frac{e^{-\bar{\sigma}(s)}-e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(s)}}\right]^{\Delta d}\left[\frac{1-e^{-\bar{\sigma}(s)}}{1-e^{-\bar{\sigma}(t)}}\right]^{d_{1}}\frac{p_{0}({\bm{x}}_{s}^{\textrm{UM}})}{p_{0}({\bm{x}}_{t}^{\textrm{UM}})},&{\bm{x}}_{t}\subseteq_{\text{UM}}{\bm{x}}_{s},\\ 0,&{\bm{x}}_{t}\not\subseteq_{\text{UM}}{\bm{x}}_{s},\end{cases}$ (D.1) $$

where ${\bm{x}}_{t}\subseteq_{\text{UM}}{\bm{x}}_{s}$ denotes $\forall i:$ ${\bm{x}}_{t}^{i}\neq[\textbf{M}]$, we have ${\bm{x}}_{t}^{i}={\bm{x}}_{s}^{i}$.

###### Proof.

Using Bayes’ theorem, $p_{s|t}({\bm{x}}_{s}|{\bm{x}}_{t})=p_{t|s}({\bm{x}}_{t}|{\bm{x}}_{s})\frac{p_{s}({\bm{x}}_{s})}{p_{t}({\bm{x}}_{t})}$.

From [Proposition˜1](#Thmproposition1):

$$ $\displaystyle p_{t}({\bm{x}}_{t})$ $\displaystyle=[1-e^{-\bar{\sigma}(t)}]^{d_{1}}[e^{-\bar{\sigma}(t)}]^{d_{2}}p_{0}({\bm{x}}_{t}^{\textrm{UM}}),$ (D.2) $\displaystyle p_{s}({\bm{x}}_{s})$ $\displaystyle=[1-e^{-\bar{\sigma}(s)}]^{d_{1}-\Delta d}[e^{-\bar{\sigma}(s)}]^{d_{2}+\Delta d}p_{0}({\bm{x}}_{s}^{\textrm{UM}}).$ (D.3) $$

Utilizing [Eq.˜B.12](#A2.E12), we get

$$ $p_{t|s}({\bm{x}}_{t}|{\bm{x}}_{s})=\prod_{i=1}^{d}p_{t|s}^{i}(x_{t}^{i}|x_{s}^{i})=\begin{cases}\left[e^{-\left(\bar{\sigma}(t)-\bar{\sigma}(s)\right)}\right]^{d_{2}}\left[1-e^{-\left(\bar{\sigma}(t)-\bar{\sigma}(s)\right)}\right]^{\Delta d},&{\bm{x}}_{t}\subseteq_{\text{UM}}{\bm{x}}_{s},\\ 0,&{\bm{x}}_{t}\not\subseteq_{\text{UM}}{\bm{x}}_{s}.\end{cases}$ (D.4) $$

Simplifying these equations, we can express $p_{s|t}({\bm{x}}_{s}|{\bm{x}}_{t})$ as

$$ $p_{s|t}({\bm{x}}_{s}|{\bm{x}}_{t})=\begin{cases}\left[\frac{e^{-\bar{\sigma}(s)}-e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(s)}}\right]^{\Delta d}\left[\frac{1-e^{-\bar{\sigma}(s)}}{1-e^{-\bar{\sigma}(t)}}\right]^{d_{1}}\frac{p_{0}({\bm{x}}_{s}^{\textrm{UM}})}{p_{0}({\bm{x}}_{t}^{\textrm{UM}})},&{\bm{x}}_{t}\subseteq_{\text{UM}}{\bm{x}}_{s},\\ 0,&{\bm{x}}_{t}\not\subseteq_{\text{UM}}{\bm{x}}_{s}.\end{cases}$ (D.5) $$

∎

It should be noted that when ${\bm{x}}_{t}\subseteq_{\text{UM}}{\bm{x}}_{s}$, the ratio $\frac{p_{0}({\bm{x}}_{s}^{\textrm{UM}})}{p_{0}({\bm{x}}_{t}^{\textrm{UM}})}$ can be reformulated as a $d_{1}$-dimensional conditional distribution $p_{0}({\bm{x}}_{s}^{\textrm{UM}}|{\bm{x}}_{t}^{\textrm{UM}})$ with $N^{d_{1}}$ states. This is not accessible using our one-dimensional conditional distribution $p_{0}(\hat{x}_{t}^{i}|{\bm{x}}_{t}^{\textrm{UM}})$ in [Theorem˜1](#Thmtheorem1) if $d_{1}>1$. Therefore, for efficiency, existing samplers assume that each dimension is independent within a small interval and update each dimension in parallel .

###### Lemma 2 .

###### Proof.

### D.2 Tweedie τ \tau -leaping method and its simplified form in RADD

Given the vector ${\bm{x}}_{t}$, if we sample each $x_{s}^{i}$ independently, the factorization of marginal distribution $p_{s|t}^{\text{tweedie}}$ results in the minimum KL divergence with true reverse $p_{s|t}({\bm{x}}_{s}|{\bm{x}}_{t})$ (proof in  , Appendix A). This assumption formally defines $p_{s|t}^{\text{tweedie}}$ as follows:

$$ $p_{s|t}^{\text{tweedie}}({\bm{x}}_{s}|{\bm{x}}_{t})=\prod_{i=1}^{d}p_{s|t}^{\text{tweedie},i}(x_{s}^{i}|{\bm{x}}_{t})=\prod_{i=1}^{d}p_{s|t}^{i}(x_{s}^{i}|{\bm{x}}_{t}).$ (D.6) $$

To sample from $p_{s|t}^{\text{tweedie}}$, we need to derive the analytic form of $p_{s|t}^{i}(x_{s}^{i}|{\bm{x}}_{t})$. Without loss of generality, let’s assume that the preceding $d_{1}$ terms of ${\bm{x}}_{t}$ are all $[\textbf{M}]$, and the remaining $d_{2}$ terms are unmasked tokens.

For illustration, we can take $i=1$ as an example. Let $\tilde{\mathcal{X}}_{k}$ denote the sample space of length $d_{1}-1$ sequence where each sequence has exact $k$ masked tokens, with $|\tilde{\mathcal{X}}_{k}|=C_{d_{1}-1}^{k}N^{d_{1}-1-k}$. When $x_{s}^{1}\neq[\textbf{M}]$, According to [Lemma˜2](#Thmlemma2):

$$ $\displaystyle p_{s|t}^{1}(x_{s}^{1}|{\bm{x}}_{t})$ $\displaystyle=\sum_{{\bm{x}}_{s}^{2:d}}p_{s|t}({\bm{x}}_{s}|{\bm{x}}_{t})$ $\displaystyle=\sum_{k=0}^{d_{1}-1}\sum_{{\bm{x}}_{s}^{2:d}\in\tilde{\mathcal{X}}_{k}}\left[\frac{e^{-\bar{\sigma}(s)}-e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(s)}}\right]^{k+1}\left[\frac{1-e^{-\bar{\sigma}(s)}}{1-e^{-\bar{\sigma}(t)}}\right]^{d_{1}}\frac{p_{0}(x_{s}^{1},{\bm{x}}_{s}^{2:d,\textrm{UM}},{\bm{x}}_{t}^{d_{1}+1:d})}{p_{0}({\bm{x}}_{t}^{d_{1}+1:d})}$ $\displaystyle=\sum_{k=0}^{d_{1}-1}C_{d_{1}-1}^{k}\left[\frac{e^{-\bar{\sigma}(s)}-e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(s)}}\right]^{k+1}\left[\frac{1-e^{-\bar{\sigma}(s)}}{1-e^{-\bar{\sigma}(t)}}\right]^{d_{1}}\frac{p_{0}(x_{s}^{1},{\bm{x}}_{t}^{d_{1}+1:d})}{p_{0}({\bm{x}}_{t}^{d_{1}+1:d})}$ $\displaystyle=\frac{e^{-\bar{\sigma}(s)}-e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(s)}}\left[1+\frac{e^{-\bar{\sigma}(s)}-e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(s)}}\right]^{d_{1}-1}\left[\frac{1-e^{-\bar{\sigma}(s)}}{1-e^{-\bar{\sigma}(t)}}\right]^{d_{1}}\frac{p_{0}(x_{s}^{1},{\bm{x}}_{t}^{d_{1}+1:d})}{p_{0}({\bm{x}}_{t}^{d_{1}+1:d})}$ $\displaystyle=\frac{e^{-\bar{\sigma}(s)}-e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}\frac{p_{0}(x_{s}^{1},{\bm{x}}_{t}^{d_{1}+1:d})}{p_{0}({\bm{x}}_{t}^{d_{1}+1:d})}$ $\displaystyle=\frac{e^{-\bar{\sigma}(s)}-e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}{p_{0}(x_{s}^{1}|{\bm{x}}_{t}^{d_{1}+1:d})}.$ $$

Here, we used the binomial expansion identity:

$$ $\sum_{k=0}^{d_{1}-1}C_{d_{1}-1}^{k}\left[\frac{e^{-\bar{\sigma}(s)}-e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(s)}}\right]^{k}=\left[1+\frac{e^{-\bar{\sigma}(s)}-e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(s)}}\right]^{d_{1}-1}.$ $$

Similarly, for $x_{s}^{1}=[\textbf{M}]$:

$$ $p_{s|t}^{1}([\textbf{M}]|{\bm{x}}_{t})=\frac{1-e^{-\bar{\sigma}(s)}}{1-e^{-\bar{\sigma}(t)}}.$ (D.7) $$

In general, we have

$$ $p_{s|t}^{i}(x_{s}^{i}|{\bm{x}}_{t})=\begin{cases}\frac{e^{-\bar{\sigma}(s)}-e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}{p_{0}(x_{s}^{i}|{\bm{x}}_{t}^{\textrm{UM}})},&x_{s}^{i}\neq[\textbf{M}],x_{t}^{i}=[\textbf{M}],\\ \frac{1-e^{-\bar{\sigma}(s)}}{1-e^{-\bar{\sigma}(t)}},&x_{s}^{i}=[\textbf{M}],x_{t}^{i}=[\textbf{M}],\\ \delta_{x_{s}^{i}x_{t}^{i}},&x_{t}^{i}\neq[\textbf{M}].\end{cases}$ (D.8) $$

With trained ${\bm{c}}_{\theta}$, we can use ${\bm{c}}_{\theta}({\bm{x}}_{t})[i,x_{s}^{i}]$ to approximate the true conditional distribution $p_{0}(x_{s}^{i}|{\bm{x}}_{t}^{\textrm{UM}})$ and sample by [Eq.˜D.8](#A4.E8).

### D.3 Euler method and its simplified form in RADD

According to theory of CTMC , given a particular one-dimensional input $x_{t}$, the transition probabilities to $x_{s}$ can be approximately calculated using [Eq.˜2.1](#S2.E1) and [Eq.˜2.12](#S2.E12) as follows:

$$ $\displaystyle p_{s|t}(x_{s}|x_{t})$ $\displaystyle=\delta_{x_{t}x_{s}}+\tilde{{\bm{Q}}}_{t}(x_{t},x_{s})(t-s)+o(t-s),$ (D.9) $\displaystyle\approx\delta_{x_{t}x_{s}}+\tilde{{\bm{Q}}}_{t}(x_{t},x_{s})(t-s),$ (D.10) $$

where

$$ $\displaystyle\tilde{{\bm{Q}}}_{t}(x_{t},x_{s})$ $\displaystyle=\begin{cases}{\bm{Q}}_{t}(x_{s},x_{t})\frac{p_{t}(x_{s})}{p_{t}(x_{t})},&x_{t}\neq x_{s},\\ -\sum_{k\neq x_{t}}\tilde{{\bm{Q}}}_{t}(x_{t},k),&x_{t}=x_{s}.\end{cases}$ (D.11) $$

Therefore, we can define the Euler approximation of the transition probability :

$$ $p_{s|t}^{\text{euler}}(x_{s}|x_{t})=\delta_{x_{t}x_{s}}+\tilde{{\bm{Q}}}_{t}(x_{t},x_{s})(t-s)$ (D.12) $$

For multi-dimensional case, we factorize $p_{s|t}^{\text{euler}}({\bm{x}}_{s}|{\bm{x}}_{t})$
as $\prod_{i=1}^{d}p_{s|t}^{\text{euler},i}(x_{s}^{i}|{\bm{x}}_{t})$, where $p_{s|t}^{\text{euler},i}(x_{s}^{i}|{\bm{x}}_{t})$ is based on [Eq.˜D.12](#A4.E12) which use ${\bm{x}}_{t}$ to replace $x_{t}$ and $x_{t}^{1}\cdots x_{s}^{i}\cdots x_{t}^{d}$ to replace $x_{s}$.

In the case of absorbing diffusion, similar to Tweedie-$\tau$ leaping method in [Section˜D.2](#A4.SS2), we can use [Theorem˜1](#Thmtheorem1) and [Eq.˜2.9](#S2.E9) to simplify [Eq.˜D.12](#A4.E12), which results in

$$ $p_{s|t}^{\text{euler},i}(x_{s}^{i}|{\bm{x}}_{t})=\begin{cases}\sigma(t)\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}(t-s)p_{0}(x_{s}^{i}|{\bm{x}}_{t}^{\textrm{UM}}),&\text{if }x_{s}^{i}\neq[\textbf{M}],x_{t}^{i}=[\textbf{M}]\\ 1-\sigma(t)\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}(t-s),&\text{if }x_{s}^{i}=[\textbf{M}],x_{t}^{i}=[\textbf{M}]\\ \delta_{x_{s}^{i}x_{t}^{i}},&x_{t}^{i}\neq[\textbf{M}].\\ \end{cases}$ (D.13) $$

In practice, we also use $c_{\theta}({\bm{x}}_{t})[i,x_{s}^{i}]$ to approximate the true conditional distribution $p_{0}(x_{s}^{i}|{\bm{x}}_{t}^{\textrm{UM}})$ when sampling from [Eq.˜D.13](#A4.E13).

### D.4 Equivalence of Tweedie τ \tau -leaping and Euler method under log-linear noise schedule

By comparing [Eq.˜D.8](#A4.E8) and [Eq.˜D.13](#A4.E13), we observe that both the Tweedie $\tau$-leaping and Euler methods can be interpreted similarly:

- •
If $x_{t}^{i}$ is an unmasked token, keep it unchanged, i.e., $x_{s}^{i}=x_{t}^{i}$.
- •
If $x_{t}^{i}$ is a masked token, first determine whether it will be unmasked with a probability $\psi(t,s)$. If it is to be unmasked, then sample $x_{s}^{i}$ from $p_{0}(x_{s}^{i}|{\bm{x}}_{t}^{\textrm{UM}})$.

The only difference lies in the analytic form of $\psi(t,s)$. For the two methods, according to [Eqs.˜D.8](#A4.E8) and [D.13](#A4.E13), their corresponding $\psi(t,s)$ are given as follows:

$$ $\displaystyle\psi^{\text{tweedie}}(t,s)=\frac{e^{-\bar{\sigma}(s)}-e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}},$ (D.14) $\displaystyle\psi^{\text{euler}}(t,s)=\sigma(t)\frac{e^{-\bar{\sigma}(t)}}{1-e^{-\bar{\sigma}(t)}}(t-s).$ (D.15) $$

In general cases, these two expressions are not equivalent. However, if we choose a log-linear noise schedule $\bar{\sigma}(t)=1-\log\left(1-(1-\epsilon)t\right)$, both [Eq.˜D.15](#A4.E15) and [Eq.˜D.14](#A4.E14) can be simplified to the same form $\psi(t,s)$ as follows:

$$ $\psi(t,s)=\frac{t-s}{t},$ (D.16) $$

which shows that these two sampling methods are equivalent under a log-linear noise schedule.

### D.5 Discuss on the expectation of NFEs

In this part, we show that given the noise schedule $\sigma(t)$ and a set of time steps $\{t_{0}=0,\cdots,t_{n}=T\}$, the NFEs can be treated as a random variable with a calculable expected value for both Euler method and Tweedie $\tau$-leaping method.

Let $l$ denote the length of the generated sequence and $N_{k}\in\{0,\cdots,l\}$ denote the number of dimensions that ${\bm{x}}$ changed in $[t_{k-1},t_{k})$. Without loss of generality, we first consider the unconditional generation case where $l=d$. The NFEs, E-NFEs, and $N_{k}$ can be expressed as

$$ $\text{NFEs}(n)=\sum_{k=1}^{n}\mathbb{I}(N_{k}\neq 0),$ (D.17) $$

$$ $\text{E-NFEs}(n)=\sum_{k=1}^{n}\mathbb{E}[\mathbb{I}(N_{k}\neq 0)]=\sum_{k=1}^{n}P(N_{k}\neq 0),$ (D.18) $$

$$ $N_{k}=\sum_{i=1}^{d}\mathbb{I}(x_{t_{k-1}}^{i}\neq[\textbf{M}],x_{t_{k}}^{i}=[\textbf{M}]).$ (D.19) $$

Furthermore, we note that the $d$ dimensions are independent. According to [Eqs.˜D.8](#A4.E8) and [D.13](#A4.E13), the probability $p_{s|t}^{\cdot,i}([\textbf{M}]|{\bm{x}}_{t})$ depends only on time and $x_{t}^{i}$ while independent of the other dimensions of ${\bm{x}}_{t}$. Thus, $p_{s|t}^{\cdot,i}([\textbf{M}]|{\bm{x}}_{t})=p_{s|t}^{\cdot,i}([\textbf{M}]|x_{t}^{i})$. Therefore, whether a token changes from masked to unmasked is independent across the $d$ dimensions(^3^33The independence applies to whether a token changes from masked to unmasked. However, the specific unmasked token a masked token changes to depends on other dimensions.):

$$ $\displaystyle p\left(\mathbb{I}(x_{s}^{1}=[\textbf{M}]),\cdots,\mathbb{I}(x_{s}^{d}=[\textbf{M}])|\mathbb{I}(x_{t}^{1}=[\textbf{M}]),\cdots,\mathbb{I}(x_{t}^{d}=[\textbf{M}])\right)$ (D.20) $\displaystyle=$ $\displaystyle\prod_{i=1}^{d}p\left(\mathbb{I}(x_{s}^{i}=[\textbf{M}])|\mathbb{I}(x_{t}^{i}=[\textbf{M}])\right).$ (D.21) $$

Since ${\bm{x}}_{T}$ consists entirely of masked tokens with probability one, each dimension of $\mathbb{I}(x_{t_{k-1}}^{i}\neq[\textbf{M}],x_{t_{k}}^{i}=[\textbf{M}])$ is independent. Consequently, $N_{k}$ follows a binomial distribution with parameters $d$ and $r_{k}$, denoted as $N_{k}\sim\text{Binomial}(d,r_{k})$,
where $r_{k}=p(x_{t_{k-1}}^{i}\neq[\textbf{M}],x_{t_{k}}^{i}=[\textbf{M}])$ represents the probability that $x^{i}$ changes within the interval $[t_{k-1},t_{k})$ in each dimension. Therefore, we can further simplify [Eq.˜D.18](#A4.E18):

$$ $\text{E-NFEs}(n)=\sum_{k=1}^{n}P(N_{k}\neq 0)=\sum_{k=1}^{n}(1-(1-r_{k})^{d}).$ (D.22) $$

By definition of $r_{k}$ and the property of absorbing diffusion:

$$ $\displaystyle r_{k}$ $\displaystyle=p(x_{t_{k-1}}^{i}\neq[\textbf{M}],x_{t_{k}}^{i}=[\textbf{M}])$ (D.23) $\displaystyle=p(x_{t_{k-1}}^{i}\neq[\textbf{M}]|x_{t_{k}}^{i}=[\textbf{M}])\prod_{j=k+1}^{n}p(x_{t_{j-1}}^{i}=[\textbf{M}]|x_{t_{j}}^{i}=[\textbf{M}])p(x_{t_{n}}^{i}=[\textbf{M}])$ (D.24) $\displaystyle=\left(1-p(x_{t_{k-1}}^{i}=[\textbf{M}]|x_{t_{k}}^{i}=[\textbf{M}])\right)\prod_{j=k+1}^{n}p(x_{t_{j-1}}^{i}=[\textbf{M}]|x_{t_{j}}^{i}=[\textbf{M}]).$ (D.25) $$

[Eq.˜D.25](#A4.E25) can be determined given the sampling method and noise schedule.

For Tweedie $\tau$ -leaping, based on [Eq.˜D.8](#A4.E8), we can derive that:

$$ $p(x_{t_{j-1}}^{i}=[\textbf{M}]|x_{t_{j}}^{i}=[\textbf{M}])=\frac{1-e^{-\bar{\sigma}(t_{j-1})}}{1-e^{-\bar{\sigma}(t_{j})}}.$ (D.26) $$

Therefore, we can express $r_{k}$ as

$$ $r_{k}=(\frac{e^{-\bar{\sigma}(t_{k-1})}-e^{-\bar{\sigma}(t_{k})}}{1-e^{-\bar{\sigma}(t_{k})}})\prod_{j=k+1}^{n}(1-\frac{1-e^{-\bar{\sigma}(t_{j-1})}}{1-e^{-\bar{\sigma}(t_{j})}})=\frac{e^{-\bar{\sigma}(t_{k-1})}-e^{-\bar{\sigma}(t_{k})}}{1-e^{-\bar{\sigma}(t_{n})}}.$ (D.27) $$

For the Euler method, based on [Eq.˜D.13](#A4.E13), we can derive that:

$$ $p(x_{t_{j-1}}^{i}=[\textbf{M}]|x_{t_{j}}^{i}=[\textbf{M}])=1-\sigma(t_{j})\frac{e^{-\bar{\sigma}(t_{j})}}{1-e^{-\bar{\sigma}(t_{j})}}(t_{j}-t_{j-1}).$ (D.28) $$

$$ $r_{k}=(\sigma(t_{k})\frac{e^{-\bar{\sigma}(t_{k})}}{1-e^{-\bar{\sigma}(t_{k})}}(t_{k}-t_{k-1}))\prod_{j=k+1}^{n}(1-\sigma(t_{j})\frac{e^{-\bar{\sigma}(t_{j})}}{1-e^{-\bar{\sigma}(t_{j})}}(t_{j}-t_{j-1})).$ (D.29) $$

For conditional generation cases where the generating sequence length $l$ is less than the dimension $d$, similar results hold. The only difference is that $N_{k}\sim\text{Binomial}(l,r_{k})$ and [Eq.˜D.22](#A4.E22) changes to

$$ $\text{E-NFEs}(n)=\sum_{k=1}^{n}(1-(1-r_{k})^{l}).$ (D.30) $$

Specifically, if we adopt a log-linear noise schedule and let $t_{k}=\frac{k}{n}$, according to [Section˜D.4](#A4.SS4), the Euler method and Tweedie $\tau$-leaping method are equivalent. In this case, [Eq.˜D.27](#A4.E27) simplifies to $\frac{1}{n}$. Substituting this result into [Eq.˜D.22](#A4.E22), we obtain

$$ $\text{E-NFEs}(n)=\sum_{k=1}^{n}(1-(1-\frac{1}{n})^{l})=n(1-(1-\frac{1}{n})^{l}).$ (D.31) $$

## Appendix E Discussion for mean parameterization and RADD

#### Equivalence of modeling

Analogous to the $x_{0}$ prediction in continuous state diffusion models, and used the mean parameterization ${\bm{\mu}}_{\theta}({\bm{x}}_{t},t)$ to learn the the reverse density $p_{0|t}^{i}(x_{0}^{i}|{\bm{x}}_{t})$, $i=1\cdots d$. According to the analytic form of reverse distribution in [Eq.˜D.8](#A4.E8), letting $s=0$, we have:

$$ $p_{0|t}^{i}(x_{0}^{i}|{\bm{x}}_{t})=\begin{cases}{p_{0}(x_{0}^{i}|{\bm{x}}_{t}^{\textrm{UM}})},&x_{0}^{i}\neq[\textbf{M}],x_{t}^{i}=[\textbf{M}],\\ 0,&x_{0}^{i}=[\textbf{M}],x_{t}^{i}=[\textbf{M}],\\ \delta_{x_{0}^{i}x_{t}^{i}},&x_{t}^{i}\neq[\textbf{M}].\end{cases}$ (E.1) $$

This shows that the mean prediction is equivalent to learning conditional distributions on clean data.
In conjunction with our discussion in [Section˜3.1](#S3.SS1), the mean parameterization should be time-independent, denoted as ${\bm{\mu}}_{\theta}({\bm{x}}_{t})$, and is equivalent to our reparameterized ${\bm{c}}_{\theta}({\bm{x}}_{t})$. Empirical results like and , which demonstrate that the time-independent model ${\bm{\mu}}_{\theta}({\bm{x}}_{t})$ performs well, also validate our theory.

#### Equivalence of training objectives

proved that the training loss for score parameterization (i.e., DSE loss) and mean parameterization (i.e., negative ELBO loss) are equivalent.

#### Equivalence of sampling methods

Comparing our [Section˜D.2](#A4.SS2) with , it is evident that the Tweedie $\tau$-leaping method for score parameterization is equivalent to the sampling method for mean prediction as follows:

$$ $q_{\theta}(x_{s}|x_{t})=p(x_{s}|x_{t},x_{0}=\mu(x_{t}))=\begin{cases}\textrm{Cat}(x_{s};x_{t}),&x_{t}\neq[\textbf{M}],\\ \textrm{Cat}(x_{s};\frac{1-\alpha_{s}}{1-\alpha_{t}}\mathbf{e}_{[\textbf{M}]}+\frac{\alpha_{s}-\alpha_{t}}{1-\alpha_{t}}\mu(x_{t})),&x_{t}=[\textbf{M}].\\ \end{cases}$ (E.2) $$

Here, $\alpha_{t}$ represents the probability of a token remaining unmasked at time $t$, which equals $e^{-\bar{\sigma}(t)}$ for score parameterization. Therefore, [Eq.˜E.2](#A5.E2) and [Eq.˜D.8](#A4.E8) is equivalent.

## Appendix F Comparison with Chen et al. ( 2024 )

proposes sampling methods for discrete-time and continuous-time diffusion models individually. Consider a sequence $\bm{x}$ of length $d$:

#### Discrete-time models

For models trained over discrete timesteps $\mathcal{T}_{\text{train}}=\{1,\cdots,T\}$, proposes to pre-sample the $d$ time points $\tau_{i}\in\mathcal{T}_{\text{train}}$, where each $\tau_{i}$ corresponds to a change timepoint for a specific dimension of $\bm{x}$. These timepoints form a time set $\mathcal{T}_{\text{change}}\subset\mathcal{T}_{\text{train}}$, and updates are only applied at these steps. For absorbing diffusion models, as each token changes only once, $\text{NFEs}=|\mathcal{T}_{\text{change}}|\leq\min(d,T)$.

#### Continuous-time models

For models trained over continuous time, $d$ change points $\tau_{i}$ are pre-sampled and sorted in ascending order ($\tau_{n_{1}}<\cdots<\tau_{n_{d}}$). Updates are sequentially applied at these points, resulting in $\text{NFEs}=d$, which resembles the sampling process in AO-ARM. However, how to reduce the NEFs to less than $d$ for the continuous-time model has not been investigated.
In contrast to , which pre-sample specific time points and update tokens only at those predetermined points, RADD leverages a time-independent parameterization that updates tokens only when they change. This fundamental difference results in different applicable scenarios for the two methods:

- •
applies to both absorbing and multinomial diffusion models. However, for continuous-time models like SEDD, their sampling method results in $\text{NFEs}=d$ and, as noted, how to reduce the NEFs less than $d$ for the continuous-time model has not been investigated.
- •
RADD, on the other hand, is specifically designed for absorbing diffusion models. It is straightforward to apply the cache strategy of the RADD to continuous-time settings and reduce NEFs to less than $d$ because the input of the model is independent of the time.

Although the samplers in our paper and in originate from different formulations, the results of Theorem D.1 for the discrete-time sampler align with those of our [Eq.˜3.4](#S3.E4). However, as discussed above, the two samplers are applicable in different scenarios.

## Appendix G Details of AO-ARMs

Any-order autoregressive models (AO-ARMs)   model the joint distribution autoregressively for $d!$ different orders $\pi$ of the $d$ variables. Formally, the joint distribution is factorized as $\prod_{k=1}^{d}p(x^{\pi(k)}|x^{\pi(<k)})$ by chain rule. Therefore, AO-ARMs actually define $\sum_{k=0}^{d}C_{d}^{k}(d-k)=d2^{d-1}$ distinct univariate conditionals probabilities.

#### Architecture

AO-ARMs model all univariate conditionals via a weight-sharing neural network, by using the [M] token for variables that are not present in the condition set(^4^44Condition set corresponds to variables in $x^{\pi(<k)}$.).
For efficient parallel optimization, the architecture is designed such that given the condition set of size $k$, it can predict all $d-k$ univariate conditionals at once, similar to the output of conditional distributions in [Fig.˜1](#S3.F1).

#### Training

AO-ARMs are trained to minimize the negative joint likelihood of a datapoint ${\bm{x}}_{0}$ under the expectation over the uniform distribution $U_{\pi}$ of orders. It can be simplified by treating $l$ as a random variable with a uniform distribution over cardinalities 1 to $d$. Further, it can be transformed into a form for better parallel optimization:

$$ $\displaystyle\mathcal{L}_{AO}({\bm{x}}_{0})$ $\displaystyle=\mathbb{E}_{\pi\sim U_{\pi}}\sum_{l=1}^{d}-\log q_{\theta}(x_{0}^{\pi(l)}|x_{0}^{\pi(<l)};\pi)$ (G.1) $\displaystyle=\mathbb{E}_{\pi\sim U_{\pi}}d\cdot\mathbb{E}_{l\sim U(1,\cdots,d)}-\log q_{\theta}(x_{0}^{\pi(l)}|x_{0}^{\pi(<l)};\pi)$ (G.2) $\displaystyle=d\cdot\mathbb{E}_{l\sim U(1,\cdots,d)}\frac{1}{d-l+1}\mathbb{E}_{\pi\sim U(S_{d})}\sum_{r=l}^{d}-\log q_{\theta}(x_{0}^{\pi(r)}|x_{0}^{\pi(<l)};\pi).$ (G.3) $$

Training pseudocode of [Eq.˜G.3](#A7.E3) can be referenced in [Algorithm˜1](#alg1).

#### Sampling

The sampling process for AO-ARMs generates data points autoregressively based on a randomly sampled order $\pi$. Starting from an empty sequence initialized with [M] tokens, the model iteratively predicts the next variable based on the current condition set $x^{\pi(<k)}$. Since the order $\pi$ is chosen randomly, AO-ARMs support generating sequences with any desired ordering, aligning with their ability to model $d!$ orderings during training. Sampling pseudocode can be referenced in [Algorithm˜2](#alg2).

## Appendix H Comparison to prior works concerning equivalence discussion

and have both discussed the relationship between absorbing discrete diffusion and AO-ARMs. Below, we provide a detailed comparison of their approaches with ours.
made an early attempt to explore the connection between absorbing discrete diffusion and AO-ARMs. However, their work lacks rigorous proof. Instead, they qualitatively discuss the correlation between the two loss functions. Notably, in Appendix A.3 of , they describe the relationship by stating, "this looks very similar … it is not exactly identical."
In contrast, our work rigorously establishes this connection. By leveraging the continuous-time framework and the time-independent parameterization presented in [Theorem˜1](#Thmtheorem1), we provide a formal proof demonstrating the equivalence between absorbing discrete diffusion and AO-ARM.
establishes the equivalence between ARMs and the ELBO of the absorbing diffusion models directly.
In comparison, our approach follows a different path:

- 1.
the ELBO was first reduced to $\mathcal{L}_{\text{DSE}}^{T}(\bm{x}_{0})$ as [Eq.˜2.11](#S2.E11), as discussed in detail in .
- 2.
Using step-by-step substitutions via [Eq.˜3.7](#S3.E7), we further reduce $\mathcal{L}_{\text{DSE}}^{T}(\bm{x}_{0})$ to $\mathcal{L}_{\text{AO}}$.

Therefore, our approach offers unique contributions by:

#### A rigorous and alternative proof

We leverage the time-independent properties of the reparameterization formulation, providing an alternative and rigorous proof of this equivalence.

#### Equivalence of four losses

Our analysis extends to demonstrate the equivalence of four distinct losses, including $\mathcal{L}_{\text{DSE}}^{T}(\bm{x}_{0})$ and $\mathcal{L}_{\text{AO}}(\bm{x}_{0})$ in [Eq.˜3.7](#S3.E7), offering a deeper understanding of absorbing discrete diffusion.

#### Comprehensive experimental validation

We conduct a thorough study of these loss functions, with results presented in [Tables˜1](#S4.T1) and [2](#S4.T2). To the best of our knowledge, this exploration has not been explored in prior work.

## Appendix I Algorithms for training and sampling

Figure: Algorithm 1 AO-ARM Training

Figure: Algorithm 2 AO-ARM Sampling

Figure: Algorithm 3 Discrete Diffusion Training ($t\textrm{-DCE}$ Loss)

Figure: Algorithm 4 Discrete Diffusion Training ($\lambda\textrm{-DCE}$ Loss)

Figure: Algorithm 5 Discrete Diffusion Sampling (Unconditional)

Figure: Algorithm 6 Discrete Diffusion Sampling (Conditional)

## Appendix J Experimental details

### J.1 Model details

We implemented our RADD model based on the SEDD architecture, an decoder-only transformer model . Our model incorporates rotary positional encoding  but excludes all parts related to time conditioning (i.e., TimeEmbedding, adaLN-zero block ). Instead, we added a softmax operation at the end of the neural network to ensure the output is a valid conditional distribution. This simplified architecture is similar to the standard GPT architecture, except the lack of attention mask and multiple probability output instead of single one.

### J.2 Training details

We trained our RADD models using the following configuration settings:

- •
Batch Size: 512
- •
Learning Rate: $3\times 10^{-4}$
- •
Exponential Moving Average (EMA):0.9999
- •
Gradient Clipping: Gradient norm clipped to 1
- •
Warmup Schedule: Applied for the first 2500 iterations
- •
weight decay: 0.03
- •
dropout rate: 0.02

The hyperparameters were adapted from , with modifications referenced from . The main modifications were setting the weight decay to 0.03 and the dropout rate to 0.02. Due to limited computational resources, we did not perform a hyperparameter search and directly conducted experiments with these settings. Further tuning of hyperparameters may enhance performance.

In terms of training tokens, 400K iterations correspond to approximately 105 billion tokens, while 1000K iterations correspond to about 262 billion tokens. It’s worth noting that the entire OpenWebText dataset contains only 9 billion tokens, meaning that models went through the dataset multiple times during training.

The small models are trained on multi-accelerator compute nodes with float16 precision.

### J.3 Unconditional generation details

For unconditional generation, we employed a log-linear noise schedule.
As illustrated in [Section˜3.2](#S3.SS2), the Euler method and the Tweedie $\tau$-leaping method are equivalent under this case. In practice, the implementation of the Euler method and the Tweedie $\tau$-leaping method remains the same for RADD but differs for SEDD. So we measure the perplexity of SEDD by Tweedie $\tau$-leaping method which performs slightly better while it suffices to measure the perplexity of RADD once.

As suggested by , except for [Table˜4](#A10.T4), all samples are generated using float64 precision of Gumbel-based categorical sampling(abbreviated as fp64 precision below). No annealing methods (e.g., top-p or top-k sampling) were applied in our sampling process.

### J.4 Further evaluation of generative perplexity

#### Runtime and entropy measurement

To evaluate the efficiency of inference and the diversity of samples, we assessed the inference time and unigram entropy averaged across 1024 samples. When calculating unigram entropy, we chose the natural logarithm (ln) instead of the $\text{log}_{2}$.
We provide perplexity and entropy results under both fp64 and fp32 precision in [Tables˜3](#A10.T3) and [4](#A10.T4) respectively.

Under both precisions, RADD and SEDD exhibit comparable perplexity results for the same number of sampling steps. For large sampling steps, perplexity converges under fp64 precision but continues to decrease under fp32 precision. This discrepancy is due to precision errors in fp32, which effectively function as a form of annealing, resulting in deceptively lower perplexity values . To evaluate efficiency, we focus on the sampling time under fp64 precision.

The RADD model consistently required the shortest sampling time while maintaining similar perplexity levels to the SEDD model. Specifically, RADD achieved a speed-up of up to 2.5 to 3 times with large sampling steps, as shown in [Table˜3](#A10.T3). These findings align with the analysis of the E-NFEs in [Fig.˜2(a)](#S3.F2.sf1), validating the effectiveness of the RADD model and the caching strategy. Even with 1024 sampling steps(equal to sequence length), the cache strategy still enables about 1.5 times acceleration.

**Table 3: Average inference time, perplexity, and entropy per sample with varying sampling steps under fp64 precision. The table compares the average inference time (in seconds), perplexity (PPL), and entropy for the SEDD medium model using Tweedie $\tau$-leaping sampling methods, as well as the RADD medium model under a log-linear noise schedule with a caching strategy. The experiment is conducted on a single high-memory accelerator with a batch size of 16.**
|  | Steps | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEDD-medium | Time(s) | 0.52 | 0.99 | 1.91 | 3.78 | 7.49 | 14.93 | 29.82 | 59.56 |
|  | PPL$\downarrow$ | 159 | 113 | 94 | 87 | 84 | 86 | 82 | 81 |
|  | Entropy | 8.26 | 8.18 | 8.14 | 8.12 | 8.09 | 8.10 | 8.09 | 8.07 |
| RADD-medium | Time(s) | 0.45 | 0.80 | 1.54 | 2.96 | 5.32 | 8.39 | 12.28 | 18.20 |
|  | PPL$\downarrow$ | 158 | 113 | 96 | 89 | 84 | 83 | 81 | 81 |
|  | Entropy | 8.28 | 8.21 | 8.18 | 8.15 | 8.14 | 8.12 | 8.13 | 8.13 |

**Table 4: Average perplexity, and entropy per sample with varying sampling steps under fp32 precision. The table compares the average perplexity (PPL), and entropy for the SEDD medium model using Tweedie $\tau$-leaping sampling methods, as well as the RADD medium model under a log-linear noise schedule with a caching strategy.**
|  | Steps | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEDD-medium | PPL$\downarrow$ | 125 | 85 | 62 | 51 | 41 | 34 | 27 | 22 |
|  | Entropy | 8.18 | 8.07 | 7.96 | 7.86 | 7.73 | 7.60 | 7.44 | 7.25 |
| RADD-medium | PPL$\downarrow$ | 126 | 84 | 63 | 51 | 41 | 33 | 27 | 22 |
|  | Entropy | 8.19 | 8.09 | 7.98 | 7.88 | 7.75 | 7.59 | 7.45 | 7.27 |

#### Efficiency of our caching strategy with mini-batch

In [Section˜D.5](#A4.SS5), we explored the average NFE for the single sample case.
This concept, however, extends to the mini-batch case.
In a mini-batch, some samples may remain unchanged while others may evolve. To address this, we use a dynamic batch-size strategy. Only the samples that have changed are passed through the neural network for computation. While this still involves a "batch-level NFE", the total NFE per sample is reduced, effectively enhancing efficiency as in the single-sample case.
In comparison, concurrent work does not consider dynamic batch size in their implementation, so their practical acceleration falls short of the theoretical potential.
To further validate the efficiency, we conducted experiments generating 64 samples under various batch sizes. The results, summarized in [Table˜5](#A10.T5), demonstrate that the average generation time with our caching strategy consistently outperforms SEDD across different batch sizes and timestep configurations. We provide a detailed analysis below:

##### Batch Size and Hardware Utilization

For a fixed model, increasing the batch size leads to improved hardware utilization before reaching the maximum batch size, reducing the average sampling time per sample. In the case of RADD, sampling time decreases significantly as the batch size increases from 1 to 4. However, the reduction in sampling time becomes minimal beyond a batch size of 4, indicating that hardware utilization has nearly reached its maximum capacity at this point. This demonstrates the scalability of our approach within the limits of the available compute resources.

##### SEDD vs RADD

Under identical batch sizes and timesteps, RADD consistently outperforms SEDD in sampling speed. This highlights the efficiency of RADD’s design, where the caching mechanism reduces redundant computations and achieves faster generation, especially for larger batch sizes and higher timesteps.

**Table 5: The average inference time across batch sizes and timesteps of SEDD-medium and RADD-medium. Experiments were conducted on a single accelerator with 24GB memory (maximum batch size = 8). The average generation time per sample (in seconds) is averaged over 64 samples.**
| Batch Size $\setminus$ Steps | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEDD-medium |  |  |  |  |  |  |  |  |
| 1 | 0.98 | 1.83 | 3.57 | 6.95 | 13.80 | 27.16 | 54.76 | 109.90 |
| 2 | 0.75 | 1.43 | 2.77 | 5.46 | 10.85 | 21.62 | 43.19 | 86.23 |
| 4 | 0.67 | 1.30 | 2.56 | 5.09 | 10.15 | 20.27 | 40.50 | 80.97 |
| 8 | 0.70 | 1.34 | 2.65 | 5.25 | 10.46 | 20.85 | 41.68 | 83.32 |
| RADD-medium |  |  |  |  |  |  |  |  |
| 1 | 0.77 | 1.34 | 2.56 | 4.86 | 8.73 | 14.00 | 20.45 | 30.70 |
| 2 | 0.60 | 1.08 | 2.05 | 3.98 | 7.32 | 12.32 | 18.88 | 28.97 |
| 4 | 0.50 | 0.97 | 1.87 | 3.65 | 6.75 | 11.26 | 17.67 | 27.67 |
| 8 | 0.51 | 0.97 | 1.90 | 3.71 | 6.76 | 10.87 | 16.76 | 26.58 |

#### Sampling as any-order autoregressive models

As outlined in [Theorem˜1](#Thmtheorem1), ${\bm{c}}_{\theta}$ can be interpreted as a conditional distribution over clean data. One natural approach is to use this directly for generating samples, similar to any-order autoregressive models. However, there are $d!$ possible ways to decompose the joint distribution into conditional distributions. We tested three representative cases:

- •
forward: $p(x^{1}\cdots x^{d})=\prod_{k=1}^{d}p(x^{k}|x^{(<k)})$
- •
backward: $p(x^{1}\cdots x^{d})=\prod_{k=1}^{d}p(x^{k}|x^{(>k)})$
- •
random: $\pi\sim U_{\pi}$, $p(x^{1}\cdots x^{d})=\prod_{k=1}^{d}p(x^{\pi(k)}|x^{\pi(<k)}$)

The results are presented in [Table˜6](#A10.T6). Perplexity was calculated as the average over 1024 samples. For the random case, we calculated the average perplexity across different randomly generated $\pi$, corresponding to the standard AO-ARMs sampling method. It shows that the result of the standard AO-ARMs sampling method aligns closely with the converged perplexity of the $\tau$-leaping method under fp64 precision in large steps. Among the different decomposition orders, the forward order demonstrated the best performance.

**Table 6: Quality of unconditionally generated text evaluated by perplexity ($\downarrow$). For a fixed model, the best perplexity is bolded.**
| Method | RADD-medium |
| --- | --- |
| Forward | 81.70 |
| Backward | 103.68 |
| Random | 83.10 |

### J.5 Further evaluation of zero-shot perplexity

In this section, we provide an extended evaluation of the zero-shot language modeling performance of RADD models trained for 1000k iterations. While the results in the main text focus on models trained for 400k iterations, training for longer durations can slightly improve model performance due to the increased exposure to training data.

**Table 7: Additional zero-shot language modeling perplexity ($\downarrow$) for RADD small models. We present the perplexity for RADD models trained for 1000k iterations based on their corresponding loss.**
| Method | LAMBADA | WikiText2 | PTB | WikiText103 | 1BW |
| --- | --- | --- | --- | --- | --- |
| RADD-$t\textrm{-DCE}$ | 48.92 | 37.44 | 102.49 | 37.20 | 70.58 |
| RADD-$\lambda\textrm{-DCE}$ | 49.74 | 37.13 | 98.84 | 36.66 | 69.77 |
| RADD-AO | 49.43 | 36.86 | 102.36 | 35.25 | 70.71 |

## Appendix K Additional experimental results

### K.1 Additional samples

Additional unconditionally and conditionally generated text of RADD-$\lambda\textrm{-DCE}$ small and medium models are reported in Figs. [3](#A11.F3) to [6](#A11.F6).
All of the samples are generated with 1024 steps under a log-linear noise schedule.

Figure: Figure 3: Unconditionally generated text of RADD-$\lambda\textrm{-DCE}$ small.

Figure: Figure 4: Conditionally generated text of RADD-$\lambda\textrm{-DCE}$ small. Prompt tokens are in blue.

Figure: Figure 5: Unconditionally generated text of RADD-$\lambda\textrm{-DCE}$ medium.

Figure: Figure 6: Conditionally generated text of RADD-$\lambda\textrm{-DCE}$ medium. Prompt tokens are in blue.