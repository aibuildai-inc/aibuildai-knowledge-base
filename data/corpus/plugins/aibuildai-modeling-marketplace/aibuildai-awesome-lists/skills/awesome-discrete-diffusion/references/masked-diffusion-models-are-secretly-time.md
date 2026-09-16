---
arxiv_id: "2409.02908"
title: "Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling"
year: 2024
source: arxiv2md
---

## Abstract

Abstract Masked diffusion models (MDMs) have emerged as a popular research topic for generative modeling of discrete data, thanks to their superior performance over other discrete diffusion models, and are rivaling the auto-regressive models (ARMs) for language modeling tasks. The recent effort in simplifying the masked diffusion framework further leads to alignment with continuous-space diffusion models and more principled training and sampling recipes.
In this paper, however, we reveal that both training and sampling of MDMs are theoretically free from the time variable, arguably the key signature of diffusion models, and are instead equivalent to masked models. The connection on the sampling aspect is drawn by our proposed first-hitting sampler (FHS). Specifically, we show that the FHS is theoretically equivalent to MDMs’ original generation process while significantly alleviating the time-consuming categorical sampling and achieving a 20 × \times × speedup. In addition, our investigation raises doubts about whether MDMs can truly beat ARMs in text generation. We identify, for the first time, an underlying numerical issue, even with the commonly used 32-bit floating-point precision, which results in inaccurate categorical sampling.
We show that it lowers the effective temperature both theoretically and empirically, and the resulting decrease in token diversity makes previous evaluations, which assess the generation quality solely through the incomplete generative perplexity metric, somewhat unfair.

## 1 Introduction

Figure: Figure 1: Trilemma of generative modeling for discrete data.
Refer to caption: x1.png

There are three primary paradigms of generative models. Diffusion models  have been the prevalent way for generative modeling of continuous data with both theoretical and empirical success. They are SOTA in image, speech, video synthesis  and serve as the cornerstone of large-scale text-to-image  and text-to-video  generation systems. Auto-regressive models (ARMs) have dominated the generation of discrete data especially languages , due to the scalability and generalizability of the straightforward next-token-prediction mechanism based on transformer architectures . Masked models, such as BERT  for masked language modeling and MaskGIT  for masked image generation, are trained to reconstruct randomly masked tokens and sampled by order-agnostic decoding. They are an alternative approach to model discrete data while suffering from insufficient theoretical foundations.

Diffusion models have been extended to discrete data spaces with principled training and sampling .
Compared to ARMs, they predict all tokens simultaneously and offer a favorable trade-off between generation quality and sampling efficiency.
Recently, masked diffusion models (MDMs), the leading variant of discrete diffusion formulations, are emerging as a promising contender of ARMs . Recent works  have simplified MDMs to align with the design space of diffusion models via continuous-time forward processes, training objectives, and sampling procedures, resulting in a unified view and empirical improvements. Positioned at the intersection of diffusion models and masked models, MDMs are considered promising as they inherit both the theoretical principles from diffusion models and the simple mechanism from masked models. Moreover, it is believed that MDMs can outperform ARMs in text generation when measured by the common generative perplexity metric .

However, we argue that the current understanding of MDMs is still quite limited in both theoretical and empirical aspects. In this paper(^1^11We present our views in a straightforward manner. The earlier version is available in Appendix I.), we conduct a thorough and comprehensive investigation and demonstrate that MDMs are essentially a theoretically and empirically equivalent form of typical masked models while being complicated, inefficient, and numerically unstable:

- 1.
The training objective of MDMs is equivalent to that of masked models, differing only by nuanced likelihood-based loss weighting. The introduction of an additional time variable in the loss function provides little benefit in practice. (Section 3)
- 2.
The sampling process of MDMs, being computationally expensive and inefficient, has a theoretically equivalent alternative (our first-hitting sampler) that is up to 20$\times$ faster and mirrors the random-order, token-by-token decoding process of masked models. (Section 4)
- 3.
The previously reported superiority of MDMs over ARMs on text generation stems from numerical issues that hack the generative perplexity metric during sampling by lowering the effective temperature, rather than genuine advantages. (Section 5)

Based on these findings, we argue that the community should reconsider investing efforts in MDMs. That being said, MDMs do provide theoretical insights and supplementary perspectives to masked models, but in practice, the simpler masked models are not only sufficient but also free from above inefficiency and numerical instability issues. Moreover, scaling up MDMs on text encounters fundamental inference inefficiency challenges compared to ARMs, as the bidirectional attention in masked models is incompatible with KV caching, a crucial technique for accelerating modern large language models (LLMs) that require long context length. This has been largely overlooked or intentionally downplayed by most existing works on diffusion language models(^2^22Given our findings, it may be more appropriate to call them masked language models if based on MDMs.). Considering the critical role of infrastructure and cost-effective inference in the deployment of LLMs, MDMs (or masked models) lack a clear and compelling prospect to replace ARMs.

## 2 Background: Masked Diffusion Models (MDMs)

Let $\mathcal{X}=\{0,1,\dots,m-1\}$ be the discrete data space, with an extra mask token $m$ added to $\mathcal{X}$. Denote $\Delta^{m}=\{\bm{\pi}\in\mathbb{R}^{m+1}|\sum_{i=0}^{m}\pi_{i}=1,\bm{\pi}\geq 0\}$ as the standard $m$-simplex. For any data token or mask token $x\in\mathcal{X}$, denote $\bm{e}_{x}\in\mathbb{R}^{m+1}$ as the corresponding one-hot vector. Continuous-time discrete-space masked diffusion models (MDMs)  can be defined akin to diffusion models, with a continuous-time forward noising process

$$ $q_{t|0}(x_{t}|x_{0})=\mbox{Cat}(\alpha_{t}\bm{e}_{x_{0}}+(1-\alpha_{t})\bm{e}_ {m})$ (1) $$

where $\alpha_{t}$ is the predefined noise schedule function satisfying $\alpha_{0}\approx 1,\alpha_{1}\approx 0$, and $\mbox{Cat}(\bm{\pi})$ denotes the categorical distribution over the class probabilities $\bm{\pi}\in\Delta^{m}$. The forward process has a time reversal for $s<t$ given $x_{0}$:

$$ $q_{s|t,0}(x_{s}|x_{t},x_{0})=\begin{cases}\mbox{Cat}(\bm{e}_{x_{t}}),\quad&x_{ t}\neq m\\ \mbox{Cat}\left(\frac{(1-\alpha_{s})\bm{e}_{m}+(\alpha_{s}-\alpha_{t})\bm{e}_{ x_{0}}}{1-\alpha_{t}}\right),\quad&x_{t}=m\end{cases}$ (2) $$

Following DDPM , the parameterized model is defined by replacing $\bm{e}_{x_{0}}$ in the reversal with a data prediction model $\bm{\mu}_{\theta}:\mathcal{X}\times\mathbb{R}\mapsto\Delta^{m}$:

$$ $p_{\theta}(x_{s}|x_{t})\coloneqq q(x_{s}|x_{t},\bm{e}_{x_{0}}\leftarrow\bm{\mu }_{\theta}(x_{t},t))$ (3) $$

and $\bm{\mu}_{\theta}$ is further parameterized by $\bm{f}_{\theta}:\mathcal{X}\times\mathbb{R}\mapsto\mathbb{R}^{m}$ as

$$ $\bm{\mu}_{\theta}(x_{t},t)=\begin{cases}[\mathrm{softmax}(\bm{f}_{\theta}(x_{t },t)),0],\quad&x_{t}=m\\ \bm{e}_{x_{t}},\quad&x_{t}\neq m\end{cases}$ (4) $$

so that it satisfies (1) the predicted vector contains valid class probabilities sum to 1;
(2) the predicted $x_{0}$ has zero probability of being the mask token;
(3) if a token is already unmasked, it no longer changes.
When $\alpha_{0}\rightarrow 1,\alpha_{1}\rightarrow 0$ and the number of timesteps tends to infinity, it is proven that the parameterized model $p_{\theta}$ has an evidence lower bound (ELBO) $\log p_{\theta}(x_{0})\geq-\mathcal{L}_{\infty}$, where

$$ $\mathcal{L}_{\infty}=\int_{0}^{1}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}} \mathbb{E}_{q_{t|0}(x_{t}|x_{0})}\left[\delta_{x_{t},m}\bm{e}_{x_{0}}^{\top} \log\bm{\mu}_{\theta}(x_{t},t)\right]\mathrm{d}t$ (5) $$

is a time-weighted cross-entropy loss, $\alpha_{t}^{\prime}=\frac{\mathrm{d}\alpha_{t}}{\mathrm{d}t}$, and $\delta_{x_{t},m}$ is a indicator function. We refer to $\mathcal{L}_{\infty}$, the training objective, as the negative ELBO (NELBO).

##### Multi-Dimensional Case

For a token sequence $\bm{x}\in\mathcal{X}^{L}=\{0,1,\dots,m-1,m\}^{L}$ of length $L$, MDMs choose a factorized forward process $q_{t|0}(\bm{x}_{t}|\bm{x}_{0})=\prod_{l=1}^{L}q_{t|0}(x_{t}^{(l)}|x_{0}^{(l)})$ over different dimensions, where $x^{(l)}$ denotes the $l$-th token of $\bm{x}$. As a result, the reversal $q_{s|t,0}(\bm{x}_{s}|\bm{x}_{t},\bm{x}_{0})=\prod_{l=1}^{L}q_{s|t,0}(x_{s}^{(l
)}|x_{t}^{(l)},x_{0}^{(l)})$ and the parameterized model $p_{\theta}(\bm{x}_{s}|\bm{x}_{t})=\prod_{l=1}^{L}q(x_{s}^{(l)}|x_{t}^{(l)},\bm
{e}_{x_{0}^{(l)}}\leftarrow\bm{\mu}_{\theta}^{(l)}(\bm{x}_{t},t))$ also factorize. Here the network $\bm{\mu}_{\theta}:\mathcal{X}^{L}\times\mathbb{R}\mapsto(\Delta^{m})^{L}$ predicts the probabilities at all positions at a time, and we use $\bm{\mu}_{\theta}^{(l)}$ to denote the $l$-th column of $\bm{\mu}_{\theta}$. The ELBO loss in Eqn. (5) under multi-dimension can be written as

$$ $\mathcal{L}_{\infty}^{(L)}=\int_{0}^{1}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t} }\mathbb{E}_{q_{t|0}(\bm{x}_{t}|\bm{x}_{0})}\left[\sum\nolimits_{l:x_{t}^{(l)} =m}\bm{e}_{x_{0}^{(l)}}^{\top}\log\bm{\mu}_{\theta}^{(l)}(\bm{x}_{t},t)\right] \mathrm{d}t$ (6) $$

##### Context of Discrete Diffusion Models

MDMs described above are a simplified version of the
best-performing masked (or absorbing) case in discrete-space diffusion models. Discrete diffusion models, originated from D3PM , rely on discrete-time or continuous-time Markov chains to model transitions in discrete space. Notably, concrete score  in discrete diffusion acts as an analog of the score function in continuous diffusion, and a recent work SEDD  proposes score entropy for robust and scalable learning of the concrete score. The model definition (Markov chain, score parameterization), training objective (diffusion-weighted denoising score entropy) and sampling procedure (Tweedie $\tau$-leaping) of SEDD Absorb can be proven equivalent to the simplified expressions (Eqn. (1) (3) (4) (5)) in MDMs. Interested readers can refer to Appendix D for further details.

## 3 Revisiting the Training of MDMs

MDMs are defined and trained by the continuous-time forward process (Eqn. (1)), time-dependent network parameterization (Eqn. (4)) and continuous-time ELBO (Eqn. (5)). However, different from continuous-time diffusion models , the evolution of $\bm{x}_{t}$ is discrete. The evolution trajectories of $(\bm{x}_{t},t)$ are like pairs of “phenotype" and “genotype", where the continuous changes in time $t$ may not be reflected on the observable traits of $\bm{x}_{t}$. In this section, we aim to disentangle the internal time variable $t$ and the external traits of the masked sequence $\bm{x}_{t}$ in the training of MDMs.

### 3.1 Reformulating the ELBO with the Number of Masked Tokens

Previous works  show the invariance of the ELBO to the noise schedule $\alpha_{t}$ by performing the time change-of-variable $\gamma=\log(1-\alpha_{t})$ or $\lambda=\log\frac{\alpha_{t}}{1-\alpha_{t}}$ following VDM .
However, this does not get to the essence as they still rely on an internal continuous time.
In the following proposition, we show that the sequence NELBO of MDMs can be expressed as a partition by the number of masked tokens instead of the continuous time.

###### Proposition 3.1 (ELBO by the Number of Masked Tokens) .

For $\bm{x}_{0}$ with sequence length $L$, denote $\bm{x}_{n}$ as a sequence with $n$ masked tokens, and $\tilde{q}(\bm{x}_{n}|\bm{x}_{0})$ as the discrete forward process which randomly and uniformly masks $n$ tokens of $\bm{x}_{0}$. Suppose the noise schedule $\alpha_{t}$ satisfies $\alpha_{0}=1,\alpha_{1}=0$. The sequence NELBO in Eqn. (6) can be reformulated as

$$ $\mathcal{L}_{\infty}^{(L)}=-\sum_{n=1}^{L}\mathbb{E}_{\tilde{q}_{n|0}(\bm{x}_{ n}|\bm{x}_{0})}\left[\frac{1}{n}\sum\nolimits_{l:x_{n}^{(l)}=m}\bm{e}_{x_{0}^{ (l)}}^{\top}\log\bar{\bm{\mu}}^{(l)}_{\theta}(\bm{x}_{n})\right]$ (7) $$

where

$$ $\log\bar{\bm{\mu}}_{\theta}(\bm{x}_{n})=\mathbb{E}_{\alpha_{n}\sim\mathcal{B}( L-n+1,n)}\left[\log\bm{\mu}_{\theta}(\bm{x}_{n},\alpha^{-1}(\alpha_{n}))\right],$ (8) $$

$\alpha^{-1}$ is the inverse function of $\alpha_{t}$ satisfying $\alpha^{-1}(\alpha_{t})=t$, and $\mathcal{B}(a,b)$ denotes the Beta distribution with shape parameters $a,b>0$.

This expression offers two aspects of theoretical insights:

Figure: Figure 2: Probability density function (PDF) of $\mathcal{B}(L-n+1,n)$ with $L=1024$.
Refer to caption: x2.png

###### Proposition 3.1 (ELBO by the Number of Masked Tokens) .

##### Mixture of Experts

From Eqn. (8), the time-dependent network $\bm{\mu}_{\theta}(\bm{x},t)$ implicitly parameterizes a time-independent network $\bar{\bm{\mu}}_{\theta}(\bm{x})$ by aggregating the logarithm at the same $\bm{x}$ but different $t$, which can be seen as an ensemble. The time $t$ is sampled unevenly so that $\alpha_{t}$ follows a Beta distribution $\mathcal{B}(L-n+1,n)$. This distribution has the mode (peak) $\frac{L-n}{L-1}$ and variance $\frac{n(L-n+1)}{(L+1)^{2}(L+2)}\leq\frac{1}{4(L+2)}$. With a large sequence length $L$, the variance is small and the distribution is concentrated around the mode, as illustrated in Figure 2. Moreover, under the best-performing linear schedule $\alpha_{t}=1-t$ in MDMs , the mode of $t$ is $\frac{n-1}{L-1}$, close to the masked ratio $\frac{n}{L}$. Therefore, the time variable $t$ can be seen as a continuous relaxation and smoothing of the masked ratio, and we can directly condition the network on the discretely distributed masked ratio instead of the continuous time while yielding similar performance (Appendix J.1).

##### Discrete ELBO

From Eqn. (7), the sequence NELBO can be expressed discretely with the time-agnostic network $\bar{\bm{\mu}}_{\theta}(\bm{x})$. Therefore, Eqn. (7) can serve as a NELBO of masked models in a straightforward way: uniformly choose the number of masked tokens $n$ from $\{1,\dots,L\}$, uniformly mask $n$ random tokens in $\bm{x}_{0}$ to obtain $\bm{x}_{n}$, and compute the average cross-entropy loss of $\bar{\bm{\mu}}_{\theta}(\bm{x})$ on these $n$ positions. The weighting $\frac{1}{n}$ in this NELBO resembles the likelihood weighting in diffusion models , facilitating maximum likelihood training of masked models. Note that early works on order-agnostic auto-regressive models  already reveal this weighting from a different perspective(^3^33The relation between ELBOs of order-agnostic ARMs and MDMs was also mentioned in a recent work , while they only consider an originally time-agnostic network instead of mixture-of-experts.). While in the context of masked models, there are few discussions on the ELBO. Discussions on related work are placed in Appendix B.

### 3.2 Time-Independent Network Parameterization

When the original network $\bm{\mu}_{\theta}$ is parameterized without the time input, we have $\bar{\bm{\mu}}_{\theta}=\bm{\mu}_{\theta}$ in Eqn (7). In this case, the training of MDMs is completely free from the time variable and behaves like masked models. The rationality of time-independent network parameterization has been discussed in recent works . Here we restate this conclusion with our simplified notations from the perspective of the optimal model.

###### Proposition 3.2 (Optimal Masked Diffusion Model) .

Given unlimited model capacity, the optimal network $\theta^{*}$ that minimizes the NELBO in Eqn. (6) satisfies

$$ $\bm{\mu}_{\theta^{*}}^{(l)}(\bm{x},t)=\mathbb{E}_{\tilde{q}_{0|N(\bm{x})}(\bm{ x}_{0}|\bm{x})}\left[\bm{e}_{x_{0}^{(l)}}\right]$ (9) $$

where $N(\bm{x})$ is a deterministic function that counts the number of masked tokens in $\bm{x}$, and $\tilde{q}_{0|n}(\bm{x}_{0}|\bm{x}_{n})$ is the posterior distribution of the discrete forward process $\tilde{q}_{n|0}(\bm{x}_{n}|\bm{x}_{0})$.

From the above expression, the optimal MDM is irrelevant to the time variable, justifying
the feasibility of removing the time input. Besides, it can be extended to a general weighted cross-entropy loss $\mathcal{L}_{\bm{w}}^{(L)}=-\sum_{n=1}^{L}w_{n}\mathbb{E}_{\tilde{q}_{n|0}(\bm
{x}_{n}|\bm{x}_{0})}\left[\sum\nolimits_{l:x_{n}^{(l)}=m}\bm{e}_{x_{0}^{(l)}}^
{\top}\log\bm{\mu}^{(l)}_{\theta}(\bm{x}_{n})\right]$ of masked models. $\mathcal{L}_{\bm{w}}^{(L)}$ with arbitrary positive weights $\bm{w}>0$ yields the same optimal solution as Eqn. (9), thus acting as a surrogate objective of the NELBO. This theoretically supports a wide range of objectives for training masked models, such as the loss in MaskGIT .

###### Proposition 3.2 (Optimal Masked Diffusion Model) .

### 3.3 Practical Considerations

While there are theoretically equivalent variants for training MDMs (continuous-time/discrete ELBO, time-conditioned/time-independent network), these choices may have practical implications due to differences in network inputs and loss variances. We present some training comparisons and our attempts to improve training (e.g., variance reduction, flow matching) in Appendix J.1. Overall, all options yield similar performance, and the low-discrepancy sampler , when applied to time or the number of masked tokens, can significantly reduce the loss variance.

Note that while several works  suggest that MDMs are competitive with ARMs in language modeling (beating GPT-2  when measured by test/zero-shot perplexity), a more fair comparison (retraining ARMs with the same configurations, Appendix J.1)  indicates that MDMs are only advantageous in language understanding tasks (surpassing ARMs and BERT on the GLUE metric ).

## 4 Revisiting the Sampling of MDMs

In the previous section, we demonstrate how the training of MDMs, both theoretically and empirically, can be disentangled with the continuous time variable and behave like masked models. In this section, we turn our attention to the sampling of MDMs, which is also performed in continuous time and seems distinct from masked models. We aim to address its current inefficiency problem as well as establish essential insights into its connection with masked models.

### 4.1 Inefficiency of Current Sampling

MDMs are sampled in an ancestral way following the parameterized reverse-time process in Eqn. (3). Specifically, the sampling step $\bm{x}_{t}\rightarrow\bm{x}_{s}$ from time $t$ to $s<t$ can be expressed as

$$ $x_{s}^{(l)}\begin{cases}=x_{t}^{(l)},\quad&x_{t}^{(l)}\neq m\\ \sim\mbox{Cat}\left(\frac{(1-\alpha_{s})\bm{e}_{m}+(\alpha_{s}-\alpha_{t})\bm{ \mu}^{(l)}_{\theta}(\bm{x}_{t},t)}{1-\alpha_{t}}\right),\quad&x_{t}^{(l)}=m \end{cases},\quad\text{for every }l$ (10) $$

Given the number of sampling steps $N$, the sampling process involves first discretizing the timesteps as $0=t_{0}<t_{1}<\dots<t_{N}=1$, and then performing reverse steps $t_{N}\rightarrow t_{N-1}\rightarrow\dots\rightarrow t_{0}$ according to Eqn. (10). Notable characteristics of MDM’s sampling include: (1) Any mask token can only be unmasked once with no further changes. (2) Each sampling step requires a forward pass through the network $\bm{\mu}_{\theta}$ and conducting at most $L$ times of $|\mathcal{X}|$-dimensional categorical sampling, where $L$ is the sequence length and $|\mathcal{X}|$ is the vocabulary size. (3) The number of sampling steps $N$ can be significantly larger than $L$, and a single sampling step may result in no changes to any token in the sequence. (4) As MDMs are trained with the continuous-time ELBO which assumes an infinite number of reverse steps, it is theoretically rigorous to employ an equivalently large $N$.

Recent works propose a simple caching strategy  to speedup the sampling of MDMs: when the network $\bm{\mu}_{\theta}$ is parameterized without time input(^4^44In our practice, the time-dependent network also exhibits no performance degradation with the caching strategy, so this assumption is unnecessary.), and the sequence is not changed in a sampling step $t\rightarrow s$ (i.e., $\bm{x}_{s}=\bm{x}_{t}$), we can reuse the network output at the last step as $\bm{\mu}_{\theta}(\bm{x}_{s})=\bm{\mu}_{\theta}(\bm{x}_{t})$. As the sequence changes at most $L$ times during sampling, the number of function evaluations (NFE) can be reduced to no more than $L$. However, sampling with the caching strategy still suffers from two major inefficiency problems:

##### Categorical Sampling is Time-Consuming

In diffusion models, NFE is an efficient indicator of the sampling speed, as the computation overhead beyond the network forward passes is negligible. However, in MDMs, the Gumbel-based(^5^55We will introduce Gumbel-based categorical sampling in the next section.) categorical sampling, which requires sampling a total number of $\mathcal{O}(NL|\mathcal{X}|)$ uniform variables and performing logarithmic operations on them, can be expensive compared to network evaluations. As illustrated in Figure 3a, when the number of sampling steps $N\gg L$, the sampling time scales with $N$ instead of the NFE. Categorical sampling steps that do not result in token changes are wasted, as they contribute no information gain.

Figure: (a) Sampling time per sequence (caching strategy, batch size=1)
Refer to caption: x3.png

##### Caching Strategy Degrades in Batched Sampling

When using the caching strategy in batched sampling, the network output can only be reused directly when all the sequences in the batch remain unchanged after a sampling step(^6^66We can reuse only the unchanged part of a batch, but this potentially reduce parallel efficiency.). Suppose the batch size is $B$, and the default linear noise schedule $\alpha_{t}=1-t$ as well as uniform timesteps $t_{k}=\frac{k}{N}$ is used. The expected NFE under the caching strategy can be derived as $N(1-(1-\frac{1}{N})^{BL})$ (proof in Appendix E), similar to the $B=1$ case in . As $\lim_{N\rightarrow\infty}N(1-(1-\frac{1}{N})^{BL})=BL$, the NFE is no longer upper bounded by the sequence length but scales with the batch size (Figure 3b).

### 4.2 First-Hitting Sampler

Figure: Algorithm 1 First-Hitting Sampling of MDMs

The current sampling methods of MDMs, including the caching strategy, are neither efficient nor insightful into the essence of MDMs. To address this, we reexamine the sampling step in Eqn. (10).

Figure: Figure 4: Illustration of the first-hitting sampler in comparison to the original sampling procedure.
Refer to caption: x5.png

When the number of sampling steps $N\rightarrow\infty$ and the maximum step size $\max_{1\leq i\leq N}|t_{i}-t_{i-1}|\rightarrow 0$, Eqn. (10) tends to an infinitesimal jump. In this case, the reverse sampling process becomes a continuous-time Markov chain (or Markov process), where each mask token is unmasked at some moment according to the network prediction. Our key insight involves three folds: (1) Whether a mask token will transit or not during a time interval $[s,t]$ is independent of the network. The network output only determines which token is the transition target given the condition that the transition happens. (2) The transition probability $\frac{\alpha_{s}-\alpha_{t}}{1-\alpha_{t}}$ is equal for masked tokens at different positions. Therefore, each mask token has the same probability of being first unmasked. (3) The first-hitting time, which denotes the first moment any of the remaining masked tokens is unmasked, can be analytically sampled:

###### Proposition 4.1 (Analytic Sampling of First-Hitting Time) .

Denote $\tau_{L}=1$ as the initial time. Suppose there are $n$ masked tokens, and the last time a token is unmasked happens at $\tau_{n}$, then the next time a token is unmasked can be analytically sampled by

$$ $\tau_{n-1}=\alpha^{-1}(1-u_{n}^{1/n}(1-\alpha_{\tau_{n}})),\quad u_{n}\sim \mathcal{U}(0,1)$ (11) $$

where $\mathcal{U}(0,1)$ is the uniform distribution on $[0,1]$.

As outlined in Algorithm 1, by recursively sampling the next time when any of the remaining mask tokens is first unmasked, then uniformly choosing a mask token and unmasking it according to the network output, we obtain a token-by-token sampling procedure of MDMs. Denote $\bm{x}_{n}$ as the sequence with $n$ remaining mask tokens. Since the transition $\bm{x}_{n}\rightarrow\bm{x}_{n-1}$ can be considered to happen in the infinitesimal step $\tau_{n-1}+\mathrm{d}t\rightarrow\tau_{n-1}$, using the network output $\bm{\mu}_{\theta}(\bm{x}_{n},\tau_{n-1})$ at
time $\tau_{n-1}$ incurs no approximation errors. Therefore, the first-hitting sampler (FHS) is theoretically equivalent as simulating the continuous-time reverse Markov sampling process. We illustrate the comparison between the FHS and the original sampling procedure in Figure 4.

The FHS demonstrates appealing properties:

###### Proposition 4.1 (Analytic Sampling of First-Hitting Time) .

##### Tackling the Sampling Inefficiency

The FHS can tackle the two inefficiency problems described in Section 4.1. Firstly, as the categorical sampling is only conducted for determining the transition target of the single chosen mask token at each step, the total computation cost is reduced to $\mathcal{O}(L|\mathcal{X}|)$. Secondly, the first-hitting time $\tau_{n}$ can be sampled independently and asynchronously across different samples in a batch, avoiding performance degradation in batched sampling.

##### Connection to the Sampling of Masked Models

When the network parameterization is independent of the time, the FHS in Algorithm 1 can be completely free from the time and become a token-by-token decoding process akin to masked models. This connection serves as supporting evidence for the typical sampling procedure of masked models, as it is theoretically equivalent to the more principled reverse Markov sampling process of MDMs.

### 4.3 Parallel Decoding and High-Order Variants

Figure: Figure 5: Variants of the first-hitting sampler. $\bm{x}_{l}$ denotes the sequence with $l$ remaining mask tokens, and $\bm{\mu}_{l}=\bm{\mu}_{\theta}(\bm{x}_{l},\tau_{l-1})$ denotes the network prediction at the step $l$.
Refer to caption: x6.png

The token-by-token decoding process of MDMs can be extended to parallel decoding by unmasking multiple tokens per step, as the network $\bm{\mu}_{\theta}$ predicts tokens at all positions. This enables speed-quality trade-offs similar to diffusion models. As illustrated in Figure 5, parallel decoding essentially reuses the previous network output to reduce the NFE, thus functioning as an approximation method.

To reduce the approximation error, we follow the recipes of
high-order diffusion solvers  to develop high-order samplers of MDMs. We propose two variants: one based on extrapolating previous network outputs, and the other utilizing a predictor-corrector method to refine the samples (algorithms in Appendix G.1).

## 5 Are MDMs Better than ARMs? A Critical Fault in Low-Precision Gumbel-Based Categorical Sampling

Before we proceed to verify the effectiveness of our proposed first-hitting sampler, we have to point out a critical fault in MDMs’ original sampling implementation. As suggested by previous works , MDMs seem to surpass ARMs with a sufficient number of sampling steps when measured by the generative perplexity (Gen PPL)(^7^77The evaluation metrics used in this paper are introduced in Appendix H.1.), as shown in Figure 7a. However, in this section, we identify for the first time a hidden numerical issue existing in previous codebases that makes this observation questionable.

### 5.1 Low Token Diversity under Numerous Sampling Steps

Figure: Figure 6: Segment of generated text by SEDD Absorb at 50k sampling steps.

Figure: (a) Generative Perplexity
Refer to caption: x7.png

Empirically, a reduction in Gen PPL is observed by increasing the inference budget. In particular, an exceptionally low Gen PPL ($<15$) is achieved when the number of sampling steps approaches 50k.

However, when we check the generated content, we discover that the quality is compromised by low token diversity (an extreme case is shown in Figure 6). We further quantify this phenomenon by measuring the sentence entropy (Figure 7b). With the original sampler, the Gen PPL of MDMs surpasses ARMs at around 2k steps, but the entropy is always lower and keeps decreasing.

This low generation quality is unexpected, as theory suggests that increasing sampling steps should yield lower discretization errors and more faithfully reflect the true model performance. We therefore consider this a hidden implementation issue and investigate further to identify the root cause.

### 5.2 Identifying the Numerical Precision Problem

**Table 1: Maximum Gumbel under different floating-point precisions.**
| Data Type | Structure (bits) | Maximum Value ($<1$) Representable | Maximum Gumbel |  |  |
| --- | --- | --- | --- | --- | --- |
| Sign | Exponent | Fraction |  |  |  |
| float32 | 1 | 8 | 23 | $1-2^{-24}\approx 0.9999999404$ | $-\log(-\log(1-2^{-24}))\approx 16.6355$ |
| float64 | 1 | 11 | 52 | $1-2^{-53}\approx 0.999999999999999889$ | $-\log(-\log(1-2^{-53}))\approx 36.7368$ |

Figure: Figure 8: Code for different versions of Gumbel-based categorical sampling. The operation $\operatornamewithlimits{argmax}_{i}(\log\pi_{i}-\log(-\log u_{i}))$ is simplified to $\operatornamewithlimits{argmax}_{i}(\pi_{i}/(-\log u_{i}))$ to save computation cost.
Refer to caption: x9.png

Our key observation is that, when we alter the floating-point precision during sampling from 32-bit to 64-bit, the entropy returns to a normal level similar to ARMs, but with a generative perplexity $\approx 100$. After careful ablations, we identify the root cause as the inaccuracy in previous Gumbel-based categorical sampling. To sample from a categorical distribution with class probabilities $\{\pi_{i}\}_{i=1}^{K}$, Gumbel-max trick(^8^88A brief introduction to Gumbel tricks is provided in Appendix F) is used by first sampling $K$ independent uniform variables $u_{i}\sim\mathcal{U}(0,1)$, then transforming them into samples from the standard Gumbel distribution $\mathcal{G}(0,1)$ by $g_{i}=-\log(-\log u_{i})$, and finally obtaining the categorical sample $n=\operatornamewithlimits{argmax}_{i}(\log\pi_{i}+g_{i})$. The operation $g=-\log(-\log u)$ theoretically maps $u\in[0,1]$ to $g\in(-\infty,+\infty)$. But due to the limited representation ability of floating-point numbers in implementation, $u$ is constrained to $[0,1-\epsilon]$ and $g$ is constrained to $(-\infty,M]$, as shown in Table 1. Therefore, the sample $g$ instead follows a truncated Gumbel distribution, denoted $\mathcal{T}\mathcal{G}(0,1,M)$, which refers to the Gumbel distribution $\mathcal{G}(0,1)$ conditioned on $g\leq M$. This tricky difference theoretically makes the categorical sampling inaccurate, i.e., $\operatornamewithlimits{argmax}_{i}(\log\pi_{i}+g_{i})$ no longer follows the class probabilities $\{\pi_{i}\}_{i=1}^{K}$.

**Table 2: Results with different versions of categorical sampling.**
| Version | Gen PPL | Entropy |
| --- | --- | --- |
| 32-bit | 31.24 | 5.17 |
| 64-bit | 126.11 | 5.66 |
| 64-bit + trunc | 28.64 | 5.12 |

To verify that truncation is the fundamental issue, we conduct ablations by only modifying the categorical sampling code. As shown in Figure 8, we manually scale 64-bit uniform samples to match the truncation in the 32-bit case. We then randomly generate 8 samples with 2048 steps and compare the average generative perplexity and entropy in Table 2. The similar results between the 32-bit and truncated 64-bit cases confirm the impact of truncation.

###### Remark 5.1 .

Note that auto-regressive LLMs like Llama  and Mistral  use torch.multinomial for categorical sampling, which is also implemented with the Gumbel-max trick in the low-level C++ code of PyTorch. In contrast, we find the token-by-token decoding process of ARMs and MDMs (by our first-hitting sampler) does not suffer from notable numerical issues under 32-bit precision (illustrations and explanations in Appendix J.2.2).

###### Remark 5.1 .

### 5.3 Categorical Sampling with Truncated Gumbel

In the previous section, we empirically observe that truncated Gumbel-based categorical sampling reduces token diversity. Surprisingly, such effects can be precisely depicted in closed-form.

###### Proposition 5.2 (Closed-Form Categorical Sampling with Truncated Gumbel) .

Suppose the class probabilities are sorted as $\pi_{1}\leq\dots\leq\pi_{K}$, and $g_{i}\sim\mathcal{T}\mathcal{G}(0,1,M)$ are truncated Gumbel samples with maximum value $M$. Denote $\pi_{0}=0$. For $1\leq n\leq K$, we have $P(\operatornamewithlimits{argmax}_{i}(\log\pi_{i}+g_{i})=n)=\pi_{n}\sum_{i=1}^
{n}\beta(i)$,
where

$$ $\beta(i)=\frac{e^{\left(K+1-i-\frac{\sum_{k=i}^{K}\pi_{k}}{\pi_{i}}\right)e^{- M}}-e^{\left(K+1-i-\frac{\sum_{k=i}^{K}\pi_{k}}{\pi_{i-1}}\right)e^{-M}}}{\sum _{k=i}^{K}\pi_{k}}\geq 0$ (12) $$

To the best of our knowledge, this formulation has not been revealed in previous works. Intuitively, with truncated Gumbel, the original class probabilities $\pi_{n}$ are shifted to $\pi_{n}^{\prime}=\pi_{n}\sum_{i=1}^{n}\beta(i)$. This has two main implications: (1) As $\beta(i)\geq 0$ and $\pi_{n}$ are sorted, if $\pi_{n_{1}}>\pi_{n_{2}}$, the adjusted class probabilities satisfy $\frac{\pi_{n_{1}}^{\prime}}{\pi_{n_{2}}^{\prime}}>\frac{\pi_{n_{1}}}{\pi_{n_{2
}}}$. This indicates that relatively larger probabilities are further amplified, creating an effect similar to lowering the temperature. (2) In the sampling step, the probability of unmasking is adjusted based on the network output, resulting in unequal unmasking probabilities at different positions in a sequence. This implies that some tokens are prioritized to be unmasked, further reducing the randomness and overall entropy.

In both aspects, the inaccurate categorical sampling deviates from theoretical correctness and reduces the generation diversity, leading to unfair evaluations of MDMs’ generative performance.

###### Proposition 5.2 (Closed-Form Categorical Sampling with Truncated Gumbel) .

## 6 A Fair Evaluation of MDMs’ Generation

In this section, we will fairly evaluate the generation performance of MDMs and examine the impact of our proposed sampler and the temperature. Our experiments are based on the codebase of MDLM  which is inherited from SEDD . We fix the categorical sampling to 64-bit floating-point precision so that the numerical truncation is negligible. We directly use pretrained models (AR, SEDD Absorb, MDLM) provided by MDLM, which share the same network architecture and were trained with the same configuration. Additional experiment details are provided in Appendix H. We display some generated text in Appendix J.3.1 to illustrate the token diversity under different sampling strategies.

### 6.1 Original Sampler v.s. First-Hitting Sampler

Figure: (a) Generative Perplexity
Refer to caption: x10.png

Figure 9 compares both the generative perplexity and the entropy of different models. For the baselines, SEDD is sampled by their analytic sampler (Tweedie $\tau$-leaping), and MDLM is sampled with and without the caching strategy. For our first-hitting sampler, the parallel decoding is performed by unmasking the same number of tokens per step. High-order variants employ the extrapolation strategy when the number of sampling steps $N\leq 128$, and the predictor-corrector strategy otherwise.

After the numerical problem is fixed, the entropy returns to a normal level (5.60$\sim$5.70) for all models. Besides, our sampler can be up to 20$\times$ faster than previous sampling strategies of MDMs in terms of the wall-clock time(^9^99The efficiency gains (measured by inference wall-clock time) can depend on many factors and may not be as large as 20x in other settings (analyzed in Appendix J.3).). Despite the notable speedup, the true generative perplexity of MDMs is revealed to be around 100, significantly lagging behind that of counterpart ARMs ($<40$).

### 6.2 Trading Off Generative Perplexity and Entropy via Temperature

Figure: Figure 10: Trade-off of generative perplexity and entropy.
Refer to caption: x12.png

The truncation effect of 32-bit floating-point numbers creates a trade-off between generative perplexity and entropy by varying the number of sampling steps (Figure 7). This trade-off arises from a tricky interplay of inaccurate categorical sampling and the approximation error at limited discretization steps. In Figure 10, we demonstrate that this trade-off can be achieved at a lower time cost by using the correct sampling (our 1024-step high-order sampler) and manually adjusting the temperature within the range $[0.8,1.0]$. The trade-off curve of our method is slightly better than the original MDM sampling, while still significantly lagging behind ARMs.

## 7 Conclusion

In this work, we advance our understanding of masked diffusion models (MDMs) by revealing their theoretical equivalence to masked models and uncovering a hidden numerical issue that compromised the fairness of previous evaluations of MDMs’ generative performance. Our findings challenge earlier claims that MDMs can surpass ARMs in text generation. Despite these negative results, we acknowledge that our text-based experiments may inherently favor ARMs, as text naturally follows a left-to-right order that ARMs are better suited to model. Nevertheless, we believe that MDMs may hold potential for applications where an order-agnostic data structure is a key prior, and in practice, simply using masked models may be a better choice.

## Acknowledgments

The team would like to thank Aaron Lou, Cheng Lu from OpenAI, and Jiaxin Shi from Google DeepMind for their valuable discussions and comments. K. Z and J. Z were also supported by the National Natural Science Foundation of China (Nos. 62350080, 62106120, 92270001), Tsinghua Institute for Guo Qiang, and the High Performance Computing Center, Tsinghua University; J. Z was also supported by the XPlorer Prize.

## Appendix A Notations and Definitions

Numbers and Arrays

| $\displaystyle x$ | A scalar representing a discrete token |
| --- | --- |
| $\displaystyle\bm{x}$ | A vector representing a sequence of discrete tokens |
| $x^{(l)}$ | The $l$-th element of $\bm{x}$ |
| $x_{t},\bm{x}_{t}$ | The state(s) at time $t$ |
| $\bm{x}_{n}$ | The sequence with $n$ masked tokens |
| $t$ | The continuous time |
| $m$ | The mask token |
| $n$ | The number of masked tokens in a sequence |
| $\bm{\mu}$ | A matrix, where the $l$-th column represents the predicted transition probabilities at the $l$-th position in a sequence |
| $\bm{\mu}^{(l)}$ | The $l$-th column of $\bm{\mu}$ |
| $\bm{\pi}$ | The class probabilities |
| $\pi_{i}$ | The $i$-th element of $\bm{\pi}$ |
| $L$ | The sequence length |
| $N$ | The number of sampling steps |
| $B$ | The batch size |
| $\theta$ | The neural network parameters |
| $\tau$ | The first-hitting time |
| $\mathcal{L}_{\infty}$ | The continuous-time NELBO loss for a single token |
| $\mathcal{L}_{\infty}^{(L)}$ | The continuous-time NELBO loss for a sequence of length $L$ |

Sets

| $\displaystyle\mathbb{R}$ | The set of real numbers |
| --- | --- |
| $\displaystyle\mathcal{X}$ | The discrete data space (vocabulary) $\{0,1,\dots,m\}$ where $m$ is the added mask token |
| $\Delta^{m}$ | The standard $m$-simplex $\{\bm{\pi}\in\mathbb{R}^{m+1}|\sum_{i=0}^{m}\pi_{i}=1,\bm{\pi}\geq 0\}$ |

Functions

| $\alpha_{t}$ | The pre-defined noise schedule, which is a decreasing function of time $t$ |
| --- | --- |
| $\alpha_{t}^{\prime}$ | The derivative of the noise schedule w.r.t. the time |
| $\alpha^{-1}(a)$ | The inverse function of the noise schedule satisfying $\alpha_{\alpha^{-1}(a)}=a$ |
| $\delta_{x,y}$ | The indicator function (1 when $x=y$ and 0 when $x\neq y$) |
| $\bm{e}_{x}$ | The one-hot vector of the token $x$ |
| $\bm{\mu}_{\theta}(\bm{x},t)$ | The network prediction given the sequence $\bm{x}$ and the time $t$ as input |
| $\mathrm{softmax}(\bm{z})$ | The Softmax operation to transform logits into class probabilities |
| $\log\bm{\mu}$ | The element-wise natural logarithm |
| $N(\bm{x})$ | The function counting the number of masked tokens in the sequence $\bm{x}$ |
| $|\mathcal{X}|$ | The size of the vocabulary $\mathcal{X}$ |

Distributions

| $q$ | The continuous-time forward process |
| --- | --- |
| $\tilde{q}$ | The discrete forward process |
| $p_{\theta}$ | The parameterized reverse process |
| $\mathcal{U}(a,b)$ | The uniform distribution on the interval $[a,b]$ |
| $\mathcal{B}(a,b)$ | The Beta distribution with parameters $a,b>0$ |
| $\mathcal{G}(0,1)$ | The standard Gumbel distribution |
| $\mathcal{T}\mathcal{G}(0,1,M)$ | The right-truncated standard Gumbel distribution with threshold $M$ |
| $\mbox{Cat}(\bm{\pi})$ | The categorical distribution over the class probabilities $\bm{\pi}$ |

Abbreviations

| MDMs | Masked Diffusion Models |
| --- | --- |
| ARMs | Auto-Regressive Models |
| (N)ELBO | (Negative) Evidence Lower Bound |
| NFE | The Number of Function Evaluations |
| PPL | Perplexity |
| Gen PPL | Generative Perplexity |

## Appendix B Related Work

##### Discrete Diffusion Models

Diffusion models are originally built on discrete-time continuous-space Markov chains with Gaussian transition kernels . They are later extended to continuous time with the theory of stochastic processes and score matching .

Discrete diffusion models arise from similar contexts of Markov chains but with discrete data space . D3PM  considers discrete-time Markov chains with several types of transition matrices (uniform, absorbing, discretized Gaussian) and derives the discrete-time variational objective (or ELBO), which is further extended to continuous-time Markov chain (CTMC) and the corresponding ELBO. They employ the mean-parameterization to learn the reverse density $q_{0|t}$.

Another line of work  argues that D3PM implicitly learns the ratio of the marginal distributions $\frac{q_{t}(\hat{x})}{q_{t}(x)}$, which is referred to as the concrete score—a discrete analog to the score function in continuous diffusion. This ratio is proposed to be directly learned via a regression objective known as concrete score matching , similar to score matching in continuous diffusion. However, this approach faces challenges in practice due to the incompatibility of the $L_{2}$ loss and the fact that the ratio $\frac{q_{t}(\hat{x})}{q_{t}(x)}$ must be positive. To address this issue, SEDD  introduces the score entropy objective as a theoretically more robust surrogate, which also connects the concrete score with the continuous-time ELBO.

Though SEDD considers two types of transitions (uniform, absorb), the absorbing case (masked diffusion) is much more performant in practice. It involves adding a [MASK] token as the absorbing state and modeling the simple transitions between the mask state and unmasked states, akin to the mechanism of masked models. Recent studies  have further aligned the masked diffusion framework with continuous diffusion, resulting in simple and principled training and sampling recipes. This not only provides a unified understanding of masked diffusion models but also enables both theoretical and empirical advancements through improved parameterization and engineering techniques. We mainly follow their framework in this work.

##### Masked Models and Order-Agnostic Auto-regressive Models

Learning to reconstruct masked tokens (or patches) is an efficient self-supervised manner for both representation learning and generative modeling. The masked modeling paradigm, originally introduced by BERT , was not initially designed for generative purposes. BERT masks a fixed portion (15%) of tokens at random(^10^1010More specifically, among the 15% tokens, 80% are replaced with the [MASK] token, 10% are replaced with random tokens, and 10% remain unchanged.), which supports representation learning and language understanding rather than generating text from scratch. Similarly, the masked autoencoder (MAE)  adopts this approach for image representation learning but employs a higher masked ratio (75%).

Masked models can be generative when trained on sequences with a range of masked ratios. Mask-Predict  extends the number of masked tokens seen during training in BERT and uses the following objective to train a language generation model:

$$ $\mathcal{L}_{\text{mask}}=-\mathbb{E}_{n\sim p(n)}\mathbb{E}_{\tilde{q}_{n|0}( \bm{x}_{n}|\bm{x}_{0})}\left[\sum\nolimits_{l:x_{n}^{(l)}=m}\bm{e}_{x_{0}^{(l) }}^{\top}\log\bm{\mu}^{(l)}_{\theta}(\bm{x}_{n}))\right]$ (13) $$

where $p(n)$ is the uniform distribution over the sequence length $L$. MaskGIT  uses a similar objective for image generation, but selects the number of masked tokens $n$ according to a mask scheduling function $\gamma(t)$: sample $t\sim\mathcal{U}(0,1)$, and set $n=\lceil\gamma(t)L\rceil$. Both Mask-Predict and MaskGIT generate samples by parallel decoding. Compared to MDMs, these methods have less theoretical grounding in training and sampling. Specifically, there is no discussion of the ELBO (Eqn. (7)) where the likelihood weighting $\frac{1}{n}$ is necessary. Nevertheless, as discussed in Section 3.2, their objectives can still lead to the same optimal solution.

The ELBO of masked models is instead revealed in the context of order-agnostic auto-regressive models . They factorize the model distribution as $p_{\theta}(\bm{x}_{0})=\mathbb{E}_{\sigma\sim\mathcal{U}(S_{L})}\prod_{n=1}^{L
}p_{\theta}(x_{0}^{\sigma(n)}|\bm{x}_{0}^{\sigma(<n)})$ in the style of ARMs, but with an additional expectation over the index permutation $\sigma$ sampled from the uniform distribution on the set of $L$-permutations $S_{L}$. By applying Jensen’s inequality, the ELBO can be derived as:

$$ $\displaystyle\log p_{\theta}(\bm{x}_{0})$ $\displaystyle\geq\mathbb{E}_{\sigma\sim\mathcal{U}(S_{L})}\sum_{n=1}^{L}\log p _{\theta}(x_{0}^{\sigma(n)}|\bm{x}_{0}^{\sigma(<n)})$ (14) $\displaystyle=\mathbb{E}_{\sigma\sim\mathcal{U}(S_{L})}\sum_{n=1}^{L}\frac{1}{ L-n+1}\sum_{k\in\sigma(\geq n)}\log p_{\theta}(x_{0}^{(k)}|\bm{x}_{0}^{\sigma( <n)})$ $$

Here $\log p_{\theta}(x_{0}^{(k)}|\bm{x}_{0}^{\sigma(<n)})$ (predicted data probability given known tokens) is an equivalent expression for the cross-entropy term $\bm{e}_{x_{0}^{(k)}}^{\top}\log\bm{\mu}^{(k)}_{\theta}(\bm{x}_{0}^{\sigma(<n)})$: the cross-entropy extracts the $x_{0}^{(k)}$-th element, $\mu^{(k)}_{\theta}(\bm{x}_{0}^{\sigma(<n)})_{x_{0}^{(k)}}$, from the network prediction
$\bm{\mu}^{(k)}_{\theta}(\bm{x}_{0}^{\sigma(<n)})$ as $p_{\theta}(x_{0}^{(k)}|\bm{x}_{0}^{\sigma(<n)})$. This can be interpreted as a masked prediction where $\bm{x}_{0}^{\sigma(<n)}$ (the first $n-1$ tokens) is known and $\bm{x}_{0}^{\sigma(\geq n)}$ (the remaining $L-n+1$ tokens) is masked and to be predicted. The cross-entropy loss is averaged over the masked positions. As the last $L-n+1$ positions in a random permutation are equivalent to $L-n+1$ random positions without permutation, this ELBO is equivalent to the ELBO in Eqn. (7).

##### Training and Sampling Improvements of Diffusion Models

Since the inception of diffusion models , numerous efforts have been undertaken to enhance their performance, leading to well-established training and sampling recipes.

Prevalent training improvements include designing noise schedules, modifying the parameterization and
applying variance reduction techniques . Notably, flow matching  provides a theoretically equivalent variant of diffusion models by employing the straight-line diffusion paths and velocity parameterization. These techniques have been validated in likelihood training of diffusion models, achieving improved density estimation results on image benchmarks .
The state-of-the-art image diffusion model, EDM , designs the parameterization according to their proposed preconditioning and first principles, which is deeply connected to velocity parameterization . When targeted at maximum likelihood training with the ELBO, instead of improving generation quality (such as FID of generated images), the design space is relatively limited. This is also the case in discrete diffusion for text generation as the perplexity metric is based on likelihood.

Training-free accelerations of diffusion sampling mainly focus on two aspects: reducing stochasticity in the sampling process and leveraging higher-order information. DDIM , along with the extension to diffusion bridges , generalizes the diffusion process to non-Markovian ones with lower levels of stochasticity, enabling faster sampling. Later works connect it to the probability flow ordinary differential equation (PF-ODE) formulations of diffusion models, and build dedicated high-order numerical differential equation solvers .

However, adapting these sampling recipes to discrete diffusion is not feasible, as the underlying evolution process of discrete data cannot be described by an ODE. Designing effective samplers for discrete diffusion requires a specialized inspection of the reverse-time Markov chain. Previous works  leverage the uniformization to convert continuous-time Markov chains into discrete ones, while still requiring time discretizations or approximations of the transition time distribution. Our study is the first to demonstrate that the transition time in MDMs can be sampled analytically without hyperparameter tuning or approximation errors. Infrastructure improvements, such as quantized or sparse attention , can also be used to accelerate the inference of discrete diffusion models, which are beyond the scope of this work.

## Appendix C Proof

### C.1 Proof of Proposition 3.1

###### Proof.

Denote $n_{t}=N(\bm{x}_{t})$ as the number of masked tokens at time $t$. According to the forward process in Eqn. (1), each token is independently masked with a probability $1-\alpha_{t}$, and $n_{t}$ follows the Binomial distribution $B(L,1-\alpha_{t})$. The probability mass function is

$$ $p_{t}(n_{t})=\binom{L}{n_{t}}(1-\alpha_{t})^{n_{t}}\alpha_{t}^{L-n_{t}},\quad n _{t}=0,1,\dots,L$ (15) $$

We can rearrange the sequence NELBO in Eqn. (6) as a partition by the number of masked tokens:

$$ $\displaystyle\mathcal{L}^{(L)}_{\infty}$ $\displaystyle=\int_{0}^{1}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}\mathbb{E}_{ q_{t|0}(\bm{x}_{t}|\bm{x}_{0})}\left[\sum\nolimits_{l:x_{t}^{(l)}=m}\bm{e}_{x_ {0}^{(l)}}^{\top}\log\bm{\mu}_{\theta}^{(l)}(\bm{x}_{t},t)\right]\mathrm{d}t$ (16) $\displaystyle=\int_{0}^{1}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}\mathbb{E}_{ p_{t}(n_{t})}\mathbb{E}_{\tilde{q}_{n_{t}|0}(\bm{x}_{t}|\bm{x}_{0})}\left[\sum \nolimits_{l:x_{t}^{(l)}=m}\bm{e}_{x_{0}^{(l)}}^{\top}\log\bm{\mu}_{\theta}^{( l)}(\bm{x}_{t},t)\right]\mathrm{d}t$ $\displaystyle=\sum_{n=1}^{L}\int_{0}^{1}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t }}p_{t}(n)\mathbb{E}_{\tilde{q}_{n|0}(\bm{x}_{n}|\bm{x}_{0})}\left[\sum \nolimits_{l:x_{n}^{(l)}=m}\bm{e}_{x_{0}^{(l)}}^{\top}\log\bm{\mu}_{\theta}^{( l)}(\bm{x}_{n},t)\right]\mathrm{d}t$ $\displaystyle=\sum_{n=1}^{L}\mathbb{E}_{\tilde{q}_{n|0}(\bm{x}_{n}|\bm{x}_{0}) }\Bigg{[}\sum\nolimits_{l:x_{n}^{(l)}=m}\bm{e}_{x_{0}^{(l)}}^{\top}\Bigg{[} \underbrace{\int_{0}^{1}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}p_{t}(n)\log \bm{\mu}_{\theta}(\bm{x}_{n},t)\mathrm{d}t}_{\text{time-related term}}\Bigg{]} ^{(l)}\Bigg{]}$ $$

The time-related term can be further simplified as

$$ $\displaystyle\int_{0}^{1}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}p_{t}(n)\log \bm{\mu}_{\theta}(\bm{x}_{n},t)\mathrm{d}t$ (17) $\displaystyle=$ $\displaystyle\int_{0}^{1}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}\binom{L}{n}( 1-\alpha_{t})^{n}\alpha_{t}^{L-n}\log\bm{\mu}_{\theta}(\bm{x}_{n},t)\mathrm{d}t$ $\displaystyle=$ $\displaystyle\binom{L}{n}\int_{\alpha_{0}}^{\alpha_{1}}(1-\alpha_{t})^{n-1} \alpha_{t}^{L-n}\log\bm{\mu}_{\theta}(\bm{x}_{n},t)\mathrm{d}\alpha_{t}$ $\displaystyle=$ $\displaystyle-\binom{L}{n}\int_{0}^{1}(1-\alpha_{t})^{n-1}\alpha_{t}^{L-n}\log \bm{\mu}_{\theta}(\bm{x}_{n},t)\mathrm{d}\alpha_{t}$ $\displaystyle=$ $\displaystyle-\binom{L}{n}\frac{(n-1)!(L-n)!}{L!}\mathbb{E}_{\alpha_{n}\sim \mathcal{B}(L-n+1,n)}\left[\log\bm{\mu}_{\theta}(\bm{x}_{n},\alpha^{-1}(\alpha _{n}))\right]$ $\displaystyle=$ $\displaystyle-\frac{1}{n}\underbrace{\mathbb{E}_{\alpha_{n}\sim\mathcal{B}(L-n +1,n)}\left[\log\bm{\mu}_{\theta}(\bm{x}_{n},\alpha^{-1}(\alpha_{n}))\right]}_ {\coloneqq\log\bar{\bm{\mu}}_{\theta}(\bm{x}_{n})}$ $$

which completes the proof.
∎

###### Proof.

### C.2 Proof of Proposition 3.2

###### Proof.

We consider minimizing the sequence NELBO (Eqn. (6)) under the expectation of the data distribution $q_{0}(\bm{x}_{0})$:

$$ $\displaystyle\min_{\theta}\mathbb{E}_{q_{0}(\bm{x}_{0})}\mathcal{L}^{(L)}_{\infty}$ $\displaystyle=\int_{0}^{1}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}\mathbb{E}_{ q_{0}(\bm{x}_{0})}\mathbb{E}_{q_{t|0}(\bm{x}_{t}|\bm{x}_{0})}\left[\sum \nolimits_{l:x_{t}^{(l)}=m}\bm{e}_{x_{0}^{(l)}}^{\top}\log\bm{\mu}_{\theta}^{( l)}(\bm{x}_{t},t)\right]\mathrm{d}t$ (18) $\displaystyle=\int_{0}^{1}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}\mathbb{E}_{ q_{t}(\bm{x}_{t})}\mathbb{E}_{q_{0|t}(\bm{x}_{0}|\bm{x}_{t})}\left[\sum \nolimits_{l:x_{t}^{(l)}=m}\bm{e}_{x_{0}^{(l)}}^{\top}\log\bm{\mu}_{\theta}^{( l)}(\bm{x}_{t},t)\right]\mathrm{d}t$ $\displaystyle=\int_{0}^{1}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}\mathbb{E}_{ q_{t}(\bm{x}_{t})}\left[\sum\nolimits_{l:x_{t}^{(l)}=m}\mathbb{E}_{q_{0|t}(\bm {x}_{0}|\bm{x}_{t})}\left[\bm{e}_{x_{0}^{(l)}}\right]^{\top}\log\bm{\mu}_{ \theta}^{(l)}(\bm{x}_{t},t)\right]\mathrm{d}t$ $$

The objective is an aggregation of cross-entropy terms over different $t,\bm{x}_{t},l$. The global minimum is achieved when each cross-entropy term is optimal:

$$ $\min_{\theta}-\mathbb{E}_{q_{0|t}(\bm{x}_{0}|\bm{x}_{t})}\left[\bm{e}_{x_{0}^{ (l)}}\right]^{\top}\log\bm{\mu}_{\theta}^{(l)}(\bm{x}_{t},t)$ (19) $$

Note that $\bm{\mu}_{\theta}^{(l)}=\mathrm{softmax}(\bm{f}_{\theta}^{(l)})$ is a set of valid class probabilities that sum to 1, and $\mathbb{E}_{q_{0|t}(\bm{x}_{0}|\bm{x}_{t})}\left[\bm{e}_{x_{0}^{(l)}}\right]$ also sum to 1. Denote them as $\bm{P}$ and $\hat{\bm{P}}$ respectively, we are essentially minimizing $-\bm{P}\log\hat{\bm{P}}=D_{\mathrm{KL}}(\bm{P}\;\|\;\hat{\bm{P}})+H(\bm{P})
\geq H(\bm{P})$, where $D_{\mathrm{KL}}(\cdot\;\|\;\cdot)$ is the Kullback–Leibler (KL) divergence and $H(\cdot)$ is the entropy. According to the property of KL divergence, the equality holds if and only if $\hat{\bm{P}}=\bm{P}$. This implies that the optimal $\theta^{*}$ satisfies

$$ $\bm{\mu}_{\theta^{*}}^{(l)}(\bm{x}_{t},t)=\mathbb{E}_{q_{0|t}(\bm{x}_{0}|\bm{x }_{t})}\left[\bm{e}_{x_{0}^{(l)}}\right]$ (20) $$

This expression is similar to continuous diffusion, where the optimal data predictor is $\bm{\mu}_{\theta^{*}}(\bm{x}_{t},t)=\mathbb{E}_{q_{0|t}(\bm{x}_{0}|\bm{x}_{t})
}\left[\bm{x}_{0}\right]$. The key difference is that, the posterior $q_{0|t}(\bm{x}_{0}|\bm{x}_{t})$ in MDMs only depends on $\bm{x}_{t}$ and is irrelevant to the time $t$. Denote $n_{t}$ as the number of masked tokens at time $t$, we have

$$ $q_{0|t}(\bm{x}_{0}|\bm{x}_{t})=\frac{q_{0}(\bm{x}_{0})q_{t|0}(\bm{x}_{t}|\bm{x }_{0})}{q_{t}(\bm{x}_{t})}=\frac{q_{0}(\bm{x}_{0})q_{t|0}(\bm{x}_{t}|\bm{x}_{0 })}{\sum_{\bm{x}_{0}}q_{0}(\bm{x}_{0})q_{t|0}(\bm{x}_{t}|\bm{x}_{0})}=\frac{q_ {0}(\bm{x}_{0})\mathbb{E}_{p_{t}(n_{t})}[\tilde{q}_{n_{t}|0}(\bm{x}_{t}|\bm{x} _{0})]}{\sum_{\bm{x}_{0}}q_{0}(\bm{x}_{0})\mathbb{E}_{p_{t}(n_{t})}[\tilde{q}_ {n_{t}|0}(\bm{x}_{t}|\bm{x}_{0})]}$ (21) $$

where $p_{t}(n_{t})$ is distribution of $n_{t}$ at time $t$, and $\tilde{q}_{n_{t}|0}$ is the discrete forward process that randomly masks $n_{t}$ tokens. As $\bm{x}_{t}$ is known, the number of masked tokens is fixed as $N(\bm{x}_{t})$, and $\tilde{q}_{n_{t}|0}(\bm{x}_{t}|\bm{x}_{0})=0$ for $n_{t}\neq N(\bm{x}_{t})$. Therefore,

$$ $q_{0|t}(\bm{x}_{0}|\bm{x}_{t})=\frac{q_{0}(\bm{x}_{0})p_{t}(N(\bm{x}_{t})) \tilde{q}_{N(\bm{x}_{t})|0}(\bm{x}_{t}|\bm{x}_{0})}{\sum_{\bm{x}_{0}}q_{0}(\bm {x}_{0})p_{t}(N(\bm{x}_{t}))\tilde{q}_{N(\bm{x}_{t})|0}(\bm{x}_{t}|\bm{x}_{0}) }=\tilde{q}_{0|N(\bm{x}_{t})}(\bm{x}_{0}|\bm{x}_{t})\\$ (22) $$

which completes the proof.
∎

###### Proof.

### C.3 Proof of Proposition 4.1

We first present a lemma that enables sequential sampling of order statistics and supports the recursive process for sampling the first hitting time.

###### Lemma C.1 (Uniform Distribution Conditioned on the Maximum) .

Suppose $n$ random variables $u_{1},u_{2},\dots,u_{n}$ are independent samples from the uniform distribution $\mathcal{U}(0,\theta)$ ($\theta>0$). Given the condition that $u=\max\{u_{1},\cdots,u_{n}\}$, the remaining variables $u_{i}$ ($u_{i}\neq u$) follow the distribution $\mathcal{U}(0,u)$.

###### Proof.

Without loss of generality, we derive the conditional distribution of $u_{1}$. Other remaining variables follow the same distribution due to symmetry. For $x\leq y\leq\theta$, we have

$$ $P(u_{1}\leq x,u\leq y)=P(u_{1}\leq x,u_{2}\leq y,\dots u_{n}\leq y)=P(u_{1} \leq x)\prod_{i=2}^{n}P(u_{i}\leq y)=\frac{xy^{n-1}}{\theta^{n}}$ (23) $$

$$ $P(u_{1}\leq x,u\leq y|u_{1}=u)=P(u\leq x)=P(u_{1}\leq x,\dots,u_{n}\leq x)= \prod_{i=1}^{n}P(u_{i}\leq x)=\frac{x^{n}}{\theta^{n}}$ (24) $$

and

$$ $\quad P(u_{1}=u)=\frac{1}{n},\quad\quad P(u_{1}\neq u)=\frac{n-1}{n}$ (25) $$

Therefore,

$$ $\displaystyle P(u_{1}\leq x,u\leq y|u_{1}\neq u)$ $\displaystyle=\frac{P(u_{1}\leq x,u\leq y)-P(u_{1}=u)P(u_{1}\leq x,u\leq y|u_{ 1}=u)}{P(u_{1}\neq u)}$ (26) $\displaystyle=\frac{n}{n-1}\frac{xy^{n-1}}{\theta^{n}}-\frac{1}{n-1}\frac{x^{n }}{\theta^{n}}$ $$

By taking derivatives w.r.t. $x$ and $y$, we obtain the density $p(u_{1}=x,u=y|u_{1}\neq u)=\frac{ny^{y-2}}{\theta^{n}}$. Similarly, $P(u\leq y)=\frac{y^{n}}{\theta^{n}}$, and the density $p(u=y)=\frac{ny^{n-1}}{\theta^{n}}$. Therefore

$$ $p(u_{1}=x|u=y,u_{1}\neq u)=\frac{p(u_{1}=x,u=y|u_{1}\neq u)}{p(u=y)}=\frac{1}{y}$ (27) $$

We conclude that $u_{1}$ ($u_{1}\neq u$) follows a uniform distribution over the interval $[0,u]$.
∎

Then we prove Proposition 4.1 below.

###### Proof.

We first consider the case of a single token undergoing the reverse process described in Eqn. (10). Starting from time $t$, when $x_{t}=m$ is the mask token, we denote $\tau$ as the time at which the unmasking transition occurs (i.e., $x_{\tau+\mathrm{d}t}=m$ and $x_{\tau}\neq m$). The transition time $\tau$ is a random variable, whose cumulative distribution function (CDF) is available:

$$ $P(\tau\leq s)=p_{\theta}(x_{s}=m|x_{t}=m)=\frac{1-\alpha_{s}}{1-\alpha_{t}}$ (28) $$

Therefore, using inverse transform sampling, $\tau$ can be analytically sampled by (1) drawing $u\sim\mathcal{U}(0,1)$, and (2) solving the equation $\frac{1-\alpha_{\tau}}{1-\alpha_{t}}=u$.

Next, we consider the case of multiple tokens in a sequence of length $L$. Thanks to the theoretical assumptions of MDMs, the transition times of different tokens are independent in the reverse process. However, to enable token-by-token decoding, we need to sample the $L$ transition times in descending order, i.e., $1>\tau_{L-1}>\dots>\tau_{0}$. Starting from $t=1$ with $\alpha_{1}=0$, each transition time $\tau$ can be sampled by drawing $u\sim\mathcal{U}(0,1)$ and solving $1-\alpha_{\tau}=u$ according to the single token case. In order to sample $\tau$ sequentially, we are essentially drawing the order statistics $u_{(L-1)}>\dots>u_{(0)}$ of $L$ independent uniform variables on $[0,1]$.

According to Lemma C.1, this process can be conducted in a recursive manner without sorting. Suppose there are currently $n$ remaining masked tokens, and the most recent unmasking occurred at time $\tau_{n}$. The transition time $\tau_{n}$ corresponds to the $n$-th smallest uniform variable $u_{(n)}$ through the relation $1-\alpha_{\tau_{n}}=u_{(n)}$. To obtain the next transition time $\tau_{n-1}$, we need to sample the next order statistic $u_{(n-1)}$. By recursively applying Lemma C.1, we know that the remaining $n$ smallest uniform variables follow the distribution $\mathcal{U}(0,u_{(n)})$, if not considering their relative order. Furthermore, $u_{(n-1)}$, as the maximum of these $n$ variables, has the CDF $P(u_{(n-1)}\leq x)=\frac{x^{n}}{u_{(n)}^{n}}$ and can be sampled by solving $\frac{u_{(n-1)}^{n}}{u_{(n)}^{n}}=u_{n}$, where $u_{n}\sim\mathcal{U}(0,1)$ (using inverse transform sampling). Therefore, the next transition time $\tau_{n-1}$ satisfies

$$ $1-\alpha_{\tau_{n-1}}=u_{(n-1)}=u_{(n)}u_{n}^{\frac{1}{n}}=(1-\alpha_{\tau_{n} })u_{n}^{\frac{1}{n}}$ (29) $$

which is equivalent to Eqn. (10) using the inverse noise schedule function $\tau_{n-1}=\alpha^{-1}(\alpha_{\tau_{n-1}})$.
∎

###### Lemma C.1 (Uniform Distribution Conditioned on the Maximum) .

###### Proof.

###### Proof.

### C.4 Proof of Proposition 5.2

###### Proof.

The truncated standard Gumbel distribution $\mathcal{T}\mathcal{G}(0,1,M)$ has the probability density function (PDF) and cumulative distribution function (CDF) defined as follows:

$$ $\hat{f}(x)=\frac{f(x)}{F(M)}\mathbb{I}_{x\leq M},\quad\hat{F}(x)=\min\left\{ \frac{F(x)}{F(M)},1\right\}$ (30) $$

where

$$ $f(x)=e^{-x-e^{-x}},\quad F(x)=e^{-e^{-x}}$ (31) $$

are the PDF and CDF of the standard Gumbel distribution $\mathcal{G}(0,1)$, and $M$ is the right truncation point. Suppose the class probabilities are sorted as $\pi_{1}\leq\dots\leq\pi_{K}$, and denote $\pi_{0}=0,\theta_{n}=\log\pi_{n}$ for simplicity. To conduct truncated Gumbel-based categorical sampling, $K$ i.i.d. samples $\{g_{i}\}_{i=1}^{K}$ are drawn from $\mathcal{T}\mathcal{G}(0,1,M)$. The resulting categorical probability of class $n$ is

$$ $\displaystyle P(\operatornamewithlimits{argmax}(\theta_{i}+g_{i})=n)$ $\displaystyle=\int_{-\infty}^{+\infty}\hat{f}(g)\prod_{k\neq n}P(\theta_{k}+g_ {k}\leq\theta_{n}+g)\mathrm{d}g$ (32) $\displaystyle=\int_{-\infty}^{M}\hat{f}(g)\prod_{k\neq n}\hat{F}(\theta_{n}+g- \theta_{k})\mathrm{d}g$ $\displaystyle=e^{Ke^{-M}}\int_{-\infty}^{M}e^{-g-e^{-g}}\prod_{k\neq n}\min\{e ^{-e^{-M}},e^{-e^{-\theta_{n}-g+\theta_{k}}}\}\mathrm{d}g$ $\displaystyle=e^{Ke^{-M}}\int_{-\infty}^{M}e^{-g-e^{-g}}e^{-\sum_{k\neq n}e^{- \min\{g+\theta_{n}-\theta_{k},M\}}}\mathrm{d}g$ $\displaystyle=e^{Ke^{-M}}\sum_{i=1}^{n}\int_{\theta_{i-1}+M-\theta_{n}}^{ \theta_{i}+M-\theta_{n}}e^{-g}e^{-(\sum_{k=i}^{K}e^{\theta_{k}-\theta_{n}})e^{ -g}}e^{-(i-1)e^{-M}}\mathrm{d}g$ $$

where the integral has a closed-form solution by

$$ $\int e^{-g}e^{-Ae^{-g}}\mathrm{d}g=\frac{e^{-Ae^{-g}}}{A}+C$ (33) $$

With this, Eqn. (32) can be further simplified to

$$ $\displaystyle P(\operatornamewithlimits{argmax}(\theta_{i}+g_{i})=n)$ (34) $\displaystyle=$ $\displaystyle e^{Ke^{-M}}\sum_{i=1}^{n}\frac{e^{-(\sum_{k=i}^{K}e^{\theta_{k}- \theta_{n}})e^{\theta_{n}-\theta_{i}-M}}-e^{-(\sum_{k=i}^{K}e^{\theta_{k}- \theta_{n}})e^{\theta_{n}-\theta_{i-1}-M}}}{\sum_{k=i}^{K}e^{\theta_{k}-\theta _{n}}}e^{-(i-1)e^{-M}}$ $\displaystyle=$ $\displaystyle e^{Ke^{-M}}\pi_{n}\sum_{i=1}^{n}\frac{e^{-\frac{\sum_{k=i}^{K} \pi_{k}}{\pi_{i}}e^{-M}}-e^{\frac{-\sum_{k=i}^{K}\pi_{k}}{\pi_{i-1}}e^{-M}}}{ \sum_{k=i}^{K}\pi_{k}}e^{-(i-1)e^{-M}}$ $\displaystyle=$ $\displaystyle\pi_{n}\sum_{i=1}^{n}\frac{e^{\left(K+1-i-\frac{\sum_{k=i}^{K}\pi _{k}}{\pi_{i}}\right)e^{-M}}-e^{\left(K+1-i-\frac{\sum_{k=i}^{K}\pi_{k}}{\pi_{ i-1}}\right)e^{-M}}}{\sum_{k=i}^{K}\pi_{k}}$ $$

Therefore, the original class probabilities $\{\pi_{n}\}_{n=1}^{K}$ are shifted to $\{\pi_{n}^{\prime}\}_{n=1}^{K}$ if the Gumbel variables used in categorical sampling are right-truncated to $M$. $\pi_{n}^{\prime}$ is given by

$$ $\displaystyle\pi_{n}^{\prime}$ $\displaystyle=\pi_{n}\sum_{i=1}^{n}\frac{e^{\left(K+1-i-\frac{\sum_{k=i}^{K} \pi_{k}}{\pi_{i}}\right)e^{-M}}-e^{\left(K+1-i-\frac{\sum_{k=i}^{K}\pi_{k}}{ \pi_{i-1}}\right)e^{-M}}}{\sum_{k=i}^{K}\pi_{k}}$ (35) $\displaystyle=\pi_{n}\sum_{i=1}^{n}\frac{e^{\left(K-i-\frac{\sum_{k=i+1}^{K} \pi_{k}}{\pi_{i}}\right)e^{-M}}-e^{\left(K-(i-1)-\frac{\sum_{k=i}^{K}\pi_{k}}{ \pi_{i-1}}\right)e^{-M}}}{\sum_{k=i}^{K}\pi_{k}}$ $$

We can verify that $\{\pi_{n}^{\prime}\}_{n=1}^{K}$ are valid class probabilities that sum to 1:

$$ $\displaystyle\sum_{n=1}^{K}\pi_{n}^{\prime}$ $\displaystyle=\sum_{n=1}^{K}\pi_{n}\sum_{i=1}^{n}\frac{e^{\left(K-i-\frac{\sum _{k=i+1}^{K}\pi_{k}}{\pi_{i}}\right)e^{-M}}-e^{\left(K-(i-1)-\frac{\sum_{k=i}^ {K}\pi_{k}}{\pi_{i-1}}\right)e^{-M}}}{\sum_{k=i}^{K}\pi_{k}}$ (36) $\displaystyle=\sum_{i=1}^{K}\sum_{n=i}^{K}\pi_{n}\frac{e^{\left(K-i-\frac{\sum _{k=i+1}^{K}\pi_{k}}{\pi_{i}}\right)e^{-M}}-e^{\left(K-(i-1)-\frac{\sum_{k=i}^ {K}\pi_{k}}{\pi_{i-1}}\right)e^{-M}}}{\sum_{k=i}^{K}\pi_{k}}$ $\displaystyle=\sum_{i=1}^{K}e^{\left(K-i-\frac{\sum_{k=i+1}^{K}\pi_{k}}{\pi_{i }}\right)e^{-M}}-e^{\left(K-(i-1)-\frac{\sum_{k=i}^{K}\pi_{k}}{\pi_{i-1}} \right)e^{-M}}$ $\displaystyle=e^{(K-K)e^{-M}}-e^{\left(K-\frac{1}{\pi_{0}}\right)e^{-M}}=1$ $$

where $\pi_{0}=0$ and $e^{\left(K-\frac{1}{\pi_{0}}\right)e^{-M}}=0$.
∎

###### Proof.

## Appendix D Relationship between Masked Diffusion Models and Previous Discrete Diffusion Models

Our framework and notations are based on recent studies of MDMs , which offer a theoretically simplified and empirically improved version of the best-performing absorbing case in discrete diffusion models . In this section, we present a summary of some background information on previous formulations: generative modeling of discrete data via continuous-time Markov chains (Section D.1), robust and principled training and sampling with score parameterization (Section D.2), and their equivalence to MDMs (Section D.3).

### D.1 Discrete Diffusion via Continuous-Time Markov Chains

Continuous-time Markov chains (CTMCs)  are a fundamental concept in stochastic processes used to model systems that transition between discrete states continuously over time.

##### Forward Process

Denote $\mathcal{X}$ as the state space and $x\in\mathcal{X}$ as a state. The probability of transitioning from one state $x$ to another state $\hat{x}$ near time $t$ is governed by the transition rate matrix $\bm{Q}_{t}\in\mathbb{R}^{|\mathcal{X}|\times|\mathcal{X}|}$. Specifically, denote $Q_{t}(x,\hat{x})$ as the transition rate from $x$ to $\hat{x}$, the transition probability during a small time interval $\Delta t$ is

$$ $p_{t+\Delta t|t}(\hat{x}|x)=\delta_{x,\hat{x}}+Q_{t}(x,\hat{x})\Delta t+ \mathcal{O}((\Delta t)^{2})$ (37) $$

The off-diagonal elements $Q_{t}(x,\hat{x})$ ($x\neq\hat{x}$) are non-negative, and the diagonal elements $Q_{t}(x,x)=-\sum_{\hat{x}\neq x}Q_{t}(x,\hat{x})\leq 0$, ensuring that each row of $\bm{Q}_{t}$ sums to zero (so that $p_{t}$ does not gain or lose total mass). Equivalently, the transition rate can be defined by the transition probability as

$$ $Q_{t}(x,\hat{x})=\lim_{\Delta t\rightarrow 0}\frac{p_{t+\Delta t|t}(\hat{x}|x) -\delta_{x,\hat{x}}}{\Delta t}$ (38) $$

Denote $\bm{p}_{t}=\{p_{t}(x)\}_{x\in\mathcal{X}}$ as the marginal distributions of all states at time $t$, and $\bm{P}_{t|s}\in\mathbb{R}^{|\mathcal{X}|\times|\mathcal{X}|}$ as the forward transition matrix from time $s$ to time $t$ satisfying $P_{t|s}(x,\hat{x})=p_{t|s}(\hat{x}|x)$. The Kolmogorov forward (or Fokker-Planck) equations describe the evolution of both the marginals $\bm{p}_{t}$ (starting from the data distribution) and the transition matrix $\bm{P}_{t|s}$:

$$ $\frac{\mathrm{d}\bm{p}_{t}}{\mathrm{d}t}=\bm{p}_{t}\bm{Q}_{t},\quad\frac{ \mathrm{d}\bm{P}_{t|s}}{\mathrm{d}t}=\bm{P}_{t|s}\bm{Q}_{t}$ (39) $$

In practice, the forward process is designed to be simple degradation , such that $p_{t}$ approaches a stationary distribution $p_{\rm base}$ that is easy to sample from as $t$ increases, akin to the Gaussian noising process in continuous diffusion. Specifically, the transition rate matrix $\bm{Q}_{t}$ is set to $\sigma(t)\bm{Q}$ where $\sigma(t)$ is a scalar noise schedule function and $\bm{Q}$ is a constant matrix with low ranks. In this case, the transition matrix can be solved analytically as $\bm{P}_{t|s}=e^{(\bar{\sigma}(t)-\bar{\sigma}(s))\bm{Q}}$ where $\bar{\sigma}(t)=\int_{0}^{t}\sigma(\tau)\mathrm{d}\tau$. Common choices of $\bm{Q}$  include:

$$ $\bm{Q}_{\rm uniform}=\begin{bmatrix}1-N&1&\cdots&1\\ 1&1-N&\cdots&1\\ \vdots&\vdots&\ddots&\vdots\\ 1&1&\cdots&1-N\end{bmatrix},\quad\bm{Q}_{\rm absorb}=\begin{bmatrix}-1&0& \cdots&0&1\\ 0&-1&\cdots&0&1\\ \vdots&\vdots&\ddots&\vdots&\vdots\\ 0&0&\cdots&-1&1\\ 0&0&\cdots&0&0\end{bmatrix}$ (40) $$

The former disturbs the data distribution into a uniform one, and the latter additionally adds a [MASK] token as the absorbing state.

##### Time Reversal

Similar to continuous diffusion, discrete diffusion defined above has a time reversal  described by the reverse transition rate matrix $\bar{\bm{Q}}_{t}$ which satisfies

$$ $\bar{Q}_{t}(x,\hat{x})=\begin{cases}\displaystyle\frac{p_{t}(\hat{x})}{p_{t}(x )}Q_{t}(\hat{x},x),\quad&\hat{x}\neq x\\ -\sum_{y\neq x}\bar{Q}_{t}(x,y),\quad&\hat{x}=x\end{cases}$ (41) $$

The intractable ratio $\frac{p_{t}(\hat{x})}{p_{t}(x)}$, named concrete score , acts as an analog to the score function  in continuous diffusion. The reverse process can be described as

$$ $\frac{\mathrm{d}\bm{p}_{s}}{\mathrm{d}s}=-\bm{p}_{s}\bar{\bm{Q}}_{s},\quad \frac{\mathrm{d}\bm{P}_{s|t}}{\mathrm{d}s}=-\bm{P}_{s|t}\bar{\bm{Q}}_{s}$ (42) $$

which evolves backward in time with $s$ decreasing to 0 and $s<t$. It is sufficient to simulate the whole process as long as the concrete score, the
only unknown term in $\bar{\bm{Q}}_{t}$, is estimated.

### D.2 Score-Entropy Discrete Diffusion (SEDD)

SEDD  provides principled, robust and scalable techniques for score-based training and sampling of discrete diffusion models.

##### Parameterization

SEDD parameterizes a score prediction network $\bm{s}_{\theta}(x,t)\in\mathbb{R}^{|\mathcal{X}|}$ to learn the unknown concrete score $\left\{\frac{p_{t}(\hat{x})}{p_{t}(x)}\right\}_{\hat{x}\in\mathcal{X}}$. We use $s_{\theta}(x,t)_{y}$ to denote its $y$-th element.

##### Training Objective

SEDD proposes the diffusion-weighted denoising score entropy (DWDSE) objective to optimize $\bm{s}_{\theta}(x,t)$:

$$ $\mathcal{L}_{\rm DWDSE}(x_{0})=\int_{0}^{T}\mathbb{E}_{x_{t}\sim p_{t|0}\left( \cdot|x_{0}\right)}\sum_{{\hat{x}_{t}}\neq x_{t}}Q_{t}\left(\hat{x}_{t},x_{t} \right)I\left(s_{\theta}\left(x_{t},t\right)_{\hat{x}_{t}},\frac{p_{t\mid 0} \left({\hat{x}_{t}}\mid x_{0}\right)}{p_{t\mid 0}\left(x_{t}\mid x_{0}\right)} \right)\mathrm{d}t$ (43) $$

where $I(a,b)\coloneqq a-b\log a+K(b)$, and $K(b)\coloneqq b\log b-b$ is a normalizing constant function that ensures $I(a,b)\geq 0$. Eqn. (43) not only admits the optimal solution as the concrete score, but also serves as a NELBO for discrete diffusion models by $-\log p_{0}^{\theta}(\bm{x}_{0})\leq\mathcal{L}_{\rm DWDSE}(x_{0})+D_{\mathrm{
KL}}(p_{T|0}(\cdot|x_{0})\;\|\;p_{\rm base})$, where $p_{\rm base}$ is the stationary distribution when $T\rightarrow\infty$.

##### Sampling Procedures

Denote $\bm{s}_{t}(x)=\left\{\frac{p_{t}(\hat{x})}{p_{t}(x)}\right\}_{\hat{x}\in
\mathcal{X}}$ as the ground-truth concrete score, and $s_{t}(x)_{y}$ as its $y$-th element. We use $\bm{s}_{t}(x)$ to demonstrate the sampling process, while in practice it is replaced with the learned score $\bm{s}_{\theta}(x,t)$. SEDD offers two sampling procedures: Euler sampling and analytic sampling with Tweedie $\tau$-Leaping.

Euler sampling applies the Euler discretization to the reverse process (Eqn. (42)), producing a reverse transition similar to the forward transition in Eqn. (37):

$$ $p_{s|t}^{\rm Euler}(x_{s}|x_{t})=\delta_{x_{t},x_{s}}+\bar{Q}_{t}(x_{t},x_{s}) (t-s)$ (44) $$

It can be expressed by the concrete score $\bm{s}_{t}$ as

$$ $p_{s|t}^{\rm Euler}(x_{s}|x_{t})=\begin{cases}(t-s)Q_{t}(x_{s},x_{t})s_{t}(x_{ t})_{x_{s}},\quad&x_{s}\neq x_{t}\\ 1-\sum_{y\neq x_{t}}p_{s|t}^{\rm Euler}(y|x_{t}),\quad&x_{s}=x_{t}\end{cases}$ (45) $$

The Euler sampling implicitly assumes a constant reverse rate matrix $\bar{\bm{Q}}_{\tau}=\bar{\bm{Q}}_{t}$ for $\tau\in[s,t]$, producing approximation errors and even resulting in negative probabilities at $x_{s}=x_{t}$.

Tweedie $\tau$-leaping operates similarly to the posterior sampling in DDPM , by analytically solving the posterior $p_{s|t}(x_{s}|x_{t})$ given the ground-truth concrete score. Specifically,

$$ $p_{s|t}^{\rm Tweedie}(x_{s}|x_{t})=\frac{p_{t|s}(x_{t}|x_{s})p_{s}(x_{s})}{p_{ t}(x_{t})}$ (46) $$

Under the special choice $\bm{Q}_{t}=\sigma(t)\bm{Q}$
described in the previous section, we have

$$ $p_{t|s}(x_{t}|x_{s})=\left(\bm{P}_{t|s}\right)_{x_{s},x_{t}}=\left(e^{(\bar{ \sigma}(t)-\bar{\sigma}(s))\bm{Q}}\right)_{x_{s},x_{t}}$ (47) $$

and

$$ $p_{s}(x_{s})=\left(\bm{p}_{s}\right)_{x_{s}}=\left(\bm{p}_{t}\bm{P}_{t|s}^{-1} \right)_{x_{s}}=\left(\bm{p}_{t}e^{-(\bar{\sigma}(t)-\bar{\sigma}(s))\bm{Q}} \right)_{x_{s}}$ (48) $$

Therefore,

$$ $\displaystyle p_{s|t}^{\rm Tweedie}(x_{s}|x_{t})$ $\displaystyle=\left(e^{(\bar{\sigma}(t)-\bar{\sigma}(s))\bm{Q}}\right)_{x_{s}, x_{t}}\left(\frac{\bm{p}_{t}}{p_{t}(x_{t})}e^{-(\bar{\sigma}(t)-\bar{\sigma}(s ))\bm{Q}}\right)_{x_{s}}$ (49) $\displaystyle=\left(e^{(\bar{\sigma}(t)-\bar{\sigma}(s))\bm{Q}}\right)_{x_{s}, x_{t}}\left(\bm{s}_{t}(x_{t})e^{-(\bar{\sigma}(t)-\bar{\sigma}(s))\bm{Q}} \right)_{x_{s}}$ $$

##### Multi-Dimensional Case

For a token sequence $\bm{x}\in\mathcal{X}^{L}$ of length $L$, we use $-l$ to denote the indexes of all tokens except the $l$-th one. The concrete score $\bm{s}_{t}(\bm{x})\in\mathbb{R}^{|\mathcal{X}|\times L}$ is defined between sequences that differ by a Hamming distance of 1:

$$ $s_{t}(\bm{x})_{\hat{x},l}=\frac{p_{t}(\hat{\bm{x}})}{p_{t}(\bm{x})},\quad\text {s.t.}\ \hat{x}^{(l)}=\hat{x},\hat{\bm{x}}^{(-l)}=\bm{x}^{(-l)}$ (50) $$

The forward and reverse processes are factorized across dimensions in the same manner as in MDMs described in the main text. The parameterized score network $\bm{s}_{\theta}(\bm{x},t)\in\mathbb{R}^{|\mathcal{X}|\times L}$ also predicts the scores at all positions at a time. Consequently, both the training and sampling are conducted simultaneously and independently for all dimensions, except that the network input $\bm{x}$ contains the current sequence information.

### D.3 Connection between MDMs and SEDD Absorb

The absorbing case ($\bm{Q}=\bm{Q}_{\rm absorb}$) of discrete diffusion has demonstrated both simple formulations and superior performance. As revealed in previous and concurrent works , SEDD Absorb is theoretically equivalent to MDMs in multiple aspects. For simplicity, we focus on the single-token case, as the factorization approach used in both MDMs and SEDD Absorb ensures that equivalence in one dimension implies equivalence in multiple dimensions.

#### D.3.1 Equivalence of Training

##### Relation between Forward Processes

The forward transition matrix of SEDD Absorb is $\bm{P}_{t|0}=e^{\bar{\sigma}(t)\bm{Q}_{\rm absorb}}$. It can be verified by mathematical induction that $\bm{Q}_{\rm absorb}^{n}=(-1)^{n-1}\bm{Q}_{\rm absorb}$ for any positive integer $n$. With this identity, $\bm{P}_{t|0}$ can be simplified as

$$ $\displaystyle\bm{P}_{t|0}=e^{\bar{\sigma}(t)\bm{Q}_{\rm absorb}}=\bm{I}+\sum_{ k=1}^{\infty}\frac{\bar{\sigma}(t)^{k}\bm{Q}^{k}_{\rm absorb}}{k!}=\bm{I}-\sum _{k=1}^{\infty}\frac{(-\bar{\sigma}(t))^{k}}{k!}\bm{Q}_{\rm absorb}$ (51) $\displaystyle=$ $\displaystyle\bm{I}+\left(1-e^{-\bar{\sigma}(t)}\right)\bm{Q}_{\rm absorb}$ $$

This is equivalent to the forward process (Eqn. (1)) in MDMs with the relation $\alpha_{t}=e^{-\bar{\sigma}(t)}$.

##### Relation between Parameterizations

For coherence, we use $m$ to denote the [MASK] token. We only need to consider the concrete score $\bm{s}_{t}(m)$ at $m$, since $\bm{s}_{t}(x)$ for $x\neq m$ can be converted from $\bm{s}_{t}(m)$ by $s_{t}(x)_{\hat{x}}=\frac{s_{t}(m)_{\hat{x}}}{s_{t}(m)_{x}}$. We have

$$ $s_{t}(m)_{x}=\frac{p_{t}(x)}{p_{t}(m)}=\frac{\left(\bm{p}_{0}\bm{P}_{t|0} \right)_{x}}{\left(\bm{p}_{0}\bm{P}_{t|0}\right)_{m}}$ (52) $$

Substituting the expression of $\bm{P}_{t|0}$ in Eqn. (51) into Eqn. (52), for $x\neq m$, we have

$$ $\left(\bm{p}_{0}\bm{P}_{t|0}\right)_{m}=\sum_{x_{0}\in\mathcal{X}\backslash\{m \}}p_{0}(x_{0})p_{t|0}(m|x_{0})=(1-\alpha_{t})\sum_{x_{0}\in\mathcal{X} \backslash\{m\}}p_{0}(x_{0})=1-\alpha_{t}$ (53) $$

$$ $\left(\bm{p}_{0}\bm{P}_{t|0}\right)_{x}=\sum_{x_{0}\in\mathcal{X}\backslash\{m \}}p_{0}(x_{0})p_{t|0}(x|x_{0})=\sum_{x_{0}\in\mathcal{X}\backslash\{m\}}p_{0} (x_{0})\alpha_{t}\delta_{x_{0},x}=\alpha_{t}p_{0}(x)$ (54) $$

and

$$ $s_{t}(m)_{x}=\frac{\left(\bm{p}_{0}\bm{P}_{t|0}\right)_{x}}{\left(\bm{p}_{0} \bm{P}_{t|0}\right)_{m}}=\frac{\alpha_{t}}{1-\alpha_{t}}p_{0}(x)=\frac{\alpha_ {t}}{1-\alpha_{t}}\left(\mathbb{E}_{p_{0|t}(x_{0}|m)}\left[\bm{e}_{x_{0}} \right]\right)_{x}$ (55) $$

This implies that the score parameterization is related to the mean parameterization $\bm{\mu}_{\theta}(x,t)$ in MDMs (excluding the $m$-th dimension) by

$$ $\bm{s}_{\theta}(x,t)=\frac{\alpha_{t}}{1-\alpha_{t}}\bm{\mu}_{\theta}(x,t)$ (56) $$

##### Equivalence of ELBOs

Substituting this relation between $\bm{s}_{\theta}$ and $\bm{\mu}_{\theta}$ into the score entropy objective of SEDD (Eqn. 43), we have

$$ $\displaystyle\mathcal{L}_{\rm DWDSE}(x_{0})$ $\displaystyle=\int_{0}^{T}\mathbb{E}_{x_{t}\sim p_{t|0}\left(\cdot|x_{0}\right )}\left[\delta_{x_{t},m}\sum_{x\neq m}Q_{t}\left(x,m\right)I\left(s_{\theta} \left(m,t\right)_{x},\frac{p_{t\mid 0}\left(x\mid x_{0}\right)}{p_{t\mid 0} \left(m\mid x_{0}\right)}\right)\right]\mathrm{d}t$ (57) $\displaystyle=\int_{0}^{T}\sigma(t)\mathbb{E}_{x_{t}\sim p_{t|0}\left(\cdot|x_ {0}\right)}\left[\delta_{x_{t},m}\sum_{x\neq m}I\left(\frac{\alpha_{t}}{1- \alpha_{t}}\mu_{\theta}\left(m,t\right)_{x},\delta_{x,x_{0}}\frac{\alpha_{t}}{ 1-\alpha_{t}}\right)\right]\mathrm{d}t$ $$

Observing that $I(a,0)=a$, $\sum_{x\neq m}\mu_{\theta}(m,t)_{x}=1$, we have

$$ $\displaystyle\sum_{x\neq m}I\left(\frac{\alpha_{t}}{1-\alpha_{t}}\mu_{\theta} \left(m,t\right)_{x},\delta_{x,x_{0}}\frac{\alpha_{t}}{1-\alpha_{t}}\right)$ (58) $\displaystyle=$ $\displaystyle K\left(\frac{\alpha_{t}}{1-\alpha_{t}}\right)-\frac{\alpha_{t}}{ 1-\alpha_{t}}\log\left(\frac{\alpha_{t}}{1-\alpha_{t}}\mu_{\theta}(m,t)_{x_{0} }\right)+\frac{\alpha_{t}}{1-\alpha_{t}}\sum_{x\neq m}\mu_{\theta}(m,t)_{x}$ $\displaystyle=$ $\displaystyle-\frac{\alpha_{t}}{1-\alpha_{t}}\log\mu_{\theta}(m,t)_{x_{0}}$ $$

Using $\alpha_{t}^{\prime}=-\sigma(t)\alpha_{t}$,
the objective $\mathcal{L}_{\rm DWDSE}(x_{0})$ can be simplified to

$$ $\displaystyle\mathcal{L}_{\rm DWDSE}(x_{0})$ $\displaystyle=-\int_{0}^{T}\sigma(t)\mathbb{E}_{x_{t}\sim p_{t|0}\left(\cdot|x _{0}\right)}\left[\delta_{x_{t},m}\frac{\alpha_{t}}{1-\alpha_{t}}\log\mu_{ \theta}(m,t)_{x_{0}}\right]\mathrm{d}t$ (59) $\displaystyle=\int_{0}^{T}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}\mathbb{E}_{ x_{t}\sim p_{t|0}\left(\cdot|x_{0}\right)}\left[\delta_{x_{t},m}\log\mu_{ \theta}(x_{t},t)_{x_{0}}\right]\mathrm{d}t$ $$

As $\log\mu_{\theta}(x_{t},t)_{x_{0}}$
and $\bm{e}_{x_{0}}^{\top}\log\bm{\mu}_{\theta}(x_{t},t)$ are equivalent expressions for the cross entropy, we conclude that $\mathcal{L}_{\rm DWDSE}(x_{0})$ is equal to the NELBO for MDMs (Eqn. (5)) when $T=1$.

#### D.3.2 Equivalence of Sampling

Euler Sampler and Tweedie $\tau$-Leaping Sampler in SEDD are equivalent in the absorbing case under the linear noise schedule $\alpha_{t}=e^{-\bar{\sigma}(t)}=1-t$.

On the one hand, the Euler sampler (Eqn. (45)) with score parameterization network $\bm{s}_{\theta}$ is

$$ $p_{s|t}^{\rm Euler}(x_{s}|x_{t})=\begin{cases}(t-s)\sigma(t)\left(\bm{Q}_{\rm absorb }\right)_{x_{s},x_{t}}s_{\theta}(x_{t},t)_{x_{s}},\quad&x_{s}\neq x_{t}\\ 1-\sum_{y\neq x_{t}}p_{s|t}^{\rm Euler}(y|x_{t}),\quad&x_{s}=x_{t}\end{cases}$ (60) $$

When $x_{s}\neq x_{t}$, using the identities $s_{\theta}(x,t)_{x}=1$ and $\left(\bm{Q}_{\rm absorb}\right)_{x_{s},x_{t}}=\delta_{m,x_{t}}-\delta_{x_{s},
x_{t}}$, $p_{s|t}^{\rm Euler}$ can be simplified to

$$ $p_{s|t}^{\rm Euler}(x_{s}|x_{t})=\begin{cases}0,\quad&x_{t}\neq m\\ \left(t-s\right)\sigma(t)s_{\theta}(x_{t},t)_{x_{s}},\quad&x_{t}=m\end{cases}$ (61) $$

When $x_{s}=x_{t}$, from Eqn. (56) we know $\sum_{x\in\mathcal{X}\backslash\{m\}}s_{\theta}(m,t)_{x}=\frac{\alpha_{t}}{1-
\alpha_{t}}\sum_{x\in\mathcal{X}\backslash\{m\}}\mu_{\theta}(m,t)_{x}=\frac{
\alpha_{t}}{1-\alpha_{t}}$. Hence, $p_{s|t}^{\rm Euler}$ can be calculated as

$$ $p_{s|t}^{\rm Euler}(x_{s}|x_{t})=\begin{cases}1,\quad&x_{t}\neq m\\ 1-\frac{\sigma(t)e^{-\bar{\sigma}(t)}(t-s)}{1-e^{-\bar{\sigma}(t)}},\quad&x_{t }=m\end{cases}$ (62) $$

Combining Eqn. (61) and Eqn. (62), the Euler sampler is simplified to

$$ $p_{s|t}^{\rm Euler}(x_{s}|x_{t})=\begin{cases}\delta_{x_{s},x_{t}},\quad&x_{t} \neq m\\ \left(t-s\right)\sigma(t)s_{\theta}(x_{t},t)_{x_{s}},\quad&x_{t}=m,x_{s}\neq m \\ 1-\frac{\sigma(t)e^{-\bar{\sigma}(t)}(t-s)}{1-e^{-\bar{\sigma}(t)}},\quad&x_{t }=m,x_{s}=m\end{cases}$ (63) $$

On the other hand, the Tweedie $\tau$-Leaping sampler (Eqn. (49)) is

$$ $p_{s|t}^{\rm Tweedie}(x_{s}|x_{t})=\left(e^{(\bar{\sigma}(t)-\bar{\sigma}(s)) \bm{Q}}\right)_{x_{s},x_{t}}\left(\bm{s}_{\theta}(x_{t},t)e^{-(\bar{\sigma}(t) -\bar{\sigma}(s))\bm{Q}}\right)_{x_{s}}$ (64) $$

When $\bm{Q}=\bm{Q}_{\rm absorb}$, similar to Eqn. (51), we have

$$ $\displaystyle e^{(\bar{\sigma}(t)-\bar{\sigma}(s))\bm{Q}}$ $\displaystyle=\bm{I}+\left(1-e^{-(\bar{\sigma}(t)-\bar{\sigma}(s))}\right)\bm{ Q}_{\rm absorb}$ (65) $\displaystyle e^{-(\bar{\sigma}(t)-\bar{\sigma}(s))\bm{Q}}$ $\displaystyle=\bm{I}+\left(1-e^{\bar{\sigma}(t)-\bar{\sigma}(s)}\right)\bm{Q}_ {\rm absorb}$ $$

Using the identities $s_{\theta}(x,t)_{x}=1$ and $\left(\bm{Q}_{\rm absorb}\right)_{x_{s},x_{t}}=\delta_{m,x_{t}}-\delta_{x_{s},
x_{t}}$, we have

$$ $\displaystyle\left(e^{(\bar{\sigma}(t)-\bar{\sigma}(s))\bm{Q}}\right)_{x_{s},x _{t}}$ $\displaystyle=\delta_{x_{s},x_{t}}+\left(1-e^{-(\bar{\sigma}(t)-\bar{\sigma}(s ))}\right)\left(\bm{Q}_{\rm absorb}\right)_{x_{s},x_{t}}$ (66) $\displaystyle=e^{-(\bar{\sigma}(t)-\bar{\sigma}(s))}\delta_{x_{s},x_{t}}+\left (1-e^{-(\bar{\sigma}(t)-\bar{\sigma}(s))}\right)\delta_{m,x_{t}}$ $$

and

$$ $\sum_{y\in\mathcal{X}}s_{\theta}(x,t)_{y}=\sum_{y\in\mathcal{X}}\frac{s_{ \theta}(m,t)_{y}}{s_{\theta}(m,t)_{x}}=s_{\theta}(x,t)_{m}\left(s_{\theta}(m,t )_{m}+\sum_{x\in\mathcal{X}\backslash\{m\}}s_{\theta}(m,t)_{x}\right)=\frac{s_ {\theta}(x,t)_{m}}{1-\alpha_{t}}$ (67) $$

Therefore,

$$ $\displaystyle\left(\bm{s}_{\theta}(x_{t},t)e^{-(\bar{\sigma}(t)-\bar{\sigma}(s ))\bm{Q}}\right)_{x_{s}}$ $\displaystyle=s_{\theta}(x_{t},t)_{x_{s}}+\left(1-e^{\bar{\sigma}(t)-\bar{ \sigma}(s)}\right)\left(\bm{s}_{\theta}(x_{t},t)\bm{Q}_{\rm absorb}\right)_{x_ {s}}$ (68) $\displaystyle=s_{\theta}(x_{t},t)_{x_{s}}+\left(1-e^{\bar{\sigma}(t)-\bar{ \sigma}(s)}\right)\sum_{x\in\mathcal{X}}s_{\theta}(x_{t},t)_{x}\left(\bm{Q}_{ \rm absorb}\right)_{x,x_{s}}$ $\displaystyle=e^{\bar{\sigma}(t)-\bar{\sigma}(s)}s_{\theta}(x_{t},t)_{x_{s}}+ \delta_{x_{s},m}\left(1-e^{\bar{\sigma}(t)-\bar{\sigma}(s)}\right)\sum_{x\in \mathcal{X}}s_{\theta}(x_{t},t)_{x}$ $\displaystyle=e^{\bar{\sigma}(t)-\bar{\sigma}(s)}s_{\theta}(x_{t},t)_{x_{s}}+ \delta_{x_{s},m}\frac{1-e^{\bar{\sigma}(t)-\bar{\sigma}(s)}}{1-e^{-\bar{\sigma }(t)}}s_{\theta}(x_{t},t)_{m}$ $$

Substituting Eqn. (66) and Eqn. (68) into Eqn. (64), the Tweedie $\tau$-Leaping sampler is simplified to

$$ $p_{s|t}^{\rm Tweedie}(x_{s}|x_{t})=\begin{cases}\delta_{x_{s},x_{t}},\quad&x_{ t}\neq m\\ \left(e^{\bar{\sigma}(t)-\bar{\sigma}(s)}-1\right)s_{\theta}(x_{t},t)_{x_{s}}, \quad&x_{t}=m,x_{s}\neq m\\ \frac{1-e^{-\bar{\sigma}(s)}}{1-e^{-\bar{\sigma}(t)}},\quad&x_{t}=m,x_{s}=m \end{cases}$ (69) $$

Under the linear noise schedule, as $e^{-\bar{\sigma}(t)}=1-t$, we have $\bar{\sigma}(t)=-\log(1-t)$ and $\sigma(t)=\frac{1}{1-t}=e^{\bar{\sigma}(t)}$. Consequently, $(t-s)\sigma(t)=e^{\bar{\sigma}(t)-\bar{\sigma}(s)}-1$, and the Euler sampler in Eqn. (63) is the same as Tweedie $\tau$-Leaping sampler in Eqn. (69).

Tweedie $\tau$-Leaping Sampler in SEDD and the Reverse Sampling Process in MDMs are equivalent in the absorbing case. By the relation $\alpha_{t}=e^{-\bar{\sigma}(t)}$ and $s_{\theta}(x_{t},t)_{x_{s}}=\frac{\alpha_{t}}{1-\alpha_{t}}\mu_{\theta}(x_{t},
t)_{x_{s}}(x_{t}=m,x_{s}\neq m)$, the Tweedie $\tau$-Leaping sampler in Eqn. (69) is converted to

$$ $p_{s|t}^{\rm Tweedie}(x_{s}|x_{t})=\begin{cases}\delta_{x_{s},x_{t}},\quad&x_{ t}\neq m\\ \frac{\alpha_{s}-\alpha_{t}}{1-\alpha_{t}}\mu_{\theta}(x_{t},t)_{x_{s}},\quad& x_{t}=m,x_{s}\neq m\\ \frac{1-\alpha_{s}}{1-\alpha_{t}},\quad&x_{t}=m,x_{s}=m\end{cases}$ (70) $$

which is the same as the reverse process in MDMs (Eqn. (3)).

## Appendix E Expected NFE in Batched Sampling Using the Caching Strategy

Suppose the sampling is performed on timesteps $1=t_{N}\rightarrow t_{N-1}\rightarrow\dots\rightarrow t_{0}=0$, and denote $\bm{x}_{t}\in\mathcal{X}^{BL}$ as the concatenation of the batched sequences at time $t$. During the sampling step $t_{i}\rightarrow t_{i-1}$, the NFE increases by 1 when $\bm{x}_{t_{i-1}}\neq\bm{x}_{t_{i}}$, and remains the same otherwise. Therefore, the expected NFE (E-NFE) can be expressed by

$$ E-NFE $\displaystyle=\mathbb{E}\left[\sum_{i=1}^{N}\mathbb{I}_{\bm{x}_{t_{i-1}}\neq \bm{x}_{t_{i}}}\right]=\sum_{i=1}^{N}\mathbb{E}\left[\mathbb{I}_{\bm{x}_{t_{i- 1}}\neq\bm{x}_{t_{i}}}\right]=\sum_{i=1}^{N}P\left(\bm{x}_{t_{i-1}}\neq\bm{x}_ {t_{i}}\right)$ (71) $\displaystyle=\sum_{i=1}^{N}\left(1-P\left(\bm{x}_{t_{i-1}}=\bm{x}_{t_{i}} \right)\right)=\sum_{i=1}^{N}\left(1-P\left(x^{(l)}_{t_{i-1}}=x^{(l)}_{t_{i}}, l=1,2,\dots,BL\right)\right)$ $$

As noted in the main text, an unmasked token will no longer change, and whether a mask token will change during a sampling step is independent of the network output. Given that the reverse sampling process is factorized across dimensions, and the only interaction between dimensions is through the sequence-conditioned network, which does not affect whether a token will change, we conclude that the events $\{x^{(l)}_{t_{i-1}}=x^{(l)}_{t_{i}}\}$ are independent for different $l$. Therefore, the probability $P\left(x^{(l)}_{t_{i-1}}=x^{(l)}_{t_{i}},l=1,2,\dots,BL\right)$ can be factorized as $\prod_{l=1}^{BL}P\left(x^{(l)}_{t_{i-1}}=x^{(l)}_{t_{i}}\right)$, where

$$ $\displaystyle P\left(x^{(l)}_{t_{i-1}}=x^{(l)}_{t_{i}}\right)$ $\displaystyle=P\left(x^{(l)}_{t_{i}}=m\right)P\left(x^{(l)}_{t_{i-1}}=m|x^{(l) }_{t_{i}}=m\right)+P\left(x^{(l)}_{t_{i}}\neq m\right)$ (72) $\displaystyle=\frac{1-\alpha_{t_{i}}}{1-\alpha_{1}}\frac{1-\alpha_{t_{i-1}}}{1 -\alpha_{t_{i}}}+1-\frac{1-\alpha_{t_{i}}}{1-\alpha_{1}}$ $\displaystyle=1-(\alpha_{t_{i-1}}-\alpha_{t_{i}})$ $$

The expected NFE is finally simplified to

$$ $\text{E-NFE}=\sum_{i=1}^{N}\left(1-\left(1-(\alpha_{t_{i-1}}-\alpha_{t_{i}}) \right)^{BL}\right)$ (73) $$

Using the default linear noise schedule $\alpha_{t}=1-t$ as well as uniform timesteps $t_{k}=\frac{k}{N}$, we have $\alpha_{t_{i-1}}-\alpha_{t_{i}}=\frac{1}{N}$, and the expected NFE is $N\left(1-(1-\frac{1}{N})^{BL}\right)$.

## Appendix F A Brief Introduction to Gumbel Tricks

Gumbel tricks are widely used in machine learning and statistics to handle the challenges associated with discrete random variables. Based on the properties of the Gumbel distribution, these techniques offer powerful tools for approximating discrete distributions and optimizing over discrete spaces, facilitating the integration of discrete variables into continuous models.

### F.1 The Gumbel Distribution

The Gumbel distribution  is related to the extreme value theory and is commonly used to model the distribution of the maximum of random variables. The Gumbel distribution $\mathcal{G}(\mu,\beta)$, with the location parameter $\mu$ and the scale parameter $\beta$, has the following PDF and CDF:

$$ $F(x;\mu,\beta)=e^{-e^{-(x-\mu)/\beta}},\quad f(x;\mu,\beta)=\frac{1}{\beta}e^{ -(x-\mu)/\beta-e^{-(x-\mu)/\beta}}$ (74) $$

A widely used special case is the standard Gumbel distribution $\mathcal{G}(0,1)$, where $\mu=0$ and $\beta=1$. For this distribution, the CDF is given by $P(g\leq x)=e^{-e^{-x}}$. Using inverse transform sampling, a random variable $g$ following $\mathcal{G}(0,1)$ can be sampled by drawing $u\sim\mathcal{U}(0,1)$ and performing two negative logarithm operations $g=-\log(-\log u)$.

### F.2 Gumbel-Max Trick

The Gumbel-max trick  allows us to sample from a categorical distribution using continuous random variables. It leverages the following property of the Gumbel distribution: for $\bm{\pi}=(\pi_{1},\pi_{2},\dots,\pi_{K})$ satisfying $\bm{\pi}\geq 0$ and $\sum_{i=1}^{K}\pi_{i}>0$, and let $g_{1},\dots,g_{K}$ be independent samples from $\mathcal{G}(0,1)$, we have

$$ $\max_{i}\left(g_{i}+\log\pi_{i}\right)\sim\mathcal{G}\left(\log\sum_{i=1}^{K} \pi_{i},1\right)$ (75) $$

and

$$ $\operatornamewithlimits{argmax}_{i}\left(g_{i}+\log\pi_{i}\right)\sim\mbox{Cat }\left(\frac{\bm{\pi}}{\sum_{i=1}^{K}\pi_{i}}\right)$ (76) $$

This implies that sampling a discrete variable from a categorical distribution can be achieved by operating on continuous variables that include the known class probabilities and the sampled Gumbel variables. Therefore, the Gumbel-max trick acts as the reparameterization trick for categorical sampling, akin to the Gaussian case used in variational auto-encoders .

### F.3 Gumbel-Softmax Distribution

The Gumbel-Softmax distribution  introduces a differentiable approximation to the categorical distribution, facilitating gradient-based optimization in neural network training. It smooths the non-differentiable argmax operation in the Gumbel-max trick by replacing it with the differentiable Softmax function plus a temperature factor $T$. Specifically, the one-hot random vector $\bm{e}_{x}$, where $x\sim\mbox{Cat}(\bm{\pi})$, is approximated by a continuous vector $\bm{y}$ defined as

$$ $y_{i}=\frac{e^{(\log\pi_{i}+g_{i})/T}}{\sum_{j=1}^{K}e^{(\log\pi_{j}+g_{j})/T} },\quad i=1,2,\dots,K$ (77) $$

It approaches the one-hot representation of the categorical variable when $T\rightarrow 0$, and tends to be uniform when $T\rightarrow\infty$.

## Appendix G Implementation Details

### G.1 Algorithms

For parallel decoding, suppose the sampling step is $N$ and the sequence length is $L$, we define a decoding schedule $\{L_{n}\}_{n=1}^{N}$ which satisfies $\sum_{n=1}^{N}L_{n}=L$ to specify the number of tokens decoded at each step. This includes the token-by-token decoding as a special case where $N=L$ and $L_{n}=1$. In practice, we decode the same number of tokens per step so that $L$ is divisible by $N$.

We present the parallel decoding procedure in Algorithm 2, which can be interpreted as a first-order method. Algorithm 3 and Algorithm 4 describe two types of high-order extensions, inspired by high-order numerical differential equation solvers in diffusion models. Algorithm 3 leverages Lagrange polynomials to interpolate the previous network outputs along the time axis, yielding an approximate network prediction for the current time step. Our implementation only uses the two most recent predictions, making it a second-order method, as we empirically find that higher-order methods tend to degrade performance. Algorithm 4 employs a predictor-corrector approach, refining the first-order decoding result at the last step using the current network prediction, also resulting in a second-order method. After refining the intermediate sample, we avoid feeding it back into the network for prediction updates, thus preventing extra NFEs.

Figure: Algorithm 2 First-Hitting Sampling of MDMs (parallel decoding)

Figure: Algorithm 3 First-Hitting Sampling of MDMs (extrapolation)

Figure: Algorithm 4 First-Hitting Sampling of MDMs (predictor-corrector)

### G.2 Low-Discrepancy Sampler

VDM  proposes a low-discrepancy sampler for batched sampling of uniformly distributed continuous time variables, which reduces the loss variance in maximum likelihood training of diffusion models. Specifically, consider a batch of $B$ timesteps ${t^{(i)}}_{i=0}^{B-1}$ that needs to be sampled from $\mathcal{U}(0,1)$. Instead of sampling them independently, VDM generates correlated samples using the formula $t^{(i)}=\text{mod}(u_{0}+i/B,1)$, where $u_{0}\sim\mathcal{U}(0,1)$. This approach ensures that each $t^{(i)}$ has the correct marginal distribution over multiple batches, while each batch of timesteps more evenly covers the interval $[0,1]$.

MDLM  employs a slightly different low-discrepancy sampler where the sampled timesteps are less correlated within a batch. Specifically, $B$ independent uniform samples $\{u_{i}\}_{i=0}^{B-1}$ from $\mathcal{U}(0,1)$ are mapped into $B$ bins by $t^{(i)}=(u_{i}+i)/B$. We adopt this approach for sampling continuous timesteps. Additionally, we extend this low-discrepancy sampler to handle discrete timesteps $\{n^{(i)}\}_{i=0}^{B-1}$ drawn from $\mathcal{U}(\{0,1,\dots,L-1\})$ (e.g., the number of masked tokens in the discrete ELBO) by mapping the continuous time $t^{(i)}$ to $n^{(i)}=\lceil Lt^{(i)}\rceil$.

## Appendix H Experiment Details

### H.1 Evaluation Metrics

Perplexity is a likelihood-related metric to evaluate how well a likelihood-based model is trained. Denote the likelihood (i.e., the probability of the data under the parameterized model) for the data point $\bm{x}_{0}$ as $p_{\theta}(\bm{x}_{0})$. The log-likelihood can be expressed either exactly or through an ELBO:

$$ $\displaystyle\text{Auto-regressive Models:}\quad\log p_{\theta}(\bm{x}_{0})$ $\displaystyle=\sum_{n=1}^{L}\log p_{\theta}(x_{0}^{(n)}|\bm{x}_{0}^{(<n)})$ (78) $\displaystyle\text{Masked Models:}\quad\log p_{\theta}(\bm{x}_{0})$ $\displaystyle\geq\sum_{n=1}^{L}\mathbb{E}_{\tilde{q}_{n|0}(\bm{x}_{n}|\bm{x}_{ 0})}\left[\frac{1}{n}\sum\nolimits_{l:x_{n}^{(l)}=m}\log p_{\theta}(x_{0}^{(l) }|\bm{x}_{n})\right]$ (79) $$

Here we express the log-likelihood of masked models with our derived discrete ELBO. We use $\log p_{\theta}(x_{0}^{(l)}|\bm{x}_{n})$ (predicted data probability given known tokens) as an alternative expression for the cross-entropy term $\bm{e}_{x_{0}^{(l)}}^{\top}\log\bm{\mu}^{(l)}_{\theta}(\bm{x}_{n})$, aligning it with the common formulation in ARMs. The perplexity (PPL) is defined as:

$$ $\text{PPL}=\exp\left(\frac{\mathbb{E}_{\bm{x}_{0}\sim p_{\text{data}}}\left[- \log p_{\theta}(\bm{x}_{0})\right]}{D}\right)$ (80) $$

where $D$ is the data dimension, and $p_{\text{data}}$ is the data distribution (such as the validation/test set).

Generative Perplexity evaluates a model’s generation quality by measuring the perplexity of its generated samples under some off-the-shelf model. It is related to both training and sampling. We adopt GPT-2 Large as the off-the-shelf evaluator following previous works.

Entropy measures the diversity of tokens in a sequence. For a sequence of length $L$ that contains $K$ distinct tokens, with each token $k$ occurring $L_{k}$ times, the entropy is computed as $-\sum_{k=1}^{K}p_{k}\log p_{k}$, where $p_{k}=L_{k}/L$ represents the probability of occurrence of token $k$.

### H.2 Model and Dataset Details

Following SEDD  and MDLM , we utilize an encoder-only transformer with a DDiT  architecture, incorporating RoPE . We use the small-size model variant, which consists of 12 layers, 12 attention heads, a hidden dimension of 768, and a timestep embedding dimension of 128, amounting to approximately 170M parameters including the word embedding matrix.

Our experiments are conducted on the OpenWebText dataset , which contains around 8 million documents, with the last 100k reserved for validation. The dataset is tokenized using the GPT-2 tokenizer, resulting in a vocabulary size of 50,257 (excluding the mask token). Sequences are concatenated and wrapped to a length of 1024 tokens, with the first, last, and in-between tokens of concatenated sequences set to eos.

### H.3 Training Details

Following SEDD  and MDLM , we use the AdamW optimizer with a batch size of 512 and a learning rate that is linearly warmed up from 0 to 3e-4 over the first 2,500 steps. We apply a dropout rate of 0.1, clip the gradient norm to 1, and utilize an exponential moving average (EMA) with a rate of 0.9999. Mixed-precision training is enabled with bfloat16.

All our training experiments are conducted on 8 NVIDIA A100 40GB GPUs for slightly over 100k iterations, which takes around 1.5 days.

### H.4 Sampling Details

We directly use the pretrained models (AR, SEDD Absorb, MDLM) trained on OpenWebText provided by MDLM(^11^1111[https://github.com/kuleshov-group/mdlm/](https://github.com/kuleshov-group/mdlm/)). These models share the same architecture and size (with the exception of the final layer in AR). SEDD and MDLM are trained for 1M iterations, while the corresponding AR baseline is trained for half as many steps to ensure a comparable number of tokens seen.

For the baselines, SEDD is sampled using its analytic sampler (Tweedie $\tau$-leaping), and MDLM is sampled both with and without the caching strategy. Their sampling timesteps are uniformly discretized. In our first-hitting sampler, parallel decoding is achieved by unmasking the same number of tokens at each step. Although the pretrained MDLM model is claimed to be time-independent, we find that adding the time condition slightly improves performance in our sampler. All sampling experiments are conducted on a single NVIDIA
RTX A6000 GPU, and the reported metrics are averaged on 64 random samples.

## Appendix I Old Version of the Introduction

We highlight our key findings: (1) MDMs, in both training and sampling, are essentially time-agnostic masked models (or order-agnostic auto-regressive models), enjoying 20$\times$ faster sampling and diverging from the design choices of diffusion models. This also justifies the theoretical foundation of masked models as they are equivalent and simpler formulations of the more principled MDMs. (2) We challenge previous claims that MDMs can surpass ARMs in text generation by identifying a hidden but critical numerical issue that reduces the token diversity and renders previous evaluations unfair. After fixing it, we find MDMs significantly lagging behind ARMs in generative perplexity.

For training, we prove that the continuous-time evidence lower bound (ELBO) objective of MDMs can be expressed by the number of masked tokens with an implicitly defined mixture-of-experts model. It provides a discrete ELBO for masked models and coincides with the ELBO previously derived for order-agnostic auto-regressive models .

For sampling, by analytically sampling the time when any mask token is first unmasked, we propose a theoretically equivalent first-hitting sampler (FHS) to avoid most of the time-consuming categorical sampling and perform decoding token by token with no approximation errors. It is further extended to enable parallel decoding and incorporate high-order approximations, achieving a 20$\times$ speedup compared to previous MDM sampling procedures. When the parameterized model is independent of the time variable, we recover the sampling of masked models.

For evaluation, we discover that while MDMs exhibit extremely low generative perplexity with numerous sampling steps, the generation quality is compromised by reduced token diversity. By examining the numerical precision during sampling, we identify a previously unrecognized issue with Gumbel-based categorical sampling.
Specifically, reducing the floating-point precision from 64-bit to 32-bit significantly truncates the Gumbel variables, which theoretically lowers the temperature and empirically improves the generative perplexity of pretrained models  from 126.11 to 31.24, but with a decreased sentence entropy from 5.66 to 5.17.

## Appendix J Additional Results

### J.1 Training Results

#### J.1.1 Comparison of Training Variants

Figure: (a) Training Loss
Refer to caption: x13.png

We compare different training variants (continuous-time/discrete ELBO, time-conditioned/time-independent network) of MDMs. By default, we condition the network on time for continuous-time ELBO and on the masked ratio for discrete ELBO. We also apply the low-discrepancy sampler described in Appendix G.2. Providing the network with extra conditions as auxiliary information may potentially facilitate the training process.

As shown in Figure 11, all variants exhibit similar performance in both training and validation. Adding the time condition provides a slight improvement over the time-independent network, and the discrete ELBO performs marginally worse than the continuous-time ELBO. However, these differences are negligible. The low-discrepancy sampler notably reduces the variance in training loss, though the validation perplexity curve remains relatively stable, likely due to the smoothing effect of the EMA. Nonetheless, MDMs still significantly lag behind the counterpart ARM.

#### J.1.2 Failed Training Attempts

Training MDMs with the ELBO is analogous to maximum likelihood training of diffusion models . Therefore, we borrow well-established techniques from the SOTA likelihood model i-DODE  within the diffusion literature, including velocity parameterization and variance reduction.

##### Flow Matching/Preconditioning

Different parameterizations are theoretically equivalent but have distinct empirical implications in diffusion models. As an alternative to data or noise prediction, velocity parameterization has proven effective in the maximum likelihood training of diffusion models . It is also related to flow matching  and the preconditioning technique in EDM . As MDMs employ mean parameterization (or data prediction), exploring alternative parameterizations may enhance training performance.

In diffusion models, different parameterizations can be understood as expressing the mean prediction model $\bm{\mu}_{\theta}$ with a “skip-connection” style preconditioning:

$$ $\bm{\mu}_{\theta}(\bm{x}_{t},t)=a_{t}\bm{F}_{\theta}(\bm{x}_{t},t)+b_{t}\bm{x} _{t}$ (81) $$

where $a_{t},b_{t}$ are some specific time-related coefficients, and $\bm{F}_{\theta}$ is a free-form network. In MDMs, this general preconditioning can be formulated by the one-hot vector $\bm{e}_{x_{t}^{(l)}}$:

$$ $\bm{\mu}_{\theta}^{(l)}(\bm{x}_{t},t)=a_{t}\bm{F}^{(l)}_{\theta}(\bm{x}_{t},t) +b_{t}\bm{e}_{x_{t}^{(l)}}$ (82) $$

However, such a formulation in MDMs makes no difference to the training. $\bm{\mu}_{\theta}^{(l)}(\bm{x}_{t},t)$ is only trained when $x_{t}^{(l)}=m$ to predict the data probabilities at dimensions $0\sim m-1$. Adding $\bm{e}_{x_{t}^{(l)}}$ (which is 0 at dimension $0\sim m-1$) to the model output does not impact the functional dimensions.

To tackle this, we attempt to employ a self-conditioning technique

$$ $\bm{\mu}_{\theta}(\bm{x}_{t},t)\coloneqq\bm{F}_{\theta}(\bm{x}_{t},t,\bm{F}_{ \theta^{-}}(\bm{x}_{t},t))$ (83) $$

where $\theta^{-}$ is the stop-gradient version of $\theta$. For implementation, the extra condition $\bm{F}_{\theta^{-}}$ (1) is concatenated to the original input along the feature dimension (2) is replaced by blank with 50% probability during training, and substituted by the earlier model prediction during inference. However, we empirically find this technique highly unstable during training, adding extra model parameters and incurring excessive training costs.

Figure: (a) Training Loss
Refer to caption: x15.png

##### Variance Reduction via Importance Sampling

The NELBO for both diffusion models and MDMs can be expressed as an expectation $\mathcal{L}=\mathbb{E}_{t}\left[\mathcal{L}_{t}\right]$ over uniformly distributed $t$, where $t$ can be either continuous (e.g., the continuous time) or discrete (e.g., the discrete time or the number of masked tokens). Importance sampling (IS) for $t$ can be introduced by rewriting the training loss with a proposal distribution $p_{t}$:

$$ $\mathcal{L}=\mathbb{E}_{t\sim p_{t}}\left[\tilde{\mathcal{L}}_{t}\right],\quad \tilde{\mathcal{L}}_{t}=\frac{\mathcal{L}_{t}}{p_{t}}$ (84) $$

While the overall loss $\mathcal{L}$ remains invariant to the choice of $p_{t}$, the variance of the loss, $\mbox{Var}_{t\sim p_{t}}\left[\tilde{\mathcal{L}}_{t}\right]$, is influenced by $p_{t}$. We can optimize $p_{t}$ for variance minimization. For continuous $t$, the density $p_{t}$ can be parameterized by a monotonic neural network and learned by gradient descent .

For simplicity, we instead consider the discrete case of $t$ (the number of masked tokens in the discrete ELBO). We adopt the adaptive IS proposed by Improved DDPM . Specifically, as $\mbox{Var}_{t\sim p_{t}}\left[\tilde{\mathcal{L}}_{t}\right]=\mathbb{E}_{t\sim
p
_{t}}\left[\tilde{\mathcal{L}}_{t}^{2}\right]-\mathcal{L}^{2}$, the optimal $p_{t}$ is given by $p_{t}\propto\sqrt{\mathbb{E}\left[\mathcal{L}_{t}^{2}\right]}$. Given that $\mathbb{E}\left[\mathcal{L}_{t}^{2}\right]$ is unknown in advance and may vary throughout training, we maintain a history of the previous 10 values for each loss term and update this dynamically during training. At the beginning of training, we sample $t$ uniformly until we have 10 samples for every $t$.

We visualize the training curves and optimized importance weights at around 100k iterations in Figure 12. Unfortunately, while the adaptive IS technique reduces the variance to a level comparable to the low-discrepancy sampler, it also results in degraded performance. We hypothesize that the dynamical updated $p_{t}$ may make the loss estimator biased.

### J.2 Sampling Results

#### J.2.1 Comparison of High-Order Variants

Figure: (a) Generative Perplexity
Refer to caption: x18.png

In Figure 13, we compare the two high-order variants of our proposed first-hitting sampler. In terms of the generative perplexity, The extrapolation strategy performs best when $N\leq 128$, and the predictor-corrector strategy is more effective when $N\geq 256$. We also observe that lower generative perplexity is associated with decreased entropy, indicating an inherent trade-off.

#### J.2.2 Impact of Numerical Precision on Our Sampler

Figure: (a) Generative Perplexity
Refer to caption: x20.png

In Figure 14, we examine the impact of numerical precision on our first-hitting sampler by varying the floating-point precision in categorical sampling between 32-bit and 64-bit. For the high-order variants, we employ the extrapolation strategy when $N\leq 128$ and the predictor-corrector strategy when $N\geq 256$. In contrast to the observations under MDM’s original sampler, the temperature-lowering effect of the numerical truncation is significantly less influential under our sampler. Notably, the 32-bit second-order first-hitting sampler even results in slightly higher entropy compared to its 64-bit counterpart when $N\geq 512$.

Figure: Figure 15: Illustration of prioritized unmasking.
Refer to caption: x22.png

##### Why the numerical issue is not notable in token-by-token decoding

With our first-hitting sampler, the inference of MDMs becomes a token-by-token decoding process, except that the time variable is additionally handled. In this case, the numerical issue becomes negligible, and 32-bit floating-point precision appears sufficient. We also observe that ARMs, which also adopt a token-by-token sampling strategy, do not suffer from numerical issues under 32-bit precision. This suggests that the numerical problem is a distinctive characteristic of the vanilla sampling process of MDMs (Eqn. (10)) that performs inaccurate categorical sampling simultaneously on all mask positions.

This phenomenon can be explained as follows. As justified theoretically in Section 5.3, the implication of inaccurate categorical sampling includes two aspects: (1) temperature-lowering effect for a single token position and (2) prioritized unmasking for different positions (Figure 15). Both factors reduce the diversity and lower the entropy. However, when altering to token-by-token decoding, all remaining mask tokens have equal probability to be first unmasked, and the diversity decrease becomes less pronounced. This suggests that the prioritized unmasking caused by shifted probabilities at all individual positions is the major factor. This effect accumulates across numerous sampling steps, eventually leading to notable diversity issues, even under 32-bit floating-point precision.

### J.3 Anaysis of the Efficiency Gain by Our Sampler

Due to the theoretical equivalence between the FHS and the original MDM sampling procedure, the correctness of the FHS is guaranteed and irrelevant to vocabulary size or sequence length. However, the efficiency gains (measured by inference wall-clock time) depend on several factors.

Let the sequence length be denoted as $L$, vocabulary size as $|V|$ (excluding the mask token), the number of sampling steps (for original MDM sampling) as $N$, the number of function evaluations as $NFE$, and the number of categorical sampling operations as $NCS$. As stated in Section 4, our FHS reduces inference time by minimizing $NCS$. For original MDM sampling with the caching strategy, $NFE\approx N(1-(1-1/N)^{L})$, $NCS=NL|V|$. For the FHS, $NCS=L|V|$. The total time cost can be expressed as $NFE\times t_{1}+NCS\times t_{2}$, where

- •
$t_{1}$ is the time for one network call, influenced by model size.
- •
$t_{2}$ is the time for categorical sampling, averaged per position and class, which involves two logarithmic operations, and is fixed.

For a fair comparison, we evaluate under the same $NFE$, ensuring similar generation quality. The inference time ratio between original MDM sampling and the FHS is $(NFE\times t_{1}+NL|V|t_{2}):(NFE\times t_{1}+L|V|t_{2})$. Therefore, the FHS yields larger speedups under the following conditions:

- •
Smaller Model Size: When the model is smaller, $t_{1}$ decreases, making categorical sampling relatively more expensive compared to network evaluation.
- •
Larger $NFE$, $L$, or $|V|$: Higher values for these parameters increase $NCS$ in original MDM sampling, amplifying the speedup provided by the FHS.

Examples:

- •
Paper Case: $|V|=50{,}526$, $L=1024$. To match the original MDM sampling at $N=10{,}000$ steps, $NFE\approx 973$. The model size is approximately 600M parameters, with a time ratio of $t_{1}:L|V|t_{2}\approx 1:1.56$. In this case, the inference time ratio is around 17, corresponding to a 20$\times$ speedup. If $N=2048$, the ratio drops to around 5$\times$.
- •
DiffSound : This model is 3$\times$ larger and uses a much smaller vocabulary. Specifically, $L=265$, $|V|=256$, with $t_{1}:L|V|t_{2}\approx 14.6:1$. Here, categorical sampling is much cheaper relative to network evaluations. Using fewer steps, $N=100$, results in $NFE\approx 93$, and the speedup ratio is only around 1.07$\times$.

#### J.3.1 Examples of Generated Text

Figure: Figure 16: The counterpart ARM.

Figure: Figure 17: MDLM with the original sampler (32-bit), 50k steps.

Figure: Figure 18: MDLM with the original sampler (64-bit), 50k steps.

Figure: Figure 19: MDLM with our first-hitting predictor-corrector sampler, 1024 steps.