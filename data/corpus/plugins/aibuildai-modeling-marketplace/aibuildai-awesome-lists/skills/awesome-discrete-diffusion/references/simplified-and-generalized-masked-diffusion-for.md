---
arxiv_id: "2406.04329"
title: "Simplified and Generalized Masked Diffusion for Discrete Data"
year: 2024
source: arxiv2md
---

## Abstract

Abstract Masked (or absorbing) diffusion is actively explored as an alternative to autoregressive models for generative modeling of discrete data.
However, existing work in this area has been hindered by unnecessarily complex model formulations and unclear relationships between different perspectives, leading to suboptimal parameterization, training objectives, and ad hoc adjustments to counteract these issues.
In this work, we aim to provide a simple and general framework that unlocks the full potential of masked diffusion models.
We show that the continuous-time variational objective of masked diffusion models is a simple weighted integral of cross-entropy losses. Our framework also enables training generalized masked diffusion models with state-dependent masking schedules. When evaluated by perplexity, our models trained on OpenWebText surpass prior diffusion language models at GPT-2 scale and demonstrate superior performance on 4 out of 5 zero-shot language modeling tasks.
Furthermore, our models vastly outperform previous discrete diffusion models on pixel-level image modeling, achieving 2.75 (CIFAR-10) and 3.40 (ImageNet 64 × \times × 64) bits per dimension that are better than autoregressive models of similar sizes.
Our code is available at https://github.com/google-deepmind/md4 .

## 1 Introduction

Since their inception , diffusion models have emerged as the workhorse for generative media, achieving state-of-the-art in tasks such as image synthesis , audio  and video generation .
The majority of existing successes are for continuous state space diffusions. While diffusion models have been extended to discrete state spaces and have been successfully applied to applications ranging from graph generation , text-to-sound generation or protein design , they remain not as widely used as their continuous counterparts as they are not competitive with autoregressive models in important domains such as text modeling. This has motivated the development of continuous space diffusion models where the discrete data are embedded in the Euclidean space or the simplex . We believe that one of the reasons for the limited success of discrete diffusions is that they have been hindered by fairly complex formulations and training objectives. This paper is a step towards closing this gap.

In this work, we focus on “masked” (or “absorbing”) diffusions, a discrete diffusion formulation first presented by , and later explored by the literature from various perspectives
. We follow here a continuous-time framework which has proven very useful to improve the training and understanding of continuous state space diffusions . We make several technical contributions which simplify the training of these models and improve significantly their performance. Our contributions are as follows:

- •
Using elementary arguments, we establish several properties for the forward process induced by this model and its corresponding time reversal,
improving our understanding of this model class.
- •
We provide a remarkably simple expression of the Evidence Lower Bound (ELBO) for masked diffusion models, showing that it corresponds to a weighted integral over time of cross-entropy losses. Similarly to continuous space diffusions , this objective can be rewritten in terms of signal-to-noise ratio and exhibits invariance properties.
- •
We develop a unifying understanding of previously proposed continuous-time discrete diffusion models , revealing the changes they made to our ELBO objective and/or model parameterization.
We show that these changes either lead to expensive model evaluations, or large variance in training, or breaking the consistency between forward and reverse processes.
- •
On GPT-2 scale text modeling and pixel-level image modeling tasks, masked diffusions trained using our simple ELBO objective outperform previous proposals, leading to the best likelihood and zero-shot transfer performance among discrete diffusion models.
- •
Finally, based on our simplified masked diffusion formulation, we propose a generalized masked diffusion model that allows state-dependent masking schedules. This generalized masked diffusion model further improves predictive performance measured by test likelihoods.

Concurrent work by and derives a similar simplified expression of the ELBO. ’s derivation relies on an observation similar to the one we made in Proposition 1.

## 2 Masked Diffusion

Consider a sentence where we progressively replace each word with a special mask token, transforming the sentence into a sequence of masks.
Our goal is to train a generative model that reverses this process, effectively turning a sentence of masks back into meaningful text.
More formally, assume our data consists of tokens from a finite discrete state space with $m$ possible states, represented by integers
$0,1,\dots,m-1$ and
their corresponding one-hot vectors $e_{0},e_{1},\dots,e_{m-1}$. To accommodate the masking process, we augment this space with an additional mask state, denoted by the index $m$.
The masking process transitions each token to
the mask state at a random time.
This process, known as the forward process, is applied independently to each token (e.g., each word), progressively converting the data into a sequence of mask tokens.
By learning to reverse this masking process, we create a generative model capable of producing coherent discrete data.

##### Discrete-time forward process.

We start with the case of a single token and later expand to multiple dimensions.
We define the forward process as a Markovian sequence of discrete random variables $x_{t}$ indexed by time $t$, where $t$ runs from 0 to 1.
Throughout the work, we abuse the notation such that $x_{t}$ can be either an integer or its corresponding one-hot vector, whenever it is clear from the context.
We divide $[0,1]$ into $T$ intervals, and let $s(i)=(i-1)/T$, $t(i)=i/T$.
Following , the state transition between $[s(i),t(i)]$ is determined by a transition matrix of size $(m+1)\times(m+1)$:
$Q_{i}=(1-\beta_{i})I+\beta_{i}\mathbf{1}e_{m}^{\top},$
where $\mathbf{1}$ is an all-one vector of size $m+1$, $e_{m}$ represents a one-hot vector where element at index $m$ is 1.
Each entry $[Q_{i}]_{jk}$ denotes the probability of transition from the state $j$ to the state $k$:

$$ $\displaystyle[Q_{i}]_{jk}=q(x_{t(i)}=k|x_{s(i)}=j)=(1-\beta_{i})\delta_{jk}+ \beta_{i}\delta_{km}.$ $$

This means that, with probability $1-\beta_{i}$, $x_{t(i)}=x_{s(i)}$, otherwise it jumps to the mask state.
Given the above transition matrix, the marginal distribution at time $t(i)$ given $x_{0}$ is

$$ $\displaystyle q(x_{t(i)}|x_{0})=\mathrm{Cat}(x_{t(i)};\bar{Q}_{i}^{\top}x_{0}) =x_{0}^{\top}\bar{Q}_{i}x_{t(i)}.$ $$

Here, we use $\mathrm{Cat}(x;p)$ to denote a Categorical distribution where $p$ is the vector of probabilities of being in each category, and $\bar{Q}_{i}\triangleq\prod_{j=1}^{i}Q_{j}=\alpha_{i}I+\big{(}1-\alpha_{i}\big{
)}\mathbf{1}e_{m}^{\top}$ for $\alpha_{i}=\prod_{j=1}^{i}(1-\beta_{j})$.
We expect $\alpha_{T}$ to become very small or zero for a sufficiently large $T$ such that $q(x_{1}|x_{0})$ for any $x_{0}$ will become a delta mass at the mask state.

##### Continuous-time limit.

We can define a continuous-time forward process by taking a limit of the above discrete-time process. We first specify a continuous function $\beta(t)$ such that
$\beta_{i}=\beta(t(i))/T$.
We then let $T\to\infty$ in the discrete-time process and compute the limit of $\bar{Q}_{i}$ (proved in , see also App. A) as

$$ $\displaystyle\bar{Q}(t)\triangleq\lim_{T\to\infty}\bar{Q}_{i}$ $\displaystyle=\alpha_{t}I+(1-\alpha_{t})\mathbf{1}e_{m}^{\top},\text{ where } \alpha_{t}\triangleq\exp\Big{(}-\int_{0}^{t}\beta(s)\mathrm{d}s\Big{)},$ (1) $$

so that $q(x_{t}|x_{0})=\mathrm{Cat}(x_{t};\bar{Q}(t)^{\top}x_{0})$.
For two arbitrary times, $0\leq s<t\leq 1$, the transition distribution that is compatible with the above marginal (i.e., $q(x_{t}|x_{0})=\sum_{x_{s}}q(x_{t}|x_{s})q(x_{s}|x_{0})$) is

$$ $\displaystyle q(x_{t}|x_{s})$ $\displaystyle=\mathrm{Cat}(x_{t};\bar{Q}(s,t)^{\top}x_{s}),\text{ where }\bar{ Q}(s,t)\triangleq\bar{Q}(s)^{-1}\bar{Q}(t)=\frac{\alpha_{t}}{\alpha_{s}}I+\big {(}1-\frac{\alpha_{t}}{\alpha_{s}}\big{)}\mathbf{1}e_{m}^{\top}.$ $$

Note that did not derive this explicit form of transition matrix between two arbitrary time $s$ and $t$, which appeared later in concurrently with our work.

Figure: Figure 1: Masking schedules in the literature: (Left) $\alpha_{t}$; (Right) weight of the cross-entropy loss w.r.t. $t$; Equations for these schedules are given in Tab. 4 in Appendix.
Refer to caption: x1.png

##### Masking schedules.

From the definition of $\alpha_{t}$, we have that $\alpha_{0}=1$.
And similar to the discrete-time formulation, we would like $\alpha_{1}$ be zero or very close to zero.
We provide a summary of masking schedules from literature that satisfy these properties in Fig. 1.
The linear schedule was proposed in for binary variables and then re-derived by from mutual information for discrete-time models. The geometric schedule $\alpha_{t}$
is plotted for $\bar{\beta}_{\text{min}}=10^{-5}$ and $\bar{\beta}_{\text{max}}=20$. It was first used for continuous diffusions and then for discrete by .
The cosine schedule was originally proposed in MaskGIT , an iterative unmasking generative model inspired by diffusion.
This schedule has the property of slowing down the unmasking process at the beginning of the reverse generation.
Aligning with their observation, we find that this results in a lower chance of conflicting tokens being unmasked simultaneously at the start of generation, thereby enhancing the overall generation quality.

##### Time reversal of the forward process given x 0 subscript 𝑥 0 x_{0} italic_x start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT .

The analytic property of our forward process allows to compute many quantities of interest in closed form.
One such quantity frequently used in diffusion models is the time reversal of the forward process given $x_{0}$: $q(x_{s}|x_{t},x_{0})$ for $s\leq t$.
We derive it in App. C as

$$ $\displaystyle q(x_{s}|x_{t},x_{0})=\mathrm{Cat}(x_{s};\bar{R}^{x_{0}}(t,s)^{ \top}x_{t}),\text{ where }\bar{R}^{x_{0}}(t,s)=I+\frac{\alpha_{s}-\alpha_{t}}{ 1-\alpha_{t}}e_{m}(x_{0}-e_{m})^{\top}.$ $$

From the transition matrix $\bar{R}^{x_{0}}(t,s)\in\mathbb{R}^{(m+1)\times(m+1)}$
we can see the reverse process conditioned on $x_{0}$ has a very simple logic—if $x_{t}$ is a mask, with probability $\frac{\alpha_{s}-\alpha_{t}}{1-\alpha_{t}}$, it will jump to the state $x_{0}$ at time $s$, otherwise it will stay masked. Once $x_{t}$ is unmasked, it remains in the same state until the end.

## 3 Model and Objective

For a discrete-time masked diffusion process,
we define our generative model by approximately reversing the forward transitions using a reverse model $p_{\theta}(x_{s}|x_{t})$.
One way to define this model is

$$ $\displaystyle p_{\theta}(x_{s}|x_{t})\triangleq q(x_{s}|x_{t},\mu_{\theta}(x_{ t},t)),$ (2) $$

where $\mu_{\theta}(x_{t},t)\in\Delta^{m+1}$ is a probability vector parametrized by a neural network $f_{\theta}$ with a softmax applied to the output logits (note the $m$-th output is forced to 0 since the clean data cannot be masks):

$$ $\displaystyle\mu_{\theta}(x_{t},t)=\begin{cases}\mathrm{softmax}(f_{\theta}(x_ {t},t))&x_{t}=m,\\ x_{t}&x_{t}\neq m.\end{cases}$ (3) $$

This is known as mean-parameterization since it leverages a prediction model for the mean of $x_{0}$. A matrix-form depiction of $p_{\theta}(x_{s}|x_{t})$ is shown in Fig. 7 (right).
In fact, we can select a time-invariant parametrization $\mu_{\theta}(x_{t},t)=\mu_{\theta}(x_{t})$ as showed that $p(x_{0}|x_{t})$ given $x_{t}=x$ is identical for any $t$.

Besides $p_{\theta}(x_{s}|x_{t})$, we also need to specify $p(x_{0}|x_{t(1)})$ and the prior distribution $p(x_{t(T)})=p(x_{1})$.
Following the practice in continuous diffusion models , we choose $p(x_{0}|x_{t(1)})\propto q(x_{t(1)}|x_{0})$.
And since $q(x_{1}|x_{0})\approx\delta_{x_{1},m}$ for any $x_{0}$ as $\alpha_{1}\approx 0$, we set $p(x_{1})\approx\delta_{x_{1},m}$, see App. E.

We then write out the discrete-time diffusion model objective , which is a lower bound of the log marginal likelihood of data $x_{0}$ under the model $p$ (known as the Evidence Lower Bound, or ELBO):

$$ $\displaystyle\log p(x_{0})$ $\displaystyle\geq\mathbb{E}_{q(x_{t(1)}|x_{0})}[\log p(x_{0}|x_{t(1)})]- \mathrm{KL}({q(x_{1}|x_{0})}\|{p(x_{1})})-\mathcal{L}_{T},$ $$

where $\mathcal{L}_{T}=\sum_{i=2}^{T}\mathbb{E}_{q(x_{t(i)}|x_{0})}[\mathrm{KL}({q(x_
{s(i)}|x_{t(i)},x_{0})}\|{p_{\theta}(x_{s(i)}|x_{t(i)})})]$.
For the above choices of the prior distribution, the term $\mathrm{KL}({q(x_{1}|x_{0})}\|{p(x_{1})})$ becomes zero.
Under the reverse model (2), the KL divergence terms in $\mathcal{L}_{T}$ becomes (proof in App. D)

$$ $\displaystyle\mathrm{KL}({q(x_{s}|x_{t},x_{0})}\|{p_{\theta}(x_{s}|x_{t})})=- \frac{\alpha_{s}-\alpha_{t}}{1-\alpha_{t}}\delta_{x_{t},m}\cdot x_{0}^{\top} \log\mu_{\theta}(x_{t},t),$ $$

which is a simple cross-entropy loss between the predicted logits and the clean data.
In App. D, we show that $\mathcal{L}_{T}$ is a Riemann sum and is lower bounded by the corresponding continuous integral:

$$ $\displaystyle\mathcal{L}_{\infty}$ $\displaystyle\triangleq\lim_{T\to\infty}\mathcal{L}_{T}=\int_{t(1)}^{1}\frac{ \alpha_{t}^{\prime}}{1-\alpha_{t}}\mathbb{E}_{q(x_{t}|x_{0})}\left[\delta_{x_{ t},m}\cdot x_{0}^{\top}\log\mu_{\theta}(x_{t},t)\right]\mathrm{d}t,$ (4) $$

where $\alpha^{\prime}_{t}$ denotes the derivative of $\alpha_{t}$ with respect to $t$.
Therefore, we can obtain an ELBO that is tighter than that of any finite $T$ by pushing $T\to\infty$.
This ELBO can be further simplified by letting $t(1)\to 0$.
As a result, $\mathbb{E}_{q(x_{t(1)}|x_{0})}[\log p(x_{0}|x_{t(1)})]$ goes to 0 0 and the ELBO becomes $-\mathcal{L}_{\infty}$.

For continuous state-space diffusions, the ELBO depends on the signal-to-noise ratio (SNR) at its endpoints but is otherwise invariant to the noise schedule . We establish here a similar result for discrete diffusions.
Consider choosing $\alpha_{t}=\sigma(\lambda_{t})$, where $\sigma$ represents the sigmoid function $\sigma(x)=\frac{1}{1+e^{-x}}$.
In this context, the log-SNR is defined by $\lambda_{t}=\log\frac{\alpha_{t}}{1-\alpha_{t}}=\textup{log-SNR}(t)$.
By making a change of variables in (4) to make everything a function of the log-SNR, we obtain

$$ $\mathcal{L}_{\infty}=\int_{\lambda_{t(1)}}^{\lambda_{1}}\sigma(\lambda)\mathbb {E}_{\tilde{q}(x_{\lambda}|x_{0})}\left[\delta_{x_{\lambda},m}\cdot x_{0}^{ \top}\log\tilde{\mu}_{\theta}(x_{\lambda},\lambda)\right]\mathrm{d}\lambda.$ $$

where $\tilde{\mu}_{\theta}(x,\lambda):=\mu_{\theta}(x,t)$ and $\tilde{q}(x_{\lambda}|x_{0}):=q(x_{t}|x_{0})$ for $t=\textup{log-SNR}^{-1}(\lambda)$. This shows that the only effect $\alpha_{t}$ has on the loss is through the values of the SNR at the endpoints.
Still, because we draw uniform samples of $t$ to estimate the integral, the choice of masking schedule affects the variance.

##### Multidimensional data.

In the previous sections, $x_{t}$ was assumed to be a single discrete token.
To extend the method to multidimensional data, let $x_{t}$ be now a sequence $(x_{t}^{(1)},x_{t}^{(2)},\dots,x_{t}^{(N)})$,
where each element $x_{t}^{(n)}$
represents a discrete token. We select a forward process which factorizes across all
$N$ tokens: $q(x_{t}|x_{s})=\prod_{n=1}^{N}q(x_{t}^{(n)}|x_{s}^{(n)})$.
As a result, the forward marginals $q(x_{t}|x_{0})$ and reversal $q(x_{s}|x_{t},x_{0})$ also factorize.
In this case, we define the reverse model as
$p_{\theta}(x_{s}|x_{t})\triangleq\prod_{n=1}^{N}q(x_{s}^{(n)}|x_{t}^{(n)},\mu_
{\theta}^{(n)}(x_{t},t))$, where $\mu_{\theta}(x_{t},t)$ is a neural network that takes the full $N$ tokens as input and
outputs $N$ probability vectors.(^1^11We intentionally choose the reverse model to factorize across dimensions because the true reverse transition $q(x_{s}|x_{t})$ factorizes in the continuous-time limit (as $s$ approaches $t$).)
The $n$-th output $\mu_{\theta}^{(n)}(x_{t},t)$ is a prediction model for $\mathbb{E}[x_{0}^{(n)}|x_{t}]$, the mean value of the $n$-th token.
Repeating above derivations gives

$$ $\displaystyle\mathcal{L}_{\infty}^{(N)}$ $\displaystyle\triangleq\int_{0}^{1}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}} \mathbb{E}_{q(x_{t}|x_{0})}\Big{[}{\textstyle\sum}_{n:x_{t}^{(n)}=m}(x_{0}^{(n )})^{\top}\log\mu_{\theta}^{(n)}(x_{t},t)\Big{]}\mathrm{d}t.$ (5) $$

We term our simple masked diffusion model trained with loss (5) MD4 (Masked Discrete Diffusion for Discrete Data).
A single step of MD4 training algorithm is described in Alg. 1 in Appendix.

## 4 Sampling

We use ancestral sampling from our discrete-time reverse process for generation. We have found this yields slightly higher sample quality compared to other methods such as Euler discretization .
For conditional generation tasks such as infilling, we find that the simple approach works best — we keep the conditioning tokens unmasked throughout the generation process.
A complete description of the sampling algorithm can be found in Alg. 2 in Appendix.

##### Impact of schedules and discretization.

For comparing different sampling configurations, we primarily use the FID score  on image datasets as our evaluation metric. We favor it over text generative perplexity(^2^22Perplexity of generated samples scored by a large language model such as GPT-2.) used in prior work , as the latter can be misleadingly reduced by lowering sample diversity .
We initially trained our model using the linear schedule, which achieves the best final ELBO overall; however, we found that sampling did not perform well with a standard uniform discretization grid $t(i)=\frac{i}{T}$.
We hypothesize that time discretization can lead to conflicts by generating multiple tokens in a single step.
We then switched to the cosine schedule (Tab. 4) that slows down unmasking at the beginning of reverse process.
This drastically improves the FID on ImageNet 64$\times$64 from 70 to 17 for $T=256$ steps (Fig. 2, left).
Building on this observation, we suggest using a “cosine” discretization grid for sampling in models trained with a linear schedule:

$$ $\displaystyle t(i)=\cos\Big{(}\frac{\pi}{2}\big{(}1-\frac{i}{T}\big{)}\Big{)}.$ (6) $$

This induces the same discretization in $\alpha_{t}$ as the cosine schedule with a uniform grid, leading to comparable sample quality, as shown in Fig. 2 (left). In Fig. 2 (right), we plot the number of tokens unmasked per step for linear and cosine schedules with a uniform grid. We believe the cosine schedule performs better because it leverages information redundancy: with more tokens revealed, the remaining tokens become more predictable, reducing conflicts when unmasking them in a single step.

Figure: Figure 2: Left: FID evaluation for 50k samples randomly generated from MD4 on pixel-level modeling of ImageNet 64$\times$64 (numbers in Tab. 6). Right: Number of tokens revealed per generation step ($T=256$). Each image consists of $64\times 64\times 3=12288$ tokens.
Refer to caption: x2.png

Although these findings were originally developed on images, we find them translate well to text (see Fig. 10).
we expect other techniques such as top-$p$ sampling , classifier-free guidance , and predictor-correctors  to further improve sample quality of our models.
While we reserve these for future work, we note that the JAX  implementation of categorical sampling implicitly truncates small probabilities, creating a similar effect to top-$p$ sampling. See App. G for details.

## 5 Relation to Existing Work

We discuss how to unify several existing masked diffusion models using our framework.

##### Continuous-Time Markov Chains (CTMC).

To show the connection with the CTMC view presented in ,
we can write out the forward and reverse masked diffusion using CTMC machinery.
To see this, for a short time $\Delta t$,
given $x_{0}$, the Taylor expansions of our forward and reverse transition matrices at $t$ are

$$ $\displaystyle\bar{Q}(t,t+\Delta t)$ $\displaystyle=I+Q(t)\Delta t+o(\Delta t)\text{\quad for\quad}Q(t)\triangleq \beta(t)(\mathbf{1}e_{m}^{\top}-I),$ (7) $\displaystyle\bar{R}^{x_{0}}(t,t-\Delta t)$ $\displaystyle=I+R^{x_{0}}(t)\Delta t+o(\Delta t)\text{\quad for\quad}R^{x_{0}} (t)\triangleq-\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}e_{m}(x_{0}-e_{m})^{\top},$ (8) $$

where $Q(t)$ and $R^{x_{0}}(t)$ are known as the *transition rate* matrices.
derived the same $Q(t)$ in App. A.6 of their paper.
However, they did not explore the reverse process or a continuous-time objective.
derived an alternative ELBO expression using rate matrices, which further simplified for absorbing diffusion.
In Sec. H.1, we show how to recover their expression
by separating out a constant from our ELBO expression (4) and applying a discrete “integration-by-part”.
A key limitation of their expression
is that it needs $N$ evaluations of the prediction model $\mu_{\theta}(\cdot,t)$ to compute an inner summation. To circumvent this computational burden, they used a doubly stochastic estimate. However, this leads to significantly higher variance compared to the analytic cross-entropy (4) which only requires one pass of $\mu_{\theta}(\cdot,t)$.
Please refer to Sec. H.2 for more details.

##### Score parameterization.

While so far we used a prediction model $\mu_{\theta}(x_{t},t)$ for the mean of clean data given $x_{t}$ (i.e., mean parameterization), one can choose other ways of parameterizing the reverse model.
proposed to parameterize the discrete “score” $s(x_{t},t)_{j}\triangleq\frac{q_{t}(j)}{q_{t}(x_{t})}$
and introduced a score-based loss for discrete diffusions.
In Sec. H.3, we provide an alternative derivation of their loss
which is simpler.
We show the link between score and mean parameterizations through the following proposition.

###### Proposition 1 (Score Parameterization vs. Mean Parameterization) .

Let $q_{t}$ be the marginal distribution of the masked diffusion defined in Sec. 2 at time $t$. The discrete score $s(x_{t},t)_{j}=\frac{q_{t}(j)}{q_{t}(x_{t})}$ for a mask state $x_{t}=m$ and $j\neq m$ can be expressed as

$$ $\displaystyle s(m,t)_{j}=\frac{\alpha_{t}}{1-\alpha_{t}}\mathbb{E}[x_{0}|x_{t} =m]^{\top}e_{j}\text{, which satisfies }\sum_{j\neq m}s(m,t)_{j}=\frac{\alpha_ {t}}{1-\alpha_{t}}.$ (9) $$

Proposition 1 (proved in Sec. H.3) implies that a reasonable score model for a mask state is

$$ $\displaystyle s_{\theta}(m,t)_{j}=\frac{\alpha_{t}}{1-\alpha_{t}}\mu_{\theta}( m,t)_{j}.$ (10) $$

Indeed, substituting (10) into the score-based loss of
recovers our objective (4).
In , the score is parameterized as a neural network without enforcing the constraint in (9).
This means the learned reverse model can be incompatible with the forward process.
We find that our parameterization, which enforces the constraint, leads to more stable training and better results.

###### Proposition 1 (Score Parameterization vs. Mean Parameterization) .

##### Any-order autoregressive models.

The continuous-time reverse process of our masked diffusion model can be viewed as an any-order autoregressive model (AO-ARM) .
To see this, we reorder the tokens according to the timing of their unmasking events in the reverse process.
For all tokens, the cumulative distribution functions (CDFs) of unmasking times $\{\tau_{n}\}_{n=1}^{N}$ are identical and satisfy
$P(\tau_{n}\leq t)=P(x_{t}^{(n)}=m)=1-\alpha_{t}$.
As a result, the ordering is uniformly random across all possible arrangements, and the token prediction during each unmasking event represents a prediction step in AO-ARMs.
This connection was initially pointed out in .
The relation between our simplified ELBO (5) and the AO-ARM objective is independently clarified by .
Despite this equivalence, our work
demonstrates that the masking schedule $\alpha_{t}$ introduces a new degree of freedom in the design of such models.
Variations in $\alpha_{t}$ can lead to different distributions of unmasking times, significantly impacting performance in diffusion-style parallel sampling under time discretization, as shown in Fig. 2.

##### Other related work.

Due to space constraint, we defer the discussion on other related work, including MaskGIT , discrete flow matching , SDDM , Blackout diffusion  and SUNDAE , to Sec. H.4.

## 6 Generalization to State-dependent Masking Schedules

Consider a scenario where some tokens hold more significance than others and we would like to unmask them earlier in the process. To achieve this, we introduce state-dependent masking schedules, where the probability of unmasking a token depends not only on time, but also on the token’s value.

We first define the forward process for a single token $x_{t}$.
Let $\alpha_{t}$ be a $m+1$ dimensional vector function, i.e.,
there is a different function $\alpha_{t,i}$
for each possible value $i$ of the token $x_{t}$. Also, by vector $\frac{\alpha_{t}}{\alpha_{s}}$ we denote the element-wise division
of the two vectors.
We define the forward transition as
$q(x_{t}|x_{s})=\mathrm{Cat}(x_{t};\bar{Q}(s,t)^{\top}x_{s})$ where

$$ $\displaystyle\bar{Q}(s,t)=\mathrm{diag}\Big{(}\frac{\alpha_{t}}{\alpha_{s}} \Big{)}+\Big{(}I-\mathrm{diag}\Big{(}\frac{\alpha_{t}}{\alpha_{s}}\Big{)}\Big{ )}\mathbf{1}e_{m}^{\top}$ $$

and $\mathrm{diag}\big{(}\frac{\alpha_{t}}{\alpha_{s}}\big{)}$ is a diagonal matrix with the vector $\frac{\alpha_{t}}{\alpha_{s}}$ in its diagonal.
The probability of moving from current state $x_{s}$ to a future state $x_{t}$ (either the same as $x_{s}$ or mask) is determined by a state-dependent rate $\big{(}\frac{\alpha_{t}}{\alpha_{s}}\big{)}^{\top}x_{s}$, while
the marginal at time $s$ given $x_{0}$ is

$$ $q(x_{s}|x_{0})=\mathrm{Cat}(x_{s};\bar{Q}(s)^{\top}x_{0})\text{\quad for\quad} \bar{Q}(s)=\mathrm{diag}(\alpha_{s})+(I-\mathrm{diag}(\alpha_{s}))\mathbf{1}e_ {m}^{\top}.$ $$

Further, for any time
$0\leq s<t\leq 1$ it holds that $q(x_{t}|x_{0})=\sum_{x_{s}}q(x_{t}|x_{s})q(x_{s}|x_{0})$
so the above is a valid continuous-time Markov chain.

Given the forward conditionals and marginals, we can now compute the time reversal conditioned on $x_{0}$.
The full form of $q(x_{s}|x_{t},x_{0})$ is derived in Sec. I.1.
For $x_{t}=m$, we have

$$ $\textstyle q(x_{s}|x_{t}=m,x_{0})=q(x_{s}|x_{t}=m,{\color[rgb]{0,0,1} \definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}x_{0}},{\color[rgb]{1,0,0} \definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}x_{0}x_{0}^{\top}})=\Big{(} \frac{{\bf 1}-\alpha_{s}}{{\bf 1}-\alpha_{t}}\Big{)}^{\top}{\color[rgb]{0,0,1} \definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}x_{0}}e_{m}^{\top}x_{s}+\Big{(} \frac{\alpha_{s}-\alpha_{t}}{{\bf 1}-\alpha_{t}}\Big{)}^{\top}{\color[rgb]{ 1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}x_{0}x_{0}^{\top}}x_{s}.$ (11) $$

This suggests that the reverse model
given $x_{t}=m$ can be chosen as
$p_{\theta}(x_{s}|x_{t}=m)\triangleq q(x_{s}|x_{t}=m,{\color[rgb]{0,0,1}
\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mu_{\theta}(x_{t},t)},{\color[
rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\mathrm{diag}(\mu_{
\theta}(x_{t},t))})$
where $\mu_{\theta}(x_{t},t)$ is a neural network
that approximates $\mathbb{E}[x_{0}|x_{t}]$ while $\mathrm{diag}(\mu_{\theta}(x_{t},t))$ approximates $\mathbb{E}[x_{0}x_{0}^{\top}|x_{t}]=\mathrm{diag}(\mathbb{E}[x_{0}|x_{t}])$.
We show in Sec. I.1 that the negative continuous-time ELBO
for the state-dependent rate case is

$$ $\displaystyle\mathcal{L}_{\infty}=\int_{0}^{1}\Big{(}\frac{\alpha_{t}^{\prime} }{\mathbf{1}-\alpha_{t}}\Big{)}^{\top}\mathbb{E}_{q(x_{t}|x_{0})}\left[\delta_ {x_{t},m}\cdot(x_{0}-\mu_{\theta}(x_{t},t)+x_{0}x_{0}^{\top}\log\mu_{\theta}(x _{t},t))\right]\mathrm{d}t.$ (12) $$

Here, $\alpha_{t}^{\prime}$ is the elementwise derivative of $\alpha_{t}$.
This generalizes the MD4 loss (4), which is recovered when $\alpha_{t}$ is a scalar schedule times a vector of ones.
For $N$ tokens, the model further generalize similarly to
Sec. 3 and the loss is given in (32).
We call this generalized model GenMD4.

Figure: Figure 3: Iterative unmasking process for an unconditionally generated sample by MD4. This visualization only includes a subsequence from a generated sequence of 1024 tokens. "?" represents masks. Masked tokens are revealed sequentially: green (steps 500-700), yellow (700-850), and red (850-1000). Additional unconditional generation from MD4 can be found in Sec. K.5.
Refer to caption: x4.png

To learn the token dependent masking schedule using ELBO optimization, we parametrize the $m+1$ dimensional function $\alpha_{t}$ using the polynomial schedule (see Fig. 1) as
$\alpha_{t,i}=1-t^{w_{i}}$
and optimize each parameter $w_{i}>0$.(^3^33We only need $m$ learnable parameters $w_{i}$, for $i=0,\ldots,m-1$, since $x_{0}$ can never be the mask token. For the final mask dimension we can choose an arbitrary fixed value such as $w_{m}=0$.) The value of $w_{i}$, through the
masking probability $1-\alpha_{t,i}$, determines how fast the token with value $i$ jumps to the mask state.
Since in the loss (12) the distribution
$q(x_{t}|x_{0})$ depends on $\alpha_{t}$ and thus the vector $w$, optimizing $w$ poses a discrete gradient estimation problem .
Naive autodiff leads to biased gradients and pushes $w$ towards zero because the gradients cannot propagate through the (discrete) samples drawn from $q(x_{t}|x_{0})$.
To fix this, we used the REINFORCE leave-one-out estimator  to compute low-variance unbiased gradients for optimizing $w$.
Details are given in Sec. I.2.

## 7 Experiments

### 7.1 Text

Text is natural discrete data with rich structures.
For comparison with prior work, we evaluate likelihood on two datasets:
text8 , a character-level text modeling benchmark, and OpenWebText , an open clone of the unreleased WebText dataset used to train GPT-2 .
We also assess our model’s performance on downstream tasks by training on
FineWeb-Edu , a high-quality dataset of fine educational text commonly used by the open-source community for comparing LLMs.
Unless otherwise specified, a linear schedule and a cosine sampling grid are employed.

**Table 1: Zero-shot unconditional perplexity on five benchmark datasets from . The numbers for other methods are from except our reimplementation of SEDD Absorb. Our MD4 model achieves the best result on all benchmarks except LAMBADA where it is the second best. ^∗The GPT-2 numbers are reported for the GPT-2 checkpoint pretrained on WebText instead of OWT thus is not a direct comparison.**
| Size | Method | LAMBADA | WikiText2 | PTB | WikiText103 | IBW |
| --- | --- | --- | --- | --- | --- | --- |
| Small | GPT-2 (WebText)^∗ | 45.04 | 42.43 | 138.43 | 41.60 | 75.20 |
|  | D3PM | $\leq$ 93.47 | $\leq$ 77.28 | $\leq$ 200.82 | $\leq$ 75.16 | $\leq$ 138.92 |
|  | Plaid | $\leq$ 57.28 | $\leq$ 51.80 | $\leq$ 142.60 | $\leq$ 50.86 | $\leq$ 91.12 |
|  | SEDD Absorb | $\leq$ 50.92 | $\leq$ 41.84 | $\leq$ 114.24 | $\leq$ 40.62 | $\leq$ 79.29 |
|  | SEDD Absorb (reimpl.) | $\leq$ 49.73 | $\leq$ 38.94 | $\leq$ 107.54 | $\leq$ 39.15 | $\leq$ 72.96 |
|  | MD4 (Ours) | $\leq$ 48.43 | $\leq$ 34.94 | $\leq$ 102.26 | $\leq$ 35.90 | $\leq$ 68.10 |
| Medium | GPT-2 (WebText)^∗ | 35.66 | 31.80 | 123.14 | 31.39 | 55.72 |
|  | SEDD Absorb | $\leq$ 42.77 | $\leq$ 31.04 | $\leq$ 87.12 | $\leq$ 29.98 | $\leq$ 61.19 |
|  | MD4 (Ours) | $\leq$ 44.12 | $\leq$ 25.84 | $\leq$ 66.07 | $\leq$ 25.84 | $\leq$ 51.45 |

Figure: Figure 4: Perplexity on OpenWebText (OWT) validation set during training. The final numbers are reported in Tab. 5 in Appendix.
Refer to caption: x5.png

##### OpenWebText.

We train MD4 of GPT-2 small (S) and GPT-2 medium (M) sizes on OpenWebText and evaluate zero-shot perplexity on
five benchmark datasets used in .
We keep our evaluation setup the same as SEDD .
To ensure fair comparison, we reimplemented SEDD in our codebase.
Our implementation led to slightly better results than those
reported in their paper.

As seen in
LABEL:tab:owt-zeroshot-ppl, our small model outperforms previous best discrete diffusion models on all five tasks.
We are also better than GPT-2 on all tasks except LAMBADA where we are the second best method.
When scaling up to medium size, MD4 similarly beats SEDD and GPT-2 on 4 out of 5 tasks.

To confirm that the strong zero-shot performance stems from improved training, we plot perplexity on 2% OpenWebText validation set in Fig. 4.
Our models converge faster and have better final likelihoods than prior methods.
We also observed that SEDD  has training instabilities, likely due to score parameterization breaking consistency between forward and reverse processes (Sec. 5).
Although GenMD4 achieves lower perplexity than MD4, we observed that the learned $w$s can overfit to dataset statistics, making it less effective on zero-shot transfer tasks.

We also assess our models’ generation quality.
Fig. 3 shows a randomly selected, notably coherent sample from MD4-medium and its denoising process.
Fig. 10 demonstrates MD4’s text infilling ability and highlights a substantial quality gain when transitioning from uniform to cosine discretization (see Sec. 4).
Despite MD4’s strong performance on quantitative metrics like generative perplexity, we have placed these results in Appendix Fig. 8 due to the metric’s inherent unreliability, as noted in Sec. 4.
We emphasize the more reliable FID-based assessments found in our image experiments.

##### Text8.

Following prior work , we trained masked diffusion models on text8 and evaluate the bits-per-character on the test set
(details in Sec. J.1). As seen in Tab. 3, our models
outperform previous discrete and continuous diffusion models, as well as state-of-the-art AO-ARMs which are closely related to discrete diffusion .
Our model is only beaten by an autoregressive (AR) transformer and the AR-backbone Discrete Flow .
We believe this is because AR models only require learning a fixed generation order thus better utilize model capacity.
Text8’s small vocabulary (26 letters and a space)
led us to expect limited flexibility from our state-dependent formulation. However, using the generalized objective in (12), GenMD4 achieved significantly better BPC than MD4, demonstrating the potential of state-dependent diffusion for discrete data.

Figure: Figure 5: Hellaswag accuracy vs. training steps for MD4 and AR models at GPT-2 small, medium, and large scales.
Refer to caption: x6.png

##### FineWeb-Edu.

We train MD4 on FineWeb-Edu and evaluate its zero-shot accuracy on the Hellaswag dataset , a popular common sense inference benchmark for LLMs. We directly compared MD4 to its AR counterparts – transformers with identical configurations (except for causal masking) trained on the same data. Results are summarized in Fig. 5.

MD4 demonstrates steady performance growth with increasing scale. While outperformed by AR models of the same size, the performance gap does not widen as model size increases. For example, AR-small reaches 30% accuracy in 50k steps, while MD4-small takes 200k steps (4x data efficiency difference). At the medium scale, AR achieves 37% in 270k steps, compared to MD4’s 1 million steps.

**Table 2: Bits Per Character (BPC) on Text8 test set. All models use standard 12-layer transformers similar to GPT-2 small except Discrete Flow which uses $8\times 3$ layers.**
| Method | BPC ($\downarrow$) |
| --- | --- |
| Continuous Diffusion |  |
| Plaid (Our impl.) | $\leq$ 1.48 |
| BFN | $\leq$ 1.41 |
| Any-order Autoregressive |  |
| ARDM | $\leq$ 1.43 |
| MAC | $\leq$ 1.40 |
| Autoregressive |  |
| IAF/SCF | 1.88 |
| AR Argmax Flow | 1.39 |
| Discrete Flow | 1.23 |
| Transformer AR | 1.23 |
| Discrete Diffusion |  |
| Mult. Diffusion | $\leq$ 1.72 |
| D3PM Uniform | $\leq$ 1.61 |
| D3PM Absorb | $\leq$ 1.45 |
| SEDD Absorb | $\leq$ 1.39 |
| MD4 (Ours) | $\leq$ 1.37 |
| GenMD4 (Ours) | $\leq$ 1.34 |

### 7.2 Pixel-level image modeling

Figure: Figure 6: Non cherry-picked unconditional samples from MD4 trained on ImageNet 64x64, treating pixels as discrete tokens. More samples can be found in Fig. 9 in Appendix. The model is optimized for likelihood instead of visual quality—see e.g., for samples from a continuous diffusion model optimized similarly for likelihood.
Refer to caption: extracted/6135574/figures/imagenet/imagenet_ancestral_01.png

Unlike continuous diffusion which struggles with discrete data, we show that MD4, a discrete diffusion model, performs well on inherently continuous data, suggesting its potential for unifying modalities.
We follow and train MD4 on order-agnostic image data from CIFAR-10 and downsampled ImageNet 64$\times$64 .
Each image is treated as a set of 256-valued discrete tokens, making the model agnostic to pixel proximity.
We compare to other discrete diffusion and AR models with reported likelihood results on these datasets, although to our knowledge there are no published result on discrete diffusion for ImageNet $64\times 64$ that directly model raw pixel space.

Tab. 3 summarizes our results.
We establish a new state-of-the-art for discrete diffusion models, outperforming previous work by a significant margin.
Our CIFAR-10 result surpasses the best reported AR result.
On ImageNet $64\times 64$, our results are competitive with Transformer AR models that are 4$\times$ larger, as well as a strong continuous diffusion model VDM .
Notably, despite lacking knowledge of the ordinal structure of pixel values, MD4 outperforms models trained with this inductive bias, including D3PM Gauss and where the noising distribution is a discrete Gaussian that assigns larger probabilities to near pixel values.
To isolate the differences caused by training objectives, we also implemented the objective with the absorbing process, showing its high variance hinders learning even with our architecture.

We provide a random sample from our ImageNet 64$\times$64 model in Fig. 6.
More results can be found in App. K.
In Fig. 2, we plot the FID values of samples generated under different choices of schedules and discretization grids.
We can see that the model with the linear schedule plus a cosine grid achieves an FID close to the model with cosine schedule, both significantly outperform the linear schedule with a uniform grid.
We further trained a class-conditional model on ImageNet 64$\times$64 that boosts the FID to around 7.
Although these are not state-of-the-art FIDs on ImageNet 64$\times$64, we emphasize our models are optimized for likelihood instead of sample quality.

## 8 Conclusion

In this work, we revisit masked diffusion models, focusing on a flexible continuous-time formulation. Existing works in this area are not easily accessible to non-specialists and present ELBOs that are difficult to optimize, often resulting in performance that is not competitive with continuous diffusions and AR models. The framework we propose provides a very simple expression of the ELBO as a weighted integral of cross-entropy losses. Additionally, we propose a generalized masked diffusion formulation (GenMD4), where the masking schedule depends on the current state of the process, and derive its corresponding ELBO. On text data, our MD4 models outperform existing discrete and continuous diffusion models. For pixel-level image modeling, we significantly improve discrete diffusion results, outperforming similar-sized AR models and achieving comparable likelihoods to continuous diffusion models such as VDM.
GenMD4 provides further improvements in terms of likelihoods over the state-independent case.

Although we have improved masked diffusion models, they still suffer from limitations. First, in some tasks such as text8, masked diffusions are not yet competitive with AR models. We conjecture that this is because AR models can better leverage model capacity since they only require learning one order. It would be interesting to develop better architectures for discrete diffusions. Moreover, GenMD4 is promising, but it can easily overfit to the dataset, making it less effective for zero-shot transfer compared to simpler versions. Additionally, inference with a state-dependent schedule is more challenging.

## Appendix A Discrete-time derivation

We divide time from 0 to 1 into $T$ intervals, and let $s(i)=(i-1)/T$, $t(i)=i/T$.
The forward transition matrix $Q_{i}\in\mathbb{R}^{(m+1)\times(m+1)}$ ($m$ is vocabulary size) at time $t(i)$ is

$$ $\displaystyle[Q_{i}]_{jk}=\begin{cases}1&j=k=m\\ 1-\beta_{i}&j=k\neq m\\ \beta_{i}&k=m,j\neq m\\ 0&\text{otherwise}\end{cases}$ $$

or more compactly written as

$$ $\displaystyle Q_{i}=(1-\beta_{i})I+\beta_{i}\mathbf{1}e_{m}^{\top},$ $$

where $\mathbf{1}$ denotes an all-one vector of size $m+1$, and $e_{m}$ is an one-hot vector of size $m+1$ with the $m$-th element (recall that counting starts from 0 0) being one.
We use an one-hot vector $x_{t}$ of length $m+1$ to denote the discrete state.
The forward conditionals are defined as

$$ $\displaystyle q(x_{t(i)}|x_{s(i)})=\mathrm{Cat}(x_{t(i)};Q_{i}^{\top}x_{s(i)}) =x_{s(i)}^{\top}Q_{i}x_{t(i)},$ (13) $$

where $Q_{i}^{\top}x_{s(i)}$ is the probabilities for each of the $m+1$ categories that $x_{t(i)}$ can take.
The marginal forward distribution at time $t(i)$ given $x_{0}$ is

$$ $\displaystyle q(x_{t(i)}|x_{0})=\mathrm{Cat}(x_{t(i)};\bar{Q}_{i}^{\top}x_{0}) =x_{0}^{\top}\bar{Q}_{i}x_{t(i)},$ $$

where $\bar{Q}_{i}=\prod_{j=1}^{i}Q_{j}=\prod_{j=1}^{i}(1-\beta_{j})I+\big{(}1-\prod_
{j=1}^{i}(1-\beta_{j})\big{)}\mathbf{1}e_{m}^{\top}$.
To see what this leads to in continuous time, we let $\beta_{i}=\frac{\beta(t(i))}{T}$ and $T\to\infty$:

$$ $\displaystyle\prod_{j=1}^{i}(1-\beta_{j})$ $\displaystyle=\exp\Big{(}\sum_{j=1}^{i}\log(1-\beta_{j})\Big{)}$ $\displaystyle=\exp\Big{(}\sum_{j=1}^{i}-\frac{\beta(t(j))}{T}+o(1/T)\Big{)}$ $\displaystyle\overset{T\to\infty}{\to}\exp\Big{(}-\int_{0}^{t(i)}\beta(s) \mathrm{d}s\Big{)}.$ $$

We let $\bar{Q}(t)$ denote the limit of $\bar{Q}_{i}$ in this case:

$$ $\displaystyle\bar{Q}(t)$ $\displaystyle=\exp\big{(}-\int_{0}^{t}\beta(s)\mathrm{d}s\big{)}I+\Big{(}1- \exp\big{(}-\int_{0}^{t}\beta(s)\mathrm{d}s\big{)}\Big{)}\mathbf{1}e_{m}^{\top}$ $\displaystyle\triangleq\alpha_{t}I+(1-\alpha_{t})\mathbf{1}e_{m}^{\top}.$ $$

Here we define $\alpha_{t}\triangleq\exp(-\int_{0}^{t}\beta(s)\mathrm{d}s)$.
And the marginal forward transition is

$$ $\displaystyle q(x_{t}|x_{0})=\mathrm{Cat}(x_{t};\bar{Q}(t)^{\top}x_{0})=x_{0}^ {\top}\bar{Q}(t)x_{t}=\alpha_{t}x_{0}^{\top}x_{t}+(1-\alpha_{t})e_{m}^{\top}x_ {t}.$ (14) $$

## Appendix B Continuous-time derivation

We consider a continuous-time Markov chain with transition rates

$$ $\displaystyle Q(t)=(Q_{i}-I)/(1/T)=\beta(t)(\mathbf{1}e_{m}^{\top}-I).$ (15) $$

For simplicity, we let $Q=\mathbf{1}e_{m}^{\top}-I$.
The marginal forward distribution at time $t$ given $x_{0}$ is $q(x_{t}|x_{0})=\mathrm{Cat}(x_{t};\bar{Q}(t)^{\top}x_{0})$, where

$$ $\bar{Q}(t)=\exp\Big{(}\int_{0}^{t}Q(s)\mathrm{d}s\Big{)}=\exp\Big{(}Q\int_{0}^ {t}\beta(s)\mathrm{d}s\Big{)}=\exp(\bar{\beta}(t)Q).$ $$

Here we define $\bar{\beta}(t)\triangleq\int_{0}^{t}\beta(s)\mathrm{d}s$.
The matrix exponential can be computed via eigendecomposition:

$$ $\displaystyle\bar{\beta}(t)Q=U\Lambda U^{-1},$ $$

where

$$ $\displaystyle U$ $\displaystyle=I-e_{m}e_{m}^{\top}+\frac{1}{\sqrt{n+1}}\mathbf{1}e_{m}^{\top},$ $\displaystyle U^{-1}$ $\displaystyle=I+\sqrt{n+1}e_{m}e_{m}^{\top}-\mathbf{1}e_{m}^{\top},$ $\displaystyle\Lambda$ $\displaystyle=\bar{\beta}(t)(e_{m}e_{m}^{\top}-I),$ $$

and thus $\exp(\Lambda)=\alpha_{t}I+(1-\alpha_{t})e_{m}e_{m}^{\top}$,

$$ $\displaystyle\bar{Q}(t)=U\exp(\Lambda)U^{-1}=\alpha_{t}I+(1-\alpha_{t})\mathbf {1}e_{m}^{\top}.$ $$

A simpler derivation uses the following property:

$$ $\displaystyle Q^{2}=-Q.$ $$

Therefore,

$$ $\displaystyle\bar{Q}(t)$ $\displaystyle=\exp(\bar{\beta}(t)Q)$ $\displaystyle=I+\bar{\beta}(t)Q+\frac{1}{2}\bar{\beta}(t)^{2}Q^{2}+\frac{1}{3} \bar{\beta}(t)^{3}Q^{3}+\dots$ $\displaystyle=I+Q-(1-\bar{\beta}(t)+\frac{1}{2}\bar{\beta}(t)^{2}-\frac{1}{3} \bar{\beta}(t)^{3}+\dots)Q$ $\displaystyle=I+Q-\exp(-\bar{\beta}(t))Q$ $\displaystyle=\alpha_{t}I+(1-\alpha_{t})\mathbf{1}e_{m}^{\top}.$ $$

This marginal forward transition matrix at time $t$ coincides with the result (1) we get by taking the limit of discrete-time derivation.

##### Arbitrary discretization of the continuous-time forward process.

For the discrete-time process we have defined the per-step transition in (13).
For the continuous-time process, we can derive the transition matrix $\bar{Q}(s,t)_{ij}\triangleq q(x_{t}=j|x_{s}=i)$ between two arbitrary time $s$ and $t$ as the solution to the following differential equation (known as Kolmogorov forward equation)

$$ $\displaystyle\frac{\mathrm{d}}{\mathrm{d}t}\bar{Q}(s,t)$ $\displaystyle=\bar{Q}(s,t)Q(t)\text{ where }Q(t)=\beta(t)Q$ $$

with initial condition $\bar{Q}(s,s)=I$.
The solution is given by

$$ $\displaystyle\bar{Q}(s,t)=\exp\big{(}(\bar{\beta}(t)-\bar{\beta}(s))Q\big{)}= \bar{Q}(s)^{-1}\bar{Q}(t).$ $$

Routine work (using the Woodbury matrix inversion lemma) shows that

$$ $\displaystyle\bar{Q}(t)^{-1}=\alpha_{t}^{-1}I+(1-\alpha_{t}^{-1})\mathbf{1}e_{ m}^{\top}.$ $$

Plugging the result back, we get the forward transition distribution from $s$ to $t$:

$$ $\displaystyle q(x_{t}|x_{s})$ $\displaystyle=\mathrm{Cat}(x_{t};\bar{Q}(s,t)^{\top}x_{s})=x_{s}^{\top}\bar{Q} (s,t)x_{t},$ (16) $\displaystyle\text{ where }\bar{Q}(s,t)$ $\displaystyle\triangleq\bar{Q}(s)^{-1}\bar{Q}(t)=\frac{\alpha_{t}}{\alpha_{s}} I+\big{(}1-\frac{\alpha_{t}}{\alpha_{s}}\big{)}\mathbf{1}e_{m}^{\top}.$ $$

## Appendix C Time reversal of the forward process given x 0 subscript 𝑥 0 x_{0} italic_x start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT

The analytic property of our forward process allows to compute many quantities of interest in closed form.
One such quantity frequently used in diffusion models is the time reversal of the forward process given $x_{0}$: $q(x_{s}|x_{t},x_{0})$.
We can compute it using (14) and (16) as

$$ $\displaystyle q(x_{s}|x_{t},x_{0})$ $\displaystyle=\frac{q(x_{t}|x_{s})q(x_{s}|x_{0})}{q(x_{t}|x_{0})}$ $\displaystyle=\begin{cases}\frac{\alpha_{s}-\alpha_{t}}{1-\alpha_{t}}x_{s}^{ \top}x_{0}&x_{s}\neq m,x_{t}=m\\ \frac{1-\alpha_{s}}{1-\alpha_{t}}&x_{s}=m,x_{t}=m\\ x_{s}^{\top}x_{t}&x_{t}\neq m.\end{cases}$ (17) $$

Visually, eqn (C) is a $\mathbb{R}^{(m+1)\times(m+1)}$ matrix (Fig. 7, left) whose first index is $x_{t}$ and the second is $x_{s}$. The matrix is almost an identity matrix except the last row corresponding to $x_{t}$ is the mask token. The last row means with probability of $\frac{\alpha_{s}-\alpha_{t}}{1-\alpha_{t}}$ the mask token gets unmasked to become $x_{0}$, and with probability of $\frac{1-\alpha_{s}}{1-\alpha_{t}}$ it remains masked.

Alternatively, we can rewrite the above using reverse transition matrix $\bar{R}^{x_{0}}(t,s)\in\mathbb{R}^{(m+1)\times(m+1)}$ as

$$ $\displaystyle q(x_{s}|x_{t},x_{0})=\mathrm{Cat}(x_{s};\bar{R}^{x_{0}}(t,s)^{ \top}x_{t}),\text{ where }\bar{R}^{x_{0}}(t,s)=I+\frac{\alpha_{s}-\alpha_{t}}{ 1-\alpha_{t}}e_{m}(x_{0}-e_{m})^{\top}.$ $$

Figure: Figure 7: The reverse transition probability and our generative model. Left: $q(x_{s}=\cdot|x_{t}=\cdot,x_{0})$ in matrix form where first index is $x_{t}$ and second index is $x_{s}$. Right: $p_{\theta}(x_{s}=\cdot|x_{t}=\cdot)\triangleq q(x_{s}=\cdot|x_{t}=\cdot,\mu_{ \theta}(x_{t},t))$ also in matrix form.
Refer to caption: x7.png

We are also interested in what would happen in the infinitesimal time limit, i.e., when $s=t-\Delta t$ and $\Delta t\to 0$. Note that

$$ $\displaystyle\alpha_{t-\Delta t}-\alpha_{t}=-\alpha_{t}^{\prime}\Delta t+o( \Delta t).$ $$

Plugging it into the original formula, we get

$$ $\displaystyle\bar{R}^{x_{0}}(t,t-\Delta t)=I-\frac{\alpha_{t}^{\prime}}{1- \alpha_{t}}e_{m}(x_{0}-e_{m})^{\top}\Delta t+o(\Delta t).$ $$

Comparing the above with the transition rate matrix $R^{x_{0}}(t)$ definition

$$ $\displaystyle\bar{R}^{x_{0}}(t,t-\Delta t)=I+R^{x_{0}}(t)\Delta t+o(\Delta t),$ $$

we have determined the transition rate matrix for the reverse process conditioned on $x_{0}$:

$$ $\displaystyle R^{x_{0}}(t)$ $\displaystyle=-\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}e_{m}(x_{0}-e_{m})^{ \top}.$ (18) $$

## Appendix D Details of the ELBO

Using (C) and (3), we compute the KL divergences between forward and reverse transitions

$$ $\displaystyle\mathrm{KL}({q(x_{s}|x_{t},x_{0})}\|{p_{\theta}(x_{s}|x_{t})})$ $\displaystyle=\mathrm{KL}({q(x_{s}|x_{t},x_{0})}\|{q(x_{s}|x_{t},\mu_{\theta}( x_{t},t))})$ (19) $\displaystyle=\begin{cases}\sum_{x_{s}=0}^{m}q(x_{s}|x_{t},x_{0})\log\frac{q(x _{s}|x_{t},x_{0})}{q(x_{s}|x_{t},\mu_{\theta}(x_{t},t))}&x_{t}=m\\ 0&x_{t}\neq m\end{cases}$ $\displaystyle=\delta_{x_{t}=m}\sum_{k\neq m}\frac{\alpha_{s}-\alpha_{t}}{1- \alpha_{t}}x_{0}^{\top}e_{k}\log\frac{x_{0}^{\top}e_{k}}{\mu_{\theta}(x_{t},t) ^{\top}e_{k}}$ $\displaystyle=-\delta_{x_{t}=m}\frac{\alpha_{s}-\alpha_{t}}{1-\alpha_{t}}x_{0} ^{\top}\log\mu_{\theta}(x_{t},t).$ $$

Note that $0\log 0=0$.
Alternatively, this result can be easily obtained from the visual depictions of $q(x_{s}|x_{t},x_{0})$ and $p_{\theta}(x_{s}|x_{t})$ shown in Fig. 7.
In this case, the reconstruction term becomes

$$ $\displaystyle\mathbb{E}_{q(x_{t(1)}|x_{0})}[\log p(x_{0}|x_{t(1)})]$ $\displaystyle=\sum_{k=0}^{m}q_{t(1)|0}(k|x_{0})\log\frac{q_{t(1)|0}(k|x_{0})}{ \sum_{j\neq m}q_{t(1)|0}(k|j)}$ $\displaystyle=\alpha_{t(1)}\cdot\log\frac{\alpha_{t(1)}}{\alpha_{t(1)}}+(1- \alpha_{t(1)})\log\frac{1}{m}$ $\displaystyle=-(1-\alpha_{t(1)})\log m.$ $$

The prior KL term can be computed as

$$ $\mathrm{KL}({q(x_{1}|x_{0})}\|{p(x_{1})})=\mathrm{KL}({\delta_{x_{1},m}}\|{ \delta_{x_{1},m}})=0.$ $$

As usual, we take the continuous-time limit by letting $T\to\infty$:

$$ $\displaystyle\mathcal{L}_{\infty}$ $\displaystyle\triangleq\lim_{T\to\infty}\mathcal{L}_{T}$ $\displaystyle=\lim_{T\to\infty}\sum_{i=2}^{T}-\frac{\alpha_{s(i)}-\alpha_{t(i) }}{s(i)-t(i)}\frac{s(i)-t(i)}{1-\alpha_{t(i)}}x_{0}^{\top}\mathbb{E}_{q(x_{t(i )}|x_{0})}\left[\delta_{x_{t(i)},m}\log\mu_{\theta}(x_{t(i)},t(i))\right]$ $\displaystyle=\int_{t(1)}^{1}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}x_{0}^{ \top}\mathbb{E}_{q(x_{t}|x_{0})}\left[\delta_{x_{t},m}\log\mu_{\theta}(x_{t},t )\right]\mathrm{d}t.$ $$

## Appendix E Avoiding undefined KL divergence

When defining the forward process, we often do not want $\alpha_{1}$ to be exactly 0, or equivalently, $\lambda_{1}$ to be $\infty$ for numerical stability reasons.
Instead, we set $\lambda_{1}$ to be a finite value, and thereby $\alpha_{1}$ has a small positive value.
This has a problem that the support of $q(x_{1}|x_{0})$ is no longer $\{m\}$ and instead becomes $\{m,x_{0}\}$.
As a result, the KL divergence between $q(x_{1}|x_{0})$ and $p(x_{1})$ is undefined because $q(x_{1}|x_{0})$ is not absolutely continuous with respect to $p(x_{1})=\delta_{x_{1},m}$.
To resolve the issue, we modify the prior distribution $p(x_{1})$ such that it has support over all $m+1$ values.
One such choice is letting

$$ $p(x_{1})=\frac{\alpha_{1}}{m}\sum_{j\neq m}\delta_{x_{1},j}+(1-\alpha_{1}) \delta_{x_{1},m}.$ $$

Then, the prior KL divergence term becomes

$$ $\displaystyle\mathrm{KL}({q(x_{1}|x_{0})}\|{p(x_{1})})$ $\displaystyle=\sum_{x_{1}=0}^{m}q(x_{1}|x_{0})\log\frac{q(x_{1}|x_{0})}{p(x_{1 })}$ $\displaystyle=\sum_{x_{1}=0}^{m}(\alpha_{1}\delta_{x_{1},x_{0}}+(1-\alpha_{1}) \delta_{x_{1},m})\log\frac{\alpha_{1}\delta_{x_{1},x_{0}}+(1-\alpha_{1})\delta _{x_{1}=m}}{p(x_{1})}$ $\displaystyle=\alpha_{1}\log\frac{\alpha_{1}}{\alpha_{1}/m}+(1-\alpha_{1})\log \frac{1-\alpha_{1}}{1-\alpha_{1}}$ $\displaystyle=\alpha_{1}\log m.$ $$

## Appendix F Details of Training and Sampling with MD4

### F.1 Training

Figure: Algorithm 1 A single step of training with MD4.

### F.2 Sampling

Figure: Algorithm 2 Unconditional and conditional generation (e.g., infilling) with MD4.

## Appendix G JAX Categorical Sampling and Implicit Top- p 𝑝 p italic_p

We noticed that the following equivalent implementation of Alg. 2 leads to significantly worse sample quality in JAX:

Figure: Algorithm 3 Variant of Alg. 2 that yields lower sample quality when implemented in JAX.

However, mathetically it is equivalent to Alg. 2 and should produce identical results.
Our investigation revealed that the issue arises because Alg. 2 scales the output probabilities of $\mu_{\theta}$ by a small factor $\frac{\alpha_{s}-\alpha_{t}}{1-\alpha_{t}}$ as $s$ is close to $t$, causing some categories to have very low probabilities.
JAX, however, implements categorical sampling using Gumbel argmax, which is less numerically stable than methods like binary search.
As a result, categories with low probabilities are rarely sampled, even when their cumulative probability is significant.
In our experiment, we found that categories with probabilities below 1e-8 are rarely sampled out of a total of 50K categories.
Thus, Alg. 2 implicitly performs top-$p$ sampling (with a dynamic p) under JAX’s categorical sampling, yielding better sample quality than Alg. 3 where $\mu_{\theta}$ is not scaled by a small factor and has fewer categories truncated.

## Appendix H Unifying Existing Masked Diffusion Models

### H.1 The CTMC point of view

We first prove a lemma that connects the forward and reverse transition rate matrices. This follows from the results in but we give a proof for completeness.

###### Lemma 2 .

The forward transition rate matrix $Q(t)$ and the reverse transition rate matrix (given $x_{0}$) $R^{x_{0}}(t)$ satisfy:

$$ $\displaystyle R^{x_{0}}(t)_{kj}=Q(t)_{jk}\frac{q_{t|0}(j|x_{0})}{q_{t|0}(k|x_{ 0})}\text{ for }j\neq k.$ (20) $$

Proof
Consider the reverse transition from time $t+\tau$ to $t$.
For $j\neq k$, Bayes’ rule yields

$$ $\displaystyle q(x_{t}=j|x_{t+\tau}=k,x_{0})$ $\displaystyle=\frac{q(x_{t}=j|x_{0})q(x_{t+\tau}=k|x_{t}=j)}{q(x_{t+\tau}=k|x_ {0})}$ $\displaystyle=\frac{q(x_{t}=j|x_{0})(\delta_{jk}+Q(t)_{jk}\tau+o(\tau))}{q(x_{ t+\tau}=k|x_{0})}$ $\displaystyle\overset{\tau\to 0}{=}\delta_{kj}+\frac{q(x_{t}=j|x_{0})}{q(x_{t} =k|x_{0})}Q(t)_{jk}\tau+o(\tau).$ $$

Then, it follows from the definition of the transition rate matrix that $R^{x_{0}}(t)_{kj}=Q(t)_{jk}\frac{q_{t|0}(j|x_{0})}{q_{t|0}(k|x_{0})}$.
∎

###### Proposition 3 .

We use the shorthand $R_{\theta}(t)_{kj}$ to denote the approximate reverse transition rate from the state $k$ to $j$ obtained by substituting our prediction model $\mu_{\theta}(k)$ for $x_{0}$ in $R^{x_{0}}(t)_{kj}$.
Then, the continuous-time objective (4) can be equivalently expressed as

$$ $\displaystyle\mathcal{L}_{\infty}=-\int_{t(1)}^{1}\mathbb{E}_{q_{t|0}(k|x_{0}) }\Big{[}R_{\theta}(t)_{kk}+\sum_{j\neq k}Q(t)_{kj}\log R_{\theta}(t)_{jk}\Big{ ]}\mathrm{d}t+\text{C},$ (21) $$

where $C$ is a constant independent of $\theta$.

Proof   To rewrite our objective $\mathcal{L}_{\infty}$ with the transition rate matrices, we first go back to (19).
There, instead of plugging in the explicit form of $\bar{R}^{x_{0}}(t,s)$, we substitute it with (8) which leverages the transition rate $R^{x_{0}}(t)$.
To simplify the notation, we assume $x_{t}=k$ and use the shorthand $R_{\theta}(t)_{kj}\triangleq R^{\mu_{\theta}(k)}(t)_{kj}$. We then have

$$ $\displaystyle\mathrm{KL}({q(x_{t-\Delta t}|x_{t},x_{0})}\|{p_{\theta}(x_{t- \Delta t}|x_{t})})$ $\displaystyle=\mathrm{KL}({\mathrm{Cat}(x_{s};\bar{R}^{x_{0}}(t,t-\Delta t)^{ \top}e_{k})}\|{\mathrm{Cat}(x_{s};\bar{R}^{\mu_{\theta}(k)}(t,t-\Delta t)^{ \top}e_{k})})$ $\displaystyle=\sum_{j=0}^{m}e_{k}^{\top}(I+R^{x_{0}}(t)\Delta t+o(\Delta t))e_ {j}\log\frac{e_{k}^{\top}(I+R^{x_{0}}(t)\Delta t+o(\Delta t))e_{j}}{e_{k}^{ \top}(I+R_{\theta}(t)\Delta t+o(\Delta t))e_{j}}$ $\displaystyle=(1+R^{x_{0}}(t)_{kk}\Delta t)\log\frac{1+R^{x_{0}}(t)_{kk}\Delta t +o(\Delta t)}{1+R_{\theta}(t)_{kk}\Delta t+o(\Delta t)}$ $\displaystyle\quad+\sum_{j\neq k}(R^{x_{0}}(t)_{kj}\Delta t)\log\frac{R^{x_{0} }(t)_{kj}\Delta t+o(\Delta t)}{R_{\theta}(t)_{kj}\Delta t+o(\Delta t)}+o( \Delta t)$ $\displaystyle=(R^{x_{0}}(t)_{kk}-R_{\theta}(t)_{kk})\Delta t+\sum_{j\neq k}(R^ {x_{0}}(t)_{kj}\Delta t)\log\frac{R^{x_{0}}(t)_{kj}\Delta t+o(\Delta t)}{R_{ \theta}(t)_{kj}\Delta t+o(\Delta t)}+o(\Delta t).$ $$

For the last identity, we have used the fact that $\log(1+x)=x+o(x)$.
To obtain $\mathcal{L}_{\infty}$, we take the limit of $\mathcal{L}_{T}$ as $T\to\infty$, which is equivalent to letting $\Delta t=1/T\to 0$. We obtain

$$ $\displaystyle\mathcal{L}_{\infty}$ $\displaystyle=\lim_{T\to\infty}\sum_{i=2}^{T}\mathbb{E}_{q(x_{t(i)}|x_{0})}[ \mathrm{KL}({q(x_{s(i)}|x_{t(i)},x_{0})}\|{p_{\theta}(x_{s(i)}|x_{t(i)})})]$ $\displaystyle=\lim_{T\to\infty}\sum_{i=2}^{T}\mathbb{E}_{q(x_{t(i)}|x_{0})} \Big{[}\Big{(}R^{x_{0}}(t(i))_{kk}-R_{\theta}(t(i))_{kk}$ $\displaystyle\quad+\sum_{j\neq k}R^{x_{0}}(t(i))_{kj}\log\frac{R^{x_{0}}(t(i)) _{kj}\Delta t+o(\Delta t)}{R_{\theta}(t(i))_{kj}\Delta t+o(\Delta t)}\Big{)} \Delta t+o(\Delta t)\Big{]}$ $\displaystyle=\int_{t(1)}^{1}\mathbb{E}_{q_{t|0}(k|x_{0})}\Big{[}R^{x_{0}}(t)_ {kk}-R_{\theta}(t)_{kk}+\sum_{j\neq k}R^{x_{0}}(t)_{kj}\log\frac{R^{x_{0}}(t)_ {kj}}{R_{\theta}(t)_{kj}}\Big{]}\mathrm{d}t.$ $$

Note that $R^{x_{0}}(t)$ is a constant matrix independent of $\theta$.
Absorbing all constant terms into $C$, we have

$$ $\mathcal{L}_{\infty}=-\int_{t(1)}^{1}\mathbb{E}_{q_{t|0}(k|x_{0})}\Big{[}R_{ \theta}(t)_{kk}+\sum_{j\neq k}R^{x_{0}}(t)_{kj}\log R_{\theta}(t)_{kj}\Big{]} \mathrm{d}t+C.$ $$

Next, we subtitute $R^{x_{0}}(t)$ with the forward transition rate using Lemma 2:

$$ $\displaystyle\mathcal{L}_{\infty}$ $\displaystyle=-\int_{t(1)}^{1}\mathbb{E}_{q_{t|0}(k|x_{0})}\Big{[}R_{\theta}(t )_{kk}+\sum_{j\neq k}Q(t)_{jk}\frac{q_{t|0}(j|x_{0})}{q_{t|0}(k|x_{0})}\log R_ {\theta}(t)_{kj}\Big{]}\mathrm{d}t+C$ $\displaystyle=-\int_{t(1)}^{1}\Big{[}\sum_{k=0}^{m}q_{t|0}(k|x_{0})R_{\theta}( t)_{kk}+\sum_{k=0}^{m}\sum_{j\neq k}Q(t)_{jk}q_{t|0}(j|x_{0})\log R_{\theta}(t )_{kj}\Big{]}\mathrm{d}t+C$ $\displaystyle=-\int_{t(1)}^{1}\Big{[}\sum_{k=0}^{m}q_{t|0}(k|x_{0})R_{\theta}( t)_{kk}+\sum_{k=0}^{m}\sum_{j\neq k}Q(t)_{kj}q_{t|0}(k|x_{0})\log R_{\theta}(t )_{jk}\Big{]}\mathrm{d}t+C,$ $$

where the last identity used the discrete analog to integration-by-part (or summation-by-part): $\sum_{k=0}\sum_{j\neq k}f(j,k)=\sum_{k=0}\sum_{j\neq k}f(k,j)$.
Rearranging the terms then gives (21).

∎

###### Lemma 2 .

###### Proposition 3 .

### H.2 Differences from Campbell et al. [ 29 ]

used the first term of (21) as the training loss.
A key limitation of this loss function is from the inner summation term

$$ $\displaystyle\sum_{j\neq k}Q(t)_{kj}\log R_{\theta}(t)_{jk}.$ $$

For single dimension case, the sum is analytically computable due to the sparse structure of $R_{\theta}(t)$—if $x_{t}=k$ is mask, the second term disappears; otherwise the only possible neighbor $j$ is a mask.
However, for multidimensional data, $j$ will represent all $N-1$ neighbors in the forward process, i.e., the states we get from mask out a single unmasked dimension of $x_{t}=k$.
Recall that $R_{\theta}(t)_{jk}$ is computed as substituting our neural network prediction model $\mu_{\theta}(j)$ for $x_{0}$ in $R^{x_{0}}(t)_{jk}$.
Therefore, the summation together with $R_{\theta}(t)_{kk}$ requires $N$ evaluations of $\mu_{\theta}(\cdot)$.
This is prohibitive since the neural network model is usually expensive.
To resolve this issue, proposed to rewrite the sum as

$$ $\displaystyle\mathbb{E}_{j\sim\tilde{q}(\cdot|k)}\left[Z_{k}\log R_{\theta}(t) _{jk}\right]\text{\quad where\quad}\tilde{q}(j|k)=\frac{Q(t)_{kj}}{Z_{k}},Z_{k }\triangleq\sum_{j^{\prime}\neq k}Q(t)_{kj^{\prime}}$ $$

and estimate it through Monte Carlo.
Taking into account the outer expectation under $q_{t|0}(k|x_{0})$, the computation of the loss then becomes a doubly stochastic estimate (using $k\sim q_{t|0}(k|x_{0})$ and $j\sim\tilde{q}(j|k)$) which suffers from large variance.
In contrast, the form of our loss (4) only requires evaluating $\mu_{\theta}$ once for a single stochastic estimation of the expectation w.r.t. $q(x_{t}|x_{0})$.

### H.3 Score parameterization

We provide a simpler derivation of the score-based loss below.
We start from the form of the ELBO in (21) and rewrite it as

$$ $\displaystyle\mathcal{L}_{\infty}$ $\displaystyle=\int_{t(1)}^{1}\mathbb{E}_{q_{t|0}(k|x_{0})}\Big{[}\sum_{j\neq k }\Big{(}R^{\mu_{\theta}}(t)_{kj}-R^{x_{0}}(t)_{kj}+R^{x_{0}}(t)_{kj}\log\frac{ R^{x_{0}}(t)_{kj}}{R^{\mu_{\theta}}(t)_{kj}}\Big{)}\Big{]}\mathrm{d}t.$ (22) $$

For the last identity we used the zero-row-sum property of transition rate matrix:

$$ $R^{x_{0}}(t)_{kk}=-\sum_{j\neq k}R^{x_{0}}(t)_{kj}.$ $$

If we plug (20) into (22) and reparameterize with a score model

$$ $\displaystyle s_{\theta}(x_{t})_{j}\triangleq\frac{q_{t|0}(j|\mu_{\theta}(x_{t }))}{q(x_{t}|\mu_{\theta}(x_{t}))},$ (23) $$

we recover the score entropy loss function from :

$$ $\mathcal{L}_{\infty}=\int_{t(1)}^{1}\mathbb{E}_{q_{t|0}(k|x_{0})}\Big{[}\sum_{ j\neq k}Q(t)_{jk}\Big{(}s_{\theta}(k)_{j}-\frac{q_{t|0}(j|x_{0})}{q_{t|0}(k|x_ {0})}\log s_{\theta}(k)_{j}+\psi\Big{(}\frac{q_{t|0}(j|x_{0})}{q_{t|0}(k|x_{0} )}\Big{)}\Big{)}\Big{]}\mathrm{d}t,$ (24) $$

where $\psi(y)\triangleq y\log y-y$.
Note that our derivation above is different and simpler than that of (which is based on) since we leverage the conditional reverse transition rate given $x_{0}$ instead of the transition rate matrix of the reverse process.
We can further simplify the loss with the following relationship between the conditional score and $x_{0}$:

$$ $\displaystyle\frac{q_{t|0}(j|x_{0})}{q_{t|0}(k|x_{0})}=\frac{x_{0}^{\top}\bar{ Q}(t)e_{j}}{x_{0}^{\top}\bar{Q}(t)e_{k}}=\frac{\alpha_{t}}{1-\alpha_{t}}x_{0}^ {\top}e_{j}$ $\displaystyle\text{ for }k=m,j\neq k.$ (25) $$

Note that only the result under the case $k=m$ is needed.
This is because when $x_{t}$ is unmasked, at any time between 0 0 and $t$, the state must stay unchanged and remain $x_{0}$.
As a result, $\mathrm{KL}({q(x_{t-\Delta t}|x_{t},x_{0})}\|{p_{\theta}(x_{t-\Delta t}|x_{t})
})=0$ for $x_{t}\neq m$.
From (15), we know $Q(t)_{jk}=\beta(t)(\delta_{mk}-\delta_{jk})$.
Combining (25) and (24), we get

$$ $\displaystyle\mathcal{L}_{\infty}=\int_{t(1)}^{1}\beta(t)\Big{(}\mathbb{E}_{q_ {t|0}(k|x_{0})}\big{[}\delta_{mk}\big{(}\sum_{j\neq k}s_{\theta}(k)_{j}-\frac{ \alpha_{t}}{1-\alpha_{t}}x_{0}^{\top}\log s_{\theta}(k)\big{)}\big{]}+\psi\big {(}\frac{\alpha_{t}}{1-\alpha_{t}}\big{)}\Big{)}\mathrm{d}t.$ (26) $$

Further, we can show the connection between (26) and (4) by reverting the score parameterization to a mean parameterization using (23), or equivalently
$s_{\theta}(x_{t})_{j}=\frac{\alpha_{t}}{1-\alpha_{t}}\mu_{\theta}(x_{t})^{\top
}e_{j}$.
By doing so, we obtain

$$ $\displaystyle\mathcal{L}_{\infty}=\int_{t(1)}^{1}\beta(t)\Big{(}\mathbb{E}_{q_ {t|0}(k|x_{0})}\big{[}\delta_{mk}\big{(}\sum_{j\neq k}s_{\theta}(k)_{j}-\frac{ \alpha_{t}}{1-\alpha_{t}}x_{0}^{\top}\log\mu_{\theta}(k)\big{]}+\frac{\alpha_{ t}}{1-\alpha_{t}}\big{)}\mathrm{d}t.$ $$

Observing that

$$ $\displaystyle\sum_{j\neq m}s_{\theta}(m)_{j}=\frac{\alpha_{t}}{1-\alpha_{t}},$ (27) $$

we conclude that this recovers the objective in (4).
Interestingly, in the score parameterization is not constrained to satisfy (27).
That means the learned reverse model might be incompatible with the forward process.

Below, we prove Proposition 1 using the result from Eq. 25.

Proof of Proposition 1

$$ $\displaystyle\frac{q_{t}(j)}{q_{t}(m)}$ $\displaystyle=\frac{\sum_{x_{0}}q_{t|0}(j|x_{0})q(x_{0})}{q_{t}(m)}=\frac{\sum _{x_{0}}q_{t|0}(j|x_{0})q_{0|t}(x_{0}|m)}{q_{t|0}(m|x_{0})}=\mathbb{E}_{x_{0}| x_{t}=m}\left[\frac{q_{t|0}(j|x_{0})}{q_{t|0}(m|x_{0})}\right]$ $\displaystyle=\mathbb{E}_{x_{0}|x_{t}=m}\left[\frac{\alpha_{t}}{1-\alpha_{t}}x _{0}^{\top}e_{j}\right]=\frac{\alpha_{t}}{1-\alpha_{t}}\mathbb{E}[x_{0}|x_{t}= m]^{\top}e_{j}.$ $$

∎

### H.4 Other related work.

##### MaskGIT [ 39 ] .

MaskGIT is a diffusion-inspired iterative denoising model for discrete image tokens obtained through models such as VQ-VAE .
Training of MaskGIT follows the steps: (a) Sample $t\in[0,1]$. (b) Given a mask scheduling function $\gamma(t)$, sample $\gamma(t)N$ tokens to place masks. (c) For data $x_{0}$ of size $(m+1)\times N$ and the partially masked state $x_{t}$, minimize the negative log-likelihood

$$ $\displaystyle\mathcal{L}_{\text{MaskGIT}}=-\int^{1}_{0}\mathbb{E}_{x_{t}}\Big{ [}\textstyle\sum_{n:x_{t}^{(n)}=m}(x_{0}^{(n)})^{\top}\log\mu_{\theta}^{(n)}(x _{t},t)\Big{]}\mathrm{d}t.$ (28) $$

Our forward process satisfies
$q_{t|0}(m|x_{0})=1-\alpha_{t}$.
Therefore, when we set the mask scheduling function as
$\gamma(t)=1-\alpha_{t}$
we obtain a loss similar to (5) without the $\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}$ weighting.
Note that there remains a difference in the sampling distribution of $x_{t}$: in the masked diffusion forward process, tokens are sampled independently and do not necessarily yield exactly $(1-\alpha_{t})N$ mask tokens at time $t$, though the expected number is $(1-\alpha_{t})N$.
One might be interested in whether the uniform weighting can be recovered by selecting an appropriate schedule $\alpha_{t}$. However, solving $\alpha_{t}$ such that
$\alpha_{t}^{\prime}=\alpha_{t}-1$ yields $\alpha_{t}=ce^{t}+1$ and there is no $c$ that satisfies both $\alpha_{0}=1$ and $\alpha_{1}=0$.
This shows that training with the MaskGIT loss (28) may not be faithfully optimizing the model likelihood.

##### Discrete flow matching [ 49 ] .

For the linear schedule $\alpha_{t}=1-t$, our reverse transition rate matrix (8) conditioned on $x_{0}$ is:

$$ $\displaystyle R^{x_{0}}(t)=-\frac{\alpha^{\prime}_{t}}{1-\alpha_{t}}e_{m}(x_{0 }-e_{m})^{\top}=\frac{1}{t}e_{m}(x_{0}-e_{m})^{\top}.$ $$

This is the same as the conditional reverse transition rate used in —note that their time $t$ is reversed, and the rate matrix was therefore in the form $R^{x_{0}}(t)=\frac{1}{1-t}e_{m}(x_{0}-e_{m})^{\top}$.

##### SDDM [ 30 ] .

proposed a pseudo-likelihood-like objective for training discrete diffusion models that can also be applied to masked diffusion.
However, their objective encounters the same challenge as — requiring $N$ passes of the mask prediction model.
To mitigate this, they introduced a new transformer architecture, which unfortunately leads to some performance degradation.

##### Blackout diffusion [ 50 ] .

proposed a “blackout” diffusion process that gradually diffuses images to a black state. While this approach is similar to masked diffusion on binary data, key differences emerge when dealing with larger state spaces. In their method, image pixel intensities gradually fade out, whereas ours directly transition to a mask state. Our method offers more flexibility, being applicable to general discrete state spaces without requiring predefined structural relationships. It also demonstrates competitive performance in image generation, achieving this without knowing pixel value proximity.

##### SUNDAE [ 51 , 71 ] .

Unlike masked diffusion, SUNDAE uniformly corrupts data with random tokens in the vocab (known as uniform discrete diffusion ).
Additionally, it uses a second loss term from cross entropy between clean data and 1-step unrolled model prediction. Similar ideas have been proposed in .

## Appendix I Details for state-dependent rates

### I.1 Derivations and time continuous limit

All derivations in this section assume that $x_{t}$
is a single token, while
for $N$ tokens the masked diffusion with state-dependent rates factorises across
the $N$ tokens.
Learning from data of $N$
tokens using variational inference is discussed in Sec. I.2.

Given the forward transition $q(x_{t}|x_{s})$
and marginal $q(x_{s}|x_{0})$ derived in main text
(Sec. 6)
The reversal given $x_{0}$ is $q(x_{s}|x_{t},x_{0})=\mathrm{Cat}(x_{s};\bar{R}^{x_{0}}(t,s)^{\top}x_{t})$ for

$$ $\displaystyle\bar{R}^{x_{0}}(t,s)_{jk}=\begin{cases}\big{(}\frac{\alpha_{s}- \alpha_{t}}{\mathbf{1}-\alpha_{t}}\big{)}^{\top}x_{0}x_{0}^{\top}e_{k}&j=m,k \neq m\\ \big{(}\frac{\mathbf{1}-\alpha_{s}}{\mathbf{1}-\alpha_{t}}\big{)}^{\top}x_{0}& j=m,k=m\\ \delta_{jk}&j\neq m.\end{cases}$ $$

or alternatively can be written as

$$ $\displaystyle q(x_{s}|x_{t},x_{0})$ $\displaystyle=\frac{q(x_{t}|x_{s})q(x_{s}|x_{0})}{q(x_{t}|x_{0})}$ $\displaystyle=\frac{\left[\frac{\alpha_{t}^{\top}x_{s}}{\alpha_{s}^{\top}x_{s} }x_{s}^{\top}x_{t}+(1-\frac{\alpha_{t}^{\top}x_{s}}{\alpha_{s}^{\top}x_{s}})e_ {m}^{\top}x_{t}\right]\left[\alpha_{s}^{\top}x_{0}x_{0}^{\top}x_{s}+(1-\alpha_ {s}^{\top}x_{0})e_{m}^{\top}x_{s}\right]}{\left[\alpha_{t}^{\top}x_{0}x_{0}^{ \top}x_{t}+(1-\alpha_{t}^{\top}x_{0})e_{m}^{\top}x_{t}\right]}.$ (29) $$

To simplify this expression we consider the two cases: either $x_{t}=m$ (i.e. $x_{t}$ is mask) or $x_{t}\neq m$ where in the second case $x_{t}=x_{0}$. For the case $x_{t}=m$, the denominator in (29)
simplifies as

$$ $q(x_{t}=m|x_{0})=1-\alpha_{t}^{\top}x_{0}$ $$

due to $x_{0}^{\top}x_{t}=0$
since $x_{0}\neq m$, i.e. the observed token $x_{0}$ cannot be a mask.
Then given that $x_{t}=m$ the probability that $x_{s}=x_{t}=m$ is

$$ $\frac{1-\alpha_{s}^{\top}x_{0}}{1-\alpha_{t}^{\top}x_{0}}=\frac{({\bf 1}- \alpha_{s})^{\top}x_{0}}{({\bf 1}-\alpha_{t})^{\top}x_{0}}=\left(\frac{{\bf 1} -\alpha_{s}}{{\bf 1}-\alpha_{t}}\right)^{\top}x_{0}$ (30) $$

while the remaining probability for $x_{s}=x_{0}\neq m$ is

$$ $\frac{(\alpha_{s}-\alpha_{t})^{\top}x_{0}}{1-\alpha_{t}^{\top}x_{0}}=\frac{( \alpha_{s}-\alpha_{t})^{\top}x_{0}}{({\bf 1}-\alpha_{t})^{\top}x_{0}}=\left( \frac{\alpha_{s}-\alpha_{t}}{{\bf 1}-\alpha_{t}}\right)^{\top}x_{0}.$ (31) $$

Then, combining (30) and (31) to write $q(x_{s}|x_{t}=m,x_{0})$ in an unified way yields the expression
(11)
in the main Sec. 6. In the second case,
when $x_{t}=x_{0}\neq m$,
$q(x_{s}|x_{t}\neq m,x_{0})$ from (29) simplifies dramatically
and it becomes $q(x_{s}|x_{t}\neq m,x_{0})=x_{t}^{\top}x_{s}$ which is a point mass that sets $x_{s}=x_{t}$.

##### Derivation of the continuous-time limit of the loss in
( 12 ).

To simplify the notation, we let $\xi_{s,t}\triangleq\frac{\alpha_{s}-\alpha_{t}}{1-\alpha_{t}}$.
We first compute the KL divergence terms in the discrete-time ELBO as

$$ $\displaystyle\mathrm{KL}({q(x_{s}|x_{t},x_{0})}\|{p_{\theta}(x_{s}|x_{t})})$ $\displaystyle=\begin{cases}\sum_{x_{s}=0}^{m}q(x_{s}|x_{t},x_{0})\log\frac{q(x _{s}|x_{t},x_{0})}{p_{\theta}(x_{s}|x_{t})}&x_{t}=m\\ 0&x_{t}\neq m\end{cases}$ $\displaystyle=\delta_{x_{t},m}\Big{[}\sum_{k\neq m}\xi_{s,t}^{\top}x_{0}x_{0}^ {\top}e_{k}\log\frac{\xi_{s,t}^{\top}x_{0}x_{0}^{\top}e_{k}}{\xi_{s,t}^{\top} \mathrm{diag}(\mu_{\theta}(x_{t},t))e_{k}}+(1-\xi_{s,t})^{\top}x_{0}\log\frac{ (1-\xi_{s,t})^{\top}x_{0}}{(1-\xi_{s,t})^{\top}\mu_{\theta}(x_{t},t)}\Big{]}$ $\displaystyle=\delta_{x_{t},m}\Big{[}-\xi_{s,t}^{\top}x_{0}x_{0}^{\top}\log\mu _{\theta}(x_{t},t)+(1-\xi_{s,t})^{\top}x_{0}\log\frac{(1-\xi_{s,t})^{\top}x_{0 }}{(1-\xi_{s,t})^{\top}\mu_{\theta}(x_{t},t)}\Big{]}.$ $$

Let $\Delta_{t}\triangleq\frac{1}{T}=t(i)-s(i)$ for all $i$.
Plugging $\alpha_{t-\Delta t}=\alpha_{t}-\alpha_{t}^{\prime}\Delta t+o(\Delta t)$ into the above formula and letting $\gamma_{t}=\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}$, we get

$$ $\displaystyle\mathrm{KL}({q(x_{s}|x_{t},x_{0})}\|{p_{\theta}(x_{s}|x_{t})})$ $\displaystyle=\delta_{x_{t},m}\left[\gamma_{t}^{\top}x_{0}x_{0}^{\top}\log\mu_ {\theta}(x_{t},t)\Delta t+\left(1+\gamma_{t}^{\top}x_{0}\Delta t\right)\cdot \log\frac{1+\gamma_{t}^{\top}x_{0}\Delta t+o(\Delta t)}{1+\gamma_{t}^{\top}\mu _{\theta}(x_{t},t)\Delta t+o(\Delta t)}+o(\Delta t)\right]$ $\displaystyle=\delta_{x_{t},m}\left[\gamma_{t}^{\top}x_{0}x_{0}^{\top}\log\mu_ {\theta}(x_{t},t)\Delta t+\left(1+\gamma_{t}^{\top}x_{0}\Delta t\right)\left( \gamma_{t}^{\top}x_{0}\Delta t-\gamma_{t}^{\top}\mu_{\theta}(x_{t},t)\Delta t+ o(\Delta t)\right)+o(\Delta t)\right]$ $\displaystyle=\delta_{x_{t},m}\left[\gamma_{t}^{\top}x_{0}x_{0}^{\top}\log\mu_ {\theta}(x_{t},t)\Delta t+\gamma_{t}^{\top}x_{0}\Delta t-\gamma_{t}^{\top}\mu_ {\theta}(x_{t},t)\Delta t+o(\Delta t)\right]$ $\displaystyle=\delta_{x_{t},m}\cdot\gamma_{t}^{\top}(x_{0}x_{0}^{\top}\log\mu_ {\theta}(x_{t},t)+x_{0}-\mu_{\theta}(x_{t},t))\Delta t+o(\Delta t).$ $$

Therefore,

$$ $\displaystyle\lim_{T\to\infty}\;\sum_{i=2}^{T}\mathbb{E}_{q(x_{t(i)}|x_{0})}[ \mathrm{KL}({q(x_{s(i)}|x_{t(i)},x_{0})}\|{p_{\theta}(x_{s(i)}|x_{t(i)})})]$ $\displaystyle=\lim_{T\to\infty}\;\sum_{i=2}^{T}\mathbb{E}_{q(x_{t(i)}|x_{0})}[ \delta_{x_{t(i)},m}\cdot\gamma_{t}^{\top}(x_{0}x_{0}^{\top}\log\mu_{\theta}(x_ {t(i)},t(i))+x_{0}-\mu_{\theta}(x_{t(i)},t(i)))\Delta t+o(\Delta t)]$ $\displaystyle=\int_{t(1)}^{1}\gamma_{t}^{\top}\mathbb{E}_{q(x_{t(i)}|x_{0})}[ \delta_{x_{t},m}\cdot(x_{0}x_{0}^{\top}\log\mu_{\theta}(x_{t},t)+x_{0}-\mu_{ \theta}(x_{t},t))]\mathrm{d}t.$ $$

Letting $t(1)\to 0$ proves the result.

### I.2 Training and gradient estimation

The model is applied
to data consisted of $N$
tokens where $x_{0}=(x_{0}^{1},\ldots,x_{0}^{(N)})$ and where each state in the masked diffusion is
$x_{t}=(x_{t}^{1},\ldots,x_{t}^{(N)})$. The reverse generated
model has a factorizing transition conditional of the form
$\prod_{n=1}^{N}p_{\theta}(x_{s}^{(n)}|x_{t})$
where
$p_{\theta}(x_{s}^{(n)}|x_{t})=q(x_{s}^{(n)}|x_{t}^{(n)},\mu_{\theta}^{(n)}(x_{
t},t))$ has a form that depends on whether $x_{t}^{(n)}=m$ or
$x_{t}^{(n)}\neq m$. For the first case:

$$ $p_{\theta}(x_{s}^{(n)}|x_{t}^{(n)}=m,\{x_{t}^{(k)}\}_{k\neq n})=\Big{(}\frac{{ \bf 1}-\alpha_{s}}{{\bf 1}-\alpha_{t}}\Big{)}^{\top}\mu^{(n)}_{\theta}(x_{t},t )e_{m}^{\top}x_{s}^{(n)}+\Big{(}\frac{\alpha_{s}-\alpha_{t}}{{\bf 1}-\alpha_{t }}\Big{)}^{\top}\mathrm{diag}(\mu^{(n)}_{\theta}(x_{t},t))x_{s}^{(n)},$ $$

where $\mu^{(n)}_{\theta}(x_{t},t)=\text{softmax}(f_{\theta}(x_{t}))$ is a $m+1$ dimensional probability vector modelled by a NN (where the final value is constrained to be zero since $\mu^{(n)}_{\theta}(x_{t},t)$ is a reconstruction of
$x_{0}^{(n)}$ which
cannot be mask, so in practice the NN classifier needs to have a softmax output only over the $m$ actual token classes). Crucially, note that the NN classifier receives as input the full
state $x_{t}$ of all tokens, while additional time features to encode $t$ are also included. When
$x_{t}^{(n)}\neq m$
the reverse transition
model is set to be
$p_{\theta}(x_{s}|x_{t}^{(n)}\neq m,\{x_{t}^{(k)}\}_{k\neq n})=(x_{t}^{(n)})^{
\top}x_{s}^{(n)}$ which matches precisely
$q(x_{s}^{(n)}|x_{t}^{(n)}=m,x_{0}^{(n)})=(x_{t}^{(n)})^{\top}x_{s}^{(n)}$ from the forward process.

The full negative lower bound for state-dependent rates and
assuming $N$ tokens is given by

$$ $\displaystyle\mathcal{L}^{(N)}_{\infty}=\int_{0}^{1}\Big{(}\frac{\alpha_{t}^{ \prime}}{1-\alpha_{t}}\Big{)}^{\top}\mathbb{E}_{q(x_{t}|x_{0})}\Big{[}{ \textstyle\sum}_{n:x_{t}^{(n)}=m}(x_{0}^{(n)}-\mu_{\theta}^{(n)}(x_{t},t)+x_{0 }^{(n)}(x_{0}^{(n)})^{\top}\log\mu_{\theta}^{(n)}(x_{t},t))\Big{]}\mathrm{d}t.$ (32) $$

Given that each $\alpha_{t,i}=1-t^{w_{i}}$, the reverse model becomes

$$ $p_{\theta}(x_{s}^{(n)}|x_{t}^{(n)}\neq m,\{x_{t}^{(k)}\}_{k\neq n})=\left(e^{w \log\frac{s}{t}}\right)^{\top}\mu_{\theta}^{(n)}(x_{t},t)e_{m}^{\top}x_{s}^{(n )}+\left(1-e^{w\log\frac{s}{t}}\right)^{\top}\text{diag}(\mu_{\theta}^{(n)}(x_ {t},t))x_{s}^{(n)},$ $$

where $w$ is the $m+1$ dimensional vector of all $w_{i}$s. Note that the probability of $x_{s}^{(n)}$ staying in the mask state, i.e., $x_{s}^{(n)}=m$ depends on the full $x_{t}$ and it is given by
$\left(e^{w\log\frac{s}{t}}\right)^{\top}\mu_{\theta}^{(n)}(x_{t},t)=\sum_{i=0}
^{m-1}e^{w_{i}\log\frac{s}{t}}\mu_{\theta}^{(n)}(x_{t},t)_{i}$
while the probability for $x_{s}^{(n)}$
to take a certain non-mask token value $i$ is
$\left(1-e^{w_{i}\log\frac{s}{t}}\right)\mu_{\theta}^{(n)}(x_{t},t)_{i}.$
The gradient wrt $t$ is $\alpha_{t,i}^{\prime}=-w_{i}t^{w_{i}-1}$ and
$\frac{\alpha_{t,i}^{\prime}}{1-\alpha_{t,i}}=-\frac{w_{i}}{t}$ the above loss is written as

$$ $\displaystyle\mathcal{L}^{(N)}_{\infty}=-\int_{0}^{1}\frac{1}{t}w^{\top} \mathbb{E}_{q(x_{t}|x_{0})}\left[{\textstyle\sum}_{n:x_{t}^{(n)}=m}(x_{0}^{(n) }-\mu_{\theta}^{(n)}(x_{t},t)+x_{0}^{(n)}(x_{0}^{(n)})^{\top}\log\mu_{\theta}^ {(n)}(x_{t},t))\right]\mathrm{d}t,$ $$

where $w$ is the vector of all $w_{i}$’s.
An unbiased gradient over the NN parameters $\theta$ is straightforward
to obtain since we just need to sample one time point $t$ and an $x_{t}\sim q(x_{t}|x_{0})$ to approximate the integral and expectation and then
use the gradient:

$$ $-\nabla_{\theta}\sum_{n:x_{t}^{(n)}=m}\frac{1}{t}w^{\top}\left(x_{0}^{(n)}-\mu _{\theta}^{(n)}(x_{t},t)+x_{0}^{(n)}(x_{0}^{(n)})^{\top}\log\mu_{\theta}^{(n)} (x_{t},t)\right).$ $$

The gradient wrt the $w$ parameters is more complex since
these parameters appear also in the discrete
distribution $q(x_{t}|x_{0})$ which is not reparametrizable. To deal with this we need REINFORCE unbiased gradients  , and in our implementation we consider REINFORCE leave-one-out (RLOO)
with two samples. Firstly, the exact gradient wrt $w$ of the exact loss is written as

$$ $-\int_{0}^{1}\frac{1}{t}\mathbb{E}_{q(x_{t}|x_{0})}\left[g(x_{t},x_{0})\right] \mathrm{d}t-\int_{0}^{1}\frac{1}{t}\mathbb{E}_{q(x_{t}|x_{0})}\left[f(x_{t},x_ {0})\nabla_{w}\log q(x_{t}|x_{0})\right]\mathrm{d}t.$ (33) $$

where

$$ $g(x_{t},x_{0})=\sum_{n:x_{t}^{(n)}=m}(x_{0}^{(n)}-\mu_{\theta}^{(n)}(x_{t},t)+ x_{0}^{(n)}(x_{0}^{(n)})^{\top}\log\mu_{\theta}^{(n)}(x_{t},t)),\ \ \ f(x_{t}, x_{0})=w^{\top}g(x_{t},x_{0}).$ $$

Note that $g(x_{t},x_{0})$ is a vector while $f(x_{t},x_{0})$
is a scalar.
The left term in
(33)
is easy since it just requires
sampling $t$ and $x_{t}\sim q(x_{t}|x_{0})$, while the right term is the REINFORCE term which could have high variance. For this second term we use RLOO with two samples $x_{t}^{1},x_{t}^{2}$ and construct the unbiased estimate

$$ $-\frac{1}{2t}\left(\nabla_{w}\log q(x_{t}^{1}|x_{0})-\nabla_{w}\log q(x_{t}^{2 }|x_{0})\right)\left[f(x_{t}^{1},x_{0})-f(x_{t}^{2},x_{0})\right].$ $$

Thus, the overall unbiased gradient for $w$ we use is

$$ $-\frac{1}{2t}\left\{g(x_{t}^{1},x_{0})+g(x_{t}^{2},x_{0})+\left(\nabla_{w}\log q (x_{t}^{1}|x_{0})-\nabla_{w}\log q(x_{t}^{2}|x_{0})\right)\left[f(x_{t}^{1},x_ {0})-f(x_{t}^{2},x_{0})\right]\right\}.$ $$

## Appendix J Experimental Details

In all experiments, the model is trained with a continuous-time loss while samples are drawn from the discrete-time reverse model of 1000 timesteps unless otherwise noted.
We used an exponential moving average factor 0.9999 for all evaluation including sample generation.

### J.1 text8

We followed the standard dataset split as in and trained our models on text chunks of length 256 for 1 million steps with batch size 512.
All models in the table used a standard 12-layer transformer architecture unless otherwise noted.
Our transformer has also the same number of heads (12) and hidden dimension (784) as in .

We used the continuous-time ELBO and drew one sample of $t$ for each data to estimate the integral.
To reduce the variance of training, we used the same antithetic sampling trick described in for continuous diffusion models.
We used the linear masking schedule $\alpha_{t}=1-t$ and added a small shift $\epsilon=10^{-4}$ when $t$ is close to 0 0 and $1$ to ensure numerical stability.
The shifted schedule is $\alpha_{t}=(1-2\epsilon)(1-t)+\epsilon$.
The shift leads to a support mismatch between $q(x_{1}|x_{0})$ and the prior $p(x_{1})$, leading to an undefined KL divergence term.
We explain in app. E how to modify the prior distribution to allow small uniform probabilities in non-mask states to mitigate this problem.
The shift leads to a non-zero reconstruction term and KL divergence term for the prior distribution but both are of negligible scale so we can safely ignore them when reporting the ELBO.

We used a cosine learning rate schedule with a linear warm up of 2000 steps.
We applied channel-wise dropout of rate $0.05$ and used AdamW optimizer with learning rate 0.0003 and a weight decay factor of 0.03.
Our model is trained on 16 TPU-v5 lite for less than a day.

### J.2 OpenWebText

We kept 2% of the original training set for validation.
Our small and medium transformer model have the same number of layers, heads, and hidden dimensions as in and our tokenizer was also kept the same with a vocabulary size of around 50k.
The training objective, masking schedule and other architectural choices were kept the same with the text8 experiment.
We kept the training hyperparameters the same as text8 experiment except that we reduced the dropout rate to 0.02.

### J.3 FineWeb-Edu

We kept the same training setup as the OpenWebText experiments. Our transformer models have the same number of layers, heads, and hidden dimensions as those of GPT-2 models. We use the same GPT-2 tokenizer.

For Hellaswag evaluation, we concatenate question with each answer option, tokenize the concatenated string, pad to the length of 1024. The padded token sequence gets fed to our MD4 model’s loss function for likelihood evaluation. We average 32 Monte Carlo samples to reduce variance. The answer with the highest likelihood estimate is the model’s prediction.

### J.4 Images

We used the same linear masking schedule as in previous experiments in all likelihood results.
We used the same U-Net plus self-attention architectures from the continuous diffusion model described in for CIFAR-10, except that we did not use Fourier feature inputs and added an additional input embedding layer with embedding size the same as the hidden dimension of the model.
For ImageNet $64\times 64$, we reduced the number of residual blocks (in one side of the U-Net structure) from 64 to 48 and added a 12-layer diffusion transformer  with 768 hidden dimension and 12 heads in the middle.

For both datasets we used AdamW optimizer and trained for 2M iterations.
We used learning rate 0.0004, batch size 256, weight decay factor 0.01 for CIFAR-10 and learning rate 0.0002, batch size 512, weight decay factor 0.03 for ImageNet 64$\times$64.
The learning rate follows a cosine annealing after 100 warm up steps.
Our CIFAR-10 model is trained on 32 TPU-v5 lite for 24 hours.
Our ImageNet-$64\times 64$ model is trained on 256 TPU-v5 lite for 3.5 days.

As explained in Sec. 4, we have observed that the cosine schedule leads to better sample quality so we used it to train a cheaper model for sample visualization.
This model differs from the one that achieves best likelihood in that we used 8 residual blocks (in one side of the UNet structure) and a 20-layer diffusion transformer in the middle.
All other configurations are kept the same.

## Appendix K Additional Results

### K.1 Sample quality evaluation by GPT-2

We use the GPT-2 large model to evaluate the perplexity of samples generated by our model, following .
Results are shown in Fig. 8.

Figure: Figure 8: Generative perplexity evaluated by GPT-2 Large following . We compare MD4 against the GPT-2 checkpoint (autoregressive baseline) and SEDD (the previous best discrete diffusion model on this task) in generating 1024-token text sequences. We investigate the effects of two orthogonal factors on sample quality: model size and decoding steps. The numbers for GPT-2 and SEDD are from .
Refer to caption: extracted/6135574/figures/generative_ppl.png

### K.2 Perplexity on OpenWebText validation set

Tab. 5 reports the final perplexity number achieved on OpenWebText validation set, corresponding to Fig. 4.

**Table 5: Perplexity on OpenWebText validation set.**
| Size | Method | Perplexity ($\downarrow$) |
| --- | --- | --- |
| Small | Gaussian Diffusion | $\leq$ 27.28 |
|  | SEDD Absorb (reimpl.) | $\leq$ 24.10 |
|  | MD4 (Ours) | $\leq$ 22.13 |
|  | GenMD4 (Ours) | $\leq$ 21.80 |
| Medium | MD4 (Ours) | $\leq$ 16.64 |

### K.3 FID evaluation of MD4 trained on ImageNet 64 × \times × 64

We provide the FID numbers corresponding to Fig. 2 in Tab. 6.

**Table 6: FID of 50k samples generated by MD4 trained on ImageNet 64$\times$ 64, corresponding to Fig. 2. Top three rows show results from an unconditional model, while the bottom row is from a model conditioned on class labels. Uniform discretization grid is used in Alg. 2 unless otherwise noted.**
| Method | Timesteps $T$ |  |  |  |
| --- | --- | --- | --- | --- |
| 64 | 128 | 256 | 512 |  |
| Linear $\alpha_{t}$ | 193.81 | 128.18 | 72.94 | 50.21 |
| Linear $\alpha_{t}$, cosine grid | 42.07 | 25.16 | 18.31 | 18.22 |
| Cosine $\alpha_{t}$ | 47.46 | 23.84 | 17.8 | 18.74 |
| Cosine $\alpha_{t}$, class conditional | 30.75 | 11.39 | 7.13 | 7.8 |

### K.4 Additional unconditional generation from MD4 trained on ImageNet 64 × \times × 64

We provide more unconditional generation results from our pixel-level modeling experiments on ImageNet 64$\times$64 in Fig. 9.

Figure: Figure 9: More unconditional samples from MD4 trained on ImageNet 64$\times$64.
Refer to caption: extracted/6135574/figures/imagenet/imagenet_ancestral_02.png

### K.5 Additional unconditional generation from MD4 trained on OpenWebText

Below we include two unconditioned text samples generated by our MD4 Medium model trained on OpenWebText.

#### K.5.1 MD4-M unconditional sample 1: 1024 tokens

#### K.5.2 MD4-M unconditional sample 2: 1024 tokens

### K.6 Conditional generation from MD4 trained on OpenWebText

We share conditionally generated text samples by MD4 Medium in Fig. 10 and observe that slow unmasking near $t=1$, enabled by the cosine schedule, tends to help produce more consist and meaningful samples than uniform unmasking counterpart.

Figure: Figure 10: Conditionally generated text samples from MD4-M. Top: MD4-M trained with the linear schedule, sampled with a uniform grid; Middle: MD4-M trained with the linear schedule, sampled with the cosine grid; Bottom: MD4-M trained with the cosine schedule, sampled with a uniform grid. Context text shown in blue, model-generated text in black.
Refer to caption: x8.png

### K.7 Effect of discretization on zero-shot perplexity

We carried out ablation study on the effect of discretization on zero-shot perplexity.
Results are included in Tab. 7.
Note that this is an inference ablation with the same trained model (MD4-S trained with the continuou-time objective).

**Table 7: Effect of discretization on zero-shot perplexity.**
| Size | Timesteps | LAMBADA | WikiText2 | PTB | WikiText103 | IBW |
| --- | --- | --- | --- | --- | --- | --- |
| Small | T = 100 | $\leq$ 49.8 | $\leq$ 36.1 | $\leq$ 105.2 | $\leq$ 36.1 | $\leq$ 70.3 |
|  | T = 1000 | $\leq$ 48.5 | $\leq$ 35.0 | $\leq$ 102.5 | $\leq$ 35.0 | $\leq$ 68.4 |
|  | T = 10000 | $\leq$ 48.4 | $\leq$ 34.9 | $\leq$ 102.4 | $\leq$ 34.9 | $\leq$ 68.2 |
|  | T = $\infty$ (continuous) | $\leq$ 48.4 | $\leq$ 34.9 | $\leq$ 102.3 | $\leq$ 35.9 | $\leq$ 68.1 |