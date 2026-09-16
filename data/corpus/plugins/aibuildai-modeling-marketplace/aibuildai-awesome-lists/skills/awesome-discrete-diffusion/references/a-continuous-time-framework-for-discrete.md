---
arxiv_id: "2205.14987"
title: "A Continuous Time Framework for Discrete Denoising Models"
year: 2022
source: arxiv2md
---

## Abstract

Abstract We provide the first complete continuous time framework for denoising diffusion models of discrete data. This is achieved by formulating the forward noising process and corresponding reverse time generative process as Continuous Time Markov Chains (CTMCs). The model can be efficiently trained using a continuous time version of the ELBO. We simulate the high dimensional CTMC using techniques developed in chemical physics and exploit our continuous time framework to derive high performance samplers that we show can outperform discrete time methods for discrete data. The continuous time treatment also enables us to derive a novel theoretical result bounding the error between the generated sample distribution and the true data distribution.

## 1 Introduction

Diffusion/score-based/denoising models are a popular class of generative models that achieve state-of-the-art sample quality with good coverage of the data distribution all whilst using a stable, non-adversarial, simple to implement training objective. The general framework is to define a forward noising process that takes in data and gradually corrupts it until the data distribution is transformed into a simple distribution that is easy to sample. The model then learns to reverse this process by learning the logarithmic gradient of the noised marginal distributions known as the score.

Most previous work on denoising models operates on a continuous state space. However, there are many problems for which the data we would like to model is discrete.
This occurs, for example, in text, segmentation maps, categorical features, discrete latent spaces, and the direct 8-bit representation of images. Previous work has tried to realize the benefits of the denoising framework on discrete data problems, with promising initial results .

All of these previous approaches train and sample the model in discrete *time*.
Unfortunately, working in discrete time has notable drawbacks. It generally forces the user to pick a partition of the process at training time and the model only learns to denoise at these fixed time points. Due to the fixed partition, we are then limited to a simple ancestral sampling strategy. In continuous time, the model instead learns to denoise for any arbitrary time point in the process. This complete specification of the reverse process enables much greater flexibility in defining the reverse sampling scheme. For example, in continuous state spaces, continuous time samplers that greatly reduce the sampling time have been devised as well as ones that improve sample quality . The continuous time interpretation has also enabled the derivation of interesting theoretical properties such as error bounds in continuous state spaces.

To allow these benefits to be exploited for discrete state spaces as well, we formulate a continuous time framework for discrete denoising models. Specifically, our contributions are as follows. We formulate the forward noising process as a Continuous Time Markov Chain (CTMC) and identify the generative CTMC that is the time-reversal of this process. We then bound the log likelihood of the generated data distribution, giving a continuous time equivalent of the ELBO that can be used for efficient training of a parametric approximation to the true generative reverse process. To efficiently simulate the parametric reverse process, we leverage tau-leaping and propose a novel predictor-corrector type scheme that can be used to improve simulation accuracy. The continuous time framework allows us to derive a bound on the error between the true data distribution and the samples generated from the approximate reverse process simulated with tau-leaping. Finally, we demonstrate our proposed method on the generative modeling of images from the CIFAR-10 dataset and monophonic music sequences. Notably, we find our tau-leaping with predictor-corrector sampler can provide higher quality CIFAR10 samples than previous discrete time discrete state approaches, further closing the performance gap between when images are modeled as discrete data or as continuous data.

Proofs for all propositions and theorems are given in the Appendix.

Figure: Figure 1: The forward noising process corrupts data according to $R_{t}$, the rate of corruption events at time $t$. The noising process’ time reversal gives the generative process which is defined through $\hat{R}_{t}^{\theta}$, the rate of generative events at time $t$. $\hat{R}_{t}^{\theta}$ is parameterized through the denoising network, $p_{0|t}^{\theta}(x_{0}|x_{t})$, which outputs categorical probabilities over clean $x_{0}$ values conditioned on a noisy $x_{t}$.
Refer to caption: /html/2205.14987/assets/x1.png

## 2 Background on Discrete Denoising Models

In the discrete time, discrete state space case, we aim to model discrete data $x_{0}\in\mathcal{X}$ with finite cardinality $S=|\mathcal{X}|$. We assume $x_{0}\sim p_{\textup{data}}(x_{0})$ for some discrete data distribution $p_{\textup{data}}(x_{0})$. We define a forward noising process that transforms $p_{\textup{data}}(x_{0})$ to some distribution $q_{K}(x_{K})$ that closely approximates an easy to sample distribution $p_{\textrm{ref}}(x_{K})$. This is done by defining forward kernels $q_{k+1|k}(x_{k+1}|x_{k})$ that all admit $p_{\textrm{ref}}$ as a stationary distribution and mix reasonably quickly. For example, one can use a simple uniform kernel , $q_{k+1|k}(x_{k+1}|x_{k})=\delta_{x_{k+1},x_{k}}(1-\beta)+(1-\delta_{x_{k+1},x_{k}})\beta/(S-1)$ where $\delta$ is a Kronecker delta. The corresponding $p_{\textrm{ref}}$ is the uniform distribution over all states. Other choices include: an absorbing state kernel—where for each state there is a small probability that it transitions to some absorbing state—or a discretized Gaussian kernel—where only transitions to nearby states have significant probability (valid for spaces with ordinal structure) .

After defining $q_{k+1|k}$, we have a forward joint decomposition as follows

$$ $\textstyle q_{0:K}(x_{0:K})=p_{\textup{data}}(x_{0})\prod_{k=0}^{K-1}q_{k+1|k}(x_{k+1}|x_{k}).$ (1) $$

The joint distribution $q_{0:K}(x_{0:K})$ also admits a reverse decomposition:

$$ $\textstyle q_{0:K}(x_{0:K})=q_{K}(x_{K})\prod_{k=0}^{K-1}q_{k|k+1}(x_{k}|x_{k+1})\leavevmode\nobreak\ \leavevmode\nobreak\ \text{where}\leavevmode\nobreak\ \leavevmode\nobreak\ q_{k|k+1}(x_{k}|x_{k+1})=\frac{q_{k+1|k}(x_{k+1}|x_{k})q_{k}(x_{k})}{q_{k+1}(x_{k+1})}.$ (2) $$

Here $q_{k}(x_{k})$ denotes the marginal of $q_{0:K}(x_{0:K})$ at time $k$.
If one had access to $q_{k|k+1}$ and could sample $q_{K}$ exactly, then samples from $p_{\textup{data}}(x_{0})$ could be produced by first sampling $x_{K}\sim q_{K}(\cdot)$ and then ancestrally sampling the reverse kernels, i.e. $x_{k}\sim q_{k|k+1}(\cdot|x_{k+1})$.

However, in practice, $q_{k|k+1}$ is intractable and needs to be approximated with a parametric reverse kernel, $p^{\theta}_{k|k+1}$. This kernel is commonly defined through the analytic $q_{k|k+1,0}$ distribution and a parametric ‘denoising’ model $p^{\theta}_{0|k+1}$ ,

$$ $\textstyle p^{\theta}_{k|k+1}(x_{k}|x_{k+1})$ $\textstyle\triangleq\sum_{x_{0}}q_{k|k+1,0}(x_{k}|x_{k+1},x_{0})p^{\theta}_{0|k+1}(x_{0}|x_{k+1})$ (3) $\textstyle=q_{k+1|k}(x_{k+1}|x_{k})\sum_{x_{0}}\frac{q_{k|0}(x_{k}|x_{0})}{q_{k+1|0}(x_{k+1}|x_{0})}p_{0|k+1}^{\theta}(x_{0}|x_{k+1}).$ (4) $$

Though $q_{K}(x_{K})$ is also intractable, for large $K$ we can reliably approximate it with $p_{\textrm{ref}}(x_{K})$.
Note that the faster the transitions mix, the more accurate this approximation becomes. Approximate samples from $p_{\textup{data}}(x_{0})$ can then be obtained by sampling the generative joint distribution

$$ $\textstyle p^{\theta}_{0:K}(x_{0:K})=p_{\textrm{ref}}(x_{K})\prod_{k=0}^{K-1}p^{\theta}_{k|k+1}(x_{k}|x_{k+1}),$ (5) $$

where $\theta$ is trained through minimizing the negative discrete time (DT) ELBO which is an upper bound on the negative model log-likelihood

$$ $\textstyle\mathbb{E}_{p_{\textup{data}}(x_{0})}\left[-\log p^{\theta}_{0}(x_{0})\right]\leq\mathbb{E}_{q_{0:K}(x_{0:K})}\left[-\log\frac{p_{0:K}^{\theta}(x_{0:K})}{q_{1:K|0}(x_{1:K}|x_{0})}\right]=\mathcal{L}_{\textup{DT}}(\theta).$ (6) $$

It was shown in that $\mathcal{L}_{\textup{DT}}$ can be re-written as

$$ $\textstyle\textstyle\mathcal{L}_{\textup{DT}}(\theta)=\mathbb{E}_{p_{\textup{data}}(x_{0})}\Big{[}$ $\textstyle\text{KL}(q_{K|0}(x_{K}|x_{0})||p_{\textrm{ref}}(x_{K}))-\mathbb{E}_{q_{1|0}(x_{1}|x_{0})}\left[\log p^{\theta}_{0|1}(x_{0}|x_{1})\right]$ (7) $\textstyle+\sum_{k=1}^{K-1}\mathbb{E}_{q_{k+1|0}(x_{k+1}|x_{0})}\left[\text{KL}(q_{k|k+1,0}(x_{k}|x_{k+1},x_{0})||p_{k|k+1}^{\theta}(x_{k}|x_{k+1}))\right]\Big{]}$ (8) $$

where KL is the Kullback–Leibler divergence. The forward kernels $q_{k+1|k}$ are chosen such that $q_{k|0}(x_{k}|x_{0})$ can be computed efficiently in a time independent of $k$. With this, $\theta$ can be efficiently trained by taking a random selection of terms from $\mathcal{L}_{\textup{DT}}$ in each minibatch and performing a stochastic gradient step.

## 3 Continuous Time Framework

### 3.1 Forward process and its time reversal

Our method is built upon a continuous time process from $t=0$ to $t=T$. State transitions can occur at any time during this process as opposed to the discrete time case where transitions only occur when one of the finite number of transition kernels is applied (see Figure [1](#S1.F1)). This process is known as a Continuous Time Markov Chain (CTMC), we provide a short overview of CTMCs in Appendix [A](#A1) for completeness. Giving an intuitive introduction here, we can define a CTMC through an initial distribution $q_{0}$ and a transition rate matrix $R_{t}\in\mathbb{R}^{S\times S}$. If the current state is $\tilde{x}$, then the transition rate matrix entry $R_{t}(\tilde{x},x)$ is the instantaneous rate (occurrences per unit time) at which state $\tilde{x}$ transitions to state $x$. Loosely speaking, the next state in the process will likely be one for which $R_{t}(\tilde{x},x)$ is high, and furthermore, the higher the rate is, the less time it will take for this transition to occur.

It turns out that the transition rate, $R_{t}$, also defines the infinitesimal transition probability for the process between the two time points $t-\Delta t$ and $t$

$$ $\textstyle q_{t|t-\Delta t}(x|\tilde{x})=\delta_{x,\tilde{x}}+R_{t}(\tilde{x},x)\Delta t+o(\Delta t),$ (9) $$

where $o(\Delta t)$ represents terms that tend to zero at a faster rate than $\Delta t$.
Comparing to the discrete time case, we see that $R_{t}$ assumes an analogous role to the discrete time forward kernel $q_{k+1|k}$ in how we define the forward process.
Therefore, just as in discrete time, we design $R_{t}$ such that: i) the forward process mixes quickly towards an easy to sample (stationary) distribution, $p_{\textrm{ref}}$, (e.g. uniform), ii) we can analytically obtain $q_{t|0}(x_{t}|x_{0})$ distributions to enable efficient training (see Section [4.1](#S4.SS1) for how this is done). We initialize the forward CTMC at $q_{0}(x_{0})=p_{\textup{data}}(x_{0})$ at time $t=0$. We denote the marginal at time $t=T$ as $q_{T}(x_{T})$, which should be close to $p_{\textrm{ref}}(x_{T})$.

We now consider the time reversal of the forward process, which will take us from the marginal $q_{T}(x_{T})$ back to the data distribution $p_{\textup{data}}(x_{0})$ through a reverse transition rate matrix, $\hat{R}_{t}\in\mathbb{R}^{S\times S}$:

$$ $\textstyle q_{t|t+\Delta t}(\tilde{x}|x)=\delta_{\tilde{x},x}+\hat{R}_{t}(x,\tilde{x})\Delta t+o(\Delta t).$ (10) $$

In discrete time, one uses Bayes rule to go from $q_{k+1|k}$ to $q_{k|k+1}$.
We can use similar ideas to calculate $\hat{R}_{t}$ from $R_{t}$ as per the following result.

###### Proposition 1 .

For a forward in time CTMC, $\{x_{t}\}_{t\in[0,T]}$, with rate matrix $R_{t}$, initial distribution $p_{\textup{data}}(x_{0})$ and terminal distribution $q_{T}(x_{T})$, there exists a CTMC with initial distribution $q_{T}(x_{T})$ at $t=T$, terminal distribution $p_{\textup{data}}(x_{0})$ at $t=0$ and transition rate matrix $\hat{R}_{t}$ that runs backwards in time and is almost everywhere equivalent to the time reversal of the forward CTMC, $\{x_{t}\}_{t\in[T,0]}$. Furthermore, $\hat{R}_{t}$ is related to $R_{t}$ by the following expression

$$ $\textstyle\hat{R}_{t}(x,\tilde{x})=R_{t}(\tilde{x},x)\sum_{x_{0}}\frac{q_{t|0}(\tilde{x}|x_{0})}{q_{t|0}(x|x_{0})}q_{0|t}(x_{0}|x)\quad\text{for}\quad x\neq\tilde{x},$ (11) $$

where $q_{t|0}(x|x_{0})$ are the conditional marginals of the forward process and $q_{0|t}(x_{0}|x)=q_{t|0}(x|x_{0})p_{\textup{data}}(x_{0})/q_{t}(x)$ with $q_{t}(x)$ being the marginal of the forward process at time $t$. When $x=\tilde{x}$, $\hat{R}_{t}(x,x)=-\sum_{x^{\prime}\neq x}\hat{R}_{t}(x,x^{\prime})$ because the rows must sum to zero (see Appendix [A](#A1)).

Unfortunately, $\hat{R}_{t}$ is intractable due to the intractability of $q_{t}(x)$ and thus of $q_{0|t}(x_{0}|x)$. Therefore, we consider an approximation $\hat{R}_{t}^{\theta}$ of $\hat{R}_{t}$ by approximating $q_{0|t}(x_{0}|x)$ with a parametric denoising model, $p^{\theta}_{0|t}(x_{0}|x)$:

$$ $\textstyle\hat{R}_{t}^{\theta}(x,\tilde{x})=R_{t}(\tilde{x},x)\sum_{x_{0}}\frac{q_{t|0}(\tilde{x}|x_{0})}{q_{t|0}(x|x_{0})}p^{\theta}_{0|t}(x_{0}|x)\quad\text{for}\quad x\neq\tilde{x}$ (12) $$

and $\hat{R}^{\theta}_{t}(x,x)=-\sum_{x^{\prime}\neq x}\hat{R}^{\theta}_{t}(x,x^{\prime})$ as before. As a further analogy to the discrete time case, notice that when $x\neq\tilde{x}$, $\hat{R}_{t}^{\theta}$ has the same form as the discrete time parametric reverse kernel, $p^{\theta}_{k|k+1}$ defined in eq ([4](#S2.E4)) but with the forward kernel, $q_{k+1|k}$, replaced by the forward rate, $R_{t}$.

###### Proposition 1 .

### 3.2 Continuous Time ELBO

In discrete time, $\theta$ is trained by minimizing the discrete time negative ELBO, $\mathcal{L}_{\textup{DT}}$, formed from the forward and reverse processes. We mirror this approach in continuous time by minimizing the corresponding continuous time (CT) negative ELBO, $\mathcal{L}_{\textup{CT}}$, as derived below.

###### Proposition 2 .

For the reverse in time CTMC with initial distribution $p_{\textrm{ref}}(x_{T})$, terminal distribution $p_{0}^{\theta}(x_{0})$, and reverse rate $\hat{R}_{t}^{\theta}$, an upper bound on the negative model log-likelihood, $\mathbb{E}_{p_{\textup{data}}(x_{0})}[-\log p_{0}^{\theta}(x_{0})]$, is given by

$$ $\textstyle\mathcal{L}_{\textup{CT}}(\theta)=T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)q_{t}(x)r_{t}(\tilde{x}|x)}\Big{[}\Big{\{}\sum_{x^{\prime}\neq x}\hat{R}_{t}^{\theta}(x,x^{\prime})\Big{\}}-\mathcal{Z}^{t}(x)\log\left(\hat{R}_{t}^{\theta}(\tilde{x},x)\right)\Big{]}+C,$ (13) $$

where $C$ is a constant independent of $\theta$ and

$$ $\textstyle\mathcal{Z}^{t}(x)=\sum_{x^{\prime}\neq x}R_{t}(x,x^{\prime})\hskip 56.9055ptr_{t}(\tilde{x}|x)=(1-\delta_{\tilde{x},x})R_{t}(x,\tilde{x})/\mathcal{Z}^{t}(x).$ (14) $$

Here $r_{t}(\tilde{x}|x)$ gives the probability of transitioning from $x$ to $\tilde{x}$, given that we know a transition occurs at time $t$. We can optimize this objective efficiently with stochastic gradient descent. For a gradient update, we sample a batch of datapoints from $p_{\textup{data}}(x_{0})$, noise each datapoint using a random time, $t\sim\mathcal{U}(0,T)$, $x\sim q_{t|0}(x|x_{0})$ and finally sample an auxiliary $\tilde{x}$ from $r_{t}(\tilde{x}|x)$ for each $x$.
Intuitively, ($x$, $\tilde{x}$) are a pair of states following the forward in time noising process. Minimizing the second term in $\mathcal{L}_{\textup{CT}}$ maximizes the reverse rate for this pair, but going in the backwards direction, $\tilde{x}$ to $x$. This is how $\hat{R}_{t}^{\theta}$ learns to reverse the noising process. Intuition on the first term and a direct comparison to $\mathcal{L}_{\textup{DT}}$ is given in Appendix [C.1](#A3.SS1).

The first argument of $\smash{\hat{R}_{t}^{\theta}}$ is input into $\smash{p_{0|t}^{\theta}}$ so we naively require two network forward passes on $x$ and $\tilde{x}$ to evaluate the objective. We can avoid this by approximating the $q_{t}(x)$ sample in the first term with $\tilde{x}$ meaning we need only evaluate the network once on $\tilde{x}$. The approximation is valid because, as we show in Appendix [C.4](#A3.SS4), $\tilde{x}$ is approximately distributed according to $q_{t+\delta t}$ for $\delta t$ very small.

###### Proposition 2 .

## 4 Efficient Forward and Backward Sampling

### 4.1 Choice of Forward Process

The transition rate matrix $R_{t}$ needs to be chosen such that the forward process: i) mixes quickly towards $p_{\textrm{ref}}$, and ii) the $q_{t|0}(x|x_{0})$ distributions can be analytically obtained. The Kolmogorov differential equation for the CTMC needs to be integrated to obtain $q_{t|0}(x|x_{0})$. This can be done analytically when $R_{t}$ and $R_{t^{\prime}}$ commute for all $t,t^{\prime}$, see Appendix [E](#A5). An easy way to meet this condition is to let $R_{t}=\beta(t)R_{b}$ where $R_{b}\in\mathbb{R}^{S\times S}$ is a user-specified time independent base rate matrix and $\beta(t)\in\mathbb{R}$ is a time dependent scalar. We then obtain the analytic expression

$$ $\textstyle q_{t|0}(x=j|x_{0}=i)=\left(Q\text{exp}\left[\Lambda\int_{0}^{t}\beta(s)ds\right]Q^{-1}\right)_{ij}$ (15) $$

where $R_{b}=Q\Lambda Q^{-1}$ is the eigendecomposition of matrix $R_{b}$ and $\text{exp}[\cdot]$ the element-wise exponential.

Our choice of $\beta$ schedule is guided by , $\beta(t)=ab^{t}\log(b)$.
The hyperparameters $a$ and $b$ are selected such that $\smash{q_{T}(x)\approx p_{\textrm{ref}}(x)}$ at the terminal time $t=T$ while having a steady speed of ‘information corruption’ which ensures that $\smash{\hat{R}_{t}}$ does not vary quickly in a short span of time.

We experiment with a variety of $R_{b}$ matrices, for example, a uniform rate, $R_{b}=\mathbbm{1}\mathbbm{1}^{T}-S\mathrm{Id}$, where $\mathbbm{1}\mathbbm{1}^{T}$ is a matrix of ones and $\mathrm{Id}$ is the identity. For problems with a heavy spatial bias, e.g. images, we can instead use a forward rate that only encourages transitions to nearby states; details and the links to the corresponding discrete time processes can be found in Appendix [E](#A5).

### 4.2 Factorizing Over Dimensions

Our aim is to model data that is $D$ dimensional, with each dimension taking one value from $S$ possibilities. We now slightly redefine notation and say $\bm{x}^{1:D}\in\mathcal{X}^{D}$, $|\mathcal{X}|=S$. In this setting, calculating transition probabilities naively would require calculating $S^{D}$ rate values corresponding to each of the possible next states. This is intractable for any reasonably sized $S$ and $D$. We avoid this problem simply by factorizing the forward process such that each dimension propagates independently. Since this is a continuous time process and each dimension’s forward process is independent of the others, the probability two or more dimensions transition at exactly the same time is zero. Therefore, overall in the full dimensional forward CTMC, each transition only ever involves a change in exactly one dimension. For the time reversal CTMC, it will also be true that exactly one dimension changes in each transition. This makes computation tractable because of the $S^{D}$ rate values, only $D\times(S-1)+1$ are non-zero - those corresponding to transitions where exactly one dimension changes plus the no change transition. Finally, we note that even though dimensions propagate independently in the forward direction, they are not independent in the reverse direction because the starting points for each dimension’s forward process are not independent for non factorized $p_{\textup{data}}$.
The following proposition shows the exact forms for the forward and reverse rates in this case.

###### Proposition 3 .

If the forward process factorizes as $q_{t|s}(\bm{x}_{t}^{1:D}|\bm{x}_{s}^{1:D})=\prod_{d=1}^{D}q_{t|s}(x_{t}^{d}|x_{s}^{d})$, $t>s$, then the forward and reverse rates are of the form

$$ $\displaystyle\textstyle R_{t}^{1:D}(\tilde{\bm{x}}^{1:D},\bm{x}^{1:D})=\sum_{d=1}^{D}R_{t}^{d}(\tilde{x}^{d},x^{d})\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}},$ (16) $\displaystyle\textstyle\hat{R}_{t}^{1:D}(\bm{x}^{1:D},\tilde{\bm{x}}^{1:D})=\sum_{d=1}^{D}R_{t}^{d}(\tilde{x}^{d},x^{d})\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}}\sum_{x_{0}^{d}}q_{0|t}(x_{0}^{d}|\bm{x}^{1:D})\frac{q_{t|0}(\tilde{x}^{d}|x_{0}^{d})}{q_{t|0}(x^{d}|x_{0}^{d})},$ (17) $$

where $R_{t}^{d}\in\mathbb{R}^{S\times S}$ and $\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}}$ is 1 when all dimensions except for $d$ are equal.

To find $\hat{R}_{t}^{\theta\,1:D}$ we simply replace $q_{0|t}(x_{0}^{d}|\bm{x}^{1:D})$ with $p_{0|t}^{\theta}(x_{0}^{d}|\bm{x}^{1:D})$ which is easily modeled with a neural network that outputs conditionally independent state probabilities in each dimension. In Appendix [C.3](#A3.SS3) we derive the form of $\mathcal{L}_{\textup{CT}}$ when we use this factorized form for $R_{t}^{1:D}$ and $\hat{R}_{t}^{\theta\,1:D}$.

###### Proposition 3 .

### 4.3 Simulating the Generative Reverse Process with Tau-Leaping

The parametric generative reverse process is a CTMC with rate matrix $\hat{R}_{t}^{\theta\,1:D}$. Simulating this process from distribution $p_{\textrm{ref}}(\bm{x}_{T}^{1:D})$ at time $t=T$ back to $t=0$ will produce approximate samples from $p_{\textup{data}}(\bm{x}_{0}^{1:D})$. The process could be simulated exactly using Gillespie’s Algorithm which alternates between i) sampling a holding time to remain in the current state and ii) sampling a new state according to the current rate matrix, $\hat{R}_{t}^{\theta\,1:D}$ (see Appendix [F](#A6)). This is inefficient for large $D$ because we would need to step through each transition individually and so only one dimension would change for each simulation step.

Instead, we use tau-leaping , a very popular approximate simulation method developed in chemical physics. Rather than step back through time one transition to the next, tau-leaping leaps from $t$ to $t-\tau$ and applies all transitions that occurred in $[t-\tau,t]$ simultaneously. To make a leap, we assume $\hat{R}_{t}^{\theta\,1:D}$ and $\bm{x}_{t}^{1:D}$ remain constant in $[t-\tau,t]$. As we propagate from $t$ to $t-\tau$, we count all of the transitions that occur, but hold off on actually applying them until we reach $t-\tau$, such that $\bm{x}_{t}^{1:D}$ remains constant in $[t-\tau,t]$. Assuming $\hat{R}_{t}^{\theta\,1:D}$ and $\bm{x}_{t}^{1:D}$ remain constant, the number of times a transition from $\bm{x}_{t}^{1:D}$ to $\tilde{\bm{x}}^{1:D}$ occurs in $[t-\tau,t]$ is Poisson distributed with mean $\tau\hat{R}_{t}^{\theta\,1:D}(\bm{x}_{t}^{1:D},\tilde{\bm{x}}^{1:D})$. Once we reach $t-\tau$, we apply all transitions that occurred simultaneously i.e. $\bm{x}_{t-\tau}^{1:D}=\bm{x}_{t}^{1:D}+\sum_{i}P_{i}(\tilde{\bm{x}}_{i}^{1:D}-\bm{x}_{t}^{1:D})$ where $P_{i}$ is a Poisson random variable with mean $\tau\hat{R}_{t}^{\theta\,1:D}(\bm{x}_{t}^{1:D},\tilde{\bm{x}}_{i}^{1:D})$. Note the sum assumes a mapping from $\mathcal{X}$ to $\mathbb{Z}$.

Figure: Figure 2: 3D visualization of one tau-leaping step from $x_{t}^{1:2}=\{S_{4},S_{1}\}$ to $x_{t-\tau}^{1:2}=\{S_{2},S_{3}\}$. Here, $D=2$, $|\mathcal{X}|=5$, $P_{12}=1$, $P_{22}=2$, all other $P_{ds}=0$.
Refer to caption: /html/2205.14987/assets/x2.png

Using our knowledge of $\hat{R}_{t}^{\theta\,1:D}$, we can further unpack this update. Namely, $\hat{R}_{t}^{\theta\,1:D}(\bm{x}_{t}^{1:D},\tilde{\bm{x}}^{1:D})$ can only be non-zero when $\tilde{\bm{x}}^{1:D}$ has a different value to $\bm{x}_{t}^{1:D}$ in exactly one dimension (rates for multi-dimensional changes are zero). Explicitly summing over these options we get $\bm{x}_{t-\tau}^{1:D}=\bm{x}_{t}^{1:D}+\sum_{d=1}^{D}\sum_{s=1\backslash x_{t}^{d}}^{S}P_{ds}(s-x_{t}^{d})\bm{e}^{d}$ where $\bm{e}^{d}$ is a one-hot vector with a $1$ at dimension $d$ and $P_{ds}$ is a Poisson random variable with mean $\tau\hat{R}_{t}^{\theta\,1:D}(\bm{x}_{t}^{1:D},\bm{x}_{t}^{1:D}+(s-x_{t}^{d})\bm{e}^{d})$. Since multiple $P_{ds}$ can be non-zero, we see that tau-leaping allows $\bm{x}_{t}^{1:D}$ to change in multiple dimensions in a single step. Figure [2](#S4.F2) visualizes this idea. During the $[t-\tau,t]$ interval, one jump occurs in dimension $1$ and two jumps occur in dimension $2$. These are all applied simultaneously once we reach $t-\tau$. When our discrete data has ordinal structure (e.g. Section [6.2](#S6.SS2)) our mapping to $\mathbb{Z}$ is not arbitrary and making multiple jumps within the same dimension ($\sum_{s=1\backslash x_{t}^{d}}^{S}P_{ds}>1$) is meaningful. In the non-ordinal/categorical case (e.g. Section [6.3](#S6.SS3)) the mapping to $\mathbb{Z}$ is arbitrary and so, although taking simultaneous jumps in different dimensions is meaningful, taking multiple jumps within the same dimension is not. For this type of data, we reject changes to $x_{t}^{d}$ for any $d$ for which $\smash{\sum_{s=1\backslash x_{t}^{d}}^{S}P_{ds}>1}$. In practice, the rejection rate is very small when $\smash{R_{t}^{1:D}}$ is suitable for categorical data (e.g. uniform), see Appendix [H.3](#A8.SS3). In Section [4.5](#S4.SS5), our error bound accounts for this low probability of rejection and also the low probability of an out of bounds jump that we observe in practice in the ordinal case.

The tau-leaping approximation improves with smaller $\tau$, recovering exact simulation in the limit as $\tau\rightarrow 0$. Exact simulation is similar to an autoregressive model in that only one dimension changes per step. Increasing $\tau$ and thus the average number of dimensions changing per step gives us a natural way to modulate the ‘autoregressiveness’ of the model and trade sample quality with compute (Figure [4](#S6.F4) right). We refer to our method of using tau-leaping to simulate the reverse CTMC as $\tau$LDR (tau-leaping denoising reversal) which we formalize in Algorithm [1](#algorithm1) in Appendix [F](#A6).

We note that theoretically, one could approximate $\hat{R}_{t}^{\theta\,1:D}$ as constant in the interval $[t-\tau,t]$, and construct a transition probability matrix by solving the forward Kolmogorov equation with the matrix exponential $\smash{P_{t-\tau|t}\approx\text{exp}(\tau\hat{R}_{t}^{\theta\,1:D})}$. However, for the learned $\smash{\hat{R}_{t}^{\theta\,1:D}\in\mathbb{R}^{S^{D}\times S^{D}}}$ matrix, it is intractable to compute this matrix exponential so we use tau-leaping for sampling instead.

### 4.4 Predictor-Corrector

During approximate reverse sampling, we aim for the marginal distribution of samples at time $t$ to be close to $q_{t}(x_{t})$ (the marginal at time $t$ of the true CTMC). The continuous time framework allows us to exploit additional information to more accurately follow the reverse progression of marginals, $\{q_{t}(x_{t})\}_{t\in[T,0]}$ and improve sample quality. Namely, after a tau-leaping ‘predictor’ step using rate $\hat{R}_{t}^{\theta}$, we can apply ‘corrector’ steps with rate $R_{t}^{c}$ which has $q_{t}(x_{t})$ as its stationary distribution. The corrector steps bring the distribution of samples at time $t$ closer to the desired $q_{t}(x_{t})$ marginal. $R_{t}^{c}$ is easy to calculate as stated below

###### Proposition 4 .

For a forward CTMC with marginals $\{q_{t}(x_{t})\}_{t\in[0,T]}$, forward rate, $R_{t}$, and corresponding reverse CTMC with rate $\hat{R}_{t}$, the rate $R_{t}^{c}=R_{t}+\hat{R}_{t}$ has $q_{t}(x_{t})$ as its stationary distribution.

In practice, we approximate $R_{t}^{c}$ by replacing $\hat{R}_{t}$ with $\hat{R}_{t}^{\theta}$. This is directly analogous to Predictor-Corrector samplers in continuous state spaces that predict by integrating the reverse SDE and correct with score-based Markov chain Monte Carlo steps, see Appendix [F.2](#A6.SS2) for further discussion.

###### Proposition 4 .

### 4.5 Error Bound

Our continuous time framework also allows us to provide a novel theoretical bound on the error between the true data distribution and the sample distribution generated via tau-leaping (without predictor-corrector steps), in terms of the error in our approximation of the reverse rate and the mixing of the forward noising process.

We assume we have a time-homogeneous rate matrix $R_{t}$ on $\mathcal{X}$, from which we construct the factorized rate matrix $R_{t}^{1:D}$ on $\mathcal{X}^{D}$ by setting $R^{d}_{t}=R_{t}$ for each $d$. Note that by rescaling time by a factor of $\beta(t)$ we can transform our choice of rate from Section [4.1](#S4.SS1) to be time-homogeneous. We will denote $|R|=\sup_{t\in[0,T],x\in\mathcal{X}}|R_{t}(x,x)|$, and let $t_{\textup{mix}}$ be the (1/4)-mixing time of the CTMC with rate $R_{t}$ (see ).

###### Theorem 1 .

For any $D\geq 1$ and distribution $p_{\textup{data}}$ on $\mathcal{X}^{D}$, let $\{x_{t}\}_{t\in[0,T]}$ be a CTMC starting in $p_{\textup{data}}$ with rate matrix $R^{1:D}_{t}$ as above. Suppose that $\hat{R}_{t}^{\theta\,1:D}$ is an approximation to the reverse rate matrix and let $(y_{k})_{k=0,1,\dots,N}$ be a tau-leaping approximation to the reverse dynamics with maximum step size $\tau$. Suppose further that there is some constant $M>0$ independent of $D$ such that

$$ $\sum_{y\neq x}\left|\hat{R}_{t}^{1:D}(x,y)-\hat{R}_{t}^{\theta\,1:D}(x,y)\right|\leq M$ (18) $$

for all $t\in[0,T]$. Then under the assumptions in Appendix [B.5](#A2.SS5), there are constants $C_{1},C_{2}>0$ depending on $\mathcal{X}$ and $R_{t}$ but not $D$ such that, if $\mathcal{L}(y_{0})$ denotes the law of $y_{0}$, we have the total variation bound

$$ $\textstyle||\mathcal{L}(y_{0})-p_{\textup{data}}||_{\textup{TV}}\leq 3MT+\left\{\big{(}|R|SDC_{1}\big{)}^{2}+\frac{1}{2}C_{2}(M+C_{1}SD|R|)\right\}\tau T+2\exp\left\{-\frac{T\log^{2}2}{t_{\textup{mix}}\log 4D}\right\}$ (19) $$

The first term of the above bound captures the error introduced by our approximation of the reverse rate $\hat{R}_{t}^{1:D}$ with $\hat{R}_{t}^{\theta\,1:D}$. The second term reflects the error introduced by the tau-leaping approximation, and is linear in both $T$ and $\tau$, showing that as we take our tau-leaping steps to be arbitrarily small, the error introduced by tau-leaping goes to zero. The final term describes the mixing of the forward chain, and captures the error introduced since $p_{\textup{ref}}$ and $q_{T}$ are not exactly equal.

We choose to make the dependence of the bound on the dimension $D$ explicit, since we are specifically interested in applying tau-leaping to high dimensional problems where we make transitions in different dimensions simultaneously in a single time step. The bound grows at worst quadratically in the dimension, versus e.g. exponentially.
The bound is therefore useful in showing us that we do not need to make $\tau$ impractically small in high dimensions. Other than gaining these intuitions, we do not expect the bound to be particularly tight in practice and further it would not be practical to compute because of the difficulty in finding $M$, $C_{1}$ and $C_{2}$.

The assumptions listed in Appendix [B.5](#A2.SS5) hold approximately for tau-leaping in practice when we use spatially biased rates for ordinal data such that jump sizes are small or uniform rates for non-ordinal data such that the dimensional rejection rate is small. These assumptions could be weakened, however, Theorem [1](#Thmtheorem1) would become much more involved, obscuring the intuition and structure of the problem.

###### Theorem 1 .

## 5 Related Work

The application of denoising models to discrete data was first described in using a binomial diffusion process for a binary dataset. Each reverse kernel $\smash{p^{\theta}_{k|k+1}}$ was directly parameterized without using a denoising model $\smash{p_{0|k}^{\theta}}$. In an approach for discrete categorical data was suggested using a uniform forward noising kernel, $q_{k+1|k}$, and a reverse kernel parameterized through a denoising model, though no experiments were performed with the approach. Experiments on text and segmentation maps were then performed with a similar model in . Other forward kernels were introduced in that are more appropriate for certain data types such as the spatially biased Gaussian kernel. apply the approach to discrete latent space modeling using uniform and absorbing state forward kernels. Whilst a link to continuous time for the forward process is mentioned in , all of these approaches train and sample in discrete time. We show in Appendix [G](#A7) that this involves making an implicit approximation for multi-dimensional data. We extend this line of work by training and sampling in continuous time.

Other works also operate in discrete space but less rigidly follow the diffusion framework. A corruption process tailored to text is proposed in , whereby token deletion and insertion is also incorporated. also focus on text, creating a generative reverse chain that repeatedly applies the same denoising kernel. The corruption distribution is also defined through the same denoising kernel to reduce distribution shift between training and sampling. In , a more standard masking based forward process is used but the reversal is interpreted from an order agnostic autoregressive perspective. They also describe how their model can be interpreted as the reversal of a continuous time absorbing state diffusion but do not utilize this perspective in training or sampling.
propose a denoising type framework that can be used on binary data where the forward and reverse process share the same transition kernel. Finally, in , the discrete latent space of a VQVAE is modeled by quantizing an underlying continuous state space diffusion with probabilistic quantization functions.

## 6 Experiments

### 6.1 Demonstrative Example

Figure: (a)
Refer to caption: /html/2205.14987/assets/x3.png

We first verify the method can accurately produce samples from the entire support of the data distribution and that tau-leaping can accurately simulate the reverse CTMC. To do this, we create a dataset formed of 2d samples of a state space of 32 arranged such that the histogram of the training dataset forms a ‘$\tau$’ shape. We train a denoising model using the $\mathcal{L}_{\textup{CT}}$ objective with $\smash{p_{0|t}^{\theta}}$ parameterized through a residual MLP (full details in Appendix [H.1](#A8.SS1)). We then sample the parameterized reverse process using an exact method (up to needing to numerically integrate the reverse rate) and tau-leaping. Figure [3](#S6.F3) top-right shows the marginals during reverse simulation with $\tau=0.004$ and we indeed produce samples from the entire support of $p_{\textup{data}}$. Furthermore, we find that with sufficiently small $\tau$, we can match the fidelity of exact simulation of the reverse CTMC (Figure [3](#S6.F3) left).
The value of $\tau$ dictates the number of network evaluations in the reverse process according to $\text{NFE}=T/\tau$. In all experiments we use $T=1$. Exact simulation results in a non zero Hellinger distance between the generated and training distributions because of imperfections in the learned $\smash{\hat{R}_{t}^{\theta}}$ model.

### 6.2 Image Modeling

**Table 1: Sample quality metrics and model likelihoods for diffusion methods modeling CIFAR10 in discrete state space. Diffusion methods modeling CIFAR10 in continuous space are included for reference. The Inception Score (IS) and Fréchet Inception Distance (FID) are calculated using 50000 generated samples with respect to the training dataset as is standard practice. The ELBO values are reported on the test set in bits per dimension.**
|  | Method | IS $(\uparrow)$ | FID $(\downarrow)$ | ELBO $(\uparrow)$ |
| --- | --- | --- | --- | --- |
| Discrete state | D3PM Absorbing | $6.78$ | $30.97$ | $-4.40$ |
|  | D3PM Gauss | $8.56$ | $7.34$ | $\mathbf{-3.44}$ |
|  | $\tau$LDR-0 0 (ours) | $8.74$ | $8.10$ | $-3.59$ |
|  | $\tau$LDR-$10$ (ours) | $\mathbf{9.49}$ | $\mathbf{3.74}$ | $-3.59$ |
| Continuous state | DDPM | $9.46$ | $3.17$ | $-3.75$ |
|  | NCSN | $9.89$ | $2.20$ | - |

Figure: (a)
Refer to caption: /html/2205.14987/assets/x5.png

We now demonstrate that our continuous time framework gives us improved generative modeling performance versus operating in discrete time. We show this on the CIFAR-10 image dataset. Images are typically stored as discrete data, each pixel channel taking one value from 256 possibilities. Continuous state space methods have to somehow get around this fact by, for example, adding a discretization function at the end of the generative process or adding uniform noise to the data. Here, we model the images directly in discrete space. We parameterize $p_{0|t}^{\theta}$ using the standard U-net architecture with the modifications for discrete state space suggested by . We use a spatially biased rate matrix and train with an augmented $\mathcal{L}_{\textup{CT}}$ loss including direct $p_{0|t}^{\theta}$ supervision, full experimental details are in Appendix [H.2](#A8.SS2).

Figure [4](#S6.F4) left shows randomly generated unconditional CIFAR10 samples from the model and we report sample quality metrics in Table [1](#S6.T1). We see that our method ($\tau$LDR-0 0) with 0 0 corrector steps has better Inception Score but worse FID than the D3PM discrete time method. However, our $\tau$LDR-$10$ method with $10$ corrector steps per predictor step at the end of the reverse sampling process ($t<0.1T$) greatly improves sample quality, beating the discrete time method in both metrics and further closes the performance gap with methods modeling images as continuous data. The derivation of the corrector rate which gave us this improved performance required our continuous time framework. D3PM achieves the highest ELBO but we note that this does not correlate well with sample quality. In Table [1](#S6.T1), $\tau$ was adjusted such that both $\tau$LDR-0 0 and $\tau$LDR-$10$ used 1000 $\smash{p_{0|t}^{\theta}}$ evaluations in the reverse sampling procedure. We show how FID score varies with number of $\smash{p_{0|t}^{\theta}}$ evaluations for $\tau$LDR-$\{0,3,10\}$ in Figure [4](#S6.F4) right. The optimum number of corrector steps depends on the sampling budget, with lower numbers of corrector steps being optimal for tighter budgets. This is due to the increased $\tau$ required to maintain a fixed budget when we use a larger number of corrector steps.

### 6.3 Monophonic Music

In this experiment, we demonstrate our continuous time model improves generation quality on non-ordinal/categorical discrete data. We model songs from the Lakh pianoroll dataset . We select all monophonic sequences from the dataset such that at each of the 256 time steps either one from 128 notes is played or it is a rest. Therefore, our data has state space size $S=129$ and dimension $D=256$. We scramble the ordering of the state space when mapping to $\mathbb{Z}$ to destroy any ordinal structure. We parameterize $p_{0|t}^{\theta}$ with a transformer architecture and train using a conditional form of $\mathcal{L}_{\textup{CT}}$ targeting the conditional distribution of the final 14 bars (224 time steps) given the first 2 bars of the song. We use a uniform forward rate matrix, $R_{t}$, full experimental details are given in Appendix [H.3](#A8.SS3). Conditional completions of unseen test songs are shown in Figure [5](#S6.F5).
The model is able to faithfully complete the piece in the same style as the conditioning bars.

We quantify sample quality in Table [2](#S6.T2).
We use two metrics: the Hellinger distance between the histograms of generated and ground truth notes and the proportion of outlier notes in the generations but not in the ground truth.
Using our method, we compare between a birth/death and uniform forward rate matrix $R_{t}$.
The birth/death rate is only non-zero for adjacent states whereas the uniform rate allows transitions between arbitrary states which is more appropriate for the categorical case thus giving improved sample quality.
Adding 2 corrector steps per predictor step further improves sample quality.
We also compare to the discrete time method D3PM with its most suitable corruption process for categorical data. We find it performs worse than our continuous time method.

**Table 2: Metrics comparing generated conditional samples and ground truth completions. We compute these over the test set showing mean$\pm$std with respect to 5 samples for each test song.**
| Model | Hellinger Distance | Proportion of Outliers |
| --- | --- | --- |
| $\tau$LDR-0 Birth/Death | $0.3928\pm 0.0010$ | $0.1316\pm 0.0012$ |
| $\tau$LDR-0 Uniform | $0.3765\pm 0.0013$ | $0.1106\pm 0.0010$ |
| $\tau$LDR-2 Uniform | $\mathbf{0.3762\pm 0.0015}$ | $\mathbf{0.1091\pm 0.0014}$ |
| D3PM Uniform | $0.3839\pm 0.0002$ | $0.1137\pm 0.0010$ |

Figure: Figure 5: Conditional completions of an unseen music sequence. The conditioning 2 bars are shown to the left of the black line. More examples and audio recordings are linked in Appendix [H.3](#A8.SS3).
Refer to caption: /html/2205.14987/assets/x7.png

## 7 Discussion

We have presented a continuous time framework for discrete denoising models. We showed how to efficiently sample the generative process with tau-leaping and provided a bound on the error of the generated samples. On discrete data problems, we found our predictor-corrector sampler improved sample quality versus discrete time methods.
Regarding limitations, our model requires many model evaluations to produce a sample. Our work has opened the door to applying the work improving sampling speed on continuous data to discrete data problems too. Modeling performance on images is also slightly behind continuous state space models, we hope this gap is further closed with bespoke discrete state architectures and corruption process tuning. Finally, we note that the ELBO values for the discrete time model on CIFAR10 are better than for our method. In this work, we focused on sample quality rather than using our model to give data likelihoods e.g. for compression downstream tasks.

## Acknowledgements

Andrew Campbell and Joe Benton acknowledge support from the EPSRC CDT in Modern Statistics and Statistical Machine Learning (EP/S023151/1). Arnaud Doucet is partly supported by the EPSRC grant EP/R034710/1. He also acknowledges support of the UK Defence Science and Technology Laboratory (DSTL) and EPSRC under grant EP/R013616/1. This is part of the collaboration between US DOD, UK MOD and UK EPSRC under the Multidisciplinary University Research Initiative. This project made use of time on Tier 2 HPC facility JADE2, funded by EPSRC (EP/T022205/1).

## Appendix A Primer on Continuous Time Markov Chains

Figure: Figure 6: Schematic representation of a 1-dimensional CTMC with 3 states.
Refer to caption: /html/2205.14987/assets/x8.png

A Continuous Time Markov Chain (CTMC) is a right continuous stochastic process
$\{x_{t}\}_{t\in[0,T]}$ satisfying the Markov property, with $x_{t}$ taking
values in a discrete state space $\mathcal{X}$. Since the CTMC is Markov,
future behaviour of the process depends only on the current state and not the
history. A schematic representation of a CTMC path is shown in Figure
[6](#A1.F6). The process repeatedly transitions from one state
to another after having waited in the previous state for a randomly determined
amount of time.

A CTMC can be completely characterised by its jumps and holding times. Specifically, the time between each jump or holding time is exponentially distributed with mean $\nu(x)$ where $x$ is the state in which the process is holding. The next state that is jumped to is drawn from a jump probability distribution $r(\tilde{x}|x)$. The holding and jumping procedure is then repeated.

There is an equivalent definition involving the transition rate matrix,
$R\in\mathbb{R}^{S\times S}$, that we use in the main paper. The transition
rate matrix is defined as

$$ $R(\tilde{x},x)=\underset{\Delta t\rightarrow 0}{\text{lim}}\frac{q_{t|t-\Delta t}(x|\tilde{x})-\delta_{x,\tilde{x}}}{\Delta t}$ (20) $$

where $R(\tilde{x},x)$ is the $(\tilde{x},x)$ element of the transition rate matrix and $q_{t|t-\Delta t}(x|\tilde{x})$ is the infinitesimal transition probability of being in state $x$ at time $t$ given that the process was in state $\tilde{x}$ at time $t-\Delta t$. Conversely, the CTMC can itself be defined through this infinitesimal transition probability

$$ $q_{t|t-\Delta t}(x|\tilde{x})=\delta_{x,\tilde{x}}+R(\tilde{x},x)\Delta t+o(\Delta t)$ (21) $$

where $o(\Delta t)$ represents terms that tend to zero at a faster rate than $\Delta t$. From this definition of the transition rate matrix, we can infer the following properties:

$$ $\textstyle R(\tilde{x},x)\geq 0\quad\text{for}\quad\tilde{x}\neq x,\hskip 28.45274ptR(x,x)\leq 0,\hskip 28.45274ptR(x,x)=-\sum_{x^{\prime}\neq x}R(x,x^{\prime})$ (22) $$

$R(\tilde{x},x)$ is the rate at which probability mass moves from state $\tilde{x}$ to $x$. $R(x,x)$ is the total rate at which probability mass moves out of state $x$ and is thus negative.

In the time-homogeneous case, $R$ has simple relations to the jump and holding time definitions.

$$ $\nu(x)=-\frac{1}{R(x,x)}\hskip 42.67912ptr(\tilde{x}|x)=(1-\delta_{\tilde{x},x})\frac{R(x,\tilde{x})}{-R(x,x)}$ (23) $$

In the time-inhomogeneous case, our transition rate matrix will now depend on time, $R_{t}$, and these simple relations to the jump and holding time definition do not hold. However, $R_{t}$ will still follow equations ([20](#A1.E20)), ([21](#A1.E21)) and ([22](#A1.E22)).

The CTMC transition probabilities satisfy the Kolmogorov forward and backward equations. For $t>s$,

$$ Kolmogorov forward equation $\displaystyle\partial_{t}q_{t|s}(x|\tilde{x})=\sum_{y}q_{t|s}(y|\tilde{x})R_{t}(y,x)$ (24) Kolmogorov backward equation $\displaystyle\partial_{s}q_{t|s}(x|\tilde{x})=-\sum_{y}R_{s}(\tilde{x},y)q_{t|s}(x|y)$ (25) $$

The Kolmogorov forward equation also gives us a differential equation for the marginals of the CTMC.

$$ $\partial_{t}q_{t}(x)=\sum_{y}q_{t}(y)R_{t}(y,x).$ (26) $$

#### Exponential and Poisson Random Variables

In the time homogeneous case, holding times are exponentially distributed with mean $\nu(x)=-1/R(x,x)$. The tau-leaping algorithm relies on the fact that the number of events in interval $[0,t]$ is Poisson distributed with mean $\frac{1}{\nu}t$ when the inter-event times are exponentially distributed with mean $\nu$.

## Appendix B Proofs

### B.1 Proof of Proposition 1

###### Proof.

We recall that a process $\{x_{t}\}_{t\in[0,T]}$ taking values in
$\mathcal{X}$ is called a CTMC if it is right-continuous and satisfies the
Markov property. Denote $\{y_{t}\}_{t\in[0,T]}=\{x_{T-t}\}_{t\in[0,T]}$
except at the jump times of the forward process $\tau_{n}$ with
$n\in\mathbb{N}$, where
$y_{T-\tau_{n}}=x_{\tau_{n}}^{-}=\lim_{t\leq\tau_{n},t\to\tau_{n}}x_{t}$. Hence, $\{y_{t}\}_{t\in[0,T]}$ is almost surely equal to
$\{x_{T-t}\}_{t\in[0,T]}$ and is right-continuous. Since the Markov property
is symmetric, we get that $\{y_{t}\}_{t\in[0,T]}$ is a CTMC. We now compute
its transition matrix. Let $x,\tilde{x}\in\mathcal{X}$ with
$x\neq\tilde{x}$, using the Kolmogorov forward equation, we have

$$ $\textstyle{\partial_{t}p_{t|s}(\tilde{x}|x)=\sum_{y\in\mathcal{X}}p_{t|s}(y|x)\hat{R}_{T-t}(y,\tilde{x})\;,}$ (27) $$

where $\{p_{t|s},\ s,t\in[0,T],\ t>s\}$ is the
transition probability system associated with
$\{y_{t}\}_{t\in[0,T]}$ and $\{\hat{R}_{T-t}\}_{t\in[0,T]}$
is the transition rate matrix associated with
$\{y_{t}\}_{t\in[0,T]}$. Note that

$$ $\displaystyle p_{t|s}(x=j|\tilde{x}=i)$ $\displaystyle=\mathbb{P}(y_{t}=j\ |\ y_{s}=i)$ (28) $\displaystyle=\mathbb{P}(x_{T-t}=j\ |\ x_{T-s}=i)$ (29) $\displaystyle=\mathbb{P}(x_{T-s}=i|x_{T-t}=j)\frac{\mathbb{P}(x_{T-t}=j)}{\mathbb{P}(x_{T-s}=i)}$ (30) $\displaystyle=q_{T-s|T-t}(\tilde{x}=i|x=j)\frac{q_{T-t}(x=j)}{q_{T-s}(\tilde{x}=i)}\;$ (31) $$

where $\{q_{t|s},s,t\in[0,T],t>s\}$ is the transition probability system associated with $\{x_{t}\}_{t\in[0,T]}$ and $\{q_{t},t\in[0,T]\}$ are the marginals of $\{x_{t}\}_{t\in[0,T]}$.
Now, writing the backward Kolmogorov equation for $\{x_{t}\}_{t\in[0,T]}$

$$ $\textstyle\partial_{s}q_{t|s}(\tilde{x}|x)=-\sum_{y\in\mathcal{X}}R_{s}(x,y)q_{t|s}(\tilde{x}|y)$ (32) $$

Re-labeling the time indices we obtain,

$$ $\textstyle\partial_{T-t}q_{T-s|T-t}(\tilde{x}|x)$ $\textstyle=-\sum_{y\in\mathcal{X}}R_{T-t}(x,y)q_{T-s|T-t}(\tilde{x}|y)$ (33) $\textstyle\partial_{t}q_{T-s|T-t}(\tilde{x}|x)$ $\textstyle=\sum_{y\in\mathcal{X}}R_{T-t}(x,y)q_{T-s|T-t}(\tilde{x}|y)$ (34) $$

Letting $s\rightarrow t$ and using that $\lim_{s\to t}q_{T-s|T-t}(x|\tilde{x})=0$, we get that

$$ $\displaystyle\hat{R}_{T-t}(x,\tilde{x})$ $\displaystyle=\lim_{s\to t}\partial_{t}p_{t|s}(\tilde{x}|x)$ (35) $\displaystyle=\lim_{s\to t}\partial_{t}\left(q_{T-s|T-t}(x|\tilde{x})\frac{q_{T-t}(\tilde{x})}{q_{T-s}(x)}\right)$ (36) $\displaystyle=\lim_{s\to t}\left[\partial_{t}\left(q_{T-s|T-t}(x|\tilde{x})\right)\frac{q_{T-t}(\tilde{x})}{q_{T-s}(x)}+q_{T-s|T-t}(x|\tilde{x})\frac{\partial_{t}q_{T-t}(\tilde{x})}{q_{T-s}(x)}\right]$ (37) $\displaystyle=\lim_{s\to t}\partial_{t}\left(q_{T-s|T-t}(x|\tilde{x})\right)\frac{q_{T-t}(\tilde{x})}{q_{T-s}(x)}$ (38) $\displaystyle=R_{T-t}(\tilde{x},x)\frac{q_{T-t}(\tilde{x})}{q_{T-t}(x)}$ (39) $$

Re-labeling the time-indices on the rate matrices, we obtain

$$ $\hat{R}_{t}(x,\tilde{x})=R_{t}(\tilde{x},x)\frac{q_{t}(\tilde{x})}{q_{t}(x)}$ (40) $$

Now we write the marginal ratio $\frac{q_{t}(\tilde{x})}{q_{t}(x)}$ in a different form

$$ $\displaystyle\frac{q_{t}(\tilde{x})}{q_{t}(x)}$ $\displaystyle=\sum_{x_{0}}\frac{p_{\textup{data}}(x_{0})}{q_{t}(x)}q_{t|0}(\tilde{x}|x_{0})$ (41) $\displaystyle=\sum_{x_{0}}\frac{q_{0|t}(x_{0}|x)}{q_{t|0}(x|x_{0})}q_{t|0}(\tilde{x}|x_{0}).$ (42) $$

Substituting in this form for the marginal ratio concludes the proof.

∎

###### Proof.

### B.2 Proof of Proposition 2

In this section, we detail two proofs for Proposition [2](#Thmproposition2). The first is a formal proof using results from stochastic processes. We then provide a second informal proof for the same result to gain intuition into the $\mathcal{L}_{\textup{CT}}$ objective that only relies on elementary results from CTMCs.

#### Proof 1 - Stochastic Processes

###### Proof.

Let us write $\mathbb{Q}$ for the path measure of the forward CTMC with rate matrix $R_{t}$, $\hat{\mathbb{Q}}$ for the path measure of its exact time reversal and $\mathbb{P}^{\theta}$ for the path measure of the approximate reverse process with rate matrix $\hat{R}^{\theta}_{t}$. Also, we use superscripts to notate conditioning on the starting point, for example $\mathbb{Q}^{x_{0}}$ denotes the path measure of the forward process conditioned to start in $x_{0}$.

With this notation, we have

$$ $\displaystyle-\log p_{0}^{\theta}(x_{0})$ $\displaystyle=-\log\int p_{\textrm{ref}}(\mathrm{d}x_{T})\int_{\{\hat{W}_{T}=x_{0}\}}\mathbb{P}^{\theta,x_{T}}(\mathrm{d}w)$ (43) $\displaystyle=-\log\int q_{T|0}(\mathrm{d}x_{T})\int_{\{\hat{W}_{T}=x_{0}\}}\hat{\mathbb{Q}}^{x_{T}}(\mathrm{d}w)\frac{\mathrm{d}p_{\textrm{ref}}}{\mathrm{d}q_{T|0}}(x_{T})\frac{\mathrm{d}\mathbb{P}^{\theta,x_{T}}}{\mathrm{d}\hat{\mathbb{Q}}^{x_{T}}}(w)$ (44) $\displaystyle=-\log\int q_{T|0}(\mathrm{d}x_{T})\int\hat{\mathbb{Q}}(\mathrm{d}w|\hat{W}_{0}=x_{T},\hat{W}_{T}=x_{0})\frac{\mathrm{d}p_{\textrm{ref}}}{\mathrm{d}q_{T|0}}(x_{T})\frac{\mathrm{d}\mathbb{P}^{\theta,x_{T}}}{\mathrm{d}\hat{\mathbb{Q}}^{x_{T}}}(w)\mathbb{Q}^{x_{T}}\{\hat{W}_{0}=x_{0}\}$ (45) $\displaystyle\leq\int q_{T|0}(\mathrm{d}x_{T})\int\hat{\mathbb{Q}}(\mathrm{d}w|\hat{W}_{0}=x_{T},\hat{W}_{T}=x_{0})\left\{-\log\frac{\mathrm{d}\mathbb{P}^{\theta,x_{T}}}{\mathrm{d}\hat{\mathbb{Q}}^{x_{T}}}(w)\right\}+C,$ (46) $$

where $\mathbb{P}^{\theta},\hat{\mathbb{Q}}$ run in the reverse time direction. Writing $\hat{W}_{s}$ for a reverse path and integrating wrt $p_{\textup{data}}(\mathrm{d}x_{0})$ we have

$$ $\displaystyle\int p_{\textup{data}}(x_{0})[-\log p_{0}^{\theta}(x_{0})]$ $\displaystyle\leq\int p_{\textup{data}}(x_{0})\int q_{T|0}(\mathrm{d}x_{T})\int\hat{\mathbb{Q}}(\mathrm{d}\hat{W}|\hat{W}_{0}=x_{T},\hat{W}_{T}=x_{0})$ (47) $\displaystyle\qquad\times\left\{\int_{s=0}^{T}\hat{R}_{T-s}^{\theta}(\hat{W}_{s})\mathrm{d}s-\sum_{s:\hat{W}_{s-}\neq\hat{W}_{s}}\log\mathbb{P}_{T-s}^{\theta}(\hat{W}_{s}|\hat{W}_{s-})R_{T-s}^{\theta}(\hat{W}_{s-})\right\}+C,$ (48) $$

where $\hat{R}_{t}^{\theta}(x)$ is shorthand for $-\hat{R}_{t}^{\theta}(x,x)$.

When $x_{0}\sim p_{\textup{data}},x_{T}\sim q_{T|0}(\cdot|x_{0}),\hat{W}\sim\hat{\mathbb{Q}}(\mathrm{d}W|\hat{W}_{0}=x_{T},\hat{W}_{T}=x_{0})$,
the reverse path is distributed according to $p_{\textup{data}}(\mathrm{d}x_{0})\mathbb{Q}_{x_{0}}(\mathrm{d}W)$ and therefore
$(\hat{W}_{s-},\hat{W}_{s})$ is distributed like $(W_{T-s},W_{(T-s)-})$
and thus we have

$$ $\displaystyle\int p_{\textup{data}}(x_{0})[-\log p_{0}^{\theta}(x_{0})]$ (49) $\displaystyle\leq\int p_{\textup{data}}(x_{0})\mathbb{Q}_{x_{0}}(\mathrm{d}W)\left\{\int_{s=0}^{T}\hat{R}_{T-s}^{\theta}({W}_{(T-s)-})\mathrm{d}s-\sum_{s:{W}_{(T-s)-}\neq{W}_{T-s}}\log\mathbb{P}^{\theta}_{T-s}({W}_{(T-s)-}|{W}_{T-s})\hat{R}_{T-s}^{\theta}({W}_{T-s})\right\}+C$ (50) $$

Using Dynkin’s lemma and the fact that
$\mathbb{P}_{t}^{\theta}(x|y)\hat{R}_{t}^{\theta}(y)=\hat{R}_{t}^{\theta}(y,x)$ we can re-expresss this final line as

$$ $\displaystyle=\iint_{s=0}^{T}q_{T-s}(\mathrm{d}x)\left\{\sum_{z\neq x}\hat{R}_{T-s}^{\theta}(x,z)-\sum_{z\neq x}{R}_{T-s}(x,z)\frac{\sum_{y\neq x}{R}_{T-s}(x,y)}{\sum_{z\neq x}{R}_{T-s}(x,z)}\log\hat{R}_{T-s}^{\theta}(y,x)\right\}$ (52) $\displaystyle=\iint_{s=0}^{T}q_{T-s}(\mathrm{d}x)r_{T-s}(\mathrm{d}y|x)\left\{\sum_{z\neq x}\hat{R}_{T-s}^{\theta}(x,z)-\sum_{z\neq x}{R}_{T-s}(x,z)\log\hat{R}_{T-s}^{\theta}(y,x)\right\}$ (53) $\displaystyle=\iint_{s=0}^{T}q_{s}(\mathrm{d}x)r_{s}(\mathrm{d}y|x)\left\{\sum_{z\neq x}\hat{R}_{s}^{\theta}(x,z)-\sum_{z\neq x}{R}_{s}(x,z)\log\hat{R}_{s}^{\theta}(y,x)\right\}$ (54) $$

which rearranges to give the continuous time ELBO in the form of Proposition [2](#Thmproposition2).

∎

###### Proof.

#### Proof 2 - Limit of Discrete Time ELBO

###### Proof.

Consider a partitioning of $[0,T]$, $0=t_{0}<t_{1}<\dots<t_{k-1}<t_{k}<t_{k+1}<\dots<t_{K-1}<t_{K}=T$. Let $t_{k}-t_{k-1}=\Delta t$ for all $k$. In subscripts we use $k$ as a shorthand for $t_{k}$ when this does not cause confusion. Considering a CTMC with this time partitioning converts the problem into a discrete time Markov Chain with forward transition kernel, $q_{k+1|k}(x_{k+1}|x_{k})$ and parameterized reverse kernel, $p^{\theta}_{k|k+1}(x_{k}|x_{k+1})$. Therefore, we can write the negative ELBO in its discrete time form, $\mathcal{L}_{\textup{DT}}$

$$ $\displaystyle\textstyle\mathcal{L}_{\textup{DT}}(\theta)=\mathbb{E}_{p_{\textup{data}}(x_{0})}\Big{[}$ $\displaystyle\text{KL}(q_{K|0}(x_{K}|x_{0})||p_{\textrm{ref}}(x_{K}))-\mathbb{E}_{q_{1|0}(x_{1}|x_{0})}\left[\log p^{\theta}_{0|1}(x_{0}|x_{1})\right]$ (55) $\displaystyle+\sum_{k=1}^{K-1}\mathbb{E}_{q_{k+1|0}(x_{k+1}|x_{0})}\left[\text{KL}(q_{k|k+1,0}(x_{k}|x_{k+1},x_{0})||p_{k|k+1}^{\theta}(x_{k}|x_{k+1}))\right]\Big{]}$ (56) $$

In the following, we will write the transition kernels in terms of the CTMC rate matrices and take the limit as $\Delta t\rightarrow 0$ to obtain a continuous time negative ELBO.

First, consider one item from the inner sum of $\mathcal{L}_{\textup{DT}}$

$$ $\displaystyle L_{k}$ $\displaystyle=\mathbb{E}_{p_{\textup{data}}(x_{0})q_{k+1|0}(x_{k+1}|x_{0})}\left[\text{KL}(q_{k|k+1,0}(x_{k}|x_{k+1},x_{0})||p_{k|k+1}^{\theta}(x_{k}|x_{k+1}))\right]$ (57) $\displaystyle=-\mathbb{E}_{p_{\textup{data}}(x_{0})q_{k+1|0}(x_{k+1}|x_{0})q_{k|k+1,0}(x_{k}|x_{k+1},x_{0})}\left[\log p_{k|k+1}^{\theta}(x_{k}|x_{k+1})\right]+C$ (58) $\displaystyle=-\mathbb{E}_{q_{k}(x_{k})q_{k+1|k}(x_{k+1}|x_{k})}\left[\log p_{k|k+1}^{\theta}(x_{k}|x_{k+1})\right]+C$ (59) $$

where we have absorbed terms that do not depend on $\theta$ into $C$. We now write $p_{k|k+1}^{\theta}(x_{k}|x_{k+1})$ in terms of $\hat{R}_{k}^{\theta}$.

$$ $p_{k|k+1}^{\theta}(x_{k}|x_{k+1})=\delta_{x_{k},x_{k+1}}+\hat{R}^{\theta}_{k}(x_{k+1},x_{k})\Delta t+o(\Delta t)$ (60) $$

$$ $\displaystyle\log p_{k|k+1}^{\theta}(x_{k}|x_{k+1})=$ $\displaystyle\log\left(\delta_{x_{k},x_{k+1}}+\hat{R}^{\theta}_{k}(x_{k+1},x_{k})\Delta t+o(\Delta t)\right)$ (61) $\displaystyle=$ $\displaystyle\delta_{x_{k},x_{k+1}}\log\left(1+\hat{R}_{k}^{\theta}(x_{k},x_{k})\Delta t+o(\Delta t)\right)$ (62) $\displaystyle\,+(1-\delta_{x_{k},x_{k+1}})\log\left(\hat{R}_{k}^{\theta}(x_{k+1},x_{k})\Delta t+o(\Delta t)\right)$ (63) $\displaystyle=$ $\displaystyle\delta_{x_{k},x_{k+1}}\left(\hat{R}_{k}^{\theta}(x_{k},x_{k})\Delta t+o(\Delta t)\right)$ (64) $\displaystyle\,+(1-\delta_{x_{k},x_{k+1}})\log\left(\hat{R}_{k}^{\theta}(x_{k+1},x_{k})\Delta t+o(\Delta t)\right)$ (65) $$

where on the last line we have used the series expansion for $\log(1+z)=z-\frac{z^{2}}{2}+o(z^{2})$ valid for $|z|\leq 1,z\neq-1$. For any finite $R_{k}^{\theta}(x_{k},x_{k})$, $\Delta t$ can be taken small enough such that the series expansion holds. We now substitute this form for $\log p_{k|k+1}^{\theta}$ into $L_{k}$ and further write the expectation over $q_{k+1|k}(x_{k+1}|x_{k})=\delta_{x_{k},x_{k+1}}+R_{k}(x_{k},x_{k+1})\Delta t+o(\Delta t)$ as an explicit sum.

$$ $\displaystyle L_{k}=-\mathbb{E}_{q_{k}(x_{k})}\Bigg{[}\sum_{x_{k+1}}\Bigg{\{}$ $\displaystyle\Big{[}\delta_{x_{k},x_{k+1}}+R_{k}(x_{k},x_{k+1})\Delta t+o(\Delta t)\Big{]}\times$ (66) $\displaystyle\Big{[}\delta_{x_{k},x_{k+1}}\left(\hat{R}_{k}^{\theta}(x_{k},x_{k})\Delta t+o(\Delta t)\right)$ (67) $\displaystyle\quad+\left(1-\delta_{x_{k},x_{k+1}}\right)\log\left(\hat{R}_{k}^{\theta}(x_{k+1},x_{k})\Delta t+o(\Delta t)\right)\Big{]}\Bigg{\}}\Bigg{]}+C$ (68) $$

$$ $\displaystyle L_{k}=-\mathbb{E}_{q_{k}(x_{k})}\Bigg{[}\sum_{x_{k+1}}\Bigg{\{}$ $\displaystyle\delta_{x_{k},x_{k+1}}\hat{R}_{k}^{\theta}(x_{k},x_{k})\Delta t$ (69) $\displaystyle+(1-\delta_{x_{k},x_{k+1}})R_{k}(x_{k},x_{k+1})\Delta t\times$ (70) $\displaystyle\,\,\log\Big{(}\hat{R}_{k}^{\theta}(x_{k+1},x_{k})\Delta t+o(\Delta t)\Big{)}+o(\Delta t)\Bigg{\}}\Bigg{]}+C$ (71) $$

We can isolate $\hat{R}_{k}^{\theta}$ within the $\log$ through the following re-arrangement

$$ $\displaystyle\Delta t\log\left(\hat{R}_{k}^{\theta}(x_{k+1},x_{k})\Delta t+o(\Delta t)\right)$ (72) $\displaystyle=\Delta t\log\Delta t+\Delta t\log\left(\hat{R}_{k}^{\theta}(x_{k+1},x_{k})+o(1)\right)$ (73) $\displaystyle=\Delta t\log\Delta t+\Delta t\log\left(1+o(1)\right)+\Delta t\log\left(\hat{R}_{k}^{\theta}(x_{k+1},x_{k})\right)$ (74) $$

where the first two terms are independent of $\theta$ and tend to 0 0 as $\Delta t\rightarrow 0$. Note that we assume $\hat{R}_{k}^{\theta}(x_{k+1},x_{k})>0$ for $x_{k+1}\neq x_{k}$ pairs which have $R_{k}(x_{k},x_{k+1})>0$. This assumption is valid because, for $x_{k+1}\neq x_{k}$, we have

$$ $\hat{R}_{k}^{\theta}(x_{k+1},x_{k})=R_{k}(x_{k},x_{k+1})\sum_{x_{0}}\frac{q_{k|0}(x_{k}|x_{0})}{q_{k|0}(x_{k+1}|x_{0})}p^{\theta}_{0|k}(x_{0}|x_{k+1})$ (75) $$

and we assume $p_{0|k}^{\theta}(x_{0}|x_{k+1})>0$ which is valid when we parameterize $p_{0|k}^{\theta}$ with a softmax output. We assume an irreducible Markov chain, hence $q_{k|0}>0$ for $t_{k}>0$.

With this re-arrangement, and absorbing constant terms into $C$, we obtain

$$ $\displaystyle L_{k}=-\mathbb{E}_{q_{k}(x_{k})}\Bigg{[}\sum_{x_{k+1}}\Bigg{\{}$ $\displaystyle\delta_{x_{k},x_{k+1}}\hat{R}_{k}^{\theta}(x_{k},x_{k})\Delta t$ (76) $\displaystyle+(1-\delta_{x_{k},x_{k+1}})R_{k}(x_{k},x_{k+1})\Delta t\log\Big{(}\hat{R}_{k}^{\theta}(x_{k+1},x_{k})\Big{)}$ (77) $\displaystyle+o(\Delta t)\Bigg{\}}\Bigg{]}+C$ (78) $$

$$ $L_{k}=-\mathbb{E}_{q_{k}(x_{k})}\left[\hat{R}_{k}^{\theta}(x_{k},x_{k})\Delta t+\sum_{x_{k+1}\neq x_{k}}R_{k}(x_{k},x_{k+1})\Delta t\log\hat{R}_{k}^{\theta}(x_{k+1},x_{k})+o(\Delta t)\right]$ (79) $$

The second term can be re-written so that it is more efficient to approximate with Monte Carlo. Currently the denoising model $p_{0|k}^{\theta}$ has to be evaluated for each term in the sum $\sum_{x_{k+1}\neq x_{k}}$ which would require multiple forward passes of the neural network. We can instead create a new probability distribution to sample from as follows. Define

$$ $r_{k}(x_{k+1}|x_{k})=(1-\delta_{x_{k},x_{k+1}})\frac{R_{k}(x_{k},x_{k+1})}{\mathcal{Z}^{k}(x_{k})}$ (80) $$

where

$$ $\mathcal{Z}^{k}(x_{k})=\sum_{x^{\prime}_{k+1}\neq x_{k}}R_{k}(x_{k},x^{\prime}_{k+1})$ (81) $$

So we now have

$$ $L_{k}=-\mathbb{E}_{q_{k}(x_{k})r_{k}(x_{k+1}|x_{k})}\left[\hat{R}_{k}^{\theta}(x_{k},x_{k})\Delta t+\mathcal{Z}^{k}(x_{k})\Delta t\log\hat{R}_{k}^{\theta}(x_{k+1},x_{k})+o(\Delta t)\right]$ (82) $$

Examining the other terms in $\mathcal{L}_{\textup{DT}}$ we have $\mathbb{E}_{p_{\textup{data}}(x_{0})}\left[\text{KL}(q_{K|0}(x_{K}|x_{0})||p_{\textrm{ref}}(x_{K}))\right]$ which does not depend on $\theta$ and $\mathbb{E}_{q_{1|0}(x_{1}|x_{0})}\left[\log p_{0|1}^{\theta}(x_{0}|x_{1})\right]$ which we expand here

$$ $\displaystyle\mathbb{E}_{q_{1|0}(x_{1}|x_{0})}\left[\log p_{0|1}^{\theta}(x_{0}|x_{1})\right]$ (83) $\displaystyle\hskip 28.45274pt=\sum_{x_{1}}\left\{\delta_{x_{1},x_{0}}+\Delta tR_{1}(x_{0},x_{1})+o(\Delta t)\right\}\log p_{0|1}^{\theta}(x_{0}|x_{1})$ (84) $\displaystyle\hskip 28.45274pt=\log p_{0|1}^{\theta}(x_{0}|x_{0})+\Delta t\sum_{x_{1}}R_{1}(x_{0},x_{1})\log p_{0|1}^{\theta}(x_{0}|x_{1})+o(\Delta t)$ (85) $\displaystyle\hskip 28.45274pt=\Delta t\hat{R}_{1}^{\theta}(x_{0},x_{0})+\Delta t\sum_{x_{1}}R_{1}(x_{0},x_{1})\log p_{0|1}^{\theta}(x_{0}|x_{1})+o(\Delta t)$ (86) $$

where on the final line we have used eq [65](#A2.E65). In summary,

$$ $\displaystyle\mathcal{L}_{\textup{DT}}=$ $\displaystyle\Delta t\mathbb{E}_{p_{\textup{data}}(x_{0})q_{1|0}(x_{1}|x_{0})}\left[-\hat{R}_{1}^{\theta}(x_{0},x_{0})+\sum_{x_{1}}R_{1}(x_{0},x_{1})\log p_{0|1}^{\theta}(x_{0}|x_{1})\right]$ (87) $\displaystyle-\Delta t\sum_{k=1}^{K-1}\mathbb{E}_{q_{k}(x_{k})r_{k}(x_{k+1}|x_{k})}\left[\hat{R}_{k}^{\theta}(x_{k},x_{k})+\mathcal{Z}^{k}(x_{k})\log\hat{R}_{k}^{\theta}(x_{k+1},x_{k})\right]$ (88) $\displaystyle+o(\Delta t)+C$ (89) $$

We now take the limit of $\mathcal{L}_{\textup{DT}}$ as $\Delta t\rightarrow 0$ and $K\rightarrow\infty$.

$$ $\underset{\Delta t\rightarrow 0}{\text{lim}}\mathcal{L}_{\textup{DT}}=\mathcal{L}_{\textup{CT}}=-\int_{0}^{T}\mathbb{E}_{q_{t}(x)r_{t}(\tilde{x}|x)}\left[\hat{R}_{t}^{\theta}(x,x)+\mathcal{Z}^{t}(x)\log\left(\hat{R}_{t}^{\theta}(\tilde{x},x)\right)\right]dt+C$ (90) $$

We can estimate the integral with Monte Carlo if we consider it to be an expectation with respect to a uniform distribution over times $(0,T)$. We also write $\hat{R}_{t}^{\theta}(x,x)$ explicitly as the negative off diagonal row sum to obtain

$$ $\textstyle\mathcal{L}_{\textup{CT}}(\theta)=T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)q_{t}(x)r_{t}(\tilde{x}|x)}\Big{[}\Big{\{}\sum_{x^{\prime}\neq x}\hat{R}_{t}^{\theta}(x,x^{\prime})\Big{\}}-\mathcal{Z}^{t}(x)\log\left(\hat{R}_{t}^{\theta}(\tilde{x},x)\right)\Big{]}+C.$ (91) $$

∎

###### Proof.

### B.3 Proof of Proposition 3

###### Proof.

We assume $q_{t|s}(\bm{x}_{t}^{1:D}|\bm{x}_{s}^{1:D})$ factorizes as $\prod_{d=1}^{D}q_{t|s}(x^{d}_{t}|x_{s}^{d})$ where $q_{t|s}(x_{t}^{d}|x_{s}^{d}),\,\,d=1,\dots,D$ are the transition probabilities for independent singular dimensional CTMCs each with forward rate $R_{t}^{d}(\tilde{x}^{d},x^{d})$. In the following, we will drop time subscripts on $x$ arguments. To find the correspondence between $R_{t}^{1:D}$ and $R_{t}^{d}$, we use the Kolmogorov forward equation

$$ $\partial_{t}q_{t|s}(\bm{x}^{1:D}|\tilde{\bm{x}}^{1:D})=\sum_{\bm{y}^{1:D}}q_{t|s}(\bm{y}^{1:D}|\tilde{\bm{x}}^{1:D})R_{t}^{1:D}(\bm{y}^{1:D},\bm{x}^{1:D})$ (92) $$

Substitute in our factorized form for $q_{t|s}$ into the LHS

$$ $\displaystyle\partial_{t}q_{t|s}(\bm{x}^{1:D}|\tilde{\bm{x}}^{1:D})$ $\displaystyle=\partial_{t}\left\{\prod_{d=1}^{D}q_{t|s}(x^{d}|\tilde{x}^{d})\right\}$ (93) $\displaystyle=\sum_{d=1}^{D}q_{t|s}(\bm{x}^{1:D\backslash d}|\tilde{\bm{x}}^{1:D\backslash d})\partial_{t}q_{t|s}(x^{d}|\tilde{x}^{d})$ (94) $\displaystyle=\sum_{d=1}^{D}q_{t|s}(\bm{x}^{1:D\backslash d}|\tilde{\bm{x}}^{1:D\backslash d})\sum_{y^{d}}q_{t|s}(y^{d}|\tilde{x}^{d})R_{t}^{d}(y^{d},x^{d})$ (95) $\displaystyle=\sum_{d=1}^{D}\sum_{\bm{y}^{1:D}}q_{t|s}(\bm{x}^{1:D\backslash d}|\tilde{\bm{x}}^{1:D\backslash d})q_{t|s}(y^{d}|\tilde{x}^{d})R_{t}^{d}(y^{d},x^{d})\delta_{\bm{x}^{1:D\backslash d},\bm{y}^{1:D\backslash d}}$ (96) $\displaystyle=\sum_{d=1}^{D}\sum_{\bm{y}^{1:D}}q_{t|s}(\bm{y}^{1:D\backslash d}|\tilde{\bm{x}}^{1:D\backslash d})q_{t|s}(y^{d}|\tilde{x}^{d})R_{t}^{d}(y^{d},x^{d})\delta_{\bm{x}^{1:D\backslash d},\bm{y}^{1:D\backslash d}}$ (97) $\displaystyle=\sum_{\bm{y}^{1:D}}q_{t|s}(\bm{y}^{1:D}|\tilde{\bm{x}}^{1:D})\sum_{d=1}^{D}R_{t}^{d}(y^{d},x^{d})\delta_{\bm{x}^{1:D\backslash d},\bm{y}^{1:D\backslash d}}$ (98) $$

We therefore obtain

$$ $\sum_{\bm{y}^{1:D}}q_{t|s}(\bm{y}^{1:D}|\tilde{\bm{x}}^{1:D})R_{t}^{1:D}(\bm{y}^{1:D},\bm{x}^{1:D})=\sum_{\bm{y}^{1:D}}q_{t|s}(\bm{y}^{1:D}|\tilde{\bm{x}}^{1:D})\sum_{d=1}^{D}R_{t}^{d}(y^{d},x^{d})\delta_{\bm{x}^{1:D\backslash d},\bm{y}^{1:D\backslash d}}$ (99) $$

This must be true for all possible factorizable forward process transitions, $q_{t|s}$, including $q_{t|s}(\bm{y}^{1:D}|\tilde{\bm{x}}^{1:D})=\delta_{\bm{y}^{1:D},\tilde{\bm{x}}^{1:D}}$. This choice gives us our forward rate relation

$$ $R_{t}^{1:D}(\tilde{\bm{x}}^{1:D},\bm{x}^{1:D})=\sum_{d=1}^{D}R_{t}^{d}(\tilde{x}^{d},x^{d})\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}}$ (100) $$

Substituting this into our expression for the reverse rate from Proposition [1](#Thmproposition1) we obtain

$$ $\displaystyle\hat{R}_{t}^{1:D}(\bm{x}^{1:D},\tilde{\bm{x}}^{1:D})$ $\displaystyle=\sum_{\bm{x}_{0}^{1:D}}\sum_{d=1}^{D}R_{t}^{d}(\tilde{x}^{d},x^{d})\frac{q_{t}(\tilde{\bm{x}}^{1:D}|\bm{x}_{0}^{1:D})}{q_{t}(\bm{x}^{1:D}|\bm{x}_{0}^{1:D})}q_{0|t}(\bm{x}_{0}^{1:D}|\bm{x}^{1:D})\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}}$ (101) $\displaystyle=\sum_{\bm{x}_{0}^{1:D}}\sum_{d=1}^{D}R_{t}^{d}(\tilde{x}^{d},x^{d})\frac{q_{t|0}(\tilde{x}^{d}|x_{0}^{d})}{q_{t|0}(x^{d}|x_{0}^{d})}q_{0|t}(\bm{x}_{0}^{1:D}|\bm{x}^{1:D})\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}}$ (102) $\displaystyle=\sum_{d=1}^{D}R_{t}^{d}(\tilde{x}^{d},x^{d})\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}}\sum_{x_{0}^{d}}q_{0|t}(x_{0}^{d}|\bm{x}^{1:D})\frac{q_{t|0}(\tilde{x}^{d}|x_{0}^{d})}{q_{t|0}(x^{d}|x_{0}^{d})}\sum_{\bm{x}_{0}^{1:D\backslash d}}q_{0|t}(\bm{x}_{0}^{1:D\backslash d}|x_{0}^{d},\bm{x}^{1:D})$ (103) $\displaystyle=\sum_{d=1}^{D}R_{t}^{d}(\tilde{x}^{d},x^{d})\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}}\sum_{x_{0}^{d}}q_{0|t}(x_{0}^{d}|\bm{x}^{1:D})\frac{q_{t|0}(\tilde{x}^{d}|x_{0}^{d})}{q_{t|0}(x^{d}|x_{0}^{d})}$ (104) $$

∎

###### Proof.

### B.4 Proof of Proposition 4

###### Proof.

By the Kolmogorov forward equation applied to the forwards process, we have

$$ $\partial_{t}q_{t}(x_{t})=\sum_{y}R_{t}(y,x_{t})q_{t}(y)$ (105) $$

In addition, applying the Kolmogorov forward equation to the reverse process, which has the same marginals as the forward but time-reversed, we get

$$ $-\partial_{t}q_{t}(x_{t})=\sum_{y}\hat{R}_{t}(y,x_{t})q_{t}(y)$ (106) $$

Summing these two equations gives

$$ $\sum_{y}\left\{R_{t}(y,x_{t})+\hat{R}_{t}(y,x_{t})\right\}q_{t}(y)=0$ (107) $$

Therefore, by comparison with the Kolmogorov equation, $R_{t}+\hat{R}_{t}$ is the rate matrix of a CTMC with invariant distribution $q_{t}$.
∎

###### Proof.

### B.5 Proof of Theorem 1

In this section, we derive a bound on the error of our tau-leaping diffusion model. Because the tau-leaping approximation is only interesting in the case where multiple jumps are made along different dimensions in a single step, we choose to make the dependence of our bound on the dimension of our model explicit, rather than simply considering the case of fixed $D$ and $\tau\rightarrow 0$.

Recall from the main text that we have a time-homogeneous rate matrix $R_{t}$ on $\mathcal{X}$, from which we construct the factorised rate matrix $R_{t}^{1:D}$ on $\mathcal{X}^{D}$ by setting $R^{d}_{t}=R_{t}$ for each $d$, and will denote $|R|=\sup_{t\in[0,T],x\in\mathcal{X}}|R_{t}(x,x)|$, and let $t_{\textup{mix}}$ be the (1/4)-mixing time of the CTMC with rate $R_{t}$. We also define addition on the state space $\mathcal{X}^{D}$ using a mapping from $\mathcal{X}$ to $\mathbb{Z}$ as in Section [4.3](#S4.SS3) and component-wise addition.

###### Theorem 1 .

For any $D\geq 1$ and distribution $p_{\textup{data}}$ on $\mathcal{X}^{D}$, let $\{x_{t}\}_{t\in[0,T]}$ be a CTMC starting in $p_{\textup{data}}$ with rate matrix $R^{1:D}_{t}$ as above. Suppose that $\hat{R}_{t}^{\theta\,1:D}$ is an approximation to the reverse rate matrix and let $(y_{k})_{k=0,1,\dots,N}$ be a tau-leaping approximation to the reverse dynamics with maximum step size $\tau$. Suppose further that there is some constant $M>0$ independent of $D$ such that

$$ $\sum_{y\neq x}\left|\hat{R}_{t}^{1:D}(x,y)-\hat{R}_{t}^{\theta\,1:D}(x,y)\right|\leq M$ (108) $$

for all $t\in[0,T]$. Then under the assumptions listed below, there are constants $C_{1},C_{2}>0$ depending on $\mathcal{X}$ and $R_{t}$ but not $D$ such that, if $\mathcal{L}(y_{0})$ denotes the law of $y_{0}$, we have the total variation bound

$$ $\textstyle||\mathcal{L}(y_{0})-p_{\textup{data}}||_{\textup{TV}}\leq 3MT+\left\{\big{(}|R|SDC_{1}\big{)}^{2}+\frac{1}{2}C_{2}(M+C_{1}SD|R|)\right\}\tau T+2\exp\left\{-\frac{T\log^{2}2}{t_{\textup{mix}}\log 4D}\right\}$ (109) $$

The above theorem holds under the following assumptions, where we write $x\sim y$ for $x,y\in S^{D}$ if they differ in at most one coordinate.

###### Assumption 1 .

The data distribution $p_{\textup{data}}$ is strictly positive.

###### Assumption 2 .

There exists a constant $C_{1}>0$, depending on $S$ and $R_{t}$ but not $D$, such that for all $t\in[0,T]$ and $x,y\in S^{D}$ such that $x\sim y$, we have

$$ $\frac{q_{t}(x)}{q_{t}(y)}\leq C_{1}.$ (110) $$

###### Assumption 3 .

There exists a constant $C_{2}>0$, depending on $S$ and $R_{t}$ but not $D$, such that for all $t\in[0,T]$ and all $x,y\in S^{D}$ such that $x\sim y$, we have

$$ $\sum_{z}\left|\hat{R}_{t}(x,x+z)-\hat{R}_{t}(y,y+z)\right|\leq C_{2}.$ (111) $$

If instead we were to allow $C_{1}$ and $C_{2}$ to depend on the dimension $D$, then Assumptions [2](#Thmassumption2) and [3](#Thmassumption3) follow trivially from Assumption [1](#Thmassumption1) and the finiteness of the state space. However, we choose the stronger formulation above in order to make explicit the dependence of the error bound on the dimension, as previously explained.

As remarked in the main text, in most cases of practical interest (including the two examples explored in Section [6](#S6)), Assumption [3](#Thmassumption3) holds only approximately. However, we still expect the bound in Assumption [3](#Thmassumption3) to hold whenever $x,y$ are in addition chosen such that the tau-leaping approximation of the reverse process makes a jump between them with reasonably high probability. For example, in the case where our data is ordinal, we expect that for any $x\sim y$ jumps from $x$ to $y$ are only common when $x$ is close to $y$, and thus $\hat{R}_{t}(x,x+z)$ and $\hat{R}_{t}(y,y+z)$ should be reasonably close whenever a jump from $x$ to $y$ occurs. Under a weaker assumption of this form, the proof of Theorem [1](#Thmtheorem1) can be adapted to work along similar lines, at the cost of a significant increase in technicality. We therefore choose to focus on the simpler case where Assumption [3](#Thmassumption3) holds as it illustrates the key ideas.

In order to prove Theorem [1](#Thmtheorem1) we will require the following lemmas.

###### Proposition 5 .

Let $(x_{t})_{t\in[0,T]}$ and $(y_{t})_{t\in[0,T]}$ be continuous time Markov chains on a finite state space $S$ with generators $G_{t}$ and $H_{t}$ respectively which are both bounded and continuous in $t$. Let the Markov kernels associated to $X$ and
$Y$ be $K$ and $L$ respectively. Then for any probability distribution $\nu$ on $S$ we have

$$ $||\nu K-\nu L||_{\textup{TV}}\leq\int_{0}^{T}\sup_{x\in S}\Big{\{}\sum_{y\neq x}|G_{t}(x,y)-H_{t}(x,y)|\Big{\}}\;\mathrm{d}t$ (112) $$

###### Proof.

We define a coupling of $(x_{t})_{t\in[0,T]}$ and $(y_{t})_{t\in[0,T]}$ as follows, based on the construction in Chapter 20.1 of . First take $Z\sim\nu$ and set $x_{0}=y_{0}=Z$. Also define the variables $\tilde{x}_{0}=\tilde{y}_{0}=Z$.

Next, fix $\lambda$ such that $|G_{t}(x,x)|,|H_{t}(x,x)|\leq\lambda$ for all $x\in S$, $t\in[0,T]$, let $(N_{s})_{1\leq s\leq T}$ be a Poisson process on $[0,T]$ of rate $\lambda$, and set $N_{0}=0$. We write $N=N_{T}$, and
$S_{1},S_{2},\dots,S_{N}$ for the arrival times and set $S_{n+1}=T$.
We construct $x_{t}$ and $y_{t}$ for $t>0$ inductively as follows.
For $t\in[0,S_{1})$ let $x_{t}=y_{t}=x_{0}$.
Let $1\leq j\leq N$. Given $(x_{r}:r<S_{j})$, $(y_{r}:r<S_{j})$, and $\tilde{x}_{j},\tilde{y}_{j}$,
define the following probability measures

$$ $\displaystyle\rho_{j}(\tilde{x}_{j},w):=\begin{cases}\displaystyle G_{S_{j}}(\tilde{x}_{j},w)/\lambda,\quad w\neq\tilde{x}_{j}\\ \displaystyle 1-G_{S_{j}}(\tilde{x}_{j},w)/\lambda,\quad w=\tilde{x}_{j},\end{cases}$ (113) $\displaystyle\rho^{\prime}_{j}(\tilde{y}_{j},w):=\begin{cases}\displaystyle H_{S_{j}}(\tilde{y}_{j},w)/\lambda,\quad w\neq\tilde{y}_{j}\\ \displaystyle 1-H_{S_{j}}(\tilde{y}_{j},w)/\lambda,\quad w=\tilde{y}_{j}.\end{cases}$ (114) $$

Sample $(\tilde{x}_{j+1},\tilde{y}_{j+1})$ from a maximal coupling of $(\rho_{j},\rho^{\prime}_{j})$ and for $t\in[S_{j},S_{j+1})$ set
$x_{t}=\tilde{x}_{j+1}$, $y_{t}=\tilde{y}_{j+1}$. Finally set $x_{T}=x_{S_{N}}$ and $y_{T}=y_{S_{N}}$.

Now, observe that $(x_{t},y_{t})_{t\in[0,T]}$ defined in this way is a coupling of the given Markov chains. Moreover,

$$ $\displaystyle||\nu K-\nu L||_{\textup{TV}}$ $\displaystyle\leq\mathbb{P}(x_{T}\neq y_{T})$ (115) $\displaystyle=\mathbb{E}\left[\sum_{j=1}^{N}\mathbb{I}\left\{x_{s}=y_{s},\,s<S_{j}\right\}\mathbb{I}\left\{x_{S_{j}}\neq y_{S_{j}}\right\}\right]$ (116) $\displaystyle=\sum_{n=0}^{\infty}\frac{\lambda^{n}e^{-\lambda}}{n!}\sum_{j=0}^{n}\mathbb{E}\left[\mathbb{I}\left\{x_{s}=y_{s},\,s<S_{j}\right\}\mathbb{I}\left\{x_{S_{j}}\neq y_{S_{j}}\right\}\right]$ (117) and using the fact that jumps are coupled maximally $\displaystyle=\sum_{n=0}^{\infty}\frac{\lambda^{n}e^{-\lambda}}{n!}\sum_{j=0}^{n}\mathbb{E}\Big{[}\mathbb{I}\left\{x_{s}=y_{s},\,s<S_{j}\right\}\times\|\rho_{j}(X_{S_{j-1}},\cdot)-\tilde{\rho}_{j}(X_{S_{j-1}},\cdot)\|_{\mathrm{TV}}\Big{]}$ (118) $\displaystyle=\sum_{n=0}^{\infty}\frac{\lambda^{n}e^{-\lambda}}{n!}\sum_{j=0}^{n}\mathbb{E}\Big{[}\mathbb{I}\left\{x_{s}=y_{s},\,s<S_{j}\right\}\frac{1}{\lambda}\sum_{z}|G_{S_{j}}(x_{S_{j-1}},z)-H_{S_{j}}(x_{S_{j-1}},z)|\Big{]}$ (119) $\displaystyle=\frac{1}{\lambda}\mathbb{E}\Big{[}\sum_{s:x_{s}\neq x_{s-}}\sum_{z}|G_{s}(x_{s-},z)-H_{s}(x_{s-},z)|\Big{]}$ (120) $\displaystyle=\frac{1}{\lambda}\int_{s=0}^{T}\mathbb{E}\Big{[}\lambda\sum_{z}|G_{s}(x_{s-},z)-H_{s}(x_{s-},z)|\Big{]}$ (121) $\displaystyle=\int_{s=0}^{T}\mathbb{E}\Big{[}\sum_{z}|G_{s}(x_{s-},z)-H_{s}(x_{s-},z)|\Big{]}\mathrm{d}s$ (122) $$

as required.
∎

###### Proposition 6 .

For all $t\in[0,T]$ and $x,y\in\mathcal{X}^{D}$ such that $x\sim y$, we have

$$ $|\partial_{t}\hat{R}_{t}(x,y)|\leq 2|R|^{2}SDC_{1}^{2}$ (123) $$

Moreover, it follows that $\hat{R}_{t}$ is bounded and continuous in $t$.

###### Proof.

Omitting the superscripts for brevity where the notation is clear, we have

$$ $\displaystyle\left|\partial_{t}\hat{R}_{t}^{1:D}(x^{1:D},y^{1:D})\right|$ $\displaystyle=\left|R_{t}(y,x)\partial_{t}\left\{\frac{q_{t}(y)}{q_{t}(x)}\right\}\right|$ (124) $\displaystyle=\left|R_{t}(y,x)\left\{\frac{q_{t}(y)}{q_{t}(x)}\frac{\sum_{z}R_{t}(z,y)q_{t}(z)}{q_{t}(y)}-\frac{q_{t}(y)}{q_{t}(x)}\frac{\sum_{z}R_{t}(z,x)q_{t}(z)}{q_{t}(x)}\right\}\right|$ (125) $\displaystyle\leq 2|R|^{2}SDC_{1}^{2}$ (126) $$

where the second line follows from Kolmogorov’s forward equation and the final inequality follows from Assumption [2](#Thmassumption2) plus the fact that $R_{t}(z,x)$ (resp. $R_{t}(z,y)$) is only non-zero when $x\sim z$ (resp. $y\sim z$), and there are at most $|S||D|$ values of $x$ (resp. $y$) for which this holds.
∎

We now give the proof of Theorem [1](#Thmtheorem1).

###### Proof of Theorem 1 .

Let us label the time steps used in tau-leaping by $0=t_{0}<t_{1}<\dots<t_{N}=T$, denote $\tau_{k}=t_{k}-t_{k-1}$, and denote the target stationary distribution by $\pi^{D}(x^{1:D})=\prod_{d=1}^{D}\pi(x^{d})$, where $\pi$ is the invariant distribution of the single-dimensional transition matrix $R^{1}_{t}$.

Also, let $\mathcal{R}^{\theta,(\tau)}_{k}$ be the Markov kernel corresponding to applying the tau-leaping approximation with rate matrix $\hat{R}^{\theta}_{t_{k}}$ to move from $t_{k}$ to $t_{k-1}$, and denote $\mathcal{R}^{\theta,(\tau)}=\mathcal{R}^{\theta,(\tau)}_{N}\mathcal{R}^{\theta,(\tau)}_{N-1}\dots\mathcal{R}^{\theta,(\tau)}_{1}$ so that $\mathcal{R}^{\theta,(\tau)}$ expresses the full dynamics of the tau-leaping process and we have $\mathcal{L}(\hat{y}_{0})=\pi^{D}\mathcal{R}^{\theta,(\tau)}$.

Then, as in we can decompose

$$ $||\pi^{D}\mathcal{R}^{\theta,(\tau)}-p_{d}||_{\textup{TV}}\leq||\pi^{D}\mathcal{R}^{\theta,(\tau)}-\pi^{D}(\mathbb{P}^{R})_{T|0}||_{\textup{TV}}+||\pi^{D}-q_{T}||_{\textup{TV}}$ (127) $$

where $\mathbb{P}^{R}$ is the path measure of the exact reverse process.

We deal with the second term first. Let $t_{\textup{mix}}$ be the (1/4)-mixing time of the single-dimension CTMC with rate matrix $R^{1}_{t}$, i.e.

$$ $t_{\textup{mix}}=\inf\left\{t\geq 0:\sup_{x_{0}^{1}\in S}||q_{t|0}(\;\cdot\;|x_{0}^{1})-\pi||_{\textup{TV}}\leq\frac{1}{4}\right\}$ (128) $$

It then follows from

$$ $||q_{t|0}(\;\cdot\;|x_{0}^{1:D})-\pi^{D}||_{\textup{TV}}\leq\sum_{d=1}^{D}||q_{t|0}(\;\cdot\;|x_{0}^{d})-\pi||_{\textup{TV}}$ (129) $$

that $t_{\textup{mix}}^{D}$, the (1/4)-mixing time of the full CTMC with rate matrix $R^{1:D}_{t}$, satisfies the inequality $t^{D}_{mix}\leq\{1+\lceil\log_{2}D\rceil\}t_{\textup{mix}}$. If we view $(x_{mt^{D}_{mix}})_{m\in\mathbb{N}}$ as a discrete-time Markov chain, then standard results on Markov chain mixing (see, for example, Chapter 4.5 of ) show that

$$ $||q_{mt^{D}_{mix}|0}(\;\cdot\;|x_{0}^{1:D})-\pi^{D}||_{\textup{TV}}\leq 2^{-m}$ (130) $$

It then follows that for any $T\geq 0$ we have

$$ $||\pi^{D}-q_{T}||_{\textup{TV}}\leq 2\exp\left\{-\frac{T\log 2}{t^{D}_{mix}}\right\}\leq 2\exp\left\{-\frac{T\log^{2}2}{t_{\textup{mix}}\log 4D}\right\}$ (131) $$

completing the bound on the second term.

To bound the first term, we define $\mathcal{P}_{k}=(\mathbb{P}^{R})_{T-t_{k-1}|T-t_{k}}$ and decompose it as

$$ $\displaystyle||\pi\mathcal{R}^{\theta,(\tau)}-\pi(\mathbb{P}^{R})_{T|0}||_{\textup{TV}}$ $\displaystyle\leq\sup_{\nu}||\nu\mathcal{R}^{\theta,(\tau)}_{N}\dots\mathcal{R}^{\theta,(\tau)}_{1}-\nu\mathcal{P}_{N}\dots\mathcal{P}_{1}||_{\textup{TV}}$ (132) $\displaystyle\leq\sup_{\nu}||\nu\mathcal{R}^{\theta,(\tau)}_{N}\mathcal{R}^{\theta,(\tau)}_{N-1}\dots\mathcal{R}^{\theta,(\tau)}_{1}-\nu\mathcal{R}^{\theta,(\tau)}_{N}\mathcal{P}_{N-1}\dots\mathcal{P}_{1}||_{\textup{TV}}$ (133) $\displaystyle\hskip 14.22636pt+\sup_{\nu}||\nu\mathcal{R}^{\theta,(\tau)}_{N}\mathcal{P}_{N-1}\dots\mathcal{P}_{1}-\nu\mathcal{P}_{N}\mathcal{P}_{N-1}\dots\mathcal{P}_{1}||_{\textup{TV}}$ (134) $\displaystyle\leq\sup_{\nu}||\nu\mathcal{R}^{\theta,(\tau)}_{N-1}\dots\mathcal{R}^{\theta,(\tau)}_{1}-\nu\mathcal{P}_{N-1}\dots\mathcal{P}_{1}||_{\textup{TV}}+\sup_{\nu}||\nu\mathcal{R}^{\theta,(\tau)}_{N}-\nu\mathcal{P}_{N}||_{\textup{TV}}$ (135) $\displaystyle\leq\sum_{k=1}^{N}\sup_{\nu}||\nu\mathcal{R}^{\theta,(\tau)}_{k}-\nu\mathcal{P}_{k}||_{\textup{TV}}$ (136) $$

by proceeding inductively. So it suffices to find bounds on the total variation distance accumulated on each interval $[t_{k-1},t_{k}]$.

Let $\mathcal{R}_{k}^{\theta}$ be the Markov kernel corresponding to running the chain from $t_{k}$ to $t_{k-1}$ with constant rate matrix $\hat{R}^{\theta}_{t_{k}}$. Since by Proposition [6](#Thmproposition6) the reverse rate matrix $\hat{R}_{t}$ is bounded and continuous in $t$, using Proposition [5](#Thmproposition5) we made deduce that for any distribution $\nu$ on $S$ we have

$$ $\displaystyle||\nu\mathcal{P}_{k}-\nu\mathcal{R}_{k}^{\theta}||_{\textup{TV}}$ $\displaystyle\leq\int_{t_{k-1}}^{t_{k}}\sup_{x\in S}\Big{\{}\sum_{y\neq x}\big{|}\hat{R}_{t}(x,y)-\hat{R}^{\theta}_{t_{k}}(x,y)\big{|}\Big{\}}\;\mathrm{d}t$ (137) $\displaystyle\leq\int_{t_{k-1}}^{t_{k}}\sup_{x\in S}\Big{\{}\sum_{y\neq x}\big{|}\hat{R}_{t}(x,y)-\hat{R}_{t_{k}}(x,y)\big{|}\Big{\}}\;\mathrm{d}t$ (138) $\displaystyle\hskip 14.22636pt+\int_{t_{k-1}}^{t_{k}}\sup_{x\in S}\Big{\{}\sum_{y\neq x}\big{|}\hat{R}_{t_{k}}(x,y)-\hat{R}^{\theta}_{t_{k}}(x,y)\big{|}\Big{\}}\;\mathrm{d}t$ (139) $$

The first half of this expression can be bounded using the Mean Value Theorem, according to

$$ $\displaystyle\int_{t_{k-1}}^{t_{k}}\sup_{x\in S}\Big{\{}\sum_{y\neq x}\big{|}\hat{R}_{t}(x,y)-\hat{R}_{t_{k}}(x,y)\big{|}\Big{\}}\;\mathrm{d}t$ $\displaystyle\leq\int_{t_{k-1}}^{t_{k}}|t-t_{k}|\cdot 2|R|^{2}S^{2}D^{2}C_{1}^{2}\;\mathrm{d}t$ (140) $\displaystyle\leq\big{(}|R|SDC_{1}\tau_{k}\big{)}^{2}$ (141) $$

where in the first line we have used that the summand is only non-zero when $y\sim x$, and there are at most $|S||D|$ values of $y$ for which this holds. The second term can be bounded using condition ([18](#S4.E18)), to get

$$ $\int_{t_{k-1}}^{t_{k}}\sup_{x\in S}\Big{\{}\sum_{y\neq x}\big{|}\hat{R}_{t_{k}}(x,y)-\hat{R}^{\theta}_{t_{k}}(x,y)\big{|}\Big{\}}\;\mathrm{d}t\leq M\tau_{k}$ (142) $$

Combining these two expressions, we get a bound on $||\nu\mathcal{P}_{k}-\nu\mathcal{R}_{k}^{\theta}||_{\textup{TV}}$.

$$ $||\nu\mathcal{P}_{k}-\nu\mathcal{R}_{k}^{\theta}||_{\textup{TV}}\leq\big{(}|R|SDC_{1}\tau_{k}\big{)}^{2}+M\tau_{k}$ (143) $$

It remains to bound $||\nu\mathcal{R}^{\theta}_{k}-\nu\mathcal{R}^{\theta,(\tau)}_{k}||_{\textup{TV}}$. Note that performing tau-leaping with rate matrix $\hat{R}_{t_{k}}^{\theta}$ starting in $x_{t_{k}}$ is equivalent to running a continuous time Markov chain from time $t_{k}$ to $t_{k-1}$ with constant rate matrix $\hat{R}^{\theta,(\tau)}_{t_{k}}$ given by

$$ $\hat{R}^{\theta,(\tau)}_{t_{k}}(x,y)=\hat{R}^{\theta}_{t_{k}}(x_{t_{k}},y-x+x_{t_{k}})$ (144) $$

(followed potentially by a clamping operation to keep us within $\mathcal{X}^{D}$). By an analogous argument to the proof of Proposition [5](#Thmproposition5),

$$ $||\delta_{x_{t_{k}}}\mathcal{R}^{\theta}_{k}-\delta_{x_{t_{k}}}\mathcal{R}^{\theta,(\tau)}_{k}||_{\textup{TV}}\leq\int_{t_{k-1}}^{t_{k}}\mathbb{E}\Big{[}\sum_{y\neq x_{t}}|\hat{R}_{t_{k}}^{\theta}(x_{t},y)-\hat{R}^{\theta}_{t_{k}}(x_{t_{k}},y-x_{t}+x_{t_{k}})|\Big{]}\;\mathrm{d}t$ (145) $$

where the expectation is taken over $(x_{t})_{t\in[t_{k-1},t_{k}]}$ distributed according to the exact CTMC with rate matrix $\hat{R}_{t_{k}}^{\theta}$. (Note we have disregarded the clamping operation, since this can only decrease the resulting total variation distance.)

We may rewrite this bound in terms of the exact reverse process using condition ([18](#S4.E18)) to get

$$ $||\delta_{x_{t_{k}}}\mathcal{R}^{\theta}_{k}-\delta_{x_{t_{k}}}\mathcal{R}^{\theta,(\tau)}_{k}||_{\textup{TV}}\leq\int_{t_{k-1}}^{t_{k}}\mathbb{E}\Big{[}2M+\sum_{y\neq x_{t}}|\hat{R}_{t_{k}}(x_{t},y)-\hat{R}_{t_{k}}(x_{t_{k}},y-x_{t}+x_{t_{k}})|\Big{]}\;\mathrm{d}t$ (146) $$

Let $J_{t}$ be the number of jumps that $(x_{t})$ makes between $t_{k}$ and $t$, and label the times of these jumps as $s_{1},\dots,s_{j}$ where $t\leq s_{1}\leq\dots\leq s_{j}\leq t_{k}$ and $j=J_{t}$ for convenience. Then by Assumption [3](#Thmassumption3), we have

$$ $\displaystyle\sum_{y\neq x_{t}}|\hat{R}_{t_{k}}(x_{t},y)-\hat{R}_{t_{k}}(x_{t_{k}},y-x_{t}+x_{t_{k}})|$ $\displaystyle\leq\sum_{z}|\hat{R}_{t_{k}}(x_{t},x_{t}+z)-\hat{R}_{t_{k}}(x_{s_{1}},x_{s_{1}}+z)|+\dots$ (147) $\displaystyle\hskip 14.22636pt+\sum_{z}|\hat{R}_{t_{k}}(x_{s_{j}},x_{s_{j}}+z)-\hat{R}_{t_{k}}(x_{t_{k}},x_{t_{k}}+z)|$ (148) $\displaystyle\leq C_{2}J_{t}$ (149) $$

where we have made the substitution $z=y-x_{t_{k}}$. We conclude that

$$ $\displaystyle||\delta_{x_{t_{k}}}\mathcal{R}^{\theta}_{k}-\delta_{x_{t_{k}}}\mathcal{R}^{\theta,(\tau)}_{k}||_{\textup{TV}}$ $\displaystyle\leq\int_{t_{k-1}}^{t_{k}}\mathbb{E}\left[2M+C_{2}J_{t}\right]\;\mathrm{d}t$ (150) $\displaystyle\leq 2M|t_{k}-t_{k-1}|+C_{2}\int_{t_{k-1}}^{t_{k}}|t_{k}-t|\cdot\sup_{x}|\hat{R}^{\theta}_{t_{k}}(x,x)|\;\mathrm{d}t$ (151) $\displaystyle\leq 2M\gamma_{k}+\frac{1}{2}C_{2}|\hat{R}^{\theta}_{t_{k}}|\tau_{k}^{2}$ (152) $\displaystyle\leq 2M\tau_{k}+\frac{1}{2}C_{2}(M+C_{1}SD|R|)\tau_{k}^{2}$ (153) $$

where to bound $\mathbb{E}[J_{t}]$ we have observed that jumps of $(x_{t})$ occur at a rate bounded above by $\sup_{x}|\hat{R}^{\theta}_{t_{k}}(x,x)|$, and in the last line we have used the condition ([18](#S4.E18)) and Assumption [2](#Thmassumption2). Since the above holds for any choice of $x_{t_{k}}$, it follows that

$$ $\sup_{\nu}||\nu\mathcal{R}^{\theta}_{k}-\nu\mathcal{R}^{\theta,(\tau)}_{k}||_{\textup{TV}}\leq 2M\tau_{k}+\frac{1}{2}C_{2}(M+C_{1}SD|R|)\tau_{k}^{2}$ (154) $$

Summing over $k$ and putting all our bounds together, we get

$$ $\textstyle||\mathcal{L}(y_{0})-p_{\textup{data}}||_{\textup{TV}}\leq 3MT+\left\{\big{(}|R|SDC_{1}\big{)}^{2}+\frac{1}{2}C_{2}(M+C_{1}SD|R|)\right\}\tau T+2\exp\left\{-\frac{T\log^{2}2}{t_{\textup{mix}}\log 4D}\right\}$ (155) $$

as required.
∎

###### Theorem 1 .

###### Assumption 1 .

###### Assumption 2 .

###### Assumption 3 .

###### Proposition 5 .

###### Proof.

###### Proposition 6 .

###### Proof.

###### Proof of Theorem 1 .

## Appendix C Continuous Time ELBO Details

### C.1 Comparison with the Discrete Time ELBO

It is easiest to gain intuition on the $\mathcal{L}_{\textup{CT}}$ objective by comparing it to its discrete time counterpart, $\mathcal{L}_{\textup{DT}}$, and examining the way in which $\mathcal{L}_{\textup{DT}}$ in the limit becomes $\mathcal{L}_{\textup{CT}}$ when we take the time step size to be very small. We repeat the definition of $\mathcal{L}_{\textup{CT}}$ here for convenience

$$ $\textstyle\mathcal{L}_{\textup{CT}}(\theta)=T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)q_{t}(x)r_{t}(\tilde{x}|x)}\Big{[}\Big{\{}\sum_{x^{\prime}\neq x}\hat{R}_{t}^{\theta}(x,x^{\prime})\Big{\}}-\mathcal{Z}^{t}(x)\log\left(\hat{R}_{t}^{\theta}(\tilde{x},x)\right)\Big{]}+C.$ (156) $$

Recall that a single term from the KL sum in $\mathcal{L}_{\textup{DT}}$ up to an additive constant independent of $\theta$ is

$$ $\textstyle-\mathbb{E}_{q_{k}(x_{k})q_{k+1|k}(x_{k+1}|x_{k})}\left[\log p_{k|k+1}^{\theta}(x_{k}|x_{k+1})\right].$ (157) $$

Minimizing this term is to sample $(x_{k},x_{k+1})$ from the forward dynamics and then maximize the assigned model probability for the pairing in the reverse direction. A similar idea can be used to understand $\mathcal{L}_{\textup{CT}}$. First, we write $\log p_{k|k+1}^{\theta}(x_{k}|x_{k+1})$ in terms of $\hat{R}_{k}^{\theta}$ as

$$ $\displaystyle\log p_{k|k+1}^{\theta}(x_{k}|x_{k+1})=$ $\displaystyle\delta_{x_{k},x_{k+1}}\left(\hat{R}_{k}^{\theta}(x_{k},x_{k})\Delta t+o(\Delta t)\right)$ (158) $\displaystyle\,+(1-\delta_{x_{k},x_{k+1}})\log\left(\hat{R}_{k}^{\theta}(x_{k+1},x_{k})\Delta t+o(\Delta t)\right)$ (159) $$

where we have separated the cases when $x_{k}=x_{k+1}$ and when $x_{k}\neq x_{k+1}$ (see the proof of $\mathcal{L}_{\textup{CT}}$ for the full details). The first term will become the $\sum_{x^{\prime}\neq x}\hat{R}_{t}^{\theta}(x,x^{\prime})$ term in $\mathcal{L}_{\textup{CT}}$ whilst the second term will become the $\mathcal{Z}^{t}(x)\log\big{(}\hat{R}_{t}^{\theta}(\tilde{x},x)\big{)}$ term. Now, when we minimize $\mathcal{L}_{\textup{CT}}$, we are sampling $(x,\tilde{x})$ from the forward process and then maximizing the assigned model probability for the pairing in the reverse direction, just as in $\mathcal{L}_{\textup{DT}}$. The slight extra complexity comes from the fact we are considering the case when $x_{k}=x_{k+1}$ and the case when $x_{k}\neq x_{k+1}$ separately. When $x_{k}=x_{k+1}$, this corresponds to the first term in $\mathcal{L}_{\textup{CT}}$ which we can see is minimizing the reverse rate out of $x$ which is exactly maximizing the model probability for no transition to occur. When $x_{k}\neq x_{k+1}$, this corresponds to the second term in $\mathcal{L}_{\textup{CT}}$, which is maximizing the reverse rate from $\tilde{x}$ to $x$ which in turn maximizes the model probability for the $\tilde{x}$ to $x$ transition to occur.

### C.2 Conditional Form

For the conditional form of $\mathcal{L}_{\textup{CT}}$, denoted as $\bar{\mathcal{L}}_{\textup{CT}}$, we instead upper bound the negative conditional model log-likelihood, $\mathbb{E}_{p_{\textup{data}}(x_{0},y)}[-\log p_{0}^{\theta}(x_{0}|y)]$ where $y$ is our conditioner. $\bar{\mathcal{L}}_{\textup{CT}}$ has the following form

$$ $\textstyle\bar{\mathcal{L}}_{\textup{CT}}(\theta)=T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)p_{\textup{data}}(x_{0},y)q_{t|0}(x|x_{0})r_{t}(\tilde{x}|x)}\Big{[}\Big{\{}\sum_{x^{\prime}\neq x}\hat{R}_{t}^{\theta}(x,x^{\prime}|y)\Big{\}}-\mathcal{Z}^{t}(x)\log\left(\hat{R}_{t}^{\theta}(\tilde{x},x|y)\right)\Big{]}+C,$ (160) $$

where

$$ $\displaystyle\textstyle\hat{R}_{t}^{\theta}(x,\tilde{x}|y)=$ $\displaystyle R_{t}(\tilde{x},x)\sum_{x_{0}}\frac{q_{t|0}(\tilde{x}|x_{0})}{q_{t|0}(x|x_{0})}p^{\theta}_{0|t}(x_{0}|x,y)\quad\text{for}\quad x\neq\tilde{x}.$ (161) $\displaystyle=$ $\displaystyle-\sum_{x^{\prime}\neq x}\hat{R}_{t}^{\theta}(x,x^{\prime}|y)\quad\text{for}\quad x=\tilde{x}$ (162) $$

This follows easily from considering the conditional form of the discret time ELBO, $\bar{\mathcal{L}}_{\textup{DT}}$ and using the same arguments as before to go from discrete time to continuous time.

$$ $\mathbb{E}_{p_{\textup{data}}(x_{0},y)}[-\log p_{0}^{\theta}(x_{0}|y)]\leq\mathbb{E}_{p_{\textup{data}}(x_{0},y)q_{1:K|0}(x_{1:K}|x_{0})}\left[-\log\frac{p_{0:K}^{\theta}(x_{0:K}|y)}{q_{1:K|0}(x_{1:K}|x_{0})}\right]=\bar{\mathcal{L}}_{\textup{DT}}$ (163) $$

### C.3 Continuous Time ELBO with Factorization Assumptions

In the following Proposition, we show the form of $\mathcal{L}_{\textup{CT}}$ when we use a factorized forward process. We note that in the proof we rearrange the sampling distribution from $p_{\textup{data}}(\bm{x}_{0}^{1:D})q_{t|0}(\bm{x}^{1:D}|\bm{x}_{0}^{1:D})r_{t}(\tilde{\bm{x}}^{1:D}|\bm{x}^{1:D})$ to $p_{\textup{data}}(\bm{x}_{0}^{1:D})\psi_{t}(\tilde{\bm{x}}^{1:D}|\bm{x}_{0}^{1:D})\phi_{t}(\bm{x}^{1:D}|\tilde{\bm{x}}^{1:D},\bm{x}_{0}^{1:D})$. This is not strictly necessary but it allows us to analytically sum over the intermediate $\bm{x}^{1:D}$ variable which greatly reduces the variance of the resulting objective.

###### Proposition 7 .

The $\mathcal{L}_{\textup{CT}}$ objective when we substitute in the factorized forms for the forward and reverse process given in Proposition [3](#Thmproposition3) is

$$ $\displaystyle\mathcal{L}_{\textup{CT}}$ $\displaystyle=T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)p_{\textup{data}}(\bm{x}_{0}^{1:D})q_{t|0}(\bm{x}^{1:D}|\bm{x}_{0}^{1:D})}\left[\sum_{d=1}^{D}\sum_{x^{\prime d}\neq x^{d}}\hat{R}_{t}^{\theta\,d}(\bm{x}^{1:D},x^{\prime d})\right]$ (164) $\displaystyle-T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)p_{\textup{data}}(\bm{x}_{0}^{1:D})\psi_{t}(\tilde{\bm{x}}^{1:D}|\bm{x}_{0}^{1:D})}\Bigg{[}\sum_{d=1}^{D}\sum_{x^{d}\neq\tilde{x}^{d}}\phi_{t}(x^{d}|\tilde{\bm{x}}^{1:D},\bm{x}_{0}^{1:D})\mathcal{Z}^{t}(\tilde{\bm{x}}^{1:D/d}\circ x^{d})\log\left(\hat{R}_{t}^{\theta\,d}(\tilde{\bm{x}}^{1:D},x^{d})\right)\Bigg{]}$ (165) $\displaystyle+C$ (166) $$

with

$$ $\hat{R}_{t}^{\theta\,d}(\bm{x}^{1:D},\tilde{x}^{d})=R_{t}^{d}(\tilde{x}^{d},x^{d})\sum_{x_{0}^{d}}p_{0|t}^{\theta}(x_{0}^{d}|\bm{x}^{1:D})\frac{q_{t|0}(\tilde{x}^{d}|x_{0}^{d})}{q_{t|0}(x^{d}|x_{0}^{d})}$ (167) $$

$$ $\mathcal{Z}^{t}(\bm{x}^{1:D})=\sum_{d=1}^{D}\sum_{\tilde{x}^{d}\neq x^{d}}R_{t}^{d}(x^{d},\tilde{x}^{d})$ (168) $$

$$ $\phi_{t}(x^{d}|\tilde{\bm{x}}^{1:D},\bm{x}_{0}^{1:D})=\frac{R_{t}^{d}(x^{d},\tilde{x}^{d})q_{t|0}(\tilde{\bm{x}}^{1:D\backslash d}\circ x^{d}|\bm{x}_{0}^{1:D})}{\mathcal{Z}^{t}(\tilde{\bm{x}}^{1:D\backslash d}\circ x^{d})\sum_{d^{\prime}=1}^{D}\sum_{x^{\prime d^{\prime}}\neq\tilde{x}^{d^{\prime}}}\frac{R_{t}^{d^{\prime}}(x^{\prime d^{\prime}},\tilde{x}^{d^{\prime}})}{\mathcal{Z}^{t}(\tilde{\bm{x}}^{1:D\backslash{d^{\prime}}}\circ x^{\prime d^{\prime}})}q_{t|0}(\tilde{\bm{x}}^{1:D\backslash d^{\prime}}\circ x^{\prime d^{\prime}}|\bm{x}_{0}^{1:D})}$ (169) $$

where $\circ$ represents the concatenation of a $D-1$ dimensional vector, $\bm{x}^{1:D\backslash d}$ with a scalar $x^{d}$, such that the resultant $D$ dimensional vector has $x^{d}$ at its $d^{\textup{th}}$ dimension.
$\psi_{t}(\tilde{\bm{x}}^{1:D}|\bm{x}_{0}^{1:D})$ is defined as the marginal of the forward noising process joint, $\int q_{t|0}(\bm{x}^{1:D}|\bm{x}_{0}^{1:D})r_{t}(\tilde{\bm{x}}^{1:D}|\bm{x}^{1:D})d\bm{x}^{1:D}$.

###### Proof.

We first re-write the general form of $\mathcal{L}_{\textup{CT}}$ here

$$ $\textstyle\mathcal{L}_{\textup{CT}}(\theta)=T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)q_{t}(x)r_{t}(\tilde{x}|x)}\Big{[}\Big{\{}\sum_{x^{\prime}\neq x}\hat{R}_{t}^{\theta}(x,x^{\prime})\Big{\}}-\mathcal{Z}^{t}(x)\log\left(\hat{R}_{t}^{\theta}(\tilde{x},x)\right)\Big{]}+C$ (170) $$

where

$$ $\textstyle\mathcal{Z}^{t}(x)=\sum_{x^{\prime}\neq x}R_{t}(x,x^{\prime})\hskip 56.9055ptr_{t}(\tilde{x}|x)=(1-\delta_{\tilde{x},x})R_{t}(x,\tilde{x})/\mathcal{Z}^{t}(x).$ (171) $$

With a factorized forward process, $\hat{R}_{t}^{\theta}$ becomes

$$ $\hat{R}_{t}^{\theta\,1:D}(\bm{x}^{1:D},\tilde{\bm{x}}^{1:D})=\sum_{d=1}^{D}\hat{R}_{t}^{\theta\,d}(\bm{x}^{1:D},\tilde{x}^{d})\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}}$ (172) $$

where

$$ $\hat{R}_{t}^{\theta\,d}(\bm{x}^{1:D},\tilde{x}^{d})=R_{t}^{d}(\tilde{x}^{d},x^{d})\sum_{x_{0}^{d}}p_{0|t}^{\theta}(x_{0}^{d}|\bm{x}^{1:D})\frac{q_{t|0}(\tilde{x}^{d}|x_{0}^{d})}{q_{t|0}(x^{d}|x_{0}^{d})}$ (173) $$

Substituting this form for $\hat{R}_{t}^{\theta\,1:D}$ into the first term in $\mathcal{L}_{\textup{CT}}$ we get

$$ $\displaystyle\sum_{\bm{x}^{\prime 1:D}\neq\bm{x}^{1:D}}\sum_{d=1}^{D}\hat{R}_{t}^{\theta\,d}(\bm{x}^{1:D},x^{\prime d})\delta_{\bm{x}^{1:D\backslash d},\bm{x}^{\prime 1:D\backslash d}}$ (174) $\displaystyle=\sum_{d=1}^{D}\sum_{x^{\prime d}}\hat{R}_{t}^{\theta\,d}(\bm{x}^{1:D},x^{\prime d})\sum_{\bm{x}^{\prime 1:D\backslash d}}\delta_{\bm{x}^{1:D\backslash d},\bm{x}^{\prime 1:D\backslash d}}(1-\delta_{\bm{x}^{\prime 1:D},\bm{x}^{1:D}})$ (175) $\displaystyle=\sum_{d=1}^{D}\sum_{x^{\prime d}\neq x^{d}}\hat{R}_{t}^{\theta\,d}(\bm{x}^{1:D},x^{\prime d})$ (176) $$

Now we tackle the second term in $\mathcal{L}_{\textup{CT}}$. We first re-arrange the distribution over which we take the expectation:

$$ $p_{\textup{data}}(\bm{x}_{0}^{1:D})q_{t|0}(\bm{x}^{1:D}|\bm{x}_{0}^{1:D})r_{t}(\tilde{\bm{x}}^{1:D}|\bm{x}^{1:D})=p_{\textup{data}}(\bm{x}_{0}^{1:D})\psi_{t}(\tilde{\bm{x}}^{1:D}|\bm{x}_{0}^{1:D})\phi_{t}(\bm{x}^{1:D}|\tilde{\bm{x}}^{1:D},\bm{x}_{0}^{1:D})$ (177) $$

We have,

$$ $\displaystyle\phi_{t}(\bm{x}^{1:D}|\tilde{\bm{x}}^{1:D},\bm{x}_{0}^{1:D})$ $\displaystyle\propto q_{t|0}(\bm{x}^{1:D}|\bm{x}_{0}^{1:D})r_{t}(\tilde{\bm{x}}^{1:D}|\bm{x}^{1:D})$ (178) $\displaystyle=q_{t|0}(\bm{x}^{1:D}|\bm{x}_{0}^{1:D})(1-\delta_{\tilde{\bm{x}}^{1:D},\bm{x}^{1:D}})\frac{\sum_{d=1}^{D}R_{t}^{d}(x^{d},\tilde{x}^{d})\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}}}{\mathcal{Z}^{t}(\bm{x}^{1:D})}$ (179) $\displaystyle=\sum_{d=1}^{D}\frac{R_{t}^{d}(x^{d},\tilde{x}^{d})}{\mathcal{Z}^{t}(\tilde{\bm{x}}^{1:D\backslash d}\circ x^{d})}q_{t|0}(\tilde{\bm{x}}^{1:D\backslash d}\circ x^{d}|\bm{x}_{0}^{1:D})\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}}(1-\delta_{\tilde{\bm{x}}^{1:D},\bm{x}^{1:D}})$ (180) $$

To find the normalization constant, we can sum the proportional term over $\bm{x}^{1:D}$

$$ $\displaystyle\sum_{\bm{x}^{1:D}}\sum_{d=1}^{D}\frac{R_{t}^{d}(x^{d},\tilde{x}^{d})}{\mathcal{Z}^{t}(\tilde{\bm{x}}^{1:D\backslash d}\circ x^{d})}q_{t|0}(\tilde{\bm{x}}^{1:D\backslash d}\circ x^{d}|\bm{x}_{0}^{1:D})\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}}(1-\delta_{\tilde{\bm{x}}^{1:D},\bm{x}^{1:D}})$ (182) $\displaystyle\quad=\sum_{d=1}^{D}\sum_{x^{d}\neq\tilde{x}^{d}}\frac{R_{t}^{d}(x^{d},\tilde{x}^{d})}{\mathcal{Z}^{t}(\tilde{\bm{x}}^{1:D\backslash d}\circ x^{d})}q_{t|0}(\tilde{\bm{x}}^{1:D\backslash d}\circ x^{d}|\bm{x}_{0}^{1:D})$ (183) $$

Therefore,

$$ $\phi_{t}(\bm{x}^{1:D}|\tilde{\bm{x}}^{1:D},\bm{x}_{0}^{1:D})=(1-\delta_{\tilde{\bm{x}}^{1:D},\bm{x}^{1:D}})\sum_{d=1}^{D}\phi_{t}(x^{d}|\tilde{\bm{x}}^{1:D},\bm{x}_{0}^{1:D})\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}}$ (184) $$

where

$$ $\phi_{t}(x^{d}|\tilde{\bm{x}}^{1:D},\bm{x}_{0}^{1:D})=\frac{R_{t}^{d}(x^{d},\tilde{x}^{d})q_{t|0}(\tilde{\bm{x}}^{1:D\backslash d}\circ x^{d}|\bm{x}_{0}^{1:D})}{\mathcal{Z}^{t}(\tilde{\bm{x}}^{1:D\backslash d}\circ x^{d})\sum_{d^{\prime}=1}^{D}\sum_{x^{\prime d^{\prime}}\neq\tilde{x}^{d^{\prime}}}\frac{R_{t}^{d^{\prime}}(x^{\prime d^{\prime}},\tilde{x}^{d^{\prime}})}{\mathcal{Z}^{t}(\tilde{\bm{x}}^{1:D\backslash{d^{\prime}}}\circ x^{\prime d^{\prime}})}q_{t|0}(\tilde{\bm{x}}^{1:D\backslash d^{\prime}}\circ x^{\prime d^{\prime}}|\bm{x}_{0}^{1:D})}$ (185) $$

Now we write the second term as

$$ $\displaystyle T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)p_{\textup{data}}(\bm{x}_{0}^{1:D})\psi_{t}(\tilde{\bm{x}}^{1:D}|\bm{x}_{0}^{1:D})}\left[-\sum_{\bm{x}^{1:D}}\phi_{t}(\bm{x}^{1:D}|\tilde{\bm{x}}^{1:D},\bm{x}_{0}^{1:D})\mathcal{Z}^{t}(\bm{x}^{1:D})\log\hat{R}_{t}^{\theta\,1:D}(\tilde{\bm{x}}^{1:D},\bm{x}^{1:D})\right]$ (186) $\displaystyle=-T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)p_{\textup{data}}(\bm{x}_{0}^{1:D})\psi_{t}(\tilde{\bm{x}}^{1:D}|\bm{x}_{0}^{1:D})}\Bigg{[}\sum_{d=1}^{D}\sum_{x^{d}\neq\tilde{x}^{d}}\phi_{t}(x^{d}|\tilde{\bm{x}}^{1:D},\bm{x}_{0}^{1:D})\mathcal{Z}^{t}(\tilde{\bm{x}}^{1:D/d}\circ x^{d})\log\left(\hat{R}_{t}^{\theta\,d}(\tilde{\bm{x}}^{1:D},x^{d})\right)\Bigg{]}$ (187) $$

∎

###### Proposition 7 .

###### Proof.

### C.4 One Forward Pass

To evaluate the $\mathcal{L}_{\textup{CT}}$ objective, we naively need to perform two forward passes of the denoising network: $p_{0|t}^{\theta}(x_{0}|x)$ to calculate $\hat{R}_{t}^{\theta}(x,x^{\prime})$ and $p_{0|t}^{\theta}(x_{0}|\tilde{x})$ to calculate $\hat{R}_{t}^{\theta}(\tilde{x},x)$. This is wasteful because $\tilde{x}$ is created from $x$ by applying a single forward transition which on multi-dimensional problems means $\tilde{x}$ differs from $x$ in only a single dimension. To exploit the fact that $\tilde{x}$ and $x$ are very similar, we approximate the sample $x\sim q_{t}(x)$ with the sample $\tilde{x}\sim\sum_{x}q_{t}(x)r_{t}(\tilde{x}|x)$. This gives the more efficient objective,

$$ $\textstyle\mathcal{L}_{\textup{eCT}}(\theta)=T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)q_{t}(x)r_{t}(\tilde{x}|x)}\left[\left\{\sum_{x^{\prime}\neq\tilde{x}}\hat{R}_{t}^{\theta}(\tilde{x},x^{\prime})\right\}-\mathcal{Z}^{t}(x)\log\left(\hat{R}_{t}^{\theta}(\tilde{x},x)\right)\right]+C$ (188) $$

The approximation is valid because $q_{t}(x)$ and $\sum_{x}q_{t}(x)r_{t}(\tilde{x}|x)$ are very similar distributions, as we now show.

$$ $\displaystyle\sum_{x}q_{t}(x)r_{t}(\tilde{x}|x)$ $\displaystyle=\sum_{x}q_{t}(x)(1-\delta_{x,\tilde{x}})\frac{R_{t}(x,\tilde{x})}{\sum_{x^{\prime}\neq x}R_{t}(x,x^{\prime})}$ (189) $\displaystyle\propto-q_{t}(\tilde{x})R_{t}(\tilde{x},\tilde{x})+\sum_{x}q_{t}(x)R_{t}(x,\tilde{x})$ (190) $\displaystyle=q_{t}(\tilde{x})\sum_{x^{\prime}\neq\tilde{x}}R_{t}(\tilde{x},x^{\prime})+\partial_{t}q_{t}(\tilde{x})$ (191) $\displaystyle\propto q_{t}(\tilde{x})+\frac{1}{\sum_{x^{\prime}\neq x}R_{t}(\tilde{x},x^{\prime})}\partial_{t}q_{t}(\tilde{x})$ (192) $\displaystyle=q_{t}(\tilde{x})+\delta t\,\partial_{t}q_{t}(\tilde{x})$ (193) $$

where on the third line we have used the Kolmogorov forward equation and defined $\delta_{t}=1/\sum_{x^{\prime}\neq x}R_{t}(\tilde{x},x^{\prime})$. The distribution $\sum_{x}q_{t}(x)r_{t}(\tilde{x}|x)$ is therefore $q_{t+\delta t}(\tilde{x})$ approximated using a first-order Taylor expansion around $q_{t}(\tilde{x})$. We notice that $\delta t$ is the average time to the next transition at time $t$. $\delta t$ can be calculated for the practical settings we consider, its varies between $2\times 10^{-6}T$ and $2\times 10^{-8}T$ in the image modelling task and is $1\times 10^{-3}T$ in the monophonic music task.

We perform an ablation experiment comparing between training with $\mathcal{L}_{\textup{eCT}}$ and $\mathcal{L}_{\textup{CT}}$ on the monophonic music dataset, the results are shown in Table [3](#A3.T3). We find that we gain a small boost in performance when using the more efficient $\mathcal{L}_{\textup{eCT}}$ objective alongside the improved efficiency. We hypothesize that this is because of a slight reduction in variance for the $\mathcal{L}_{\textup{eCT}}$ objective due to increased negative correlation between the two terms in the objective when $\tilde{x}$ is shared between them.

**Table 3: Metrics on the monophonic music dataset comparing training with the efficient $\mathcal{L}_{\textup{eCT}}$ objective vs the original $\mathcal{L}_{\textup{CT}}$ objective. We compute these over the test set showing mean$\pm$std with respect to 5 samples for each test song.**
| Model | Hellinger Distance | Proportion of Outliers |
| --- | --- | --- |
| $\tau$LDR-0 Uniform $\mathcal{L}_{\textup{eCT}}$ | $0.3765\pm 0.0013$ | $0.1106\pm 0.0010$ |
| $\tau$LDR-0 Uniform $\mathcal{L}_{\textup{CT}}$ | $0.3797\pm 0.0009$ | $0.1128\pm 0.0007$ |

## Appendix D Direct Denoising Model Supervision

Following , we can introduce direct $p_{0|t}^{\theta}$ supervision into the optimization objective which has been found empirically to improve performance. We first contextualize the change by expressing $\mathcal{L}_{\textup{CT}}$ with the dependence on $p_{0|t}^{\theta}$ made explicit.

$$ $\displaystyle\mathcal{L}_{\textup{CT}}=T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)q_{t}(x)r_{t}(\tilde{x}|x)}\Big{[}$ $\displaystyle\Big{\{}\sum_{x^{\prime}\neq x}R_{t}(x^{\prime},x)\sum_{x_{0}}\frac{q_{t|0}(x^{\prime}|x_{0})}{q_{t|0}(x|x_{0})}p_{0|t}^{\theta}(x_{0}|x)\Big{\}}$ (194) $\displaystyle-\mathcal{Z}^{t}(x)\log\Big{(}R_{t}(x,\tilde{x})\sum_{x_{0}}\frac{q_{t|0}(x|x_{0})}{q_{t|0}(\tilde{x}|x_{0})}p_{0|t}^{\theta}(x_{0}|\tilde{x})\Big{)}\Big{]}+C$ (195) $$

The signal for $p_{0|t}^{\theta}(x_{0}|x)$ comes through a sum over $x_{0}$ weighted by the ratio $\frac{q_{t|0}(x|x_{0})}{q_{t|0}(\tilde{x}|x_{0})}$. We can also provide a direct denoising signal by predicting the clean datapoint $x_{0}$ from the corrupted version $x$ and using the negative log-likelihood loss.

$$ $\textstyle L_{ll}(\theta)=T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)p_{\textup{data}}(x_{0})q_{t|0}(x|x_{0})}\left[-\log p_{0|t}^{\theta}(x_{0}|x)\right]$ (196) $$

###### Proposition 8 .

The true denoising distribution, $q_{0|t}$, minimizes $L_{ll}$

###### Proof.

$$ $\textstyle T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)q_{t}(x)}\left[\text{KL}\left(q_{0|t}(x_{0}|x)\,||\,p_{0|t}^{\theta}(x_{0}|x)\right)\right]$ (197) $\textstyle=T\,\mathbb{E}_{t\sim\mathcal{U}(0,T)p_{\textup{data}}(x_{0})q_{t|0}(x|x_{0})}\left[-\log p_{0|t}^{\theta}(x_{0}|x)\right]+C$ (198) $$

where $C$ is a constant independent of $\theta$. Therefore, minimizing $L_{ll}$ is equivalent to minimizing the KL divergence between $q_{0|t}$ and $p_{0|t}^{\theta}$, which is minimized when $p_{0|t}^{\theta}=q_{0|t}$.
∎

If we obtain the true denoising distribution, $p_{0|t}^{\theta}=q_{0|t}$, then we will have the true reverse rate, $\hat{R}_{t}$. find that optimizing with an objective combining $L_{ll}$ and $\mathcal{L}_{\textup{DT}}$ performs best, which we can also do in continuous time

$$ $\underset{\theta}{\text{min}}\quad\mathcal{L}_{\textup{CT}}(\theta)+\lambda L_{ll}(\theta)$ (199) $$

where $\lambda$ is a hyperparameter. In , it was found that training with $L_{ll}$ alone resulted in poorer performance than when the ELBO was included in the loss. We provide a theoretical hypothesis as to why this may be the case here. We show that minimizing $L_{ll}$ is equivalent to minimizing an upper bound on the negative ELBO in discrete time and thus by training with $L_{ll}$ we are simply minimizing a looser bound on the negative model log-likelihood than if we were to use the negative ELBO directly.

###### Proposition 9 .

Minimizing the sum of negative log-likelihoods

$$ $\sum_{k=0}^{K-1}\mathbb{E}_{p_{\textup{data}}(x_{0})q_{k+1|0}(x_{k+1}|x_{0})}\left[-\log p_{0|k+1}^{\theta}(x_{0}|x_{k+1})\right]$ (200) $$

is equivalent to minimizing an upper bound on the negative ELBO.

###### Proof.

$$ $\textstyle\textstyle\mathcal{L}_{\textup{DT}}(\theta)=\mathbb{E}_{p_{\textup{data}}(x_{0})}\Big{[}$ $\textstyle\text{KL}(q_{K|0}(x_{K}|x_{0})||p_{\textrm{ref}}(x_{K}))-\mathbb{E}_{q_{1|0}(x_{1}|x_{0})}\left[\log p^{\theta}_{0|1}(x_{0}|x_{1})\right]$ (201) $\textstyle+\sum_{k=1}^{K-1}\mathbb{E}_{q_{k+1|0}(x_{k+1}|x_{0})}\left[\text{KL}(q_{k|k+1,0}(x_{k}|x_{k+1},x_{0})||p_{k|k+1}^{\theta}(x_{k}|x_{k+1}))\right]\Big{]}$ (202) $$

Consider one term from the sum

$$ $\displaystyle L_{k}$ $\displaystyle=\mathbb{E}_{p_{\textup{data}}(x_{0})q_{k+1|0}(x_{k+1}|x_{0})}\left[\text{KL}(q_{k|k+1,0}(x_{k}|x_{k+1},x_{0})||p_{k|k+1}^{\theta}(x_{k}|x_{k+1})\right]$ (203) $\displaystyle=\mathbb{E}_{q_{k+1}(x_{k+1})q_{k|k+1}(x_{k}|x_{k+1})}\left[-\log p_{k|k+1}^{\theta}(x_{k}|x_{k+1})\right]$ (204) $\displaystyle\quad+\mathbb{E}_{p_{\textup{data}}(x_{0})q_{k+1|0}(x_{k+1}|x_{0})q_{k|k+1,0}(x_{k}|x_{k+1},x_{0})}\left[\log q_{k|k+1,0}(x_{k}|x_{k+1},x_{0})\right]$ (205) $$

Now,

$$ $\displaystyle\mathbb{E}_{q_{k+1}(x_{k+1})q_{k|k+1}(x_{k}|x_{k+1})}\left[-\log p_{k|k+1}^{\theta}(x_{k}|x_{k+1})\right]$ (206) $\displaystyle\quad=\mathbb{E}_{q_{k+1}(x_{k+1})q_{k|k+1}(x_{k}|x_{k+1})}\left[-\log\sum_{\tilde{x}_{0}}q(x_{k}|\tilde{x}_{0},x_{k+1})p_{0|k+1}^{\theta}(\tilde{x}_{0}|x_{k+1})\right]$ (207) $\displaystyle\quad=\mathbb{E}_{q_{k+1}(x_{k+1})q_{k|k+1}(x_{k}|x_{k+1})}\left[-\log\sum_{\tilde{x}_{0}}\frac{q_{0|k}(\tilde{x}_{0}|x_{k})q_{k|k+1}(x_{k}|x_{k+1})}{q_{0|k+1}(\tilde{x}_{0}|x_{k+1})}p_{0|k+1}^{\theta}(\tilde{x}_{0}|x_{k+1})\right]$ (208) $\displaystyle\quad\leq\mathbb{E}_{q_{k+1}(x_{k+1})q_{k|k+1}(x_{k}|x_{k+1})q_{0|k}(\tilde{x}_{0}|x_{k})}\left[-\log\frac{q_{k|k+1}(x_{k}|x_{k+1})}{q_{0|k+1}(\tilde{x}_{0}|x_{k+1})}p_{0|k+1}^{\theta}(\tilde{x}_{0}|x_{k+1})\right]$ (209) $\displaystyle\quad=\mathbb{E}_{p_{\textup{data}}(x_{0})q_{k+1|0}(x_{k+1}|x_{0})}\left[-\log p_{0|k+1}^{\theta}(x_{0}|x_{k+1})\right]$ (210) $\displaystyle\hskip 28.45274pt+\mathbb{E}_{p_{\textup{data}}(x_{0})q_{k|0}(x_{k}|x_{0})q_{k+1|k}(x_{k+1}|x_{k})}\left[-\log\frac{q_{k|k+1}(x_{k}|x_{k+1})}{q_{0|k+1}(x_{0}|x_{k+1})}\right]$ (211) $$

Therefore,

$$ $\displaystyle\mathcal{L}_{\textup{DT}}\leq$ $\displaystyle\sum_{k=0}^{K-1}\Bigg{\{}\mathbb{E}_{p_{\textup{data}}(x_{0})q_{k+1|0}(x_{k+1}|x_{0})}\left[-\log p_{0|k+1}^{\theta}(x_{0}|x_{k+1})\right]\Bigg{\}}$ (212) $\displaystyle+\mathbb{E}_{p_{\textup{data}}(x_{0})}\left[\text{KL}(q_{K|0}(x_{K}|x_{0})||p_{K}(x_{K}))\right]$ (213) $\displaystyle+\sum_{k=1}^{K-1}\Bigg{\{}\mathbb{E}_{p_{\textup{data}}(x_{0})q_{k+1|0}(x_{k+1}|x_{0})q_{k|k+1,0}(x_{k}|x_{k+1},x_{0})}\left[\log q_{k|k+1,0}(x_{k}|x_{k+1},x_{0})\right]$ (214) $\displaystyle\hskip 42.67912pt+\mathbb{E}_{p_{\textup{data}}(x_{0})q_{k|0}(x_{k}|x_{0})q_{k+1|k}(x_{k+1}|x_{k})}\left[-\log\frac{q_{k|k+1}(x_{k}|x_{k+1})}{q_{0|k+1}(x_{0}|x_{k+1})}\right]\Bigg{\}}$ (215) $$

We can see that only the first term depends on $\theta$.
∎

###### Proposition 8 .

###### Proof.

###### Proposition 9 .

###### Proof.

## Appendix E Choice of Forward Process

We need to choose the structure of $R_{t}$ such that we can analytically obtain $q_{t|0}$ marginals to enable efficient training.

###### Proposition 10 .

If $R_{t}$ and $R_{t^{\prime}}$ commute for all $t$, $t^{\prime}$ then $q_{t|0}(x=j|x_{0}=i)=\big{(}\text{exp}\big{[}\int_{0}^{t}R_{s}ds\big{]}\big{)}_{ij}$ where exp here is understood to be the matrix exponential function.

###### Proof.

Let $(P_{t})_{ij}=q_{t|0}(x=j|x_{0}=i)$. We show that $P_{t}=\text{exp}\left(\int_{0}^{t}R_{s}ds\right)$ is a solution to the Kolmogorov forward equation, which in matrix form reads, $\partial_{t}P_{t}=P_{t}R_{t}$. Writing the matrix exponential in sum form

$$ $\displaystyle P_{t}$ $\displaystyle=\sum_{k=0}^{\infty}\frac{1}{k!}\left(\int_{0}^{t}R_{s}ds\right)^{k}$ (216) $\displaystyle=\mathrm{Id}+\int_{0}^{t}R_{s}ds+\frac{1}{2!}\left(\int_{0}^{t}R_{s}ds\right)^{2}+\frac{1}{3!}\left(\int_{0}^{t}R_{s}ds\right)^{3}+\dots$ (217) $$

Now, differentiating and using the fact that $R_{t}$, $R_{t^{\prime}}$ commute.

$$ $\displaystyle\partial_{t}P_{t}$ $\displaystyle=R_{t}+\int_{0}^{t}R_{s}dsR_{t}+\frac{1}{2!}\left(\int_{0}^{t}R_{s}ds\right)^{2}R_{t}+\dots$ (218) $\displaystyle=\left\{\sum_{k=0}^{\infty}\frac{1}{k!}\left(\int_{0}^{t}R_{s}ds\right)^{k}\right\}R_{t}$ (219) $\displaystyle=P_{t}R_{t}$ (220) $$

∎

As stated in the main text, we achieve the commutative property by selecting $R_{t}=\beta(t)R_{b}$ where $\beta(t)$ is a time dependent scalar and $R_{b}$ is a constant base matrix. We can utilize the eigendecomposition of $R_{b}=Q\Lambda Q^{-1}$ to efficiently calculate $P_{t}$,

$$ $\displaystyle P_{t}$ $\displaystyle=\text{exp}\left(\int_{0}^{t}\beta(s)R_{b}ds\right)$ (221) $\displaystyle=\sum_{k=0}^{\infty}\frac{1}{k!}\left(\int_{0}^{t}\beta(s)R_{b}ds\right)^{k}$ (222) $\displaystyle=\sum_{k=0}^{\infty}\frac{1}{k!}\left(Q\Lambda Q^{-1}\int_{0}^{t}\beta(s)ds\right)^{k}$ (223) $\displaystyle=\sum_{k=0}^{\infty}\frac{1}{k!}Q\left(\Lambda\int_{0}^{t}\beta(s)ds\right)^{k}Q^{-1}$ (224) $\displaystyle=Q\left\{\sum_{k=0}^{\infty}\frac{1}{k!}\left(\Lambda\int_{0}^{t}\beta(s)ds\right)^{k}\right\}Q^{-1}$ (225) $\displaystyle=Q\text{exp}\left[\Lambda\int_{0}^{t}\beta(s)ds\right]Q^{-1}$ (226) $$

Since $\Lambda$ is a diagonal matrix, the matrix exponential coincides with the element wise exponential making the final expression tractable to compute. We choose $\beta(t)=ab^{t}\log b$ because this makes the integral which dictates the variance of $q_{t|0}$ have a simple form $\int_{0}^{t}\beta(s)ds=ab^{t}-a$.

For categorical problems, we found a uniform rate matrix works well, $R_{t}=\beta\mathbbm{1}\mathbbm{1}^{T}-\beta S\mathrm{Id}$.
This is directly analogous to the discrete time uniform transition matrix: $P=\alpha\mathbbm{1}\mathbbm{1}^{T}+(1-S\alpha)\mathrm{Id}$ with $\alpha$ depending on the time discretization used. Indeed, if one calculates the corresponding discrete transition matrix for the uniform $R_{t}$ rate through the matrix exponential, the uniform transition matrix is obtained. Another categorical corruption process is the absorbing state process. In discrete time, the transition matrix is given by $P=\alpha\mathbbm{1}\mathbf{e}_{\ast}^{T}+(1-\alpha)\mathrm{Id}$ where $\mathbf{e}_{\ast}$ is the one-hot encoding of the absorbing state. The corresponding absorbing state continuous time process has transition rate matrix: $R_{t}=\beta\mathbf{1}\mathbf{e}_{\ast}^{T}-\beta\mathrm{Id}$. The correspondence for more complex transition matrices e.g. the Discretized Gaussian matrix in is much harder to find analytically especially if the time inhomogeneous case is considered. For datasets with an ordinal structure, we construct a new rate matrix that maintains a bias towards nearby states using a similar approach as that taken by to construct the Discretized Gaussian matrix.

We construct this matrix by first picking a desired stationary distribution, $p_{\textrm{ref}}$, and then filling in matrix entries such that we encourage transitions to nearby states whilst keeping $p_{\textrm{ref}}$ as our stationary distribution. Specifically, we let $p_{\textrm{ref}}$ be a discretized Gaussian over the state space, i.e.

$$ $p_{\textrm{ref}}(x)\propto\text{exp}\left[-\frac{(x-\mu_{0})^{2}}{2\sigma_{0}^{2}}\right]$ (227) $$

To find a condition on the rate such that this is the case, recall the Kolmogorov differential equation for the marginals

$$ $\partial_{t}q_{t}(x)=\sum_{\tilde{x}}q_{t}(\tilde{x})R_{b}(\tilde{x},x)$ (228) $$

Now, consider a rate that is in detailed balance with $p_{\textrm{ref}}$

$$ $p_{\textrm{ref}}(\tilde{x})R_{b}(\tilde{x},x)=p_{\textrm{ref}}(x)R_{b}(x,\tilde{x})$ (229) $$

Substituting this rate into the Kolmogorov equation, we see that $p_{\textrm{ref}}$ is the stationary distribution

$$ $\displaystyle\partial_{t}p_{\textrm{ref}}(x)$ $\displaystyle=\sum_{\tilde{x}}p_{\textrm{ref}}(\tilde{x})R_{b}(\tilde{x},x)$ (230) $\displaystyle=\sum_{\tilde{x}}p_{\textrm{ref}}(x)R_{b}(x,\tilde{x})$ (231) $\displaystyle=p_{\textrm{ref}}(x)\sum_{\tilde{x}}R_{b}(x,\tilde{x})$ (232) $\displaystyle=0$ (233) $$

where the last line follows from the fact that the row sum of a rate matrix is zero. Note that any $R_{t}=\beta(t)R_{b}$ will also have this stationary distribution as the multiplication by $\beta(t)$ can be seen as just a scaling of the time axis. From the detailed balance equation, we gain a condition on $R_{b}$ such that our desired $p_{\textrm{ref}}$ is the stationary distribution

$$ $\frac{R_{b}(\tilde{x},x)}{R_{b}(x,\tilde{x})}=\frac{p_{\textrm{ref}}(x)}{p_{\textrm{ref}}(\tilde{x})}=\text{exp}\left[\frac{(\tilde{x}-\mu_{0})^{2}}{2\sigma_{0}^{2}}-\frac{(x-\mu_{0})^{2}}{2\sigma_{0}^{2}}\right]$ (234) $$

This gives constraints on diagonal elements within $R_{b}$ but does not fully define the entire matrix. To do this, we first make the assumption that $\mu$ is selected to be at the center of the state space. Then we set off diagonal terms to the right of the diagonal in the top half of the rate matrix and off diagonal terms to the left of the diagonal in the bottom half to be 1. Finally, progressing in from the top and bottom of the rate matrix we make definitions of rate matrix values that have not already been defined by the detailed balance condition.
For clarity, we provide a pictorial representation of this scheme for an $8\times 8$ rate matrix below

$$ $\begin{bmatrix}\cdot&1&\square&\square&\square&\square&\square&\square\\ \triangle&\cdot&1&\square&\square&\square&\square&\triangle\\ \triangle&\triangle&\cdot&1&\square&\square&\triangle&\triangle\\ \triangle&\triangle&\triangle&\cdot&1&\triangle&\triangle&\triangle\\ \triangle&\triangle&\triangle&\triangle&\cdot&\triangle&\triangle&\triangle\\ \triangle&\triangle&\triangle&\square&1&\cdot&\triangle&\triangle\\ \triangle&\triangle&\square&\square&\square&1&\cdot&\triangle\\ \triangle&\square&\square&\square&\square&\square&1&\cdot\\ \end{bmatrix}$ (235) $$

where $\square$ represents a value we will define, $\triangle$ represents a value that is defined relative to another entry through the detailed balance condition and $\cdot$ is a diagonal entry that is equal to the negative off diagonal row sum. We could define $\square$ values to be 0 to gain a sparse rate matrix, however, we found in early experiments that allowing transitions to further away states greatly reduces the mixing time and gives better performance. We define $\square$ in each row similarly, by setting it equal to $\text{exp}[-i^{2}/\sigma_{r}^{2}]$ where $i$ is the distance away from the ‘$1$’ value in that row and $\sigma_{r}$ is a hyperparameter defining the length scale in state space of a typical transition. This biases our forward process to make transitions between nearby states, at a length scale of $\sigma_{r}$.

###### Proposition 10 .

###### Proof.

## Appendix F CTMC Simulation

### F.1 Exact CTMC and Tau-Leaping

In this section, we first describe exact CTMC simulation before giving an algorithmic description of tau-leaping.

When a CTMC has a time-homogeneous rate matrix, we can use Gillespie’s Algorithm to exactly simulate it. This algorithm is based on the jump chain/holding time definition of the CTMC. It repeats the following two steps:

- •
Draw a holding time from an exponential distribution with mean $-1/R(x,x)$ and wait in the current state $x$ for that amount of time.
- •
Sample the next state from $r(\tilde{x}|x)=(1-\delta_{x,\tilde{x}})\frac{R(x,\tilde{x})}{\sum_{x^{\prime}\neq x}R(x,x^{\prime})}$

This Algorithm can be adjusted for the case when we have a time-inhomogeneous rate matrix using the modified next reaction method . However, both algorithms still step through each transition in the CTMC individually and are thus unsuitable in our case because only one dimension would change for each simulation step making it very computationally expensive to produce a sample. Instead we use tau-leaping that allows multiple dimensions to change in a single simulation step. We detail this method in Algorithm [1](#algorithm1).

Figure: Algorithm 1 Generative Reverse Process Simulation with Tau-Leaping

### F.2 Predictor-Corrector Discussion

In this section we compare predictor-corrector sampling schemes as applied to continuous state spaces and discrete state spaces.

The predictor-corrector scheme in continuous state spaces was introduced in . It consists of alternating between a predictor step and a corrector step:

$$ $\displaystyle\text{Predictor}\quad\bm{x}_{i}\leftarrow\bm{x}_{i+1}+\gamma_{i}s_{\theta}(\bm{x}_{i+1},i+1)+\sqrt{\gamma_{i}}z,\quad z\sim\mathcal{N}(0,I)$ (236) $\displaystyle\text{Corrector}\quad\bm{x}_{i}\leftarrow\bm{x}_{i}+\epsilon_{i}s_{\theta}(\bm{x}_{i},i)+\sqrt{2\epsilon_{i}}z,\quad z\sim\mathcal{N}(0,I)$ (237) $$

where $\bm{x}_{i}$ is the state at sampling step $i$, $s_{\theta}$ is the learned score model approximating $\nabla_{\bm{x}}\log q_{t}(\bm{x})$ and $\gamma_{i}$, $\epsilon_{i}$ are the step sizes for the predictor and corrector respectively. We see that both take similar forms, except the corrector adds in a factor $\sqrt{2}$ more Gaussian noise during the update step.

In discrete state spaces, rather than sampling using gradient guided stochastic steps as in the continuous state space case, we sample by simulating CTMCs with defined rates. When we take a predictor step, we simulate using $\hat{R}_{t}^{\theta}$ and when we take a corrector step we simulate using $R_{t}^{c\,\theta}=\hat{R}_{t}^{\theta}+R_{t}$. If we simulate the CTMC exactly, we have seen in the previous section that this amounts to sampling next states from the categorical distribution defined by normalizing the row of the rate matrix corresponding to the current state. Therefore, corrector sampling can be seen as sampling from a slightly noisier categorical distribution defined through $R_{t}^{c\,\theta}$ as compared to the predictor categorical distribution defined through $\hat{R}_{t}^{\theta}$. This is analogous to the increased Gaussian noise applied during a corrector step in continuous state spaces.

Adding corrector steps brings the marginal of the samples closer to $q_{t}(\bm{x})$ and continued application of the corrector will further explore the domain of $q_{t}(\bm{x})$. In previous work on continuous state predictor-corrector methods, the number of corrector steps has been small (e.g. 1 or 2 corrector steps per predictor step) or indeed the corrector steps have been removed altogether. In this work we have found that using up to 10 corrector steps per predictor steps can be beneficial during certain regions of the reverse generative process. Additionally, in continuous state spaces, it has been observed that too many corrector steps can result in unwanted noise in the generated data .

We hypothesize that corrector steps are better utilized in discrete state spaces to explore the domain of $q_{t}(\bm{x})$ than in continuous state spaces. This is because, the corrector update is defined largely through the reverse rate itself, $\hat{R}_{t}^{\theta}$, just with the categorical probabilities being annealed slightly more towards uniform through the addition of the forward rate $R_{t}$. This may be a more effective update than simply adding extra Gaussian noise in the continuous state space case. Furthermore, the denoising model in continuous state spaces can be seen as outputting a point estimate of $\bm{x}_{0}$ of dimension $D$. However, in discrete state spaces, the denoising model outputs a categorical distribution over every dimension (output dimension $D\times S$) allowing it to express some uncertainty information in the $\bm{x}_{0}$ prediction, albeit with conditional independence between the dimensions. Adding corrector steps in discrete state spaces would then allow information to mix between dimensions for the current time step, exploring modes of $q_{t}(\bm{x})$.

We explore this idea on the image modelling task in Figure [7](#A6.F7). We run the reverse generative process until time $t=0.4$ at which point we hold the time constant and apply $1000$ corrector steps. We see that the resulting progression of $\bm{x}_{t}$ states explores potential local modes of $q_{t}(\bm{x})$ in the local region of image space.

Figure: Figure 7: Progression of $\bm{x}_{t}$ for $t=0.4$ by repeated application of corrector steps. In each pair of rows, the top row is $\bm{x}_{t}$ whilst the bottom row is the $\bm{x}_{0}$ prediction made by $p^{\theta}_{0|t}(\bm{x}_{0}|\bm{x}_{t})$ (argmax of the categorical probabilities in each dimension). Each column represents an additional $100$ corrector steps.
Refer to caption: /html/2205.14987/assets/x9.png

## Appendix G Implicit Dimensional Assumptions Made in Discrete Time

In discrete time, the parametric reverse kernel, $p_{k|k+1}^{\theta}$, is commonly defined through a denoising model $p_{0|k+1}^{\theta}$. Here, we examine this definition in the multi-dimensional case where the forward process factorizes, as in Appendix [C.3](#A3.SS3) and previous discrete time work . We begin by writing the true full dimensional reverse kernel, $q_{k|k+1}$, in terms of the true denoising distribution, $q_{0|k+1}$.

$$ $\displaystyle q_{k|k+1}(\bm{x}_{k}^{1:D}|\bm{x}_{k+1}^{1:D})$ $\displaystyle=\prod_{d=1}^{D}q_{k|k+1}(x_{k}^{d}|\bm{x}_{k}^{1:d-1},\bm{x}_{k+1}^{1:D})$ (238) $\displaystyle=\prod_{d=1}^{D}\sum_{x_{0}^{d}}q_{k,0|k+1}(x_{k}^{d},x_{0}^{d}|\bm{x}_{k}^{1:d-1},\bm{x}_{k+1}^{1:D})$ (239) $\displaystyle=\prod_{d=1}^{D}\sum_{x_{0}^{d}}q_{0|k+1}(x_{0}^{d}|\bm{x}_{k}^{1:d-1},\bm{x}_{k+1}^{1:D})q_{k|0,k+1}(x_{k}^{d}|x_{0}^{d},x_{k+1}^{d})$ (240) $$

where on the final line we have used the fact that the forward process is independent across dimensions. To create our approximate reverse kernel, $p_{k|k+1}^{\theta}$, we approximate $q_{0|k+1}(x_{0}^{d}|\bm{x}_{k}^{1:d-1},\bm{x}_{k+1}^{1:D})$ with $p_{0|k+1}^{\theta}(x_{0}^{d}|\bm{x}_{k+1}^{1:D})$,

$$ $p^{\theta}_{k|k+1}(\bm{x}_{k}^{1:D}|\bm{x}_{k+1}^{1:D})=\prod_{d=1}^{D}\sum_{x_{0}^{d}}p^{\theta}_{0|k+1}(x_{0}^{d}|\bm{x}_{k+1}^{1:D})q_{k|0,k+1}(x_{k}^{d}|x_{0}^{d},x_{k+1}^{d})$ (241) $$

We throw away the extra $\bm{x}_{k}^{1:d-1}$ conditioning because we use a non-autoregressive model that takes in $\bm{x}_{k+1}^{1:D}$ and in a single forward pass gives conditionally independent probabilities over $x_{0}^{d}$, $d=1,\dots,D$. For finite $K$, this approximation can never match the true kernel because we are not conditioning on all relevant information. Of course, as $K$ gets larger, this approximation becomes more accurate. Since we operate in the continuous regime, we do not have to make this approximation because the conditionally independent denoising model, $q_{0|t}(x_{0}^{d}|\bm{x}^{1:D})$, appears directly in our reverse rate, $\hat{R}_{t}^{1:D}$, when we factorize the forward process (see Proposition [3](#Thmproposition3)).

## Appendix H Experimental Details

In this section, we provide additional details for the experiments we performed applying our method to practical problems. The code for our models is available at [https://github.com/andrew-cr/tauLDR](https://github.com/andrew-cr/tauLDR). Before describing the specifics for each experiment, we first explain the implementation details common to all.

When we evaluate the objective $\mathcal{L}_{\textup{CT}}$ on each minibatch of training datapoints, we must sample a time for each from $t\sim\mathcal{U}(0,T)$ which represents the point in the forward process which we will noise to. Training instabilities can be found if $t$ is sampled very close to 0 0 because the reverse rate, $\hat{R}_{t}$, becomes ill-conditioned in this region. This phenomenon is also observed in continuous state space models because the score, $\nabla_{x}\log q_{t}(x)$, becomes ill-conditioned close to $t=0$. The reverse rate and score become ill conditioned close to the start of the forward process because the marginal probability, $q_{t}(x)$, will be highly peaked around the data manifold and $\log q_{t}(x)$ will explode in regions that are not close to the data. To avoid these issues, a common trick is to set a minimum time such that $t\sim\mathcal{U}(\epsilon,T)$. $\epsilon$ is set such that the level of noising at $t=\epsilon$ is very small and reverse sampling to this point will produce samples very close to $p_{\textup{data}}$. In our experiments, we set $\epsilon=0.01T$.

During reverse sampling, we use tau-leaping to simulate the reverse process from $t=T$ until $t=\epsilon$ because the reverse rate is not trained for $t<\epsilon$. This produces a sample close to $p_{\textup{data}}$. We found improved performance in metrics such as FID if we then complete a final step to remove the small amount of noise that may still be present in the sample. Specifically, we pass the sample through the denoising model $p_{0|t}^{\theta}(x_{0}|x_{t})$ with $t=\epsilon$ to obtain an output of shape $D\times S$ where $D$ is the dimensionality of the problem. This is a probability distribution over the states for each of the dimensions. We set the value of each dimension to the state with the highest probability. This then produces a sample which has all of the noise removed.

The specific value of $T$ within our model is arbitrary because the forward process can be scaled in the time axis to provide the same noising process for any $T$. Therefore, we simply set $T=1$.

### H.1 Demonstrative Example

Our 2d dataset is created by sampling 1M 2d points from a $32\times 32$ state space with probability proportional to the pixel values of a $32\times 32$ grayscale image of a $\tau$ character.

For our forward process, we use a Gaussian rate (see Appendix [E](#A5)) with stationary distribution standard deviation $\sigma_{0}=8$ and rate length scale $\sigma_{r}=1$. We use a rate schedule of $\beta(t)=5\times 5^{t}\log(5)$.

To represent $p_{0|t}^{\theta}$ we use a residual MLP. The architecture consists of an input linear layer to lift the input dimension of $2$ to the internal network dimension of $16$. Then, there are $2$ residual blocks each consisting of: a single hidden layer MLP of hidden dimension $32$, a residual connection to the input of the MLP, a layer norm, and finally a FiLM layer modulated by the time embedding. At the output, there is a single linear layer with output size of $2\times 32=64$ representing state probabilities in each of the $2$ dimensions. The time is embedded using the Transformer sinusoidal position embedding creating an embedding of size $32$. Then, the embedding is further processed by a single hidden layer MLP with hidden layer size $32$ and output size $128$. To create the FiLM parameters in each residual block, the time embedding is passed through a linear layer with output of size $32$ to provide a multiplicative and additive modulation to the state dimension of $16$. We minimize the $\mathcal{L}_{\textup{CT}}$ objective using Adam with a learning rate of $0.0001$ and batch size of 32 for 1M steps.

For the exact simulation we use the next reaction method with modifications for time dependent transition rates . This method steps through each transition in the exact simulation path individually by calculating the time to the next occurrence of each transition type and applying the transition that occurs soonest. Exact algorithmic details can be found in . To calculate the time to the next occurrence for a transition, we need to integrate the reverse rate matrix (eq (13) in ). We do this with euler integration with a step size of 0.001.

### H.2 Image Modeling

We train on the CIFAR10 training dataset that contains 50000 images of dimension $3\times 32\times 32$. We evaluate the test ELBO on the CIFAR10 test dataset which consists of 10000 images. For the forward noising process, we use the the Gaussian rate (see Appendix [E](#A5)) with stationary distribution standard deviation of $\sigma_{0}=512$ and rate length scale $\sigma_{r}=6$. This effectively defines a uniform stationary distribution since the state space is of size $256$. We found this performs better than a more concentrated Gaussian. Our $\beta$ schedule is $\beta(t)=3\times 100^{t}\log 100$. This was selected in accordance with $\sigma_{r}$ such that the overall shape of progression of the $q_{t|0}$ variances approximately matches that of the schedule proposed in .

Our $p_{0|t}^{\theta}$ model is parameterized with the standard U-net architecture introduced in . The network follows the PixelCNN++ backbone with group normalization layers. There are four feature map resolutions ($32\times 32$ to $4\times 4$) in the downsampling/upsampling stacks. At each resolution there are two convolutional residual blocks. There is a self-attention block between the residual blocks at the $16\times 16$ resolution level . The time is input into the network by first embedding with the Transformer sinusoidal position embedding . This time embedding is passed into each residual block by passing it through a SiLU activation and then a linear layer before adding it onto the hidden state within the residual block between the two convolution operations.

The original architecture of has an output of dimension $3\times 32\times 32$ as it makes a point prediction of $x_{0}$ given $x_{t}$. In order for the model to output probabilities over $x_{0}$ (i.e. an output dimension of $3\times 32\times 32\times 256$) we make the adjustments suggested in . Specifically, we use their truncated logistic distribution parameterization where the model outputs the mean and log scale of a logistic distribution i.e. an output dimension of $3\times 32\times 32\times 2$. The probability for a state is then the integral of this continuous distribution between this state and the next when mapped onto the real line. To impart a residual inductive bias on the output, the mean of the logistic distribution is taken to be $\text{tanh}(x_{t}+\mu^{\prime})$ where $x_{t}$ is the normalized input into the model and $\mu^{\prime}$ is mean outputted from the network. The normalization operation takes the input in the range $0,\dots,255$ and maps it to $[-1,1]$. In total, our network has approximately $35.7$ million parameters.

We optimize with the auxiliary objective described in Appendix [D](#A4) with $\lambda=0.001$. Within the auxiliary objective, we use the one-forward pass version of the continuous time ELBO, $\mathcal{L}_{\textup{eCT}}$. We optimize with Adam for 2M steps with a learning rate of 0.0002 and batch size of 128. We use the standard set of training tricks to improve optimization . Throughout training we maintain an exponential moving average of the parameters with decay factor 0.9999. These average parameters are used during testing. At the start of optimization we use a linear learning rate warm-up for the first 5000 steps. We clip the gradient norm at a norm value of 1.0. We set the dropout rate for the network at 0.1. The skip connections for each residual block are rescaled by a factor of $\frac{1}{\sqrt{2}}$. The input images have random horizontal flips applied to them during training.

For sampling in Table [1](#S6.T1) we set $\tau=0.001$ for $\tau$LDR-0 0 and set $\tau=0.002$ for $\tau$LDR-$10$. The 10 corrector steps per predictor steps for $\tau$LDR-$10$ are introduced after $t<0.1T$. We found that introducing the corrector steps near the end of the reverse sampling process had the best improvement in sample quality for the smallest increase in computational cost. When performing tau-leaping with the corrector rate, $R_{t}^{c}$, we have control over what $\tau$ we use since we are sampling a different CTMC (with $q_{t}$ as its stationary distribution) to the original reverse CTMC. We found that setting the corrector rate $\tau$ to be $1.5$ times the original $\tau$ for the reverse CTMC achieves the best performance in this example.

We train using 4 V100 GPUs on an academic research cluster. To calculate Inception and FID values, we use pytorch-fid and a further development (^1^11[https:/github.com/w86763777/pytorch-gan-metrics](https:/github.com/w86763777/pytorch-gan-metrics)). We verified this library produced comparable values to previous work by calculating the Inception and FID scores for the published images from the DDPM method.

We show a large array of unconditional samples from the $\tau$LDR-$10$ model in Figure [8](#A8.F8). We now also present statistics from the reverse sampling process with standard tau-leaping with $\tau=0.001$. Figure [9](#A8.F9) shows the proportion of dimensions that transition during a single step of tau-leaping. We see that during the initial stages, every dimensions changes during every tau-leaping step, but nearer the end of the process, more dimensions will have settled in their final positions and the proportion is less. In Figure [10](#A8.F10) we show the proportion of dimensions that are clipped due to proposing an out of bounds jump. Overall, the proportion is small. It is largest at the start of the process when we have initially sampled from the approximately uniform $p_{\textrm{ref}}$ and there will be dimensions close to the boundary. As pixel values settle to their final values, the proportion reduces. Figure [11](#A8.F11) shows the progression of a selection of dimensions during the reverse sampling process. A similar picture emerges where dimensions eventually settle in a region of the state space. We also note that larger jumps are made in a single tau-leaping step nearer the start of the reverse process and smaller jumps are made nearer the end.

Figure: Figure 8: Unconditional CIFAR10 samples from our $\tau$LDR-$10$ model.
Refer to caption: /html/2205.14987/assets/x10.png

Figure: Figure 9: Proportion of dimensions that transition during a single step of tau-leaping during the reverse sampling process.
Refer to caption: /html/2205.14987/assets/x11.png

Figure: Figure 10: Proportion of dimensions that are clipped during a tau-leaping step due to proposing an out of bounds jump during the reverse sampling process.
Refer to caption: /html/2205.14987/assets/x12.png

Figure: Figure 11: The progression of a selection of dimensions during the reverse sampling process.

### H.3 Monophonic Music

We generate our training dataset from the Lakh pianoroll dataset (license CC By 4.0). This dataset consists of 174,154 multitrack pianorolls. We go through all songs and all tracks within each song and select sequences that match the following criteria: they are monophonic (only one note played at a time), there is not a period longer than one bar in which no note is played, there is more than one type of note played in the sequence and finally there is no one note played for more than 50 time steps out of the total 256 time steps. This removes the uninteresting and trivial sequences present within the dataset. We then remove any duplicates in the result. This leaves us with 6000 training examples and 950 testing examples. Each song consists of 256 time steps (16 per bar) and each time step takes one from 129 values i.e. we have $D=256$ and $S=129$. This state value represents either a note from 128 options or a rest. We scramble the ordering of this state space when mapping to the integers from 0 to 128. When we input into the denoising network, we input as one-hot $129$ dimensional vectors.

For the forward noising process, we use a uniform rate matrix, $R_{b}=\mathbbm{1}\mathbbm{1}^{T}-S\mathrm{Id}$ and set $\beta(t)=0.03$. We found a constant in time $\beta(t)$ was sufficient for this dataset. In our comparison, we used a birth/death rate matrix defined as

$$ $\begin{bmatrix}-\lambda&\lambda&0&0&\dots&0&0\\ \lambda&-2\lambda&\lambda&0&\dots&0&0\\ 0&\lambda&-2\lambda&\lambda&\dots&0&0\\ \vdots&\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\ 0&0&0&0&\dots&\lambda&-\lambda\end{bmatrix}$ (242) $$

this is the rate matrix for a birth/death process. We set $\lambda=1$ and $\beta(t)=\frac{1}{2}\times 10000^{t}\log 10000$. These hyperparameters were selected such that the forward process has a steady rate of noising whilst still having $q_{T}$ very close to $p_{\textrm{ref}}$. We chose to compare these types of rate matrix because the birth/death rate is inappropriate for this categorical data as adjacent states have no meaning since the mapping to the integers was arbitrary. The uniform rate is suitable for this categorical data because, during a time interval, it has a uniform probability to transition to any other state. The D3PM baseline was implemented also with a time homogeneous uniform forward kernel set such that the rate of noising is matched in the discrete and continuous time cases.

We define our conditional denoising network, $\smash{p_{0|t}^{\theta}}(x_{0}|x,y)$ using a transformer architecture inspired by . It takes an input of shape $(B,D,S)$ where $B$ is the batch size, $D$ is the dimensionality ($256$) and $S$ is the state size ($129$). This final dimension contains the one-hot vectors. The conditioning on the initial bars is achieved by concatenating the conditioning information $y$ with the noisy input $x$. At the start of the network, there is an input embedding linear layer with output of size $128$ which is our model dimension for the transformer. Then a transformer positional embedding is added to the hidden state. Next a stack of $6$ transformer encoder layers are applied which consist of a self attention block and a one hidden layer MLP. The self attention block uses $8$ heads and the MLP has a hidden layer size of $2048$. At the output of each internal block, we apply dropout with rate $0.1$. Finally, there is a stack of 2 residual MLP layers. Each consists of a one hidden layer MLP with a hidden dimension of $2048$. There is a residual connection between the input and output of the MLP. A layer norm is applied to the output of the block. To create the output of the network, there is an output linear layer with an output shape of $(B,D,S)$ where now the $S$ dimension has logit probabilities. To instill a residual bias into the network, we add the one-hot input to the output logits. All activations are ReLU. The time is input into the network through FiLM layers . First, the time is embedded using the sinusoidal transformer position embedding as in the U-net architecture used for image modeling to create an embedding size of $128$. This is then passed into a single hidden layer MLP with hidden size $2048$ and output size $512$. Within each encoder and residual MLP block, there is a FiLM linear layer which takes in the $512$ time embedding and outputs two FiLM parameters each of size $128$. These are the scale and offset applied to the hidden state. In the encoder blocks, this FiLM transform is applied after the self attention block and again after the fully connected block. In the residual MLP blocks, it is applied after the layer norm operation. Our network has approximately $7$ million parameters in total.

We optimize using Adam for 1M steps with a batch size of 64 and learning rate of 0.0002. We use the conditional $\bar{\mathcal{L}}_{\textup{CT}}$ objective with additional direct $p_{0|t}^{\theta}$ supervision as described in Appendix [D](#A4) with weight $\lambda=0.001$. We also make the same one forward pass approximation as explained in Appendix [C.4](#A3.SS4). We use the standard set of training tricks to improve optimization . Throughout training we maintain an exponential moving average of the parameters with decay factor 0.9999. These average parameters are used during testing. At the start of optimization we use a linear learning rate warm-up for the first 5000 steps. We clip the gradient norm at a norm value of 1.0. We train on a single V100 GPU on an academic cluster.

For sampling with $\tau$LDR-0 0 we use $\tau=0.001$ and for sampling with $\tau$LDR-$2$ we include $2$ corrector steps per predictor step after $t<0.9T$. The corrector rate is simulated with $\tau=0.0001$ which we found to perform best. We reject any dimension in which 2 or more jumps are proposed as this is categorical data. We plot the rejection rate in Figure [12](#A8.F12). Most of the time, the rejection rate is zero and there are few steps for which it increases slightly. We show a large batch of samples from the first 10 songs in the test dataset in Figure [13](#A8.F13). We see that there is variation between the sampled completions and they consistently follow the style of the conditioning first two bars of the song. Audio samples from the model are available at [https://github.com/andrew-cr/tauLDR](https://github.com/andrew-cr/tauLDR). Finally, we examine the progression of a random selection of dimensions during reverse sampling for the uniform and birth/death rate matrix cases. Figure [14](#A8.F14) shows the progression for the uniform case, we see that large jumps through the state space are made throughout the reverse process. Figure [15](#A8.F15) shows the progression for the birth/death case. At the start of reverse sampling, no dimensions move as the rejection rate is high in this case because the rate matrix is not suitable for categorical data. Nearer the end, small jumps are made between adjacent states but since large jumps between any category do not occur for this rate matrix, the performance will overall be worse.

Figure: Figure 12: Proportion of jumps rejected during reverse sampling. The rejection rate is calculated as the proportion of dimensions in a tau leaping step that have their jump rejected. The results are averaged over a batch of 16.
Refer to caption: /html/2205.14987/assets/x14.png

Figure: Figure 13: Two conditional samples from the $\tau$LDR-0 0 model for each of the first 10 songs in the test dataset.
Refer to caption: /html/2205.14987/assets/x15.png

Figure: Figure 14: The progression of a selection of dimensions during the reverse sampling process for the uniform rate matrix.
Refer to caption: /html/2205.14987/assets/x16.png

Figure: Figure 15: The progression of a selection of dimensions during the reverse sampling process for the birth/death rate matrix.
Refer to caption: /html/2205.14987/assets/x17.png

## Appendix I Ethical Considerations

Our work increases our theoretical understanding of denoising generative models and also improves generation capabilities within some discrete datasets. Deep generative models are generic methods for learning from unstructured data and can have negative social impacts when misused. For example, they can be used to spread misinformation by reducing the resources required to create realistic fake content. Furthermore, generative models will produce samples that accurately reflect the statistics of their training dataset. Therefore, if samples from these models are interpreted as an objective truth without fully considering the biases present in the original data, then they can perpetuate discrimination against minority groups.

In this work, we train on datasets that contain less sensitive data such as pictures of objects and music samples. The methods we presented, however, could be used to model images of people or text from the internet which will contain biases and potentially harmful content that the model will then learn from and reproduce. Great care must be taken when training these models on real world datasets and when deploying them so as to mitigate and prevent the harms that they can cause.