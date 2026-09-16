---
arxiv_id: "2311.03886"
title: "Formulating Discrete Probability Flow Through Optimal Transport"
year: 2023
source: arxiv2md
---

## Abstract

Abstract Continuous diffusion models are commonly acknowledged to display a deterministic probability flow, whereas discrete diffusion models do not. In this paper, we aim to establish the fundamental theory for the probability flow of discrete diffusion models. Specifically, we first prove that the continuous probability flow is the Monge optimal transport map under certain conditions, and also present an equivalent evidence for discrete cases. In view of these findings, we are then able to define the discrete probability flow in line with the principles of optimal transport. Finally, drawing upon our newly established definitions, we propose a novel sampling method that surpasses previous discrete diffusion models in its ability to generate more certain outcomes. Extensive experiments on the synthetic toy dataset and the CIFAR-10 dataset have validated the effectiveness of our proposed discrete probability flow. Code is released at: https://github.com/PangzeCheung/Discrete-Probability-Flow .

## 1 Introduction

The emerging diffusion-based models have been proven to be an effective technique for modeling data distribution, and generating high-quality texts , images and videos . Considering their generative capabilities have surpassed the previous state-of-the-art results achieved by generative adversarial networks , there has been a growing interest in exploring the potential of diffusion models in various advanced applications .

Diffusion models are widely recognized for generating samples in a stochastic manner , which complicates the task of defining an encoder that translates a sample to a certain latent space. For instance, by following the configuration proposed by , it has been observed that generated samples from any given initial point have the potential to span the entire support of the data distribution. To achieve a deterministic sampling process while preserving the generative capability, Song *et al.* proposed the probability flow, which provides a deterministic map between the data space and the latent space for continuous diffusion models. Unfortunately, the situation differs when it comes to discrete models. For instance, considering two binary distributions $(P_{0}=\frac{1}{2},P_{1}=\frac{1}{2})$ and $(P_{0}=\frac{1}{3},P_{1}=\frac{2}{3})$, there is no deterministic map that can transform the former distribution to the latter one, as it would simply be a permutation. Although some previous research has been conducted on discrete diffusion models with discrete and continuous time configurations, these works primarily focus on improving the sampling quality and efficiency, while sampling certainty has received less attention. More specifically, there is a conspicuous absence of existing literature addressing the probability flow in discrete diffusion models.

The aim of this study is to establish the fundamental theory of the probability flow for discrete diffusion models. Our paper contributes in the following ways. Firstly, we provide proof that under some conditions the probability flow of continuous diffusion coincides with the Monge optimal transport map during any finite time interval within the range of $\left(0,\infty\right)$. Secondly, we propose a discrete analogue of the probability flow under the framework of optimal transport, which we have defined as the *discrete probability flow*. Additionally, we identify several properties that are shared by both the continuous and discrete probability flow. Lastly, we propose a novel sampling method based on the aforementioned observations, and we demonstrate its effectiveness in significantly improving the certainty of the sampling outcomes on both synthetic toy dataset and CIFAR-10 dataset.

Proofs for all Propositions are given in the Appendix. For consistency, the probability flow and infinitesimal transport of a process $X_{t}$ is signified by $\hat{X}_{t}$ and $\tilde{X}_{t}$ respectively.

## 2 Background on Diffusion Models and Optimal Transport

First of all, we review some important concepts from the theory of diffusion models, optimal transport and gradient flow.

### 2.1 Continuous state diffusion models

Diffusion models are generative models that consist of a forward process and a backward process. The forward process transforms the data distribution $p_{data}(x_{0})$ into a tractable reference distribution $p_{T}(x_{T})$. The backward process then generates samples from the initial points drawn from $p_{T}(x_{T})$. According to , the forward process is modeled as the (time-dependent) Ornstein-Uhlenbeck (OU) process:

$$ $dX_{t}=-\theta_{t}X_{t}dt+\sigma_{t}dB_{t},$ (1) $$

where $\theta_{t}\geq 0,\sigma_{t}>0,\forall t\geq 0$ and $B_{t}$ is the Brownian Motion (BM). The backward process is the reverse-time process of the forward process :

$$ $dX_{t}=[-\theta_{t}X_{t}-\sigma_{t}^{2}\nabla_{X_{t}}\log p(X_{t},t)]dt+\sigma_{t}d\tilde{B}_{t},$ (2) $$

where $\tilde{B}_{t}$ is the reverse-time Brownian motion and $p(X_{t},t)$ is the single-time marginal distribution of the forward process, which also serves as the solution to the Fokker-Planck equation :

$$ $\frac{\partial}{\partial t}p(x,t)=\theta_{t}\nabla_{x}(xp(x,t))+\frac{1}{2}\sigma_{t}^{2}\Delta_{x}p(x,t).$ (3) $$

In order to train a diffusion model, the primary objective is to minimize the discrepancy between the model output $s_{\theta}(x_{t},t)$ and the Stein score function $s(x_{t},t)=\nabla_{x_{t}}\log p(x_{t},t)$ . Song *et al.* demonstrate that, it is equivalent to match $s_{\theta}(x_{t},t)$ with the conditional score function:

$$ $\displaystyle\theta^{*}=\mathop{\arg\min}\limits_{\theta}\mathbb{E}_{t}\left\{\lambda_{t}\mathbb{E}_{x_{0},x_{t}}\left[\lVert s_{\theta}(x_{t},t)-\nabla_{x_{t}}\log p(x_{t},t|x_{0},0)\rVert^{2}\right]\right\},$ (4) $$

where $\lambda_{t}$ is a weighting function, $t$ is uniformly sampled over $\left[0,T\right]$ and $p(x_{t},t|x_{0},0)$ is the forward conditional distribution.

It is noted that every Ornstein-Uhlenbeck process has an associated probability flow, which is a deterministic process that shares the same single-time marginal distribution . The probability flow is governed by the following Ordinary Differential Equation (ODE):

$$ $d\hat{X}_{t}=[-\theta_{t}\hat{X}_{t}-\frac{1}{2}\sigma_{t}^{2}s(\hat{X}_{t},t)]dt.$ (5) $$

In accordance with the global version of Picard-Lindelöf theorem and the adjoint method, the map

$$ $\displaystyle T_{s,t}:\mathbb{R}^{n}$ $\displaystyle\longrightarrow\mathbb{R}^{n},$ (6) $\displaystyle\hat{X}_{s}$ $\displaystyle\longmapsto\hat{X}_{t}.$ $$

is a diffeomorphism $\forall t\geq s>0$. The diffeomorphism naturally gives a transport map.

### 2.2 Discrete state diffusion models

In the realm of discrete state diffusion models, there are two primary classifications: the Discrete Time Discrete State (DTDS) models and the Continuous Time Discrete State (CTDS) models, which are founded on Discrete Time Markov Chains (DTMC) and Continuous Time Markov Chains (CTMC), correspondingly. Campbell *et al.* conducted a comparative analysis of these models and determined that CTDS outperforms DTDS. The DTDS models construct the forward process through the utilization of the conditional distribution $q_{t+1|t}(x_{t+1}|x_{t})$ and employ a neural network to approximate the reverse conditional distribution $q_{t|t+1}(x_{t}|x_{t+1})=\frac{q_{t+1|t}(x_{t+1}|x_{t})q_{t}(x_{t})}{q_{t+1}(x_{t+1})}$. In practical applications, it is preferable to parameterize this model using $p^{\theta}_{0|t+1}$ and obtain $p^{\theta}_{k|k+1}$ through

$$ $\begin{split}p^{\theta}_{k|k+1}(x_{k}|x_{k+1})&=\sum_{x_{0}}q_{k|k+1,0}(x_{k}|x_{k+1},x_{0})p^{\theta}_{0|k+1}(x_{0}|x_{k+1})\\ &=\sum_{x_{0}}q_{k+1|k}(x_{k+1}|x_{k})\frac{q_{k|0}(x_{k}|x_{0})}{q_{k+1|0}(x_{k+1}|x_{0})}p^{\theta}_{0|k+1}(x_{0}|x_{k+1}).\end{split}$ (7) $$

In contrast to DTDS models, a CTDS model is characterized by the (infinitesimal) generator , or transition rate, $Q_{t}(x,y)$. The Kolmogorov forward equation is:

$$ $\frac{\partial}{\partial t}q_{t|s}(x_{t}|x_{s})=\sum_{y}q_{t|s}(y|x_{s})Q_{t}(y,x_{t}).$ (8) $$

The reverse process is:

$$ $\frac{\partial}{\partial s}q_{s|t}(x_{s}|x_{t})=\sum_{y}q_{s|t}(y|x_{t})R_{t}(y,x_{s}).$ (9) $$

The generator of the reverse process can be written by :

$$ $R_{t}(y,x)=\frac{q_{t}(x)}{q_{t}(y)}Q_{t}(x,y)=\sum_{y_{0}}\frac{q_{t|0}(x|y_{0})}{q_{t|0}(y|y_{0})}q_{0|t}(y_{0}|y)Q_{t}(x,y).$ (10) $$

There are various approaches to train the model, such as the Evidence Lower Bound (ELBO) technique , and the score-based approach . It has been observed that the reverse generator can be factorized over dimensions, allowing parallel sampling for each dimension during the reverse process. However, it is important to note that this independence is only possible when the time interval for each step is small.

### 2.3 Optimal transport

The *optimal transport problem* can be formulated in two primary ways, namely the Monge formulation and the Kantorovich formulation . Suppose there are two probability measures $\mu$ and $\nu$ on $(\mathbb{R}^{n},\mathcal{B})$, and a cost function $c:\mathbb{R}^{n}\times\mathbb{R}^{n}\rightarrow\left[0,+\infty\right]$. The *Monge problem* is

$$ $\text{(MP) }\inf_{\text{T}}\left\{\int c(x,\text{T}(x))\,{\rm d}\mu(x):\text{T}_{\texttt{\#}}\mu=\nu\right\}.$ (11) $$

The measure $\text{T}_{\texttt{\#}}\mu$ is defined through $\text{T}_{\texttt{\#}}\mu(A)=\mu(\text{T}^{-1}(A))$ for every $A\in\mathcal{B}$ and is called the *pushforward* of $\mu$ through T.

It is evident that the Monge Problem (MP) transports the entire mass from a particular point, denoted as $x$, to a single point $\text{T}(x)$. In contrast, Kantorovich provided a more general formulation, referred to as the *Kantorovich problem*:

$$ $\text{(KP) }\inf_{\gamma}\left\{\int_{\mathbb{R}^{n}\times\mathbb{R}^{n}}c\,{\rm d}\gamma:\gamma\in\mit\Pi(\mu,\nu)\right\},$ (12) $$

where $\mit\Pi(\mu,\nu)$ is the set of *transport plans*, i.e.,

$$ $\mit\Pi(\mu,\nu)=\left\{\gamma\in\mathcal{P}(\mathbb{R}^{n}\times\mathbb{R}^{n}):(\pi_{x})_{\texttt{\#}}\gamma=\mu,(\pi_{y})_{\texttt{\#}}\gamma=\nu\right\},$ (13) $$

where $\pi_{x}$ and $\pi_{y}$ are the two projections of $\mathbb{R}^{n}\times\mathbb{R}^{n}$ onto $\mathbb{R}^{n}$.
For measures absolutely continuous with respect to the Lebesgue measure, these two problems are equivalent . However, when the measures are discrete, they are entirely distinct as the constraint of the Monge Problem may never be fulfilled.

### 2.4 Fokker-Planck equation by gradient flow

According to , the Fokker-Planck equation represents the gradient flow of a functional in a metric space. In particular, for Brownian motion, its Fokker-Planck equation, which is also known as the heat diffusion equation, can be expressed as:

$$ $\frac{\partial}{\partial t}p(x,t)=\frac{1}{2}\Delta p(x,t),$ (14) $$

and it represents the gradient flow of the Gibbs-Boltzmann entropy multiplied by $-\frac{1}{2}$:

$$ $-\frac{1}{2}S(p)=\frac{1}{2}\int_{\mathbb{R}^{n}}p(x)\log p(x)\,{\rm d}x.$ (15) $$

It is worth noting that Eq. [15](#S2.E15) is the gradient flow of Eq. [14](#S2.E14) under the
*2-wasserstein metric* ($W_{2}$).

Chow *et al.* have developed an analogue in the discrete setting by introducing the discrete Gibbs-Boltzmann entropy:

$$ $\displaystyle S(p)=\sum_{i}p_{i}~{}log~{}p_{i},$ (16) $$

and deriving the gradient flow using a newly defined metric (Definition 1 in ). Since the discrete model is defined on graph $G(V,E)$, where $V=\{a_{1},...,a_{N}\}$ is the set of vertices, and $E$ is the set of edges, the discrete Fokker-Planck equation with a constant potential can be written as:

$$ $\frac{d}{dt}p_{i}=\sum_{j\in N(i)}p_{j}-p_{i},$ (17) $$

where $N(i)=\{j\in\{1,2,...,N\}|\{a_{i},a_{j}\}\in E\}$ represents the one-ring neighborhood.

## 3 Continuous probability flow

### 3.1 The equivalence of Ornstein-Uhlenbeck processes and Brownian motion

The diffusion models that are commonly utilized in machine learning are founded on Ornstein-Uhlenbeck processes. First of all, we demonstrate that it is feasible to deterministically convert a time-dependent Ornstein-Uhlenbeck process into a standard Brownian motion.

#### Proposition 1.

Let $X_{t}$ and $Y_{t}$ be a time-dependent Ornstein-Uhlenbeck process and a Brownian motion respectively: $dX_{t}=-\theta_{t}X_{t}dt+\sigma_{t}dB^{(1)}_{t}$, $dY_{t}=dB^{(2)}_{t}$, where $B^{(1)}_{t}$ and $B^{(2)}_{t}$ are two independent Brownian motions and $\theta_{t}\geq 0,\sigma_{t}>0,\forall t\geq 0$. Let $\phi_{t}=\exp(\int_{0}^{t}\theta_{\tau}\,{\rm d}\tau)$, $\beta_{t}=\int_{0}^{t}(\sigma_{\tau}\phi_{\tau})^{2}\,{\rm d}\tau$. Then $X_{t}$ coincides in law with $\phi^{-1}_{t}Y_{\beta_{t}}$.

Building upon the aforementioned proposition, the primary focus of this paper is centered around the standard Brownian motion $dY_{t}=dB_{t}$.

### 3.2 Probability flow is a Monge map

Khrulkov *et al.* have proposed a conjecture that the probability flow of Ornstein-Uhlenbeck process is a Monge map. However, they only provided a proof for a simplified case. We demonstrate that under some conditions, the conjecture is correct.

It is important to highlight that the continuous optimal transports presented in this paper are defined exclusively with the cost function: $c(x,y)=\frac{1}{2}|x-y|^{2}$.

Within the context of generative models, a collection of training samples denoted as $\{x_{i}\}_{i=1}^{N}$ is typically provided, and these samples are intrinsically defined by a distribution:

$$ $p(x,0)=\frac{1}{N}\sum_{i=1}^{N}\delta(x-x_{i}),$ (18) $$

where $\delta(x)$ represents the Dirac delta function. Given a Brownian motion with an initial distribution in the form of Equation ([18](#S3.E18)), the single-time marginal distribution is

$$ $p_{B}(x,t)=\frac{1}{N}\sum_{i=1}^{N}(2\pi t)^{-\frac{n}{2}}\exp(-\frac{\left|x-x_{i}\right|^{2}}{2t}).$ (19) $$

The probability flow is defined as :

$$ $d\hat{Y}_{t}=-\frac{1}{2}\nabla_{\hat{Y}_{t}}\log p_{B}(\hat{Y}_{t},t)dt.$ (20) $$

According to , the solution exists for all $t>0$ and the map $\hat{Y}_{t+s}(\hat{Y}_{t})$ is a diffeomorphism for all $t>0,s\geq 0$. We have discovered that $\hat{Y}_{t+s}(\hat{Y}_{t})$ is the Monge map under some conditions and the time does not reach 0 0 or $+\infty$.

#### Proposition 2.

Given that $Y_{0}$ follows the initial condition ([18](#S3.E18)), and all $x_{i}$s lie on the same line, the diffeomorphism $\hat{Y}_{t+s}(\hat{Y}_{t})$ is the Monge optimal transport map between $p_{B}(x,t)$ and $p_{B}(x,t+s)$, $\forall~{}t>0,s\geq 0$.

There is a counterexample to demonstrate that the probability flow map does not necessarily provide optimal transport. It is important to note that their case differs from our assumptions in two ways. Firstly, they consider the limit case of $\hat{Y}_{+\infty}(\hat{Y}_{0})$. Secondly, the initial distribution of the counterexample does not conform to the form specified in Equation ([18](#S3.E18)). Therefore, their counterexample is not applicable to our situation.

It has been shown that the heat diffusion equation can be regarded as the *gradient flow* of the Gibbs-Boltzmann entropy concerning the $W_{2}$ *metric* . As $W_{2}$ is associated with optimal transport, it is reasonable to anticipate that the "infinitesimal transport" $\hat{Y}_{t+dt}(\hat{Y}_{t})$ is optimal .

In order to interpret the concept of "infinitesimal transport", we utilize the generator of the process $Y_{t}$. Let $C_{c}^{2}(\mathbb{R}^{n})$ denote the set of twice continuously differentiable functions on $\mathbb{R}^{n}$ with compact support. The generator $A_{t}$ is defined as follows :

$$ $\hat{A}_{t}f=\lim_{\Delta t\rightarrow 0^{+}}\frac{f(\hat{Y}_{t+\Delta t})-f(\hat{Y}_{t})}{\Delta t},\forall f\in C_{c}^{2}(\mathbb{R}^{n}).$ (21) $$

It is straightforward to verify that

$$ $\hat{A}_{t}=-\frac{1}{2}\nabla_{x}\log{p_{B}(x,t)^{T}}\nabla_{x}.$ (22) $$

We define the "infinitesimal transport" to be the diffeomorphism $\tilde{Y}_{t+s}(\tilde{Y}_{t})$ where $\tilde{Y}_{t+s}$ evolves according to the following equation

$$ $d\tilde{Y}_{t+s}=-\frac{1}{2}\nabla_{\tilde{Y}_{t}}\log{p_{B}(\tilde{Y}_{t}(\tilde{Y}_{t+s}),t)}ds,$ (23) $$

with the initial condition $\tilde{Y}_{t}=\hat{Y}_{t}$. The generator of $\tilde{Y}_{t+s}$ is

$$ $\tilde{A}_{t+s}=-\frac{1}{2}\nabla_{\tilde{Y}_{t}}\log{p_{B}(\tilde{Y}_{t}(\tilde{Y}_{t+s}),t)}\nabla_{x}.$ (24) $$

#### Proposition 3.

Given any $t>0$, there exists a $\delta_{t}>0$ s.t. $\forall~{}0<s<\delta_{t}$, the diffeomorphism $\tilde{Y}_{t+s}(\tilde{Y}_{t})$ with the initial condition $\tilde{Y}_{t}=\hat{Y}_{t}$ is the Monge optimal transport map.

Let us return to the original Ornstein-Uhlenbeck process $X_{t}$. As it is merely a deterministic transformation of the Brownian motion $Y_{t}$, we can anticipate that the probability flow of $X_{t}$, denoted by $\hat{X}_{t}$, will be a Monge map. In fact, this expectation holds true:

#### Proposition 4.

Given that $X_{0}$ follows the initial condition ([18](#S3.E18)), and all $x_{i}$s lie on the same line, the diffeomorphism $\hat{X}_{t+s}(\hat{X}_{t})$ is the Monge optimal transport map for all $t>0,s\geq 0$.

## 4 Discrete probability flow

The continuous probability flow is deterministic, which means the "mass" at $\hat{Y}_{t}$ is entirely transported to $\hat{Y}_{t+s}$ during the time interval $\left[t,t+s\right]$. However, it is widely acknowledged that for discrete distributions $\mu$ and $\nu$, there may not exist a T such that $\text{T}_{\texttt{\#}}\mu=\nu$. As a result, discrete diffusions cannot possess a deterministic probability flow. To establish the concept of the *discrete probability flow*, we employ the methodology of optimal transport. First of all, a discrete diffusion model is proposed as an analogue of Brownian motion. Secondly, we modified the forward process to create an optimal transport map, which is used to define the discrete probability flow. Finally, a novel sampling technique is introduced, which significantly improves the certainty of the sampling outcomes.

### 4.1 Constructing discrete probability flow

It is demonstrated that the process described by Equation ([17](#S2.E17)) is a discrete equivalent of the heat diffusion process ([14](#S2.E14)) . We adopt this process as our discrete diffusion model and represent it in a more comprehensive notation.

The discrete diffusion model has $K$ dimensions and $S$ states. The states are denoted by $i=(i_{1},i_{2},\dots,i_{K})$, where $i_{j}\in\{1,2,\dots,S\}$. The Kolmogorov forward equation for this process is

$$ $\frac{d}{dt}{P}^{i}_{j}(t|s)=\sum_{j^{\prime}}{P}^{i}_{j^{\prime}}(t|s){Q_{D}}_{j}^{j^{\prime}}(t),$ (25) $$

where ${P}^{i}_{j}(t|s)$ means $P(x_{t}=j|x_{s}=i)$ and $Q_{D}$ is defined as:

$$ ${Q_{D}}^{i}_{j}=\begin{cases}1,&d_{D}(i,j)=1,\\ -\sum_{j^{\prime}\in\{k:d_{D}(i,k)=1\}}{Q_{D}}^{i}_{j^{\prime}},&d_{D}(i,j)=0,\\ 0,&otherwise,\end{cases}$ (26) $$

where $d_{D}(i,j)=\sum_{l=1}^{K}\left|i_{l}-j_{l}\right|$.
If we let the solution of the Equation ([25](#S4.E25)) be denoted by $P_{D}(t|s)$ and assume an initial condition $P_{0}$, the single-time marginal distribution can be computed as follows:

$$ ${P_{D}}_{i}(t)=\sum_{j}{P_{0}}_{j}{Q_{D}}^{j}_{i}(t|0).$ (27) $$

It is noteworthy that the process defined by $Q_{D}$ is not an optimal transport map, as there exist *mutual flows* between the states (i.e., there exists two states $i$, $j$ with $Q^{i}_{j}>0$ and $Q^{j}_{i}>0$). Therefore, we propose a modified version that will be proved to be a solution to the Kantorovich problem, namely, an optimal transport plan. The modified version is defined by the following $Q$:

$$ $Q^{i}_{j}(t)=\begin{cases}\frac{ReLU({P_{D}}_{i}(t)-{P_{D}}_{j}(t))}{{P_{D}}_{i}(t)},&d_{D}(i,j)=1,\\ -\sum_{j^{\prime}\in\{k:d_{D}(i,k)=1\}}{Q}^{i}_{j^{\prime}}(t),&d_{D}(i,j)=0,\\ 0,&otherwise.\end{cases}$ (28) $$

where

$$ $ReLU(x)=\begin{cases}x,&x>0,\\ 0,&x\leq 0.\end{cases}$ (29) $$

In order to avoid singular cases, We define $Q^{i}_{j}(t)$ to be 0 0 when ${P_{D}}_{i}(t)=0$. In fact, it is easy to verify that ${P_{D}}_{i}(t)>0$ for all $t>0$ , $i\in\left\{1,2,\dots,K\right\}$. We will show that the process defined by $Q$ is equivalent in distribution to the one generated by $Q_{D}$.

#### Proposition 5.

The processes generated by $Q_{D}$ and $Q$ have the same single-time marginal distribution $\forall t>0$.

#### Proposition 6.

Given any $t>0$, there exists a $\delta_{t}>0$ s.t. $\forall~{}0<s<\delta_{t}$, the process generated by $Q$ provides an optimal transport map from $P_{D}(t)$ to $P_{D}(t+s)$ under the cost $d_{D}$.

Proposition 6 demonstrates that $Q_{D}$ generates a Kantorovich plan between $P_{D}(t)$ and $P_{D}(t+s)$ under a certain cost function. On the other hand, the continuous probability flow is the Monge map between $p_{B}(x,t)$ and $p_{B}(x,t+s)$. Therefore, it is reasonable to define the process defined by $Q_{D}$ as the *discrete probability flow* of the original process defined by $Q$.

Furthermore, the "infinitesimal transport" of the discrete process, which is defined by $\frac{d}{ds}\hat{P}(t+s)=\hat{P}(t+s)Q(t)$, also provides an optimal transport map.

#### Proposition 7.

Given any $t>0$, there exists a $\delta_{t}>0$ s.t. $\forall~{}0<s<\delta_{t}$, the process above provides an optimal transport map from $\hat{P}(t)$ to $\hat{P}(t+s)$ under the cost $d_{D}$.

### 4.2 Sampling by discrete probability flow

In order to train the modified model, we employ a score-based method described in the Score-based Continuous-time Discrete Diffusion Model (SDDM) . Specifically, we directly learn the conditional probability $P^{\theta}({i_{l}}(t)|\left\{i_{1},\dots,i_{l-1},i_{l+1},\dots,i_{K}\right\}(t))$. According to proposition 5, it follows that $P^{\theta}={P^{\theta}}_{D}$, and consequently, the training process is identical to that of . For the sake of brevity, we will employ the notation $P^{\theta}_{i_{l}|i\backslash i_{l}}(t)$ to replace $P^{\theta}({i_{l}}(t)|\left\{i_{1},\dots,i_{l-1},i_{l+1},\dots,i_{K}\right\}(t))$.

The generator of the reverse process is

$$ $R^{i}_{j}(t)=\begin{cases}{ReLU}(\frac{{P^{\theta}_{D}}_{j_{l}|i\backslash i_{l}}(t)}{{P^{\theta}_{D}}_{i_{l}|i\backslash i_{l}}(t)}-1),&d_{D}(i,j)=1~{}\text{and}~{}i_{l}\neq j_{l},\\ -\sum_{j^{\prime}\in\{k:d_{D}(i,k)=1\}}{R}^{i}_{j^{\prime}}(t),&d_{D}(i,j)=0,\\ 0,&otherwise.\end{cases}$ (30) $$

We use the Euler’s method to generate samples. Given the time step length $\epsilon$, the transition probabilities for dimension $l$ is:

$$ $P^{\theta}({i_{l}}(t-\epsilon)|i(t))=\begin{cases}\epsilon R^{i(t)}_{i_{1}(t),\dots,i_{l}(t-\epsilon),\dots,i_{k}(t)}(t),&i_{l}(t-\epsilon)\neq i_{l}(t),\\ 1+\epsilon R^{i(t)}_{i(t)}(t),&i_{l}(t-\epsilon)=i_{l}(t).\end{cases}$ (31) $$

When $\epsilon$ is small, the reverse conditional distribution has the factorized probability:

$$ $P^{\theta}(i(t-\epsilon)|i(t))={\Pi}_{l=1}^{K}P^{\theta}({i_{l}}(t-\epsilon)|i(t))$ (32) $$

In this way, it becomes possible to generate samples by sequentially sampling from the reverse conditional distribution [32](#S4.E32).

#### Transition to higher probability states

The reverse process of the continuous probability flow, as described in Equation ([20](#S3.E20)), causes particles to move towards areas with higher logarithmic probability densities. As the logarithm function is monotonically increasing, this reverse flow pushes particles to higher probability density states. This phenomenon is also observed in the discrete probability flow. By examining the reverse generator, as shown in Equation ([30](#S4.E30)), it can be determined that the transition rate $R^{i}_{j}(t)>0$ only when the destination state $j$ has a higher probability than the source state $i$. This implies that transitions only occur in higher probability states. In contrast, the original continuous reverse process ([2](#S2.E2)) and the discrete reverse process from ([10](#S2.E10)) allow any transitions.

#### Reduction of Standard Deviation

We measure the certainty of the sampling method by the expectation of the Conditional Standard Deviation (CSD):

$$ $CSD_{s,t}(X)=\mathbb{E}_{X_{t}}[\text{Std}(X_{s}|X_{t})],$ (33) $$

where $\text{Std}(X_{s}|X_{t})=\text{Var}^{\frac{1}{2}}(X_{s}|X_{t})=\mathbb{E}^{\frac{1}{2}}_{X_{s}}[X_{s}-\mathbb{E}_{X_{s}}[X_{s}|X_{t}]|X_{t}]$. $CSD_{s,t}(X)$ is 0 0 when the process is deterministic, such as the continuous probability flow. In the discrete situation, there does not exist any deterministic map. However, our discrete probability flow significantly reduces $CSD_{s,t}(X)$. Table [2](#S6.T2) presents numerical evidence of this phenomenon. Therefore, we posit that the discrete probability flow enhances the certainty of the sampling outcomes.

## 5 Related Work

The concept of probability flow was initially introduced in as a deterministic alternative to the Itô diffusion. In the work , they presented the Denoising Diffusion Implicit Model (DDIM) and demonstrated its equivalence to the probability flow. Subsequently, investigated the relationship between the probability flow and optimal transport. They hypothesized that the probability flow could be considered a Monge optimal transition map and provided a proof for a specific case. Additionally, they conducted numerical experiments that supported their conjecture, showing negligible errors. However, has discovered an initial distribution that renders probability flow not optimal.

The discrete diffusion models were first introduced by , who considered a binary model. Following the success of continuous diffusion models, discrete models have garnered more attention. The bulk of research on discrete models has focused primarily on the design of the forward process . Continuous time discrete state models were introduced by and subsequently developed by .

## 6 Experiments

We conduct numerical experiments using our novel sampling method by Discrete Probability Flow (DPF) on synthetic data. The primary goal is to demonstrate that our method can generate samples of comparable quality with higher certainty.

**Table 1: Comparison of generation quality for SDDM and DPF, in terms of MMD with Laplace kernel using bandwith=0.1. Lower values indicate superior quality.**
|  | 2spirals | 8gaussians | checkerboard | circles | moons | pinwheel | swissroll |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | discrete dimension = 32, state size = 2 |  |  |  |  |  |  |
| SDDM | 2.18e-06 | 4.28e-06 | 1.33e-06 | 6.22e-06 | 5.62e-06 | 2.10e-06 | 4.27e-06 |
| DPF (ours) | 1.89e-05 | 1.09e-05 | 2.22e-05 | 3.27e-05 | 2.42e-05 | 1.60e-05 | 2.18e-05 |
|  | discrete dimension = 16, state size = 5 |  |  |  |  |  |  |
| SDDM | 2.06e-4 | 1.01e-4 | 2.43e-4 | 1.74e-4 | 2.20e-4 | 3.37e-4 | 1.43e-4 |
| DPF (ours) | 3.87e-4 | 5.87e-4 | 4.93e-4 | 3.83e-4 | 3.43e-4 | 6.64e-4 | 3.20e-4 |
|  | discrete dimension = 12,<br>state size = 10 |  |  |  |  |  |  |
| SDDM | 5.52e-4 | 3.01e-4 | 4.39e-4 | 4.22e-4 | 2.71e-4 | 2.90e-4 | 3.39e-4 |
| DPF (ours) | 7.19e-4 | 3.49e-4 | 5.99e-4 | 6.65e-4 | 4.34e-4 | 4.14e-4 | 5.17e-4 |
|  |  |  |  |  |  |  |  |

**Table 2: Comparison of certainty for SDDM and DPF, in terms of $CSD$ on 4,000 initial points, each of which has 10 generated samples. Lower values indicate superior certainty.**
|  | 2spirals | 8gaussians | checkerboard | circles | moons | pinwheel | swissroll |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | discrete dimension = 32, state size = 2 |  |  |  |  |  |  |
| SDDM | 14.3053 | 14.1882 | 14.7433 | 14.4327 | 14.1739 | 14.0450 | 14.0548 |
| DPF (ours) | 2.1719 | 1.7945 | 2.0693 | 1.7210 | 2.0573 | 2.1834 | 1.8892 |
|  | discrete dimension = 16, state size = 5 |  |  |  |  |  |  |
| SDDM | 14.4645 | 14.6143 | 14.6963 | 14.4807 | 14.2397 | 14.2466 | 14.2659 |
| DPF (ours) | 1.9711 | 1.9367 | 1.4172 | 1.7185 | 1.7668 | 1.9633 | 1.6665 |
|  | discrete dimension = 12, state size = 10 |  |  |  |  |  |  |
| SDDM | 12.8463 | 12.7933 | 13.0158 | 12.9232 | 12.6665 | 12.7634 | 12.7880 |
| DPF (ours) | 1.8123 | 1.3178 | 1.1348 | 1.4625 | 1.4859 | 1.8435 | 1.5227 |
|  |  |  |  |  |  |  |  |

Figure: Figure 1: Visualization of the generation quality on generated binary samples for SDDM and DPF.
Refer to caption: /html/2311.03886/assets/x1.png

Figure: Figure 2: Visualization of the generating certainty on generated binary samples for SDDM and DPF. All the samples (in blue) are randomly generated from the single initial point (in red).
Refer to caption: /html/2311.03886/assets/x2.png

Figure: Figure 3: Visualization of the generated binary samples from the given initial points $\bm{x}_{T}$. Different colors distinguish the generated samples from different initial points $\bm{x}_{T}$.
Refer to caption: /html/2311.03886/assets/x3.png

Experiments are conducted on synthetic data using the same setup as SDDM , with the exception that we replaced the generator $Q$ with Equation ([26](#S4.E26)). In addition to the binary situation ($S=2$) studied in , we also perform experiments on synthetic data with the state size $S$ set to 5 and 10. To evaluate the quality of the generated samples, we generated 40,000 / 4,000 samples for binary data / other type of data using SDDM and DPF, and measured the Maximum Mean Discrepancy (MMD) with the Laplace kernel . The results are shown in Table [1](#S6.T1). It can be seen that the MMD value obtained using DPF is slightly higher than that of SDDM, which may be attributed to the structure of the reverse generator [10](#S2.E10). Specifically, DPF approximates an additional term, $Q_{t}(y,x)$, with the neural network, which potentially introduces additional errors to the sampling process, leading to a higher MMD value compared to SDDM. However, such difference is minimal and does not significantly impact the quality of the generated samples. As evident from the visualization of the distributions obtained from SDDM and DPF in Figure [1](#S6.F1), it is clear that DPF can generate samples that are comparable to those generated by SDDM.

In addition, we also compare the sampling certainty of DPF and SDDM by computing $CSD_{s,t}$ using a Monte-Carlo based method. Specifically, we set $s=0$ and $t=T$, and sample 4,000 $x_{t}$s with 10 $x_{s}$s for each $x_{t}$. We then estimate $\mathbb{E}(x_{s}|x_{t})$ and $\text{Std}(x_{s}|x_{t})$ using the sample mean and sample standard deviation, respectively. The results of certainty are presented in Table [2](#S6.T2). Our findings indicate that DPF significantly reduces the $CSD$, which suggests a higher certainty. Additionally, we visualize the results of 4,000 generated samples (in blue) from a single initial point (in red) in the binary case in Figure [2](#S6.F2). It is apparent that the sampling of SDDM exhibits high uncertainty, as it can sample the entire pattern from a single initial point. In contrast, our method reduces such uncertainty and is only able to sample a limited number of states.

To provide a more intuitive representation of the generated samples originating from various initial points, we select $20\times 20$ initial points arranged in the grid, and distinguish them using different colors. Subsequently, we visualize the results by sampling 10 outcomes from each initial point, as shown in Figure [3](#S6.F3). We observe that the visualization of SDDM samples appears disorganized, indicating significant uncertainty. In contrast, the visualization of DPF samples exhibits clear regularity, manifesting in two aspects: (1) the generated samples from the same initial point using DPF are clustered by color, demonstrating the better sampling certainty of our DPF. (2) Both of the generated samples and initial points are colored similarly at each position. For example, in the lower right area, a majority of the generated samples are colored purple, which corresponds to the color assigned to the initial points $x_{T}$ in that area. This observation demonstrates that most of the sampling results obtained through DPF are closer to their respective initial points, aligning with our design intention of optimal transport. It is worth noting that similar phenomena are observed across different state sizes, and we have provided these results in the Appendix.

Finally, we extended our DPF to the CIFAR-10 dataset, and compare it with the $\tau$LDR-0 method proposed in . The visualization results are shown in Figure [4](#S8.F4). It can be seen that our method greatly reduces the uncertainty of generating images by sampling from the same initial $x_{T}$. Detailed experimental settings and more experimental results are presented in the Appendix.

## 7 Discussion

In this study, we introduce a discrete counterpart of the probability flow and established its connections with the continuous formulations. We began by demonstrating that the continuous probability flow corresponds to a Monge optimal transport map. Subsequently, we proposed a method to modify a discrete diffusion model to achieve a Kantorovich plan, which naturally defines the discrete probability flow. We also discovered shared properties between continuous and discrete probability flows. Finally, we propose a novel sampling method that significantly reduces sampling uncertainty. However, there are still remaining aspects to be explored in the context of the discrete probability flow. For instance, to obtain more general conclusions under a general initial condition, the semi-group method could be employed. Additionally, while we have proven the existence of a Kantorovich plan in a small time interval, it is possible to extend this to a global solution. Moreover, the definition of the probability flow has been limited to a specific type of discrete diffusion model, which also could be extended to a broader range of models. These topics remain open for future studies.

## 8 Acknowledgments and Disclosure of Funding

We would like to thank all the reviewers for their constructive comments. Our work was supported in National Natural Science Foundation of China (NSFC) under Grant No.U22A2095 and No.62072482.

Figure: Figure 4: Image modeling on CIFAR-10 dataset. The figure is divided into three groups: initial points $x_{T}$, sampling results of $\tau$LDR-0, and sampling results of our DPF. For each row, the sampled images are obtained from the same initial point.
Refer to caption: /html/2311.03886/assets/x4.png

## Appendix A Overview of our DPF

To elucidate our methodology more intuitively, we include schematic diagrams in Figure [5](#A1.F5), illustrating the sampling procedure from various diffusion models. Broadly speaking, diffusion models can be classified into two categories based on the nature of the underlying data space: continuous diffusion models and discrete diffusion models. Figure [5](#A1.F5) (a) provides an illustration of a continuous diffusion model using a Stochastic Differential Equation (SDE) that transforms a prior noise distribution into the data distribution. The stochastic nature of the sampling process in continuous diffusion models allows samples generated from a single initial point to span the entire space (green line), but this feature limits its practical applicability. To overcome this limitation, probability flow is introduced, which ensures that the generated sample from an initial point follows a deterministic path (red line). This enhancement enables the continuous diffusion model to be more manageable and applicable in a broader range of scenarios.

Figure: Figure 5: Schematic representation of different diffusion models.
Refer to caption: /html/2311.03886/assets/x5.png

In this paper, our concentration is primarily on discrete diffusion models. An example of such a model, based on SDDM with 15 states, is depicted in Figure [5](#A1.F5) (b). Similar to SDE, it is observed that the sampling process is also susceptible to uncertainty (green line). One potential solution could involve incorporating probability flow into discrete diffusion in a similar manner as in the continuous models. Nonetheless, as previously mentioned in the introduction, this is not a viable option in discrete models due to the lack of a deterministic mapping between the latent space and the data space. Thus, there is a necessity for a redefined probability flow that is tailored to discrete diffusion models, and this forms the core of this paper. This study examines the probability flow of discrete diffusion models through the concept of optimal transport. Initially, we demonstrate that the continuous probability flow coincides with the Monge optimal transport map (Proposition 4). We then leverage this result to develop a similar probability flow for discrete diffusion models using optimal transport (Proposition 6). Finally, we propose a novel sampling methodology for discrete models that significantly reduces the uncertainty (red line) in the sampling process.

## Appendix B Definitions and Theorems Employed in this Appendix

For the sake of reader convenience, we hereby provide a comprehensive list of the definitions and theorems utilized in this paper. Additionally, we limit our representation to the case within $\mathbb{R}^{n}$.

#### Theorem 1.

(Theorem 1.48 in ) Suppose that $\mu$ is a probability measure on $(\mathbb{R}^{n},\mathcal{B})$ such that $\int|x|^{2}\,{\rm d}\mu(x)<\infty$ and that $u:\mathbb{R}^{n}\rightarrow\mathbb{R}\cup\{+\infty\}$ is convex and differentiable $\mu{\text{-}}a.e$. Set $\text{T}=\nabla u$ and suppose $\int|\text{T}(x)|^{2}\,{\rm d}\mu(x)<\infty$. Then T is optimal for the transport cost $c(x,y)=\frac{1}{2}|x-y|^{2}$ between the measures $\mu$ and $\nu=\text{T}_{\texttt{\#}}\mu$.

#### Definition 2.

The optimization problem under constraint is formally defined as follows:

$$ $\min_{x\in\mathbb{R}^{n}}f(x)~{}~{}~{}~{}\text{subject to}\begin{cases}c_{i}(x)=0,i\in\mathcal{E}\\ c_{i}(x)\geq 0,i\in\mathcal{I}.\end{cases}$ (34) $$

The Lagrangian for this constrained optimization problem is defined as:

$$ $\mathcal{L}(x,\lambda)=f(x)-\sum_{i\in\mathcal{E}\cup\mathcal{I}}\lambda_{i}c_{i}(x).$ (35) $$

Here, $\lambda_{i}$ represents the Lagrange multiplier associated with the $i^{th}$ constraint. The active set at any feasible $x$ is defined as the union of the set $\mathcal{E}$ with the indices of the active inequality constraints, that is:

$$ $\mathcal{A}(x)=\mathcal{E}\cup\{i\in\mathcal{I}:c_{i}(x)=0\}.$ (36) $$

#### Definition 3.

(Definition 12.1 in )
Given the point $x^{*}$, we say that the Linear Independence Constraint Qualification (LICQ) holds if the set of active constraint gradients $\{\nabla c_{i}(x^{*}),i\in\mathcal{A}(x^{*})\}$ is linearly independent.

#### Theorem 4.

(Theorem 12.1 in , the Karush-Kuhn-Tucker (KKT) conditions) Suppose that $x^{*}$ is a local solution of the problem ([34](#A2.E34)) and that the LICQ holds at $x^{*}$. Then there is a Lagrange multiplier vector $\lambda^{*}$ , with components $\lambda_{i}^{*},i\in\mathcal{E}\cup\mathcal{I}$, such that the following conditions are satisfied at $(x^{*},\lambda^{*})$

$$ $\displaystyle\nabla_{x}\mathcal{L}(x^{*},\lambda^{*})=0,$ (37a) $\displaystyle c_{i}(x^{*})=0,$ $\displaystyle~{}~{}~{}~{}~{}\forall i\in\mathcal{E},$ (37b) $\displaystyle c_{i}(x^{*})\geq 0,$ $\displaystyle~{}~{}~{}~{}~{}\forall i\in\mathcal{I},$ (37c) $\displaystyle\lambda^{*}\geq 0,$ $\displaystyle~{}~{}~{}~{}~{}\forall i\in\mathcal{I},$ (37d) $\displaystyle\lambda_{i}^{*}c_{i}(x^{*})=0,$ $\displaystyle~{}~{}~{}~{}~{}\forall i\in\mathcal{E}\cup\mathcal{I}.$ (37e) $$

Remark 5. According to Theorem 4, the Karush-Kuhn-Tucker (KKT) conditions serve as necessary conditions. In the case of linear programming, these conditions are not only necessary but also sufficient. To demonstrate this, let us consider the standard form of a linear programming problem:

$$ $\min c^{T}x,~{}~{}~{}~{}\text{subject to}~{}Ax=b,x\geq 0.$ (38) $$

We can write the Lagrangian function for [38](#A2.E38) as

$$ $\mathcal{L}(x,\pi,s)=c^{T}x-\pi^{T}(Ax-b)-s^{T}x.$ (39) $$

The KKT conditions are

$$ $\displaystyle A^{T}\pi+s=c,$ (40a) $\displaystyle Ax=b,$ (40b) $\displaystyle x\geq 0,$ (40c) $\displaystyle s\geq 0,$ (40d) $\displaystyle x^{T}s=0.$ (40e) $$

Suppose we have a vector triple $(x^{*},\pi^{*},s^{*})$ that satisfies Equation ([40](#A2.E40)). In such a scenario, we can deduce that:

$$ $c^{T}x^{*}=(A^{T}\pi^{*}+s)^{T}x^{*}=(\pi^{*})^{T}Ax^{*}=b^{T}\pi^{*}.$ (41) $$

Let us consider another feasible point denoted by $\bar{x}$, which satisfies the conditions $A\bar{x}=b$ and $\bar{x}\geq 0$. we can conclude that:

$$ $c^{T}\bar{x}=(A^{T}\pi^{*}+s^{*})^{T}\bar{x}=b^{T}\pi^{*}+\bar{x}^{T}s^{*}\geq b^{T}\pi^{*}=c^{T}x^{*}.$ (42) $$

The inequality ([42](#A2.E42)) demonstrates that the KKT conditions serve as sufficient conditions.

#### Theorem 6.

(Theorem 8.5.1 in ) Let $X_{t}$ be an Itô diffusion given by

$$ ${\rm d}X_{t}=b(X_{t})\,{\rm d}t+\sigma(X_{t})\,{\rm d}B_{t},~{}~{}b\in\mathbb{R}^{n},\sigma\in\mathbb{R}^{n\times m},X_{0}=x,$ (43) $$

and let $Y_{t}$ be an Itô process given by

$$ ${\rm d}Y_{t}=u(t,\omega)\,{\rm d}t+v(t,\omega)\,{\rm d}B_{t},~{}~{}u\in\mathbb{R}^{n},v\in\mathbb{R}^{n\times m},Y_{0}=x.$ (44) $$

Assume that

$$ $u(t,\omega)=c(t,\omega)b(Y_{t})~{}~{}and~{}~{}vv^{T}(t,\omega)=c(t,\omega)\sigma\sigma^{T}(Y_{t}),$ (45) $$

for a.a. $t,\omega$. Define $\beta_{t}$ and $\alpha_{t}$ as:

$$ $\beta_{t}=\beta(t,\omega)=\int_{0}^{t}c(s,\omega)\,{\rm d}s~{}~{}and~{}~{}\alpha_{t}=\inf\{s:\beta_{s}>t\}.$ (46) $$

Then $Y_{\alpha_{t}}$ coincides in law with $X_{t}$, denoted by $Y_{\alpha_{t}}\simeq X_{t}$.

#### Theorem 7.

(Theorem 4.1 of Chapter V, §4 in , Poincaré’s lemma). Let $U$ be an open ball in $\mathbb{R}^{n}$ and let $\omega$ be a differential form of degree $\geq 1$ on $U$ such that ${\rm d}\omega=0$. Then there exists a differential form $\phi$ on $U$ such that ${\rm d}\phi=\omega$.

Remarks. The conclusion remains valid when the open ball $U$ is substituted with the entirety of $\mathbb{R}^{n}$.

#### Theorem 8.

(Theorem 4 in ) The solution of the differential equation $Y^{\prime}=A(t)Y$ with initial condition $Y(0)=Y_{0}$ can be written as $Y(t)=\exp(\Omega(t))Y_{0}$ with $\Omega(t)$ defined by

$$ $\Omega^{\prime}=\,{\rm d}\exp_{\Omega}^{-1}(A(t)),~{}~{}\Omega(0)={\displaystyle O}.$ (47) $$

where

$$ $\,{\rm d}\exp_{\Omega}^{-1}(A(t))=\sum_{0}^{\infty}\frac{B_{k}}{k!}{\rm ad}_{\Omega}^{k}(A),$ (48) $$

and $B_{k}$ is the Bernoulli numbers. ${\rm ad}_{\Omega}^{k}(A)$ is defined through

$$ ${\rm ad}_{\Omega}(A)=[\Omega,A],~{}~{}{\rm ad}_{\Omega}^{j}(A)=[\Omega,{\rm ad}_{\Omega}^{j-1}(A)],~{}~{}{\rm ad}_{\Omega}^{0}(A)=A,~{}~{}~{}~{}j\in\mathbb{N},$ (49) $$

where $[A,B]=AB-BA$ is the Lie-bracket.

Remarks. If $A(s)A(t)=A(t)A(s),\forall s,t\geq 0$, $\Omega(t)$ has the simple form $\Omega(t)=\int_{0}^{t}A(s)\,{\rm d}s$.

## Appendix C Proofs

### C.1 Proof of Proposition 1

*Proof.* By Itô formula:

$$ $\displaystyle{\rm d}(\phi_{t}X_{t})=$ $\displaystyle\phi_{t}\theta_{t}X_{t}\,{\rm d}t+\phi_{t}\,{\rm d}X_{t}$ (50) $\displaystyle=$ $\displaystyle\phi_{t}\theta_{t}X_{t}\,{\rm d}t-\phi_{t}\theta_{t}X_{t}\,{\rm d}t+\phi_{t}\sigma_{t}\,{\rm d}B_{t}$ $\displaystyle=$ $\displaystyle\phi_{t}\sigma_{t}\,{\rm d}B_{t}.$ $$

By Theorem 6, $\phi_{\alpha_{t}}X_{\alpha_{t}}\simeq Y_{t}$, which means $X_{t}$ coincides in law with $\phi_{t}^{-1}Y_{\beta_{t}}$ $\square$

Remarks.  Proposition 1 posits that the Ornstein-Uhlenbeck (OU) process is essentially a scaling of Brownian motion with a change in time. Consequently, the VE SDEs, VP SDEs, sub-VP SDEs in , as well as the models presented in , can be regarded as equivalent.

### C.2 Proof of Proposition 2

#### Lemma B.2.1

Let $H_{t}$ be the Hessian matrices $\nabla^{2}_{x_{t}}\log p_{B}(x_{t},t)$, then $H_{s}H_{t}=H_{t}H_{s},\forall s,t\geq 0$.

*Proof.*

$$ $\displaystyle H_{t}=$ $\displaystyle\nabla^{2}_{x_{t}}\log p_{B}(x_{t},t)$ (51) $\displaystyle=$ $\displaystyle\nabla_{x_{t}}\frac{\sum_{i}\exp(-\frac{|x_{t}-x_{i}|^{2}}{2t})(-\frac{x_{t}-x_{i}}{t})}{\sum_{j}\exp(-\frac{|x_{t}-x_{j}|^{2}}{2t})}$ $\displaystyle=$ $\displaystyle\underbrace{\sum_{i}\nabla_{x_{t}}(\frac{\exp(-\frac{|x_{t}-x_{i}|^{2}}{2t})}{\sum_{j}\exp(-\frac{|x_{t}-x_{j}|^{2}}{2t})})(-\frac{x_{t}-x_{i}}{t})}_{A}+\underbrace{\frac{\sum_{i}\exp(-\frac{|x_{t}-x_{i}|^{2}}{2t})}{\sum_{j}\exp(-\frac{|x_{t}-x_{j}|^{2}}{2t})}(-\frac{1}{t})I}_{B}.$ $$

$B$ is a scalar matrix, then it commutes with any matrix.

$$ $\displaystyle A=$ $\displaystyle\underbrace{(\sum_{j}\exp(-\frac{|x_{t}-x_{j}|^{2}}{2t}))^{-2}}_{C}\sum_{i}[\exp(-\frac{|x_{t}-x_{i}|^{2}}{2t})(-\frac{x_{t}-x_{i}}{t})\sum_{j}\exp(-\frac{|x_{t}-x_{j}|^{2}}{2t})$ (52) $\displaystyle-\exp(-\frac{|x_{t}-x_{i}|^{2}}{2t})\sum_{j}\exp(-\frac{|x_{t}-x_{j}|^{2}}{2t})(-\frac{x_{t}-x_{j}}{t})](-\frac{x_{t}-x_{i}}{t})^{T}$ $\displaystyle=$ $\displaystyle C\sum_{i,j}\exp(-\frac{|x_{t}-x_{i}|^{2}}{2t}-\frac{|x_{t}-x_{j}|^{2}}{2t})(\frac{x_{t}-x_{i}}{t})(\frac{x_{t}-x_{i}}{t})^{T}$ $\displaystyle-C\sum_{i,j}\exp(-\frac{|x_{t}-x_{i}|^{2}}{2t}-\frac{|x_{t}-x_{j}|^{2}}{2t})(\frac{x_{t}-x_{j}}{t})(\frac{x_{t}-x_{i}}{t})^{T}$ $\displaystyle=$ $\displaystyle C\sum_{i<j}\exp(-\frac{|x_{t}-x_{i}|^{2}}{2t}-\frac{|x_{t}-x_{j}|^{2}}{2t})\frac{1}{t^{2}}[(x_{t}-x_{i})(x_{t}-x_{i})+(x_{t}-x_{j})(x_{t}-x_{j})$ $\displaystyle-(x_{t}-x_{j})(x_{t}-x_{i})^{T}-(x_{t}-x_{i})(x_{t}-x_{j})^{T}]$ $\displaystyle=$ $\displaystyle C\sum_{i<j}\exp(-\frac{|x_{t}-x_{i}|^{2}}{2t}-\frac{|x_{t}-x_{j}|^{2}}{2t})\frac{1}{t^{2}}(x_{j}-x_{i})(x_{j}-x_{i})^{T}.$ $$

As ${x_{i}}$s lie on the same line, $x_{j}-x_{i}$ can be denoted by $x_{j}-x_{i}=C_{i,j}v,\forall i,j$, where $v$ is a fixed vector. It has $(x_{j}-x_{i})(x_{j}-x_{i})^{T}={C^{2}_{i,j}}v{v}^{T}$. It is clear that ${C^{2}_{i,j}}v{v}^{T}$ and ${C^{2}_{k,l}}v{v}^{T}$ commutes $\forall i,j,k,l$. Furthermore, as $t$ and $x_{t}$ only appear in the coefficients, $H_{t}$ and $H_{s}$ commute with one another. $\square$

Proof of Proposition2. If $Y_{0}$ follows the initial condition ([18](#S3.E18)), and $x_{i}$s lie on the same line, $Y_{t}$ will governed by the equation ([20](#S3.E20)). Employing the trick in , For a fixed $T$, we define

$$ $a(t)=\nabla_{\hat{Y}_{t}}\hat{Y}_{T}.$ (53) $$

Then we can derive

$$ $\displaystyle\frac{\,{\rm d}a(t)}{\,{\rm d}t}=$ $\displaystyle\lim_{\epsilon\rightarrow 0^{+}}\frac{a(t+\epsilon)-a(t)}{\epsilon}$ (54) $\displaystyle=$ $\displaystyle\lim_{\epsilon\rightarrow 0^{+}}\frac{a(t+\epsilon)-a(t+\epsilon)\nabla_{\hat{Y}_{t}}(\hat{Y}_{t}-\epsilon\frac{1}{2}\nabla_{\hat{Y}_{t}}\log p_{B}(\hat{Y}_{t},t)+\mathcal{O}(\epsilon^{2}))}{\epsilon}$ $\displaystyle=$ $\displaystyle\lim_{\epsilon\rightarrow 0^{+}}\frac{\epsilon a(t+\epsilon)\nabla^{2}_{\hat{Y}_{t}}\log p_{B}(\hat{Y}_{t},t)+\mathcal{O}(\epsilon^{2})}{2\epsilon}$ $\displaystyle=$ $\displaystyle\frac{1}{2}a(t)\nabla^{2}_{\hat{Y}_{t}}\log p_{B}(\hat{Y}_{t},t),$ $$

where $\nabla^{2}$ is the Hessian operator. Based on Lemma B.2.1, theorem 8 and the fact that $a(T)=\nabla_{\hat{Y}_{T}}\hat{Y}_{T}=I$, $a(t)=\nabla_{\hat{Y}_{t}}\hat{Y}_{T}$ is symmetric. Then theorem 7 shows that the equation $\nabla_{\hat{Y}_{t}}u(\hat{Y}_{t})=\hat{Y}_{T}(\hat{Y}_{t})$ has a solution. Furthermore, since $a(t)$ is a matrix exponential of a symmetric matrix, it must be positive semi-definite. Consequently, the solution $u$ is convex. According to theorem 1, the map $\hat{Y}_{T}(\hat{Y}_{t})$ is optimal for the quadratic transport cost. $\square$

### C.3 Proof of Proposition 3

*Proof.* The definition of $\tilde{Y}_{t+s}$ is given by Equation ([23](#S3.E23)). It can be observed that the term $\tilde{Y}_{t}(\tilde{Y}_{t+s})$ on the right-hand side indicates that the evolution speed of $\tilde{Y}_{t+s}$ is constant, which implies that all particles travel at a uniform rate. Consequently, for a given initial condition $\tilde{Y}_{t}$,

$$ $\tilde{Y}_{t+s}=\tilde{Y}_{t}-\frac{1}{2}\nabla_{\tilde{Y}_{t}}\log p_{B}(\tilde{Y}_{t})s.$ (55) $$

Further, we have:

$$ $\nabla_{\tilde{Y}_{t}}\tilde{Y}_{t+s}=I-\frac{1}{2}\nabla^{2}_{\tilde{Y}_{t}}\log p_{B}(\tilde{Y}_{t})s.$ (56) $$

It is evident that $\nabla_{\tilde{Y}_{t}}\tilde{Y}_{t+s}$ is symmetric and for small values of $s$, it is also positive semi-definite. Based on the same reasoning as the proof of Proposition 2, we can conclude that the map $\tilde{Y}_{t+s}(\tilde{Y}_{t})$ is optimal for the quadratic transport cost. $\square$

### C.4 Proof of Proposition 4

*proof.* As Proposition 1 establishes that $X_{t}=\phi_{t}^{-1}Y_{\beta_{t}}$, the single-time marginal distribution $p_{OU}(x_{t},t)$ for the Ornstein-Uhlenbeck process can be expressed as follows:

$$ $p_{OU}(xt,t)=\frac{1}{N}\sum_{i}^{N}(2\pi\beta_{t}\phi_{t}^{-2})^{-\frac{n}{2}}\exp(-\frac{|x_{t}-\phi_{t}^{-1}x_{i}|^{2}}{2\beta_{t}\phi_{t}^{-2}}).$ (57) $$

The probability flow ODE for Ornstein-Uhlenbeck process is:

$$ $\,{\rm d}\hat{X}_{t}=[-\theta_{t}\hat{X}_{t}-\frac{1}{2}\sigma_{t}^{2}\nabla_{\hat{X}_{t}}\log p_{OU}(\hat{X}_{t},t)]dt.$ (58) $$

We start from $\hat{Y}_{t}$ with the change of variable $Z_{t}=\phi_{t}^{-1}\hat{Y}_{\beta_{t}}$:

$$ $\displaystyle\frac{\,{\rm d}}{\,{\rm d}t}Z_{t}=$ $\displaystyle\frac{\,{\rm d}\phi_{t}^{-1}}{\,{\rm d}t}\hat{Y}_{\beta_{t}}+\phi_{t}^{-1}\frac{\,{\rm d}\hat{Y}_{\beta_{t}}}{\,{\rm d}\beta_{t}}\frac{\,{\rm d}\beta_{t}}{\,{\rm d}t}$ (59) $\displaystyle=$ $\displaystyle-\phi_{t}^{-1}\theta_{t}\hat{Y}_{\beta_{t}}+\phi_{t}^{-1}(\sigma_{t}\phi_{t})^{2}[-\frac{1}{2}\nabla_{\hat{Y}_{\beta_{t}}}p_{B}(\hat{Y}_{t},\beta_{t})]$ $\displaystyle=$ $\displaystyle-\theta_{t}Z_{t}-\frac{1}{2}\phi_{t}\sigma_{t}^{2}\frac{\sum_{i}\exp(-\frac{|\hat{Y}_{\beta_{t}}-x_{i}|^{2}}{2\beta_{t}})\frac{\hat{Y}_{\beta_{t}}-x_{i}}{\beta_{t}}}{\sum_{j}\exp(-\frac{|\hat{Y}_{\beta_{t}}-x_{j}|^{2}}{2\beta_{t}})}$ $\displaystyle=$ $\displaystyle-\theta_{t}Z_{t}-\frac{1}{2}\sigma_{t}^{2}\frac{\sum_{i}\exp(-\frac{|\phi_{t}^{-1}\hat{Y}_{\beta_{t}}-\phi_{t}^{-1}x_{i}|^{2}}{2\beta_{t}\phi_{t}^{-2}})\frac{\phi_{t}^{-1}\hat{Y}_{t}-\phi_{t}^{-1}x_{i}}{\beta_{t}\phi_{t}^{-2}}}{\sum_{j}\exp(-\frac{|\phi_{t}^{-1}\hat{Y}_{\beta_{t}}-\phi_{t}^{-1}x_{j}|^{2}}{2\beta_{t}\phi_{t}^{-2}})}$ $\displaystyle=$ $\displaystyle-\theta_{t}Z_{t}-\frac{1}{2}\sigma_{t}^{2}\frac{\sum_{i}\exp(-\frac{Z_{t}-\phi_{t}^{-1}x_{i}}{2\beta_{t}\phi_{t}^{-2}})\frac{Z_{t}-\phi_{t}^{-1}x_{i}}{\beta_{t}\phi_{t}^{-2}}}{\sum_{j}\exp(-\frac{|Z_{t}-\phi_{t}^{-1}x_{j}|^{2}}{2\beta_{t}\phi_{t}^{-2}})}$ $\displaystyle=$ $\displaystyle-\theta_{t}Z_{t}-\frac{1}{2}\sigma_{t}^{2}\nabla_{Z_{t}}\log p_{OU}(Z_{t},t).$ $$

As $\phi_{0}=1$ and $\beta_{0}=0$, the initial distribution of $X_{0}$ and $Z_{0}$ is the same. Consequently, $\hat{X}_{t}$ and $Z_{t}$ follow the same ODE with identical initial conditions. Thus, we have $\hat{X}_{t}=Z_{t}=\phi_{t}^{-1}\hat{Y}_{\beta_{t}}$ and $\nabla_{\hat{X}_{t}}\hat{X}_{t+s}=\frac{\phi_{t}}{\phi_{t+s}}\nabla_{\hat{Y}_{\beta_{t}}}\hat{Y}_{\beta_{t+s}}$, which is symmetric and positive semi-definite by Proposition 2. Therefore, we can conclude that the map $\hat{X}_{t+s}(\hat{X}_{t})$ is optimal for the quadratic transport cost. $\square$

### C.5 Proof of Proposition 5

*Proof.* For the original process (discrete analogue of Brownian motion), the transition rate is:

$$ ${{Q_{D}}^{i}_{j}=}\begin{cases}1,&d_{D}(i,j)=1,\\ \sum\limits_{j\in N(i)}-{Q_{D}}^{i}_{j},&i=j,\\ 0,&others,\end{cases}$ (60) $$

where $N(i)=\{k:d_{D}(i,k)=1\}$. The Kolmogorov forward equation of this process is written as:

$$ $\displaystyle\frac{d{{P_{D}}_{i}{(t)}}}{dt}=$ $\displaystyle\sum\limits_{i^{{}^{\prime}}}{P_{D}}_{i{{}^{\prime}}}{(t)}{Q_{D}}^{i^{{}^{\prime}}}_{i}$ (61) $\displaystyle=$ $\displaystyle{P_{D}}_{i}{(t)}\times\sum\limits_{i^{{}^{\prime}}\in N(i)}-{Q_{D}}^{i}_{i^{{}^{\prime}}}+\sum\limits_{i^{{}^{\prime}}\in N(i)}{P_{D}}_{i^{{}^{\prime}}}{(t)}\times 1$ $\displaystyle+\sum\limits_{i^{{}^{\prime}}\in\{k:d_{D}(i,k)>1\}}{P_{D}}_{i^{{}^{\prime}}}{(t)}\times 0$ $\displaystyle=$ $\displaystyle\sum\limits_{i^{{}^{\prime}}\in N(i)}({P_{D}}_{i^{{}^{\prime}}}{(t)}-{P_{D}}_{i}{(t)}).$ $$

In contrast, the transition rate of our new process is:

$$ ${{Q}^{i}_{j}=}\begin{cases}\frac{ReLU({P_{D}}_{i}{(t)}-{P_{D}}_{j}{(t)})}{{P_{D}}_{i}{(t)}},&d_{D}(i,j)=1\\ \sum\limits_{d_{D}(i,j)=1}-{Q}^{i}_{j},&i=j\\ 0,&others.\end{cases}$ (62) $$

Our new process can be written as:

$$ $\displaystyle\frac{d{P_{i}{(t)}}}{dt}=$ $\displaystyle\sum\limits_{i^{{}^{\prime}}}P_{i^{{}^{\prime}}}{(t)}{Q}^{i^{{}^{\prime}}}_{i}$ (63) $\displaystyle=$ $\displaystyle P_{i}{(t)}\times\sum\limits_{i^{{}^{\prime}}\in N(i)}-{Q}^{i}_{i^{{}^{\prime}}}$ $\displaystyle+\sum\limits_{i^{{}^{\prime}}\in N(i)}P_{i^{{}^{\prime}}}{(t)}\times\frac{ReLU({P_{D}}_{i^{{}^{\prime}}}{(t)}-{P_{D}}_{i}{(t)})}{{P_{D}}_{i^{{}^{\prime}}}{(t)}}$ $\displaystyle+\sum\limits_{i^{{}^{\prime}}\in\{k:d_{D}(i,k)>1\}}P_{i^{{}^{\prime}}}{(t)}\times 0$ $\displaystyle=$ $\displaystyle-\sum\limits_{i^{{}^{\prime}}\in N(i)}P_{i}{(t)}\frac{ReLU({P_{D}}_{i}{(t)}-{P_{D}}_{i^{{}^{\prime}}}{(t)})}{{P_{D}}_{i}(t)}$ $\displaystyle+\sum\limits_{i^{{}^{\prime}}\in N(i)}P_{i^{{}^{\prime}}}(t)\frac{ReLU({P_{D}}_{i^{{}^{\prime}}}{(t)}-{P_{D}}_{i}{(t)})}{{P_{D}}_{i}^{{}^{\prime}}(t)}.$ $$

Substitute $P=P_{D}$ in Equation ([63](#A3.E63)), we get the same form in [61](#A3.E61), which means $P_{D}$ also solves the Equation ([63](#A3.E63)). Thus, $P_{i}(t)={P_{D}}_{i}(t)$, $\forall t\geq 0,i\in\{1,2,\cdots,S\}^{K}$, according to Picard-Lindelöf theorem. $\square$

### C.6 Proof of Proposition 6

Let $a=P(t)$, $b=P(t+\varepsilon)$. As our generator only allows flux between adjacent states, we define the transport map $\Pi^{*}\in\mathbb{R}^{k\times k}$ as:

$$ ${\Pi^{*}}^{i}_{j}=\int_{t}^{t+\epsilon}P_{i}(t)Q^{i}_{j}(t)\,{\rm d}t,$ (64) $$

which is the probability transported from state $i$ to state $j$ in the time interval $[t,t+\epsilon]$. As the probability $P(t)$ is continuous with respect to time $t$, we choose the $\epsilon$ such that the sign of all the quantities $P_{i}(t)-P_{j}(t)$ for $\{i,j\in\{1,2,\cdots,S\}^{K}:d_{D}(i,j)=1\}$ do not change. Under this assumption, the flux directions do not change at every state.

We claim that $\Pi^{*}$ solves the optimal transport problem:

$$ $\displaystyle\min\limits_{\Pi}\sum\limits_{i,j}\Pi_{j}^{i}d_{D}(i,j),$ (65) $\displaystyle s.t.\begin{cases}&\sum\limits_{i}\Pi^{i}=b,\\ &\sum\limits_{j}\Pi_{j}=a,\\ &\Pi\geq 0.\end{cases}$ $$

*Proof.* The Lagrangian function for this optimization problem is:

$$ $L(\Pi,\psi,\phi,\lambda)=\sum\limits_{i,j}\Pi^{i}_{j}d_{D}(i,j)+\psi_{i}(\Pi^{i}_{j}-a_{i})+\phi_{j}(\Pi^{i}_{j}-b_{j})-\lambda^{i}_{j}\Pi^{i}_{j},$ (66) $$

where $\psi_{i}\text{,~{}}\phi_{j}\text{~{}and~{}}\lambda^{i}_{j}$ are Lagrange multipliers.
According to Remark 5 and the fact that this is a linear programming, $\Pi^{*}$ is optimal if and only if there exists a set of $\phi_{i}\text{,~{}}\psi_{j}\text{~{}and~{}}\lambda^{i}_{j}$ that satisfy the following equations for $\forall~{}i,j$:

$$ $\displaystyle d_{D}(i,j)+\psi_{i}+\phi_{j}-\lambda_{j}^{i}$ $\displaystyle=0$ (67a) $\displaystyle\lambda^{i}_{j}$ $\displaystyle\geq 0$ (67b) $\displaystyle\lambda^{i}_{j}\Pi^{i}_{j}$ $\displaystyle=0$ (67c) $\displaystyle\sum\limits_{i}\Pi^{i}_{j}$ $\displaystyle=b_{j}$ (67d) $\displaystyle\sum\limits_{j}\Pi_{j}^{i}$ $\displaystyle=a_{i}$ (67e) $\displaystyle\Pi^{i}_{j}$ $\displaystyle\geq 0.$ (67f) $$

Firstly, we consider the $i,j$ pairs where $d_{D}(i,j)\leq 1$. In this case ${\Pi^{*}}_{j}^{i}~{}\text{may}~{}>0$ (thus $\lambda^{i}_{j}=0$). Besides, the Equation ([64](#A3.E64)) indicates that ${\Pi^{*}}^{i}_{i}>0$, thus we have $\lambda^{i}_{i}=0$. Then the Equation ([67a](#A3.E67.1)) comes to:

$$ $\psi_{i}+\phi_{i}=0.$ (68) $$

According to the construction of our generator $Q$, there is no mutual flux, thus we obtain:

$$ $\Pi^{i_{1},\dots,i_{l},\dots,i_{K}}_{i_{1},\dots,i_{l}+1,\dots,i_{K}}\neq 0\text{ or }\Pi^{i_{1},\dots,i_{l}+1,\dots,i_{K}}_{i_{1},\dots,i_{l},\dots,i_{K}}\neq 0,~{}~{}\forall~{}i.$ (69) $$

By substituting this result into complementary slackness condition [67c](#A3.E67.3), we have:

$$ $\lambda^{i_{1},\dots,i_{l},\dots,i_{K},}_{i_{1},\dots,i_{l}+1,\dots,i_{K}}=0\text{ or }\lambda^{i_{1},\dots,i_{l}+1,\dots,i_{K}}_{i_{1},\dots,i_{l},\dots,i_{K}}=0.$ (70) $$

Since $d_{D}([i_{1},\dots,i_{l},\dots,i_{K}],[i_{1},\dots,i_{l}+1,\dots,i_{K}])=1$, from Equation ([67a](#A3.E67.1)), we can obtain:

$$ $1+\psi_{i_{1},\dots,i_{l},\dots,i_{K}}+\phi_{i_{1},\dots,i_{l}+1,\dots,i_{K}}=0\text{ or }1+\psi_{i_{1},\dots,i_{l}+1,\dots,i_{K}}+\phi_{i_{1},\dots,i_{l},\dots,i_{K}}=0.$ (71) $$

Solving Equations ([68](#A3.E68)) and ([71](#A3.E71)) simultaneously, we get:

$$ $\left\{\begin{aligned} &\psi_{i_{1},\dots,i_{l}+1,\dots,i_{K}}=1+\psi_{i_{1},\dots,i_{l},\dots,i_{K}}\\ &\phi_{i_{1},\dots,i_{l}+1,\dots,i_{K}}=-1+\phi_{i_{1},\dots,i_{l},\dots,i_{K}}\end{aligned}\right.\text{ or }\left\{\begin{aligned} &\psi_{i_{1},\dots,i_{l}+1,\dots,i_{K}}=-1+\psi_{i_{1},\dots,i_{l},\dots,i_{K}}\\ &\phi_{i_{1},\dots,i_{l}+1,\dots,i_{K}}=1+\phi_{i_{1},\dots,i_{l},\dots,i_{K}}\end{aligned}\right..$ (72) $$

Therefore, given $\psi_{0,\dots,0}$, $\psi_{i_{1},\dots,i_{K}}$ and $\phi_{i_{1},\dots,i_{K}}$ can be calculated by:

$$ $\displaystyle\psi_{i_{1},\dots,i_{K}}$ $\displaystyle=\psi_{0,\dots,0}+m^{i_{1},\dots,i_{K}}_{0,\dots,0}-n^{i_{1},\dots,i_{K}}_{0,\dots,0},$ (73) $\displaystyle\phi_{i_{1},\dots,i_{K}}$ $\displaystyle=-\psi_{i_{1},\dots,i_{K}},$ (74) $$

where $m^{i_{1},\dots,i_{K}}_{0,\dots,0}+n^{i_{1},\dots,i_{K}}_{0,\dots,0}=d_{D}(0,i)$, $m^{i_{1},\dots,i_{K}}_{0,\dots,0}\in\mathbb{N}_{0}$, and $n^{i_{1},\dots,i_{K}}_{0,\dots,0}\in\mathbb{N}_{0}$. $\mathbb{N}_{0}$ represents the set of all non-negative integers. The quantity $m^{i_{1},\dots,i_{K}}_{0,\dots,0}$ is the number where $\Pi^{i_{1},\dots,i_{l},\dots,i_{K}}_{i_{1},\dots,i_{l}+1,\dots,i_{K}}\neq 0$ and $n^{i_{1},\dots,i_{K}}_{0,\dots,0}$ is the number where $\Pi^{i_{1},\dots,i_{l}+1,\dots,i_{K}}_{i_{1},\dots,i_{l},\dots,i_{K}}\neq 0$. Consequently, we find all the Lagrange multipliers for $d_{D}(i,j)\leq 1$

Then, we consider $i,j$ pairs when $d_{D}(i,j)>1$ which indicates ${\Pi^{*}}^{i}_{j}=0$. We use $\psi_{i}$ and $\phi_{j}$ in Equation ([73](#A3.E73)) and ([74](#A3.E74)). To satisfy the KKT condition, we only need to verify that there is $\lambda^{i}_{j}$ satisfies Equation ([67b](#A3.E67.2)) and Equation ([67a](#A3.E67.1)). From Equation ([67a](#A3.E67.1)), $\lambda^{i}_{j}$ can be written as:

$$ $\lambda^{i}_{j}=d_{D}(i,j)+\psi_{i}+\phi_{j}$ (75) $$

Let $r_{l}=\min(i_{l},j_{l})$, it has:

$$ $\displaystyle d_{D}{(i,j)}$ $\displaystyle=d_{D}{(i,r)}+d_{D}{(j,r)}$ (76) $\displaystyle\psi_{i}$ $\displaystyle=\psi_{r}+m_{i}^{r}-n_{i}^{r}$ (77) $\displaystyle\phi_{j}$ $\displaystyle=-\psi_{j}=-\psi_{r}-m_{j}^{r}+n_{j}^{r}$ (78) $\displaystyle d_{D}{(i,r)}$ $\displaystyle=m_{i}^{r}+n_{i}^{r}$ (79) $\displaystyle d_{D}{(j,r)}$ $\displaystyle=m_{j}^{r}+n_{j}^{r}$ (80) $\displaystyle m_{i}^{r},n_{i}^{r},m_{j}^{r},n_{j}^{r}$ $\displaystyle\geq 0.$ (81) $$

Substitute the above results to Equation ([75](#A3.E75)), we have:

$$ $\displaystyle\lambda^{i}_{j}$ $\displaystyle=2(m_{i}^{r}+n_{j}^{r})\geq 0$ (82) $$

As a result, we find all the Lagrange multipliers. Since Equations ([67d](#A3.E67.4)) and ([67e](#A3.E67.5)) are naturally satisfied by the construction of $\Pi^{*}$, we conclude that the KKT conditions are met at $\Pi^{*}$:

\small{1}⃝ Primal Feasibility: ([67d](#A3.E67.4)), ([67e](#A3.E67.5)), ([67f](#A3.E67.6))

\small{2}⃝ Dual Feasibility: ([67a](#A3.E67.1)), ([67b](#A3.E67.2))

\small{3}⃝ Complementary slackness: ([67c](#A3.E67.3))

According to Remark 5, the KKT conditions indicate $\Pi^{*}$ is a solution to the optimal transport problem ([65](#A3.E65)). $\square$

### C.7 Proof of Proposition 7

This is a special case of Proposition 6, where the generator $Q$ remains constant throughout time.

Figure: Algorithm 1 Generative Reverse Process with Discrete Probability Flow (DPF)

## Appendix D Experiment

### D.1 Algorithm

Our training process follows the same procedure as SDDM, with the distinction that our forward process incorporates the rate we formulated in Equation ([28](#S4.E28)) to align with optimal transport theory. The loss function employed during the training process is as follows:

$$ $\theta^{*}=\arg\min\limits_{\theta}\int_{0}^{T}\sum\limits_{i_{t}\in\{1,2,\cdots,S\}^{K}}q_{t}(i_{t})\left[\sum\limits_{l=1}^{K}-\log P_{t}(i_{t}^{l}|i_{t}\backslash i_{t}^{l})\right]dt.$ (83) $$

The sampling process with the proposed discrete probability flow is shown in Algorithm [1](#algorithm1). In our algorithm, as $R$ is non-zero only when $d_{D}{(i_{t},i_{t-\tau})}\leq 1$, the calculation of the reverse transition rate $R$ (as defined in Equation [30](#S4.E30)) is divided into three cases: staying in the current state ( $i^{l}_{t-\tau}=i^{l}_{t}$, i.e., "stay"), jumping to the next state ($i^{l}_{t-\tau}=i^{l}_{t}+1$, i.e., "add"), and jumping to the previous state ($i^{l}_{t-\tau}=i^{l}_{t}-1$, i.e., "sub"). By combining the rate in these situations, we can derive $P_{\theta}(i_{t-\tau}^{l}|i_{t})$ from ([31](#S4.E31)) and ([32](#S4.E32)), which allows us to sample the next state accordingly. This process continues iteratively until $t=0$.

**Table 3: Average MMD between different distributions of data.**
| State size | 2 | 5 | 10 |
| --- | --- | --- | --- |
| Average MMD | 5.336e-3 | 2.201e-2 | 6.531e-3 |
|  |  |  |  |

Figure: Figure 6: Visualization of the generation quality on generated samples with state size = 5 for SDDM and DPF.
Refer to caption: /html/2311.03886/assets/x6.png

Figure: Figure 7: Visualization of the generation quality on generated samples with state size = 10 for SDDM and DPF.
Refer to caption: /html/2311.03886/assets/x7.png

### D.2 Synthetic Dataset

Following , we utilize synthetic data for model validation. Initially, we generate 2D floating-point data from seven distinct distributions using an infinite data oracle. By employing the same settings as , we convert each dimension of the data into 16-bit Gray code, resulting in a dataset with discrete dimension = 32 and state size = 2. However, it is not sufficient to validate our method solely on the dataset with state size = 2, since $Q$ in Equation ([28](#S4.E28)) does not cover cases where $d_{D}(i,j)>1$. Therefore, we further transform the same data into 8-bit 5-base code and 6-bit decimal code respectively, thereby creating two additional datasets: one with dimension = 16 and state size = 5, and another with dimension = 12 and state size = 10.

### D.3 Experiment Details

In the experiments, our neural network consists of a 3-layer MLP with 256 channels . We employ the Adam optimizer with a learning rate of 1e-4. The model is trained on a single NVIDIA Quadro RTX 8000, utilizing a batch size of 128 for 300,000 iterations. During training, the parameter $t$ is uniformly sampled from the range of 0 to 1. For the sampling process, the data is generated through 1,000 steps (i.e.,$\tau$ is set to 0.001).

Figure: Figure 8: Visualization of the generating certainty on generated samples with state size = 5 for SDDM and DPF. All the samples (in blue) are randomly generated from the single initial point (in red).
Refer to caption: /html/2311.03886/assets/x8.png

Figure: Figure 9: Visualization of the generating certainty on generated samples with state size = 10 for SDDM and DPF. All the samples (in blue) are randomly generated from the single initial point (in red).
Refer to caption: /html/2311.03886/assets/x9.png

### D.4 Quality of Generated Samples

To evaluate the sampling quality, we generate 40,000 samples for binary data, and 4,000 samples for other type of synthetic data by SDDM and our method. Then we compare these generated samples to true data using Laplace MMD. This evaluation is repeated 10 times, and the average results are presented in Table [1](#S6.T1). It is worth noting that the unbiased estimation of MMD is an approximation by Monte Carlo method, which may cause negative results. It is observed that MMD score of our method is slightly higher than that of SDDM. This is mainly caused by the approximation of $Q_{t}$. In the sampling process, two terms are present on the right-hand side of Eq. [10](#S2.E10): $\frac{q_{t}(y)}{q_{t}(x)}$ and $Q_{t}(y,x)$. In SDDM, only one term, i.e., $\frac{q_{t}(y)}{q_{t}(x)}$ is estimated using a neural network, as ${Q_{D}}_{t}(y,x)$ is known. Different from SDDM, both terms in our method are evaluated using quantities approximated by the neural network, since our $Q_{t}(y,x)$ is dependent on $\frac{q_{t}(y)}{q_{t}(x)}$ (Eq. [27](#S4.E27)). This approximation may lead to slightly inferior quality than the SDDM using precise ${Q_{D}}_{t}(y,x)$. Due to this being a neural network fitting error, we currently have no feasible alternative approximations to achieve a superior outcome.

To assess the significance of these differences, we presented the MMD between different distributions of real data in Table [3](#A4.T3). Taking this result as a reference, we can find that the gap of the MMD score between DPF and SDDM is very small, which is not enough to affect the quality of the generation. This conclusion is further supported by the visualization of the generated samples in Figure [1](#S6.F1), Figure [6](#A4.F6) and Figure [7](#A4.F7) also confirm this point, which demonstrates that DPF produces samples of comparable quality to SDDM.

**Table 4: Comparison of the average $L_{1}$ distance between the generated samples and initial point. Lower values indicate that the generated sample is closer to initial point.**
|  | 2spirals | 8gaussians | checkerboard | circles | moons | pinwheel | swissroll |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | discrete dimension = 32, state size = 2 |  |  |  |  |  |  |
| SDDM | 13.5595 | 13.3025 | 13.4710 | 13.5848 | 13.6485 | 13.4875 | 13.6962 |
| DPF (ours) | 1.5965 | 1.3855 | 0.7525 | 1.1875 | 1.8693 | 1.8135 | 1.6955 |
|  | discrete dimension = 16, state size = 5 |  |  |  |  |  |  |
| SDDM | 12.7220 | 12.4698 | 12.4833 | 12.5390 | 12.6745 | 12.6238 | 12.7510 |
| DPF (ours) | 1.5265 | 1.6155 | 0.7090 | 1.1668 | 1.7038 | 1.8088 | 1.5888 |
|  | discrete dimension = 12,<br>state size = 10 |  |  |  |  |  |  |
| SDDM | 11.3433 | 11.0083 | 11.0243 | 11.2205 | 11.6850 | 11.3895 | 11.6333 |
| DPF (ours) | 1.7655 | 1.1940 | 0.7143 | 1.1588 | 2.0493 | 1.9283 | 1.7695 |
|  |  |  |  |  |  |  |  |

Figure: Figure 10: Visualization of the generated samples with state size = 5 from the given initial points $\bm{x}_{T}$. Different colors distinguish the generated samples from different initial points $\bm{x}_{T}$.
Refer to caption: /html/2311.03886/assets/x10.png

Figure: Figure 11: Visualization of the generated samples with state size = 10 from the given initial points $\bm{x}_{T}$. Different colors distinguish the generated samples from different initial points $\bm{x}_{T}$.
Refer to caption: /html/2311.03886/assets/x11.png

### D.5 Standard Deviation of Generated Samples

To evaluate the certainty of generated samples, we randomly select a 2D float-point and fix it as the initial point. In this experiments, we use the same initial point (-1.91, 1.57). In the binary case, the point is converted into Gray code, whereas in the 5-base and decimal cases, the original code is utilized. For each dataset, we generate 4,000 points and compute the Expectation of the Conditional Standard Deviation ([33](#S4.E33)). Our DPF method results in a significant reduction in the $CSD$ score, as presented in Table [2](#S6.T2). For example, on the checkerboard dataset with $S=5$, our DPF achieves the best score of 1.4103, which is 89% lower than the score achieved by SDDM. We also visualize these results in Figure [2](#S6.F2), Figure [8](#A4.F8), and Figure [9](#A4.F9), where the red star represents the initial point, and the blue points denote the generated samples. Furthermore, it is evident that SDDM generates samples from a single initial point across the entire space, especially for datasets with $S=2$. In contrast, our method can only reach a limited number of states from the initial point, indicating the superior sampling certainty of our approach.

We can also observe that our sampling results tend to form rectangles in Figure [2](#S6.F2), Figure [8](#A4.F8), and Figure [9](#A4.F9). This phenomenon arises from the construction of our synthetic dataset. Specifically, we construct the synthetic dataset states and dimensions by encoding the $x$-axis and y-axis coordinates of the toy dataset (normalized to [0, 1]) into $K/2$-bit $S$-ary. This is equivalent to dividing the data space into rectangle regions, where the first few dimensions determine the approximate location of the data. Since our proposed method significantly reduces the uncertainty, each dimension (including the first few dimensions) has only a limited number of possible values. As a result, the points in Figure [2](#S6.F2), Figure [8](#A4.F8), and Figure [9](#A4.F9) appear to form rectangles.

### D.6 Generated Samples from Different Initial Points

To display the generated samples from various initial points, we select a $20\times 20$ grid of initial points and mark them with distinct colors. Subsequently, we generate 10 samples for each initial point and presented the results in Figure [3](#S6.F3), Figure [10](#A4.F10), and Figure [11](#A4.F11). It is apparent that the samples obtained through SDDM sampling are mixed together. In contrast, the results obtained by our method exhibit strong regularity, with the generated samples clustering together based on their respective colors. This observation suggests that our method offers improved certainty in the sampling process.

**Table 5: Comparison of average trajectory length. Lower value indicates a better transport plan.**
|  | 2spirals | 8gaussians | checkerboard | circles | moons | pinwheel | swissroll |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | discrete dimension = 32, state size = 2 |  |  |  |  |  |  |
| SDDM | 32.0075 | 31.7275 | 31.8010 | 32.0258 | 32.0240 | 31.8650 | 31.9988 |
| DPF (ours) | 1.6135 | 1.3980 | 0.7640 | 1.1995 | 1.8883 | 1.8265 | 1.7065 |
|  | discrete dimension = 16, state size = 5 |  |  |  |  |  |  |
| SDDM | 26.0680 | 25.3258 | 25.9708 | 25.7565 | 25.8815 | 25.9793 | 26.0810 |
| DPF (ours) | 1.5425 | 1.6390 | 0.7275 | 1.1758 | 1.7102 | 1.8178 | 1.5973 |
|  | discrete dimension = 12,<br>state size = 10 |  |  |  |  |  |  |
| SDDM | 21.8558 | 21.7828 | 21.7778 | 21.9100 | 22.3180 | 21.9985 | 22.2688 |
| DPF (ours) | 1.7835 | 1.1995 | 0.7213 | 1.1793 | 2.0623 | 1.9433 | 1.7870 |
|  |  |  |  |  |  |  |  |

### D.7 Distance Between the Generated Samples and Initial Points

Our DPF is designed based on the theory of optimal transport, as demonstrated in Proposition 6. Here, we aim to reflect this finding through experimentation as well. To accomplish this, we utilize the generated samples from Figure [3](#S6.F3), Figure [10](#A4.F10) and Figure [11](#A4.F11), and calculate the average $L_{1}$ distance from the generated samples to the corresponding initial point:

$$ $d_{D}(i(0),i(T))=\sum\limits_{l=1}^{S}|i^{l}(0)-i^{l}(T)|.$ (84) $$

The results are presented in Table [4](#A4.T4). It is evidence that DPF greatly reduces the distance between the generated samples and the initial point. Moreover, combined with the visualization results in Figure [3](#S6.F3), Figure [10](#A4.F10) and Figure [11](#A4.F11), we observe that our method’s sampling outcomes tend to concentrate around the high probability states near the initial point. This outcome aligns with our optimal transport design, further verifying the efficacy of our approach.

However, there is an illusion that the difference between SDDM and PDF decreases as the state size increases. This is mainly because that the Figure [3](#S6.F3), Figure [10](#A4.F10) and Figure [11](#A4.F11) are visualized in the ’float space’ instead of the ’encoding space’. Specifically, our synthetic data with a state size of and a dimension size of is established by encoding the $x$ and $y$ coordinates of the toy dataset (normalized to [0, 1]) to $K/2$-bit $S$-ary respectively. In this encoding, the first dimension of the encoding has the greatest impact on the data position. For example, in binary encoding (state size = 2), the first bit divides the data space into two parts, and determines the part in which it resides. However, as the number of states increases, the space is divided into more parts, and the small change of the first bit can not significantly change the position of the number it represents. This will lead to a narrowing of the gap between our DPF and SDDM in the visualization. Therefore, in such situations, the quantitative results in Table [2](#S6.T2) are more appropriate for verifying the reduction of uncertainty in the encoding space.

**Table 6: Comparison of transport efficiency. Larger values indicate better transport efficiency.**
|  | 2spirals | 8gaussians | checkerboard | circles | moons | pinwheel | swissroll |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | discrete dimension = 32, state size = 2 |  |  |  |  |  |  |
| SDDM | 42.36% | 41.93% | 42.36% | 42.42% | 42.62% | 42.33% | 42.80% |
| DPF (ours) | 98.95% | 99.11% | 98.49% | 99.00% | 98.99% | 99.29% | 99.36% |
|  | discrete dimension = 16, state size = 5 |  |  |  |  |  |  |
| SDDM | 48.80% | 49.24% | 48.07% | 48.68% | 48.97% | 48.59% | 48.89% |
| DPF (ours) | 98.96% | 98.56% | 97.46% | 99.23% | 99.63% | 99.50% | 99.47% |
|  | discrete dimension = 12,<br>state size = 10 |  |  |  |  |  |  |
| SDDM | 51.90% | 50.54% | 50.62% | 51.21% | 52.36% | 51.77% | 52.24% |
| DPF (ours) | 98.99% | 99.54% | 99.02% | 98.26% | 99.36% | 99.22% | 99.02% |
|  |  |  |  |  |  |  |  |

Figure: Figure 12: Visualization of the sampling trajectory. The yellow box highlights the duplicated trajectories encountered during the sampling process.
Refer to caption: /html/2311.03886/assets/x12.png

### D.8 Sampling Trajectory Length

Merely examining the distance between the initial point and the generated samples is insufficient to verify that our DPF aligns with optimal transport principles, as the sampling process may follow different trajectories. Therefore, we calculate the cumulative consumption of the sampling trajectory in Figure [3](#S6.F3), Figure [10](#A4.F10) and Figure [11](#A4.F11) according to the following formula:

$$ $d_{tra}(i(0),\dots,i(T))=\sum\limits_{t\in\{\tau,2\tau\dots,T\}}d_{D}(i(t),i(t-\tau)).$ (85) $$

The results are shown in Table [5](#A4.T5). It is evidence that there is a significant decrease in the trajectory length of DPF compared to the SDDM. For instance, our DPF achieves the best score on the checkerboard dataset with S = 5 with a score of 0.7640, which is 97% lower than the score of SDDM. This suggests that the consumption during our sampling process is lower, which is in line with the optimal transport design.

### D.9 Transport Efficiency

We also examine the transport efficiency during the sampling process, which can be calculated as the ratio of $L_{1}$ distance between the initial point and generated sample to the sampling trajectory length:

$$ $E_{(i(0),\dots,i(T))}=\frac{d_{D}(i(0),i(T))}{d_{tra}(i(0),\dots,i(T))}.$ (86) $$

A higher value indicates a more optimal sampling trajectory selected by the model from the initial point to the generated sample, i.e., the higher transport efficiency. The results are presented in Table [6](#A4.T6). Notably, we observed that only approximately 50% of the trajectory length of SDDM contributes to the actual distance between the initial point and generated samples. In contrast, the transport efficiency of our DPF is close to 100%, which means most jumps in our trajectory efficiently contribute to the final transition. This finding demonstrates that the transport plan selected by our DPF is more effective, aligning with our theoretical derivation.

### D.10 Visualization of Sampling Trajectory

The visualization of the sampling trajectory for the 0-th dimension of the dataset is shown in Figure [12](#A4.F12). It is evident that the sampling trajectory of SDDM often exhibits duplicate trajectories, which is also the reason for the low transport efficiency of SDDM in Table [6](#A4.T6). In contrast, our method, which adheres to the principles of optimal transport theory, ensures that the sampling process only moves toward high probability states, thereby avoiding the occurrence of duplicate trajectories.

### D.11 Higher dimension or state scenarios

To further verify our method is still applicable in higher dimension or state scenarios, we increased the number of states and dimension to 50 for experiments. Specifically, we set $S=50,K=20$ and $S=5,K=50$ to avoid dimension redundancy in the $K/2$-bit $S$-ary encoding for the toy dataset (float64) coordinates (i.e., $50^{20/2}<2^{64}$ and $5^{50/2}<2^{64}$). The results of this experiment are shown in Table [7](#A4.T7), which clearly demonstrate that our method can significantly reduce sampling uncertainty even with larger state and dimension sizes.

### D.12 Image Modeling

In addition to the transition rate designed in Eq. [26](#S4.E26), our method can also be extended to a broad range of transition rates. For example, we can extend our discrete probability flow to the method in .
For general $Q_{D_{t}}$ with ${Q_{D}}^{i}_{j}(t)={Q_{D}}^{j}_{i}(t)$, define:

$$ $Q^{i}_{j}(t)=\begin{cases}{Q_{D}}^{i}_{j}(t)\frac{{ReLU}(P_{D_{i}}(t)-P_{D_{j}}(t))}{P_{D_{i}}(t)},&i\neq j,\\ -\sum_{j\neq i}Q^{i}_{j},&i=j.\end{cases}$ (87) $$

**Table 7: Application on higher dimension or state scenarios. Lower $CSD$ indicate superior certainty.**
|  | 2spirals | 8gaussians | checkerboard | circles | moons | pinwheel | swissroll |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | discrete dimension = 20, state size = 50 |  |  |  |  |  |  |
| SDDM | 25.8777 | 26.5288 | 25.5106 | 25.7398 | 25.6984 | 25.6984 | 6.4767 |
| DPF (ours) | 2.7113 | 4.4274 | 2.6217 | 3.7554 | 3.0774 | 3.3054 | 3.8183 |
|  | discrete dimension = 50, state size = 5 |  |  |  |  |  |  |
| SDDM | 47.2706 | 47.5810 | 47.4964 | 47.2733 | 47.0047 | 46.9103 | 46.9819 |
| DPF (ours) | 2.0335 | 1.8134 | 0.7418 | 1.7143 | 1.2840 | 1.4245 | 1.5720 |
|  |  |  |  |  |  |  |  |

${Q_{D_{t}}}$ and ${Q_{t}}$ have the same single-time marginal distribution. Let $q_{t}=P_{D}(t)$ and $x\neq y$, the reverse transition rate can be written as:

$$ $\displaystyle R_{t}(x,y)$ $\displaystyle=\frac{q_{t}(y)}{q_{t}(x)}Q_{t}(y,x)$ (88a) $\displaystyle=\frac{q_{t}(y)}{q_{t}(x)}Q_{D_{t}}(y,x)\frac{ReLU(q_{t}(y)-q_{t}(x))}{q_{t}(y)}$ (88b) $\displaystyle=Q_{D_{t}}(y,x)RELU(\frac{q_{t}(y)}{q_{t}(x)}-1)$ (88c) $$

In the same way, the reverse rate in the paper can be written into the following form:

$$ $\hat{R}_{t}^{1:D}(\bm{x}^{1:D},\tilde{\bm{x}}^{1:D})=\sum_{d=1}^{D}R_{t}^{d}(\tilde{x}^{d},x^{d})\delta_{\bm{x}^{1:D\backslash d},\tilde{\bm{x}}^{1:D\backslash d}}RELU(\sum_{x_{0}^{d}}q_{0|t}(x_{0}^{d}|\bm{x}^{1:D})\frac{q_{t|0}(\tilde{x}^{d}|x_{0}^{d})}{q_{t|0}(x^{d}|x_{0}^{d})}-1)$ (89) $$

**Table 8: Comparison of certainty for $\tau$LDR-0 and DPF on the Cifar-10 dataset. Here, $CSD$ , class-std, and class-entropy are calculated on 1,000 initial points, each of which has 10 generated images. Lower values indicate superior certainty.**
|  | $CSD$ | class-std | class-entropy |
| --- | --- | --- | --- |
| $\tau$LDR-0 | 57.6898 | 2.6628 | 1.7703 |
| DPF (ours) | 9.4420 | 1.1819 | 0.5291 |
|  |  |  |  |

In this way, the mutual flow between states is eliminated, greatly reducing the sampling uncertainty. To validate this, we validated our DPF on the CIFAR-10 dataset, using the pre-trained discrete diffusion model provided by the paper . Firstly, we selected 1,000 initial points, and sampled 10 images from each initial point. To measure the sampling certainty on the image data, we used a pre-trained CIFAR-10 classifier to classify the image, and introduce two new metrics, i.e., class-std and class-entropy. The class-std calculates the standard deviation of the categories of the images sampled from the same initial point. While the class-entropy calculates the entropy of the category distribution of the images sampled from the same initial point. Lower class-std and class-entropy indicate better sampling certainty. The experimental results, shown in Table [8](#A4.T8), demonstrate that our method can significantly reduce the sample uncertainty compared to the $\tau$LDR-0 method. Additionally, we visualized the sampled images in Fig. [4](#S8.F4). It was clear that from an initial point, our method samples almost the same images, while the original sampling method obtains totally different images.

## Appendix E Discussion

### E.1 Narrow time interval limited in Proposition 6.

We limit the time frame to a narrow interval, as the validity of the proof hinges on the constancy of the sign of $P_{i}(t)-P_{j}(t)$.
Alternatively, if both equations in Eq. (70) are established concurrently, a contradiction arises whereby 2 equals 0.
Consequently, the KKT condition cannot be satisfied by any suitable Lagrange multipliers, thereby rendering the plan sub-optimal.

From an intuitive standpoint, DPF only avoids instantaneous mutual flow, which does not ensure the elimination of mutual flow during finite interval. For example, if we assume $P_{i}(t)>P_{j}(t)$ in the interval $[t,t+\epsilon/2)$ and $P_{i}(t)<P_{j}(t)$ in $(t+\epsilon/2,t+\epsilon]$, it follows that $\Pi^{i}_{j}>0$ and $\Pi^{j}_{i}>0$. Assuming $\Pi^{i}_{j}>\Pi^{j}_{i}$, we can demonstrate that the given plan is sub-optimal. If we define a new plan as ${\Pi^{*}}^{i}_{j}=\Pi^{i}_{j}-\Pi^{j}_{i}$ and ${\Pi^{*}}^{j}_{i}=0$, we can verify that the resultant plan ${\Pi^{*}}$ incurs a lower transportation cost than $\Pi$.
The preceding derivation establishes the tightness of our announcement, indicating that the optimal transport plan cannot be extended across the entire time interval.

In order to confirm the existence of such a scenario, we explicitly construct it in the case where $K=1$.
Since $P(t)=P(0)e^{Q_{D}t}$, we can obtain the analytical solution through eigen decomposition with difference equations, yielding the following outcomes: $\lambda_{i}=2cos(i\pi/S)-2$ and $v_{i}=(1,cos(\theta_{i}/2),...,cos((2S-1)\theta_{i}/2))$, where $\theta_{i}=\arccos((\lambda_{i}+2)/2)$.
Subsequently, we can assess a basic scenario wherein $S=3$ and $P(t=0)=(0.1,0,0.9)$. It can be observed that $P_{0}(t=0)>P_{1}(t=0)$ and $P_{0}(t=0.1)<P_{1}(t=0.1)$.
However, the discussion presented above does not deny the existence of a long term optimal transport process. And from an application perspective, it is worth finding out a process with minimal uncertainty.

### E.2 Definition of probability flow on universal discrete process.

In contrast to continuous processes, which necessitate the stochastic term to be a Brownian motion, there are few assumptions regarding the discrete stochastic term. As a result, the consideration of the drift term becomes unnecessary as it can be assimilated into the stochastic term. However, it is worth exploring the potential distinctive properties that may arise from treating these two terms separately.

### E.3 Practical applications.

Analogously to the effect of continuous probability flow on the continuous diffusion model, we believe that reducing uncertainty can also bring many benefits to discrete diffusion models. For instance, by selecting appropriate initial data, we can generate results that are pertinent to the initial data to attain controllable generation. Additionally, due to the excellent property of sampling certainty reduction, we can perform operations such as interpolation in latent code to complete data editing.

### E.4 Infinite horizon case in Proposition 2.

The infinite horizon case is not addressed in our study due to the presence of singularities that pose significant challenges. For instance, in the case of Brownian motion, the distribution at $t=\infty$ assumes a uniform distribution over the entire ${\mathbb{R}^{n}}$, which is not well-defined.

Additionally, the probability flow of Brownian motion at $t=0$ also experiences a singularity. By taking the limit of the right-hand side of Eq. [20](#S3.E20) and let $d_{i}=x_{t}-x_{i}$, we obtain:

$$ $\displaystyle\lim_{t\to 0^{+}}-\frac{1}{2}\nabla_{x_{t}}p_{B}(x_{t},t)=$ $\displaystyle\lim_{t\to 0^{+}}\frac{\sum_{i}exp(-\frac{d_{i}^{2}}{2t})\frac{d_{i}}{2t}}{\sum_{j}exp(-\frac{d_{j}^{2}}{2t})}$ (90) $\displaystyle=$ $\displaystyle\lim_{z\to+\infty}\sum_{i}\frac{d_{i}}{\sum_{j}exp((d_{i}^{2}-d_{j}^{2})z)/z}.$ $$

Since

$$ $\lim_{z\to+\infty}exp((d_{i}^{2}-d_{j}^{2})z)/z=\begin{cases}+\infty~{}~{}~{}~{}\text{if}~{}~{}d_{i}^{2}>d_{j}^{2},\\ 0~{}~{}~{}~{}~{}~{}~{}~{}~{}\text{if}~{}~{}d_{i}^{2}\leq d_{j}^{2},\end{cases}$ (91) $$

we have

$$ $\lim_{t\to 0}-\frac{1}{2}\nabla_{x_{t}}p_{B}(x_{t},t)=\lim_{z\to+\infty}\frac{d_{i_{min}}}{\sum_{j}exp((d_{i_{min}}^{2}-d_{j}^{2})z)/z},$ (92) $$

where $i_{min}=\mathop{\arg\min}\limits_{i}d_{i}^{2}$. (noting that $i_{min}$ may not be unique, but we exclude this scenario as it does not significantly affect our analysis). Consequently, we obtain:

$$ $\lim_{t\to 0^{+}}-\frac{1}{2}\nabla_{x_{t}}p_{B}(x_{t},t)=\begin{cases}0,&\text{if}~{}~{}x_{t=0}=x_{i_{min}},\\ d_{i_{min}}*\infty,&\text{else}.\end{cases}$ (93) $$

where $d_{i_{min}}*\infty$ indicates that the vector is oriented in the direction of $d_{i_{min}}$ and has an infinite norm. Consequently, the right-hand side of Eq. [20](#S3.E20) lacks Lipschitz continuity, leading to non-unique solutions. Actually, if Eq. [20](#S3.E20) has a unique solution near $t=0$, the distribution $p_{B}(x,t)$ will always be a summation of Dirac deltas, which contradicts Eq. [18](#S3.E18). Due to our reliance on the solution of ODE, we are unable to analyze the behavior in the vicinity of $t=0$. Consequently, we have limited our study to finite intervals.