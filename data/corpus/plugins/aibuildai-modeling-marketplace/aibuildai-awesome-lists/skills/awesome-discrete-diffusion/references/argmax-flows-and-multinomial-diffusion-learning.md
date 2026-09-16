---
arxiv_id: "2102.05379"
title: "Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions"
year: 2021
source: arxiv2md
---

## Abstract

Abstract Generative flows and diffusion models have been predominantly trained on ordinal data, for example natural images. This paper introduces two extensions of flows and diffusion for categorical data such as language or image segmentation: Argmax Flows and Multinomial Diffusion . Argmax Flows are defined by a composition of a continuous distribution (such as a normalizing flow), and an argmax function.
To optimize this model, we learn a probabilistic inverse for the argmax that lifts the categorical data to a continuous space.
Multinomial Diffusion gradually adds categorical noise in a diffusion process, for which the generative denoising process is learned. We demonstrate that our method outperforms existing dequantization approaches on text modelling and modelling on image segmentation maps in log-likelihood.

## 1 Introduction

Figure: (a) Argmax Flow: Composition of a flow $p({\bm{v}})$ and argmax transformation which gives the model $P({\bm{x}})$. The flow maps from a base distribution $p({\bm{z}})$ using a bijection $g$.
Refer to caption: /html/2102.05379/assets/x1.png

Many sources of high-dimensional data are categorical, for example language and image segmentation. Although natural images have been studied to a large extent with generative flows and diffusion models, categorical data has not had the same extensive treatment. Currently they are primarily modelled by autoregressive models, which are expensive to sample from .

Normalizing flows are attractive because they can be designed to be fast both in the evaluation and sampling direction. Typically, normalizing flows model continuous distributions. As a result, directly optimizing a flow on discrete data may lead to arbitrarily high likelihoods. In literature this problem is resolved for ordinal data by adding noise in a unit interval around the discrete value . However, because these methods have been designed for ordinal data, they do not work well on categorical data.

Other attractive generative models are diffusion models , which are fast to train due to an objective that decomposes over time steps . Diffusion models typically have a fixed diffusion process that gradually adds noise. This process is complemented by a learnable generative process that denoises the signal. have shown that diffusion models can also be designed for fast sampling. Thus far, diffusion models have been primarily trained to learn ordinal data distributions, such as natural images.

**Table 1: Surjective flow layers for applying continuous flow models to discrete data. The layers are deterministic in the generative direction, but stochastic in the inference direction. Rounding corresponds to the commonly-used dequantization for ordinal data.**
| Layer | Generation | Inference | Applications |
| --- | --- | --- | --- |
| Rounding | ${\bm{x}}=\lfloor{\bm{v}}\rfloor$ | ${\bm{v}}\sim q({\bm{v}}|{\bm{x}})$ with support | Ordinal Data |
| $\mathcal{S}({\bm{x}})=\{{\bm{v}}|{\bm{x}}=\lfloor{\bm{v}}\rfloor\}$ | e.g. images, audio |  |  |
| Argmax | ${\bm{x}}=\operatorname*{arg\,max}{\bm{v}}$ | ${\bm{v}}\sim q({\bm{v}}|{\bm{x}})$ with support | Categorical Data |
| $\mathcal{S}({\bm{x}})=\{{\bm{v}}|{\bm{x}}=\operatorname*{arg\,max}{\bm{v}}\}$ | e.g. text, segmentation |  |  |

Therefore, in this paper we introduce extensions of flows and diffusion models for categorical variables (depicted in Figure [1](#S1.F1)): i) Argmax Flows bridge the gap between categorical data and continuous normalizing flows using an argmax transformation and a corresponding family of probabilistic inverses for the argmax. In addition ii) we introduce Multinomial Diffusion, which is a diffusion model directly defined on categorical variables. Opposed to normalizing flows, defining diffusion for discrete variables directly does not require gradient approximations, because the diffusion trajectory is fixed. As a result of our work, generative normalizing flows and diffusion models can directly learn categorical data.

## 2 Background

#### Normalizing Flows

Given ${\mathcal{V}}=\mathbb{R}^{d}$ and ${\mathcal{Z}}=\mathbb{R}^{d}$ with densities $p_{V}$ and $p_{Z}$ respectively, normalizing flows learn a bijective and differentiable transformation $g:{\mathcal{Z}}\to{\mathcal{V}}$ such that the change-of-variables formula gives the density at any point ${\bm{v}}\in{\mathcal{V}}$:

$$ $p_{V}({\bm{v}})=p_{Z}({\bm{z}})\cdot\left|\det\frac{\mathrm{d}{\bm{z}}}{\mathrm{d}{\bm{v}}}\right|,\qquad{\bm{v}}=g({\bm{z}}),$ (1) $$

where $p_{Z}$ can be any density (usually chosen as a standard Gaussian). Thus, normalizing flows provide a powerful framework to learn *exact* density functions. However, [Equation 1](#S2.E1) is restricted to continuous densities.

To learn densities on ordinal discrete data (such as natural images), typically dequantization noise is added . reinterpreted dequantization as a surjective flow layer ${\bm{v}}\mapsto{\bm{x}}$ that is deterministic in one direction (${\bm{x}}=\mathsf{round}({\bm{v}})$) and stochastic in the other (${\bm{v}}={\bm{x}}+{\bm{u}}$ where ${\bm{u}}\sim q({\bm{u}}|{\bm{x}})$). Using this interpretation, dequantization can be seen as a probabilistic right-inverse for the rounding operation in the latent variable model given by:

$$ $\small{P({\bm{x}})=\!\int\!P({\bm{x}}|{\bm{v}})p({\bm{v}})\mathop{}\!\mathrm{d}{\bm{v}},\quad P({\bm{x}}|{\bm{v}})\!=\!\delta\big{(}{\bm{x}}\!=\!\mathsf{round}({\bm{v}})\big{)}},$ $$

where $\mathsf{round}$ is applied elementwise. In this case, the density model $p({\bm{v}})$ is modeled using a normalizing flow. Learning proceeds by introducing the variational distribution $q({\bm{v}}|{\bm{x}})$ that models the probabilistic right-inverse for the rounding surjection and optimizing the evidence lower bound (ELBO):

$$ $\displaystyle\log P({\bm{x}})$ $\displaystyle\geq\mathbb{E}_{{\bm{v}}\sim q({\bm{v}}|{\bm{x}})}\left[\log P({\bm{x}}|{\bm{v}})+\log p({\bm{v}})-\log q({\bm{v}}|{\bm{x}})\right]=\mathbb{E}_{{\bm{v}}\sim q({\bm{v}}|{\bm{x}})}\left[\log p({\bm{v}})-\log q({\bm{v}}|{\bm{x}})\right].$ (2) $$

The last equality holds under the constraint that the support of $q({\bm{v}}|{\bm{x}})$ is enforced to be only over the region $\mathcal{S}=\{{\bm{v}}\in\mathds{R}^{d}:{\bm{x}}=\mathsf{round}({\bm{v}})\}$ which ensures that $P({\bm{x}}|{\bm{v}})=1$.

#### Diffusion Models

Given data ${\bm{x}}_{0}$, a diffusion model consists of predefined variational distributions $q({\bm{x}}_{t}|{\bm{x}}_{t-1})$ that gradually add noise over time steps $t\in\{1,\ldots,T\}$. The diffusion trajectory is defined such that $q({\bm{x}}_{t}|{\bm{x}}_{t-1})$ adds a small amount of noise around ${\bm{x}}_{t-1}$. This way, information is gradually destroyed such that at the final time step, ${\bm{x}}_{T}$ carries almost no information about ${\bm{x}}_{0}$. Their generative counterparts consists of learnable distributions $p({\bm{x}}_{t-1}|{\bm{x}}_{t})$ that learn to denoise the data.
When the diffusion process adds sufficiently small amounts of noise, it suffices to define the denoising trajectory using distributions that are factorized (without correlation) over the dimension axis. The distribution $p({\bm{x}}_{T})$ is chosen to be similar to the distribution that the diffusion trajectory approaches. Diffusion models can be optimized using variational inference:

$$ $\log P({\bm{x}}_{0})\geq\mathbb{E}_{x_{1},\ldots x_{T}\sim q}\Big{[}\log p({\bm{x}}_{T})+\sum_{t=1}^{T}\log\frac{p({\bm{x}}_{t-1}|{\bm{x}}_{t})}{q({\bm{x}}_{t}|{\bm{x}}_{t-1})}\Big{]}.$ $$

An important insight in diffusion is that by conditioning on ${\bm{x}}_{0}$, the posterior probability $q({\bm{x}}_{t-1}|{\bm{x}}_{t},{\bm{x}}_{0})=q({\bm{x}}_{t}|{\bm{x}}_{t-1})q({\bm{x}}_{t-1}|{\bm{x}}_{0})/q({\bm{x}}_{t}|{\bm{x}}_{0})$ is tractable and straightforward to compute, permitting a reformulation in terms of KL divergences that has lower variance . Note that $\mathrm{KL}\big{(}q({\bm{x}}_{T}|{\bm{x}}_{0})|p({\bm{x}}_{T})\big{)}\approx 0$ if the diffusion trajectory $q$ is defined well:

$$ $\displaystyle\begin{split}&\log P({\bm{x}}_{0})\geq\mathbb{E}_{q}\Big{[}\log p({\bm{x}}_{0}|{\bm{x}}_{1})-\mathrm{KL}\big{(}q({\bm{x}}_{T}|{\bm{x}}_{0})|p({\bm{x}}_{T})\big{)}-\sum_{t=2}^{T}\mathrm{KL}\big{(}q({\bm{x}}_{t-1}|{\bm{x}}_{t},{\bm{x}}_{0})|p({\bm{x}}_{t-1}|{\bm{x}}_{t})\big{)}\Big{]}\end{split}$ (3) $$

## 3 Argmax Flows

Argmax flows define discrete distributions using 1) a density model $p({\bm{v}})$, such as a normalizing flow, and 2) an argmax layer that maps the continuous ${\bm{v}}\in\mathbb{R}^{D\times K}$ to a discrete ${\bm{x}}\in\{1,2,...,K\}^{D}$ using

$$ $\small{\bm{x}}=\operatorname*{arg\,max}{\bm{v}}\quad\text{ where }\quad x_{d}=\operatorname*{arg\,max}_{k}v_{dk}.$ (4) $$

This is a natural choice to model categorical variables, because it divides the entire continuous space of ${\bm{v}}$ into symmetric partitions corresponding to categories in ${\bm{x}}$. To sample from an argmax flow sample ${\bm{v}}\sim p({\bm{v}})$ and compute ${\bm{x}}=\operatorname*{arg\,max}{\bm{v}}$ (Algorithm [1](#alg1)). To generate reasonable samples, it is up to the density model $p({\bm{v}})$ to capture any complicated dependencies between the different dimensions. While sampling from an argmax flow is straightforward, the main difficulty lies in optimizing this generative model. To compute the likelihood of a datapoint ${\bm{x}}$, we have to compute

$$ $\small P({\bm{x}})=\!\int P({\bm{x}}|{\bm{v}})p({\bm{v}})d{\bm{v}},~{}~{}P({\bm{x}}|{\bm{v}})\!=\!\delta\big{(}{\bm{x}}\!=\!\operatorname*{arg\,max}({\bm{v}})\big{)},$ (5) $$

which is intractable. Consequently, we resort to variational inference and specify a variational distribution $q({\bm{v}}|{\bm{x}})$. We note that naïvely choosing any variational distribution may lead to samples ${\bm{v}}\sim q({\bm{v}}|{\bm{x}})$ where $\delta({\bm{x}}=\operatorname*{arg\,max}{\bm{v}})=0$, which yields an ELBO of negative infinity. To avoid this, we need a variational distribution $q({\bm{v}}|{\bm{x}})$ that satisfies what we term the *argmax constraint*:

$$ $\small{\bm{x}}=\operatorname*{arg\,max}{\bm{v}}\quad\text{for all}\quad{\bm{v}}\sim q({\bm{v}}|{\bm{x}}).$ $$

That is, the variational distribution $q({\bm{v}}|{\bm{x}})$ should have support limited to $\mathcal{S}({\bm{x}})=\{{\bm{v}}\in\mathbb{R}^{D\times K}:{\bm{x}}=\operatorname*{arg\,max}{\bm{v}}\}$. Recall that under this condition, the ELBO simplifies to $\mathbb{E}_{{\bm{v}}\sim q({\bm{v}}|{\bm{x}})}\left[\log p({\bm{v}})-\log q({\bm{v}}|{\bm{x}})\right]$, as shown in Algorithm [2](#alg2). For an illustration of the method see Figure [1(a)](#S1.F1.sf1).

Table: Algorithm 1 Sampling from Argmax Flows

### 3.1 Probabilistic Inverse

The argmax layer may be viewed as a surjective flow layer . With this view, the variational distribution $q({\bm{v}}|{\bm{x}})$ specifies a distribution over the possible right-inverses of the argmax function, also known as a *stochastic inverse* or *probabilistic inverse*.
Recall that the commonly-used dequantization layer for ordinal data corresponds to the probabilistic inverse of a rounding operation. As summarized in Table [1](#S1.T1), this layer may thus be viewed as analogous to the argmax layer, where the round is for ordinal data while the argmax is for categorical data.

We are free to specify any variational distribution $q({\bm{v}}|{\bm{x}})$ that satisfies the argmax constraint. In the next paragraphs we outline three possible approaches. Since operations are performed independently across dimensions, we omit the dimension axis and let ${\bm{v}}\in\mathbb{R}^{K}$ and $x\in\{1,\ldots,K\}$.

#### Thresholding (Alg. 3 ).

A straightforward method to construct a distribution $q({\bm{v}}|x)$ satisfying the argmax constraint is to use thresholding. That is, we first sample an unbounded variable ${\bm{u}}\in\mathbb{R}^{K}$ from $q({\bm{u}}|x)$, which can be for example a conditional Gaussian or normalizing flow. Next, we map ${\bm{u}}$ to ${\bm{v}}$ such that element $x$ is the largest:

$$ $v_{x}=u_{x}\quad\text{and}\quad{\bm{v}}_{-x}=\mathrm{threshold}_{T}({\bm{u}}_{-x})$ (6) $$

where the thresholding is applied elementwise with threshold value $T=v_{x}$. This ensures that element $v_{x}$ is the largest, and consequently that $q({\bm{v}}|x)$ satisfies the argmax constraint.
Note that we require the threshold function to be bijective, $\mathrm{threshold}_{T}:\mathbb{R}\rightarrow(-\infty,T)$, so that we can use the change-of-variables formula to compute $\log q({\bm{v}}|x)$.
In our implementation, thresholding is implemented using a softplus such that all values are mapped below a limit $T$:

$$ $\small v=\mathrm{threshold}_{T}(u)=T-\mathrm{softplus}(T-u),$ (7) $$

where $\mathrm{softplus}(z)=\log(1+e^{z})$ and for which it is guaranteed that $v\in(-\infty,T)$.

Table: Algorithm 3 Thresholding-based $q({\bm{v}}|{\bm{x}})$

#### Gumbel (Alg. 4 ).

An alternative approach is to let $q({\bm{v}}|x)=\mathrm{Gumbel}({\bm{v}}|\bm{\phi})$ restricted to $\operatorname*{arg\,max}{\bm{v}}=x$, where the location parameters $\bm{\phi}\leftarrow\mathrm{NN}(x)$ are predicted using a neural network $\mathrm{NN}$.
The Gumbel distribution has favourable properties: The $\operatorname*{arg\,max}$ and $\max$ are independent and the $\max$ is also distributed as a Gumbel:

$$ $\small\max_{i}v_{i}\sim\mathrm{Gumbel}(\phi_{\max}),$ (8) $$

where $\phi_{\max}=\log\sum_{i}\exp\phi_{i}$.
For a more extensive introduction see . To sample ${\bm{v}}\sim q({\bm{v}}|x)$, we thus first sample the maximum $v_{x}$ according to Eq. [8](#S3.E8). Next, given the sample $v_{x}$, the remaining values can be sampled using truncated Gumbel distributions:

$$ $\small v_{i}\sim\mathrm{TruncGumbel}(\phi_{i};T)\text{ where }i\not=x$ (9) $$

where the truncation value $T$ is given by $v_{x}$ which ensures that the argmax constraint $v_{x}>v_{i}$ for $i\not=x$ is satisfied. Recall that to optimize Eq. [2](#S2.E2), $\log q({\bm{v}}|{\bm{x}})$ is also required, which can be computed using the closed-form expressions for the log density functions (see Table [5](#A2.T5)). Another property of Gumbel distributions is that

$$ $\small P(\operatorname*{arg\,max}{\bm{v}}=i)=\exp\phi_{i}/\sum_{i}\exp\phi_{i},$ (10) $$

which we use to initialize the location parameters $\bm{\phi}$ to match the empirical distribution of the first minibatch of the data.

#### Gumbel Thresholding.

This method unifies the methods from the previous two sections: Gumbel distributions and thresholding. The key insight is that the Gumbel sampling procedures as defined above can be seen as a reparametrization of a uniform noise distribution $\mathcal{U}(0,1)^{K}$ which is put through the inverse CDF of the Gumbel distributions (see Table [5](#A2.T5)). From the perspective of change-of-variables, the log likelihood denotes the log volume change of this transformation. To increase expressitivity the uniform distribution can be replaced by a normalizing flow $q({\bm{u}}|x)$ that has support on the interval $(0,1)^{K}$, which can be enforced using a sigmoid transformation. This section shows that a large collection of thresholding functions can be found by studying (truncated) inverse CDFs. In practice we find that performance is reasonably similar as long as the underlying noise ${\bm{u}}$ is learned.

#### Behavior of the Variational Posterior

Although several methods to learn $q$ have been proposed, it is unclear what expressitivity is required. In the following, the interactions between $q({\bm{v}}|{\bm{x}})$ and the density model $p({\bm{v}})$ are discussed. Recall that the variational bound that is optimized under expectation of a data distribution $\mathcal{D}$ can be seen as minimizing the $\operatorname{KL}$ distance between the aggregated posterior $q({\bm{v}})=\mathbb{E}_{{\bm{x}}\sim\mathcal{D}}q({\bm{v}}|{\bm{x}})$ and the density model $p({\bm{v}})$, so $\operatorname{KL}(q({\bm{v}})|p({\bm{v}}))$. There are two distinct reasons which can cause this distance to be large: Firstly, the density model $p({\bm{v}})$ may not have the right probability mass in each argmax region. These desired probabilities solely depend on the data distribution $\mathcal{D}$. Secondly, the variational posterior $q({\bm{v}}|{\bm{x}})$ may not have the correct shape compared to $p({\bm{v}})$, within an argmax region. At initialization, the thresholding within $q$ can create low density regions at argmax boundaries.

In theory, if $p({\bm{v}})$ is a universal density approximator, then the model can be fitted for any well-behaved $q({\bm{v}}|{\bm{x}})$. Then $p({\bm{v}})$ can even fit the low density regions in the boundaries. This argument is trivial, as one can simply set $p({\bm{v}})$ to $q({\bm{v}})=\mathbb{E}_{{\bm{x}}\sim\mathcal{D}}q({\bm{v}}|{\bm{x}})$. In practice, over training steps we find that $q$ does smooth out these boundary artifacts, and counteracts the thresholding so that the aggregated posterior becomes smoother.

### 3.2 Cartesian Products of Argmax Flows

In the current description, Argmax Flows require the same number of dimensions in ${\bm{v}}$ as there are classes in ${\bm{x}}$. To alleviate this constraint we introduce Cartesian products of Argmax Flows. To illustrate our method, consider a 256 class problem. One class can be represented using a single number in $\{1,\ldots,256\}$, but also using two hexadecimal numbers $\{1,\ldots,16\}^{2}$ or alternatively using eight binary numbers. Specifically, any base $K$ variable ${\bm{x}}^{(K)}\in\{1,\ldots,K\}^{D}$ can be converted to a base $M$ variable ${\bm{x}}^{(M)}\in\{1,\ldots,M\}^{d_{m}\times D}$ where $d_{m}=\lceil\log_{M}K\rceil$.
Then the variable ${\bm{x}}^{(M)}$ with dimensionality $M\cdot d_{m}\cdot D$ represents the variable ${\bm{x}}^{(K)}$ with dimensionality $K\cdot D$, trading off symmetry for dimensionality. Even though this may lead to some unused additional classes, the ELBO objective in Equation [2](#S2.E2) can still be optimized using an $M$-categorical Argmax Flow. Finally, note that Cartesian products of binary spaces are a special case where the variable can be encoded symmetrically into a single dimension to the positive and negative part using binary dequantization . In this case, by trading-off symmetry the dimensionality increases only proportional to $\log_{2}K$ .

## 4 Multinomial Diffusion

In this section we introduce an alternative likelihood-based model for categorical data: Multinomial Diffusion. In contrast with previous sections, ${\bm{x}}_{t}$ will be represented in one-hot encoded format ${\bm{x}}_{t}\in\{0,1\}^{K}$. Specifically, for category $k$, $x_{k}=1$ and $x_{j}=0$ for $j\not=k$. Note that again the dimension axis is omitted for clarity as all distributions are independent over the dimension axis. We define the multinomial diffusion process using a categorical distribution that has a $\beta_{t}$ chance of resampling a category uniformly:

$$ $q({\bm{x}}_{t}|{\bm{x}}_{t-1})=\mathcal{C}({\bm{x}}_{t}|(1-\beta_{t}){\bm{x}}_{t-1}+\beta_{t}/K),$ (11) $$

where $\mathcal{C}$ denotes a categorical distribution with probability parameters after $|$. Further addition (and subtraction) between scalars and vectors is done elementwise. This convention kept throughout this section.
Since these distributions form a Markov chain, we can express the probability of any ${\bm{x}}_{t}$ given ${\bm{x}}_{0}$ as:

$$ $q({\bm{x}}_{t}|{\bm{x}}_{0})=\mathcal{C}({\bm{x}}_{t}|\bar{\alpha}_{t}{\bm{x}}_{0}+(1-\bar{\alpha}_{t})/K)$ (12) $$

where $\alpha_{t}=1-\beta_{t}$ and $\bar{\alpha}_{t}=\prod_{\tau=1}^{t}\alpha_{\tau}$. Intuïtively, for each next timestep, a little amount of uniform noise $\beta_{t}$ over the $K$ classes is introduced, and with a large probability $(1-\beta_{t})$ the previous value ${\bm{x}}_{t-1}$ is sampled. Using Equation [11](#S4.E11) and [12](#S4.E12) the categorical posterior $q({\bm{x}}_{t-1}|{\bm{x}}_{t},{\bm{x}}_{0})$ can be computed in closed-form:

$$ $\displaystyle\begin{split}q({\bm{x}}_{t-1}|{\bm{x}}_{t},{\bm{x}}_{0})&=\mathcal{C}({\bm{x}}_{t-1}|{\bm{\theta}}_{\mathrm{post}}({\bm{x}}_{t},{\bm{x}}_{0})),\,\,\text{ where }\,\,{\bm{\theta}}_{\mathrm{post}}({\bm{x}}_{t},{\bm{x}}_{0})=\tilde{{\bm{\theta}}}/\sum_{k=1}^{K}\tilde{\theta}_{k}\\ \text{and }\,\tilde{{\bm{\theta}}}&=[\alpha_{t}{\bm{x}}_{t}+(1-\alpha_{t})/K]\odot[\bar{\alpha}_{t-1}\bm{x}_{0}+(1-\bar{\alpha}_{t-1})/K].\end{split}$ (13) $$

Figure: Figure 2: Overview of multinomial diffusion. A generative model $p({\bm{x}}_{t-1}|{\bm{x}}_{t})$ learns to gradually denoise a signal from left to right. An inference diffusion process $q({\bm{x}}_{t}|{\bm{x}}_{t-1})$ gradually adds noise form right to left.
Refer to caption: /html/2102.05379/assets/x3.png

One of the innovations in was the insight to not predict the parameters for the generative trajectory directly, but rather to predict the noise using the posterior equation for $q$. Although predicting the noise is difficult for discrete data, we predict a probability vector for $\hat{{\bm{x}}}_{0}$ from ${\bm{x}}_{t}$ and subsequently parametrize $p({\bm{x}}_{t-1}|{\bm{x}}_{t})$ using the probability vector from $q({\bm{x}}_{t-1}|{\bm{x}}_{t},\hat{{\bm{x}}}_{0})$, where ${\bm{x}}_{0}$ is approximated using a neural network $\hat{{\bm{x}}}_{0}=\mu({\bm{x}}_{t},t)$. Equation [13](#S4.E13) will produce valid probability vectors that are non-negative and sums to one under the condition that the prediction $\hat{{\bm{x}}}_{0}$ is non-negative and sums to one, which is ensured with a softmax function in $\mu$. To summarize:

$$ $\displaystyle\begin{split}p({\bm{x}}_{0}|{\bm{x}}_{1})=\mathcal{C}({\bm{x}}_{0}|\hat{{\bm{x}}}_{0})\,\text{ and }\,p({\bm{x}}_{t-1}|{\bm{x}}_{t})=\mathcal{C}({\bm{x}}_{t-1}|\bm{\theta}_{\mathrm{post}}({\bm{x}}_{t},\hat{{\bm{x}}}_{0}))\,\text{ where }\,\hat{{\bm{x}}}_{0}=\mu({\bm{x}}_{t},t)\end{split}$ (14) $$

The KL terms in Equation [3](#S2.E3) can be simply computed by enumerating the probabilities in Equation [13](#S4.E13) and [14](#S4.E14) and computing the KL divergence for discrete distributions in $L_{t-1}$ with $t\geq 2$:

$$ $\displaystyle\begin{split}\mathrm{KL}\big{(}q({\bm{x}}_{t-1}|{\bm{x}}_{t},{\bm{x}}_{0})|p({\bm{x}}_{t-1}|{\bm{x}}_{t})\big{)}&=\mathrm{KL}\big{(}\mathcal{C}(\bm{\theta}_{\mathrm{post}}({\bm{x}}_{t},{\bm{x}}_{0}))|\mathcal{C}(\bm{\theta}_{\mathrm{post}}({\bm{x}}_{t},\hat{{\bm{x}}}_{0}))\big{)},\end{split}$ (15) $$

which can be computed using $\sum_{k}\bm{\theta}_{\mathrm{post}}({\bm{x}}_{t},{\bm{x}}_{0}))_{k}\cdot\log\frac{\bm{\theta}_{\mathrm{post}}({\bm{x}}_{t},{\bm{x}}_{0}))_{k}}{\bm{\theta}_{\mathrm{post}}({\bm{x}}_{t},\hat{{\bm{x}}}_{0}))_{k}}$. Furtermore, to compute $\log p({\bm{x}}_{0}|{\bm{x}}_{1})$ use that ${\bm{x}}_{0}$ is onehot:

$$ $\log p({\bm{x}}_{0}|{\bm{x}}_{1})=\sum_{k}{\bm{x}}_{0,k}\log\hat{{\bm{x}}}_{0,k}$ (16) $$

## 5 Related Work

Deep generative models broadly fall into the categories autoregressive models ARMs , Variational Autoencoders (VAEs) , Adversarial Network (GANs) , Normalizing Flows , Energy-Based Models (EBMs) and Diffusion Models .

Normalizing Flows typically learn a continuous distribution and dequantization is required to train these methods on ordinal data such as images. A large body of work is dedicated to building more expressive continuous normalizing flows .
To learn ordinal discrete distributions with normalizing flows, adding uniform noise in-between ordinal classes was proposed in and later theoretically justified in . An extension for more powerful dequantization based on variational inference was proposed in , and connected to autoregressive models in . Dequantization for binary variables was proposed in . propose invertible transformations for categorical variables directly. However, these methods can be difficult to train because of gradient bias and results on images have thus far not been demonstrated. In addition flows for ordinal discrete data (integers) have been explored in . In other works, VAEs have been adapted to learn a normalizing flow for the latent space . However, these approaches typically still utilize an argmax heuristic to sample, even though this is not the distribution specified during training.

Diffusion models were first introduced in , who developed diffusion for Gaussian and Bernoulli distributions. Recently, Denoising Diffusion models have been shown capable of generating high-dimensional images by architectural improvements and reparametrization of the predictions. Diffusion models are relatively fast to train, but slow to sample from as they require iterations over the many timesteps in the chain. showed that in practice samples can be generated using significantly fewer steps. demonstrated that importance-weighting the objective components greatly improves log-likelihood performance. In a continuous-time extension of denoising diffusion models was proposed. After initial release of this paper we discovered that concurrently also describe a framework for discrete diffusion, but without empirical evaluation.

## 6 Experiments

In our experiments we compare the performance of our methods on language modelling tasks and learning image segmentation maps unconditionally.

**Table 2: Comparison of a coupling and autoregressive generative flows with uniform and variational dequantization and our proposed Argmax flows.**
| Dequantization | Flow type | text8 (bpc) | enwik8 (bits per raw byte) |
| --- | --- | --- | --- |
| Uniform dequantization | Autoregressive | 1.90 | 2.14 |
| Variational dequantization | 1.43 | 1.44 |  |
| Argmax Flow (ours) | 1.38 | 1.42 |  |
| Uniform dequantization | Coupling | 2.01 | 2.33 |
| Variational dequantization | 2.08 | 2.28 |  |
| Argmax Flow (ours) | 1.82 | 1.93 |  |

**Table 3: Comparison of different methods on text8 and enwik8. Results are reported in negative log-likelihood with units bits per character (bpc) for text8 and bits per raw byte (bpb) for enwik8.**
| Model type |  | Model | text8 (bpc) | enwik8 (bpb) |
| --- | --- | --- | --- | --- |
| ARM |  | 64 Layer Transformer | 1.13 | 1.06 |
|  | TransformerXL | 1.08 | 0.99 |  |
| VAE |  | AF/AF^⋆ (AR) | 1.62 | 1.72 |
|  | IAF / SCF^⋆ | 1.88 | 2.03 |  |
|  | CategoricalNF (AR) | 1.45 | - |  |
| Generative Flow |  | Argmax Flow, AR (ours) | 1.39 | 1.42 |
|  | Argmax Coupling Flow (ours) | 1.82 | 1.93 |  |
| Diffusion |  | Multinomial Text Diffusion (ours) | 1.72 | 1.75 |

### 6.1 Language data

In this section we compare our methods on two language datasets, text8 and enwik8. text8 contains 27 categories (‘a’ through ‘z’ and ‘ ’) and for enwik8 the bytes are directly modelled which results in 256 categories.

#### Model description

Two versions of generative argmax flows are tested: using an autoregressive (AR) flow and a coupling-based flow for $p({\bm{v}})$. In these experiments the probabilistic inverse is based on the thresholding approach. Specifically, a conditional diagonal Gaussian $q({\bm{u}}|{\bm{x}})$ is trained and thresholded which gives the distribution $q({\bm{v}}|{\bm{x}})$.
The argmax flow is defined on binary Cartesian products. This means that for $K=27$, a $5$-dimensional binary space is used and for $K=256$ an $8$-dimensional binary space. The argmax flow is compared to the current standard of training generative flows directly on discrete data: dequantization. We compare to both uniform and variational dequantization, where noise on a (0, 1) interval is added to the onehot representation of the categorical data. The autoregressive density model is based on the model proposed in . The coupling density model consists of 8 flow layers where each layer consists of a $1$ $\times$ $1$ convolution and mixture of logistics transformations . In the multinomial text diffusion model, the $\mu$ network is modeled by a 12-layer Transformer. For more extensive details about the experiment setup see Appendix [B](#A2).

#### Comparison with Generative Flows

Firstly we compare the performance of generative flows directly trained on language data (Table [2](#S6.T2)). These experiments are using the same underlying normalizing flow: either a coupling-based flow or an autoregressive flow. Note that Argmax Flows consistently outperform both uniform and variational dequantization. This indicates that it is easier for a generative flow to learn the lifted continuous distribution using an argmax flow. An advantage of Argmax flows that may explain this difference is that they lift the variables into the entire Euclidean space, whereas traditional dequantization only introduce probability density on $(0,1)$ intervals, leaving gaps with no probability density. The performance improvements of Argmax flows are even more pronounced when comparing coupling-based approaches. Also note that coupling flows have worse performance than autoregressive flows, with a difference that is generally smaller for images. This indicates that designing more expressive coupling layers for text is an interesting future research direction.

#### Comparison with other generative models

The performance compared to models in literature is presented in Table [3](#S6.T3) alongside the performance of our Argmax Flows and Multinomial Diffusion. The latent variable approaches containing autoregressive components are marked using (AR). Although autoregressive flows still have the same disadvantages as ARMs, they provide perspective on where performance deficiencies are coming from. We find that our autoregressive Argmax Flows achieve better performance than the VAE approaches, they outperform AF/AF and CategoricalNF .

Figure: (a) Samples from Multinomial Text Diffusion.
Refer to caption: /html/2102.05379/assets/x4.png

When comparing non-autoregressive models, Argmax Flows also outperforms the method that lifts the categorical space to a continuous space: IAF / SCF . Interestingly, the multinomial text diffusion is a non-autoregressive model that performs even better than the argmax coupling flow, but performs worse than the autoregressive version. For this model it is possible that different diffusion trajectories for $q$ would result in even better performance, because in the current form the denoising model has to be very robust to input noise. These experiments also highlight that there is still a distinct performance gap between standard ARMs and (autoregressive) continuous density model on text, possibly related to the dequantization gap . Samples from different models trained on text8 are depicted in Figure [3](#S6.F3). Because of difficulties in reproducing results from Discrete Flows, a comparison and analysis of discrete flows are left out of this section. Instead they are extensively discussed in Appendix [C](#A3). For additional experiments regarding Cartesian products and sampling time see Appendix [D](#A4).

Figure: (a) Ground truth sequence from text8.
Refer to caption: /html/2102.05379/assets/x7.png

#### Unsupervised spell-checking

An interesting by-product of the text diffusion model is that it can be used to spell-check text using a single forward pass. To demonstrate this, a sentence taken from the test data is corrupted by changing a few characters. This corrupted sequence is given as ${\bm{x}}_{1}$ to the generative denoising model, which is close to the data at step 0 0. Then the denoising model predicts $p({\bm{x}}_{0}|{\bm{x}}_{1})$ and the most-likely ${\bm{x}}_{0}$ can be suggested. Note that this model only works for character-level corruption, not insertions. An example is depicted in Figure [5](#S6.F5). Since the model chooses the most-likely matching word, larger corruptions will at some point lead to word changes.

### 6.2 Segmentation maps

For image-type data, we introduce a categorical image dataset: the cityscapes dataset is repurposed for unconditional image segmentation learning. In contrast with the standard setting, the distribution over the segmentation targets needs to be learned without conditioning on the photograph. To reduce computational cost, we rescale the segmentation maps from cityscapes to $32\times 64$ images using nearest neighbour interpolation. We utilize the global categories as prediction targets which results in an 8-class problem.

Figure: Table 4: Performance of different dequantization methods on squares and cityscapes dataset, in bits per pixel, lower is better.

#### Model description

The Argmax Flows are defined directly on the $K=8$ categorical space. The density model $p({\bm{v}})$ is defined using affine coupling layers parametrized by DenseNets . For the probabilistic inverse we learn a conditional flow $q({\bm{u}}|{\bm{x}})$ which is also based on the affine coupling structure. Depending on the method, either softplus or Gumbel thresholding is applied to obtain ${\bm{v}}$. Recall that for our first Gumbel approach it is equivalent to set $q({\bm{u}}|{\bm{x}})$ to the unit uniform distribution, whereas $q({\bm{u}}|{\bm{x}})$ is learned for Gumbel thresholding. We compare to existing dequantization strategies in literature: uniform and variational dequantization which are applied on the onehot representation. All models utilize the same underlying flow architectures and thus the number of parameters is roughly the same. The exception are uniform dequantization and the Gumbel distribution, since no additional variational flow distribution is needed. For more extensive details see Appendix [B](#A2).

#### Comparison

The results of this experiment are shown in Table [4](#S6.T4) in terms of ELBO and if available the IWBO (importance weighted bound) with $1000$ samples measured in bits per pixel. Consistent with the language experiments, the traditional dequantization approaches (uniform / variational) are outperformed by Argmax Flows. Interestingly, although argmax flows with softplus thresholding achieves the best ELBO, the argmax flow with Gumbel thresholding approach achieves a better IWBO. The Multinomial Diffusion model performs somewhat worse with 0.37 bpp on test whereas it scored 0.33 bpp on train. Interestingly, this the only model where overfitting was an issue and data augmentation was required, which may explain this portion of the performance difference. For all other models training performance was comparable to test and validation performance. Samples from the different models trained on cityscapes are depicted in Figure [4](#S6.F4). Another interesting point is that coupling flows had difficulty producing coherent text samples (Figure [3](#S6.F3)) but do not suffer from this problem on the cityscapes data which is more image-like. As coupling layers where initially designed for images , they may require adjustments to increase their expressiveness on text.

## 7 Social Impact and Conclusion

#### Social Impact

The methods described in this paper can be used to learn categorical distributions. For that reason, they can potentially be used to generate high-dimensional categorical data, such as text or image segmentation maps, faster than iterative approaches. Possibly negative influences are the generation of fake media in the form of text, or very unhelpful automated chat bots for customer service. Our work could positively influence new methods for text generation, or improved segmentation for self-driving cars. In addition, our work may also be used for outlier detection to flag fake content. Also, we believe the method in its current form is still distant from direct applications as the ones mentioned above.

#### Conclusion

In this paper we propose two extensions for Normalizing Flows and Diffusion models to learn categorical data: Argmax Flows and Multinomial Diffusion. Our experiments show that our methods outperform comparable models in terms of negative log-likelihood. In addition, our experiments highlight distinct performance gaps in the field: Between standard ARMs, continuous autoregressive models and non-autoregressive continuous models. This indicates that future work could focus on two sources of decreased performance: 1) when discrete variables are lifted to a continuous space and further 2) when removing autoregressive components.

Funding Disclosure
There are no additional sources of funding to disclose, beyond the affiliations of the authors.

## Appendix A Numerically stable Multinomial Diffusion in log space

In this section we explain how Multinomial Diffusion models can be implemented in a numerically safe manner in log-space. Note that in addition to this appendix with pseudo-code, the actual source code will also be released. First we define a few helper functions:

Then we can initialize the variables we are planning to utilize for the multinomial diffusion model. This is done with float64 variables to limit the precision loss in the log_1_min_a computation. Since these are precomputed and later converted to float32, there is no meaningful increase in computation time.

Then we can define the functions that we utilize to compute the log probabilities of the categorical distributions of the forward process. The functions below compute the probability vectors for $q({\bm{x}}_{t}|{\bm{x}}_{t-1})$, $q({\bm{x}}_{t}|{\bm{x}}_{0})$ and $q({\bm{x}}_{t-1}|{\bm{x}}_{t},{\bm{x}}_{0})$.

Some magic is happening in q_pred_one_timestep. Recall that at some point we need to compute $\mathcal{C}({\bm{x}}_{t}|(1-\beta_{t}){\bm{x}}_{t-1}+\beta_{t}/K)$ for different values of ${\bm{x}}_{t}$, which when treated as a function outputs $(1-\beta_{t})+\beta_{t}/K$ if ${\bm{x}}_{t}={\bm{x}}_{t-1}$ and $\beta_{t}/K$ otherwise. This function is symmetric, meaning that $\mathcal{C}({\bm{x}}_{t}|(1-\beta_{t}){\bm{x}}_{t-1}+\beta_{t}/K)=\mathcal{C}({\bm{x}}_{t-1}|(1-\beta_{t}){\bm{x}}_{t}+\beta_{t}/K)$. This is why we can switch the conditioning and immediately return the different probability vectors for ${\bm{x}}_{t}$. This also corresponds to Equation [13](#S4.E13).

Then using the q_posterior function as parametrization we predict the probability vector for $p({\bm{x}}_{t-1}|{\bm{x}}_{t})$ using a neural network.

And then finally we can compute the loss term $L_{t}$ using the KL divergence for categorical distributions:

Coincidentally this code even works for $L_{0}$ because ${\bm{x}}_{0}$ is onehot and then:

$$ $-\log\mathcal{C}({\bm{x}}_{0}|\hat{{\bm{x}}}_{0})-\sum_{k}{\bm{x}}_{0,k}\log\hat{{\bm{x}}}_{0,k}=\sum_{k}{\bm{x}}_{0,k}[\underbrace{\log{\bm{x}}_{0,k}}_{0\text{ or }\log 0}-\log\hat{{\bm{x}}}_{0,k}]=\mathrm{KL}(\mathcal{C}({\bm{x}}_{0})||\mathcal{C}(\hat{{\bm{x}}}_{0})),$ $$

where in the last term ${\bm{x}}_{0}$ and $\hat{{\bm{x}}}_{0}$ are probability vectors and $0\log 0$ is defined to be 0 0.

## Appendix B Experimental details

This section gives details on experimental setup, architectures and optimization hyperparameters. In addition, the code to reproduce experiments will be released publicly.

#### Diffusion settings

For diffusion we use the cosine schedule for $\{\alpha_{t}\}$ from with the difference that what was previously $\sqrt{\bar{\alpha}_{t}}$ is now $\bar{\alpha}_{t}$, so that their factor $\sqrt{\bar{\alpha}_{t}}$ for the Gaussian mean is equal to our factor $\bar{\alpha}_{t}$ for categorical parameters. Specifically, our $\bar{\alpha}_{t}$ are defined using:

$$ $\bar{\alpha}_{t}=\frac{f(t)}{f(0)}\quad f(t)=\cos\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right),\quad s=0.008,$ $$

where $T$ is the total number of diffusion steps. show that instead of sampling $t$ uniformly, variance is reduced when $t$ is importance-sampled with $q(t)\propto\sqrt{\mathbb{E}[L_{t}^{2}]}$, which is estimated using training statistics, and we use their approach. The objective can be summarized as:

$$ $\log P({\bm{x}}_{0})\geq\mathbb{E}_{t\sim q(t),{\bm{x}}_{t}\sim q({\bm{x}}_{t}|{\bm{x}}_{0})}\left[-\frac{1}{q(t)}\mathrm{KL}\big{(}q({\bm{x}}_{t-1}|{\bm{x}}_{t},{\bm{x}}_{0})|p({\bm{x}}_{t-1}|{\bm{x}}_{t})\right].$ (17) $$

#### Gumbel properties

In Table [5](#A2.T5) a useful overview of Gumbel properties are given. These equations can be used to sample and compute the likelihood of the (truncated) Gumbel distributions. For a more extensive treatment see .

**Table 5: Summary of Gumbel properties.**
| Description | $\log p$ | Sample |
| --- | --- | --- |
| $\mathrm{Gumbel}(g|\phi)$ | $\phi-g-\exp(\phi-g)$ | $g=-\log(-\log(u))+\phi$<br>$u\sim\mathcal{U}(0,1)$ |
| $\max_{i}\mathrm{Gumbel}(g_{i}|\phi)$ | $\log\mathrm{Gumbel}(g_{\max}|\phi_{\max})$<br>$\phi_{\max}=\log\sum_{i}\exp\phi_{i}$ | $g_{\max}\sim\mathrm{Gumbel}(\phi_{\max})$<br>$\phi_{\max}=\log\sum_{i}\exp\phi_{i}$ |
| $\mathrm{TruncGumbel}(g|\phi,T)$ | $\phi-g-\exp(\phi-g)+\exp(\phi-T)$<br>if $g<T$ else $-\infty$ | $g=\phi-\log(\exp(\phi-T)-\log u)$<br>$u\sim\mathcal{U}(0,1)$ |

### B.1 Language Modelling

For the language modelling experiments we utilize the standard text8 dataset with sequence length $256$ and enwik8 dataset with sequence length $320$. The train/val/test splits are 90000000/5000000/5000000 for both text8 and enwik8, as is standard in literature. The Multinomial Text Diffusion models are trained for $300$ epochs, whereas the Argmax Flows are trained for $40$ epochs, with the exception of the Argmax Coupling Flow on enwik8 which only needs to be trained for $20$ epochs. Further details are presented in Tables [6](#A2.T6) and [7](#A2.T7). In addition, the code to reproduce results will be publicly available. There are no known ethics issues with these datasets at the time of writing.

**Table 6: Optimization details for text models.**
| Model | batch size | lr | lr decay | optimizer | dropout |
| --- | --- | --- | --- | --- | --- |
| Multinomial Text Diffusion (text8) | 32 | 0.0001 | 0.99 | Adam | 0 |
| Multinomial Text Diffusion (enwik8) | 32 | 0.0001 | 0.99 | Adam | 0 |
| Argmax AR Flow (text8) | 64 | 0.001 | 0.995 | Adam | 0.25 |
| Argmax AR Flow (enwik8) | 64 | 0.001 | 0.995 | Adam | 0.25 |
| Argmax Coupling Flow (text8) | 16 | 0.001 | 0.995 | Adamax | 0.05 |
| Argmax Coupling Flow (enwik8) | 32 | 0.001 | 0.995 | Adamax | 0.1 |

**Table 7: Architecture description for text models.**
| Model | Architecture description |
| --- | --- |
| Multinomial Text Diffusion (text8) | 12-layer transformer 8 global, 8 local heads / 1000 diffusion steps |
| Multinomial Text Diffusion (enwik8) | 12-layer transformer 8 global, 8 local heads / 4000 diffusion steps |
| Argmax AR Flow (text8) | 2-layer LSTM, 2048 hidden units |
| Argmax AR Flow (enwik8) | 2-layer LSTM, 2048 hidden units |
| Argmax Coupling Flow (text8) | 2-layer bi-directional LSTM, 512 hidden units |
| Argmax Coupling Flow (enwik8) | 2-layer bi-directional LSTM, 768 hidden units |

### B.2 Cityscapes

#### Preprocessing

The Cityscapes segmentation maps are re-sampled to a $32$ by $64$ pixel image using nearest neighbour interpolation. The original segmentation maps are downloaded from [https://www.cityscapes-dataset.com/downloads/](https://www.cityscapes-dataset.com/downloads/) where all files are contained in gtFine_trainvaltest.zip. Note that we train on a $8$-class problem since we only consider what is called the category_id field in torchvision. We re-purpose the validation set as test set, containing $500$ maps. The original train set containing $2975$ maps is split into $2500$ maps for training and $475$ maps for validation. The original test set is not utilized. To aid reproducibility we will publish source code that includes the preprocessing and the dataloaders. There are no known ethics issues with the segmentation maps at the time of writing. License is located at [https://www.cityscapes-dataset.com/license/](https://www.cityscapes-dataset.com/license/).

#### Architectures

For Cityscapes all models utilize the same architectures, although they represent a different part for their respective model designs. The density model $p({\bm{v}})$ consist of $4$ levels with $10$ subflows each, separated by squeeze layers, where each subflow consists of a $1$ $\times$ $1$ convolution and an affine coupling layer. The coupling layers are parametrized by DenseNets . The same model is used for the latent distribution in the VAE (usually referred to as $p({\bm{z}})$ in literature). The probabilistic inverse $q({\bm{v}}|{\bm{x}})$ is modelled by a single level flow that has $8$ subflows, again consisting of affine coupling layers and $1$ $\times$ $1$ convolutions. To condition on ${\bm{x}}$ it is processed by a DenseNet which outputs a representation for the coupling layers that is concatenated to the original input. The same model is utilized to parametrize the VAE encoder (commonly referred to as $q({\bm{z}}|{\bm{x}})$). The VAE additionally has a model for the decoder $p({\bm{x}}|{\bm{z}})$ which is parametrized by a DenseNet which outputs the parameters for a categorical distribution. The models are optimized using the same settings, and no hyperparameter search was performed. Specifically, the models are optimized with minibatch size $64$ for $2000$ epochs with the Adamax optimizer with learning rate $0.001$ and a linear learning rate warmup of $10$ epochs and a decay factor of $0.995$.

### B.3 Range of considered hyperparameters

For Multinomial Text Diffusion we experimented with the depth of transformers $\{1,2,4,8,12,16,20\}$ and the hidden size $\{128,256,512,1024\}$. We found that models with depth $12$ and $512$ could be trained in a reasonable amount of time while giving good performance. For the cityscapes experiments no hyperparameter search was performed.

### B.4 Details on latent normalizing flows for text8

We utilize the official code repository from in here(^1^11[https://github.com/harvardnlp/TextFlow](https://github.com/harvardnlp/TextFlow)). The original code utilizes $10$ ELBO samples, which is relatively expensive. For that reason we instead opt for $1$ ELBO sample and find it gives similar results. The batch size is increased from $16$ to $32$. Additionally we reduce the KL scheduling from $4$ initial $10^{-5}$ epochs to only $2$ initial $10^{-5}$ epoch and we anneal linearly over the next $4$ epochs instead of over the next $10$ epochs. In total the models are optimized for $30$ epochs. We verify that the resulting models still achieve similar performance on the Penn Tree Bank experiment compared to the original paper in terms of ELBO values: Our hyperparameter setup for AF/AF achieves slightly better performance with 1.46 versus 1.47 bpc and for IAF/SCF achieves slightly worse 1.78 versus 1.76 bpc.

### B.5 Computing infrastructure

Experiments where run on NVIDIA-GTX 1080Ti GPUs, CUDA 10.1 with Python version 3.7.6 in Pytorch 1.5.1 or 1.7.1.

## Appendix C Reproducing Discrete Flows

In this section we detail our efforts to reproduce the results from discrete flows . Specifically, we are interested in the discrete flows models that map to factorized distributions, for instance the discrete bipartite (coupling) flow. We avoid situations where an autoregressive base distribution is used, it may be difficult to identify how much the flow is actually learning versus the ARM as base. For this paper an official implementation was released at [https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/](https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/) in the files discrete_flows.py and utils.py. However, this codebase contains only the high-level modules and code for the toy example, it does not contain the specific code related to the language experiments. These high-level modules and the toy problem were ported to PyTorch here: [https://github.com/TrentBrick/PyTorchDiscreteFlows](https://github.com/TrentBrick/PyTorchDiscreteFlows). Using this codebase, we were able to compare on the quantized eight Gaussians toy dataset, as depicted in Figure [6](#A3.F6). In this experiment we clearly see that argmax flows outperform discrete flows both numerically (6.32 versus 7.0 nats) and visually by comparing the samples or probability mass function.

Figure: (a) Samples from Discrete Flow using a single layer, taken from .
Refer to caption: /html/2102.05379/assets/x10.png

Subsequent efforts by others to reproduce the language experiments failed (see [https://github.com/TrentBrick/PyTorchDiscreteFlows/issues/1](https://github.com/TrentBrick/PyTorchDiscreteFlows/issues/1)). In another work, also noticed the difficulty of getting discrete flows to succesfully optimize, as detailed in the set shuffling/summation experiment corresponding to Table 5 in the paper.

For this paper we also tried to reproduce the language experiments. After verifying the correctness of the one_hot_argmax, one_hot_minus and one_hot_add functions in [https://github.com/TrentBrick/PyTorchDiscreteFlows](https://github.com/TrentBrick/PyTorchDiscreteFlows), we implemented an autoregressive discrete flow layer with an expressive network, in an effort to limit the accumulated gradient bias. Recall that an autoregressive layer is more expressive than a coupling layer as it has more dependencies between dimensions. As can be seen in Table [8](#A3.T8) our re-implementation also performed considerably worse, matching the experience of the others described above.

**Table 8: Discrete Flows on text8. Note that AR is more expressive than coupling.**
| Model | text8 (bpc) |
| --- | --- |
| Discrete Flows from paper (coupling, factorized base, without scale) | 1.29 |
| Discrete Flows from paper (coupling, factorized base, with scale) | 1.23 |
| Discrete Flows reimplementation (AR, factorized base, without scale) | 4.13 |
| Argmax Flow, AR (ours) | 1.38 |
| Argmax Coupling Flow (ours) | 1.80 |

#### Final remarks

We have had extensive contact with the authors of to resolve this issue over the course of several months. Unfortunately it is not possible for them to share the code for the language flows due to internal dependencies. Also, we have not been able to find any implementation of discrete flows online that achieves the reported performance on text. The authors generously offered to look at our reimplementation, which we have shared with them. At the time of writing we have not yet heard anything back on the code. For the reasons described in this appendix, we currently assume that the language experiments in discrete flows are not reproducible.

## Appendix D Additional experiments

A comparison of the performance for Cartesian products with different bases is shown in Table [9](#A4.T9). Note that this experiment was performed using a somewhat smaller architecture then in the main text. As can be seen, the performance difference between different Cartesian products is relatively small. The performance does decreases slightly over larger base numbers, indicating that it is better to choose a small base that results in fewer overall dimensions.

**Table 9: Cartesian Products with different base numbers trained using a slightly smaller version of the Argmax AR Flow on text8.**
| Model | text8 (bpc) |
| --- | --- |
| $d_{m}=1,M=27$ | 1.45 |
| $d_{m}=2,M=6$ | 1.44 |
| $d_{m}=3,M=3$ | 1.44 |
| $d_{m}=5,M=2$ | 1.44 |

A comparison of sampling time speeds are shown in Table [10](#A4.T10). A couple of orders in magnitude difference can be seen comparing autoregressive versus non-autoregressive models. This highlights the importance of researching generative models that can be built from non-autoregressive components. The main source of difference between our coupling approach and IAF/SCF is that we utilize mixture of discretized logistics as coupling transformation, which requires a iterative process to invert over 1 dimension. The multinomial diffusion takes in-between the time of autoregressive and coupling models. Also reducing steps reduces the required sampling time, as is expected.

**Table 10: Comparison of different methods in terms of sample time. Sample time is measured by generating a single text sample of length 256 averaged over $10$ runs, unless specified otherwise.**
| Model type |  | Model | Sample time (s) |
| --- | --- | --- | --- |
| ARM |  | 64 Layer Transformer | 35.5^† |
| VAE |  | AF/AF^⋆ (AR) | 156 $\pm 1.8$ |
|  | IAF / SCF^⋆ | 0.04 $\pm 0.004$ |  |
| Generative Flow |  | Argmax Flow, AR (ours) | 115 $\pm 0.03$ |
|  | Argmax Coupling Flow (ours) | 0.40 $\pm 0.03$ |  |
|  | Discrete Flow | 0.16^† |  |
| Diffusion |  | Multinomial Text Diffusion (ours) | 26.6 $\pm 2.2$^‡ |
|  | Multinomial Text Diffusion, 100 steps (ours) | 2.4 $\pm 0.16$ |  |

Due to the computational cost of running normalizing flows, it is not possible for us to run every model many times. However, generally single-run results suffice, as the performance variance of these models is relatively small. In Table [11](#A4.T11) the standard deviation and average performance for a selection of models is shown, taken over $3$ runs. Observe that these standard deviations are small compared to the reported differences between the models. Notice that standard deviations for coupling models are larger, but the performance difference between those types of models is also larger.

**Table 11: Average and standard deviations of several models.**
| Dequantization | Flow type | Dataset | average | stdev |
| --- | --- | --- | --- | --- |
| Argmax Flow (ours) | AR | text8 | 1.38 | $0.001$ |
| Argmax Flow (ours) | AR | enwik8 | 1.42 | $0.008$ |
| Argmax Flow (ours) | Coupling | text8 | 1.82 | $0.017$ |
| Argmax Flow (ours) | Coupling | enwik8 | 1.93 | $0.012$ |

Finally, we also compare argmax flows to a situation where its density model exactly matches the density model in on text8. In this experiment Argmax Flows (1.43 bpc) outperform CategoricalNF (1.45 bpc) in an equal setting.

## Appendix E Samples from the text models

Samples from our proposed models are presented in Table [12](#A5.T12) and a Multinomial Text Diffusion train is shown in Figure [7](#A5.F7), these results were not cherry-picked.

**Table 12: Samples from models trained on text8.**
| Model | Nr | Text |
| --- | --- | --- |
| Multinomial Diffusion | 1 | that the role of tellings not be required also action characters passed on constitution ahmad a nobilitis first be closest to the cope and dhur and nophosons she criticized itm specifically on august one three movement and a renouncing local party of exte |
| 2 | nt is in this meant the replicat today through the understanding element thinks the sometimes seven five his final form of contair you are lotur and me es to ultimately this work on the future all all machine the silon words thereis greatly usaged up not t |  |
| 3 | arity island louis has convinced privatist provinces the restrained marriage of his income ted guilds which in gulick performed in one nine six seven then sponly onward the bambat loving in separate including tichatta westell s doubled a bound of his futur |  |
| 4 | same early duration without education as a golden core power to the pirit of spain arriving wise speech art and r t plain firman q one five six the same as part of herald h rogenszers a art poetic of literature at shaft bressen three five three five eight |  |
| AR Argmax Flow | 1 | heartedness frege thematically infered by the famous existence of a function f from the laplace definition we can analyze a definition of binary operations with additional size so their functionality cannot be reviewed here there is no change because its |
| 2 | otal cost of learning objects from language to platonic linguistics examines why animate to indicate wild amphibious substances animal and marine life constituents of animals and bird sciences medieval biology biology and central medicine full discovery re |  |
| 3 | o use language combined with any of its subsets evolved into the group containing the primary concepts of a daily line on off the road and the material emulation of welcomes and prospects of pleasure and exercise have been committed projects in the economy |  |
| 4 | en that are beginning to forge since october one nine five zero the mandate was planted at k nigsberg during the car horizon at first please refer to a small government situated as well as in all these countries finally giving birth to a band here he was a |  |
| Coupling Argmax Flow | 1 | ns fergenur d alpha and le heigu man notabhe leglon lm n two six a gg<br>opa movement as sympathetic dutch the term bilirubhah acquired the bava<br>rian cheeh segt thmamouinaire vhvinus lihnos ineoneartis or medical iod<br>ine the rave wesp published harsy varb hhgh |
| 2 | and inequalities syllee mike jean demet in standard rather than fmxed liga and a piare nut is gruncionde aodadneveshiopyhabally uchc one viredtlty three ben yi agricultariis the only mefamantia or nuil and mid satio for kigou wore not on the war rits af |  |
| 3 | e g chain within the sale of cooperative oppine p nge tyae yarot bouatta real frequency one mbj or rorbepetam iw by someone c langt b kindoms is the single yenta valve nor eosed collagen surkeys in the goubark cuisine of animum and two trantual measurement |  |
| 4 | hilepuin the king pete was added to or who cefralded to kiark n and panhpur not souhhvestern bat batas mudtlu for this creatures chew palenque lii lasron gentla tzanemi derived from oo four issais nivissos with the name convertinus magaa named wes orieanr |  |

Figure: Figure 7: Intermediate steps of the generation chain of the Multinomial Text Diffusion model trained on text8.
Refer to caption: /html/2102.05379/assets/x14.png