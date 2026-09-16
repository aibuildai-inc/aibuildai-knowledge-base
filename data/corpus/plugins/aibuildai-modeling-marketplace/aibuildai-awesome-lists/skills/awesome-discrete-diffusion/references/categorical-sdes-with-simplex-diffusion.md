---
arxiv_id: "2210.14784"
title: "Categorical SDEs with Simplex Diffusion"
year: 2022
source: arxiv2md
---

## Abstract

Abstract Diffusion models typically operate in the standard framework of generative modelling by producing continuously-valued datapoints. To this end, they rely on a progressive Gaussian smoothing of the original data distribution, which admits an SDE interpretation involving increments of a standard Brownian motion. However, some applications such as text generation or reinforcement learning might naturally be better served by diffusing categorical-valued data, i.e., lifting the diffusion to a space of probability distributions. To this end, this short theoretical note proposes simplex diffusion , a means to directly diffuse datapoints located on an n 𝑛 n -dimensional probability simplex. We show how this relates to the Dirichlet distribution on the simplex and how the analogous SDE is realized thanks to a multi-dimensional Cox–Ingersoll–Ross process (abbreviated as CIR), previously used in economics and mathematical finance. Finally, we make remarks as to the numerical implementation of trajectories of the CIR process, and discuss some limitations of our approach.

## 1 Introduction and background

Diffusion models are a now well-established class of generative models that find applications notably in the image , video , speech domains, and even for molecule generation . These models proceed as follows. One adds noise progressively to data using a diffusion process to transform the complex data distribution to a simple easy-to-sample distribution. The generative model is obtained by simulating an approximation of the time-reversal of this process. The resulting “denoising” process is also a diffusion whose drift depends on the logarithmic gradients of
the noised data densities , i.e. the Stein *scores*. These scores are estimated using a neural network via score matching
. In the usual case where Gaussian noise is progressively added to the generative distribution, the score matching objective simply reduces to a least squares denoising term easily amenable to gradient descent.

In all which precedes, the datapoints are assumed to be vectors taking continuous values. Being able to proceed with diffusion when those datapoints are instead discrete-valued would further widen the applicability domain of diffusion models, in particular to language modelling and even reinforcement learning . We propose a construction of such a discrete diffusion in this short technical note. Our approach consists in directly deriving a tractable stochastic process that operates on the probability simplex itself, lifting traditional diffusion schemes to categorical distributions, rather than relying on auxiliary methods such as binary encoding . Because of this, we can use simplex diffusion in conjunction with the now standard mathematical machinery of diffusion models, including equivalent ODE formulation, and computation of an evidence lower bound (ELBO). Finally, we also discuss some specific limitations of our approach, namely the issues once encounters in practice when simulating high-dimensional simplices (i.e., for large values of $n$).

## 2 Simplex diffusion with the Cox–Ingersoll–Ross process

We first proceed to recall how one can sample from the Dirichlet distribution on the probability simplex using independent Gamma random variables. Then, we introduce a compatible stochastic process, the Cox–Ingersoll–Ross process.

### 2.1 Dirichlet distribution on the simplex

For a given integer $n\geq 2$, the $n-1$ dimensions probability simplex is the set of $n$-dimensional vectors $\bm{X}$ in $\mathbb{R}^{n}$ whose components $\bm{X}:=(X_{1},...,X_{n})$ satisfy $X_{i}\geq 0$ and $\sum_{i=1}^{n}X_{i}=1$. A point on the simplex is hence assimilated to an $n$-way categorical distribution.

The Dirichlet distribution is defined over the simplex as the conjugate prior of the categorical distribution. It is a multivariate, continuous distribution, parametrized by an arbitrary vector of strictly positive scalars $\bm{\alpha}$. The Dirichlet distribution $\mathcal{D}(\bm{\alpha})$ with parameters $\bm{\alpha}:=(\alpha_{1},...,\alpha_{n})$, where $\alpha_{1},...,\alpha_{n}>0$, has probability density function $f_{\mathcal{D}}$ given (w.r.t. the standard Lebesgue measure on $\mathbb{R}^{n}$) by

$$ $f_{\mathcal{D}}(x_{1},\cdots,x_{n};\alpha_{1},\cdots,\alpha_{n})=\frac{1}{Z(\bm{\alpha})}\prod_{i=1}^{n}{x_{i}^{\alpha_{i}-1}}$ (1) $$

with $Z(\bm{\alpha})$ a normalizing constant. In particular, the choice $\alpha_{i}=1$ for all $i$ recovers a uniform distribution over the simplex. In our construction, the Dirichlet distribution plays a role somewhat analogous to that of the Gaussian distribution in standard diffusions - in that it represents the desired stationary distribution of the diffusion process we will build below. Hence, and given its flexibility we focus on it, although other choices of simplex distributions are possible .

Sampling. It is well known that sampling from the Dirichlet distribution reduces to a two-step procedure: first, sampling $n$ *independent* Gamma random variables $Y_{1}\sim\mathcal{G}(\alpha_{1},\beta),...,Y_{n}\sim\mathcal{G}(\alpha_{n},\beta)$ where $\alpha_{i}$ is their shape parameter, and $\beta$ their common rate parameter. Second, normalizing those random variables to sum to $1$ then yields the Dirichlet-distributed random vector

$$ $\bm{X}=\bigg{(}\frac{Y_{1}}{\sum_{i=1}^{n}Y_{i}},...,\frac{Y_{n}}{\sum_{i=1}^{n}Y_{i}}\bigg{)}\sim\mathcal{D}(\bm{\alpha}).$ (2) $$

This result holds for any $\beta>0$ so we can use $\beta=1$ specifically. Taking these observations together, we now seek to find an $n$-dimensional stochastic process whose marginal distributions each converge to Gamma laws in the large-time limit. We exhibit such a process below.

### 2.2 The Cox-Ingersoll-Ross process

The Cox–Ingersoll–Ross (or CIR) process introduced in is a popular real-valued diffusion process used in econometrics and quantitative finance, both for yield curve (usually, the instantaneous interest rate) and stochastic equity volatility modelling. It is an instance of *square-root diffusion* defined by the following SDE in $\theta_{t}$: for any $\theta_{0}\geq 0$ and $a,b,\sigma>0$

$$ $\mathrm{d}\theta_{t}=b(a-\theta_{t})\mathrm{d}t+\sigma\sqrt{\theta_{t}}\mathrm{d}W_{t},$ (3) $$

where $(W_{t})_{t\geq 0}$ is a standard Brownian motion (or Wiener process). The solution to this SDE exists and is unique , despite the non-regularity of the square root term near zero. The CIR process is ergodic, almost surely non-negative and admits as invariant limiting distribution the Gamma distribution $\mathcal{G}(2ab/\sigma^{2},2b/\sigma^{2})$. If $2ab\geq\sigma^{2}$ and $\theta_{0}>0$, then the process is strictly positive, pathwise.

For our purpose, we can set $2b=\sigma^{2}$ so that ([3](#S2.E3)) becomes

$$ $\mathrm{d}\theta_{t}=b(a-\theta_{t})\mathrm{d}t+\sqrt{2b\theta_{t}}\mathrm{d}W_{t}$ (4) $$

and admits the Gamma distribution $\mathcal{G}(a,1)$ as limiting distribution.

Conditional mean and variance. One can readily check that for $t>0$

$$ $\mathbb{E}[\theta_{t}|\theta_{0}]=\theta_{0}\exp(-bt)+a(1-\exp(-bt))=a+\exp(-bt)(\theta_{0}-1)$ (5) $$

while

$$ $\displaystyle\textup{var}[\theta_{t}|\theta_{0}]$ $\displaystyle=2\theta_{0}(\exp(-bt)-\exp(-2bt))+a(1-\exp(-bt))^{2}$ $\displaystyle=2\theta_{0}(\exp(-bt)-\exp(-2bt))+a(1+\exp(-2bt)-2\exp(-bt))$ $\displaystyle=a+2\exp(-bt)(\theta_{0}-a)+\exp(-2t)(a-2\theta_{0}).$ (6) $$

$b$ can be thought of as the parameter governing diffusion speed. As $t\rightarrow\infty$, we have $\mathbb{E}[\theta_{t}|\theta_{0}]\rightarrow a$ and $\textup{var}[\theta_{t}|\theta_{0}]\rightarrow a$. At any point in time, the drift term in equation [3](#S2.E3) pushes $\theta_{t}$ back towards its long-term average $a$, a phenomenon known as *mean-reversion*. For this reason $b$ is also indicative of, and sometimes called, the speed of mean-reversion.

Density of increments. The transition density of the CIR process is available in closed-form thanks to Laplace transform techniques and can be sampled from exactly; i.e. we have

$$ $\theta_{t}|\theta_{0}\sim\frac{1-\exp(-bt)}{2}K,\qquad K\sim\chi^{2}\Big{(}2a,2\theta_{0}\frac{\exp(-bt)}{1-\exp(-bt)}\Big{)},$ (7) $$

where $\chi^{2}(\nu,\mu)$ denotes the non-central chi-squared distribution with $\nu$ degrees of freedom and non-centrality parameter $\mu$. We can write this density explicitly as

$$ $f(\theta_{t}|\theta_{0})=c\exp\Big{(}-c(\theta_{0}\exp(-bt)+\theta_{t})\Big{)}\Big{(}\frac{\theta_{t}\exp(bt)}{\theta_{0}}\Big{)}^{\frac{a-1}{2}}I_{a-1}\Big{(}2c\sqrt{\theta_{0}\theta_{t}\exp(-bt)}\Big{)},$ (8) $$

for $c=(1-\exp(-bt))^{-1}$ and $I_{a-1}$ being the modified Bessel function of the first kind of order $a-1$. This closed-form expression for the transition density for the CIR model makes usual denoising score matching techniques applicable, as we’ll see below.

### 2.3 Simplex diffusion

Simplex SDE. Our original purpose is to exhibit a diffusion whose marginal distribution, in the large time limit, provides samples from a Dirichlet distribution $\mathcal{D}(\bm{\alpha})$. It follows directly from previous section that this can be achieved by simulating first $n$ independent CIR processes $Y^{i}$ in parallel, resulting in a process $\bm{Y}_{t}$ with values in the positive orthant, following (with $\mathrm{d}W^{i}_{t}$ the independent increments of a standard $n$-dimensional Brownian motion $\bm{W}_{t}$, so that $\langle\mathrm{d}W^{i}_{t},\mathrm{d}W^{j}_{t}\rangle=\delta_{i,j}t$) :

$$ $\mathrm{d}Y^{i}_{t}=b(\alpha_{i}-Y^{i}_{t})\mathrm{d}t+\sqrt{2bY^{i}_{t}}\mathrm{d}W^{i}_{t}$ (9) $$

each $Y^{i}$ thus having limiting distribution $\mathcal{G}(\alpha_{i},1)$. We then consider the normalized, unit-sum vector

$$ $\bm{X}_{t}=\bigg{(}\frac{Y^{1}_{t}}{\sum_{i=1}^{n}Y^{i}_{t}},...,\frac{Y^{n}_{t}}{\sum_{i=1}^{n}Y^{i}_{t}}\bigg{)}$ (10) $$

This normalization projects $\bm{Y}_{t}$ from the positive orthant to the probability simplex. By construction, we have $\bm{X}_{t}=(X^{1}_{t},...,X^{n}_{t})\sim\mathcal{D}(\bm{\alpha})$ as $t\rightarrow\infty$. This is our main result, and enables us to perform diffusion towards a vertex of the simplex (a one-hot vector, representing the state of a categorical variable) in the time-reversal process.

Now since this multidimensional SDE retains a standard Brownian increment, both the time-reversal of the SDE, and reformulation as a standard ODE for ’probability flow’-type sampling proceed as usual. We detail those aspects below.

Time reversal. The SDE in equation ([9](#S2.E9)) is of the general (vector) form

$$ $\mathrm{d}\bm{Y}_{t}=\bm{f}_{t}(t,\bm{Y}_{t})\mathrm{d}t+\bm{G}(t,\bm{Y}_{t})\mathrm{d}\bm{W}_{t},$ (11) $$

where $\bm{f}(t,\bm{Y}_{t})=b(\bm{\alpha}-\bm{Y}_{t})$, $\bm{\alpha}=(\alpha_{1},...,\alpha_{n})^{\top}$ and $\bm{G}(t,\bm{Y}_{t})=\sqrt{2b}\cdot\textup{diag}(\sqrt{Y_{t}^{1}},...,\sqrt{Y^{n}_{t}})$. Let us also introduce further notation: $p_{t}=\textup{Law}(\bm{Y}_{t})$, the law of the probability density function of $\bm{Y}_{t}$, and $\bm{\Sigma}(t,\bm{Y}_{t})=\bm{G}(t,\bm{Y}_{t})\bm{G}(t,\bm{Y}_{t})^{\top}=2b\cdot\textup{diag}(\bm{Y}_{t})$.

The *time reversal* of the multidimensional CIR given by equation ([9](#S2.E9)) is the process $(\bm{Z}_{t})_{t\in[0,T]}$ such that $\bm{Z}_{t}=\bm{Y}_{T-t}$ satisfies

$$ $\displaystyle\mathrm{d}\bm{Z}_{t}$ $\displaystyle=\big{[}-b(\bm{\alpha}-\bm{Z}_{t})+2b\cdot\textup{diag}(\bm{Z}_{t})~{}\nabla_{\bm{Z}}\log p_{T-t}(\bm{Z}_{t})+2b\textbf{1}\big{]}\mathrm{d}t+\bm{G}(T-t,\bm{Z}_{t})\mathrm{d}\bm{W}_{t}$ (12) $$

with $\bm{Z}_{0}\sim p_{T}$. In practice, we will approximate this time reversal by the diffusion

$$ $\mathrm{d}\bm{Z}_{t}=\big{[}-b(\bm{\alpha}-\bm{Z}_{t})+2b\cdot\textup{diag}(\bm{Z}_{t})~{}\bm{s}_{T-t}(\bm{Z}_{t})+2b\textbf{1}\big{]}\mathrm{d}t+\bm{G}(T-t,\bm{Z}_{t})\mathrm{d}\bm{W}_{t},$ (13) $$

with $\bm{Z}_{0}\sim p_{\textup{ref}}$ where $p_{\textup{ref}}(z^{1},...,z^{n})=\prod_{i=1}^{n}\mathcal{G}(z^{i};\alpha_{i},1)$. Here $\bm{s}_{t}(\bm{x})$ is a neural score network approximating $\nabla_{\bm{x}}\log p_{t}(\bm{x})$.

Max likelihood training. This form also lends itself to computation of an evidence lower bound. Maximizing the likelihood of the data is equivalent to minimizing the KL divergence between the terminal time marginal induced by our SDE and the data distribution, which we compute exactly as in .
Let $\mathcal{P}$ and $\mathcal{\hat{P}}$ the path measures corresponding respectively to equations ([12](#S2.E12)) and ([13](#S2.E13)). Then by Girsanov theorem, the KL-divergence $\textup{KL}(\mathcal{P}||\mathcal{\hat{P}})$ satisfies

$$ $\textup{KL}(\mathcal{P}||\mathcal{\hat{P}})=\textup{KL}(p_{T}||p_{\textup{ref}})+\Delta I$ (14) $$

with the integral difference $\Delta I$ given by

$$ $\displaystyle\Delta I$ $\displaystyle=\frac{1}{2}\mathbb{E}_{\mathcal{P}}\left[\int_{0}^{T}\Big{\|}\bm{\Sigma}(T-t,\bm{Z}_{t})\nabla_{\bm{Z}}\log p_{T-t}(\bm{Z}_{t})-\bm{\Sigma}(T-t,\bm{Z}_{t})\bm{s}_{T-t}(\bm{Z}_{t})\Big{\|}^{2}_{\bm{\Sigma}^{-1}(T-t,\bm{Z}_{t})}\mathrm{d}t\right]$ $\displaystyle=\frac{1}{2}\mathbb{E}_{\mathcal{P}}\left[\int_{0}^{T}\Big{\|}\nabla_{\bm{Y}}\log p_{t}(\bm{Y}_{t})-\bm{s}_{t}(\bm{Y}_{t})\Big{\|}^{2}_{\bm{\Sigma}(t,\bm{Y}_{t})}\mathrm{d}t\right],$ $$

where we use the notation $||\bm{x}||_{\bm{A}}=\bm{x}^{\top}\bm{A}\bm{x}$.
Now thanks to the denoising score matching trick, we get that, up to the additive constant (w.r.t. optimization) term $\textup{KL}(p_{T}||p_{\textup{ref}})$,

$$ $\displaystyle\textup{KL}(\mathcal{P}||\mathcal{\hat{P}})$ $\displaystyle\equiv\frac{1}{2}\mathbb{E}_{\mathcal{P}}\left[\int_{0}^{T}\Big{\|}\nabla_{\bm{Y}}\log p_{t|0}(\bm{Y}_{t}|\bm{Y}_{0})-\bm{s}_{t}(\bm{Y}_{t})\Big{\|}^{2}_{\bm{\Sigma}(t,\bm{Y}_{t})}\mathrm{d}t\right]$ $\displaystyle\equiv\mathbb{E}_{\mathcal{P}}\left[\int_{0}^{T}b\Big{(}\nabla_{\bm{Y}}\log p_{t|0}(\bm{Y}_{t}|\bm{Y}_{0})-\bm{s}_{t}(\bm{Y}_{t})\Big{)}^{\top}\textup{diag}(\bm{Y}_{t})\Big{(}\nabla_{\bm{Y}}\log p_{t|0}(\bm{Y}_{t}|\bm{Y}_{0})-\bm{s}_{t}(\bm{Y}_{t})\Big{)}\mathrm{d}t\right]$ $$

ODE formulation for sampling. The ODE formulation consists in finding an ODE

$$ $\frac{\mathrm{d}\bm{Y}_{t}}{\mathrm{d}t}=\tilde{\bm{f}}_{t}(t,\bm{Y}_{t})$ (15) $$

that admits the same temporal marginals as the solution of equation ([9](#S2.E9)). Using the formulation in , or simply by applying Ito’s lemma, one gets:

$$ $\tilde{\bm{f}}_{t}(t,\bm{Y}_{t})=\bm{f}_{t}(t,\bm{Y}_{t})-\frac{1}{2}\nabla\cdot\bm{\Sigma}(t,\bm{Y}_{t})-\frac{1}{2}\bm{\Sigma}(t,\bm{Y}_{t})\nabla_{\bm{Y}}\log p_{t}(\bm{Y}_{t}),$ (16) $$

which in our case results in

$$ $\frac{\mathrm{d}\bm{Y}_{t}}{\mathrm{d}t}=b(\bm{\alpha}-\textbf{1}-\bm{Y}_{t}-\textup{diag}(\bm{Y}_{t})\nabla_{\bm{Y}}\log p_{t}(\bm{Y}_{t}))$ (17) $$

This highlights another benefit of the ODE formulation: we can simulate the ODE in the log-domain and get an equation of the form $\frac{\mathrm{d}\log\bm{y}}{\mathrm{d}t}=-b(1+\nabla_{\bm{y}}\log p_{t}(\bm{y}))$, to promote numerical stability.

Remarks on numerical simulation. The CIR process has been extensively used and studied within Monte Carlo methods in quantitative finance. Care must be taken in simulating its trajectories; this can typically require an additional scalar $\epsilon$ stabilization parameter inside of the square-root diffusion term in equation [9](#S2.E9) in order to avoid path termination due to discretization error. Another avenue is to observe that under specific conditions on their parameters, the sum of independent, squared Ornstein–Uhlenbeck processes is identical in law to a CIR process ; this observation relates to Bessel processes . This enables substituting a single CIR path for multiple Ornstein-Uhlenbeck paths, trading off stability for computation.

Limitations. We might want to use our approach on very high dimensional simplices in order to simulate *one-of-many* categoricals - for instance, when modelling language tokens over a sizeable vocabulary, or in the case of a large action-space policy. This comes with practical issues, chief amongst those being the potential presence of outliers in the categorical distribution. When we draw a sample from the transition density of the CIR process for a given $t$, we can determine the rank of the ground truth token in the resulting (unnormalized) vector. We observed in practice that the distribution of that rank - whose closed form law involves large, and possibly intractable integrals - is extremely heavy-tailed. Informally, this can lead to noisy results. We found empirically this phenomenon to be particularly relevant in high dimensions.

Finally, we note that while the interpretation of noisy vectors as unnormalized probability distributions via a Dirichlet prior is useful to build intuition, it is not rigorous. When one considers the posterior distribution at token level $p_{t}(\bm{x}_{0}|\bm{x}_{t})$, where $\bm{x}_{0}$ is a one-hot vector representing a token, and $\bm{x}_{t}$ is the noisy unnormalized probability input vector, we can apply Bayes’ rule and get

$$ $p_{t}(\bm{x}_{0}|\bm{x}_{t})=\frac{p_{t}(\bm{x}_{t}|\bm{x}_{0})p(\bm{x}_{0})}{\sum_{\bm{x}_{0}}p_{t}(\bm{x}_{t}|\bm{x}_{0})p(\bm{x}_{0})}$ (18) $$

thus showing that $p(\bm{x}_{0}|\bm{x}_{t})$ is actually nonlinear in $\bm{x}_{t}$.

Related and alternative approaches. The Cox–Ingersoll–Ross process is seldom used in machine learning. Similar derivations to ours nonetheless previously appeared in , where a CIR process is also used to approximate a Dirichlet distribution, but in a Bayesian inference context, with the very different purpose of obviating discretization error in stochastic gradient MCMC . Other stochastic processes than the CIR can be built that admit the Dirichlet distribution as a limiting distribution. considers functions of the components of a multivariate Brownian motion running on a hypersphere. When those functions are all identically a squaring, by construction the squared components sum to $1$ and can thus represent a categorical probability vector. In that setting the invariant distribution of the squared-components vector is proven to be symmetric Dirichlet with parameter $1/2$. Unlike ours, that approach is however not fully compatible with standard diffusion score matching, since the transition density of the Brownian motion on the sphere is to our knowledge not known in closed form - it is merely possible to sample from . Other choices than a Dirichlet limiting distribution are also possible, even as it represents a reasonable and flexible prior family; proposes a generic *log-ratio transform* projecting unconstrained, multivariate distributions defined on $\mathbb{R}^{n}$ onto the simplex. Separately, perform an asymptotic expansion of the heat kernel on statistical manifolds (including an approximation of the simplex), with application to the multinomial family of distributions towards text classification.

## 3 Conclusion

We have introduced *simplex diffusion*, a simple method that uses a multi-dimensional Cox-Ingersoll-Ross process, via a unit-sum normalization of its time marginals, to diffuse categorical distributions directly on the probability simplex. Our approach is tractable and compatible with the tools of standard stochastic calculus central to diffusion models. Further research will involve operationalizing and evaluating deep learning models that leverage this principle.