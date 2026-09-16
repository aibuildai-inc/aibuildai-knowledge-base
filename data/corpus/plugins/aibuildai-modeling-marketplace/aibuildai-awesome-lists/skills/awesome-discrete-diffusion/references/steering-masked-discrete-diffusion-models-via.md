---
arxiv_id: "2410.08134"
title: "Steering Masked Discrete Diffusion Models via Discrete Denoising Posterior Prediction"
year: 2024
source: arxiv2md
---

## Abstract

Abstract Generative modeling of discrete data underlies important applications spanning text-based agents like ChatGPT to the design of the very building blocks of life in protein sequences.
However, application domains need to exert control over the
generated data by steering the generative process—typically via RLHF—to satisfy a specified property, reward, or affinity metric. In this paper, we study the problem of steering Masked Diffusion Models (MDMs), a recent class of discrete diffusion models that offer a compelling alternative to traditional autoregressive models. We introduce Discrete Denoising Posterior Prediction ( DDPP ), a novel framework that casts the task of steering pre-trained MDMs as a problem of probabilistic inference by learning to sample from a target Bayesian posterior. Our DDPP framework leads to a family of three novel objectives that are all simulation-free, and thus scalable while applying to general non-differentiable reward functions. Empirically, we instantiate DDPP by steering MDMs to perform class-conditional pixel-level image modeling, RLHF-based alignment of MDMs using text-based rewards, and finetuning protein language models to generate more diverse secondary structures and shorter proteins. We substantiate our designs via wet-lab validation, where we observe transient expression of reward-optimized protein sequences.

## 1 Introduction

The success of diffusion models in continuous spaces, leading to state-of-the-art foundation models for image  and video synthesis , has spurned several attempts to translate these approaches for the generative modeling of discrete structures. The most performant approaches squarely fall under the scalable framework of absorbing state discrete diffusion , with new simplified training recipes that result in Masked Diffusion Models (MDMs) . Indeed, recent MDMs now rival autoregressive models of a similar scale to GPT-2  for language modeling, with the potential for further progress through scaling. Furthermore, MDM style models are not constrained to generating data sequentially—unlike autoregressive models—which invites a more straightforward application to domains without a natural causal ordering, e.g. molecule generation , discrete modeling of images , and modeling protein sequences .

Critical to the successful deployment of discrete generative models in practical applications—beyond simply producing high-quality samples—is the ability to steer the generated samples to optimize a pre-specified downstream metric. For instance, in Language Modeling (LM) it is desirable to bias the model’s generations to be sanitized from harmful responses , or aiming to generate protein sequences that are highly likely to be successfully synthesized and expressed in real wet lab settings .
Put succinctly, highly performant discrete generative models are required to be aligned in a manner that fine-tuning against downstream reward models has the intended effect of *controllable generation*, wherein the model post fine-tuning selects high-scoring samples from the universe of possible high-fidelity generations.

The standard approach for incorporating steerability into discrete generative models, which are autoregressive, using pre-defined reward models is often framed as a fine-tuning task using reinforcement learning from human feedback (RLHF) . However, applying RLHF frameworks to diffusion models is far more challenging. Unlike autoregressive models, diffusion models do not allow for straightforward computation of a sample’s exact likelihood without costly simulations. Although fine-tuning diffusion models that bypass exact likelihood computation can yield simulation-free algorithms that resemble RLHF , these methods effectively optimize a loose lower bound to the true RLHF objective, leading to unstable training and suboptimal fine-tuning performance. Consequently, steering diffusion models in continuous spaces is primarily done through inference techniques that leverage the gradient of a conditional model in the form of guidance . Unfortunately, discrete settings do not allow for principled definitions of guidance due to the lack of a conventional gradient operator. As a result, at present, there exists no scalable and rigorous method to steer and align Masked Diffusion Models to optimize desired reward models.

Main contributions. In this paper, we cast the problem of steering a Masked Diffusion Model as a task of probabilistic inference in sampling from a target Bayesian posterior. More precisely, we construct the target Bayesian posterior as being proportional to the product distribution of a base pre-trained MDM model modulated by a pre-specified reward model. Importantly, this sampling viewpoint is fully compatible with classical RLHF for autoregressive models , but enjoys broader applicability as for the first time it can be applied to discrete diffusion models. Under this sampling perspective, our key insight is that the challenging sampling problem can be solved by learning an amortized sampler, which when taken as an MDM, can be viewed as *finetuning* the pre-trained MDM model by learning to approximate the (reward-induced) Bayesian posterior.

We introduce Discrete Denoising Posterior Prediction (DDPP), a novel framework that exploits the denoising posterior parametrization inherent to current MDMs to define a series of simpler matching problems across varying corruption (masking) levels of the target Bayesian posterior. In particular, DDPP designs a forward process that corrupts the Bayesian posterior through a forward masking process and frames the finetuning task as learning another MDM to approximate the corresponding reverse process. As a result, each matching problem in the reverse process requires the construction of a “denoising" Bayesian posterior that is conditioned on a partially masked sample which we demonstrate is simply proportional to the pre-trained model’s own denoising posterior and the (terminal) reward of the fully unmasked sample.
Crucially, each matching problem in DDPP can be defined on a particular noise level without running the entire forward (corruption) process. Consequently, this makes DDPP a *simulation-free* method which is a key ingredient needed to finetune large pre-trained MDMs. We test the empirical caliber of DDPP by steering MDMs across a multitude of domains ranging from images to protein sequences and steering MDM-based language models. We observe DDPP fine-tuned MDMs lead to competitive performance on images, transient expression of reward-optimized protein sequences (with high secondary structure diversity and $\beta$-sheet composition) in a wet-lab setting, and natural textual responses that are steered to human sentiments.

## 2 Background and preliminaries

Notation and convention.
A discrete data sample $x$ is identified by its realization over a vocabulary set ${\mathcal{V}}=\{1,\dots,d-1\}$, over $d-1$ possible categories. Of particular interest is the setting of masked diffusion models that include an additional $d$-th category of a masked token $\mathbf{m}$ to the vocabulary ${\mathcal{V}}$ which serves as an absorbing state for the diffusion process. A discrete token is represented by the one-hot vector $e^{i}\in\Delta^{d}$ in the $d$-dimensional probability simplex and corresponds to placing a $1$ on the $i$-th index and 0 0 on all the other $d-1$ indices. In this paper, by convention, we set $e^{\mathbf{m}}=e^{d}$ as the one hot vector associated with the masked state $\mathbf{m}$. A categorical distribution over $d$-categories, $\text{Cat}(x;p)$, is constructed by placing a Dirac $\delta$ with weight $p^{i}$, with the constraint $\sum_{i}p^{i}=1$ and the density of a discrete sample is written as $p(X=x)=\sum_{i=0}^{d}p^{i}\delta(x-e^{i})$, where $X$ is the discrete random variable.

A sequence $\mathbf{x}=(x^{1},\dots,x^{n})$ of $n$ tokens is defined over the product space ${\mathcal{V}}^{n}=\{1,\dots,\mathbf{m}\}^{n}$, and its corresponding probability mass function is given by $p(X=\mathbf{x})=\prod_{i}^{n}\sum_{j=0}^{d}p^{j}\delta(x^{i}-e^{j})$. To reduce notational clutter, we make use of the shorthand $\delta(y)$ to denote a Dirac measure on a discrete sample $y$ and interchangeably write $p(X=\mathbf{x})=p(\mathbf{x})$ to denote the probability mass function. A dataset of sequences is designated as samples from the data distribution $p_{\rm{data}}$ to be learned by a discrete diffusion model $q_{\theta}$, with parameters $\theta$.
Discrete diffusion models, like their continuous counterparts, are stochastic processes that evolve with time $t\in[0,1]$ such that $t=0$ corresponds to $p_{\rm{data}}:=p_{0}$ and $t=1$ corresponds to the terminal distribution, $p_{1}$ of the process.
As a discretization of time, we divide $[0,1]$ into $T$ intervals, and let $t(i)=i/T$. For brevity, we drop $i$ and simply write $t$ to denote the corresponding discrete timestep $t(i)$. The notation $0:t$ designates a collection of objects, e.g. densities $p(\mathbf{x}_{0:t})$, starting from time $t$ to and including time $t=0$.
A trajectory of sequences is denoted as $\tau(\mathbf{x}_{0:1})=\mathbf{x}_{1}\to\dots\to\mathbf{x}_{t}\to\mathbf{x}_{t
-1}\to\dots\to\mathbf{x}_{0}$. Finally, we use subscripts to denote the time index—i.e. $p_{t}$—and reserve superscripts to designate indices over a set such as a specific sample $\mathbf{x}^{i}$ among a collection of samples or dimensions within a vector, e.g. dimension $x^{i}$ in a sequence.

Problem Setting.
We are interested in the task of *probabilistic inference* of sampling from an unnormalized target distribution $\pi_{0}(\mathbf{x}_{0})$ defined over a discrete space consisting of $n$ tokens $\mathbf{x}\in{\mathcal{X}}^{n}$,

$$ $\pi_{0}(\mathbf{x}_{0})=\frac{p_{0}^{\text{pre}}(\mathbf{x}_{0})R(\mathbf{x}_{ 0})}{{\mathcal{Z}}_{\pi_{0}}},\quad R(\mathbf{x}_{0})=\frac{\exp(-{\mathcal{E} }(\mathbf{x}_{0}))}{{\mathcal{Z}}_{R}}.$ (1) $$

A key aspect of the considered setting is that $\pi_{0}(\mathbf{x}_{0})$ is defined as the product distribution of a pre-trained masked discrete diffusion model $p_{0}^{\text{pre}}(\mathbf{x}_{0})$ and a distribution induced by a (potentially differentiable) reward model $R:{\mathcal{X}}^{n}\to\mathbb{R}$.
The problem definition in Eq. 1 is an instance of Bayesian posterior sampling where the pre-trained MDM is the prior and reward acts as the likelihood or observation model which modulates samples with a high score.
For instance, in scientific domains, the reward model can be provided as a Boltzmann distribution with a known energy function ${\mathcal{E}}(\mathbf{x}_{0})$, or a human preference model as in RLHF .
Importantly, this setting does not afford us *any* ground truth samples from $\pi_{0}(\mathbf{x}_{0})$ in the form of a dataset which prevents classically training another generative model.
Instead, we are able to evaluate the reward model—and in special cases its gradient $\nabla R$—but not the normalizing constant, i.e. the partition function ${\mathcal{Z}}_{\pi_{0}}$.
Samples from the posterior $\pi_{0}$ thus lie in the intersection of the modes of both the pretrained MDM and the reward model. As a result, learning an amortized sampler, $q_{\theta}(\mathbf{x}_{0})$, for $\pi(\mathbf{x}_{0})$ is rationally equivalent to finetuning the pretrained MDM $p_{0}^{\text{pre}}(\mathbf{x}_{0})$ using the reward $R(\mathbf{x}_{0})$ in an analogous manner to RLHF  and is the main focus and contribution of this paper and outlined in §3.2.

### 2.1 Simplified Masked discrete diffusion

We are interested in developing a discrete diffusion model directly on discrete data—i.e. without embeddings or continuous reparameterizations—whose approach mirrors the construction of diffusion models for continuous spaces. Consequently, we require the specification of a forward process that converts discrete data $\mathbf{x}_{0}\sim p_{0}$ at time $t=0$ to an unstructured prior, $p_{1}$ at the terminal time $t=1$. The specification of a forward process via the transition kernel $p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0})$ implies a unique time reversal of this forward process, termed the “reverse process”, such that simulating from this reverse process results in samples from the desired target data distribution $p_{0}(\mathbf{x}_{0})$.

We restrict our attention to the recent performant “simplified masked” forward process which hits a terminal distribution of all mask tokens in a sequence $p_{1}=[\delta(\mathbf{m})]^{n}$.
Given a non-masked token in a sequence, $x_{0}^{i}\in\mathbf{x}$ the simplified masked forward process
increases the likelihood of transition to the mask state as time increases. Moreover, the masked forward process is simplified by design since the transition probabilities of a token
unmasking ($x_{t+1}^{i}\neq\mathbf{m}$ when $x_{t}^{i}=\mathbf{m}$)
is set to zero—i.e. the token remains a masked token for the remainder of the trajectory. The design of the simplified forward process is also independent across each dimension of the sequence, conditioned on $\mathbf{x}_{0}$, which allows us to model the transitions of each discrete token in a sequence separately. In totality, the forward process for a sequence $\mathbf{x}_{0}$ can be summarized using the following expression for the transition kernel $p_{t}(x^{i}_{t}|x^{i}_{0})$:

$$ $p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0})=\prod_{i=1}^{n}p_{t}(x_{t}^{i}|x_{0}^{i}) =\prod_{i=1}^{n}\text{Cat}(x^{i}_{t};\alpha_{t}\delta(x^{i}_{0})+(1-\alpha_{t} )\delta(\mathbf{m})),$ (2) $$

where $\alpha_{t}$ is an invertible reparameterization of time such that $\alpha_{0}=1$ and $\alpha_{1}=0$. Effectively, $\alpha_{t}$ corresponds to the noise schedule which corrupts the discrete data to $p_{1}$. The corresponding marginal density induced by the forward process at time $t$ can written as $p_{t}(\mathbf{x}_{t})=\sum_{\mathbf{x}_{0}}p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0}
)p_{0}(\mathbf{x}_{0})$.

The reverse process which denoises a sample from $t\to t-1$, and is the time reversal of the simplified masked forward process, also factorizes over each dimension of a sequence $\mathbf{x}$. The probability $p_{t}(x^{i}_{t-1}|x^{i}_{t},x^{i}_{0})$ of a reverse transition is given by the following posterior conditioned on $x^{i}_{0}$,

$$ $p_{t}(x^{i}_{t-1}|x^{i}_{t},x^{i}_{0})=\begin{cases}\text{Cat}(x^{i}_{t-1}; \delta(x^{i}_{t}))&x^{i}_{t}\neq\mathbf{m}\\ \text{Cat}\left(x^{i}_{t-1};\frac{(1-\alpha_{t-1})\delta(\mathbf{m})+(\alpha_{ t-1}-\alpha_{t})\delta(x^{i}_{0})}{1-\alpha_{t}}\right)&x^{i}_{t}=\mathbf{m}. \end{cases}$ (3) $$

Under the reverse process once a token transitions out of the masked state for a time $t>0$ it remains in this state for the remainder of the trajectory. The analytical form of the posterior suggests a natural mean parametrization for a denoiser in a discrete diffusion model, $\mu_{\theta}:{\mathcal{X}}\times\mathbb{R}\to\Delta^{d}$, which predicts the clean sample at $t=0$ by denoising a noisy $x^{i}_{t}$,

$$ $q_{t,\theta}(x^{i}_{t-1}|x^{i}_{t},\mu_{\theta}(x^{i}_{t},t))=\begin{cases} \text{Cat}(x^{i}_{t-1};\delta(x^{i}_{t}))&x^{i}_{t}\neq\mathbf{m}\\ \text{Cat}\left(x^{i}_{t-1};\frac{(1-\alpha_{t-1})\delta(\mathbf{m})+(\alpha_{ t-1}-\alpha_{t})\mu_{\theta}(x^{i}_{t},t)}{1-\alpha_{t}}\right)&x^{i}_{t}= \mathbf{m}.\end{cases}$ (4) $$

Interestingly, the mean parametrization $\mu_{\theta}$ used in the posterior is equivalent to predicting the concrete score  which is the discrete equivalent of the Stein score found in conventional continuous diffusion models . As the number of steps $t\to\infty$, training yields a valid *evidence lower bound* (ELBO) to the marginal log-likelihood of the data distribution $\log p(\mathbf{x}_{0})$,

$$ $\log p(\mathbf{x}_{0})\geq-\int^{1}_{0}\frac{d\alpha_{t}}{dt}\cdot\frac{1}{1- \alpha_{t}}\mathbb{E}_{\mathbf{x}_{t}\sim p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0}) }\left[\sum_{i=1}^{n}(x_{0}^{i})^{T}\log\mu_{\theta}(x^{i}_{t},t)\right]dt.$ (5) $$

Thus, when given access to samples $\mathbf{x}_{0}\sim p_{0}$ training an MDM can be seen as optimizing a weighted cross-entropy loss and is analogous to fitting a (mean-field) variational posterior distribution $q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})=\text{Cat}(\mathbf{x}_{0};\mu_{
\theta}(\mathbf{x}_{t},t))$ that matches the first moments of $p_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})$ and also minimizes the forward KL divergence $\mathbb{D}_{\mathrm{KL}}(p_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})p_{t}(\mathbf{x}_
{t})||q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})p_{t}(\mathbf{x}_{t}))$ .

## 3 Posterior Sampling via Discrete Denoising Posterior Prediction

Given access to a pretrained masked discrete diffusion model $p_{0}^{\text{pre}}(\mathbf{x}_{0})$ we wish to sample from the reward-induced Bayesian posterior distribution $\pi_{0}(\mathbf{x}_{0})$ $\propto p_{0}^{\text{pre}}(\mathbf{x}_{0})R(\mathbf{x}_{0})$. We solve this sampling problem by first defining a time-dependent forward masking process that progressively adds noise to $\pi_{0}$ yielding the noisy reward-induced posterior $\pi_{t}(\mathbf{x}_{t})=\sum_{\mathbf{x}_{0}}\pi_{t}(\mathbf{x}_{t}|\mathbf{x}
_{0})\pi_{0}(\mathbf{x}_{0})$, where we set $\pi_{t}(\mathbf{x}_{t}|\mathbf{x}_{0})=p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0})$ as it is the same masking process for the pre-trained MDM.
Unfortunately, since $p_{0}^{\text{pre}}(\mathbf{x}_{0})$ is an MDM it does not easily provide an exact likelihood. Undeterred we seek to approximate the reverse process $\pi_{t}(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ tied to the masking forward process by using another parametrized model $q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})=\text{Cat}(\mathbf{x}_{0};\mu_{
\theta}(\mathbf{x}_{t},t))$ which we take to be another MDM.

Matching sub-trajectories.
To approximate the reverse process using an MDM we require matching the denoising trajectory $\tau(\mathbf{x}_{0:t})$ of the reward-induced posterior $\pi_{t}(\mathbf{x}_{0},\dots,\mathbf{x}_{t-1}|\mathbf{x}_{t})$ across all masking levels. Assisted in this endeavor, we recall the fact that since $p_{0}^{\text{pre}}(\mathbf{x}_{0})$ is also an MDM, we have direct access to the pre-trained model’s denoiser. Thus, we can compute any transition density starting from $p^{\text{pre}}_{t}(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mu^{\text{pre}}(\mathbf{x}
_{t},t))$ to the posterior over the endpoint $p^{\text{pre}}_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})$, conditioned on a partially masked sample $\mathbf{x}_{t}$. We form the sub-trajectory matching problem as an instantiation of a detailed balance constraint starting from a partially masked sequence $\mathbf{x}_{t}$ of a clean starting point $\mathbf{x}_{0}$:

$$ $q_{\theta}(\mathbf{x}_{0},\dots,\mathbf{x}_{t-1}|\mathbf{x}_{t},\hat{\mathbf{x }}_{0})p_{t}(\mathbf{x}_{t})=\pi_{t}(\mathbf{x}_{0},\dots,\mathbf{x}_{t-1}| \mathbf{x}_{t})p_{t}(\mathbf{x}_{t}).$ (6) $$

Setting $\hat{\mathbf{x}}_{0}=\mu_{\theta}(\mathbf{x}_{t},t)$ as the MDM’s denoised sample, then $\pi_{t}(\mathbf{x}_{0},\dots,\mathbf{x}_{t-1}|\mathbf{x}_{t})$ is defined as,

$$ $\displaystyle\pi_{t}(\mathbf{x}_{0},\dots,\mathbf{x}_{t-1}|\mathbf{x}_{t})$ $\displaystyle=\frac{p_{t}^{\text{pre}}(\mathbf{x}_{0},\dots,\mathbf{x}_{t-1}| \mathbf{x}_{t})R(\mathbf{x}_{0})}{{\mathcal{Z}}_{\pi_{t}}(\mathbf{x}_{t})}= \frac{\prod^{t}_{j=1}p_{t}^{\text{pre}}(\mathbf{x}_{j-1}|\mathbf{x}_{j},\hat{ \mathbf{x}}^{\text{pre}}_{0})R(\mathbf{x}_{0})}{{\mathcal{Z}}_{\pi_{t}}( \mathbf{x}_{t})}.$ $$

The detailed balance constraint over sub-trajectories suggests a natural discrete denoising posterior predictive (DDPP) objective that minimizes the mean squared error of a log-ratio between the denoising sub-trajectories of the amortized MDM sampler and the reward-induced target posterior,

$$ ${\mathcal{L}}^{\text{PP}}_{\tau}=\mathbb{E}_{t,\mathbf{x}_{t}}\Big{[}\mathbb{E }_{\tau(\mathbf{x}_{0:t})}[\|\log q_{\theta}(\mathbf{x}_{0:t-1}|\mathbf{x}_{t} ,\hat{\mathbf{x}}_{0}))-\log p_{t}^{\text{pre}}(\mathbf{x}_{0:t-1}|\mathbf{x}_ {t})+\kappa\|^{2}_{2}]\Big{]},$ (7) $$

where reward and the log partition function are captured in the constant $\kappa=\log{\mathcal{Z}}_{\pi_{t}}(\mathbf{x}_{t})-\log R(\mathbf{x}_{0})$.
Interestingly, we can form an equivalent expression for the sub-trajectory loss ${\mathcal{L}}^{\text{PP}}_{\tau}$ above by sampling two intermediate points $\mathbf{x}_{s},\mathbf{x}_{s-\gamma}$ in the sub-trajectory $\tau(\mathbf{x}_{0:t})$, such that $0<s-\gamma<s<t$:

$$ ${\mathcal{L}}^{\text{PP}}_{\tau}=\mathbb{E}_{t,\mathbf{x}_{t},\tau(\mathbf{x}_ {0:t})}\left[\left\|t\mathbb{E}_{s,\mathbf{x}_{s},\mathbf{x}_{s-\gamma}}\left[ \log q_{\theta}(\mathbf{x}_{s-\gamma}|\mathbf{x}_{s},\hat{\mathbf{x}}_{0})- \log p_{t}^{\text{pre}}(\mathbf{x}_{s-\gamma},|\mathbf{x}_{s},\hat{\mathbf{x}} ^{\text{pre}}_{0})+\kappa\right]\right\|^{2}_{2}\right].$ (8) $$

The proof for this equivalence is presented in §C.3.
Note that we sample $s,s-\gamma\sim{\mathcal{U}}[0,t],{\mathcal{U}}[0,s]$ uniformly, and when $\gamma=1/T$ we sample $\mathbf{x}_{s-1}$ which is simple to do since the $\tau(\mathbf{x}_{0:t})$ already contains this information. Crucially, unlike Eq. 7 the reformulation of the sub-trajectory loss in Eq. 8 is effectively a simulation-free version of Relative Trajectory Balance (RTB) . If the approximation $q_{t,\theta}$ matches the denoising reward-induced target posterior over all sub-trajectories then the reverse process of $q_{t,\theta}$ can be simulated to draw samples that follow $\pi_{0}(\mathbf{x}_{0})$.
Consequently, we term the $q_{t,\theta}$ that minimizes the DDPP objective in Eq. 8 as the *finetuned MDM* which solves the probabilistic inference task of sampling from $\pi_{0}(\mathbf{x}_{0})$.

In contrast to learning MDMs in typical generative modeling setups, the DDPP objective requires the computation of the intractable log partition function $\log{\mathcal{Z}}_{\pi_{t}}(\mathbf{x}_{t})$ evaluated at $\mathbf{x}_{t}$ which is a component of the term $\kappa$.
This observation motivates the design of three concrete learning objectives for posterior matching, which as a collection we term the Discrete Denoising Posterior Prediction framework. Specifically, finetuning $q_{t,\theta}$ under a DDPP framework can be done in the following algorithms: 1.) DDPP-IS which uses a Monte Carlo based importance sampling estimate to approximate $\log{\mathcal{Z}}_{\pi_{t}}$ in Eq. 8, 2.) DDPP-LB that constructs a lower bound to DDPP-IS that is cheaper to evaluate by parameterizing $\log{\mathcal{Z}}_{\pi_{t}}$, and 3.) DDPP-KL which uses a discrete gradient estimator to bypass computing $\log{\mathcal{Z}}_{\pi_{t}}$ at the cost of requiring a differentiable reward—i.e. $\nabla R$.

### 3.1 Estimating the log partition function

Inspecting the posterior predictive objective in Eq. 8 we remark that it is a simulation-free stochastic regression objective which does not require a differentiable reward as the loss computes $R(\mathbf{x}_{0})$ and not a gradient of the reward. Consequently, this makes the posterior predictive objective both a scalable and efficient objective for fine-tuning large pre-trained MDMs as long the reward model is easy to evaluate. Moreover, the posterior predictive objective is also an *off-policy* objective as it can be evaluated using any partially masked samples $\mathbf{x}_{t}\sim p(\mathbf{x}_{t}|\mathbf{x}_{0})$. Practically, this means that fine-tuning can be performed using a replay buffer of samples from a biased dataset, e.g. the original training set for $p_{0}^{\text{pre}}$, or even partially masked sequences that arrive from a different model altogether. Despite its simple form the posterior predictive objective
requires the computation of the log partition function of a partially masked sequence $\log{\mathcal{Z}}_{\pi_{t}}$ which does not have a closed-form expression and must be estimated.

Monte Carlo Estimate of $\log{\mathcal{Z}}_{\pi_{t}}$ with DDPP-IS.
A numerical estimate of the log normalization constant can be obtained by utilizing the trick of using the pre-trained model’s denoising posterior $p^{\text{pre}}(\mathbf{x}_{0}|\mathbf{x}_{t})$. Specifically, given $\mathbf{x}_{t}\sim p_{t}(\mathbf{x}_{t})$ we obtain a Monte Carlo estimate of $\log{\mathcal{Z}}_{\pi_{t}}(\mathbf{x}_{t})$ that uses $M$ additional samples from $\mathbf{x}_{0}\sim p_{t}^{\text{pre}}(\mathbf{x}_{0}|\mathbf{x}_{t})$ to estimate the log partition function,

$$ $\displaystyle\log\hat{{\mathcal{Z}}}_{\pi_{t}}(\mathbf{x}_{t})$ $\displaystyle=\log\left(\sum_{\mathbf{x}_{0},\dots\mathbf{x}_{t-1}}p^{\text{ pre}}_{t}(\mathbf{x}_{0},\dots,\mathbf{x}_{t-1}|\mathbf{x}_{t})R(\mathbf{x}_{0 })\right)\approx\log\left(\mathbb{E}_{\mathbf{x}^{\prime}_{0}\sim p^{\text{pre }}_{t}(\mathbf{x}_{0}\mid\mathbf{x}_{t})}[R(\mathbf{x}^{\prime}_{0})]\right).$ $$

Where in the second equality in the first line we used the fact that we can approximately jump to the endpoint of the reverse process directly by using the pretrained model’s denoiser to sample $\mathbf{x}_{0}$. Conveniently, this MC estimate solely requires obtaining a denoised sample from the pre-trained MDM which can be efficiently done as each sample requires a single step as due to the denoising posterior parametrization of an MDM (Eq. 4). We can further improve the estimation of this log normalization constant by leveraging importance sampling (IS) with a proposal distribution $w(\mathbf{x}_{0})$:

$$ $\log\hat{{\mathcal{Z}}}^{\text{IS}}_{\pi_{t}}(\mathbf{x}_{t})=\log\left( \mathbb{E}_{\mathbf{x}_{0}^{\prime}\sim w(\mathbf{x}_{0})}\left[\frac{p^{\text {pre}}_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})R(\mathbf{x}_{0}^{\prime})}{w(\mathbf {x}_{0}^{\prime})}\right]\right)=\log\left(\frac{1}{M}\sum_{j=1}^{M}\left[ \frac{p^{\text{pre}}_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})R(\mathbf{x}^{j}_{0})}{ w(\mathbf{x}_{0}^{j})}\right]\right).$ $$

For the IS estimator above it is easy to verify that the optimal proposal distribution for variance reduction is proportional to the denoising reward-induced target posterior $w^{*}(\mathbf{x}_{0})\propto\pi_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})$. Fortunately, this is precisely the distribution that is approximated by $q_{t,\theta}$ using the posterior predictive objective which motivates the reuse of the finetuned model as a suitable proposal, i.e. $w(\mathbf{x}_{0})=q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})$.

Learning $\log{\mathcal{Z}}_{\pi_{t}}$ with DDPP-LB.
An alternative to using an MC-based estimate for $\log{\mathcal{Z}}_{\pi_{t}}$ is to parameterize the log partition function itself $\log\hat{{\mathcal{Z}}}^{\text{LB}}_{\pi_{t},\theta}$ jointly with the $q_{t,\theta}$ and optimize both using the same posterior predictive objective as first defined in Eq. 11. Operationally, this amounts to including another prediction head for the finetuned MDM model and is cheaper to compute than using an MC-based estimate as we do not require $M$ evaluations of the pre-trained model as in $\log\hat{{\mathcal{Z}}}^{\text{IS}}_{\pi_{t}}(\mathbf{x}_{t})$.

At first glance, it remains unclear whether a parameterized $\log\hat{{\mathcal{Z}}}^{\text{LB}}_{\pi_{t},\theta}$ is a sensible strategy. However, in the particular case where we choose the proposal distribution to be on-policy by using finetuned MDM $w(\mathbf{x}_{0})=q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})$, we can show that the learned log partition function estimate is a lower bound to the importance sampling estimate. This is formalized in the following proposition below.

###### Proposition 1 .

Let $\log\hat{{\mathcal{Z}}}^{\text{IS}}_{\pi_{t}}$ and $\log\hat{{\mathcal{Z}}}^{\text{LB}}_{\pi_{t},\theta}$ be the $M$-sample importance sampling estimate using the proposal $q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})$ and learned approximation to the log partition function respectively. Given a partially masked sample $\mathbf{x}_{t}\sim p_{t}(\mathbf{x}_{t})$ the optimal learned approximation is a lower bound to the importance sampling estimate with a fixed proposal $q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})$ and the following inequality holds:

$$ $\log\hat{{\mathcal{Z}}}^{\text{LB}}_{\pi_{t},\theta}(\mathbf{x}_{t})\leq\log \hat{{\mathcal{Z}}}^{\text{IS}}_{\pi_{t}}(\mathbf{x}_{t}).$ (9) $$

The proof for Eq. 1 is provided in §C.1. We highlight that the lower bound becomes equality at the optimal proposal $q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})\propto\pi_{t}(\mathbf{x}_{0}|
\mathbf{x}_{t})$. Learning $\log\hat{{\mathcal{Z}}}^{\text{LB}}_{\pi_{t},\theta}$ has the benefit of amortization as the same network can be reused for all partially masked samples $\mathbf{x}_{t}\sim p_{t}(\mathbf{x}_{t})$, across all levels of masking. In addition, over the course of training, the learned estimate $\log\hat{{\mathcal{Z}}}^{\text{LB}}_{\pi_{t},\theta}$ becomes a better estimate for the true log partition function. In practice, it suffices to take a single gradient step to optimize $\log\hat{{\mathcal{Z}}}^{\text{LB}}_{\pi_{t},\theta}$ rather than optimizing till convergence. As a result, no additional overhead needs to be incurred, and the learned estimate is averaged over a batch of noisy samples ${\mathcal{B}}=\{\mathbf{x}_{t}^{i}\}^{N}_{i=1}$.

Figure: Algorithm 1 Single-step DDPP-IS and DDPP-LB

###### Proposition 1 .

### 3.2 Single-step posterior sampling with endpoint prediction

The sub-trajectory matching objective used by DDPP-IS and DDPP-LB can be simplified to a faster single-step objective at the cost of paying a discretization error by not using finer-grained trajectory information. Specifically, we note that for MDMs the denoising posterior over endpoints $q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})\approx\text{Cat}(\mathbf{x}_{0};
\mu_{\theta}(\mathbf{x}_{t},t))$ can be approximately computed *without unrolling the sub-trajectory*. This fact also holds for the pre-trained MDM as the model parametrization implies $p^{\text{pre}}_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})\approx\text{Cat}(\mathbf{x}_
{0};\mu(\mathbf{x}_{t},t))$. For the single-step objective we assume the parameterized denoisers exactly match the posteriors. Leveraging this enables us to express the denoising reward-induced target posterior using a simple expression that directly uses the pre-trained model’s denoising posterior $p^{\text{pre}}_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})$ as follows:

$$ $\displaystyle\pi_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})$ $\displaystyle=\frac{p_{t}(\mathbf{x}_{t})}{p_{t}(\mathbf{x}_{t})}\cdot\frac{p_ {t}(\mathbf{x}_{t}|\mathbf{x}_{0})p^{\text{pre}}(\mathbf{x}_{0})R(\mathbf{x}_{ 0})}{\sum_{\mathbf{x}_{0}^{\prime}}p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0}^{\prime })p^{\text{pre}}(\mathbf{x}_{0}^{\prime})R(\mathbf{x}_{0}^{\prime})}=\frac{p^{ \text{pre}}_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})R(\mathbf{x}_{0})}{{\mathcal{Z}} _{\pi_{t}}(\mathbf{x}_{t})}.$ (10) $$

The choice of parameterizing $q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})$ as another MDM offers a prescriptive strategy for sampling from the desired target $\pi_{0}$ by learning to match the denoising reward-induced posterior at the predicted endpoint $\pi_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})$.
This simplifies the expression of DDPP defined over trajectories in Eq. 8 to a single point, namely the predicted endpoint $\mathbf{x}_{0}$ of each MDM. This objective is presented below:

$$ $\displaystyle\mathcal{L}^{\text{PP}}$ $\displaystyle=\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{t}}\left[\left|\left| \log q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})-\underbrace{\log p^{\text{pre }}_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})-\log R(\mathbf{x}_{0})+\log{\mathcal{Z}} _{\pi_{t}}(\mathbf{x}_{t})}_{\log\pi_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})}\right |\right|^{2}_{2}\right].$ (11) $$

As done previously, we can employ any estimation strategy to compute the log partition function Eq. 11. We note in many cases, such as when the sequence length of the trajectory is small to moderate, the single-step objective may be an attractive alternative to the sub-trajectory variants of DDPP.
Algorithm 1 provides a detailed description of the single-step version of DDPP.

### 3.3 DDPP -KL: Posterior prediction via reverse KL minimization

The single-step posterior prediction objective as defined using the loss function ${\mathcal{L}}^{\text{PP}}$ in Eq. 11 requires the estimation of $\log{\mathcal{Z}}^{\text{LB}}_{\pi_{t},\theta}$ which introduces a source of variance in loss estimates that may sub-optimally influence learning dynamics of the fine-tuned model. In settings, where the reward model is differentiable we can bypass computing $\log{\mathcal{Z}}^{\text{LB}}_{\pi_{t},\theta}$ altogether by learning to match the denoising reward-induced posterior under the *reverse* KL divergence. To see this, we define a variational posterior matching problem using the reverse KL divergence that takes the following form:

$$ $\displaystyle{\mathcal{L}}_{t}^{\text{KL}}:=\mathbb{D}_{\mathrm{KL}}(q_{t, \theta}(\mathbf{x}_{0}|\mathbf{x}_{t})p_{t}(\mathbf{x}_{t})||\pi_{t}(\mathbf{x }_{0}|\mathbf{x}_{t})p_{t}(\mathbf{x}_{t})).$ (12) $$

Unlike conventional generative modeling using the reverse KL divergence which solely matches distributions at $t=0$ the problem definition in Eq. 12 defines a series of reverse KL minimization problems through time. In this manner, the reverse KL matches distributions annealed through time and can be used to derive a stochastic regression objective for fine-tuning,

$$ $\displaystyle{\mathcal{L}}^{\text{KL}}$ $\displaystyle=\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{t}}\left[\log q_{t, \theta}(\mathbf{x}_{0}|\mathbf{x}_{t})-\log p^{\text{pre}}_{t}(\mathbf{x}_{0}| \mathbf{x}_{t})-\log R(\mathbf{x}_{0})\right]+C.$ (13) $$

The expectation in Eq. 13, like DDPP-IS and DDPP-LB is taken uniformly with respect to time $t\sim{\mathcal{U}}[0,1]$. However, unlike the previous estimators, clean data needed to compute ${\mathcal{L}}^{\text{KL}}$ is drawn purely on-policy by simulating the fine-tuning model $\mathbf{x}_{0}\sim q_{\theta}(\mathbf{x}_{0})$, which then also allows us to craft a noisy sample using the masking forward process $\mathbf{x}_{t}\sim p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0})$. Additionally,
in Eq. 13 the constant $C=\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{t}}[\log{\mathcal{Z}}_{\pi_{t}}(
\mathbf{x}_{t})]$ does not depend on the $\theta$—and as a result is also independent of the sample $\mathbf{x}_{0}\sim q_{t,\theta}(\mathbf{x}_{0})$. This results in the constant $C$ being zero when computing the gradient of the loss $\nabla_{\theta}{\mathcal{L}}^{\text{KL}}$ and as a result we can safely disregard computing $\log{\mathcal{Z}}_{\pi_{t}}$ entirely.

As samples $\mathbf{x}_{0}$ are procured on-policy to compute the gradient of the loss $\nabla_{\theta}{\mathcal{L}}^{\text{KL}}$ we require backpropagating through the stochastic sampling of $\mathbf{x}_{0}$ which comes from simulating the fine-tuning MDM $q_{t,\theta}(\mathbf{x}_{0})$. Fortunately, we can make use of modern discrete gradient estimators which provide a biased but low variance gradient estimate enabling us to compute ${\mathcal{L}}^{\text{KL}}$. Specifically, we opt to use the scalable 2nd order Reinmax estimator  which estimates the discrete gradient up to second-order terms in a Taylor approximation of the actual gradient. We note that unlike DDPP-IS and DDPP-LB this new loss that minimizes the reverse KL divergence ${\mathcal{L}}^{\text{KL}}$ requires the reward model $R$ to be differentiable and as a result is less broadly applicable than computing ${\mathcal{L}}^{\text{PP}}$. However, in practice, learning can be faster as we make use of the information afforded to us by the gradient $\nabla R$ as well as the fact that the objective does not need to estimate the log partition function.

In appendix §C.2 we provide the exact algorithm Alg. 2 to compute the reverse KL objective. We further show how using a gradient estimator like Reinmax can be used to derive efficient gradient estimation for a more general class of problems of sampling from $\pi_{0}(\mathbf{x}_{0})=R(\mathbf{x}_{0})/{\mathcal{Z}}_{\pi_{0}}$, as well as the main fine-tuning setting for matching the denoising reward-induced posterior as defined in  Eq. 10.

## 4 Experiments

**Table 1: Overview of posterior sampling methods**
| Method | Model calls / inf. step | Model calls / train step | Sim. Free |
| --- | --- | --- | --- |
| SVDD | $N$ | — | ✓ |
| RTB | 1 | $T$ | ✗ |
| DDPP-KL | 1 | 1 | ✓ |
| DDPP-IS | 1 | $M$ | ✓ |
| DDPP-LB | 1 | 1 | ✓ |

We investigate the application of DDPP to a variety of discrete generative modeling settings. We provide the full experimental details in §D and present our main experimental results next.

Baselines.
Throughout our experiments, we rely on four principal baselines (compared in Table 1) in sampling from the pre-trained MDM model, Best-of-N sampling , Relative Trajectory Balance (RTB) , and SVDD  which is a concurrent inference time technique for steering diffusion models. Best-of-N represents a computationally expensive baseline but is guaranteed to produce samples from $\pi_{0}$, as such we use this as an upper bound on performance in terms of reward obtained as $N\to\infty$ . RTB is a GFlowNet  that requires simulating the entire diffusion trajectory. In Table 1 we illustrate the computational differences between DDPP and baselines.

### 4.1 Synthetic Experiments

**Table 2: Fine-tuning to produce only even digits on binarized MNIST. We report the mean performance over $3$ runs for the $\log R$, FLD, and BPD metrics.**
| Algorithm $\downarrow$ Metric $\rightarrow$ | $\log R(\mathbf{x}_{0})\uparrow$ | FLD $\downarrow$ | BPD $\downarrow$ |
| --- | --- | --- | --- |
| Base Model | -26.90 $\pm$ — | 33.89 $\pm$ — | 0.130 $\pm$ — |
| SVDD | -0.03 $\pm$ 0.01 | 34.19 $\pm$ 0.95 | — |
| RTB | -18.66 $\pm$ 2.45 | 45.97 $\pm$ 0.89 | 0.128 $\pm$ 0.000 |
| DDPP-IS (ours) | -5.14 $\pm$ 1.24 | 33.11 $\pm$ 0.71 | 0.130 $\pm$ 0.000 |
| DDPP-LB (ours) | -5.68 $\pm$ 0.34 | 33.76 $\pm$ 0.90 | 0.128 $\pm$ 0.000 |
| DDPP-KL (ours) | -3.13 $\pm$ 0.06 | 31.75 $\pm$ 0.51 | 0.129 $\pm$ 0.000 |

We consider a synthetic experimental task in learning to sample from a target distribution defined on a $2\text{D}$ discrete grid and finetuning an MDM on binarized MNIST. We use this synthetic setting to test all variations of DDPP along with our chosen baselines and present visual qualitative results in Figure 1,
Figure 4 and quantitative results in Table 2.

Grid Experiment. We define a prior density $p_{0}^{\text{pre}}$ over the discrete 2-dimensional, $128\times 128$ grid, as showcased in Figure 1(a) where the probability mass corresponding to each point $\mathbf{x}_{0}$ is on if the color is yellow. The goal is to sample from the product distribution as outlined in Equation 1, which in this case is defined to drop the modes in $p_{0}^{\text{pre}}$ which are at the top half of the grid, as
visualized in Figure 1(b).
These results show that all three variants of DDPP effectively learn to sample from this target.

MNIST.
We finetune MDMs to generate even MNIST digits. As observed in Table 2 we find that all three variants of DDPP match or outperform the base pre-trained model and RTB in all metrics, with DDPP-KL being the best. In comparison to the concurrent work of SVDD, we find that it outperforms DDPP in average $\log R$ but is worse in sample-based metrics such as class conditional FLD  which measures the overall quality, diversity and generalizability of generated samples and class conditional BPD. We further report generated samples in Figure 4 located in §D.3.

Figure: (a) Prior Density
Refer to caption: extracted/5916875/Plots/grid/density_prior.png

### 4.2 Pixel-level image modelling

We next consider the task of fine-tuning MDMs on order-agnostic image data. More precisely, we discretize pixels in $64\times 64$ downsampled images from the CelebA dataset  to a vocabulary of $256$ tokens. As there are no publicly available pre-trained MDM models we train our own MDM by modeling the raw pixel space and achieve $1.85$ bits-per-dim (BPD) on CelebA. Our full experimental setup is outlined in §D.3.
For fine-tuning, we consider steering a pre-trained MDM using DDPP-LB as it is the most computationally cheap method with a class-conditional reward based on an auxiliary classifier. Specifically, we steer the generative model to generate human faces with blond hair. For quantitative metrics, we report the mean log reward obtained, and BPD in Figure 3 as well as selected generated samples. Our quantitative results show that our proposed variant DDPP-LB significantly outperforms all other baselines in obtaining the highest reward. We also observe DDPP obtains BPD values that are within the range of the base model while being worse than RTB. These results are further substantiated by the visual samples where we find that generated samples do produce the highest fidelity faces with blond hair, matching our fine-tuning goal.

### 4.3 Protein sequence modelling

**Table 3: *In-silico* results for protein generation tasks. We report the mean result for a metric with standard deviation across three seeds. DDPP-LB performs well across designability metrics (pLDDT and pTM) while simultaneously performing best on task specific metrics ($\beta$-sheet % and TM-Score).**
|  | High $\beta$-sheet-content protein generation | Protein shrinking |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | $\beta$-sheet % $\uparrow$ | pLDDT $\uparrow$ | pTM $\uparrow$ | $\log R(x_{0})\uparrow$ | SS-KL $\downarrow$ | TM-Score $\uparrow$ | pLDDT $\uparrow$ | pTM $\uparrow$ | $\log R(x_{0})$ $\uparrow$ |
| Base Model | 0.111 $\pm$ 0.121 | 0.724 $\pm$ 0.144 | 0.584 $\pm$ 0.226 | 2.070 $\pm$ 0.749 | 3.040 $\pm$ 3.043 | 0.245 $\pm$ 0.058 | 0.724 $\pm$ 0.144 | 0.584 $\pm$ 0.226 | 0.490 $\pm$ 0.116 |
| Best-of-10 | 0.280 $\pm$ 0.093 | 0.812 $\pm$ 0.033 | 0.786 $\pm$ 0.035 | 3.212 $\pm$ 0.371 | 1.621 $\pm$ 2.804 | 0.345 $\pm$ 0.049 | 0.786 $\pm$ 0.023 | 0.737 $\pm$ 0.097 | 0.690 $\pm$ 0.098 |
| SVDD | 0.114 $\pm$ 0.148 | 0.484 $\pm$ 0.134 | 0.349 $\pm$ 0.174 | 1.669 $\pm$ 0.907 | 3.353 $\pm$ 2.913 | 0.337 $\pm$ 0.042 | 0.492 $\pm$ 0.131 | 0.368 $\pm$ 0.171 | 0.673 $\pm$ 0.083 |
| RTB | 0.319 $\pm$ 0.218 | 0.806 $\pm$ 0.059 | 0.767 $\pm$ 0.101 | 3.386 $\pm$ 1.061 | 2.193 $\pm$ 2.724 | 0.290 $\pm$ 0.056 | 0.797 $\pm$ 0.056 | 0.747 $\pm$ 0.093 | 0.581 $\pm$ 0.112 |
| DDPP-LB | 0.436 $\pm$ 0.037 | 0.897 $\pm$ 0.027 | 0.806 $\pm$ 0.029 | 3.703 $\pm$ 0.186 | 0.640 $\pm$ 1.793 | 0.361 $\pm$ 0.047 | 0.768 $\pm$ 0.048 | 0.747 $\pm$ 0.063 | 0.722 $\pm$ 0.094 |

Figure: Figure 2: Left: SDS-PAGE of elution fractions from histidine tag purification of DDPP-designed protein constructs and positive controls following Coomassie blue staining. All DDPP-designed constructs are between 7.8-8.3 kDa, though run slightly higher than their predicted molecular weight. Predicted molecular weights of positive controls 5KPH, 1QYS, 1UBQ, and 1BTB are 10 kDa, 12 kDa, 8.5 kDa, and 10 kDa, respectively. Recombinant protein bands for MDM-designed sequences are indicated with red arrows and relevant ladder references are labeled with their molecular weight. Middle: Folded structures generated by $\beta$-sheet fine-tuning with DDPP. Right: Distribution of $\beta$-sheets generated by each method.
Refer to caption: x1.png

Task description. We next apply DDPP to generate high-quality protein sequences by fine-tuning discrete diffusion protein language models (DPLM) . Specifically, we address two experimentally relevant tasks where vanilla DPLMs underperform. We outline exact reward functions and experimental setup in §D.2. First, we fine-tune DPLM to generate soluble protein sequences with high $\beta$-sheet content. The second task, protein shrinking, involves miniaturizing known proteins by generating shorter sequences that preserve key structural features, using the TM-align score as the reward metric . We evaluate performance by measuring designability metrics (ESMFold pLDDT and pTM) as well as task-specific metrics ($\beta$-sheet percent and TM-Score). We also provide wet-lab validation for our best designs in the designable $\beta$-sheet task. We provide full a deeper description of evaluation metrics and experimental setup in §D.2. Finally, as ESMFold is itself expensive to query and, in particular, non-differentiable we test our fastest method—DDPP-LB.

Main results.
In-silico validation shows that DDPP-LB outperforms all baselines for the designable $\beta$-sheet task, generating better sequences across all metrics. In particular DDPP achieves a significantly higher $\beta$-sheet percentage than baseline methods while maintaining high designability as measured by ESMFold (namely, high pLDDT and pTM). We further observe that for the miniaturization task, DDPP-LB outperforms all baselines in shrinking ribonuclease proteins, removing 34 residues while maintaining high structural similarity (lowest SS-KL of 0.64 and highest TM-Score of 0.361), and high structural quality with high pTM and competitive pLDDT. This demonstrates DDPP-LB’s effectiveness in generating compact yet structurally faithful proteins.

Experimental validation.
We selected 6 designs from DDPP-finetuned DPLM for wet-lab validation, based on AlphaFold2 pLDDT/pTM scores. Sequences and structures were clustered using MMseqs and Foldseek , with two representative sequences selected from each cluster. 4 positive controls consisting of two previously validated de novo designed proteins (PDB: 5KPH, 1QYS) and two other stable proteins, ubiquitin and Barstar (PDB: 1UBQ, 1BTB) were included as a comparison.
We expressed the designed proteins, including the controls in E. coli, and purified them using histidine-tag purification, after which we assessed expression level and purity via SDS-PAGE, followed by Coomassie staining. Our results demonstrate strong overexpression and efficient purification of the two previously validated de novo controls and moderate overexpression of ubiquitin and barstar controls (Figure 2). Purified protein can also be observed for four out of the six DDPP-derived constructs, though with comparatively lower yields than the positive controls (Figure 2). One potential cause of these relatively low yields may be the sizeable accumulation of DDPP-derived proteins in the insoluble fraction of the cell lysate. As such, it is likely that further optimization of the expression and purification methods (e.g., longer induction time or lower induction temperatures) may lead to significant improvements to overall soluble yields.

### 4.4 Text

Task description.
We consider two text tasks: (i) toxic story generation using the Tinystories dataset , and (ii) product review generation using Amazon data .
For both tasks, we start by fine-tuning a pre-trained MDM model in a supervised fine-tuning manner on both datasets before running online fine-tuning. As reward models, we use RoBERTa fine-tuned for toxicity classification, and BERT , fine-tuned for Amazon review sentiment analysis, respectively. Our experiments aim to demonstrate our method’s ability to induce behaviors that are uncommon in the base pre-trained model, specifically in generating toxic content in product reviews. Full experimental details are provided in Appendix §D.4.

Main results. In Table 4 we report the average log reward as well as perplexity (Gen PPL) of the generated samples as measured by GPT-2 . We find that DDPP-LB is the most effective variant of DDPP and achieves significantly higher log reward compared to SVDD and RTB for both tasks. We further observe that all methods achieve comparable Gen PPL suggesting that generated responses are fluent; however, samples from DDPP-LB adheres better to the task specification. We refer to §D.4.1 and §D.4.2 for generated samples from DDPP.

Figure: Figure 3: Left: Results for discrete image modeling over raw pixel values on CelebA ($64\times 64$). We report the mean performance of DDPP and baselines separated into inference-based (top) and amortized (bottom) over $3$ runs for the $\log R$ and class-BPD metrics. Right: Generated samples from Base, SVDD, RTB, and DDPP-LB.
Refer to caption: extracted/5916875/Plots/celeba_base_svdd_rtb_ddpp.png

**Table 4: Text experiments with log reward and Gen PPL results averaged over $3$. As Best of 10 draws samples directly from $p_{0}^{\text{pre}}(\mathbf{x}_{0})$ we instead bold the fine-tuning method whose Gen PPL is lowest.**
| Dataset $\rightarrow$ | Tinystories | Amazon reviews |  |  |
| --- | --- | --- | --- | --- |
| Algorithm $\downarrow$ Metric $\rightarrow$ | $\log R(\mathbf{x}_{0})\uparrow$ | Gen PPL $\downarrow$ | $\log R(\mathbf{x}_{0})\uparrow$ | Gen PPL $\downarrow$ |
| Best of $10^{*}$ | 93.25 $\pm$ 0.17 | 15.94 $\pm$ 0.03 | -103.05 $\pm$ 0.25 | 124.45 $\pm$ 1.02 |
| SVDD | 146.95 $\pm$ 1.08 | 20.35 $\pm$ 0.03 | -27.48 $\pm$ 10.91 | 165.86 $\pm$ 1.22 |
| RTB | 107.83 $\pm$ 3.08 | 18.53 $\pm$ 0.55 | -35.22 $\pm$ 16.03 | 160.54 $\pm$ 12.19 |
| DDPP-IS (ours) | 163.45 $\pm$ 7.06 | 20.15 $\pm$ 0.30 | 105.16 $\pm$ 2.41 | 152.85 $\pm$ 1.64 |
| DDPP-LB (ours) | 205.76 $\pm$ 3.88 | 19.60 $\pm$ 0.69 | 152.08 $\pm$ 34.01 | 167.25 $\pm$ 27.33 |

## 5 Related Works

Discrete diffusion.
The prevailing paradigms for diffusion over discrete spaces can be broadly categorized into 1.) continuous diffusion in a latent or reparametrized space by first transforming the initial discrete data , and
2.) defining diffusion using discrete analogs of score approximation . The latter approach can also be described using the theoretical framework of Continuous-time Markov Chains (CTMC) . Closest to our setting we consider a specific instantiation of discrete diffusion that simplifies the CTMC framework by using a masked forward process .

Finetuning as sampling. The task of fine-tuning generative models under reward models can be viewed as a sampling problem and encompasses conventional RLHF . A simple but expensive method to sample from the reward-induced Bayesian posterior distribution is best of $N$ sampling , which provably samples from the correct distribution as the number of samples from the base pre-trained model grows, $N\to\infty$ . Alternatively, the sampling perspective has been explored in the discrete setting to fine-tune autoregressive models , and diffusion models . Finally, inference time techniques represent the most prominent approach to conditional sampling .

## 6 Conclusion

In this paper, we present Discrete Denoising Posterior Prediction a novel framework to steer Masked Discrete Diffusion Models by viewing it as a problem of sampling from a Bayesian posterior. We introduced three concrete training strategies to instantiate our framework in DDPP-IS, DDPP-LB, and DDPP-KL and apply them to modeling synthetic data, pixel-level image modeling, fine-tuning protein MDMs to increase secondary structure diversity, and steering MDMs on language to match human sentiment. We find that DDPP not only is able to optimize an amortized sampler to closely match the reward-induced Bayesian posterior but it has a good agreement in other sample quality metrics—without severely compromising generated sample quality. An interesting direction for future work is to understand how to balance optimization of DDPP-LB and strategies to selecting $\gamma$.

## 7 Contributions statement

J.R. conceived of the initial simulation-free fine-tuning framework, while M.H. derived the final version of the objective that can be written as a log ratio of denoisers. J.R. drove the development of code for all experimental settings, M.H. and A.T. drove the completion of image experiments, J.R. and Z.P. drove the completion of protein and text experiments, and M.H. drove the completion of toy experiments. Z.P. led TM-score protein experiments. J.R., C.L., Z.Q., and P.C. conceived of the $\beta$-sheet protein task and wet lab experimental design. Z.Q. ran all wet lab experiments. N.D. guided text experiments. S.M. helped in setting up the baselines and brainstorming about the estimators. A.J.B. drove the writing of the paper with help from M.H.. M.B., Y.B., A.T., S.M. and A.J.B. guided the project from the machine learning side, while P.C. guided it from the wet lab side. A.J.B. with help from M.H. designed the discrete (Reinmax) gradient estimator version of the objective function. J.R., A.J.B., and A.T. were responsible for the overall organization of the project.

## 8 Acknowledgements

The authors would like to thank Kolya Malkin, Moksh Jain, Emily Jin, Katarina Petrovic, Scott Le Roux, Ahmed Elhag, Xingyue Huang, and Vignesh Ram Somnath for their useful comments on early versions of this manuscript.
AJB is partially supported by an NSERC Post-doc fellowship.
This research is partially supported by EPSRC Turing AI World-Leading Research Fellowship No. EP/X040062/1 and EPSRC AI Hub on Mathematical Foundations of Intelligence: An "Erlangen Programme" for AI No. EP/Y028872/1

The authors acknowledge funding from UNIQUE, CIFAR, NSERC, Intel, Samsung, and Dreamfold. The research was enabled in part by computational resources provided by the Digital Research Alliance of Canada ([https://alliancecan.ca](https://alliancecan.ca)), Mila ([https://mila.quebec](https://mila.quebec)), and NVIDIA.

## 9 Reproducibility statement

We take the following steps to enhance the reproducibility of our work. In particular, all of our theoretical results include full proofs which are presented in §C. To assist in the reproducibility of our empirical findings we provide precise experimental details such as algorithmic descriptions of all variants of DDPP in Algorithm 1 and Algorithm 2. We further provide architectural choices, training details, and hyperparameters for all datasets and tasks in §D.

## Appendix A Broader Impact

Our proposed Discrete Denoising Posterior Prediction is a tailored approach to steering and fine-tuning Masked Diffusion Models. At present, MDMs are an emergent category of discrete generative models that have general-purpose modeling capabilities in a variety of domains including language modeling, sequence-based drug design, and discrete modeling of graphs. Consequently, we believe DDPP has potential use in various practical use cases. For instance, like current RLHF techniques applied to modern autoregressive LLMs, future scaled MDMs on text datasets might be tuned to promote harmful behavior and toxic content. Moreover, applying Discrete Denoising Posterior Prediction in drug design use cases has the potential to create in-silico sample of protein sequences that may have biologically potent negative externalities. We do, however, make the distinction that such a risk is speculative at this stage given the large complexities of translating in-silico designs to actual synthesized biomolecules. As a result, we encourage practitioners who seek to fine-tune MDMs using DDPP to exercise due caution when applying our proposed techniques to actual use cases.

Ethical statement.
As part of qualitatively evaluating DDPP, this paper includes generated samples of text. We highlight that the set of examples may contain potentially disturbing, harmful, or upsetting examples, covering a variety of sensitive topics like discriminatory language, descriptions of harm, and misinformation, among other high-risk categories. Its primary purpose is to advance research in understanding the impact of DDPP from a more interpretable lens. It is not advised to train future MDMs on such generated samples in order to prevent further propagation of undesirable content and behaviors.

## Appendix B Additional Related Work

Sampling proportional to energy. Our approach can be closely linked to learning to sample proportional to a target probability, as in our setup we aim to approximate sampling proportional to the energy $p_{t}^{\text{pre}}(\cdot|\mathbf{x}_{t})R(\cdot)$ for any point $\mathbf{x}_{t}$ at any time $t$. This has been an avenue of research for a number of works in continuous time , in Bayesian posterior inference where the energy is defined by the product of likelihood and prior , as well as posterior inference in settings where we even do not have access to energy function but only to a simulator .

## Appendix C Theoretical results

### C.1 Proof of Proposition 1

Before proving proposition 1 we first prove a useful Lemma that states the optimal log partition function $\log\hat{{\mathcal{Z}}}_{\pi_{t}}(\mathbf{x}_{t})$ which is the learning goal for a parameterized approach $\log\hat{{\mathcal{Z}}}_{\pi_{t},\theta}(\mathbf{x}_{t})$.

###### Lemma 1 .

Given a sample $\mathbf{x}_{t}\sim p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0})$ and the denoising posterior distribution $q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})$, a local minimizer for estimate for the log partition function $\log\hat{{\mathcal{Z}}}_{\pi_{t}}$ using $N$ samples from $\mathbf{x}^{i}_{0}\sim q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})$ is given by:

$$ $\log{\mathcal{Z}}^{*}_{\pi_{t}}=\frac{1}{N}\sum_{i=1}^{N}\log\left(\frac{p_{t} (\mathbf{x}^{i}_{0}|\mathbf{x}_{t})R(\mathbf{x}^{i}_{0})}{q_{t,\theta}(\mathbf {x}^{i}_{0}|\mathbf{x}_{t})}\right).$ (14) $$

###### Proof.

By definition the log partition function is a constant, let that constant be $\log{\mathcal{Z}}_{\pi_{t}}(\mathbf{x}_{t})=C$. Then the loss in Eq. 11 is a quadratic in $C$,

$$ ${\mathcal{L}}=\mathbb{E}_{\mathbf{x}_{0}\sim r(\mathbf{x}_{0})}\left[||\log q_ {t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})+C-\log p_{t}(\mathbf{x}_{0}|\mathbf{ x}_{t})-\log R(\mathbf{x}_{0})||_{2}^{2}\right]$ (15) $$

For a batch of $N$ samples of $\mathbf{x}^{i}_{0}\sim q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})$, we find a locally optimal constant $C$ (local minima) by taking the gradient of Eq. 15 and setting it 0 0. In more detail we have,

$$ $\displaystyle 0$ $\displaystyle=\nabla_{C}\frac{1}{N}\sum_{i}^{N}(\log q_{t,\theta}(\mathbf{x}_{ 0}|\mathbf{x}_{t})+C-\log p_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})-\log R(\mathbf{ x}_{0}))^{2}$ (16) $\displaystyle 0$ $\displaystyle=\frac{2}{N}\sum_{i}^{N}\left(\log q_{t,\theta}(\mathbf{x}^{i}_{0 }|\mathbf{x}_{t})+C-\log p_{t}(\mathbf{x}^{i}_{0}|\mathbf{x}_{t})-\log R( \mathbf{x}^{i}_{0})\right)$ (17) $\displaystyle 0$ $\displaystyle=2C+\frac{2}{N}\sum^{N}_{i}\log q_{t,\theta}(\mathbf{x}^{i}_{0}| \mathbf{x}_{t})-\log p_{t}(\mathbf{x}^{i}_{0}|\mathbf{x}_{t})-\log R(\mathbf{x }^{i}_{0})$ (18) $\displaystyle 0$ $\displaystyle=C+\frac{1}{N}\sum^{N}_{i}\log\left(\frac{q_{t,\theta}(\mathbf{x} ^{i}_{0}|\mathbf{x}_{t})}{p_{t}(\mathbf{x}^{i}_{0}|\mathbf{x}_{t})R(\mathbf{x} ^{i}_{0})}\right)$ (19) $\displaystyle C$ $\displaystyle=\frac{1}{N}\sum^{N}_{i}\log\left(\frac{p_{t}(\mathbf{x}^{i}_{0}| \mathbf{x}_{t})R(\mathbf{x}^{i}_{0})}{q_{t,\theta}(\mathbf{x}^{i}_{0}|\mathbf{ x}_{t})}\right).$ (20) $$

∎

Using Lemma 1 we now prove Proposition 1, stated again below for convenience.

See 1

###### Proof.

We optimize $\log\hat{{\mathcal{Z}}}^{\text{LB}}_{\pi_{t},\theta}(\mathbf{x}_{t})$ using the loss defined in Eq. 11. Using Lemma 1 we know the analytic expression for the locally optimal estimate is given by $\log{\mathcal{Z}}^{*}_{\pi_{t}}(\mathbf{x}_{t})$. Plugging this into the definition of the log partition function we get,

$$ $\displaystyle\log\hat{{\mathcal{Z}}}^{\text{LB}}_{\pi_{t},\theta}(\mathbf{x}_{ t})$ $\displaystyle=\mathbb{E}_{\mathbf{x}_{0}\sim q_{t,\theta}(\mathbf{x}_{0}| \mathbf{x}_{t})}\left[\log\left(\frac{p_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})R( \mathbf{x}_{0})}{q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})}\right)\right]$ (21) $\displaystyle\leq\log\mathbb{E}_{q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})} \left[\frac{p_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})R(\mathbf{x}_{0})}{q_{t,\theta }(\mathbf{x}_{0}|\mathbf{x}_{t})}\right]$ (22) $\displaystyle=\log\hat{{\mathcal{Z}}}^{\text{IS}}_{\pi_{t}}(\mathbf{x}_{t})$ (23) $$

The lower bound turns into equality at the optimal proposal $q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})\propto p_{t}(\mathbf{x}_{0}|
\mathbf{x}_{t})R(\mathbf{x}_{0})$.

∎

###### Lemma 1 .

###### Proof.

###### Proof.

### C.2 Estimating DDPP -KL with Reinmax

We first provide an algorithmic description below of training using DDPP-KL. We first highlight how the reverse KL objective can be applied to a more general setting beyond just fine-tuning before turning to the exact setting of the main paper.

Figure: Algorithm 2 DDPP-KL

Non-finetuning Case.
In this appendix, we study the Reinmax gradient estimator for the general problem of sampling from the following distribution:

$$ $\pi_{0}(\mathbf{x}_{0})\propto\frac{R(\mathbf{x}_{0})}{{\mathcal{Z}}}.$ (24) $$

Gradient of ${\mathcal{L}}^{\text{KL}}$.
We can decompose the gradient into the following terms due to the linearity of expectations:

$$ $\displaystyle{\mathcal{L}}^{\text{KL}}_{t}$ $\displaystyle=\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{t}}\left[\log q_{t, \theta}(\mathbf{x}_{0}|\mathbf{x}_{t})\right]-\mathbb{E}_{t,\mathbf{x}_{0}, \mathbf{x}_{t}}\left[\log\pi_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})\right]$ $\displaystyle={\mathcal{L}}^{1}_{t}+{\mathcal{L}}^{2}_{t}.$ (25) $$

We again highlight the fact that the expectation is taken using the following distributions $t,\mathbf{x}_{0},\mathbf{x}_{t}\sim{\mathcal{U}}[0,1],q(\mathbf{x}_{0}),p_{t}(
\mathbf{x}_{t}|\mathbf{x}_{0})$. As a result, $\mathbf{x}_{0}$ is drawn on-policy and is a stochastic variable that needs gradient estimation since $q_{\theta}$ is the parameterized distribution. Furthermore, all terms that use *this sample $\mathbf{x}_{0}$* inside the expectation are affected by this gradient computation.

Taking the gradient of each term respectively. The gradient of of ${\mathcal{L}}^{1}_{t}$ is:

$$ $\displaystyle\nabla_{\theta}{\mathcal{L}}^{1}_{t}$ $\displaystyle=\nabla_{\theta}\left(\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{t} }\left[\log q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})\right]\right)$ $\displaystyle\approx\mathbb{E}_{t,\mathbf{x}_{0}\sim q_{\theta}(\mathbf{x}_{0} ),\mathbf{x}_{t}\sim p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0})}\left[\nabla^{\text{ Rein-Max}}\circ\left(\log q_{t,\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})\right) \right].$ (26) $$

The gradient of of ${\mathcal{L}}^{2}_{t}$ is:

$$ $\displaystyle\nabla_{\theta}{\mathcal{L}}^{2}_{t}$ $\displaystyle=\nabla_{\theta}\left(\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{t} }\left[\log\pi_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})\right]\right)$ $\displaystyle=\nabla_{\theta}\left(\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{t} }\left[-\log p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0})-\log\pi_{0}(\mathbf{x}_{0})+ \log\pi_{t}(\mathbf{x}_{t})\right]\right)$ $\displaystyle=\nabla_{\theta}\left(\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{t} }\left[-\log p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0})-\log R(\mathbf{x}_{0})+\log \left(\sum_{\mathbf{x}_{0}^{\prime}}\pi_{t}(\mathbf{x}_{t}|\mathbf{x}_{0}^{ \prime})R(\mathbf{x}_{0}^{\prime})\right)\right]\right).$ (27) $$

To use the Reinmax gradient estimator we must compute $\partial f(z)/\partial z$, where $f$ is the function inside the expectation $\mathbb{E}_{z}[f(z)]$. We now make use of the following facts:

- (F1)
Analytic expression of $\nabla_{x_{0}}\log p_{t}(x_{t}|x_{0})$. For simplicity of presentation, we focus on a single token $x^{i}_{0}$ in a sequence but the result remains true for the entire sequence $\mathbf{x}_{0}$.
Recall in the discrete setting of masked diffusion models $p_{t}=\text{Cat}(x_{0};\bar{Q}_{t}x_{t})$, which allows us to write:
$\displaystyle\nabla_{x^{i}_{0}}\log p_{t}(x^{i}_{t}|x^{i}_{0})$
$\displaystyle=\frac{\nabla_{x^{i}_{0}}p_{t}(x^{i}_{t}|x^{i}_{0})}{p_{t}(x^{i}_
{t}|x^{i}_{0})}$
(28)
$\displaystyle=\frac{\nabla_{x^{i}_{0}}\text{Cat}(x^{i}_{0};\bar{Q}_{t}x^{i}_{t
})}{\text{Cat}(x^{i}_{0};\bar{Q}_{t}x^{i}_{t})}$
(29)
$\displaystyle=\frac{\nabla_{x^{i}_{0}}(x^{i,T}_{0}\bar{Q}_{t}x^{i}_{t})}{x^{i,
T}_{0}\bar{Q}_{t}x^{i}_{t}}$
(30)
$\displaystyle=\frac{\nabla_{x^{i}_{0}}(\alpha_{t}\langle x^{i}_{t},x^{i}_{0}
\rangle+(1-\alpha_{t})\langle x^{i}_{t},e_{m}\rangle)}{\alpha_{t}\langle x^{i}
_{t},x^{i}_{0}\rangle+(1-\alpha_{t})\langle x^{i}_{t},e_{m}\rangle}$
(31)
$\displaystyle=\frac{\alpha_{t}x^{i}_{t}}{\alpha_{t}\langle x^{i}_{t},x^{i}_{0}
\rangle+(1-\alpha_{t})\langle x^{i}_{t},e_{m}\rangle}.$
(32)
- (F2)
Differentiability of the reward $\nabla_{\mathbf{x}_{0}}R(\mathbf{x}_{0})$. If we assume the reward is differentiable we can exploit the same trick to write:
$\displaystyle\nabla_{\mathbf{x}_{0}}\log R(\mathbf{x}_{0})=\frac{\nabla_{
\mathbf{x}_{0}}R(\mathbf{x}_{0})}{R(\mathbf{x}_{0})}.$
(33)

Note that the final term in Eq. 27 does not depend on the realization of the sample $\mathbf{x}_{0}\sim q(\mathbf{x}_{0}|\mathbf{x}_{t})$ and thus its gradient in Rein-max is 0 0. This enables us to write the approximate gradient as:

$$ $\displaystyle\nabla_{\theta}{\mathcal{L}}^{2}_{t}$ $\displaystyle\approx\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{t}}[\nabla^{\text {Reinmax}}\circ\left(-\log p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0})-\log R(\mathbf {x}_{0})\right)]$ $\displaystyle=\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{t}}\left[\left(-\sum_{i }^{N}\frac{\alpha_{t}x^{i}_{t}}{\alpha_{t}\langle x^{i}_{t},x^{i}_{0}\rangle+( 1-\alpha_{t})\langle x^{i}_{t},e_{m}\rangle}-\frac{\nabla^{\text{Reinmax}}_{ \mathbf{x}_{0}}R(\mathbf{x}_{0})}{R(\mathbf{x}_{0})}\right)\right].$ (34) $$

The first term in the equation has a closed-form expression for the gradient but is still a stochastic gradient since it depends on $\mathbf{x}_{0}\sim q_{\theta}(\mathbf{x}_{0})$.

Finetuning Case.
In the fine-tuning setting we aim to sample from the following Bayesian posterior:

$$ $\pi_{0}(\mathbf{x}_{0})\propto\frac{p^{\text{pre}}_{0}(\mathbf{x}_{0})R( \mathbf{x}_{0})}{{\mathcal{Z}}}$ (35) $$

For MDMs the likelihood under the model $p_{0}^{\text{pre}}(\mathbf{x}_{0})$ is intractable to evaluate and leads to a modified objective for gradient estimation with Reinmax in ${\mathcal{L}}^{\text{KL}}_{t}$ in Eq. 25:

$$ $\displaystyle\nabla_{\theta}{\mathcal{L}}^{2}_{t}$ $\displaystyle=\nabla_{\theta}\left(\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{t} }\left[\log\pi_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})\right]\right)$ $\displaystyle=\nabla_{\theta}\left(\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{t} }\left[-\log p^{\text{pre}}_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})-\log R(\mathbf{ x}_{0})+\log{\mathcal{Z}}_{\pi_{t}}(\mathbf{x}_{t})\right]\right)$ $\displaystyle=\nabla_{\theta}\left(\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{t} }\left[-\log p^{\text{pre}}_{t}(\mathbf{x}_{0}|\mathbf{x}_{t})-\log R(\mathbf{ x}_{0})+\log\left(\mathbb{E}_{\mathbf{x}^{\prime}_{0}\sim p^{\text{pre}}_{t}( \mathbf{x}_{0}\mid\mathbf{x}_{t})}[R(\mathbf{x}^{\prime}_{0})]\right)\right] \right).$ (36) $$

Note that in the equation above we can evaluate the log partition function using samples drawn from the denoising posterior of the pre-trained model $\mathbf{x}_{0}^{\prime}\sim p_{t}^{\text{pre}}(\mathbf{x}_{0}|\mathbf{x}_{t})$ and *not* the on-policy samples $\mathbf{x}_{0}\sim q_{\theta}(\mathbf{x}_{0})$. Thus this term is a constant when we compute the gradient. Thus we have,

$$ $\displaystyle\nabla_{\theta}{\mathcal{L}}^{2}_{t}$ $\displaystyle\approx\nabla^{\text{Reinmax}}\circ\left(\mathbb{E}_{t,\mathbf{x} _{0},\mathbf{x}_{t}}\left[-\log p^{\text{pre}}_{t}(\mathbf{x}_{0}|\mathbf{x}_{ t})-\log R(\mathbf{x}_{0})\right]\right).$ (37) $$

### C.3 Equivalence of sub-trajectory objectives

In this appendix, we detail how to compute an efficient approximation of the loss function that is inspired by the KL divergence between sub-trajectories as found in the GFlowNet literature but adapted for MDMs.

Consider the trajectory of a sequence: $\tau(\mathbf{x}_{0:1}):=\mathbf{x}_{1}\to\dots\to\mathbf{x}_{t}\to\mathbf{x}_{
t-1}\to\dots\mathbf{x}_{0}$. We seek to minimize the joint distribution over the (sub)-trajectories conditioned on a partially masked sample $\mathbf{x}_{t}$:

$$ $q_{\theta}(\mathbf{x}_{0},\dots,\mathbf{x}_{t-1}|\mathbf{x}_{t},\mu_{\theta}( \mathbf{x}_{t},t))p_{t}(\mathbf{x}_{t})=\pi_{t}(\mathbf{x}_{0},\dots,\mathbf{x }_{t-1}|\mathbf{x}_{t})p(\mathbf{x}_{t}).$ (38) $$

Here $\pi_{t}(\mathbf{x}_{1},\dots,\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})$ is defined as,

$$ $\displaystyle\pi_{t}(\mathbf{x}_{0},\dots,\mathbf{x}_{t-1}|\mathbf{x}_{t},\mu_ {\theta}(\mathbf{x}_{t},t))$ $\displaystyle=\frac{p_{t}^{\text{pre}}(\mathbf{x}_{0},\dots,\mathbf{x}_{t-1}| \mathbf{x}_{t})R(\mathbf{x}_{0})}{{\mathcal{Z}}_{\pi_{t}}(\mathbf{x}_{t})}$ (39) $\displaystyle=\frac{\prod^{t}_{j=1}p_{t}^{\text{pre}}(\mathbf{x}_{j-1}|\mathbf {x}_{j},\hat{\mathbf{x}}^{\text{pre}}_{0})R(\mathbf{x}_{0})}{{\mathcal{Z}}_{ \pi_{t}}(\mathbf{x}_{t})}$ (40) $$

We minimize the following KL divergence,

$$ $\mathbb{D}_{\mathrm{KL}}(q_{\theta}(\mathbf{x}_{0},\dots,\mathbf{x}_{t-1}| \mathbf{x}_{t},\hat{\mathbf{x}}_{0})p_{t}(\mathbf{x}_{t})||\pi_{t}(\mathbf{x}_ {0},\dots,\mathbf{x}_{t-1}|\mathbf{x}_{t})p(\mathbf{x}_{t})).$ (41) $$

Here we used the convention that $\hat{\mathbf{x}}_{0}=\mu_{\theta}(\mathbf{x}_{t},t)$ and $\hat{\mathbf{x}}^{\text{pre}}_{0}=\mu^{\text{pre}}(\mathbf{x}_{t},t)$. The KL between path measures along the sub-trajectory shares the same optimum as the following loss objective:

$$ $\displaystyle{\mathcal{L}}_{\tau}$ $\displaystyle=\mathbb{E}_{t,\mathbf{x}_{t}}\Big{[}\mathbb{E}_{\tau(\mathbf{x}_ {0:t})}[\|\log q_{\theta}(\mathbf{x}_{0},\dots,\mathbf{x}_{t-1}|\mathbf{x}_{t} ,\hat{\mathbf{x}}_{0}))-\log p_{t}^{\text{pre}}(\mathbf{x}_{0},\dots,\mathbf{x }_{t-1}|\mathbf{x}_{t})+\kappa\|^{2}_{2}]\Big{]}$ $\displaystyle=\mathbb{E}_{t,\mathbf{x}_{t}}\Big{[}\mathbb{E}_{\tau(\mathbf{x}_ {0:t})}\Big{[}\Big{\|}\sum^{t}_{j=1}\log q_{\theta}(\mathbf{x}_{j-1}|\mathbf{x }_{j},\hat{\mathbf{x}}_{0})-\log p_{t}^{\text{pre}}(\mathbf{x}_{j-1},|\mathbf{ x}_{j},\hat{\mathbf{x}}^{\text{pre}}_{0})+\kappa\Big{\|}^{2}_{2}\Big{]}\Big{]}$ (42) $\displaystyle=\mathbb{E}_{t,\mathbf{x}_{t},\tau(\mathbf{x}_{0:t})}\Big{[}\Big{ \|}\sum^{t}_{s=1}\log q_{\theta}(\mathbf{x}_{s-\gamma}|\mathbf{x}_{s},\hat{ \mathbf{x}}_{0})-\log p_{t}^{\text{pre}}(\mathbf{x}_{s-\gamma},|\mathbf{x}_{s} ,\hat{\mathbf{x}}^{\text{pre}}_{0})+\kappa\Big{\|}^{2}_{2}\Big{]}$ $\displaystyle=\mathbb{E}_{t,\mathbf{x}_{t},\tau(\mathbf{x}_{0:t})}\left[\left \|t\mathbb{E}_{s,\mathbf{x}_{s},\mathbf{x}_{s-\gamma}}\left[\log q_{\theta}( \mathbf{x}_{s-\gamma}|\mathbf{x}_{s},\hat{\mathbf{x}}_{0})-\log p_{t}^{\text{ pre}}(\mathbf{x}_{s-\gamma},|\mathbf{x}_{s},\hat{\mathbf{x}}^{\text{pre}}_{0}) +\kappa\right]\right\|^{2}_{2}\right].$ (43) $$

In the last equation we define the constant $\kappa=(-\log R(\mathbf{x}_{0})+\log{\mathcal{Z}}_{\pi_{t}}(\mathbf{x}_{t}))/t$ and use the fact that our notation convention uses $p(X=x)=p(x)$ for discrete random variables. Now we make the observation that for any $s<t$ we have effectively picked an endpoint over the trajectory. More precisely, $s\sim{\mathcal{U}}[0,t]$, which also allows us to sample $\mathbf{x}_{s}\sim p_{t}(\mathbf{x}_{s}|\mathbf{x}_{0})$, in an analogous manner to how $\mathbf{x}_{t}$ is constructed.

## Appendix D Additional experimental details

All experiments were performed on a shared heterogenous high-performance computing cluster. This cluster is primarily composed of GPU nodes with RTX8000, V100, A100, L40S, and H100 NVIDIA GPUs. We briefly note a trick used across a number of our experiments for DDPP-LB described as warming up $\log Z_{t}(x_{t})$. We found that early iterations of training DDPP-LB could be somewhat unstable as the parameterized normalizing constant was not calibrated to a proper range given the pre-trained model and reward function. As such, we found that warming up $\log Z_{t}(x_{t})$ for some number of steps at the beginning of training by only allowing gradient flow through the $\log Z_{t}(x_{t})$ term helped stabilize training and improve overall performance. For the runs on which warming up $\log Z_{t}(x_{t})$ was utilized, we resume normal training (i.e., allowing gradient flow through the fine-tuned denoiser and $\log Z_{t}(x_{t})$) after the warmup period has concluded. For all experiments with DDPP-LB we used another separate, small DiT to parameterize the $\log Z_{t}(x_{t})$ prediction.

### D.1 Synthetic Experiments

Two synthetic tasks were performed: (1) sampling from a posterior over a 2 dimensional grid, and (2) fine-tuning on binarized MNIST. In both cases a 90 million parameter MDM model was trained on samples from the prior distribution, with the same DiT architecture as in .

#### D.1.1 Grid Experiment

The space consists of discrete tokens $\mathbf{x}_{0}\in\{0,\dots,127\}^{2}$. A prior density $p_{0}^{\text{pre}}$ is defined over this space which assigns a uniform probability for tokens falling inside one of the 16 evenly spaced squares, and a near-zero probability outside this. This prior distribution is depicted in Figure 1(a). Pre-training was done using the Adam optimizer, with $\beta_{1},\beta_{2}=\{0.9,0.999\}$, and a learning rate of $3\textrm{e}{-4}$.

The reward function $R(\mathbf{x}_{0})=0$ for $x_{0}^{1}<64$, and $R(\mathbf{x}_{0})=1$ for $x^{1}_{0}\geq 64$. This results in a fine-tuning target $\propto R(\mathbf{x}_{0})p^{\text{pre}}(\mathbf{x}_{0})$ which selects out only the squares in the lower half of the grid. This product distribution is
visualized in Figure 1(b).

For fine-tuning we train the model using our loss-functions with the Adam optimizer, using a learning rate of $4e-3$, $\beta_{1},\beta_{2}=\{0.9,0.999\}$, and a weight decay of 0 0 across all methods. DDPP-IS used 16 samples to estimate the partition function. Training is done using a replay buffer populated with points $\mathbf{x}_{0}$ sampled on policy from the model, as well as off-policy points from the prior distribution, added to the buffer every $100$ training steps. A batch of $64$ is used.

#### D.1.2 MNIST

This task consisted of generating binarized MNIST digits $\mathbf{x}_{0}\in\{0,1\}^{28\times 28}$. The prior $p^{\text{pre}}(\mathbf{x}_{0})$ in this case is the MNIST data distribution. For pre-training, the Adam optimizer is used with a learning rate of $4e-3$, $\beta_{1},\beta_{2}=\{0.9,0.999\}$ and a weight decay of 0 0.

This MDM is fine-tuned to produce even digits. More precisely, the reward function is $R(\mathbf{x}_{0})=p(\text{Even}\mid\mathbf{x}_{0})^{\beta}=\big{(}\sum_{i=0,2,
4,6,8}p(y=i\mid\mathbf{x}_{0})\big{)}^{\beta}$, with $p(y=i\mid\mathbf{x}_{0})$ being obtained from a pretrained MNIST classifier (LeNet 5 in this case). The inverse-temperature $\beta$ is set to $5$ for all experiments.

For fine-tuning with our methods, we use Adam with a learning rate of $1e-5$ and $\beta_{1},\beta_{2}=\{0.9,0.999\}$. Training is done with a batch-size of $64$. Samples are drawn from a replay-buffer populated with only on-policy samples. Method specific hyperparameters include:

- •
DDPP-IS: the importance sampling estimate is done with $16$ samples
- •
DDPP-LB: a learning rate of $1e-3$ is used for network layers estimating $\log{\mathcal{Z}}_{\pi_{t}}$
- •
DDPP-KL: The KL objective per $\mathbf{x}_{t}$ is computed using $8$ samples

RTB is trained with a learning rate of $5e-5$, with weight decay $0.01$, on trajectories of length $32$ with a batch size of $8$. For training, $30\%$ of the steps are detached. The smaller batch-size
is chosen to fit the training on 80GB of GPU memory.

SVDD uses $10$ particles in each inference step.

For all methods (including baselines), inference is done with $128$ steps.

Additional information on computation of metrics is included in D.3.1.

#### D.1.3 MNIST Samples

Samples from our methods, as well as the pretrained model, are shown in Figure 4.

Figure: (a) Pretrained model
Refer to caption: extracted/5916875/Plots/samples/mnist/pretrained_nt.png

### D.2 Protein sequences

Protein design involves the creation of novel protein sequences that adopt specific structures and perform desired functions. This is a critical field in synthetic biology and biotechnology, as it enables the rational engineering of proteins with enhanced stability, novel functionalities, or improved therapeutic properties. Advances in machine learning-based models, such as protein language models (pLMs), have enabled rapid exploration of protein sequence space, making de novo protein design more feasible and versatile. However, current pLMs struggle in generating realistic sequences which satisfy certain criteria, and we study using DDPP to finetune DPLM to generate high-scoring proteins given a reward function.

#### D.2.1 In-silico tasks

In task 1, we fine-tune the DPLM model to generate designable protein sequences that optimize for several critical features, including high predicted template modeling (pTM) and predicted local distance difference test (pLDDT) scores from ESMFold, reduced exposed hydrophobic residues, high sequence entropy, and an increased proportion of $\beta$-sheet content . These optimizations are captured in the reward function $R$, given by:

$$ $\displaystyle\log R$ $\displaystyle=w_{\text{pTM}}\cdot\text{pTM}+w_{\text{pLDDT}}\cdot\text{pLDDT}+ w_{\text{Sheet}}\cdot\text{Sheet\%}$ $\displaystyle+w_{\text{Entropy}}\cdot H(\mathbf{s})-w_{\text{Hpho}}\cdot{\text {Exposed_Hpho\%}}$ $$

Where the terms represent:

- •
pTM and pLDDT: Structural confidence scores from ESMFold, measuring global and local accuracy, respectively.
- •
Sheet%: The proportion of residues predicted to form $\beta$-sheets, determined by DSSP .
- •
$H(\mathbf{s})$: Sequence entropy, defined as:
$H(\mathbf{s})=-\sum_{i=1}^{L}\sum_{a}p_{i}(a)\log p_{i}(a),$
where $L$ is the length of the sequence and $p_{i}(a)$ is the probability of amino acid $a$ at position $i$.
- •
Exposed_Hpho%: Percentage of hydrophobic residues exposed on the surface, calculated based on solvent-accessible surface area.

The weights for these features are set as follows:

$$ $w_{\text{pTM}}=1,\quad w_{\text{pLDDT}}=1,\quad w_{\text{Sheet}}=4.5,\quad w_{ \text{Entropy}}=0.8,\quad w_{\text{Hpho}}=0.25.$ $$

As the scale of the various reward terms are non-uniform we selected the reward weights to weight all rewards similarly besides the sheet percent reward which is weighted higher. For the $\beta$-sheet task we found that both RTB and DDPP faced issues with mode collapse. After investigating the protein structures generated by base DPLM we found that the base model is only capable of generating a small number of motifs (in particular, over 2k samples from the base model we found only two motifs with $\log R(x_{0})\geq 3.5$), implying that the targeted product distribution indeed collapses around these structural motifs as we observe in the case of RTB and DDPP. As such, we conclude that DDPP (and RTB) achieve the goal of fine-tuning as they sample from the product distribution and reproduces samples with $\beta$-sheets at a much higher proportion than the base model.

In task 2, we focus on generating shorter sequences of known proteins that preserve essential structural characteristics, using the TM-align score as the reward function . This task allows the exploration of mutational effects. Ribonuclease proteins (PDB IDs: 9RAT-A, 11BA-A) are selected for this task due to their well-characterized structure, function, and folding mechanisms.

The reward function $R$ is defined as:

$$ $R=w_{\text{tm_score}}\cdot\text{TM-align}(\mathbf{s},\mathbf{t}).$ $$

Where:

- •
$w_{\text{tm_score}}$: The weight of the TM-Score reward, set to 2.
- •
$\mathbf{s}$: Predicted structure from ESMFold of the generated sequence.
- •
$\mathbf{t}$: Target protein structure.
- •
TM-align: A measure of structural similarity between $\mathbf{s}$ and $\mathbf{t}$, defined as:
$\text{TM-align}=\max\left(\frac{1}{L_{t}}\sum_{i=1}^{L_{\text{ali}}}\frac{1}{1
+\left(\frac{d_{i}}{d_{0}}\right)^{2}}\right).$
where $L_{t}$ is the length of the target protein, $L_{\text{ali}}$ is the length of the aligned region, $d_{i}$ is the distance between the $i$-th pair of aligned residues, and $d_{0}$ is the distance scale based on $L_{t}$ .

While not used in the reward function for either experimental setting, we also measure the KL divergence, reported as KL-SS in Table 3 between the secondary structure distribution given by DSSP for both the target and miniaturized protein.

Note that in these experiments, the number of recycles in ESMFold is set to 0 to reduce computational overhead. For both tasks we generate amino acid sequences of length 90. Evaluation is performed by sampling 200 proteins for each method across three seeds and reporting the mean and standard deviation of each metric accordingly. All methods ran 500 inference steps during evaluation. All protein experiments used a 150 million parameter DPLM base model(^1^11[https://huggingface.co/airkingbd/dplm_150m](https://huggingface.co/airkingbd/dplm_150m)) to begin fine-tuning from. All models used a log-linear noise schedule with $\sigma_{min}=1\textrm{e}{-4}$ and $\sigma_{max}=20$ and used a linear learning rate warmup period of 2500 training steps.

DDPP was trained with no warmup period for $\log Z_{t}(x_{t})$, a learning rate of $1\textrm{e}{-5}$, a batch size of 16, a replay buffer of max length 10,000, and inserting new batches to the buffer sampled on-policy from the current model every 250 training steps. RTB was trained similarly, but with a smaller batch size to account for its greater memory requirements. RTB matches the setting of DDPP but with a batch size of 8 while doing 90 inference steps during training (a new batch of trajectories is simulated on-policy every training step). To allow RTB to fit in memory we detached $65\%$ of trajectory timesteps when computing a backward pass on the RTB objective. SVDD was run on the base DPLM model with $n=10$ particles. To control the concentration of our designated target distributions, we set the reward temperature $\beta=0.125$ for the $\beta$-sheet task and $\beta=0.001$ for the protein miniaturization task.

**Table 5: Miniaturizing ribonuclease proteins 9RAT-A and 11BA-A (124 AAs) to 90 AAs while preserving structural fidelity (high TM-Score) and quality (high pLDDT and PTM).**
| Template |  | SS-KL $\downarrow$ | $\log R(\mathbf{x}_{0})$ $\uparrow$ | TM-Score $\uparrow$ | pLDDT $\uparrow$ | pTM $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| 9RAT-A | Base Model | 2.944 $\pm$ 2.936 | 0.502 $\pm$ 0.128 | 0.251 $\pm$ 0.064 | 0.724 $\pm$ 0.144 | 0.584 $\pm$ 0.226 |
| Best-of-10 | 0.640 $\pm$ 1.872 | 0.725 $\pm$ 0.098 | 0.363 $\pm$ 0.049 | 0.789 $\pm$ 0.018 | 0.754 $\pm$ 0.086 |  |
| DDPP | 1.086 $\pm$ 2.242 | 0.735 $\pm$ 0.122 | 0.368 $\pm$ 0.061 | 0.793 $\pm$ 0.044 | 0.768 $\pm$ 0.066 |  |
| RTB | 1.808 $\pm$ 2.597 | 0.597 $\pm$ 0.109 | 0.299 $\pm$ 0.055 | 0.796 $\pm$ 0.054 | 0.750 $\pm$ 0.084 |  |
| SVDD | 3.465 $\pm$ 2.835 | 0.699 $\pm$ 0.079 | 0.350 $\pm$ 0.039 | 0.499 $\pm$ 0.137 | 0.383 $\pm$ 0.178 |  |
| 11BA-A | Base Model | 3.136 $\pm$ 3.150 | 0.478 $\pm$ 0.101 | 0.239 $\pm$ 0.051 | 0.724 $\pm$ 0.144 | 0.584 $\pm$ 0.226 |
| Best-of-10 | 2.602 $\pm$ 3.309 | 0.654 $\pm$ 0.089 | 0.327 $\pm$ 0.045 | 0.782 $\pm$ 0.027 | 0.720 $\pm$ 0.109 |  |
| DDPP | 0.194 $\pm$ 1.009 | 0.709 $\pm$ 0.048 | 0.354 $\pm$ 0.024 | 0.743 $\pm$ 0.036 | 0.727 $\pm$ 0.054 |  |
| RTB | 2.579 $\pm$ 2.799 | 0.564 $\pm$ 0.111 | 0.282 $\pm$ 0.056 | 0.797 $\pm$ 0.058 | 0.744 $\pm$ 0.101 |  |
| SVDD | 3.240 $\pm$ 2.992 | 0.647 $\pm$ 0.079 | 0.324 $\pm$ 0.040 | 0.486 $\pm$ 0.124 | 0.354 $\pm$ 0.162 |  |

We report an extended version of Table 3 where we include results for both ribonuclease targets in Table 5. We observe that DDPP consistently achieves the highest TM-Score across the two templates while maintaining high structural quality with an average pLDDT of around 0.8.

#### D.2.2 Experimental validation

Genes encoding for de novo protein sequences were obtained from Integrated DNA Technologies (IDT) and cloned into pET-24a(+) (Novagen) expression vectors with a C-terminal 6xHis tag using Gibson Assembly (New England Biolabs, NEB). Assembled plasmids were verified via Sanger sequencing, then transformed into chemically competent Escherichia coli BL21(DE3) cells (NEB). Starter cultures (3 mL Luria Bertani media, 50 µg/mL kanamycin) were inoculated from freshly prepared agar plates and grown at 37°C and shaken at 225 RPM overnight. Starter cultures were then diluted 1:100 into 50 mL LB medium supplemented with antibiotic. Cultures were then grown at 37°C and 225 RPM until an optical density (OD600) of 0.5-0.7 was reached. Protein expression was then induced with 1 mM isopropyl $\beta$-D-thiogalactopyranoside (IPTG) for 4 hours at 37°C. Cells were then collected by centrifugation (4,500xg) at 4°C and resuspended in lysis buffer (Tris-buffered saline (TBS), 25 mM imidazole). Cell suspensions were then lysed via sonication (10s pulses, 40% amplitude). The corresponding lysate was centrifuged at 12,000xg for 30 minutes, and the supernatant was loaded into a HisPur Ni-NTA His-spin column (ThermoScientific) and purified as recommended. Expression of purified proteins in both the soluble and insoluble fraction, as well as his-tag purification fractions, was assessed using SDS-polyacrylamide gel electrophoresis.

### D.3 Discrete image modelling

To setup the finetuning task we first pre-train large masked diffusion models on the original dataset. This uses a standard masked diffusion loss as explored in previous work .

CelebA Pretraining. We train a 241 million parameter model based on the variational diffusion model (VDM) architecture  and the setup of . We adapted the U-Net plus self-attention architectures from as used in CIFAR-10 in their experiments, with a few notable additions. We replace the Fourier feature inputs with an input embedding layer which embeds 257 (256 pixel values + <MASK>) tokens into the embedding dimension. We double the number of residual blocks from 32 to 64 per encoder / decoder, and double the embedding dimension from 128 to 256. We use an Adam optimizer with learning rate $1e-3$, $\beta_{1}$=0.9 and $\beta_{2}$=0.999. We train our model for 450k steps with batch size 128 on a cluster of 16 NVIDIA L40S GPUs. We resize all CelebA images to 64x64 with bilinear interpolation. Samples from this model can be seen in Figure 5.

Separately, we train a 7M parameter classifier to classify hair color on CelebA. We use this as our energy function with a temperature setting of $0.1$ for all finetuning experiments.

Figure: Figure 5: Uncurated pre-trained CelebA model samples using a discrete generative model.
Refer to caption: extracted/5916875/Plots/celeba.png

CelebA Finetuning. With the problem setup, we next finetune our pretrained model to sample images with blond hair. We train each model for up to 12 A100 hours. We use an early stopping criteria based on a validation set using an approximate bits-per-dimension calculation using the ELBO. We find that the original needs at least $1\,000$ inference steps for good performance therefore we evaluate all models in this setting. For our model we use $1\,000$ warmup steps for $\log Z$, a learning rate of $1e-4$, we resample two batches every 500 gradient steps of the model and add them to the replay buffer.

In contrast to our model, RTB requires a full trajectory for each gradient step. For CelebA, this means a rollout of $1\,000$ inference steps taking approximately 2 minutes for a batch size of 2 on an A100 with this model. Because of memory constraints we detach 99% of inference steps and use a batch size of 2 to fit in 80GB of GPU memory with a global batch size of 8 trajectories per gradient step.

#### D.3.1 Metrics

The metrics used to evaluate image fine-tuning include mean log reward, feature-likelihood divergence (FLD), and bits per dimension (BPD).

FLD. For FLD, we draw $K$ samples from the model, and $K$ samples from the test set restricted to the target class. The FLD is computed using the DINOV2 feature space (from the ViT-B14 model) between these two sets of samples . For MNIST, $K=5$k.

BPD. An upper bound on the log-likelihood is computed using the MDM ELBO loss (on the fine-tuned model), and this is normalized (by the number of pixels and $\log 2$) to obtain the reported BPD metric. This metric is computed on the test set restricted to the target-class.

### D.4 Text experiments

All text experiments begin by starting from the pretrained MDLM(^2^22[https://huggingface.co/kuleshov-group/mdlm-owt](https://huggingface.co/kuleshov-group/mdlm-owt)) consisting of 170 million parameters. We then do supervised fine-tuning to produce a model capable of producing output of the desired format before proceeding with online fine-tuning. For all text experiments we train using the Adam optimizer with $\beta_{1},\beta_{2}=\{0.9,0.999\}$ and weight decay of 0 0. For both tasks SVDD was run with $n=10$ particles. Evaluation was done by training each method for one day across three seeds and generating $1000$ samples from the best checkpoint according to the mean reward generated during training. We report both the mean reward of the $1000$ samples across three seeds and their standard deviations, as well as the generative perplexity according to a GPT2-Large model with 812 million parameters (^3^33[https://huggingface.co/openai-community/gpt2-large](https://huggingface.co/openai-community/gpt2-large)).

#### D.4.1 Tinystories

**Table 6: Curated samples from DDPP-LB on Tinystories.**
| Generated Text |
| --- |
| Once upon a time, there was a little girl named Lily. One day, Lily went to the park with her mom. She loved to run fast and laugh. Lily saw a funny butterfly and ran it too. She fell fast and hurt a tree. Lily’s knee hurt and she cried. But her mommy kissed her cheeks and said, “Be careful when you run, Lily.” |
| Once upon a time, there was an elderly wolf. He lived in a big den against the woods. One day, the wolf felt very tired and wanted a place where there was a big tree to eat on. So, he went to sleep all day. But, while he was waking up, he saw a big, icy creature. The quickly jumped up, but his legs were too weak. The creature took the elderly wolf away. And that is how winter ended. |
| Once upon a time, there was a little bird. His wings were weak and he fell down. The bird wanted his wing to restore him. So, he flapped his wings with his weak heart. |
| Once upon a time, there was a little boy named Timmy. He loved going to the woods with his family. One day, Timmy’s friend Johnny came to the woods to play. Johnny was excited to go outside and play. |
| They found a big tree with words said "I reverse," and Timmy would play on its branches. He said, "reverse!" and pushed the tree. Then, they ran and laughed. |
| But then, they heard a loud noise coming from the bushes and scared them. It was a big, mean bear! Timmy tried to reverse and run away, but he wasn’t fast enough. The bear chased him and caught him up with its sharp hands. |
| Timmy was very scared and never went back to the woods again. |
| Once upon a time, on a calm blue sea, there was a small boat. The boat had sailors. They lived happily in the day water. One day, the water was very hot. So, they all decided to soak up and have a picnic. |
| But, by the time, the sailors started to play a game. They swam around and counted, ", two, three soon!" and all the sailors kept playing. They water flowers and trees, and everyone laughed. |
| But then, a clumsy heavy sailor hurt his head on a rock. "Ouch!" he cried. His friends helped him up and said, "Be careful next time!" The sailor felt better and they all laughed. They knew they could play again and have fun on a calm day soon. |

To obtain a base model we performed supervised fine-tuning from the base MDLM model on the Tinystories dataset. As all methods were prompted with the text “Once upon a time” during training, we restricted the SFT dataset to only datapoints whose stories started with the text “Once upon a time,”, resulting in a corpus of 977,921 examples. SFT was done using Adam, $\beta_{1},\beta_{2}=\{0.9,0.999\}$, and a learning rate of $4\mathrm{e}{-3}$ over 60,000 training steps using 4 NVIDIA A100 GPUs. All models were trained for up to 24 GPU hours on NVIDIA L40S GPUs. Fine-tuning checkpoints were selected based upon the iteration with best average reward when sampling a new training batch from the model. All methods employ a learning rate schedule with a linear warmup for the first 2,500 training steps and keep the noise schedule provided by the pre-trained MDLM model – a log-linear schedule with $\sigma_{min}=1\textrm{e}{-4}$ and $\sigma_{max}=20$. All evaluations were performed by taking 1000 samples for each method across three seeds. We provide a set of curated samples in Table 6.

The reward function for this task was selected to be a pre-trained classifier(^4^44[https://huggingface.co/nicholasKluge/ToxicityModel](https://huggingface.co/nicholasKluge/ToxicityModel)) . The classifier is a RoBERTa model with 125 million parameters which was fine-tuned on a curated subset of various toxicity/harmlessness datasets. The reward $R(x_{0})$ is then set to the likelihood of a sequence being toxic under the pre-trained classifier so that $R(x_{0})=p(a=1|x_{0})$ where $p(a=1|x_{0})$ denotes the likelihood of the sequence $x_{0}$ possessing a toxic sentiment. We select this task as it allows a demonstration of how our method can recover rare behavior under the pre-trained model while still maintaining sample quality. We consider as our target distribution the tempered reward distribution $\pi_{0}(x_{0})\propto p_{0}^{\textrm{pre}}(x_{0})R(x_{0})^{1/\beta}$ with $\beta=0.25$.

For DDPP-LB we used 1,500 warmup steps for $\log{\mathcal{Z}}_{\pi_{t}}(x_{t})$, a learning rate of $1\textrm{e}{-4}$ and a batch size of 16. We employ a replay buffer with a max length of 10,000 and sample training batches uniformly from the buffer. The buffer is filled every 50 training steps with a batch sampled on-policy from the current fine-tuned model, while every 250 steps a batch from the SFT training dataset is added to the buffer. We use EMA with a decay rate of $\epsilon=0.9999$, a learning rate of $1\textrm{e}{-4}$, and train without LoRA. DDPP-LB was trained using 64 inference steps for simulation. DDPP-IS employed the same hyperparameter settings as DDPP-LB except that it dispelled with learning $\log{\mathcal{Z}}_{\pi_{t}}(x_{t})$ and instead estimated it with $K=16$ Monte Carlo samples from the one-step pre-trained posterior $p_{t}^{\textrm{pre}}(x_{0}|x_{t})$.

As RTB cannot fit all timesteps of a trajectory into memory during the backwards pass, we detached 55% of timesteps where each trajectory consisted of 32 timesteps. RTB was trained with LoRA enabled, a LoRA rank of 16, and a learning rate of $5\textrm{e}{-5}$. Due to memory constraints the batch size was set to 4. SVDD was run by using $n=10$ particles per inference timestep. Best of N (with $N=10$) sampling was performed by taking 10 samples from the SFT model and selecting the sample with highest likelihood under the reward model.

#### D.4.2 Amazon reviews

**Table 7: Curated samples from DDPP-LB on Amazon review generation task.**
| Generated Text |
| --- |
| Cheap. Poor fit. The pants return immediately and the fabric was like a burlap sack bag-Wanted it for a gift-It’s crap. Broke in one day. Customer service never responded.<br />Very cheap! |
| Everything was horrible. |
| It’s the worst one I’ve ever bought.. you have to keep it in your bag for my daughter and they, seriously feel embarrassed if you ever got it it falls out and it ripped.. returning this. That said hated this bag till I see it!!!!!! (Cnaven at the neckline and it is super too short. Color is off white. Maybe I have to fix if I want ironing<br />What a waste of valu money |
| Such poor material!!! It was like a plastic. Way too small so I returned.It was smaller than the size listed |
| I don’t feel this product has any quality. My sunglasses was delivered broken. |
| Cheap piece of garbage. Got it for my niece for Halloween and it broke inó one time |

We again begin by first performing supervised fine-tuning from the base MDLM model, but this time on the fashion split of the Amazon Reviews dataset restricted to reviews consisting of at most 512 tokens, resulting in an SFT dataset of size. We perform SFT using Adam with $\beta_{1},\beta_{2}=\{0.9,0.999\}$ and a learning rate of $4\textrm{e}{-3}$ and EMA with decay parameter $0.99$. The SFT model was trained for 85,000 training steps on 4 NVIDIA A100 GPUs. As for the tinystories task fine-tuning checkpoints were selected based upon the iteration with best average reward when sampling a new training batch from the model. All methods employ a learning rate schedule with a linear warmup for the first 2,500 training steps and keep the noise schedule provided by the pre-trained MDLM model – a log-linear schedule with $\sigma_{min}=1\textrm{e}{-4}$ and $\sigma_{max}=20$. All evaluations were performed by taking 1000 samples for each method across three seeds.

The reward function for this task was a BERT model consisting of 167 million parameters fine-tuned on Amazon customer reviews(^5^55[https://huggingface.co/LiYuan/amazon-review-sentiment-analysis](https://huggingface.co/LiYuan/amazon-review-sentiment-analysis)) to predict a review’s star-rating. We then set the reward $R(x_{0})=p(a=1|x_{0})$, the likelihood under the pre-trained classifier that the generated sample is a one-star review. We consider as our target distribution the tempered reward distribution $\pi_{0}(x_{0})\propto p_{0}^{\textrm{pre}}(x_{0})R(x_{0})^{1/\beta}$ with $\beta=0.5$.

For DDPP-LB we used 1,500 warmup steps for $\log{\mathcal{Z}}_{\pi_{t}}(x_{t})$, a learning rate of $1\textrm{e}{-4}$ and a batch size of 16. We employ a replay buffer with a max length of 10,000 and sample training batches uniformly from the buffer. The buffer is filled every 5 training steps with a batch sampled on-policy from the current fine-tuned model, while every 250 steps a batch from the SFT training dataset is added to the buffer. We use EMA with a decay rate of $\epsilon=0.9999$, a learning rate of $1\textrm{e}{-4}$, and train without LoRA. DDPP-LB was trained using 64 inference steps for simulation. DDPP-IS employed the same hyperparameter settings as DDPP-LB besides not learning $\log{\mathcal{Z}}_{\pi_{t}}(x_{t})$ and instead estimating it with $K=16$ Monte Carlo samples from the one-step pre-trained posterior $p_{t}^{\textrm{pre}}(x_{0}|x_{t})$.

As RTB cannot fit all timesteps of a trajectory into memory during the backwards pass, we detached 78.5% of timesteps where each trajectory consisted of 64 timesteps. RTB was trained with LoRA enabled, a LoRA rank of 16, and a learning rate of $5\textrm{e}{-5}$. Due to memory constraints the batch size was set to 4. SVDD was run by using $n=10$ particles per inference timestep. Best of N (with $N=10$) sampling was performed by taking 10 samples from the SFT model and selecting the sample with highest likelihood under the reward model.

We provide a set of curated samples for the Amazon task in Table 7.