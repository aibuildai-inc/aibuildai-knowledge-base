---
arxiv_id: "2410.01949"
title: "Discrete Copula Diffusion"
year: 2024
source: arxiv2md
---

## Abstract

Abstract Discrete diffusion models have recently shown significant progress in modeling complex data, such as natural languages and DNA sequences. However, unlike diffusion models for continuous data, which can generate high-quality samples in just a few denoising steps, modern discrete diffusion models still require hundreds or even thousands of denoising steps to perform well. In this paper, we identify a fundamental limitation that prevents discrete diffusion models from achieving strong performance with fewer steps – they fail to capture dependencies between output variables at each denoising step. To address this issue, we provide a formal explanation and introduce a general approach to supplement the missing dependency information by incorporating another deep generative model, termed the copula model. Our method does not require fine-tuning either the diffusion model or the copula model, yet it enables high-quality sample generation with significantly fewer denoising steps. When we apply this approach to autoregressive copula models, the combined model outperforms both models individually in unconditional and conditional text generation. Specifically, the hybrid model achieves better (un)conditional text generation using 8 to 32 times fewer denoising steps than the diffusion model alone. In addition to presenting an effective discrete diffusion generation algorithm, this paper emphasizes the importance of modeling inter-variable dependencies in discrete diffusion. 1 1 1 Code is available at https://github.com/liuanji/Copula-Diffusion .

## 1 Introduction

Discrete diffusion models have recently achieved significant progress in modeling complex data such as natural languages , protein sequences , and graphs . In particular, recent discrete diffusion models for text generation have matched or even surpassed the performance of autoregressive models at the scale of GPT-2 . Additionally, discrete diffusion models offer improved inference-time controllability using guidance from auxiliary models such as classifiers , making them suitable for controlled generation tasks .

Despite these promising results, discrete diffusion models still require hundreds to thousands of denoising steps to produce high-quality samples , significantly affecting their efficiency. In this paper, we identify a fundamental limitation in most discrete diffusion models that hinders their ability to generate high-quality samples in just a few steps.

We illustrate the problem in Figure 1. At each denoising step, a partially completed sample shown in the top-left is fed into a sequence-to-sequence denoising model, which predicts the univariate marginal distributions for each masked token independently. A new output sequence is then sampled based on these univariate marginals before proceeding to the next denoising step. The key issue with this process is that when multiple “edits” (i.e., replacing masked tokens with data tokens) are made simultaneously, the model does not account for the joint probability of these changes occurring together. As a result, the generated samples often lack coherence, as shown in the bottom-left of Figure 1. This problem is exacerbated in few-step generation, where many tokens must be edited simultaneously. We formally demonstrate that if the diffusion model predicts each variable independently, an irreducible term (in addition to the data entropy) remains in the negative evidence lower bound (ELBO), preventing the model from perfectly capturing the data distribution.

We propose using a generative model, which we refer to as the copula model, to compensate for the missing dependency information between output variables at each denoising step. Our method operates only at inference time and can be adapted to any discrete diffusion model and a wide range of copula models.
As illustrated on the right side of Figure 1, the input sequence is also fed into a copula model that (implicitly) produces information on inter-variable dependencies. This information is combined with the univariate marginals predicted by the diffusion model to produce a more accurate distribution, resulting in higher-quality samples, shown in the bottom-right corner.

We formally show that the univariate marginals from the diffusion model and the dependencies captured by the copula model can be combined in a principled way, leading to a better approximation of the true denoising distribution under mild assumptions. Further, finding this combined distribution reduces to solving a convex optimization problem that can be efficiently approximated in practice.

By instantiating the copula model as an autoregressive deep generative model such as GPT , we propose an algorithm that combines any pretrained discrete diffusion model with an autoregressive model to form a hybrid model called Discrete Copula Diffusion (DCD). This model is capable of producing high-quality (un)conditional samples with only a few denoising steps. Empirical results on text and antibody generation show that DCD significantly outperforms both of its base models. Moreover, DCD achieves comparable or better performance using 8 to 32 times fewer denoising steps compared to the base discrete diffusion model. In addition to proposing a discrete diffusion model capable of few-step generation, we emphasize the importance of modeling inter-variable dependencies in discrete diffusion models.

Figure: Figure 1: Discrete Copula Diffusion (DCD). At each denoising step, a partially completed sequence is given as input (top-left). The diffusion model independently predicts the univariate marginals for each masked token, which leads to the samples in the bottom-left. DCD introduces an additional copula model (top-right) to capture the inter-variable dependencies, thereby supplementing the information missed by the diffusion model. By combining outputs from both models in a principled way, DCD achieves better performance than either model individually (see improved samples in the bottom-right), enabling few-step discrete diffusion generation.
Refer to caption: x1.png

## 2 Preliminaries

We aim to model the joint distribution of variables $\mathbf{X}_{0}$, a set of categorical variables with $C$ categories. Discrete diffusion models learn to sample from ${p}(\mathbf{X}_{0})$ by modeling the reversal of the following noising process involving $\mathbf{X}_{0}$ and a set of auxiliary variables $\{\mathbf{X}_{t}\}_{t=1}^{T}$:

$$ $\displaystyle\forall t\in\{1,\dots,T\}\quad{q}(\bm{x}_{t}|\bm{x}_{t-1}):= \mathrm{Cat}(\bm{x}_{t};Q_{t}\!\cdot\!\bm{x}_{t-1}),$ (1) $$

where $\mathrm{Cat}(\bm{x};\mathbf{p})$ refers to the Categorical distribution over $\bm{x}$ with class probabilities $\mathbf{p}$, and $Q_{t}$ is a $C\!\times\!C$ transition matrix that is applied independently to every variable $x_{t-1}^{i}$ (denote $x^{i}_{t-1}$ as the $i$th variable of $\bm{x}_{t-1}$) to get the corresponding categorical distribution of $x_{t}^{i}$. Specifically, each variable $x_{t-1}^{i}$ is treated as a one-hot vector of size $C\!\times\!1$, which is then multiplied by $Q_{t}$ to compute the class probabilities of $x_{t}^{i}$. The noising process is designed such that ${p}(\bm{x}_{T})$ follows a simple distribution regardless of the data distribution.

Instead of using a fixed number of predefined time steps, we can treat $t$ as a continuous variable within the range $[0,T]$. The noising process is now defined by the rate of change of ${p}(\bm{x}_{t})$ w.r.t. $t$: $\frac{d{p}(\bm{x}_{t})}{dt}\!=\!Q\!\cdot\!{p}(\bm{x}_{t})$, where $Q\!\in\!\mathbb{R}^{C\times C}$ is a transition rate matrix. For any $0\!\leq\!s\!<\!t\!\leq\!T$, we have

$$ $\displaystyle{q}(\bm{x}_{t}|\bm{x}_{s}):=\mathrm{Cat}(\bm{x}_{t};\exp((t\!-\!s )\!\cdot\!Q)\!\cdot\!\bm{x}_{s}),$ $$

where $\exp(\cdot)$ denotes the matrix exponential.

Discrete diffusion models represent the reverse diffusion process as a Markov chain from $\bm{x}_{T}$ to $\bm{x}_{0}$, effectively reversing the noising process. Specifically, the reverse diffusion is modeled as:

$$ $\displaystyle{p}_{\theta}(\bm{x}_{0:T}):={p}(\bm{x}_{T})\prod_{t=0}^{T-1}{p}_{ \theta}(\bm{x}_{t}|\bm{x}_{t+1}).$ $$

In the discrete-time framework, the model is trained by maximizing the ELBO, which is defined by the forward joint distribution (${q}(\bm{x}_{1:T}|\bm{x}_{0}){p}(\bm{x}_{0})$) and the reverse joint distribution (${p}_{\theta}(\bm{x}_{0:T})$) . In the continuous-time framework, we can either adopt an extended ELBO objective or to learn the likelihood ratios $\{{p}(\bm{x}^{\prime}_{t})/{p}(\bm{x}_{t})\}_{\bm{x}_{t},\bm{x}^{\prime}_{t}}$, allowing for the recovery of ${p}(\bm{x}_{s}|\bm{x}_{t})$ ($s\!<\!t$) in an indirect manner . Following the reverse diffusion process, sampling from a diffusion model involves first sampling from the prior ${p}(\bm{x}_{T})$ and then recursively sampling $\bm{x}_{T-1},\dots,\bm{x}_{0}$ following $\{{p}_{\theta}(\bm{x}_{t}|\bm{x}_{t-1})\}_{t=0}^{T-1}$.

## 3 Challenge of Modeling Variable Dependencies

Unlike continuous diffusion models, which can produce high-quality samples with just a few steps (e.g., ), discrete diffusion models exhibit a strong positive correlation between sample quality and the number of denoising steps. For instance, to generate $1024$ text tokens, a recent discrete diffusion model SEDD requires $1024$ steps to reach around $35$ perplexity (PPL), while with $32$ denoising steps the PPL is only around $130$.

We argue that the need for a large number of sampling steps in discrete diffusion models stems from their inability to capture inter-variable dependence among the outputs. Specifically, at each time step $t$, discrete diffusion models independently sample each variable from $\bm{x}_{t}$ conditioned on $\bm{x}_{t+1}$, i.e., ${p}(\bm{x}_{t}|\bm{x}_{t+1})\!:=\!\prod_{i}{p}(x_{t}^{i}|\bm{x}_{t+1})$. As a result, when changing multiple variables from $\bm{x}_{t+1}$ to $\bm{x}_{t}$, the model fails to account for the joint probability of these modifications happening together.
In the following, we first quantitatively analyze the performance degradation caused by this independent denoising assumption. We then discuss approaches to mitigate this issue.

Quantifying the Performance Drop.
The total correlation of a distribution ${p}(\mathbf{X})$ is the KL-divergence between itself and the product of its univariate marginals:

$$ $\displaystyle\mathrm{D}_{\mathrm{TC}}({p}(\mathbf{X}))\!:=\!\sum_{\bm{x}}{p}( \bm{x})\log\Big{(}{p}(\bm{x})/\prod_{i}{p}(x_{i})\Big{)}.$ $$

The following result demonstrates that, under the independent denoising assumption, there is an irreducible component in the ELBO that directly stems from ignoring inter-variable dependencies.

###### Proposition 1 .

Assume the denoising distributions $\{{p}_{\theta}(\bm{x}_{t}|\bm{x}_{t+1})\}_{t=0}^{T-1}$ are fully factorized. Let $\mathrm{H}({p}(\mathbf{X}))$ denote the entropy of ${p}(\mathbf{X})$. For any choice of denoising distributions (or equivalently, any parameterization $\theta$), the negative ELBO of the diffusion model is lower bounded by

$$ $\displaystyle\mathrm{H}({p}(\mathbf{X}_{0}))+\sum_{t=1}^{T}\mathrm{D}_{\mathrm {TC}}(q(\mathbf{X}_{t-1}|\mathbf{X}_{t})),\text{~{}~{}~{}~{}where~{}}\mathrm{D }_{\mathrm{TC}}({p}(\mathbf{Y}|\mathbf{X})):=\mathbb{E}_{\bm{x}\sim{p}}\big{[} \mathrm{D}_{\mathrm{TC}}({p}(\mathbf{Y}|\bm{x}))\big{]}.$ (2) $$

The first term represents the entropy of the data distribution and is irreducible. The second term additionally depends on the noising process and the chosen noise levels, which set an upper limit on the performance of discrete diffusion models that use the independent denoising assumption. Note that although $\mathrm{D}_{\mathrm{TC}}({q}(\mathbf{X}_{t}|\mathbf{X}_{t-1}))$ is zero according to the definition of the noising process, $\mathrm{D}_{\mathrm{TC}}({q}(\mathbf{X}_{t-1}|\mathbf{X}_{t}))$ is not unless the data distribution is fully factorized.

Closing the Performance Gap.
While increasing the number of denoising steps can improve sample quality, it also introduces significant computational overhead during inference. Our goal is to use fewer denoising steps while maintaining good sample quality. As shown in Proposition 1, given a fixed noising strategy and the number of denoising steps, the only way to reduce the negative ELBO lower bound in Equation 2 is to relax the independent denoising assumption. That is, in addition to modeling the univariate marginals, we must also account for dependencies between variables.

The challenge of capturing inter-variable dependencies during each denoising step can be addressed through adjustments during either training or inference. A direct approach involves modeling both the univariate marginals and the inter-variable dependencies within the diffusion model. However, this requires improving existing sequence-to-sequence architectures (e.g., ) to capture dependencies between output variables directly, which is not very well studied in the literature.

Instead, we propose an inference-time solution that complements the information missed by the pretrained discrete diffusion model. Specifically, we aim to combine the univariate marginals produced by the diffusion model with the inter-variable dependencies learned by another (possibly smaller) deep generative model, which we refer to as the *copula model*. The term “copula” traditionally refers to the dependencies between random variables in statistics .

###### Proposition 1 .

## 4 Modeling Variable Dependencies with Copula Models

As motivated in the previous section, our main goal is to combine the univariate marginals produced by the diffusion model with the inter-variable dependencies captured by a copula model. In this section, we first formalize the concept of “combining” two such distributions in a general context (Sec. 4.1). We then specialize the formulation to the case of diffusion models (Sec. 4.2).

### 4.1 Combining Univariate Marginals with Inter-Variable Dependencies

In this section, we discuss how to best inject inter-variable dependence using copula models given a target distribution ${p}_{\mathrm{tar}}$ over $\mathbf{X}$. Assume we have access to ${p}_{\mathrm{tar}}$ through two sources: (i) the set of all univariate marginal distributions $\{{p}_{\mathrm{tar}}(X_{i})\}_{i}$, and (ii) an estimate ${p}_{\mathrm{est}}$ of the target distribution coming from the copula model, which is also a generative model. Our goal is to combine these two estimates to construct $\hat{{p}}$ that is “closer” to the true distribution ${p}_{\mathrm{tar}}$ than either estimate individually.

We construct $\hat{{p}}$ as the distribution that (i) matches the set of univariate marginals $\{{p}_{\mathrm{tar}}(X_{i})\}_{i}$, and (ii) minimizes the KL divergence to ${p}_{\mathrm{est}}$. The intuition is that by ensuring $\hat{{p}}$ has the correct univariate marginals, we can achieve a good approximation of ${p}_{\mathrm{tar}}$ even if ${p}_{\mathrm{est}}$ is biased. To formalize this, we first define information projection (I-projection).

###### Definition 1 .

The I-projection of a distribution $q(\mathbf{X})$ onto a set of distributions $\mathcal{P}$ over $\mathbf{X}$ is

$$ $\displaystyle{p}^{*}=\operatorname*{arg\,min}_{{p}\in\mathcal{P}}\mathrm{D}_{ \mathrm{KL}}({p}\;\|\;{q}).$ $$

Let $\mathcal{P}^{{p}}_{\mathrm{mar}}$ denote the set of distributions over $\mathbf{X}$ that share the same univariate marginals as ${p}$. We define $\hat{{p}}$ as the I-projection of ${p}_{\mathrm{est}}$ onto $\mathcal{P}^{{p}_{\mathrm{tar}}}_{\mathrm{mar}}$. The following proposition shows that regardless of the initial estimate ${p}_{\mathrm{est}}$ of ${p}_{\mathrm{tar}}$, the I-projection $\hat{{p}}$ will be an improved estimate of ${p}_{\mathrm{tar}}$ in KL-divergence.

###### Proposition 2 .

If there exists $i$ and $x_{i}$ s.t. ${p}_{\mathrm{tar}}(x_{i})\!\neq\!{p}_{\mathrm{est}}(x_{i})$, then $\mathrm{D}_{\mathrm{KL}}({p}_{\mathrm{tar}}\|\;\hat{{p}})\!<\!\mathrm{D}_{
\mathrm{KL}}({p}_{\mathrm{tar}}\|\;{p}_{\mathrm{est}}).$

Having now seen that $\hat{{p}}$ is an improved estimate of ${p}_{\mathrm{tar}}$, we next explore whether it is feasible to compute $\hat{{p}}$ given $\{{p}_{\mathrm{tar}}(X_{i})\}_{i}$ and ${p}_{\mathrm{est}}$. We start by showing that $\hat{{p}}$ has a simple form.

###### Proposition 3 .

Assume $\forall\bm{x}$, ${p}_{\mathrm{tar}}(\bm{x})\!>\!0$ and ${p}_{\mathrm{est}}(\bm{x})\!>\!0$. Then $\hat{{p}}$ exists and has the form

$$ $\displaystyle\hat{{p}}(\bm{x})={p}_{\mathrm{est}}(\bm{x})\cdot\prod_{i}\sigma_ {i}(x_{i}),$ $$

where $\sigma_{i}$ is a positive function
that depends on $x_{i}$.

Assume $\mathbf{X}$ consists of $N$ categorical variables, each with $C$ categories, we can represent the factors $\{\sigma_{i}\}_{i}$ using a matrix $\mathbf{V}\!\in\!\mathbb{R}^{N\times C}$. Under this representation, the combined distribution is

$$ $\displaystyle\hat{{p}}(\bm{x})={p}_{\mathrm{est}}(\bm{x})\cdot\prod_{i}\exp( \mathbf{V}[i,x_{i}]),$ (3) $$

where $\mathbf{V}[i,j]$ denotes the element at the $i$th row and $j$th column of $\mathbf{V}$ and $\mathbf{V}[i,x_{i}]\!=\!\log\sigma_{i}(x_{i})$. Determining the true matrix $\mathbf{V}^{*}$ corresponding to $\hat{{p}}$, which is the I-projection of ${p}_{\mathrm{est}}$ onto $\mathcal{P}_{\mathrm{mar}}^{{p}_{\mathrm{tar}}}$, can be reformulated as solving the following convex optimization problem.

###### Theorem 1 .

If $\mathbf{V}^{*}$ minimizes the following convex objective function, then the corresponding $\hat{{p}}$ defined by Equation 3 is the I-projection of ${p}_{\mathrm{est}}$ onto $\mathcal{P}_{\mathrm{mar}}^{{p}_{\mathrm{tar}}}$.(^2^22Equation 4 closely resembles the matrix scaling problem . See Appendix B for details.)

$$ $\displaystyle\mathcal{L}(\mathbf{V};{p}_{\mathrm{tar}},{p}_{\mathrm{est}}):= \sum_{\bm{x}}{p}_{\mathrm{est}}(\bm{x})\cdot\prod_{i}\exp(\mathbf{V}[i,x_{i}]) -\sum_{i=1}^{N}\sum_{x_{i}=1}^{C}\mathbf{V}[i,x_{i}]\cdot{p}_{\mathrm{tar}}(x_ {i}).$ (4) $$

We proceed to explain why I-projecting ${p}_{\mathrm{est}}$ leads to a better estimate of ${p}_{\mathrm{tar}}$ as suggested by Proposition 2. In general, a joint distribution can be viewed as combining two independent pieces of information: (i) a set of univariate marginal distributions and (ii) a copula describing the association or dependence among the variables. By the classical work of , for continuous variables the copula can take the form of a joint distribution with uniform margins and can be combined quite simply with univariate marginal distributions to recover the full joint distribution, a fact heavily exploited in statistics .
While the discrete case is somewhat less straightforward, recent work of has developed the fundamental notions of discrete copula modeling as well, where the information of a copula can be parameterized by odds ratios.

Figure 2 shows an example consisting of two binary variables $X$ and $Y$. The probability table on the left can be equivalently expressed using univariate marginals (i.e., ${p}_{0\cdot},{p}_{1\cdot}$, ${p}_{\cdot 0}$, ${p}_{\cdot 1}$) and the odds ratio (i.e., copula) $\omega\!:=\!\frac{{p}_{00}{p}_{11}}{{p}_{01}{p}_{10}}$ as shown in the middle of Figure 2. Intuitively, $\omega\!=\!125$ indicates that the phrases “alpine skiing” and “scuba diving” are more likely than others (e.g., “alpine diving”), and the marginals decide which of the two phrases appears more frequently. The idea of representing the copula with odds ratios generalizes to the multivariate case and is presented in Appendix C.

The following result demonstrates that, under its functional form in Equation 3, I-projecting ${p}_{\mathrm{est}}$ onto $\mathcal{P}_{\mathrm{mar}}^{{p}_{\mathrm{tar}}}$ only improves the univariate marginals and leaves the copula unchanged regardless of $\mathbf{V}$.

###### Proposition 4 .

For a positive distribution $p$ and any $\mathbf{V}\!\in\!\mathbb{R}^{N\times C}$, the distribution $q(\bm{x})\!\propto\!{p}(\bm{x})\!\cdot\!\prod_{i}\exp(\mathbf{V}[i,x_{i}])$ has the same copula as $p$.

In general, Proposition 4 holds because scaling factors (e.g., $\exp(\mathbf{V}[i,x_{i}])$) cancel in odds ratios. For example, in the $2\!\times\!2$ case in Figure 2, scaling the top row of the probability table by $a$ would result in the odds ratio
$\omega\!=\!\frac{a{p}_{00}{p}_{11}}{a{p}_{01}{p}_{10}}\!=\!\frac{{p}_{00}{p}_{
11}}{{p}_{01}{p}_{10}}$.

Figure: Figure 2: Illustration of the decomposition of a distribution into univariate marginals and a copula.
Refer to caption: x2.png

###### Definition 1 .

###### Proposition 2 .

###### Proposition 3 .

###### Theorem 1 .

###### Proposition 4 .

### 4.2 Modeling Dependence in Discrete Diffusion Models

Recall from Section 3 that our goal is to capture inter-variable dependencies between the output variables at each denoising step (e.g., sampling $\bm{x}_{t}$ from ${q}(\mathbf{X}_{t}|\bm{x}_{t+1})$). Similar to the general case shown in Section 4.1, we first have a set of univariate marginals $\{{p}_{\mathrm{dm}}(X_{t}^{i}|\bm{x}_{t+1})\}_{i}$ from the diffusion model. Notably, these univariate marginals are fairly accurate since for both discrete-time and continuous-time diffusion models, if their respective training losses are minimized, the model recovers the true univariate marginals. This is formally justified in Appendix D.

Alongside the univariate marginals, we assume access to a copula model that encodes a distribution over $\mathbf{X}_{t}$. Following Section 4.1, combining the copula model’s distribution with the univariate marginals from the diffusion model will lead to an improved estimate of ${q}(\mathbf{X}_{t}|\bm{x}_{t+1})$ (Prop. 2).

The performance of the augmented diffusion model hinges on two key questions: (i) how well can the copula model capture the inter-variable dependencies in ${q}(\mathbf{X}_{t}|\bm{x}_{t+1})$ (defined by the data distribution and the noising process); (ii) given a good copula distribution, how to effectively combine it with the univariate marginals obtained from the diffusion model, i.e., how to solve Equation 4.

## 5 Autoregressive Models as Copula Models

This section answers the two questions above tailored to the case where the copula model is an autoregressive model such as GPT and State Space Models . Specifically, Section 5.1 discusses how to approximate ${q}(\mathbf{X}_{t}|\bm{x}_{t+1})$ using an autoregressive model trained on the clean data distribution ${p}(\mathbf{X}_{0})$ under certain noising processes. Section 5.2 explores the process of performing I-projection from the (autoregressive) copula distribution onto the set of distributions with univariate marginals $\{{p}_{\mathrm{dm}}(X_{t}^{i}|\bm{x}_{t+1})\}_{i}$. Finally, Section 5.3 summarizes the sampling procedure with a discrete diffusion model and an autoregressive copula model.

### 5.1 Extracting Copula Distributions from Autoregressive Models

At step $t$, to sample $\bm{x}_{t}$ conditioned on $\bm{x}_{t+1}$, we need a copula distribution ${p}_{\mathrm{copula}}(\mathbf{X}_{t})$ that closely approximates ${q}(\mathbf{X}_{t}|\bm{x}_{t+1})$. While this might suggest that the copula model should also be trained with a diffusion model objective, which brings us back to the problem of modeling inter-variable dependencies, we show that any model trained on the clean data distribution can serve as a copula model that indirectly approximates ${q}(\mathbf{X}_{t}|\bm{x}_{t+1})$ under the absorbing mask forward noising process.

The absorbing mask noising process gradually converts data tokens in $\bm{x}_{0}\!\sim\!{p}(\mathbf{X}_{0})$ to a new category denoted <MASK> through the sequence $\bm{x}_{1},\dots,\bm{x}_{T}$. Specifically, each token in $\bm{x}_{0}$ is independently converted to <MASK> with probabilities $0\!<\!\alpha_{1}\!<\!\dots\!<\!\alpha_{T}\!=\!1$ in $\bm{x}_{1},\dots,\bm{x}_{T}$, respectively. This is a widely used noising strategy for discrete diffusion models. Since this process only transforms data tokens into the mask token, it preserves the dependencies between the remaining unmasked tokens. Therefore, we can decompose ${q}(\mathbf{X}_{t}|\bm{x}_{t+1})$ as ${q}(\bm{x}_{t}|\bm{x}_{t+1})\!=\!\sum_{\tilde{\bm{x}}_{t}}{q}(\tilde{\bm{x}}_{
t}|\bm{x}_{t+1}){q}(\bm{x}_{t}|\tilde{\bm{x}}_{t},\bm{x}_{t+1})$, where ${q}(\tilde{\bm{x}}_{t}|\bm{x}_{t+1})$ is inuitively capturing the joint distribution of generating all currently masked tokens, and ${q}(\bm{x}_{t}|\tilde{\bm{x}}_{t},\bm{x}_{t+1})$ captures only the choice of which currently masked tokens will actually be generated. Formally, define $I$ as the set of variables $i$ such that $x_{t+1}^{i}\!=\!\text{{<MASK>}}$ and $J$ as its complement. The auxiliary distributions have the following form.

###### Proposition 5 .

Assume ${p}(\mathbf{X}_{0})$ is the clean data distribution and $\{{q}(\mathbf{X}_{t}|\bm{x}_{t-1})\}_{t=1}^{T}$ follows the absorbing mask noising process. Let $\alpha_{t}$ be the probability of conversion to the mask state from $X_{0}^{i}$ to $X_{t}^{i}$ ($\forall i$). Define $\tilde{\mathbf{X}}_{t}$ as a set of auxiliary variables such that

$$ $\displaystyle{q}(\tilde{\bm{x}}_{t}|\bm{x}_{t+1})={p}(\mathbf{X}_{0}^{I}= \tilde{\bm{x}}_{t}^{I}|\mathbf{X}_{0}^{J}=\bm{x}_{t+1}^{J})\cdot\mathbbm{1}[ \tilde{\bm{x}}_{t}^{J}=\bm{x}_{t+1}^{J}].$ (5) $$

Then, the distribution ${q}(\mathbf{X}_{t}|\tilde{\bm{x}}_{t},\bm{x}_{t+1})$ is the following: ${q}(\mathbf{X}_{t}|\tilde{\bm{x}}_{t},\bm{x}_{t+1})\!=\!\prod_{i}{q}(x_{t}^{i}
|\tilde{x}_{t}^{i},x_{t+1}^{i})$.

– For $i\!\in\!I$, ${q}(x_{t}^{i}|\tilde{x}_{t}^{i},x_{t+1}^{i})$ equals $\alpha_{t}/\alpha_{t+1}$ if $x_{t}^{i}\!=\!\text{{<MASK>}}$ and equals $1\!-\!\alpha_{t}/\alpha_{t+1}$ if $x_{t}^{i}\!=\!\tilde{x}_{t}^{i}$.

– For $i\!\in\!J$, ${q}(x_{t}^{i}|\tilde{x}_{t}^{i},x_{t+1}^{i})\!=\!1$ if and only if $x_{t}^{i}\!=\!x_{t+1}^{i}$.

Since ${q}(\mathbf{X}_{t}|\tilde{\bm{x}}_{t},\bm{x}_{t+1})$ is fully factorized, the copula model only needs to account for inter-variable dependencies in ${q}(\tilde{\mathbf{X}}_{t}|\bm{x}_{t+1})$. Following Equation 5, we can transform ${p}_{\mathrm{copula}}(\mathbf{X}_{0})$, which estimates the clean data distribution, into ${p}_{\mathrm{copula}}(\tilde{\mathbf{X}}_{t}|\bm{x}_{t+1})$ that approximates ${q}(\tilde{\mathbf{X}}_{t}|\bm{x}_{t+1})$ by conditioning it on the unmasked tokens in $\bm{x}_{t+1}$ (i.e., $\bm{x}_{t+1}^{J}$). Specifically, for autoregressive copula models (i.e., ${p}_{\mathrm{copula}}(\bm{x})\!:=\!\prod_{i}{p}_{\mathrm{copula}}(x_{i}|\bm{x}
_{<i})$), we construct ${p}_{\mathrm{copula}}(\tilde{\mathbf{X}}_{t}|\bm{x}_{t+1})$ by conditioning each variable on the corresponding preceding tokens in $\bm{x}_{t+1}^{J}$ while enforcing $\tilde{x}_{t}^{j}\!=\!x_{t+1}^{j}$ ($\forall j\!\in\!J$):

$$ $\displaystyle{p}_{\mathrm{copula}}(\tilde{\bm{x}}_{t}|\bm{x}_{t+1}):=\prod_{i \in I}{p}_{\mathrm{copula}}(X_{0}^{i}=\tilde{x}_{t}^{i}|\mathbf{X}_{0}^{<i}= \tilde{\bm{x}}_{t}^{<i})\cdot\prod_{j\in J}\mathbbm{1}[\tilde{x}_{t}^{j}=x_{t+ 1}^{j}].$ (6) $$

This copula distribution is biased even if the autoregressive model perfectly captures the data distribution since it cannot condition on subsequent unmasked tokens in $\bm{x}_{t+1}$. In contrast, while being able to condition on all unmasked tokens, diffusion models cannot capture dependence between variables. Combining the two estimates in a proper way will lead to better empirical performance.

Continuing with the example in Figure 2, we assume an autoregressive copula model encodes the probability table on the left. As shown on the right, when provided with the suffix prompt “in Switzerland”, the copula model alone cannot adjust its probabilities, as it can only condition on prefix prompts. However, a diffusion model that captures the strong dependence between “Switzerland” and $Y\!=\!\text{``skiing''}$ can, through I-projection, set the correct marginal probabilities of $Y$, while keeping the copula unchanged. This allows the model to reliably generate “how about alpine skiing.”

Lastly, we need the univariate marginals of ${q}(\tilde{\mathbf{X}}_{t}|\bm{x}_{t+1})$, which can be derived by renormalizing $\{{q}(X_{t}^{i}|\bm{x}_{t+1})\}_{i}$ to zero out the probability of the mask state according to the following result.

###### Proposition 6 .

For each $i$ and data category $c\!\neq\!\text{{<MASK>}}$, ${q}(\tilde{X}_{t}^{i}=c|\bm{x}_{t+1})\propto{q}(X_{t}^{i}=c|\bm{x}_{t+1})$.

As a result, for each $i$, the distribution ${p}_{\mathrm{dm}}(\tilde{X}_{t}^{i}|\bm{x}_{t+1})$ can be similarly obtained by renormalizing ${p}_{\mathrm{dm}}(X_{t}^{i}|\bm{x}_{t+1})$, which is directly obtained from the denoising model, to exclude the mask state.

###### Proposition 5 .

###### Proposition 6 .

### 5.2 Approximate I-Projection with Autoregressive Models

Figure: Algorithm 1 Draw samples from a discrete diffusion model with the help of a copula model

Given univariate marginals $\{{p}_{\mathrm{dm}}(\tilde{X}_{t}^{i}|\bm{x}_{t+1})\}_{i}$ and an autoregressive copula distribution ${p}_{\mathrm{copula}}(\tilde{\mathbf{X}}_{t}|\bm{x}_{t+1})$, both of which estimate the target distribution ${q}(\tilde{\mathbf{X}}_{t}|\bm{x}_{t+1})$, our goal is to combine them following the I-projection procedure described in Section 4.1. Specifically, this involves solving the convex optimization problem in Equation 4, which is specialized to the following:

$$ $\displaystyle\sum_{\tilde{\bm{x}}_{t}}{p}_{\mathrm{copula}}(\tilde{\bm{x}}_{t} |\bm{x}_{t+1})\cdot\prod_{i}\exp(\mathbf{V}[i,\tilde{x}_{t}^{i}])-\sum_{i=1}^{ N}\sum_{\tilde{x}_{t}=1}^{C}\mathbf{V}[i,\tilde{x}_{t}^{i}]\cdot{p}_{\mathrm{ dm}}(\tilde{x}_{t}^{i}|\bm{x}_{t+1}).$ (7) $$

Following Theorem 1, if $\mathbf{V}$ minimizes Equation 7, then the distribution defined by $\hat{{p}}(\tilde{\bm{x}}_{t}|\bm{x}_{t+1})\!=\!{p}_{\mathrm{copula}}(\tilde{
\bm{x}}_{t}|\bm{x}_{t+1})\!\cdot\!\prod_{i}\exp(\mathbf{V}[i,\tilde{x}_{t}^{i}])$ is the I-projection of ${p}_{\mathrm{copula}}(\tilde{\bm{x}}_{t}|\bm{x}_{t+1})$ onto the set of distributions with the univariate marginals $\{{p}_{\mathrm{dm}}(\tilde{X}_{t}^{i}|\bm{x}_{t+1})\}_{i}$, which is the desired combined distribution.

Consider initializing all coefficients in $\mathbf{V}$ to zero, i.e., $\hat{{p}}(\tilde{\bm{x}}_{t}|\bm{x}_{t+1})\!=\!{p}_{\mathrm{copula}}(\tilde{
\bm{x}}_{t}|\bm{x}_{t+1})$. For each row $i$, if we only optimize the values $\mathbf{V}[i,:]$ and fix the rest to zero, the optimal coefficients are

$$ $\displaystyle\forall c,\;\mathbf{V}[i,c]=\log{p}_{\mathrm{dm}}(\tilde{X}_{t}^{ i}=c|\bm{x}_{t+1})-\log{p}_{\mathrm{copula}}(\tilde{X}_{t}^{i}=c|\bm{x}_{t+1}).$ (8) $$

We approximate the solution to Equation 7 by applying the above update (Eq. (8)) to each row in $\mathbf{V}$ independently, as it strikes a proper balance between efficiency and empirical performance.

While the first term on the right-hand side of Equation 8 can be acquired from the diffusion model, the second term is not accessible through the copula model. Plug in the definition in Equation 6, the required marginal probabilities can be written as (for $j\!\in\!J$, ${p}_{\mathrm{copula}}(\tilde{x}_{t}^{j}|\bm{x}_{t+1})\!=\!1$ iff $\tilde{x}_{t}^{j}\!=\!x_{t+1}^{j}$)

$$ $\displaystyle\forall i\!\in\!I,\;{p}_{\mathrm{copula}}(\tilde{x}_{t}^{i}|\bm{x }_{t+1})={p}_{\mathrm{copula}}(X_{i}=\tilde{x}_{t}^{i}|\mathbf{X}_{K_{i}}\!=\! \bm{x}_{t+1}^{K_{i}}),\text{~{}where~{}}K_{i}\!=\!\{j:j\!\in\!J\text{~{}and~{} }j\!<\!i\}.$ (9) $$

The above probabilities cannot be computed from the autoregressive model since we need to “marginalize out” preceding tokens that are not in $K_{i}$ (i.e., those not given as evidence in $\bm{x}_{t+1}$). However, these terms can be estimated using the diffusion model. Assume both the diffusion model and the autoregressive model perfectly encode the data distribution. According to Proposition 6, the diffusion model computes ${p}_{\mathrm{dm}}(\tilde{X}_{t}^{i}|\bm{x}_{t+1})\!=\!{q}(\tilde{X}_{t}^{i}|
\bm{x}_{t+1})$. Comparing it to Equation 9, which gives ${p}_{\mathrm{copula}}(\tilde{X}_{t}^{i}|\bm{x}_{t+1})\!=\!{q}(\tilde{X}_{t}^{i
}|\bm{x}_{t+1}^{K_{i}})$, we only need to additionally restrict the diffusion model to only condition on preceding unmasked tokens in $\bm{x}_{t+1}$, since $K_{i}$ is the intersection of $J$ and $\{j:j\!<\!i\}$. Therefore, if both models well-approximate the data distribution, we have ${p}_{\mathrm{copula}}(\tilde{x}_{t}^{i}|\bm{x}_{t+1})\!\approx\!{q}(\tilde{x}_
{t}^{i}|\bm{x}_{t+1}^{K_{i}})\!=\!{q}(\tilde{x}_{t}^{i}|\bm{x}_{t+1}^{<i})\!
\approx\!{p}_{\mathrm{dm}}(\tilde{x}_{t}^{i}|\bm{x}_{t+1}^{<i})$, where the equality holds since all values in $\bm{x}_{t+1}^{<i}$ but not in $\bm{x}_{t+1}^{K_{i}}$ are <MASK>, and does not “contribute to” the distribution of $\tilde{X}_{t}^{i}$ according to Proposition 5). Correspondingly, we update $\mathbf{V}$ following

$$ $\displaystyle\forall i,c,\;\mathbf{V}[i,c]=\log{p}_{\mathrm{dm}}(\tilde{X}_{t} ^{i}=c|\bm{x}_{t+1})-\log{p}_{\mathrm{dm}}(\tilde{X}_{t}^{i}=c|\bm{x}_{t+1}^{< i}).$ (10) $$

For denoising neural networks that are implemented with bidirectional Transformers, we can simply apply causal attention masks to the self-attention layers to obtain $\{{p}_{\mathrm{dm}}(\tilde{X}_{t}^{i}|\bm{x}_{t+1}^{<i})\}_{i}$.

### 5.3 The Overall Diffusion Sampling Process

Given a diffusion model ${p}_{\mathrm{dm}}$ and an autoregressive copula model ${p}_{\mathrm{copula}}$, the sampling procedure is outlined in Algorithm 1. First, we sample $\bm{x}_{T}$ from the prior noise distribution ${p}(\mathbf{X}_{T})$ (line 3). During each denoising step $t$, we compute the univariate marginals $\{{p}_{\mathrm{dm}}(\tilde{X}_{t}^{i}|\bm{x}_{t+1})\}_{i}$ and $\{{p}_{\mathrm{dm}}(\tilde{X}_{t}^{i}|\bm{x}_{t+1}^{<i})\}_{i}$ based on the previously obtained $\bm{x}_{t+1}$ (line 5). These marginals are then used to compute the entries in $\mathbf{V}$ (line 6), which approximates the I-projection of ${p}_{\mathrm{copula}}(\tilde{\mathbf{X}}_{t}|\bm{x}_{t+1})$ onto the set of distributions with univariate marginals $\{{p}_{\mathrm{dm}}(\tilde{X}_{t}^{i}|\bm{x}_{t+1})\}_{i}$ (cf. Sec. 5.2).

Afterwards, we sample $\tilde{\bm{x}}_{t}$ from the combined distribution $\hat{{p}}(\tilde{\mathbf{X}}_{t}|\bm{x}_{t+1})$ (line 7). Specifically, following Equation 6, we sample autoregressively following $\hat{{p}}(\tilde{\bm{x}}_{t}|\bm{x}_{t+1})\!=\!\prod_{i}\hat{{p}}(\tilde{x}_{t
}^{i}|\bm{x}_{t+1},\tilde{\bm{x}}_{t}^{<i})$, where

$$ $\displaystyle\hat{{p}}(\tilde{x}_{t}^{i}|\bm{x}_{t+1},\tilde{\bm{x}}_{t}^{<i}) \propto{p}_{\mathrm{copula}}(X_{i}=\tilde{x}_{t}^{i}|\mathbf{X}_{<t}=\tilde{ \bm{x}}_{t}^{<i})\cdot\exp(\mathbf{V}[i,\tilde{x}_{t}^{i}])\cdot\mathbbm{1}[ \tilde{x}_{t}^{i}=x_{t+1}^{i}].$ $$

Finally, we sample $\bm{x}_{t}$ from ${q}(\mathbf{X}_{t}|\tilde{\bm{x}}_{t},\bm{x}_{t+1})$ (line 8) as defined in Proposition 5. To improve the algorithm’s efficiency, we introduce a variant that unmasks tokens in an autoregressive manner. Specifically, at step $t$, all tokens except the first $(T\!-\!t)/T$ portion of the tokens in $\bm{x}_{t}$ are converted to <MASK>. Since $\hat{{p}}$ is sampled autoregressively, this allows us to use techniques such as KV-caching for autoregressive Transformers to significantly reduce computation cost introduced by the copula model. See Appendix E for details and the concrete algorithm.

## 6 Experiments

We empirically validate the proposed method, Discrete Copula Diffusion (DCD), on language modeling tasks (Sec. 6.1 and 6.2) and antibody sequence infilling tasks (Sec. 6.3). For all tasks, we evaluate whether DCD can effectively reduce the number of diffusion steps while maintaining strong performance. Specifically, since DCD combines two pretrained models: a discrete diffusion model and an autoregressive copula model, we examine whether DCD outperforms each individual model.

### 6.1 Unconditional Text Generation

We first compare the quality of unconditional samples generated by models trained on either WebText or OpenWebText , which contain web content extracted from URLs shared on Reddit with a minimum number of upvotes. We adopt the medium-sized SEDD model ($\text{SEDD}_{\text{{M}}}$) since it is a SoTA discrete diffusion model for text generation. The GPT-2-small model ($\text{GPT-2}_{\text{{S}}}$) serves as the copula model.

We generate samples of $128$ tokens each. Following , we evaluate sample quality using their generative perplexity, which is the perplexity of the samples when evaluated with the GPT-2-large model. Since previous studies have observed that this metric can be affected by distribution annealing methods such as nucleus sampling, we always sample directly from the models. $\text{SEDD}_{\text{{M}}}$ is evaluated with $2$ to $256$ diffusion steps and DCD (i.e., $\text{SEDD}_{\text{{M}}}$ with $\text{GPT-2}_{\text{{S}}}$ as the copula model) is run with diffusion steps ranging from $2$ to $32$. We adopt the log-linear noise schedule suggested by the SEDD paper. See Section G.1 for more details.

Figure: Figure 3: Generative perplexity ($\downarrow$) with different numbers of denoising steps.
Refer to caption: x3.png

For each configuration, we draw 10,000 samples and report the average perplexity in Figure 4. First, when fixing the number of denoising steps between $2$ to $32$, we observe that DCD outperforms both $\text{SEDD}_{\text{{M}}}$ with the same number of denoising steps and $\text{GPT-2}_{\text{{S}}}$. This provides empirical validation of the effectiveness of the I-projection procedure for modeling inter-variable dependencies.

Additionally, DCD with just $4$ denoising steps achieves performance comparable to $\text{SEDD}_{\text{{M}}}$ with $128$ steps, representing a $32$x reduction in the number of denoising steps. This result not only demonstrates the efficiency of DCD but also underscores the importance of modeling inter-variable dependencies in discrete diffusion models, particularly in few-step generation settings.

Finally, as shown in Figure 4, SEDD fails to generate fluent and meaningful sentences given only a few diffusion steps, as too many tokens have to be generated in each step. In contrast, by modeling the inter-variable dependencies, DCD generates smooth sentences with only $4$ denoising steps.

Efficiency.
We compare the sample time and the generative perplexity of DCD against competitive baselines in Figure 6. We additionally adopt another recent discrete diffusion baseline MDLM . We adopt the autoregressive version of DCD as described in Section 5.3 and Appendix E. Compared to the baselines, DCD consistently achieves better generative perplexity given a fixed runtime constraint. It also requires less time to reach a desired perplexity value. We defer a comprehensive study of DCD’s efficiency to Appendix F.

### 6.2 Conditional Text Generation

We now move on to conditional text generation, where certain tokens are provided in advance, and the task is to generate the remaining tokens. As shown in the first column of Table 1, we use five mask strategies, where tokens in specific prompt ranges are given (we use a sequence length of $128$). We adopt the MAUVE score with the default settings to compare the difference between the generated and original texts. See Section G.2 for further details.

For all methods, we use the same set of 2,000 text sequences from the validation set of WikiText-103 . After applying the prompt mask, we generate $5$ samples for each prompt, resulting in a total number of 10,000 samples.

In addition to $\text{SEDD}_{\text{{M}}}$ and $\text{GPT-2}_{\text{{S}}}$, we compare against SSD-LM , which is a semi-autoregressive diffusion model designed for text infilling. We adopt the autoregressive unmasking variant of DCD described in the last paragraph of Section 5.3.

Results are presented in Table 1. First, DCD outperforms all three baselines in all five tasks. Additionally, when fixing the number of denoising steps between 2 and 32, DCD surpasses both of its base models. Notably, while both $\text{GPT-2}_{\text{{S}}}$ and the 2-step $\text{SEDD}_{\text{{M}}}$ performs poorly on the first, the second, and the fifth tasks, combining them in a principled way allows DCD to achieve significantly better performance using only two denoising steps.

**Table 1: Evaluation of text infilling performance using the MAUVE score ($\uparrow$) with 5 prompt masks. Scores of DCD are all better than (i) SEDD with the same # denoising steps, and (ii) $\text{GPT-2}_{\text{{S}}}$.**
| Prompt ranges<br>(remainder is masked) | SSD-LM | $\text{GPT-2}_{\text{{S}}}$ | $\text{SEDD}_{\text{{M}}}$ | DCD (ours) |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 500 | N/A | 2 | 4 | 8 | 16 | 32 | 2 | 4 | 8 | 16 | 32 |  |
| [0.1,0.2] & [0.5,0.7] | 0.057 | 0.083 | 0.079 | 0.013 | 0.051 | 0.122 | 0.152 | 0.201 | 0.158 | 0.187 | 0.185 | 0.195 | 0.211 |
| [0.25,0.75] | 0.072 | 0.108 | 0.188 | 0.027 | 0.110 | 0.226 | 0.237 | 0.278 | 0.249 | 0.251 | 0.257 | 0.314 | 0.298 |
| [0.0,0.1] & [0.4,0.6] & [0.9,1.0] | 0.333 | 0.681 | 0.928 | 0.827 | 0.940 | 0.972 | 0.980 | 0.979 | 0.962 | 0.976 | 0.979 | 0.982 | 0.983 |
| [0.4,0.5] & [0.8,1.0] | 0.436 | 0.565 | 0.914 | 0.896 | 0.944 | 0.978 | 0.978 | 0.980 | 0.963 | 0.975 | 0.975 | 0.976 | 0.981 |
| [0.2,0.3] & [0.6,0.8] | 0.041 | 0.054 | 0.069 | 0.016 | 0.056 | 0.128 | 0.207 | 0.215 | 0.171 | 0.178 | 0.215 | 0.217 | 0.403 |

### 6.3 Antibody Sequence Infilling

We consider the task of unguided antibody infilling, where certain complementarity determining regions (CDRs) of antibodies (i.e., sequences of amino acids) are missing and to be generated by the model. We adopt NOC-D , which is a discrete diffusion model trained on 104K antibody sequences from the Observed Antibody Space dataset . We further train a GPT model on the same dataset as the copula model. See Section G.3 for training details.

We follow to select the same 10 antibody seed sequences from paired OAS . We consider two infilling tasks: (i) three CDRs $\{\text{HCDR1},\text{HCDR2},\text{HCDR3}\}$ are masked, and (ii) two CDRs $\{\text{HCDR1},\text{LCDR1}\}$ are masked. We follow the original paper and run $64$ diffusion steps for NOS-D. For DCD (i.e., combining NOS-D with the trained GPT model as the copula model), we use $4$ denoising steps. We measure the sequence recovery rate, i.e., the accuracy of the infilled sequences given the ground truth sequence.

As shown in Figure 6, by combining the univariate marginals from NOS-D and the dependencies captured by the GPT model, DCD can also perform well in antibody sequence infilling tasks.

Figure: Figure 5: Sampling time vs. generative perplexity (the autoregressive version of DCD is used).
Refer to caption: x5.png

## 7 Related Work and Conclusion

Diffusion models have been widely applied to model discrete data such as text and DNA sequences. Encouraged by the successes of continuous diffusion models (e.g., ), initial attempts convert discrete data into continuous embeddings with either predefined or learned mappings. This enables the use of continuous diffusion models for discrete data . However, due to the need for ad-hoc mappings between the discrete data space and the continuous embedding space, which have to be pre-defined or pre-trained, continuous diffusion models are not as effective for modeling discrete distributions .

proposed the first diffusion model designed directly for discrete data. Later works further improved discrete diffusion models from various aspects such as better loss functions/learning objectives , better model architectures , better sampling algorithms , and unifying and scaling up existing techniques .

Despite the recent breakthroughs of discrete diffusion models, few papers address the challenge of sampling in a few denoising steps. Some works attribute the failure to perform high-quality few-step generation to a scaling problem of the model. However, we show that the fundamental problem lies in the assumption made by discrete diffusion models that each variable is denoised independently at each step. In addition to identifying this problem, we propose a general solution Discrete Copula Diffusion that combines a discrete diffusion model with a copula model at inference time to obtain a better estimate of the denoising distribution at each step. Concurrently, show that energy-based models can also be used as copula models to capture inter-variable dependencies.

There are a few limitations of DCD. First, in addition to a discrete diffusion model, it requires another copula model, which may require additional training for certain applications. Second, although the I-projected distribution is guaranteed as a better estimate of the target distribution, the I-projection step often needs to be approximated in practice. Finally, although DCD requires fewer denoising steps, the computation cost of each step is higher than in discrete diffusion models. Therefore, DCD may not always provide a notable speedup. However, DCD points out the inter-dependency modeling problem and opens up the possibility of combining different types of generative models for better overall performance.

## Acknowledgements

This work was funded in part by the DARPA ANSR program under award FA8750-23-2-0004, the DARPA CODORD program under award HR00112590089, the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy - EXC 2075 – 390740016, NSF grant #IIS-1943641, and gifts from Adobe Research and Amazon. We acknowledge the support of the Stuttgart Center for Simulation Science (SimTech). MN thanks IMPRS-IS (International Max Planck Research School for Intelligent Systems) for the support.

## Appendix A Proof of the Theoretical Results

###### Proof of Proposition 1 .

Following , the negative ELBO $\mathcal{L}$ can be decomposed as follows:

$$ $\displaystyle\mathcal{L}$ $\displaystyle=\mathbb{E}_{{q}}\left[-\log{p}(\bm{x}_{T})-\sum_{t=1}^{T}\log \frac{{p}_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})}{q(\bm{x}_{t}|\bm{x}_{t-1})}\right],$ $\displaystyle=\mathbb{E}_{{q}}\left[-\log{p}(\bm{x}_{T})-\sum_{t=1}^{T}\log \frac{{p}_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})\cdot{q}(\bm{x}_{t-1})}{q(\bm{x}_{t -1}|\bm{x}_{t})\cdot{q}(\bm{x}_{t})}\right],$ $\displaystyle=\mathbb{E}_{{q}}\left[-\log\frac{{p}(\bm{x}_{T})}{{q}(\bm{x}_{T} )}-\sum_{t=1}^{T}\log\frac{{p}_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})}{q(\bm{x}_{t- 1}|\bm{x}_{t})}-\log{p}(\bm{x}_{0})\right],$ $\displaystyle=\mathrm{D}_{\mathrm{KL}}({q}(\bm{x}_{T})\;\|\;{p}(\bm{x}_{T}))+ \mathbb{E}_{q}\left[\sum_{t=1}^{T}\mathrm{D}_{\mathrm{KL}}(q(\bm{x}_{t-1}|\bm{ x}_{t})\;\|\;{p}_{\theta}(\bm{x}_{t-1}|\bm{x}_{t}))\right]+\mathrm{H}(\bm{x}_{ 0}).$ (11) $$

The first term equals 0 0 as we assume the noise distribution ${p}(\mathbf{X}_{T})$ is consistent in the noising and the denoising processes. Given the independent denoising assumption, when the denoising distribution are optimal, we have

$$ $\displaystyle\forall t\in\{1,\dots,T\},\;{p}_{\theta}(\bm{x}_{t-1}|\bm{x}_{t}) =\prod_{i}{q}(x_{t-1}^{i}|\bm{x}_{t}).$ $$

Plug in Equation 11 and using the definition of total correlation, we have:

$$ $\displaystyle\mathcal{L}$ $\displaystyle=\mathrm{D}_{\mathrm{KL}}({q}(\bm{x}_{T})\;\|\;{p}(\bm{x}_{T}))+ \mathbb{E}_{q}\left[\sum_{t=1}^{T}\mathrm{D}_{\mathrm{KL}}(q(\bm{x}_{t-1}|\bm{ x}_{t})\;\|\;\prod_{i}{q}(x_{t-1}^{i}|\bm{x}_{t}))\right]+\mathrm{H}(\bm{x}_{0})$ $\displaystyle=\mathrm{D}_{\mathrm{KL}}({q}(\bm{x}_{T})\;\|\;{p}(\bm{x}_{T}))+ \sum_{t=1}^{T}\mathrm{D}_{\mathrm{TC}}(q(\mathbf{X}_{t-1}|\mathbf{X}_{t}))+ \mathrm{H}({p}(\mathbf{X}_{0}))$ $\displaystyle\geq\mathrm{H}({p}(\mathbf{X}_{0}))+\sum_{t=1}^{T}\mathrm{D}_{ \mathrm{TC}}(q(\mathbf{X}_{t-1}|\mathbf{X}_{t})).$ $$

∎

###### Proof of Proposition 2 .

According to Pythagoras’ triangle-inequality theorem, if $\hat{{p}}$ is the I-projection of ${p}_{\mathrm{est}}$ onto $\mathcal{P}_{\mathrm{mar}}^{{p}_{\mathrm{tar}}}$, and $\mathcal{P}_{\mathrm{mar}}^{{p}_{\mathrm{tar}}}$ is convex (this can be shown by applying the definition of a convex set), the following holds for any ${p}^{\prime}\!\in\!\mathcal{P}_{\mathrm{mar}}^{{p}_{\mathrm{tar}}}$:

$$ $\displaystyle\mathrm{D}_{\mathrm{KL}}({p}^{\prime}\;\|\;{p}_{\mathrm{est}}) \geq\mathrm{D}_{\mathrm{KL}}({p}^{\prime}\;\|\;\hat{{p}})+\mathrm{D}_{\mathrm{ KL}}(\hat{{p}}\;\|\;{p}_{\mathrm{est}}).$ (12) $$

Choosing ${p}^{\prime}\!=\!{p}_{\mathrm{tar}}$, we have

$$ $\displaystyle\mathrm{D}_{\mathrm{KL}}({p}_{\mathrm{tar}}\;\|\;\hat{{p}})\leq \mathrm{D}_{\mathrm{KL}}({p}_{\mathrm{tar}}\;\|\;{p}_{\mathrm{est}})-\mathrm{D }_{\mathrm{KL}}(\hat{{p}}\;\|\;{p}_{\mathrm{est}})<\mathrm{D}_{\mathrm{KL}}({p }_{\mathrm{tar}}\;\|\;{p}_{\mathrm{est}}),$ $$

where the last inequality holds since $\mathrm{D}_{\mathrm{KL}}(\hat{{p}}\;\|\;{p}_{\mathrm{est}})\!>\!0$ if the set of univariate marginals of ${p}_{\mathrm{est}}$ and ${p}_{\mathrm{tar}}$ are different (as assumed in the proposition).

∎

###### Proof of Proposition 3 .

Following the definition of $\hat{{p}}$, we write down the constrained optimization problem as follows

$$ $\displaystyle\operatorname*{minimize}_{{p}^{\prime}}\;\mathrm{D}_{\mathrm{KL}} ({p}^{\prime}\;\|\;{p}_{\mathrm{est}})$ $\displaystyle\text{s.t.~{}}\forall i\in\{1,\dots,N\}$ $\displaystyle,x_{i}\in\{1,\dots,C\},\;\sum_{\bm{x}_{\backslash i}}{p}^{\prime} (\bm{x}_{\backslash i},x_{i})={p}_{\mathrm{tar}}(x_{i}).$ $$

To incorporate the constraints, we use the method of Lagrange multipliers. The Lagrangian for this problem is

$$ $\displaystyle\mathcal{L}({p}^{\prime},\{\lambda_{i}\}_{i=1}^{N})=\sum_{\bm{x}} {p}^{\prime}(\bm{x})\log\frac{{p}^{\prime}(\bm{x})}{{p}_{\mathrm{est}}(\bm{x}) }+\sum_{i=1}^{N}\sum_{x_{i}=1}^{C}\lambda_{i}(x_{i})\cdot\left(\sum_{\bm{x}_{ \backslash i}}{p}^{\prime}(\bm{x}_{\backslash i},x_{i})-{p}_{\mathrm{tar}}(x_{ i})\right),$ $$

where the Lagrange multipliers $\{\lambda_{i}\}_{i=1}^{N}$ enforce the univariate marginal constraints.

To minimize the Lagrangian with respect to ${p}^{\prime}(\bm{x})$, we take the partial derivative of $\mathcal{L}({p}^{\prime},\{\lambda_{i}\}_{i=1}^{N})$ with respect to ${p}^{\prime}(\bm{x})$ and set it to 0 0:

$$ $\displaystyle\frac{\partial\mathcal{L}({p}^{\prime},\{\lambda_{i}\}_{i=1}^{N}) }{\partial{p}^{\prime}(\bm{x})}=\log\frac{{p}^{\prime}(\bm{x})}{{p}_{\mathrm{ est}}(\bm{x})}+1+\sum_{i}\lambda_{i}(x_{i})=0.$ $$

Simplifying this equation gives

$$ $\displaystyle{p}^{\prime}(\bm{x})={p}_{\mathrm{est}}(\bm{x})\cdot\exp\left(-1- \sum_{i}\lambda_{i}(x_{i})\right).$ $$

Defining $\sigma_{i}(x_{i})\!:=\!\exp(-\lambda_{i}(x_{i})-1/N)$ gives ${p}^{\prime}(\bm{x})={p}_{\mathrm{est}}(\bm{x})\prod_{i}\sigma_{i}(x_{i})$.

Existence of the solution follows from the fact that (i) the objective function is convex and bounded (since probability values are in $[0,1]$), and (ii) the set of constraints is feasible (e.g., ${p}^{\prime}(\bm{x})\!=\!\prod_{i}{p}_{\mathrm{tar}}(x_{i})$ or ${p}^{\prime}(\bm{x})\!=\!{p}_{\mathrm{tar}}(\bm{x})$).

∎

###### Proof of Theorem 1 .

We show that for any $\mathbf{V}^{*}$ that minimizes the objective function $\mathcal{L}(\mathbf{V};{p}_{\mathrm{tar}},{p}_{\mathrm{est}})$, the corresponding ${p}^{\prime}$ defined by ${p}^{\prime}(\bm{x})\!=\!{p}_{\mathrm{est}}(\bm{x})\cdot\prod_{i}\exp(\mathbf{
V}[i,x_{i}])$ belongs to the set $\mathcal{P}_{\mathrm{mar}}^{{p}_{\mathrm{tar}}}$. Specifically, for any $\mathbf{V}$ that minimizes the objective, the partial derivative of $\mathcal{L}(\mathbf{V};{p}_{\mathrm{tar}},{p}_{\mathrm{est}})$ with respect to any $\mathbf{V}[i,x_{i}]$ should be 0 0:

$$ $\displaystyle\frac{\partial\mathcal{L}(\mathbf{V};{p}_{\mathrm{tar}},{p}_{ \mathrm{est}})}{\partial\mathbf{V}[i,x_{i}]}=\exp(\mathbf{V}[i,x_{i}])\sum_{ \bm{x}_{\backslash i}}{p}_{\mathrm{est}}(\bm{x}_{\backslash i},x_{i})\prod_{j \neq i}\exp(\mathbf{V}[j,x_{j}])-{p}_{\mathrm{tar}}(x_{i})=0.$ $$

Plug in the definition of ${p}^{\prime}$, we have

$$ $\displaystyle 0=\sum_{\bm{x}_{\backslash i}}{p}^{\prime}(\bm{x}_{\backslash i} ,x_{i})-{p}_{\mathrm{tar}}(x_{i})={p}^{\prime}(x_{i})-{p}_{\mathrm{tar}}(x_{i}).$ (13) $$

Since Equation 13 holds for all $(i,x_{i})$ pairs, we have that every minimizer of $\mathcal{L}(\mathbf{V};{p}_{\mathrm{tar}},{p}_{\mathrm{est}})$ corresponds to a distribution ${p}^{\prime}$ in $\mathcal{P}_{\mathrm{mar}}^{{p}_{\mathrm{tar}}}$. Since $\mathcal{L}(\mathbf{V};{p}_{\mathrm{tar}},{p}_{\mathrm{est}})$ is convex, we can also argue the converse: if a distribution ${p}^{\prime}$ with the above-defined form belongs to $\mathcal{P}_{\mathrm{mar}}^{{p}_{\mathrm{tar}}}$, then the corresponding $\mathbf{V}$ is a minimizer of $\mathcal{L}(\mathbf{V};{p}_{\mathrm{tar}},{p}_{\mathrm{est}})$.

According to Proposition 3, the solution to the following I-projection exists and its solution $\hat{{p}}$ has the same form as ${p}^{\prime}$.

$$ $\displaystyle\hat{{p}}=\operatorname*{arg\,min}_{{p}^{\prime}\in\mathcal{P}_{ \mathrm{mar}}^{{p}}}\mathrm{D}_{\mathrm{KL}}({p}^{\prime}\;\|\;{p}_{\mathrm{ est}}).$ $$

Since $\hat{{p}}$ has the same form as ${p}^{\prime}$ (by Prop. 3) and belongs to $\mathcal{P}_{\mathrm{mar}}^{{p}_{\mathrm{tar}}}$, it is the a minimizer of $\mathcal{L}(\mathbf{V};{p}_{\mathrm{tar}},{p}_{\mathrm{est}})$.
∎

###### Proof of Proposition 4 .

The copula of $p$ is shown to be invariant under rescalings of the form $q(\bm{x})\propto{p}(\bm{x})\cdot\prod_{i}\exp(\mathbf{V}[i,x_{i}])$ for any $\mathbf{V}\in\mathbb{R}^{N\times C}$ by using the parameterization of a discrete copula by conditional odds ratios (Definition 2). The scaling factors cancel in the ratios as shown, e.g. by .
∎

###### Proof of Proposition 5 .

We start by writing the probability ${q}(\bm{x}_{t}|\bm{x}_{t+1})$ using the Bayes’ rule:

$$ $\displaystyle{q}(\bm{x}_{t}|\bm{x}_{t+1})$ $\displaystyle={q}(\bm{x}_{t+1}|\bm{x}_{t})\cdot\frac{{q}(\bm{x}_{t})}{{q}(\bm{ x}_{t+1})},$ $\displaystyle=\sum_{\bm{x}_{0}}\frac{1}{{q}(\bm{x}_{t+1})}\cdot{q}(\bm{x}_{t+1 }|\bm{x}_{t})\cdot{q}(\bm{x}_{t}|\bm{x}_{0})\cdot{p}(\bm{x}_{0}),$ (14) $$

where the last equality follows from ${q}(\bm{x}_{t})\!=\!\sum_{\bm{x}_{0}}{q}(\bm{x}_{t}|\bm{x}_{0})\!\cdot\!{p}(
\bm{x}_{0})$. Recall from the proposition that $I$ is defined as the set of variables $i$ such that $x_{t+1}^{i}\!=\!\text{{<MASK>}}$ and $J$ is the complement of $I$.

First, we must have $x_{t}^{j}\!=\!x_{t+1}^{j}$ for $j\!\in\!J$ since for any other value of $X_{t}^{j}$, we have ${q}(\bm{x}_{t+1}|\bm{x}_{t})\!=\!0$ in Equation 14. As a result, ${q}(\bm{x}_{t}|\bm{x}_{t+1})$ is also zero.

We then move our attention to the variables in $I$. We first consider the probability ${q}(X_{t}^{i}\!=\!\text{{<MASK>}}|\bm{x}_{t+1})$ for any $i\!\in\!I$. Following Equation 14, we have

$$ $\displaystyle{q}(X_{t}^{i}=\text{{<MASK>}}|\bm{x}_{t+1})$ $\displaystyle=\sum_{\bm{x}_{0}}\sum_{\bm{x}_{t}^{\backslash i}}\frac{1}{{q}( \bm{x}_{t+1})}\cdot{q}(\bm{x}_{t+1}|\bm{x}_{t})\cdot{q}(\bm{x}_{t}|\bm{x}_{0}) \cdot{p}(\bm{x}_{0}),$ $\displaystyle=\sum_{\bm{x}_{t}^{\backslash i}}\frac{1}{{q}(\bm{x}_{t+1})}\cdot {q}(\bm{x}_{t+1}|\bm{x}_{t})\cdot{q}(\bm{x}_{t}),$ $\displaystyle=\frac{{q}(X_{t+1}^{i}=\text{{<MASK>}}|X_{t}^{i}=\text{{<MASK>}}) \cdot{q}(X_{t}^{i}=\text{{<MASK>}})}{{q}(X_{t+1}^{i}=\text{{<MASK>}})},$ $\displaystyle=\frac{{q}(X_{t}^{i}=\text{{<MASK>}})}{{q}(X_{t+1}^{i}=\text{{< MASK>}})}=\frac{\alpha_{t}}{\alpha_{t+1}}.$ (15) $$

We then focus on $\mathbf{X}_{t}^{I}\!=\!\bm{x}_{t}^{I}$, where none of the value in $\bm{x}_{t}^{I}$ is <MASK>. Note that we also need to have $\mathbf{X}_{t}^{J}\!=\!\bm{x}_{t+1}^{J}$.

$$ $\displaystyle{q}(\bm{x}_{t}|\bm{x}_{t+1})$ $\displaystyle\propto\sum_{\bm{x}_{0}}{q}(\bm{x}_{t+1}|\bm{x}_{t})\cdot{q}(\bm{ x}_{t}|\bm{x}_{0})\cdot{q}(\bm{x}_{0}),$ $\displaystyle\overset{(a)}{=}{q}(\bm{x}_{t+1}|\bm{x}_{t})\cdot{q}(\mathbf{X}_{ 0}=\bm{x}_{t}),$ $\displaystyle=\left(\frac{\alpha_{t+1}-\alpha_{t}}{1-\alpha_{t}}\right)^{|I|} \cdot{q}(\mathbf{X}_{0}=\bm{x}_{t}),$ $\displaystyle\propto{q}(\mathbf{X}_{0}=\bm{x}_{t}),$ (16) $$

where ${p}(\mathbf{X}_{0})$ is the data distribution; $(a)$ follows from the fact that no value in $\bm{x}_{t}$ is <MASK>, hence $\bm{x}_{0}\!=\!\bm{x}_{t}$; $\frac{\alpha_{t+1}-\alpha_{t}}{1-\alpha_{t}}$ is the probability of transitioning into the mask state from time $t$ to time $t\!+\!1$.

Denote $\tilde{\mathbf{X}}_{t}$ as a set of variables with the same configuration and semantics as $\mathbf{X}_{t}$, with the only difference that the category <MASK> is excluded. By following Equation 16 and apply normalization, we conclude that

$$ $\displaystyle{q}(\tilde{\bm{x}}_{t}|\bm{x}_{t+1})={p}(\mathbf{X}_{0}^{I}= \tilde{\bm{x}}_{t}^{I}|\mathbf{X}_{0}^{J}=\bm{x}_{t+1}^{J})\cdot\mathbbm{1}[ \tilde{\bm{x}}_{t}^{J}=\bm{x}_{t+1}^{J}].$ (17) $$

This matches the definition in Equation 5.

Finally, we verify the correctness of the distribution ${q}(\mathbf{X}_{t}|\tilde{\bm{x}}_{t},\bm{x}_{t+1})$ defined in the proposition by verifying the following for any $\bm{x}_{t}$

$$ $\displaystyle{q}(\bm{x}_{t}|\bm{x}_{t+1})=\sum_{\tilde{\bm{x}}_{t}}{q}(\tilde{ \bm{x}}_{t}|\bm{x}_{t+1})\cdot{q}(\bm{x}_{t}|\tilde{\bm{x}}_{t},\bm{x}_{t+1}).$ (18) $$

Denote $K$ as the set of variables $i$ such that $\bm{x}_{t}\!=\!\text{{<MASK>}}$ and $L$ as its complement. First, if $L\subseteq J$ (i.e., $I\subseteq K$), then both the left-hand side (LHS) and the right-hand sides (RHS) are zero. Specifically, the RHS is zero since according to the definition, $\forall i\!\in\!J\,\&\,i\!\in\!K$, we have ${q}(x_{t}^{i}|\tilde{x}_{t}^{i},x_{t+1}^{i})\!=\!0$.

Next, if $K\subseteq I$, we can decompose ${q}(\bm{x}_{t}|\bm{x}_{t+1})$ as follows

$$ $\displaystyle{q}(\bm{x}_{t}|\bm{x}_{t+1})={q}(\bm{x}_{t}^{I\backslash K}|\bm{x }_{t+1})\cdot\prod_{i\in K}{q}(x_{t}^{i}|\bm{x}_{t+1})\cdot\prod_{j\in J}{q}(x _{t}^{j}|\bm{x}_{t+1}).$ (19) $$

For any $j\!\in\!J$, if $x_{t}^{j}\!\neq\!x_{t+1}^{j}$ then both the LHS and the RHS of Equation 18 are zero. Otherwise we always have ${q}(x_{t}^{j}|\bm{x}_{t+1})\!=\!1$. Therefore, Equation 19 can be further simplified as

$$ $\displaystyle{q}(\bm{x}_{t}|\bm{x}_{t+1})={q}(\bm{x}_{t}^{I\backslash K}|\bm{x }_{t+1})\cdot\prod_{i\in K}{q}(x_{t}^{i}|\bm{x}_{t+1}).$ (20) $$

We then proceed to simplify the RHS of Equation 18:

$$ $\displaystyle\sum_{\tilde{\bm{x}}_{t}}{q}(\tilde{\bm{x}}_{t}|\bm{x}_{t+1}) \cdot{q}(\bm{x}_{t}|\tilde{\bm{x}}_{t},\bm{x}_{t+1}),$ $\displaystyle=$ $\displaystyle\sum_{\tilde{\bm{x}}_{t}^{K}}{q}(\tilde{\bm{x}}_{t}^{K},\tilde{ \bm{x}}_{t}^{I\backslash K}|\bm{x}_{t+1})\cdot\left(\frac{\alpha_{t}}{\alpha_{ t+1}}\right)^{|K|}\cdot\left(\frac{\alpha_{t+1}-\alpha_{t}}{\alpha_{t+1}} \right)^{|I|-|K|},$ $\displaystyle\overset{(a)}{=}$ $\displaystyle\sum_{\tilde{\bm{x}}_{t}^{K}}{q}(\tilde{\bm{x}}_{t}^{K},\tilde{ \bm{x}}_{t}^{I\backslash K}|\bm{x}_{t+1})\cdot\left(\frac{\alpha_{t+1}-\alpha_ {t}}{\alpha_{t+1}}\right)^{|I|-|K|}\cdot\prod_{i\in K}{q}(x_{t}^{i}|\bm{x}_{t+ 1}),$ $\displaystyle=$ $\displaystyle\,{q}(\tilde{\bm{x}}_{t}^{I\backslash K}|\bm{x}_{t+1})\cdot\left( \frac{\alpha_{t+1}-\alpha_{t}}{\alpha_{t+1}}\right)^{|I|-|K|}\cdot\prod_{i\in K }{q}(x_{t}^{i}|\bm{x}_{t+1}),$ $\displaystyle\overset{(b)}{\propto}$ $\displaystyle\,{p}(\mathbf{X}_{0}^{I\backslash K}=\tilde{\bm{x}}_{t}^{I \backslash K},\mathbf{X}_{0}^{J}=\tilde{\bm{x}}_{t}^{J})\cdot\prod_{i\in K}{q} (x_{t}^{i}|\bm{x}_{t+1}),$ $\displaystyle\overset{(c)}{\propto}$ $\displaystyle\,{q}(\mathbf{X}_{t}^{I\backslash K}=\tilde{\bm{x}}_{t}^{I \backslash K}|\bm{x}_{t+1})\cdot\prod_{i\in K}{q}(x_{t}^{i}|\bm{x}_{t+1}),$ (21) $$

where $(a)$ follows from Equation 15, $(b)$ applies the definition in Equation 17, and $(c)$ is a result of applying Equation 16 to the case where $\tilde{\bm{x}}_{t}^{L}\!=\!\{\tilde{\bm{x}}_{t}^{I\backslash K},\tilde{\bm{x}}
_{t}^{J}\}$ are not <MASK>.

By combining Equations 20 and 21, we conclude that the LHS and the RHS of Equation 18 are proportional to each other. Since they are both properly-normalized distributions, they must also match exactly.

∎

###### Proof of Proposition 6 .

We first state a more detailed version of the proposition: for each variable $i$ and data category $c$ ($c\!\neq\!\text{{<MASK>}}$), we have

$$ $\displaystyle{q}(\tilde{X}_{t}^{i}=c|\bm{x}_{t+1})=\frac{1}{Z}\cdot{q}(X_{t}^{ i}=c|\bm{x}_{t+1}),\text{~{}where~{}}Z=\sum_{c\neq\text{{<MASK>}}}{q}(X_{t}^{i }=c|\bm{x}_{t+1}).$ $$

According to the proof of Proposition 5, Equation 18 holds for all $\bm{x}_{t}$. Therefore, we have that for each $i$ and each data category $x_{t}^{i}\!\neq\!\text{{<MASK>}}$,

$$ $\displaystyle{q}(x_{t}^{i}|\bm{x}_{t+1})=\sum_{\tilde{\bm{x}}_{t}}{q}(\tilde{ \bm{x}}_{t}|\bm{x}_{t+1})\cdot{q}(x_{t}^{i}|\tilde{\bm{x}}_{t},\bm{x}_{t+1}).$ (22) $$

If $i\!\in\!J$, then both the LHS of the above equation and ${q}(x_{t}^{i}|\tilde{\bm{x}}_{t},\bm{x}_{t+1})$ equals one if and only if $x_{t}^{i}\!=\!x_{t+1}^{i}$. Therefore, the result holds trivially.

Next, if $i\!\in\!I$, denote $I_{\backslash i}\!:=\!I\backslash\{i\}$, Equation 22 is simplified to

$$ $\displaystyle{q}(x_{t}^{i}|\bm{x}_{t+1})$ $\displaystyle=\sum_{\tilde{\bm{x}}_{t}}{q}(\tilde{\bm{x}}_{t}|\bm{x}_{t+1}) \cdot{q}(x_{t}^{i}|\tilde{\bm{x}}_{t},\bm{x}_{t+1}),$ $\displaystyle=\sum_{\tilde{x}_{t}^{i}}\sum_{\tilde{\bm{x}}_{t}^{I_{\backslash i }}}{q}(\tilde{x}_{t}^{i},\tilde{\bm{x}}_{t}^{I_{\backslash i}}|\bm{x}_{t+1}) \cdot{q}(x_{t}^{i}|\tilde{x}_{t}^{i},x_{t+1}^{i}),$ $\displaystyle={q}(\tilde{X}_{t}^{i}=x_{t}^{i}|\bm{x}_{t+1})\cdot{q}(x_{t}^{i}| \tilde{X}_{t}^{i}=x_{t}^{i},x_{t+1}^{i}),$ $\displaystyle={q}(\tilde{X}_{t}^{i}=x_{t}^{i}|\bm{x}_{t+1})\cdot\frac{\alpha_{ t+1}-\alpha_{t}}{\alpha_{t+1}}.$ $$

Therefore, we have

$$ $\displaystyle{q}(\tilde{X}_{t}^{i}=x_{t}^{i}|\bm{x}_{t+1})=\frac{1}{Z}\cdot{q} (X_{t}^{i}=x_{t}^{i}|\bm{x}_{t+1}),\text{~{}where~{}}Z=\sum_{x_{t}^{i}\neq \text{{<MASK>}}}{q}(X_{t}^{i}=x_{t}^{i}|\bm{x}_{t+1}).$ $$

∎

###### Proof of Proposition 1 .

###### Proof of Proposition 2 .

###### Proof of Proposition 3 .

###### Proof of Theorem 1 .

###### Proof of Proposition 4 .

###### Proof of Proposition 5 .

###### Proof of Proposition 6 .

## Appendix B Relation Between ℒ ⁢ ( 𝐕 ; p tar , p est ) ℒ 𝐕 subscript 𝑝 tar subscript 𝑝 est \mathcal{L}(\mathbf{V};{p}_{\mathrm{tar}},{p}_{\mathrm{est}}) caligraphic_L ( bold_V ; italic_p start_POSTSUBSCRIPT roman_tar end_POSTSUBSCRIPT , italic_p start_POSTSUBSCRIPT roman_est end_POSTSUBSCRIPT ) and Matrix Scaling

The matrix scaling problem gives a matrix $A$ as input and asks for diagonal ‘scaling’ matrices $X$ and $Y$ such that $XAY$ is doubly stochastic (its row and column sums are all one). More generally, target row and column sum vectors $r$ and $c$ are provided and need not contain only ones. The solvability of this problem for positive matrices was established by , and its algorithms (sometimes called iterative proportional fitting), generalizations, and numerous applications have been studied thoroughly ; see for a review. Taking the multidimensional generalization of the problem and interpreting the tensor as a (unnormalized) probability distribution yields the connection to our problem, with the target sums being the univariate marginal distributions.

## Appendix C Parameterizing Discrete Copulas by Odds Ratios

We start by formally defining odds ratios.

###### Definition 2 ( Rudas ( 2018 ) ) .

Let $p$ be a distribution over variables $\bm{X}$ each taking values in $\{0,1\}$. For a partition of $\bm{X}$ into sets $\bm{A}$ and $\bm{B}$, the *conditional odds ratio* of variables $\bm{A}$ conditioned on the assignment $\bm{B}=\bm{b}$ is

$$ $\text{COR}_{p}(\bm{A}|\bm{B}=\bm{b})=\frac{\prod_{\bm{a}\in s}p(\bm{a},\bm{b}) }{\prod_{\bm{a}\in d}p(\bm{a},\bm{b})}$ $$

where $s$ is the set of assignments to $\bm{A}$ whose parity is the same as the number of variables in $\bm{A}$, and $d$ is the set of assignments whose parity is different.

In the case of more than two categories per variable, $\text{COR}_{p}(\bm{A}|\bm{B}=\bm{b})$ can generalized further to be a set of similarly defined ratios (see, e.g., ). Together the set of all conditional odds ratios $\text{COR}_{p}(\bm{A}|\bm{B}=\bm{b})$ for partitions of $\bm{X}$ into sets $\bm{A}$ and $\bm{B}$ with $|\bm{A}|\geq 2$, completely specifies the association among the variables in the joint distribution $p$, as established by the following theorem.

###### Theorem 2 ( Rudas ( 2018 ) ) .

Let $q$ and $r$ be positive probability distributions on a the set of variables $\bm{X}$ each taking values in $\{0,1,\ldots,k\}$. Then there exists a unique probability distribution $p$ such that $p$ has the same univariate marginal distributions as $q$, that is, for all $i$

$$ $p(x_{i})=q(x_{i}),$ $$

and $p$ has the same copula as $q$, that is for all partitions of $\bm{X}$ into sets $\bm{A}$ and $\bm{B}$ with $|\bm{A}|\geq 2$,

$$ $\text{COR}_{p}(\bm{A}|\bm{B}=\bm{b})=\text{COR}_{r}(\bm{A}|\bm{B}=\bm{b}).$ $$

###### Proof.

This follows from by taking the descending set to contain the empty set and all singletons (and the ascending set, its complement).
∎

Theorem 2 shows how any distribution $p$ can be viewed as combining independent marginal distributions (i.e., from $r$) and odds ratios (i.e., from $q$).
Such a combination has desirable properties. For example, in the case of two variables with possibly many categories, it has been shown that among all distributions with the same margins as $r$, the distribution $p$ minimizes the KL-divergence to $q$ , i.e. that $p$ is the information projection of $q$ onto the set of distributions with the margins of $r$.

###### Definition 2 ( Rudas ( 2018 ) ) .

###### Theorem 2 ( Rudas ( 2018 ) ) .

###### Proof.

## Appendix D Unbiased Univariate Marginals from Discrete Diffusion Models

In this section, we show that when their respective training losses are minimized, discrete-time and continuous-time discrete diffusion models recover the true univariate marginals.

Discrete-Time Diffusion Models.
Discrete-time diffusion models are trained to maximize the ELBO between the forward joint distribution ${p}(\bm{x}_{0}){q}(\bm{x}_{1:T}|\bm{x}_{0})$, where ${p}(\bm{x}_{0})$ is the data distribution, and the reverse joint distribution ${p}_{\theta}(\bm{x}_{0:T})$. The ELBO can be simplified to

$$ $\displaystyle\mathbb{E}_{{q}}\left[\log\frac{{p}(\bm{x}_{T})}{{q}(\bm{x}_{T})} +\sum_{t=1}^{T}\log\frac{{p}_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})}{{q}(\bm{x}_{t- 1}|\bm{x}_{t})}+\log{p}(\bm{x}_{0})\right].$ $$

Assume that ${p}_{\theta}(\bm{x}_{t-1}|\bm{x}_{t})$ encodes fully-factorized distribution, the above objective can be simplified as

$$ $\displaystyle\sum_{t=1}^{T}\sum_{i}{q}(x_{t-1}^{i}|\bm{x}_{t})\log\frac{{p}_{ \theta}(x_{t-1}^{i}|\bm{x}_{t})}{{q}(x_{t-1}^{i}|\bm{x}_{t})}+\mathbb{E}_{{q}} \left[\log\frac{{p}(\bm{x}_{T})}{{q}(\bm{x}_{T})}+\log{p}(\bm{x}_{0})\right],$ $$

where the second term is independent to ${p}_{\theta}$. From the first term of the above formula, we can conclude that the ELBO objective is maximized when ${p}_{\theta}(x_{t-1}^{i}|\bm{x}_{t})\!=\!{q}(x_{t-1}^{i}|\bm{x}_{t})$ for every $t$ and every $i$.

Continuous-Time Diffusion Models.   As described in Section 2, many continuous-time diffusion models learn to approximate the likelihood ratio (defined as $s_{\theta}(\bm{x}_{t},\bm{x}^{\prime}_{t};t)$) at all noise levels $t\!\in\![0,T]$:

$$ $\displaystyle s_{\theta}(\bm{x}_{t},\bm{x}^{\prime}_{t};t):=\frac{{q}(\mathbf{ X}_{t}=\bm{x}^{\prime}_{t})}{{q}(\mathbf{X}_{t}=\bm{x}_{t})}.$ $$

Specifically, directly parameterize a neural network to approximate the likelihood ratios, and approximates the likelihood ratios with the conditional distributions ${p}_{\theta}(X_{t}^{i}|\bm{x}_{t}^{\backslash i})$ ($\forall i,t$).

For each $\bm{x}_{t}$, since there are exponentially many possible $\bm{x}^{\prime}_{t}$, it is infeasible to have a neural network to directly model the likelihood ratio for all pairs of $(\bm{x}_{t},\bm{x}^{\prime}_{t})$. Instead, they focus on $(\bm{x}_{t},\bm{x}^{\prime}_{t})$ pairs where $\bm{x}_{t}$ and $\bm{x}^{\prime}_{t}$ are only different in one single variable, i.e., their Hamming distance is one. For example, in , they represent $s_{\theta}$ as $s_{\theta}(\bm{x}_{t},y^{i}_{t};t,i)$, which computes the likelihood ratio between $\bm{x}_{t}$ and $\bm{x}^{\prime}_{t}\!=\!\{\bm{x}_{t}^{\backslash i},y^{i}_{t}\}$. $s_{\theta}$ is trained by minimizing the following objective:

$$ $\displaystyle\mathbb{E}_{t,\bm{x}_{t}\sim{q}(\mathbf{X}_{t})}\left[\sum_{i} \sum_{y_{t}^{i}\neq x_{t}^{i}}w_{t}\left(s_{\theta}(\bm{x}_{t},y_{t}^{i};t,i)- \frac{{q}(\mathbf{X}_{t}=\{\bm{x}_{t}^{\backslash i},y_{t}^{i}\})}{{q}(\mathbf {X}_{t}=\bm{x}_{t})}\log s_{\theta}(\bm{x}_{t},y_{t}^{i};t,i)\right)\right],$ $$

where $\{w_{t}\}_{t}$ are positive weights. When the above objective is minimized, $s_{\theta}$ recovers the correct likelihood ratios:

$$ $\displaystyle\forall i,t,\;s_{\theta}(\bm{x}_{t},y_{t}^{i};t,i)=\frac{{q}( \mathbf{X}_{t}=\{\bm{x}_{t}^{\backslash i},y_{t}^{i}\})}{{q}(\mathbf{X}_{t}= \bm{x}_{t})}.$ (23) $$

At inference time, continuous-time discrete diffusion models select a list of time steps $0\!<\!t_{0}\!<\!\cdots\!<\!t_{k}\!=\!T$ to sample from: first sample from the prior ${p}(\mathbf{X}_{t_{k}})$ and then sample recursively from $\{{p}_{\theta}(\bm{x}_{t_{i-1}}|\bm{x}_{t_{i}})\}_{i=1}^{k}$, where ${p}_{\theta}(\bm{x}_{t_{i-1}}|\bm{x}_{t_{i}})$ is obtained from $s_{\theta}(\bm{x}_{t},y_{t}^{i};t,i)$ in an indirect manner. Specifically, assume $\frac{d{p}(\bm{x}_{t})}{dt}\!=\!Q\!\cdot\!{p}(\bm{x}_{t})$, we have(^3^33This argument largely follows Theorem 4.1 in . We include it for the sake of completeness.)

$$ $\displaystyle{q}(\bm{x}_{t_{i-1}}|\bm{x}_{t_{i}})$ $\displaystyle={q}(\bm{x}_{t_{i}}|\bm{x}_{t_{i-1}})\cdot\frac{{q}(\bm{x}_{t_{i- 1}})}{{q}(\bm{x}_{t_{i}})},$ $\displaystyle={q}(\bm{x}_{t_{i}}|\bm{x}_{t_{i-1}})\cdot\left(\sum_{\bm{x}}\exp (-\Delta t\!\cdot\!Q)(\bm{x}_{t_{i-1}},\bm{x})\cdot\frac{{q}(\mathbf{X}_{t_{i} }=\bm{x})}{{q}(\mathbf{X}_{t_{i}}=\bm{x}_{t_{i}})}\right),$ $$

where $\Delta t\!:=\!t_{i}\!-\!t_{i-1}$ and $\exp(-\Delta t\!\cdot\!Q)(\bm{x}_{t_{i-1}},\bm{x})$ denotes the product of $\exp(-\Delta t\!\cdot\!Q)(x_{t_{i-1}}^{j},x^{j})$, the $x_{t_{i-1}}^{j}$-th row and $x^{j}$-th column of $\exp(-\Delta t\!\cdot\!Q)$.

Plug in Equation 23, we can compute the marginal of $x_{t_{i-1}}^{j}$ (i.e., ${p}_{\theta}(x_{t_{i-1}}^{j}|\bm{x}_{t_{i}})$) following

$$ $\displaystyle{q}(X_{t_{i-1}}^{j}=y|\bm{x}_{t_{i}})$ $\displaystyle\propto{q}(\bm{x}_{t_{i}}|\bm{x}_{t_{i-1}})\cdot\left(\sum_{y^{ \prime}}\exp(-\Delta t\!\cdot\!Q)(y,y^{\prime})\cdot s_{\theta}(\bm{x}_{t_{i}} ,y^{\prime};t_{i},j)\right),$ $\displaystyle=\exp(\Delta t\!\cdot\!Q)(y,x_{t_{i}}^{j})\cdot\left(\sum_{y^{ \prime}}\exp(-\Delta t\!\cdot\!Q)(y,y^{\prime})\cdot s_{\theta}(\bm{x}_{t_{i}} ,y^{\prime};t_{i},j)\right).$ $$

Therefore, if $s_{\theta}$ perfectly learns the likelihood ratios between inputs with Hamming distance at most one, then the correct marginals ${q}(x_{t_{i-1}}^{j}|\bm{x}_{t_{i}})$ can be computed using $s_{\theta}$.

## Appendix E Implementation Details of DCD

Figure: Figure 7: Sampling time of DCD and its two base models with 2 to 128 denoising steps.
Refer to caption: x6.png

We describe details about the “autoregressive” version of DCD introduced in Section 5.3. According to Section 5.3, the first $(T\!-\!t\!-\!1)/T$ portion of the tokens in $\bm{x}_{t+1}$ are unmasked. At step $t$, we only need to additionally unmask the tokens spanning the $(T\!-\!t\!-\!1)/T$ to $(T\!-\!t)/T$ fraction of the sequence $\bm{x}_{t}$. We do this by caching the keys and values generated by the attention layers of tokens generated in previous denoising steps. So at step $t$, we will have the KV-caches of the first $(T\!-\!t\!-\!1)/T$ fraction of tokens. As a result, the computational cost for running the autoregressive Transformer is independent of the number of denoising steps.

Figure: Algorithm 2 DCD with Autoregressive Copula Models and Using Autoregressive Sampling

Additional Runtime Analysis.
Figure 7 displays the generation time per sample for $\text{SEDD}_{\text{{M}}}$, $\text{GPT-2}_{\text{{S}}}$, and DCD. When the number of denoising steps is small, the computation cost of running $\text{GPT-2}_{\text{{S}}}$ dominates the total runtime of DCD. However, as the number of denoising steps increases, this cost is amortized because, with KV-caching, the total computation cost for running $\text{GPT-2}_{\text{{S}}}$ stays constant.

## Appendix F Additional Unconditional Generation Experiments

Figure: Figure 8: Comparison between generative perplexity ($\downarrow$), diversity (measured by sentence entropy; $\uparrow$), and runtime ($\downarrow$) of DCD with baselines.
Refer to caption: x7.png

To better understand the relation between quality (measured by generative perplexity), diversity (measured by sentence entropy(^4^44The sentence entropy of a sequence is the entropy of its token frequency distribution. The reported number is averaged across all samples.)), and speed for DCD and its baselines. Specifically, we run the more efficient version of DCD described in the last paragraph of Section 5.3 and Appendix E to generate text sequences of lengths 128 and 1024. In addition to SEDD and GPT2, the two base models used by DCD, we compare them with MDLM , a more recent discrete diffusion model that is more efficient than SEDD. Note that DCD can use any discrete diffusion model as its base model.

First, we compare the sample time and the generative perplexity (the second and the fourth sub-plot in Figure 8). Compared to SEDD, GPT, and MDLM, DCD consistently achieves better generative perplexity given a fixed runtime constraint. It also requires less time to achieve a desired perplexity value.

Additionally, we compare the perplexity and diversity of the generated text sequences. Following community standards, we adopt the sentence entropy to measure the diversity of generated text. Specifically, the entropy of each text sequence is the entropy of its token frequency distribution, and the final sentence entropy is the average entropy over all generated sequences. The desired behavior is to have low generative perplexity and high sentence entropy (which means high diversity). Results are shown in the table below and Figure 8’s first and third sub-plot. Compared to the two discrete diffusion models (SEDD and MDLM), DCD achieves better generative perplexity under the same entropy, which offers a better perplexity-diversity tradeoff. Compared to the autoregressive GPT model, although the entropy of DCD is lower, it achieves better generative perplexity with slightly worse entropy.

## Appendix G Additional Experimental Details

This section provides additional details of the experiments.

### G.1 Unconditional Text Generation

SEDD.
We adopt the SEDD-medium model with 320M non-embedding parameters trained on OpenWebText. The model is accessed through HuggingFace: [https://huggingface.co/louaaron/sedd-medium](https://huggingface.co/louaaron/sedd-medium). We follow the original paper and use the log-linear noise schedule $\sigma(t)\!=\!-\log(1\!-\!(1\!-\!\epsilon t))$, which leads to the forward transition probabilities ($0\!\leq\!s\!<\!t\!\leq\!T$):

$$ $\displaystyle{q}(\bm{x}_{t}|\bm{x}_{s}):=\mathrm{Cat}(\bm{x}_{t};\exp(\sigma(t -s)\cdot Q)\cdot\bm{x}_{s}).$ $$

The absorbing mask forward noising process is used. The corresponding transition rate matrix is

$$ $\displaystyle Q:=\begin{bmatrix}-1&0&\cdots&0&0\\ 0&-1&\cdots&0&0\\ \vdots&\vdots&\ddots&\vdots&\vdots\\ 0&0&\cdots&-1&0\\ 1&1&\cdots&1&0\\ \end{bmatrix},$ $$

where the last category is <MASK>.

GPT.
The GPT-2-small model is obtained from HuggingFace: [https://huggingface.co/openai-community/gpt2](https://huggingface.co/openai-community/gpt2).

DCD.
We implement DCD by combining $\text{SEDD}_{\text{{M}}}$ and $\text{GPT-2}_{\text{{S}}}$ following the steps in Algorithm 1. In line 8, instead of masking tokens independently, we group chunks of 8 tokens together and mask/unmask them with the same probability given the noise schedule (i.e., $\alpha_{t}/\alpha_{t+1}$ as shown in Prop. 5).

### G.2 Conditional Text Generation

MAUVE Score.
We adopt the MAUVE implementation available in the Python package evaluate. We use the default hyperparameters established by the original paper , which is also the default used by the package. We found that the number of samples and the number of samples given a fixed prompt influenced the score. Therefore, we randomly selected the 2,000 prompts and generated 5 samples for each prompt for all methods.

Detailed Runtime Analysis.
As shown in Algorithm 1, in each denoising step of DCD, we need to run the discrete diffusion model twice: first to compute $\{{p}(\tilde{X}_{t}^{i}|\bm{x}_{t+1})\}_{i}$ and next to compute $\{{p}(\tilde{X}_{t}^{i}|\bm{x}_{t+1}^{<i})\}_{i}$ by applying causal attention masks to the same denoising neural network given that it is based on the Transformer architecture. Next, as discussed in Appendix E, the total runtime consumed by the autoregressive model remains constant across different numbers of denoising steps thanks to the KV-caching mechanism. Therefore, the runtime of DCD will be dominated by the computation cost of the autoregressive model with only a few denoising steps. As the number of denoising steps increases, the runtime of the autoregressive model will be amortized and the total computation cost will be dominated by the cost to evaluate the diffusion model.

SSD-LM.
SSD-LM is a semi-autoregressive model that uses techniques from discrete diffusion models to predict/denoise chunks of sequences in an autoregressive manner. Specifically, given a predefined chunk size, SSD-LM diffuses tokens in each chunk one by one conditioned on all previous chunks. As a result, the model is semi-autoregressive and cannot see suffix prompts.

While the official implementation on GitHub ([https://github.com/xhan77/ssd-lm](https://github.com/xhan77/ssd-lm)) only allows conditioning on tokens in previous prompts, we improved their code to also allow conditioning on tokens in the current chunk that is being diffused. Specifically, we replace the diffusion model’s input corresponding to the prompt tokens with the ground truth token embeddings.

We followed the original paper to choose a chunk size of 32 and use top-p sampling with $p\!=\!0.95$. The remaining hyperparameters are kept as default.

### G.3 Antibody Sequence Infilling

Detailed Task Description.
The adopted antibodies with an immunoglobulin G (IgG) format, which comprises a heavy (H) chain and a light (L) chain. Each chain has three complementarity determining regions (CDRs) that are crucial toward the binding affinity to the target antigen.

Training NOS-D.
We use the training script as well as the dataset provided in the official GitHub repo of NOS-D ([https://github.com/ngruver/NOS](https://github.com/ngruver/NOS)). The model is trained with 50 epochs using the default settings (e.g., learning rate and its schedule).

Training GPT.
We use the same dataset provided in the repository of NOS-D and use the GPT implementation from [https://github.com/karpathy/nanoGPT/tree/master](https://github.com/karpathy/nanoGPT/tree/master). The GPT model has 6 layers, an embedding size of 512, and 16 attention heads. The model is trained for 10 epochs with the default settings in the nanoGPT repository.

DCD.
When implementing DCD for the antibody sequence infilling task, we add an additional scaling factor to the coefficients in $\mathbf{V}$. That is, $\mathbf{V}$ is updated in line 6 of Algorithm 1 following

$$ $\displaystyle\forall i,\tilde{x}_{t}^{i},\,\mathbf{V}[i,\tilde{x}_{t}^{i}]= \beta\cdot\left(\log{p}_{\mathrm{dm}}(\tilde{x}_{t}^{i}|\bm{x}_{t+1})-\log{p}_ {\mathrm{dm}}(\tilde{x}_{t}^{i}|\bm{x}_{t+1}^{<i})\right),$ $$

where we set $\beta\!=\!0.1$ for this task. We note that $\beta\!=\!1$ works well for the language modeling tasks. The need to choose a smaller $\beta$ in this task may be caused by the fact that the dataset and the models are much smaller and are more prone to overfitting.

## Appendix H Additional Related Work

We briefly review a class of related works that perform (semi-)autoregressive diffusion, which is weakly related to our work since we also “combine” discrete diffusion models with autoregressive models. Specifically, perform diffusion in an autoregressive manner by allowing the noising schedule to be variable-dependent. Variables at the beginning are kept unchanged at small $t$s and are corrupted only when $t$ is close to $T$; variables at the end are corrupted in the first few time steps. During sampling where $t$ moves from $T$ to 0 0, initial variables/tokens are first denoised, followed by later tokens. This allows the diffusion model to perform “autoregressive denoising”.

## Appendix I Additional Text Samples

We provide randomly selected unconditional samples in Figure 9 and 10 and conditional samples in Figure 11 and 12.

Figure: Figure 9: Randomly selected unconditional samples from DCD ($\text{SEDD}_{\text{{M}}}+\text{GPT-2}_{\text{{S}}}$) with 4 denoising steps.
Refer to caption: x8.png

Figure: Figure 10: Randomly selected unconditional samples from DCD ($\text{SEDD}_{\text{{M}}}+\text{GPT-2}_{\text{{S}}}$) with 32 denoising steps.
Refer to caption: x9.png

Figure: Figure 11: Randomly selected conditional samples from DCD ($\text{SEDD}_{\text{{M}}}+\text{GPT-2}_{\text{{S}}}$) with 4 denoising steps. Prompt texts are bolded and in blue.
Refer to caption: x10.png

Figure: Figure 12: Randomly selected conditional samples from DCD ($\text{SEDD}_{\text{{M}}}+\text{GPT-2}_{\text{{S}}}$) with 32 denoising steps. Prompt texts are bolded and in blue.
Refer to caption: x11.png