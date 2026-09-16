---
arxiv_id: "2305.18619"
title: "Likelihood-Based Diffusion Language Models"
year: 2023
source: arxiv2md
---

## Abstract

Abstract Despite a growing interest in diffusion-based language models, existing work has not shown that these models can attain nontrivial likelihoods on standard language modeling benchmarks.
In this work, we take the first steps towards closing the likelihood gap between autoregressive and diffusion-based language models, with the goal of building and releasing a diffusion model which outperforms a small but widely-known autoregressive model.
We pursue this goal through algorithmic improvements, scaling laws, and increased compute.
On the algorithmic front, we introduce several methodological improvements for the maximum-likelihood training of diffusion language models.
We then study scaling laws for our diffusion models and find compute-optimal training regimes which differ substantially from autoregressive models.
Using our methods and scaling analysis, we train and release Plaid 1B, a large diffusion language model which outperforms GPT-2 124M in likelihood on benchmark datasets and generates fluent samples in unconditional and zero-shot control settings. 1 1 1 We release our code and pretrained models at https://github.com/igul222/plaid .

## 1 Introduction

Large language models lie at the center of recent advances in artificial intelligence.
Shared across nearly all such language models is a common recipe: learn a model that maximizes data likelihoods using an autoregressive, left-to-right factorization.
Maximum-likelihood pretraining has been a remarkably successful paradigm, leading to models that perform well on a range of downstream tasks and display complex behaviors like in-context learning .

Thus far, autoregressive modeling has been a core part of this process due to its computational efficiency and empirical performance.
However, this choice carries drawbacks.
Autoregressive models generate tokens one at a time, making it difficult to perform long-range planning or controllable generation .
In addition, certain sequence distributions may be fundamentally more difficult to model autoregressively .

Given the importance of language modeling, these potential drawbacks motivate us to explore alternatives to the autoregressive approach.
As a promising candidate, we turn to continuous diffusion models , which have achieved state-of-the-art results in image modeling .
In language, prior works on diffusion models exist , but these optimize non-likelihood-based objectives.
Without the ability to use standard likelihood-based benchmarks , it is difficult to say precisely how these models compare to autoregressive models (see [Section 7](#S7) for a discussion).
Somewhat concerningly, there is no work showing that it is possible for diffusion language models to achieve any nontrivial likelihoods on standard benchmarks.

In this work, we explore the limits of likelihood-based diffusion language models.
Our goal is to train and release a diffusion model which achieves better likelihoods than GPT-2 124M , which we consider the smallest widely-adopted autoregressive model today.
To achieve this goal, we first develop an algorithmic framework, then study its scaling laws to enable compute-optimal training, and finally train a large model called Plaid 1B.

Our contributions are as follows:

- 1.
We explore the design space of likelihood-based diffusion language models and propose an algorithmic framework called Plaid.
We validate the design choices of Plaid through compute-matched ablations.
- 2.
We study the scaling laws of Plaid training.
Our analysis shows that the log-likelihood of Plaid models improves predictably with more compute.
We derive a recipe for compute-optimal training which differs substantially from the usual autoregressive rule.
- 3.
We train and release Plaid 1B, a large diffusion language model pretrained on OpenWebText2 .
Plaid 1B outperforms GPT-2 124M in zero-shot likelihood across six standard benchmarks.
We demonstrate Plaid 1B’s ability to perform fluent and controllable text generation.

Figure: Figure 1: Plaid models scale predictably across five orders of magnitude. Our largest model, Plaid 1B, outperforms GPT-2 124M in zero-shot likelihood (see [Table 2](#S6.T2)).
Refer to caption: /html/2305.18619/assets/x1.png

## 2 Variational Diffusion Models for language

In this background section, we formally define continuous diffusion models over text sequences, adopting the Variational Diffusion Models (VDM) framework which is a natural fit for likelihood-based training (see for a survey on other formalisms).
For brevity, we simplify some details in our exposition and refer the reader to for details.

Consistent with prior work (e.g. ), our basic approach will be to map discrete text sequences into a continuous space with a token-wise embedding function and then construct a diffusion model on the embedded data.

### 2.1 Forward diffusion process

Consider a sequence of tokens $x=[x^{(1)},\ldots,x^{(L)}]$ drawn from the data distribution $q(x)$.
We transform $x$ into a sequence $\tilde{x}$ of embedding vectors using an invertible token-wise embedding function $\mathrm{Embed}(\cdot)$, such that $\tilde{x}^{(i)}:=\mathrm{Embed}(x^{(i)})$.

The forward process is a Markov chain over latent variables $z_{t}$ from $t=0$ to $t=1$ which progressively adds Gaussian noise to $\tilde{x}$.
Let $\sigma^{2}(t)$ be some monotonic function that specifies the total noise added by time $t$.
We then define the forward process distribution $q$ with $T$ discrete timesteps as

$$ $\displaystyle q(x,z):=q(x)q(z_{0}|x)\prod_{i=1}^{T}q(z_{i/T}|z_{(i-1)/T})$ (1) $$

where $q(z_{0}|x):=\mathcal{N}(\tilde{x},\sigma^{2}(0))$ and $q(z_{t}|z_{s}):=\mathcal{N}(z_{s},\sigma^{2}(t)-\sigma^{2}(s))$.
It follows from this that $q(z_{s}|z_{t},\tilde{x})$ is also Gaussian, which will be useful later.

### 2.2 Reverse generative process

We can approximate the forward process distribution $q$ by a learned Markov reverse process where time runs backward from $t=1$ to $t=0$.
The reverse process with $T$ timesteps is defined via the decomposition

$$ $\displaystyle p_{\theta}(x,z):=p(z_{1})\left(\prod_{i=1}^{T}p_{\theta}(z_{(i-1)/T}|z_{i/T})\right)p(x|z_{0}).$ (2) $$

Let $z_{t}^{(i)}$ denote the portion of $z_{t}$ at sequence position $i$.
Then we choose $p(z_{1}):=\mathcal{N}(0,\sigma^{2}(1)I)$ and $p(x|z_{0}):=\prod_{i}p(x^{(i)}|z^{(i)}_{0})$ where $p(x^{(i)}|z^{(i)}_{0})\propto q(z^{(i)}_{0}|x_{i})$.
Recalling that $q(z_{(i-1)/T}|z_{i/T},\tilde{x})$ is Gaussian, for the remaining factors we choose $p_{\theta}(z_{(i-1)/T}|z_{i/T}):=q(z_{(i-1)/T}|z_{i/T},\tilde{x}=\hat{x}_{\theta}(z_{i/T}))$ where $\hat{x}_{\theta}(z_{t})$ is a denoiser neural network that approximates $\mathbb{E}_{q}[\tilde{x}|z_{t}]$.
Finally, our generative model is given by the marginal distribution $p_{\theta}(x)=\int_{z}p_{\theta}(x,z)$.
If $\hat{x}_{\theta}$ is optimal, then the forward and reverse processes express the same joint distribution as $\sigma^{2}(0)\rightarrow 0$, $\sigma^{2}(1)\rightarrow\infty$, and $T\rightarrow\infty$.

### 2.3 Likelihood bound

To optimize and evaluate the likelihood, we can write a variational lower bound (VLB) for the log-likelihood as

$$ $\displaystyle-\log p_{\theta}(x)\leq-\mathrm{VLB}(x):=D_{\mathrm{KL}}(q(z_{1}|x)\|p(z_{1}))\;+\;\mathbb{E}_{q(z_{0}|x)}[-\log p(x|z_{0})]\;+\;\mathcal{L}_{T}$ (3) $$

where

$$ $\displaystyle\mathcal{L}_{T}:=\sum_{i=1}^{T}\mathbb{E}_{q(z_{i/T}|x)}[D_{\mathrm{KL}}(q(z_{(i-1)/T}|z_{i/T},x)\|p_{\theta}(z_{(i-1)/T}|z_{i/T}))].$ (4) $$

In the $T\rightarrow\infty$ limit, $\mathcal{L}_{T}$ simplifies to

$$ $\displaystyle\mathcal{L}_{\infty}=-\frac{1}{2}\mathbb{E}_{t\sim U[0,1],z_{t}\sim q(z_{t}|x)}[\mathrm{SNR}^{\prime}(t)\|x-\hat{x}_{\theta}(z_{t})\|_{2}^{2}]$ (5) $$

where $\mathrm{SNR}^{\prime}(t):=\frac{d}{dt}\frac{1}{\sigma^{2}(t)}$.
We use Monte-Carlo estimates of the resulting continuous-time likelihood bound to train and evaluate our model.

### 2.4 Learned noise schedule

A crucial hyperparameter in diffusion models is the noise schedule $\sigma^{2}(t)$, which specifies how much noise to add at each time in the diffusion process.
In our setting, the VLB is differentiable with respect to $\sigma^{2}(t)$ via the reparameterization trick.
Moreover, the VLB is invariant to the value of $\sigma^{2}(t)$ except at $t=0$ and $t=1$ in the continuous-time limit.

We can therefore parameterize $\sigma^{2}(t)$ as a scalar-to-scalar neural network and learn it by gradient descent.
We train the endpoints $\sigma^{2}(0)$ and $\sigma^{2}(1)$ to maximize the VLB, and the schedule in between the endpoints to minimize the variance of the Monte-Carlo estimate of the VLB.
Minimizing the loss variance is a proxy for minimizing the gradient covariance trace, which generally speeds up learning.
See for further implementation details about this training procedure.

## 3 The Plaid framework

In this section, we present a series of algorithmic improvements to the basic setup described in [Section 2](#S2).
The result is a framework for diffusion language models which we refer to as Plaid (Perplexity-based LAnguage Inverse Diffusion).

### 3.1 Learned embeddings

In an autoregressive language model, the embedding operation is simply the first layer of the neural network and thus can be treated as just another part of the network.
This is not true of embeddings in diffusion language models, which play a more fundamental role: they determine the order in which different tokens get generated.
Tokens whose embeddings are far apart become distinguishable early in the reverse process, whereas nearby embeddings are distinguishable only later, at low noise levels.

Despite the importance of embeddings in diffusion language models, the loss functions used in prior work lead to ill-posed problems when optimized over $W_{\text{Embed}}$: for example, if our objective is $L_{2}$ reconstruction, then collapsing the embeddings by setting $W_{\text{Embed}}=0$ and $\hat{x}_{\theta}(z_{t})=0$ yields a degenerate solution with zero loss.
Prior work addresses this with workarounds like choosing $W_{\mathrm{Embed}}$ by hand or using heuristic regularizers or constraints .

In contrast, the Plaid loss function is a bound on the log-likelihood of the discrete data, which is a meaningful objective over both the model weights and embeddings.
We therefore optimize the embedding matrix $W_{\mathrm{Embed}}$ jointly with the rest of the model without additional constraints.

### 3.2 Categorical reparameterization

When optimally trained, $\hat{x}_{\theta}(z_{t})$ learns to approximate a conditional expectation $\mathbb{E}[\tilde{x}|z_{t}]$ over sequences of word embeddings $\tilde{x}$.
At low noise levels, some or all of the embeddings in $\tilde{x}$ are deterministic given $z_{t}$, so an optimal $\hat{x}_{\theta}(z_{t})$ should output these exactly.
However, doing so requires memorizing embedding vectors to high precision somewhere inside the model parameters, which is a poor use of capacity.

Instead of forcing the model to memorize the embedding vectors, we reparameterize $\hat{x}_{\theta}(z_{t})$ as an average of embeddings weighted by a softmax over tokens.
More formally, let $f_{\theta}(z_{t})$ be a neural network which outputs logits and define $\hat{x}$ as an average over embeddings $\hat{x}_{\theta}^{(i)}(z_{t}):=W_{\mathrm{Embed}}\mathrm{softmax}(f_{\theta}^{(i)}(z_{t}))$.
We can interpret $f$ as learning a posterior over each discrete token $x^{(i)}$ given $z_{t}$.
This relates to methods proposed in prior work, but these either require proxy objectives or consider image models .

### 3.3 Output prior

When we interpret $f_{\theta}$ as a posterior over tokens, the optimal value of $f_{\theta}(z_{t})$ is $\log q(x^{(i)}|z_{t})+Z$, which decomposes as $\log q(z_{t}^{(i)}|x^{(i)})+\log q(x^{(i)}|z^{(\neq i)}_{t})+Z$ where $z^{(\neq i)}_{t}:=\{z^{(j)}_{t}:j\neq i\}$.
We view the first term as a prior constraining the model’s predictions to those which are plausible given $z_{t}$, while the second term models relationships between different tokens.

To make it easier to model $f_{\theta}$, we compute the first term in closed form as the log-density of a Gaussian $\mathcal{N}(\tilde{x}^{(i)},\sigma^{2}(t)I)$ and add it to the output.
This leaves the neural network with only the easier task of estimating $\log p(x^{(i)}|z^{(\neq i)}_{t})$.
Empirically, we found it helpful to linearly anneal in this prior over the first 5000 steps of training.

### 3.4 Learned conditional likelihood

Recall that our loss function ([3](#S2.E3)) includes a conditional likelihood term $\log p(x|z_{0})$.
We are free to choose $p$ however we wish, and in [Section 2](#S2) we chose a position-wise factorial model $p(x|z_{0}):=\prod_{i}p(x^{(i)}|z^{(i)}_{0})$, with a simple fixed distribution for each factor.
This choice is optimal for sufficiently small $\sigma^{2}(0)$, but using a more powerful model allows $\sigma^{2}(0)$ to take a larger value, effectively truncating the reverse process and therefore making it simpler to learn.

Here we leverage the fact that, after applying the categorical reparameterization ([Section 3.2](#S3.SS2)), our neural network $f_{\theta}(z_{t})$ can be interpreted as learning the logits for $q(x^{(i)}|z_{t})$ at all positions $i$.
We therefore choose to keep $p(x|z_{0})$ as a factorial model, but define each factor $p(x_{i}|z^{(i)}_{0})$ using the more powerful learned model $\mathrm{softmax}(f^{(i)}_{\theta}(z_{t}))$.

Implementing this change naively requires two evaluations of $f_{\theta}$ for each minibatch example during training, corresponding to the two terms of ([3](#S2.E3)) $\mathcal{L}_{\infty}$ and $\log p_{\theta}(x|z_{0})$.
We instead split each minibatch, using some examples to compute $\mathcal{L}_{\infty}$ and the rest to compute $\log p_{\theta}(x|z_{0})$.
We allocate examples between the two terms according to the ratio $\sqrt{\frac{\mathrm{Var}(\mathcal{L}_{\infty})}{\mathrm{Var}(\log p_{\theta}(x|z_{0}))}}$, where we compute the variances using running estimates of each term’s first and second moments.
This minimizes the variance of the full loss ([3](#S2.E3)).

### 3.5 Self-conditioning

Self-conditioning is a technique which improve the performance of diffusion language models.
The core idea is to reparameterize the denoiser $\hat{x}_{\theta}(z_{t})$ as the fixed point $y_{\infty}$ of a recurrence $y_{0}:=0,y_{i+1}:=\hat{x}_{\theta}^{\prime}(z_{t},y_{i})$ where $\hat{x}_{\theta}^{\prime}$ is a neural network which now takes two inputs instead of one.
During training, we approximate the fixed point $y_{\infty}$ by randomly unrolling the recurrence to either $y_{1}$ (with probability $0.75$) or $y_{2}$ (otherwise).
When we unroll to $y_{2}$ during training, we zero the gradients with respect to $y_{1}$, the noise schedule, and the embeddings.
During held-out likelihood evaluation, we always unroll to $y_{2}$.
During sampling, instead of solving the recurrence from scratch at each step of the diffusion chain, we compute $\hat{x}_{\theta}^{\prime}(z_{1},0)$ for the first step and $\hat{x}_{\theta}^{\prime}(z_{t},\hat{x}_{\theta}^{\prime}(z_{t+(1/T)},\ldots))$ for subsequent steps.

### 3.6 Other details

We perform all forward and backward computations in double precision except for the Transformer layers themselves, which happen in bfloat16 mixed precision. This comes at a negligible extra cost since the Transformer layers dominate the overall cost.

#### Architecture choices

We condition $\hat{x}_{\theta}(z_{t})$ on the timestep $t$ by adding a sinusoidal encoding of $t$ to the Transformer’s residual stream before the first layer.
Before feeding $z_{t}$ into the Transformer, we rescale it by a factor of $\sqrt{1+\sigma^{2}(t)}$ which makes each input dimension approximately unit-variance.
Whereas autoregressive Transformers are relatively insensitive to aspect ratio , we find that Plaid performance increases significantly with Transformer depth up to about 16 layers.
We also find that performance is sensitive to the choice of embedding dimension, with small values performing best. In all experiments, we use embedding dimension $16$.

#### Stochastic sequence length

Unlike autoregressive models, diffusion language models can only operate on sequences of exactly the same length as those seen during training.
To enable our model to generalize to shorter sequence lengths, we truncate a small random subset of examples seen during training to random lengths.
We observe that truncating even 3% of examples allows the model to generalize well across lengths without impacting full-length performance.
Short-sequence performance does not improve substantially as we increase the number of truncated examples.

**Table 1: Compute-matched ablations of algorithmic components on OpenWebText2.**
|  | NLL bound (val.) |
| --- | --- |
| Our full method | $\mathbf{3.89}$ |
| Our full method ($0.5\times$ compute) | $4.01$ |
| No learned noise schedule | $4.17$ |
| No learned embeddings | $4.54$ |
| No categorical reparameterization | $4.25$ |
| No output prior | $3.95$ |
| No learned conditional likelihood | $4.03$ |
| No self-conditioning | $3.98$ |
| CDCD  (our reimplementation) | $4.23$ |

## 4 Ablation experiments

In this section, we validate different aspects of the Plaid framework through compute-matched ablation experiments.

### 4.1 Validating likelihood-based training

We take a likelihood-based approach in this work for multiple reasons: it has a principled interpretation, it simplifies training and evaluation, and it has yielded strong results in autoregressive models.
Here, we validate that the log-likelihood objective can attain competitive sample quality through a human evaluation.

In diffusion models, the log-likelihood bound is an expectation over noise levels of a reconstruction loss weighted by a specific function of the noise level.
In contrast, most prior work on diffusion models for language  as well as images  use heuristic weight schedules.
Motivated by the intuition that human perception is more sensitive to coarse structure than fine details, these typically assign more weight to higher noise levels than the likelihood weight schedule.

We train three Plaid models: one with the likelihood weight schedule (“VLB”) and two with heuristic weight schedules (“Schedule A” and “Schedule B”) which we plot in  [Appendix A](#A1).
The models are trained on a large dataset of short children’s stories which we constructed by finetuning GPT-J  on ROCStories .
Because learning embeddings is only straightforward when training against the likelihood bound, all models use fixed embeddings obtained from a previously-trained known-good model.

We repeatedly asked crowdworkers to choose from a pair of model samples, where one sample came from the likelihood-trained model and the other came from a heuristically-trained model.
On average, crowdworkers preferred the likelihood-trained model over both alternatives: Weighting A’s win rate was $0.449$ ($p=0.001$, 95% CI $[0.417,0.482]$) and Weighting B’s win rate was $0.457$ ($p=0.005$, 95% CI $[0.425,0.490]$).
Because we only consider two alternative weight schedules, we do not claim that the likelihood objective yields optimal sample quality, but our results suggest that it performs at least comparably to other choices.

### 4.2 Validating algorithmic components

Having validated our likelihood-based approach, we show in this section that each of the algorithmic components described in [Section 3](#S3) lead to improved likelihoods in a compute-matched ablation study.

We train Plaid models on OpenWebText2 and report their log-likelihood bounds on held-out data in [Table 1](#S3.T1).
Our reference model (“full method”) is a $16\times 384$ Transformer with $28\mathrm{M}$ non-embedding parameters, trained for $92\mathrm{K}$ steps at batch size $256$ and sequence length $256$, corresponding to $1.12\times 10^{18}$ non-embedding FLOPs.
For each ablation model, we stay as close to this configuration as possible while preserving the number of non-embedding FLOPs (we exclude FLOPs from the embedding and output projections because these become negligible at large scale).
We observe that ablating each of the components described in [Section 3](#S3) results in a worse log-likelihood.
As a comparison point, we also train a model at half the compute budget ($5.6\times 10^{17}$ FLOPs) by halving the model size.
See [Appendix B](#A2) for more training details.

Finally, as a comparison to prior work, we reimplement CDCD , train it following the same configuration, and report its log-likelihood.
We follow the authors’ description as faithfully as possible except for the noise schedule endpoints, embedding dimension, and embedding weight initialization, which we tune to maximize log-likelihood.
We observe in [Table 1](#S3.T1) that even the half-compute-budget version of Plaid surpasses our CDCD implementation in likelihood.
Note that CDCD was not developed as a likelihood-based model, and the lack of a public implementation means that there are most likely differences between our implementation and the original.

## 5 Scaling laws for Plaid

Figure: Figure 2: Plaid models improve with compute at a similar rate to autoregressive models, but Plaid is less efficient by a constant factor of $64\times$.
Refer to caption: /html/2305.18619/assets/x2.png

Having developed an algorithmic framework for diffusion language models, we now study its scaling properties in order to guide large-scale training of Plaid models.
In the case of autoregressive models, the work of demonstrates that model log-likelihoods follow a log-linear scaling law: across many orders of magnitude, training with more compute predictably improves likelihood.
Using these scaling laws, and accurately predict the optimal model size as a function of the given compute budget across many orders of magnitude.
Both results together enable effective large-scale training.
In this section, we experimentally determine them for Plaid models.

### 5.1 Methodology

Our main experimental method will be an IsoFLOP analysis .
We first fix a set of FLOP budgets $\{C_{1},\ldots,C_{K}\}$.
For each budget $C$, we train models with different sizes $\{N_{C,1},\ldots,N_{C,M}\}$ and perform a quadratic fit of the loss $L$ to $\log N$.
We plot all the data along with quadratic fits in [Appendix C](#A3) and find that the fits approximate the data well.

The minimum of the quadratic gives us the compute-optimal loss $L^{*}_{C}$ and corresponding model size $N^{*}_{C}$ for that budget.

Given the compute-optimal loss $L^{*}_{C_{i}}$ for each FLOP budget $C_{i}$, we fit the parameters of a loss scaling law

$$ $\min_{\alpha,\beta}\sum_{i}(\log(L^{*}_{C_{i}})-\beta\log(C_{i})-\alpha)^{2}$ $$

which can then be used to predict the compute-optimal loss as $L^{*}(C)=\alpha C^{\beta}$.
We also fit a parameter scaling law $N^{*}(C)$ in the same fashion from the model sizes $N^{*}_{C_{i}}$.

We perform IsoFLOP analyses for both Plaid and autoregressive models in order to compare the results.
We choose compute budgets log-uniformly between $10^{16}$ and $10^{19}$ FLOPs and corresponding model sizes heuristically.
We choose learning rates using $\mu$Transfer  and batch sizes, weight decays, and aspect ratios by well-tuned heuristics.
When computing FLOPs, we exclude FLOPs from the embedding layers and output projections.
This enables us to use much smaller compute budgets than , but it causes our autoregressive scaling law to differ slightly from theirs.
We consider this acceptable since we are mainly interested in the differences between our autoregressive and Plaid scaling laws.

### 5.2 Loss improves predictably with compute

We plot both of our scaling laws in [Figure 3](#S5.F3).
Our first finding is that over many orders of magnitude, the compute-optimal log-likelihood of Plaid models closely matches a power law function of the compute.
Surprisingly, we find that the slopes of both the autoregressive and diffusion scaling laws are almost exactly the same.
These results validate Plaid’s scalability and suggest that we can obtain strong improvements by training at larger scale.

Regardless of scale, Plaid models require a constant factor of about $64\times$ more compute to match their autoregressive equivalents.
While this factor is large, our work represents the very first attempt at efficient diffusion model training and focused engineering effort on constant-factor improvements to diffusion models may enable them to perform similarly to autoregressive models in the future.

### 5.3 Compute-optimal training recipe

Our next goal is to understand how to optimally use a given compute budget $C$ to maximize the held-out likelihood of a model.
Specifically, we must choose between training a large model for fewer iterations or training a small model for longer.
For this, we leverage our parameter scaling law $N^{*}(C)$ which predicts the optimal model size given a compute budget.

We plot both of our parameter scaling laws in [Figure 3](#S5.F3) and again find that the trends have nearly the same slope but differ by a constant factor.
Specifically, compute-optimal Plaid models should be about $4\times$ smaller (and therefore trained for $4\times$ longer) than compute-optimal autoregressive models.
The large gap in compute-optimal settings suggests that selecting model sizes based on existing scaling laws , which were developed for autoregressive models, could incur a substantial loss in the effective compute budget.

## 6 Plaid 1B

**Table 2: Plaid 1B outperforms GPT-2 124M in zero-shot likelihood across six benchmark datasets from . Our GPT-2 numbers differ from the originals due to striding and detokenization (see [Section 6.1](#S6.SS1)).**
|  | PTB | enwik8 | text8 | WikiText2 | WikiText103 | 1BW |
| --- | --- | --- | --- | --- | --- | --- |
|  | (PPL) | (BPC) | (BPC) | (PPL) | (PPL) | (PPL) |
| Plaid 1B (ours) | 74.33 | 1.18 | 1.12 | 29.42 | 28.28 | 77.64 |
| GPT-2 124M | 87.97 | 1.24 | 1.22 | 35.01 | 35.92 | 87.85 |
| GPT-2 345M | 64.92 | 1.09 | 1.11 | 26.80 | 26.13 | 67.34 |
| GPT-2 762M | 53.42 | 1.04 | 1.06 | 23.30 | 22.24 | 59.48 |
| GPT-2 1.5B | 47.59 | 1.00 | 1.02 | 21.33 | 20.13 | 54.09 |

**Table 3: Chosen unconditional samples from Plaid 1B demonstrate fluent syntax and long-range coherence. See [Appendix D](#A4) for un-picked random samples.**
| New research rolled out at an annual scientist meeting finds that the industry will need to recover between 4,000 and 10,000 tons every year of fracked and produced oil and gas fossil reserves in order to do that, according to an analysis done by lead author Dr. Ernesto Monteiro of the University of Alberta in Canada.\n\nThe eye-watering figure represents the total amount of oil and gas — shale and natural gas produced, extracted and sold — will likely need to be recovered in coming years to meet the carbon mitigation goals. …[698 words omitted]… the team concluded in a report published in an academic journal prepared for the annual meeting of the National Academies of Sciences. | The Barcelona Golf Course doesn’t look like a golf course, but it is an oasis for gardening in this busy city.\n\nIt’s a massive course spread over 120 acres with fairways that are split right down the middle, unlike the designs on most golf courses. The course enjoys the stunning view of the skyline above it.\n\nA giant oak tree almost 40 meters in diameter serves as one main highlight to the golf course’s design. …[414 words omitted]… The new golf course is accessible on the area’s busy streets with shops and restaurants, so the community can enjoy all the leisure activities in the green space.\n\nThe team uprooted the previously existing Pérez Tree, to make room for the new trees to complement Gillet Park. |
| --- | --- |

**Table 4: Chosen conditional samples from Plaid 1B in different zero-shot control settings. Highlighted spans are prompts. See [Appendix D](#A4) for un-picked random samples.**
| Prefix completion: |
| --- |
| Generative models of text are very versatile: they can be used as a data classification model and also incorporated into multiple data processing engines.\n\nIn this article, we present two new neural memory models capable of processing terabytes of data and the neural networks and computational techniques that are used in those models. |
| Infilling: |
| A year ago in Paris, prior to the tournament, I went to Elijah’s to eat and get drunk. Everyone in the venue was seventeen. I was there for a few minutes and then I went back to the event. Wow, what a great day! So relaxed and too happy. I do not think I was always like that. |
| Token-level weights (5$\times$ weight on “law”): |
| Let’s talk about law and medicine.\n\n\n\nIn her dissent, Justice Ron Sen, a veteran administrative law judge, points out that the decision "ignores the fact that the original separation agreement was reached by binding arbitration" that responded to "the legitimate ethical concerns of the university administration," which is what lies "at the heart of law and medicine." |
| Token-level weights (5$\times$ weight on “medicine”): |
| Let’s talk about law and medicine.\n\nIn part because of advancements in technology, personal information about medical and drug use is spreading. Healthcare professionals across the nation rely on this personal data to make decisions about drug prescriptions and clinical trials and monitor people at immediate risk of serious or chronic diseases. |
| Lexical constraints (“Donald” anywhere): |
| Also facing legal challenges is Donald Trump’s executive order banning immigration from seven Muslim-majority countries that is facing a temporary halt, with nothing scheduled to go into effect. Two federal judges have ruled that such an order violates the establishment clause. |
| Composition and negation (“Donald” anywhere and “Trump” nowhere): |
| A month later, with little time to spare, the government hired Donald V. Davis, a former senior aide to Senator Tom Mondale of Minnesota and former Chief Security Operations Officer at the White House, to lead tactical centers. |

To demonstrate the scalability of Plaid models and achieve our goal of outperforming an autoregressive model in likelihoods, we train, evaluate, and release a large Plaid model called Plaid 1B.
Plaid 1B is a Transformer-based Plaid model with 1.3B parameters, trained for 314B tokens on OpenWebText2 .
In total, Plaid 1B was trained for $2.5\times 10^{21}$ FLOPs, which to our knowledge equals the largest purely diffusion-based language model trained in prior work .
We give further training details in [Appendix B](#A2).

### 6.1 Likelihood evaluation

We evaluate Plaid 1B’s likelihood in a zero-shot setting on a suite of six benchmark datasets originally used in : Penn Treebank , enwik8 and text8 , WikiText2 and WikiText103 , and the One Billion Word corpus .

use sliding windows of size 32 in their likelihood computation.
As a non-autoregressive model, Plaid doesn’t support sliding-window likelihood evaluations, so we use non-overlapping 1024-token sequences(^2^22We choose the splitting boundaries using the Plaid tokenizer, yielding sequences slightly shorter than 1024 tokens under the GPT-2 tokenizer.) when computing likelihoods.
Following , we use heuristic invertible detokenizers for PTB, 1BW, and WikiText to minimize the effect of tokenization artifacts on the perplexity results.
For a fair comparison, we also recompute GPT-2 likelihoods using the same protocol, resulting in different numbers than .

In [Table 2](#S6.T2) we observe that Plaid 1B consistently outperforms the 124M parameter GPT-2 model, demonstrating that diffusion models are capable of scaling to perplexities on par with a small modern autoregressive model.

### 6.2 Unconditional samples

We generate from Plaid 1B by starting from $z_{1}\sim\mathcal{N}(0,\sigma^{2}(1)I)$, performing ancestral sampling of $p(z_{t-(1/T)}|z_{t})$ for $T=4096$ steps, and finally $\arg\max p_{\theta}(x|z_{0})$.
Following , we sample using a score temperature of $\tau=0.9$, which in our formulation corresponds to adding $\frac{1-\tau}{\tau}(\hat{x}_{\theta}(z_{t})-z_{t})$ to $\hat{x}_{\theta}(z_{t})$ at each step.

We generate unconditional samples with sequence length 1024 and present chosen samples in [Table 3](#S6.T3).
We observe that the model is capable of generating fluent text and remaining on-topic over several hundred words.
We provide random un-picked samples in [Appendix D](#A4).

### 6.3 Zero-shot control

Although Plaid models are trained in a purely unconditional fashion, we present a zero-shot control technique called token guidance which allows us to implement a number of conditioning structures at generation time.
We begin with classifier guidance, a technique which allows diffusion models to generate samples conditioned on an arbitrary attribute $y$.
Classifier guidance first trains a probabilistic classifier $p(y|z_{t})$ of $y$ given noisy latents $z_{t}$, and then biases the diffusion model’s sampling steps by a guidance term derived from the gradient of the classifier probability $\nabla_{z_{t}}\log p(y|z_{t})$.
Now, recall from [Section 3.2](#S3.SS2) that our denoiser $\hat{x}_{\theta}(z_{t})$ is parameterized in terms of a model $f_{\theta}(z_{t})$ which learns the distribution over the token $x^{(i)}$ at each position $i$ given $z_{t}$.
We can therefore implement many different conditioning structures via classifier guidance on probabilities derived from $f_{\theta}$ itself.
We give a few examples:

Conditioning on a span: We perform guidance on the joint probability of the span under the factorial model $p(x^{(a:b)}|z_{t})\propto\prod_{i=a}^{b}p(x^{(i)}|z_{t})$, where $f_{\theta}$ estimates each factor in the product. This lets us implement prefix completion and infilling as special cases.
Lexical constraints: In order to condition on the presence of a token without specifying its location, we perform guidance on the token’s probability under the unigram distribution $p(x^{\mathrm{(any)}}|z_{t})\propto\sum_{i}p(x^{(i)}|z_{t})$, where $f_{\theta}$ estimates each term in the sum.
Token-level weights: We can emphasize a specific conditioning token by multiplying the corresponding guidance term by a scalar weight.
Negation: We condition on the negation of an attribute $y$ by performing guidance on the complement probability $1-p(y|z_{t})$.

Using Plaid 1B and token guidance, we generate samples under various zero-shot control settings.
We present chosen samples in [Table 4](#S6.T4) and random samples in [Appendix D](#A4).
Despite being trained unconditionally, Plaid 1B is able to follow diverse conditioning structures.

## 7 Related work

We contribute to a growing body of work on diffusion-based language models .
Our biggest departure from those works is that we aim for strong likelihood performance, which to our knowledge has not been attempted in any prior work except for an appendix result from .
We therefore benchmark against well-known autoregressive models instead of prior diffusion language models.

The work most comparable to ours is CDCD , which is also a strong general-purpose diffusion language model.
However, without the ability to use standard likelihood-based benchmarks , it is difficult to say precisely where CDCD stands in comparison to autoregressive models: in every result, either CDCD underperforms the autoregressive baseline, or the evaluation metric saturates and lacks the statistical power to distinguish the models.
Many of the other works above share similar difficulties.
In contrast, our likelihood-based approach enables unambiguous comparisons to widely-known models.

Other diffusion language model works consider more constrained settings like controllable generation or sequence-to-sequence tasks , or propose hybrid approaches involving pretrained autoregressive models . Particularly, in concurrent work, finetune an OPT 13B  model into a hybrid model which is autoregressive over 25-token blocks and uses diffusion within blocks. Compared to their work, we focus on the more general setting of training a fully diffusion-based language model from scratch.

Finally, our work builds on recent advances in diffusion models for images .
Most notably, we adopt the framework of Variational Diffusion Models and extend it to language modeling.

## 8 Conclusion

In this work, we have taken the first steps toward a competitive likelihood-based diffusion language model.
We built Plaid 1B, which matches GPT-2 124M in likelihood by combining several algorithmic improvements and a scaling law analysis.
Our ablations show that maximizing likelihood does not substantially harm sample quality, and we show samples are fluent in both unconditional and zero-shot conditional settings.
Despite this progress, substantial work remains: Plaid narrows the compute-efficiency gap between diffusion and autoregressive language models to $64\times$, and we view this gap as a tractable and exciting open problem that may be addressed with further research.

## Appendix A VLB and heuristic weight schedules

Figure: Figure 4: VLB weight schedule and and heuristic weight schedules used in ablation experiments.
Refer to caption: /html/2305.18619/assets/x4.png

## Appendix B Experiment details

### B.1 Dataset

Unless otherwise noted, all models in this work are trained on a subset of OpenWebText2 which we filter to remove documents labeled as non-English.
The data is tokenized using a 32K-token BPE tokenizer which we train on the OpenWebText2 training split.

### B.2 Architecture

We use standard pre-activation Transformers models with RMSNorm normalization layers and GeLU nonlinearities throughout.
Unless otherwise noted, all Plaid models have 16 Transformer layers, which we found to be approximately optimal for our scale.
We scale autoregressive model depth approximately following .
For efficiency, our implementation uses FlashAttention and other fused kernels wherever applicable.
We train at sequence length 256 for all experiments except Plaid 1B.

### B.3 Optimization

We optimize all models using AdamW with parameter-specific learning rates derived by $\mu$Transfer based on a learning rate of $1.4\times 10^{-3}$ at width 256.
Each parameter’s weight decay is set to $\frac{4\times 10^{-5}}{\eta}$ where $\eta$ is that parameter’s learning rate.
We use a linear warmup on the learning rate and weight decay over the first 2500 steps, followed by a linear decay to zero over training.
We train at batch size 256 for algorithm ablations and 128 for scaling law experiments.

### B.4 Plaid 1B Training

We increase the base $\mu$Transfer learning rate to $2\times 10^{-3}$ (at width 256).
The denoiser network is a Transformer with 24 layers of width 2048 and a vocabulary size of 32K tokens, for a total of 1.3B parameters.
We train for 1.2M steps at batch size 256 and sequence length 1024, for a total of 314B tokens.
All other details are as written above.

## Appendix C IsoFLOP profiles

Figure: Figure 5: IsoFLOP profiles for autoregressive models (left) and diffusion models (right).
Refer to caption: /html/2305.18619/assets/x5.png

## Appendix D Plaid 1B Random Samples

### D.1 Unconditional

### D.2 Prefix completion: “Generative models of text are very versatile: they can be used”

### D.3 Infilling: “A year ago in Paris, […] Wow, what a great day!”

### D.4 Word-level weights: “Let’s talk about law and medicine.” with 5 × \times weight on “law”

### D.5 Word-level weights: “Let’s talk about law and medicine.” with 5 × \times weight on “medicine”

### D.6 Lexical constraints: “Donald” anywhere

### D.7 Negation: “Donald” anywhere and “Trump” nowhere