---
arxiv_id: "2308.12219"
title: "Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning"
year: 2023
source: arxiv2md
---

## Abstract

Abstract The recent surge of generative AI has been fueled by the generative power of diffusion probabilistic models and the scalable capabilities of large language models (LLMs).
Despite their potential, it remains elusive whether Diffusion-LLM can solve general language tasks comparable to their autoregressive counterparts.
This paper demonstrates that scaling masked discrete diffusion models w.r.t . data, sizes, and tasks can effectively make them strong language learners.
We introduce Diffusion-LLM s at scale by first acquiring knowledge from massive data via masked language modeling pre-training thanks to their intrinsic connections.
We then reprogram pre-trained Masked LMs into Diffusion-LLM s via diffusive adaptation, wherein task-specific finetuning and instruction finetuning are explored to unlock their versatility in solving general language tasks.
Experiments show that scaling Diffusion-LLM s consistently improves performance across downstream language tasks. We further discover that instruction finetuning can elicit zero-shot and few-shot in-context learning abilities that help tackle many unseen tasks by following both natural language instructions and even visual instructions for multimodal understanding,
and show promise in advanced and challenging abilities such as reasoning.

## 1 Introduction

Recent advances in generative modeling have led to remarkable progress in the field of generative AI.
In domains of continuous signals, diffusion probabilistic models have shown great success in rendering photorealistic images , immersive videos  and synthesizing high-quality audio  through iterative denoising, outperforming GANs and autoregressive (AR) models, even contributing to the surge of AI art.
The story is different in the domains of discrete signals comprising symbolic sequences such as natural languages, where AR large language models  have dominated the scene, delivering impressive generalist language abilities in language understanding and generating human-like texts, and can even follow natural language instructions to perform unseen tasks.

While many recent endeavors try unifying the generation paradigms by enabling large language models to draw  or speak , few explore generating discrete sequences such as languages with diffusion models.
We suggest that the revolutionized generative abilities of diffusion models give the promise of a strong complement to autoregressive LMs for several favorable reasons, including (1) global receptive field vs. one-sided context, and (2) non-autoregressive drafting-then-revising manner vs. restrictive unidirectional generation/autoregression.
Hence, an intriguing question arises: can diffusion models speak languages well?

The key to the great success of modern LLMs lies in their scalability which fosters powerful generalist capabilities.
The question of the capability of Diffusion-LLMs is thus in turn to ask about their scalability, which can be further boiled down into the following specific research questions regarding the three key ingredients of the success of large-scale LMs, i.e., data, model sizes, and tasks:

- (i)
On scaling data.
Acquiring general knowledge via self-supervised pre-training from massive unlabeled data plays a crucial role in the success of the modern NLP paradigms , hence it is also of importance to enable Diffusion LMs to learn from massive data.
Can Diffusion-LLMs leverage knowledge from large-scale data?
- (ii)
On scaling model sizes. It has been widely observed that the larger the model size, the more competent the LMs become. Can enlarging
Diffusion-LLMs effectively improve downstream tasks?
- (iii)
On scaling tasks.
What makes LLMs most attractive is they can tackle new tasks that they were never exposed to during training by following natural language and even multimodal instructions with little to no demonstrations.
Can Diffusion-LLMs exhibit general zero-shot and few-shot in-context learning capabilities to generalize to unseen tasks?

Figure: Figure 1: Overview. (A) Comparative illustration of LM paradigms, i.e., autoregressive LMs vs. Diffusion LMs. (B) Overall illustration of the proposed Diffusion-LLM where large-scale pre-trained masked LMs are reprogrammed to Diffusion-LLMs via generative surgery.
Refer to caption: x1.png

In this paper, we delve into the potential of Diffusion-LLMs through the three research questions.
We highlight our contributions and findings as follows:

(1) We first demonstrate the intrinsic connection between masked LMs and discrete diffusion models, which permits us to treat pre-trained masked LMs of various scales as pre-trained Diffusion-LLMs, without the need for expensive learning from scratch.
We then reprogram pre-trained masked LMs into Diffusion-LLMs via diffusive adaptation, where task-specific finetuning and instruction finetuning  are explored for solving certain downstream tasks or general language problems, showing Diffusion-LLMs benefit from pre-training on large scale data.
(2) We reveal that large-scale Diffusion-LLMs can serve as strong sequence generative models to tackle multitasks, exhibiting competitive performance compared with autoregressive LMs.
And the performance consistently improves as the model sizes scale up. (3) We further elicit zero-shot and few-shot abilities for Diffusion-LLMs to tackle multiple unseen tasks, spanning from language and visual ones, through both language and vision instruction finetuning.
Notably, Diffusion-LLMs demonstrate promising structured reasoning behaviors thanks to their flexible non-autoregressive generation order. Nevertheless, their capacity to tackle complex reasoning tasks remains an ongoing challenge awaiting resolution.

To sum up, we hope that our explorations provide valuable insights into the scalability of Diffusion-LLMs and their potential as a viable complement in tackling generative language tasks across the board.

## 2 Preliminaries: Diffusion Models for Sequence Generation

Language processing tasks can be unified as sequence-to-sequence problems , modeling the conditional distribution $p_{\theta}(\mathbf{x}|\mathbf{c})$, where $\mathbf{x}=(\mathbf{x}^{[1]},\mathbf{x}^{[2]},\dots,\mathbf{x}^{[N]})$ is a target sequence composing $N$ tokens and $\mathbf{c}$ is the given context.
For example, we may want to generate responses $\mathbf{x}$ conditioned on the prompt $\mathbf{c}$, or it can be unconditional generation if no context is provided (i.e., $\mathbf{c}=\phi$).
As a result, one thing we care about is the capability of generative models for sequence data $\mathbf{x}$, e.g., the prevailing autoregressive models or diffusion models.
In this section, we provide the necessary background on diffusion-based sequence generative models, where we abuse the notation and use $p_{\theta}(\mathbf{x})$ for both conditional $p_{\theta}(\mathbf{x}|\mathbf{c})$ and unconditional $p_{\theta}(\mathbf{x}|\mathbf{c}=\phi)$ for brevity.

##### Diffusion Models

are a class of generative models characterized by a pair of Markov processes, i.e., a forward diffusion process and a backward denoising process.
The forward process ${q(\mathbf{x}_{1:T}|\mathbf{x}_{0})=\prod_{t=1}^{T}q(\mathbf{x}_{t}|\mathbf{x}
_{t-1})}$ gradually perturb the data $\mathbf{x}_{0}\sim q(\mathbf{x}_{0})$ into a stationary distribution $q(\mathbf{x}_{T})$ with $T$ increasingly noisy steps $\mathbf{x}_{1:T}=\mathbf{x}_{1},\mathbf{x}_{2},\dots,\mathbf{x}_{T}$.
The learned backward process ${p_{\mathbf{\theta}}(\mathbf{x}_{0:T})=p(\mathbf{x}_{T})\prod_{t=1}^{T}p_{
\mathbf{\theta}}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}$, reversely, gradually denoises the samples towards the data distribution.
To fit the model $p_{\mathbf{\theta}}(\mathbf{x}_{0})$ to the data distribution $q(\mathbf{x}_{0})$, the denoiser model is typically optimized by the variational bound of the negative log-likelihood :

$$ $\displaystyle\mathbb{E}_{q(\mathbf{x}_{0})}\left[-\log p_{\theta}(\mathbf{x}_{ 0})\right]\leq\mathbb{E}_{q(\mathbf{x}_{0:T})}\left[-\log\frac{p_{\mathbf{ \theta}}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_{0})}\right]= \mathcal{L}_{0}+\sum_{t=2}^{T}\mathcal{L}_{t}+\text{const.},$ (1) $$

where $\mathcal{L}_{0}=\mathbb{E}_{q}\left[-\log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}
_{1})\right]$, and $\mathcal{L}_{t}=\mathbb{E}_{q}\left[\text{KL}[q(\mathbf{x}_{t-1}|\mathbf{x}_{t
},\mathbf{x}_{0})\|p_{\mathbf{\theta}}(\mathbf{x}_{t-1}|\mathbf{x}_{t})]\right]$ for $t\in[1,T]$.

In general, diffusion models can be categorized into continuous and discrete diffusion models according to distribution type for data perturbation.
Continuous diffusion models with Gaussian perturbation have demonstrated impressive performance in generating continuous signals  but still struggle with satisfactory generation quality in natural languages .
A critical challenge herein is the pitfall of discreteness  that makes Gaussian perturbation on embeddings hardly provide effective training signals.
In contrast, discrete diffusion models directly operate over the discrete state space of tokens, providing an attractive alternative for generative sequence learning. Therefore in this paper, we explore developing Diffusion-LLMs upon discrete diffusion.

##### Discrete Diffusion Models

cover a subset of diffusion models for which transition probabilities between timesteps are discrete distributions.
Since the forward diffusion process is applied independently to each token of a sequence $\mathbf{x}$, for the sake of brevity, we abuse the notation $\mathbf{x}_{t}$ for arbitrary tokens at diffusion timestep $t$.
Formally, $\mathbf{x}_{t}\in\{0,1\}^{|\mathcal{V}|}$ is a token represented as a one-hot vector, where $\mathcal{V}$ is the vocabulary of all possible tokens.
Let $\texttt{Cat}(\mathbf{x};\mathbf{p})$ be a categorical distribution on $\mathbf{x}$ with probabilities given by vector $\mathbf{p}$ on $|\mathcal{V}|-1$ dimensional probability simplex, and the forward transition be $q(\mathbf{x}_{t}|\mathbf{x}_{t-1})=\texttt{Cat}\left(\mathbf{x}_{t};\mathbf{p}
=\beta_{t}\mathbf{x}_{t-1}+(1-\beta_{t})\mathbf{q}_{\text{noise}}\right),$
where $0\ll\beta_{t}<1$ is the noise schedule controlling the degree of perturbation at timestep $t$, and $\mathbf{q}_{\text{noise}}$ is the probability vector of stationary distribution $q(\mathbf{x}_{T})$, i.e., $q(\mathbf{x}_{T})=\texttt{Cat}(\mathbf{x}_{T};\mathbf{p}=\mathbf{q}_{\text{
noise}})$.
In this case, the distribution of corrupted sample $\mathbf{x}_{t}$ given its original data $\mathbf{x}_{0}$ has a closed-form expression:

$$ $q(\mathbf{x}_{t}|\mathbf{x}_{0})=\texttt{Cat}\left(\mathbf{x}_{t};\mathbf{p}= \alpha_{t}\mathbf{x}_{0}+(1-\alpha_{t})\mathbf{q}_{\text{noise}}\right),$ $$

where $\alpha_{t}=\prod_{i=1}^{t}\beta_{i}$. This shows that the diffusion process is intuitively a convex combination between data and noise where the $\alpha_{t}$ controls the degree of corruption at different timesteps. In particular, $\alpha_{t}$ decreases as the timestep increases. With sufficiently large timesteps, we have $\alpha_{T}\approx 0$, which preserves no information from the data at the end of the diffusion process.

Different stationary distributions $\mathbf{q}_{\text{noise}}$ lead to different formulations of discrete diffusion models. One typical design is the absorbing diffusion with $q(\mathbf{x}_{T})=\{1~{}\text{if}~{}\mathbf{x}_{T}=\texttt{[MASK]};~{}0~{}
\text{if}~{}\mathbf{x}_{T}\not=\texttt{[MASK]}\}$, where [MASK] is an absorbing state.
According to Eq. (2), this formulation results in $\mathbf{x}_{t}$ either being masked or the same as $\mathbf{x}_{0}$, with a masking ratio $(1-\alpha_{t})$.
This makes absorbing diffusion resemble masked LMs  as points out.

##### Reparameterized Discrete Diffusion Models

reparameterize the backward transition of Diffusion-LLMs that reformulates the training objective of discrete diffusion models into

$$ $\displaystyle\mathcal{L}_{t}=\mathbb{E}\big{[}-\lambda_{t-1}^{(2)}\left(1- \mathds{1}(\mathbf{x}_{t}=\mathbf{x}_{0})\right)\log p_{\mathbf{\theta}}( \mathbf{x}_{0}|\mathbf{x}_{t})\big{]},$ (2) $$

where $\mathds{1}(\cdot)$ is indicator function.
Under the formulation of absorbing diffusion, Eqn. 2 resembles a weighted MLM objective .
demonstrate that Eqn. 2 is a more effective training protocol compared to Eqn. 1 for generative discrete diffusion models, showing performance on par with autoregressive LMs
on representative machine translation benchmarks for the first time.
In this paper, we use RDM as our primary training objective for building our Diffusion-LLMs (see §A for more details).

##### Generative Process of Discrete Diffusion Models.

Diffusion models yield new samples by their reverse generative process of iterative denoising.
Under the formulation of absorbing diffusion, the denoising process can be characterized in an iterative mask-predict manner .
Specifically, the starting sequence is initialized by all [MASK] tokens, and in each iteration, some masked tokens are replaced by the model predictions from $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ while some unmasked tokens are remasked, according to specific strategies/schedules .
In this paper, we follow to unmask positions with top-$k$ predicted $\log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{t})$, and mask all the rest position in each denoising step(^1^11See §A for concrete noise schedules, and for the justification of this sampling strategy.).

## 3 Scaling Diffusion Language Models w.r.t . Data, Sizes and Tasks

Developing Diffusion-LLMs that leverage the advantages of both the generative power of both diffusion models and the scalability of large pre-trained LMs is a promising yet challenging endeavor.
The key to the success of the current standard paradigm of large generative LMs is acquiring knowledge via massive pre-training and generating in a prompt-response manner for preferable output for many tasks.
For Diffusion-LLMs, (1) how to benefit from pre-training at scale, and (2) how to best fit the prompt-response paradigm, are the crucial open questions.
In this section, we will elaborate on how to empower Diffusion-LLMs with knowledge from pre-training of large-scale data as well as model sizes, and extend their generative capabilities for extensive downstream tasks.

### 3.1 Knowledge Acquisition via MLM pre-training

The theoretical framework of discrete diffusion models has an intrinsic connection to masked language modeling (MLM), which was discussed in  and .
Among various types of discrete diffusion models, the absorbing diffusion  resembles a generalized masked language modeling, which has been shown to be an effective training objective in pre-training foundation models .
Specifically, absorbing diffusion defines a stationary distribution: $q(\mathbf{x}_{T})=\{1~{}\text{if}~{}\mathbf{x}_{T}=\texttt{[MASK]};~{}0~{}
\text{if}~{}\mathbf{x}_{T}\not=\texttt{[MASK]}\}$,
where [MASK] is an absorbing token.
According to Eq. (2), this formulation results in $\mathbf{x}_{t}$ either being masked or the same as $\mathbf{x}_{0}$, with a masking ratio $(1-\alpha_{t})$.
Consequently, $\mathbf{x}_{t}=\mathbf{x}_{0}$ if and only if $\mathbf{x}_{t}\not=\texttt{[MASK]}$, which aligns the reparameterized training objective in Eq. (2) exactly with the masked language modeling objective.

This connection allows us to establish Diffusion-LLMs by pre-training with MLM objectives from massive raw textual data.
We can even treat abundant community-available pre-trained MLMs  as pre-trained Diffusion-LLMs, and can depart from them for downstream tasks at a very low cost, bypassing the expensive pre-training stage.

### 3.2 Diffusive Adaptation: Reprogramming pre-trained MLMs to Diffusion-LLM s

Existing masked LMs are primarily designed to serve as sequence encoders, and are not able to generate sequences by default.
Despite their connections to absorbing discrete diffusion, it is non-trivial to naively sample from masked LMs through the iterative denoising process of absorbing diffusion.
One major reason is that absorbing diffusion generates sampling by iterative applying $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ from complete noise to the final prediction (i.e., ranging gradually from $100\%$ to $0\%\texttt{[MASK]}$ tokens) through different timesteps, whereas vanilla masked LMs are only pre-trained with a limited and constant masking ratio (e.g., 15%).

In order to elicit the pre-trained masked LMs’ ability for sequence generation, we propose diffusion adaptation to eliminate the gap between pre-trained masked and Diffusion-LLMs, where we further finetune pre-trained Masked LMs with diffusion training objective such that sampling with the denoising process becomes possible.
In particular, we follow the reparameterized training and sampling method in  as described in §2.
Similar to the time agnostic design in , we do not introduce any extra parameters to differentiate different diffusion timesteps.

For different purposes, we perform diffusive adaptation for Diffusion-LLMs in two ways:

- •
Optimizing specialist capabilities on certain downstream tasks via task-specific finetuning.
To verify the feasibility of diffusive adaptation, we finetune pre-trained masked LMs on specific datasets for each downstream task.
Moreover, we further perform finetuning on pre-trained models of different scales so as to study the scalability of Diffusion-LLMs.
- •
Eliciting generalist capabilities on extensive tasks via instruction finetuning.
Finetuning on a collection of tasks phrased as instructions (i.e., instruction finetuning) enables LMs to better respond to instruction prompts and generalize to unseen tasks .
Inspired by this, we apply diffusive adaptation to pre-trained masked LMs by instruction finetuning to study whether Diffusion-LLMs can acquire few-shot and zero-shot abilities like autoregressive LLMs.

##### Engineering considerations for scaling.

Both scenarios above handle conditional sequence generation tasks from input to output, which require the model to generate target sequences according to the given prompts.
Instead of incorporating an extra encoder,
we handle conditional generation by organizing data in a prompt-response format(^2^22 A prompt-response formatted example for German$\to$English translation (Vielen dank$\to$Thank you): “Translate the German sentence to English. German: Vielen dank. English: Thank you.”).
This enables our model to share the same training infrastructure as prevailing large decoder-only AR LMs  while incorporating extra encoder adds to the complexity in key techniques for scaling LLMs such as pipeline parallelism .
Our design decision ensures the scalability of our Diffusion-LLMs in engineering.

Besides, we incorporate a length predictor, a common practice in non-autoregressive text generation , to determine the lengths of predicted sequences.
We pick its top-$k$ length predictions for parallel length beam search, where $k$ is referred to as the length beam size.
During tuning, we only apply the diffusion process to the target response tokens and compute loss on them.
During inference, we append the initial fully masked sequences to the prompts and denoise from them.

## 4 Experiments

In this section, we first introduce our general experimental setups in §4. Then we conduct three parts of experiments progressively regarding scaling on data (§4.2), model sizes (§4.3), and the number of tasks (§4.4).

**Table 1: SacreBLEU on IWSLT14 De$\rightarrow$En and WMT14 En$\rightarrow$De, and Rouge-L on Gigaword-10k. We use 10 length beams for all the results with length prediction. Results out of (inside) parentheses are obtained with length prediction (oracle target length). Results of DiffusionLM and DiNoiser are quoted from . And the results of GENIE are our reimplementation obtained with the open-source code and checkpoint of the orignal paper. “#Params.”: Number of non-embedding parameters. “Type”: whether the training objective and sampling method are autoregressive or follow reparameterized diffusion models . “pre-trained”: whether initialized from pre-trained models. “${\dagger}$”: The architecture is in the format of # layers / hidden dimension. For models w/ encoder, the # layers consist of layers in both encoder and decoder. “${\ddagger}$”: For IWSLT14, we follow previous practice and use a smaller version (39M) of Transformer-BASE, which has a dimension of 1024 in the feed-forward layers is 1024 and 4 attention heads.**
|  | Method | Architecture^† | #Params. | Pre-trained | IWSLT14 | WMT14 | Gigaword-10K | Gigaword |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w/ encoder | AR | 12/512 | 39-43M^‡ | ✗ | 33.30 | 26.85 | 10.42 | 34.5 |
| DiffusionLM | 12/512 | 39-43M^‡ | ✗ | 29.11 | 17.41 | - | - |  |
| DiNoiSer | 12/512 | 39-43M^‡ | ✗ | 31.44 | 24.62 | - | - |  |
| RDM | 12/512 | 39-43M^‡ | ✗ | 32.14 | 26.54 | 16.25 | - |  |
| GENIE | 12/768 | 86M | ✓ | - | - | - | 30.24 |  |
| w/o encoder | AR | 12/768 | 86M | ✗ | 26.07 | - | 4.7 | - |
| RDM | 12/768 | 86M | ✗ | 28.79 (29.12) | 26.09 (26.86) | 10.01 (10.66) | - |  |
| Diffusion-LLM (RDM + XLM-R) | 12/768 | 86M | ✓ | 34.10 (35.78) | 26.65 (26.64) | 27.52 (28.83) | 34.08(35.65) |  |
| Diffusion-LLM (RDM + XLM-R) | 48/4096 | 9.7B | ✓ | 38.57 (40.65) | 30.34 (32.81) | 31.54 (32.57) | 34.24(34.26) |  |

##### Model.

Throughout the experiments, we use XLM-RoBERTa  as our foundation models, which is pre-trained on CC100 , a multilingual corpus containing 167B tokens of 100 languages, with four model sizes (numbers of non-embedding parameters) at different scales, i.e., 86M, 304M, 2.8B, and 9.7B.

##### Data & task.

We investigate our approach for its specialist ability in respective downstream tasks and generalist ability to solve massive unseen tasks using natural language instructions. The datasets we use to finetune our model are as follows:

(1) Downstream task datasets.
We evaluate whether our approach can help Diffusion-LLMs serve as strong specialized models on multiple representative downstream tasks: (1) IWSLT14 for De$\rightarrow$En translation; (2) WMT14 for En$\rightarrow$De translation; (3) Gigaword-10K for text summarization; and (4) the full Gigaword summarization.

(2) Instruction finetuning datasets.
We follow and finetuned XLM-R models of different scales with the Flan 2022 Collection  with diffusion training objective.
It is the publicly available version of the instruction tuning data for Flan-T5 and Flan-PaLM, covering over 1.8K tasks.
It combines several multitask learning datasets with instructions , combined with a few extra chain-of-thought and dialog data.

Figure: Figure 2: An exemplary generation process of Diffusion-LLM for machine translation. Notice that the target translation contains three segments, which are generated simultaneously by the diffusion language model.
Refer to caption: extracted/6217187/figs/qualitative_mt.png

### 4.1 Exploring the Design Space of Diffusion-LLM s

We first validate that our design decisions lead to a competitive Diffusion-LLM, which includes (1) time-agnostic RDM and (2) no incorporation of an extra encoder. As shown in Tab. 1:

- •
RDM is the most performant Diffusion-LLM, being an ideal candidate for scaling.
It outperforms both continuous diffusion language models, DiffusionLM and DiNoiSer.
Moreover, it performs on par with AR model on IWSLT14 and WMT14, outperforming AR on Gigaword-10K, indicating that RDM is a strong enough diffusion model to exploit.
- •
Using an architecture without encoders has a minor effect on model performance.
Despite that model without encoders slightly underperforms their counterpart with encoders, bidirectional perception of diffusion models and scaling in data can mitigate this gap, similar to the findings in .
As results in Tab. 1 show, compared to AR models, RDM has a smaller performance gap between models with and without encoders.
Such discrepancy becomes neglectable on large datasets like WMT14 and models initialized from the pre-trained XLM-R models.

The results verify the competence of our model at moderate scales without pre-training, preparing it to be an ideal candidate for scaling.
We then build upon this to investigate its scalability w.r.t., data, model sizes, and tasks.

### 4.2 Empowering Diffusion-LLM s with Large-scale Data

We apply diffusive adaptation on XLM-R-BASE  model architecture on sequence generation benchmarks.
In this way, we verify the feasibility of diffusive adaptation.
Meanwhile, by comparing the performance of diffusive adapted RDMs to models trained from scratch, we confirm that our Diffusion-LLM takes advantage of large-scale self-supervised learning, namely the scaling with data.

##### pre-training at scale benefit Diffusion-LLM s.

The results in Tab. 1 demonstrate that Training RDM through diffusive adaptation from pre-trained MLMs boosts the model performance compared to the models trained from scratch, whose impact is particularly significant on small datasets like Gigaword-10K.
Compared to GENIE, a pre-trained Diffusion-LLM based on continuous diffusion models, RDM adapted from XLM-R is also stronger, further confirming the advantage of discrete Diffusion-LLM.
As qualitatively shown in Fig. 2, Diffusion-LLMs generate fluent and semantically accurate translation(^3^33The intermediate steps demonstrate that the models generate three clauses simultaneously, implying a global perception that plans the generation of the whole sequence. We consider this benefits the model on more complex generation tasks, which we discuss in §4.5.2.), further confirming the feasibility of our generative surgery to the pre-trained MLMs.
With these findings, we confirm the feasibility of our diffusive adaptation method and verify the scalability of our Diffusion-LLMs regarding training data.

Figure: Figure 3: Scaling curves of task-specific finetuning on IWSLT14, WMT14 and Gigaword-10K. We obtain results of mT5 on IWSLT14 by ourselves. The results of T5 on WMT14 are from . “OL”: results obtained with oracle target lengths. “LB=10”: length prediction results with 10 length beams. “#Params.”: Number of effective parameters (i.e., non-embedding parameters).
Refer to caption: extracted/6217187/figs/scaling_task_w_size4.png

### 4.3 Scaling up the Size of Diffusion-LLM s Boosts Downstream Tasks

We now move on to the scalability with respect to model sizes.
We finetune XLM-R models of different scales , whose effective parameters (i.e., number of non-embedding parameters) range from $<$100M to 10B.
Notably, when scaling up to 10B, the model shows impressive performance that surpasses base-sized models by a remarkable margin (Tab. 1).

Fig. 3 shows the scaling curve of model performance with respect to model sizes.
It demonstrates that the performance of the finetuned diffusion models substantially increases as the model size increases.
This shows the scaling law of Diffusion-LLMs in terms of model size.
In addition, we also include the performance of (m)T5  at similar scales as references to intuitively understand how scalable our Diffusion-LLMs are.
Note that the performance of different models is intricately affected by not only the model size but also numerous factors including model designs, pre-training budget, pre-training objectives, as well as pre-training data .
In Fig. 3, although we see a performance gap between the finetuned (m)T5 and XLM-R models at similar scales, the discrepancy is minor and does not seem amplified as models scale up.
Therefore, while there is still ample room for improving large-scale pre-trained Diffusion-LLMs, we believe that the path of scaling up these models holds great promise.

### 4.4 Instruction-Finetuning Helps Generalize to Unseen Tasks

**Table 2: Zero-shot SacreBLEU of instruction-tuned Diffusion-LLMs on IWSLT14 De$\rightarrow$En translation. For Flan 2021, we explicitly remove all German data for strict evaluation. Results are obtained with oracle length.**
| Architecture | Strict Flan’21 | Flan’22 |
| --- | --- | --- |
| instruction-tuned Diffusion-LLM: |  |  |
| XLM-R-BASE (85M) | 7.15 | 21.26 |
| XLM-R-LARGE (304M) | 22.52 | 25.24 |
| XLM-R-XL (2.8B) | 27.27 | 28.13 |
| XLM-R-XXL (9.7B) | 28.74 | 29.59 |
| ref: supervised AR on 160k De$\rightarrow$En data: 33.30 |  |  |

A fascinating property that motivates scaling LMs up is that LLMs can follow instructions and show impressive few-shot or even zero-shot performance .
We now investigate whether diffusion models can also exhibit zero-shot and few-shot performance when being scaled up.

#### 4.4.1 Instruction finetuning elicits scalable zero-shot performance

Figure: Figure 4: Zero-shot performance of instruction-tuned Diffusion-LLMs (Flan-XLM-Rs) of different scales. OL means the results are obtained with oracle length, while LB means the number of length beams to sample the target with length prediction. The model sizes refer to the number of non-embedding parameters.
Refer to caption: extracted/6217187/figs/zero-shot-new.png

##### Strict zero-shot evaluation on IWSLT14 De → → \rightarrow → En .

We first conduct a strict zero-shot evaluation to study if Diffusion-LLMs can acquire zero-shot capabilities through instruction finetuning.
Specifically, we evaluate on IWSLT14 De$\rightarrow$En translation task, for which we instruction-finetune Diffusion-LLMs on Flan 2021 Collection  with all German data removed to ensure that the De$\rightarrow$En translation becomes a strictly unseen task.
As shown in Tab. 2, the instruction-tuned Diffusion-LLMs demonstrate scalable zero-shot performance even without finetuning with German data, signifying that Diffusion-LLMs are able to follow instructions.

##### Extensive zero-shot evaluation with large-scale instruction tuning.

We then follow the recommended settings and conduct larger-scale instructions tuning on the full Flan 2022 Collection  and run extensive evaluations(^4^44We continue to evaluate on the IWSLT14 dataset. Besides, we also evaluate several datasets used in . In detail, MMLU includes multiple-choice exam questions from 57 tasks covering humanities, social science, STEM, and more. TyDiQA is an open-book question-answering benchmark across 8 typologically diverse languages).
Following , we named our instruction-tuned checkpoints on Flan 2022 Collection as Flan-XLM-R.
The results in Fig. 4 suggest that the Flan-XLM-R models are indeed general-purpose zero-shot learners, and their zero-shot performance substantially improves as the model scales.
In particular, we highlight the results on IWSLT14.
The largest model, Flan-XLM-R-XXL even achieves a 30.90 zero-shot ScareBLEU score, only 2.4 below the performance of widely adopted supervised transformer baselines (33.30 as shown in Tab. 2).
This indicates the Flan-XLM-R models produce a very good language generation quality.

#### 4.4.2 Diffusion-LLM s can learn in context

**Table 3: SacreBLEU of instruction-tuned Diffusion-LLMs on IWSLT14 De$\rightarrow$En under oracle lengths with removed instruction, where the model can only figure out the objective of the task through demonstrations.**
| Diffusion-LLM Arch. | w/o demonstration | w/ demonstration |
| --- | --- | --- |
| XLM-R-LARGE (304M) | 2.58 | 5.38 |
| XLM-R-XL (2.8B) | 4.42 | 12.18 |
| XLM-R-XXL (9.7B) | 4.16 | 19.16 |

We also evaluate the in-context learning ability of the Diffusion-LLMs.
We construct an experiment on IWSLT14 with removed instructions.
In this case, the model can only rely on the demonstration to figure out the task objective.

The result supports that Diffusion-LLMs can learn in context.
The model is unable to produce the desired outcome without prior knowledge of the task.
However, when given a demonstration, it can learn to treat the task as a translation task, showing obvious performance improvement, which also scales with model sizes.

### 4.5 Exploring Reasoning with Diffusion-LLM s

We are also interested in exploring the reasoning abilities of our Diffusion-LLMs as it is a crucial emergent ability that distinguishes LLMs from the small ones .
Understanding how these models develop reasoning capabilities could provide insights into their scalability, generalization, and potential applications in complex problem-solving tasks. Moreover, investigating their reasoning mechanisms may help bridge the gap between traditional autoregressive LLMs and diffusion-based architectures, shedding light on their respective strengths and limitations in various domains.
In this section, we will highlight our key findings and include detailed discussions.

#### 4.5.1 Quantitative Results

As shown in Fig. 5, we find, by simply instruction tuning, even Flan-XLM-R-XXL fails to emerge non-trivial reasoning performance on GSM8K , a benchmark dataset for mathematical reasoning, and its German translated version in MGSM .

Figure: Figure 5: Evaluating reasoning on GSM datasets. (A) Performance of Diffusion-LLM (Flan-XLM-R) and autoregressive LLMs (Flan-T5) of different scales. Although Flan-XLM-R-XXL fails to emerge with non-trivial performance, the performance improves with (B) task-specific tuning on GSM8K training set or better foundation model (adapted from LLaMa3.1 8B Instruct by continual training on RedPajama). We also include the result of fine-tuning autoregressive LLaMa3.1 8B on GSM8k for reference.
Refer to caption: x2.png

We conduct further analyses to understand the unsatisfying performance and conclude that this is due to the limitation of the pre-trained recipe instead of the Diffusion-LLM paradigm.

- •
Diffusion-LLM can perform reasoning tasks. Given that task-specific fine-tuning offers an effective strategy to predict the existence of emergent ability , we finetune Flan-XLM-R-XXL on the training set of GSM8K. We find that the model performance rockets to a non-trivial accuracy of 26.73%, confirming the capability of Diffusion-LLMs in reasoning.
- •
Reasoning performance of
Diffusion-LLMs can be improved with better pre-training recipes. We suggest that the reasoning performance of Flan-XLM-R-XXL is limited by its pre-training recipe, which significantly falls short of modern designs . As a preliminary verification, we adapt an LLaMa3.1 8B into Diffusion-LLMs by fine-tuning on RedPajama(^5^55We adapt the autoregressive LLaMa into a Diffusion-LLM by replacing the causal attentions with bidirectional ones and continue pre-train the model with diffusion objective. Meanwhile, we also post-process the output logits by right shifting for one token to fill the gap that the output of autoregressive models predicts the next token while the output of Diffusion-LLMs predicts the token at the same positions as the mask tokens .) and obtain 13.48 after instruction tuning on Flan 2022 and 42.77(^6^66This performance is comparable to fine-tuning autoregressive LLaMa 3.1 with GSM8K training set, which is 40.36. Although we note that fine-tuning on GSM8K actually degrades the performance LLaMa 3.1 8B from near 80, we consider the fine-tuning results can represent model performance with sufficient training under the training distribution and implies similar capabilities between autoregressive and Diffusion-LLMs.) after task-specific fine-tuning on GSM8K. Both greatly outperform XLM-R-XXL with the same fine-tuning settings.

These findings support that Diffusion-LLMs are also able to perform reasoning as autoregressive LMs do while we need an improved pre-trained models to fully unveil the potential.

Figure: Figure 6: Qualitative investigation into the reasoning abilities of diffusion language models. (a) A causal graph that represents the dependencies between the reasoning steps. (b) An example question, its reference answer, and answers from autoregressive models. (c) The answer from Diffusion-LLM (Flan-XLM-R-XXL) and its generation process.
Refer to caption: x3.png

#### 4.5.2 Qualitative Analysis

In reasoning tasks, a model needs to generate intermediate reasoning steps to approach the final answers, where the model heavily relies on the intermediate results generated by itself to predict the final answer.
This leads to constraints on the generation order when performing reasoning tasks.
Given the non-autoregressive nature of Diffusion-LLMs, we wonder whether they show different behaviors in generation orders compared to fixed left-to-right autoregressive reasoning.

##### Understanding target dependencies with causal graphs.

Fig. 6(a) depicts the causal graph for the exemplary problem and its solution shown in Fig. 6(b).
We argue that to solve the task with reasoning, language models must generate tokens in an order that conforms to a topological sort of the causal graph.
Specifically, it means the following requirements for the generation order:
(1) the final results should come after the last intermediate result;
(2) the intermediate results should come after listing the corresponding equation;
(3) to correctly list an equation, models need to have the idea for this equation, copying calculation results from previous steps or numbers provided by the question; and
(4) before these, models need to propose the idea for each step first.

##### Diffusion-LLM s can figure out feasible topological sorts on the causal graph.

A follow-up question is whether the generation process of autoregressive models and our diffusion language models conform to possible topological sorts.
One feasible topological sort is exactly the left-to-right traversal on the chain-of-thought text and is implicitly provided to autoregressive models during training.
Diffusion language models, on the other hand, learn without a fixed generation order due to random masking.
Fig. 6(c) demonstrate its generation process of solving the exemplary question.
Despite incorrect final answers, the generative process does conform to a topological sort of the causal graph in Fig. 6(a).
The model generates the ideas first, then writes the formulas, and finally calculates the answers.
We randomly sampled 30 samples generated by Diffusion-LLMs and found that 21 out of these samples conformed to a topological order.
This implies that diffusion language models learn to figure out feasible topological sorts, namely a structure reasoning ability.

##### Diffusion-LLM s reason with a flexible mind.

Notably, diffusion language models are able to explore different topological sorts different from autoregressive models thanks to less constrained generative orders.
We highlight some of the interesting patterns resulting from this.

- •
Easy first.
Fig. 6(c) shows that the model fills up the fixed pattern (i.e., “the final answer is”) at first, showing a quite smart easy-to-hard generation behavior.
- •
Planning ahead.
In Fig. 6(c), the model constructs the framework for the solution before diving into arithmetic. Actually, we have seen similar behavior in Fig. 2 where the model generates three clauses simultaneously. Both cases demonstrate the models’ global perception which helps plan the generation of the whole sequence.
- •
Forward and backward reasoning.
During the reasoning process in Fig. 2, on STEP 31, the model begins the solution with the idea for the last reasoning step. This shows backward reasoning behavior, a very common human behavior that is especially helpful for challenging reasoning activities such as finding mathematical proofs .
- •
Backtracing.
The backward transition of diffusion models formally supports backtracing by remasking. In Fig. 6(c), STEP 47 erases a “the” token. This ability helps avoid accumulating errors in predicted tokens .

#### 4.5.3 Diffusion-LLM s Show Superiority with Reasoning That Nessitates Implicit Planning

Figure: Figure 7: An example of Path-Finding on Path-Star Graph with degree of central start as 4 and length of each path is 3. The task inputs the graph as well as the node indices of the central start (Node 4 in the example) and goal node (Node 1 in the example), and requires a model to figure out the path from the central start and the goal node (“Node 4, Node 3, Node 1" in the example). This task serves as a minimal planning task as it can be simply solved only if the models look ahead to see which first step leads to the goal.
Refer to caption: x4.png

Inspired by the qualitative observations, we consider that Diffusion-LLMs could be helpful when the logical reasoning process differs from the sequential (i.e., left-to-right) order of words in the text.
This is common, especially for challenging reasoning tasks, implicit thought processes are required to plan before giving the outcome rationales, which is also known as meta-CoT .

##### Path-Finding on Path-Star Graphs.

To verify this, we experiment to see whether Diffusion-LLMs are able to solve Path-Finding on Path-Star Graphs , a simple planning-related problem that autoregressive models struggle with(^7^77We refer readers to which elaborate on the planning capability of Diffusion-LLM in more details.).
As shown in Fig. 7, a path-star graph contains a central start node with multiple paths of equal length radiating outward from it, where one of the paths leads to the designated goal node.
The task for the model herein is to find the correct path from start to goal.
The task is simple only if the model can look ahead to discover which first step leads to the goal.
The straightforward failure mode makes it representative of studying the planning capability of the models.

Figure: Figure 8: Comparison between autoregressive and Diffusion-LLMs on Path-Finding on Path-Star Graph . We experiment on two settings where the degree of the central start is 2 and 4, respectively. “Random”: a random walk predictor that randomly selects a path on the star graph; “Autoregressive”: fintuned LLaMa3.1-8B-Instruct; “Diffusion”: Diffusion-LLM fine-tuned from Flan-XLM-R-XXL.
Refer to caption: x5.png

##### Comparison with AR-LMs.

The comparison between autoregressive models and Diffusion-LLMs in Fig. 8 reveals a significant advantage of Diffusion-LLMs in solving reasoning tasks that require implicit planning. To investigate this further, we fine-tune two representative models—LLaMa3.1-8B-Instruct and Flan-XLM-R-XXL—on the Path-Finding on Path-Star Graphs tasks and evaluate their generalization performance on unseen graph structures.
Our evaluation encompasses two different experimental settings designed to test the models’ ability to perform multi-step reasoning and implicit planning. In both cases, the autoregressive models exhibit limited success, struggling to capture the underlying graph structure and plan effectively. Their performance closely resembles a random walk, indicating an inability to leverage structural information for accurate predictions. In contrast, Diffusion-LLMs consistently produce near-perfect predictions, demonstrating a remarkable capacity for handling implicit planning tasks.

This performance disparity highlights a key architectural difference: the bidirectional receptive field in Diffusion-LLMs allows the model to capture global dependencies across the input more effectively. This not only facilitates better reasoning in structured environments but also gives Diffusion-LLMs a clear advantage in tasks where planning and multi-step inference are required. These findings suggest that Diffusion-LLMs are better equipped to model complex, non-sequential dependencies, offering new possibilities for reasoning-based applications beyond the capabilities of conventional autoregressive models.

### 4.6 Diffusion-LLM s as Multimodal Learners

Figure: Figure 9: Visual question answering with Diffusion-LLMs. We set the iterations steps for the Diffusion-LLMs to generate final output as 50. Similar to the discussion in Sec. 4.5.2, the model first generates easy content and fills the answer at the end.
Refer to caption: x6.png

Recent advances in large language models extend beyond language processing, aiming to unify multiple modalities end to end for seamless multimodal interactions .
Given the dominance of diffusion models in generating continuous signals  and excellent language capabilities shown in this study, we believe Diffusion-LLMs contribute to a promising paradigm for developing unified multimodal models.
This motivates us to explore Diffusion-LLMs for multimodal tasks.

In particular, we investigate whether Diffusion-LLMs can tackle visual question answering (VQA).
We follow LLaVa  to conduct two-phase training upon Flan-XLM-R-XXL.
In the first stage, we freeze the language model backbone and train a projector to map vision feature extracted from pre-trained CLIP visual encoder ViT-L/14  to embeddings using the 558k subset of the LAION-CC-SBU .
We then jointly tune the LM backbone and projectors for VQA with LLaVA-v1.5-mix665k data .

**Table 4: Zero-shot exact match performance of Diffusion-LLM and AR-LLMs on the dev set of GQA .**
| Model | Exact Match |
| --- | --- |
| Diffusion-LLM (Flan-XLM-R-XXL) | 39.93 |
| AR-LLM (Flan-T5-XXL) | 44.71 |

Tab. 4 shows the zero-shot performance of our models on the dev set of GQA .
For reference, we accordingly augment a Flan-T5-XXL model with vision understanding capability with the same recipe.
The results shows meaningful performance that supports the vision understanding capability of our model, which is close to that adapted from Flan-T5.
Case study on Fig. 9 shows that the model demonstrate similar behavior to what we find in the qualitative study for reasoning tasks (Sec. 4.5.2) when generating answers for vision tasks.
In the three cases, the models answer in an easy-first order where they first generate content that can be copied from the question to build up the framework of the sentence and then fill in the key answers at the end.
In Fig. 9(C), where the key answer contains multiple entities, Diffusion-LLMs fill them simultaneously, showing the capabilities to parallelly process different vision information.

These results demonstrate Diffusion-LLMs can also understand multimodal information similar to recent autoregressive LMs.
Together with the generation capabilities, for both well-tested vision generation  as well as language generation abilities verified in our study, diffusion models shed light on an appealing unified paradigm for multimodal foundation models.

## 5 Discussions

In this work, we pioneer the study of the scalability of Diffusion-LLMs to catch up with the recent advances of LLMs and facilitate the exploration of their potential.
Our experiments verify their scalability regarding data, model sizes, and tasks.
Further, we showcase positive prospects about their reasoning capabilities such as casual-order generation and implicit planning for further exploitation.

##### Latest advancement on diffusion language models.

After the first release of our study of Diffusion-LLM, diffusion language models have attracted broad attention and plenty of closely related studies have emerged.
As such, we would like to highlight the latest progress to facilitate more upcoming advancements in this field.

- •
Foundation. To formulate a diffusion language model, also study masked language modeling-like objectives similar to ours and validate their effectiveness.
Alternatively, explores learning discrete diffusion language models by learning probability ratios as the extension of score matching and extends the flow matching formulation.
All these studies confirm the practicality of building capable diffusion language models to serve as an alternative paradigm to autoregressive language models, with specific manners evolving.
- •
Scaling validation. The scalability of diffusion language models under masked language modeling-like objectives has rapid progress.
successfully build large-scale diffusion language models by adapting from autoregressive language models, offering another promising routine to gain large diffusion language models with relatively low cost.
investigate the pretraining of diffusion language models from scratch and show a scaling law parallel to autoregressive language models, indicating similar scaling trends of the two paradigms.
further scales up the pretrained diffusion language models to 8B parameters and 2.3T pre-training tokens with up-to-date recipes, with results highlighting the competitiveness of diffusion language models to frontier open-source autoregressive models on well-recognized benchmarks for large language models and showcase a helpful chatbot built upon large diffusion language models.
Besides natural language, scales diffusion language models to empower generative modeling of proteins.
- •
Capabilities and applications.
Diffusion language models have a bidirectional receptive field and can perform refinement by nature.
For this reason, recent progress has confirmed that diffusion language models show superiority in scenarios where left-to-right generation order is suboptimal.
For instance, shows their advantage in tasks requiring implicit planning and shows diffusion language models can fix the reversal curse of autoregressive models .
Notably, the applications of diffusion language models demonstrate significant advances in scientific domains.
With diffusion language models, DPLM  build frontier foundation models for protein and DPLM-2  further investigate multimodal diffusion language models to unify generative modeling of protein sequences and structures.

We hope that our findings as well as our discussion on the latest progress can fuel the success of diffusion models in broader domains and also encourage investment into this compelling complement to autoregressive LLMs, which might push forward the boundary of techniques to pursue more advanced machine intelligence.

## Appendix A Implementation Details

### A.1 Model

Throughout this work, we mainly follow to train and sample from our diffusion language models.
Specifically, we set $\lambda_{t-1}^{(2)}=1-\frac{t-1}{T}$ in the training objective where $t$ is the current timestep and $T$ is the number of total timesteps which is 50 in our experiments. Additionally, we apply label smoothing with a factor of 0.1 when we train a model without pretraining.
During sampling, we also follow and denoise tokens with high scores in each step instead of naively sampling from the Bernoulli distributions.
We use the same cosine schedule as in to decide the number of denoised tokens in each step $k=\lfloor N\cdot\cos{\frac{\pi t}{2T}}\rfloor$, where $N$ is the sequence length.
For full details, we refer readers to the pseudocode in the original paper .
For length prediction, we feed model outputs into a one-layer transformer, apply mean pooling to the features and feed the pooled feature into an MLP classifier head.
For task-specific finetuning, we remove both input and output embeddings of the tokens that do not appear in the training set.

### A.2 Data

For IWSLT14 and WMT14 machine translation tasks, we download and preprocess data following the example scripts provided by Fairseq(^8^88[https://github.com/facebookresearch/fairseq/tree/main/examples/translation](https://github.com/facebookresearch/fairseq/tree/main/examples/translation)), and we use SacreBleu  for evaluation(^9^99The signature of sacrebleu for IWSLT14 De$\rightarrow$En is nrefs:1|case:mixed|eff:no|tok:13a| smooth:exp|version:2.3.1, and for WMT14 En$\rightarrow$De nrefs:1|case:mixed|eff:no|tok:intl| smooth:exp|version:2.3.1, respectively.).
And we download Gigaword-10K data from the repository of LGEB(^10^1010[https://github.com/CLUEbenchmark/LGEB](https://github.com/CLUEbenchmark/LGEB)).
For (M)GSM, we follow the instruction(^11^1111[https://github.com/google-research/url-nlp/tree/main/mgsm](https://github.com/google-research/url-nlp/tree/main/mgsm)) in the official repository of to process the data and prompts.
Besides, we obtain the preprocessed Flan 2021(^12^1212[https://huggingface.co/datasets/Muennighoff/flan](https://huggingface.co/datasets/Muennighoff/flan)), Flan 2022(^13^1313[https://huggingface.co/datasets/SirNeural/flan_v2](https://huggingface.co/datasets/SirNeural/flan_v2)), MMLU(^14^1414[https://huggingface.co/datasets/cais/mmlu](https://huggingface.co/datasets/cais/mmlu)),
and TydiQA(^15^1515[https://huggingface.co/datasets/khalidalt/tydiqa-goldp](https://huggingface.co/datasets/khalidalt/tydiqa-goldp)) from shared datasets on HuggingFace(^16^1616[https://huggingface.co/datasets](https://huggingface.co/datasets)).
During training with Flan 2022, we follow the recommended ratios in to sample training data from different subsets.
We follow to report the MMLU performance on the validation set and adopt the GoldP setting for TyDiQA as in .
On the few-shot settings, we randomly select demonstrations.
We will also release our code and data for better reproducibility.

### A.3 Training details

We use Adam optimizer  throughout our study.
The dropout rate is consistent with the original configuration of the models which is 0.1.
For task-specific tuning, we use 8 Nvidia A100 GPUs.
For instruction tuning, we use 8 Nvidia V100 GPUs for BASE and LARGE-sized models, 32 for XL, and 64 for XXL.
The overall batch size and other detailed hyperparameters for the two settings are in Tab. 5 and Tab. 6, respectively.

**Table 5: The training hyperparameters for task-specific finetuning.**
| Dataset | Pretrained model | Batch size (#. tokens) | Learning rate | #. training steps |
| --- | --- | --- | --- | --- |
| IWSLT14 De$\rightarrow$En | XLM-R-BASE | 32K | 5e-5 | 150,000 |
| XLM-R-LARGE | 32K | 5e-5 | 150,000 |  |
| XLM-R-XL | 32K | 5e-5 | 100,000 |  |
| XLM-R-XXL | 32K | 5e-5 | 30,000 |  |
| WMT14 En$\rightarrow$De | XLM-R-BASE | 128K | 5e-5 | 300,000 |
| XLM-R-LARGE | 128K | 5e-5 | 300,000 |  |
| XLM-R-XL | 128K | 5e-5 | 150,000 |  |
| XLM-R-XXL | 128K | 5e-5 | 100,000 |  |
| Gigaword-10K | XLM-R-BASE | 16K | 5e-5 | 30,000 |
| XLM-R-LARGE | 16K | 5e-5 | 10,000 |  |
| XLM-R-XL | 16K | 5e-5 | 5,000 |  |
| XLM-R-XXL | 16K | 5e-5 | 1,000 |  |

**Table 6: The training hyperparameters for instruction finetuning.**
| Training data | Pretrained model | Batch size (#. sequence) | Learning rate | #. training steps |
| --- | --- | --- | --- | --- |
| Flan 2021 | XLM-R-BASE | 512 | 5e-5 | 5,000 |
| XLM-R-LARGE | 512 | 5e-5 | 5,000 |  |
| XLM-R-XL | 512 | 5e-5 | 3,000 |  |
| XLM-R-XXL | 256 | 5e-5 | 1,000 |  |
| Flan 2022 | XLM-R-BASE | 512 | 1e-5 | 70,000 |
| XLM-R-LARGE | 512 | 1e-5 | 30,000 |  |
| XLM-R-XL | 1024 | 1e-5 | 17,000 |  |
| XLM-R-XXL | 2048 | 1e-5 | 4,000 |  |

## Appendix B Full Experimental Results

The experimental results for task-specific tuning and instruction tuning on Flan 2022 are in Tab. 7 and Tab. 8, respectively.

**Table 7: Full experimental results of task-specific finetuning. OL: the results are obtained with oracle length. LB: the size of length beam for length prediction.**
| Dataset (Metric) | Setting | XLM-R-BASE | XLM-R-LARGE | XLM-R-XL | XLM-R-XXL |
| --- | --- | --- | --- | --- | --- |
| IWSLT14 De$\rightarrow$En<br>(SacreBLEU) | OL | 35.78 | 38.84 | 40.11 | 40.65 |
| LB=10 | 34.10 | 37.33 | 38.54 | 38.57 |  |
| WMT14 En$\rightarrow$De<br>(SacreBLEU) | OL | 26.65 | 30.22 | 30.91 | 32.81 |
| LB=10 | 26.72 | 29.04 | 30.23 | 30.34 |  |
| Gigaword-10K<br>(Rouge-L) | OL | 28.83 | 31.33 | 31.72 | 32.57 |
| LB=10 | 27.52 | 30.11 | 31.42 | 31.54 |  |

**Table 8: Full experimental results of instruction tuning on Flan 2022. OL: the results are obtained with oracle length. LB: the size of length beam for length prediction.**
| Dataset (Metric) | Setting | XLM-R-BASE | XLM-R-LARGE | XLM-R-XL | XLM-R-XXL |
| --- | --- | --- | --- | --- | --- |
| IWSLT14 De$\rightarrow$En<br>(SacreBLEU) | 0-shot (OL) | 21.26 | 25.24 | 28.13 | 29.59 |
| 2-shot (OL) | 20.97 | 25.70 | 29.19 | 30.31 |  |
| 0-shot (LB=3) | 17.76 | 25.12 | 26.42 | 30.90 |  |
| 2-shot (LB=3) | 15.91 | 23.49 | 27.29 | 31.04 |  |
| MMLU<br>(Accuracy%) | 0-shot | 31.28 | 32.79 | 40.17 | 42.13 |
| 2-shot | 28.74 | 32.72 | 38.08 | 42.06 |  |
| TyDiQA<br>(Rouge) | 0-shot (OL) | 23.42 | 62.28 | 83.39 | 84.76 |
| 1-shot (OL) | 23.46 | 66.86 | 86.12 | 84.36 |  |
| 0-shot (LB=3) | 13.54 | 62.28 | 83.39 | 84.76 |  |
| 1-shot (LB=3) | 12.63 | 44.49 | 55.78 | 84.36 |  |
| MGSM (De)<br>(Accuracy%) | 0-shot | 0.9 | 2.8 | 1.6 | 3.6 |
| 3-shot | 1.6 | 2.8 | 5.2 | 4.4 |  |
| GSM8K<br>(Accuracy%) | 0-shot | 3.6 | 3.2 | 5.2 | 4.4 |
| 3-shot | 3.2 | 2.0 | 3.6 | 5.6 |  |

Figure: Figure 10: Few-shot performance of Flan-XLM-R and Flan-T5 models. “OL” means the results are obtained with oracle length, while “LB” means the number of length beams to sample the target with length prediction. The model sizes refer to the number of non-embedding parameters.
Refer to caption: extracted/6217187/figs/few_shot.png