---
arxiv_id: "2410.21035"
title: "Beyond Autoregression: Fast LLMs via Self-Distillation Through Time"
year: 2025
source: arxiv2md
---

## Abstract

Abstract Autoregressive (AR) Large Language Models (LLMs) have demonstrated significant success across numerous tasks. However, the AR modeling paradigm presents certain limitations; for instance, contemporary autoregressive LLMs are trained to generate one token at a time, which can result in noticeable latency. Recent advances have indicated that search and repeated sampling can enhance performance in various applications, such as theorem proving, code generation, and alignment, by utilizing greater computational resources during inference. In this study, we demonstrate that diffusion language models are capable of generating at least 32 tokens simultaneously, while exceeding the performance of AR models in text quality and on the LAMBADA natural language understanding benchmark. This outcome is achieved through a novel distillation method for discrete diffusion models, which reduces the number of inference steps by a factor of 32-64. Practically, at the 1.3B parameters scale, diffusion models, even without caching, can generate tokens at a rate that is up to 8 times faster than AR models employing KV-caching, and we anticipate further improvements with the inclusion of caching. Moreover, we demonstrate the efficacy of our approach for diffusion language models with up to 860M parameters.

## 1 Introduction

Figure: Figure 1: Perplexity versus latency. The diffusion models (169M) use 16, 32, 64, 128 and 256 decoding step.
Refer to caption: x1.png

In recent years, autoregressive (AR) large language models (LLM) have exceeded expectations . Importantly, many breakthroughs in coding , mathematics, and reasoning were achieved based on decoding large amounts of completions from a base LLM.

Importantly, the benefits of repeated sampling can be so significant that it is often more efficient to use a smaller, faster model rather than a larger, slower one. More generally, one can improve the performance of a fixed model by scaling up computational resources at inference time , a phenomenon that was previously observed for games . Hence, when tackling reasoning tasks, a major bottleneck is the latency of the model. In this work, we improve the decoding speed of LLMs by moving away from AR modeling. We build on recent breakthroughs in discrete diffusion . Our approach can generate text up to 8 times faster than AR models that use KV caching .
Diffusion models are typically trained to maximize the evidence lower bound (ELBO), which does not consider the desired number of inference steps. Hence, vanilla diffusion models typically require thousands of decoding steps. Fortunately, it is possible to drastically reduce the inference costs of continuous diffusion models via distillation . Continuous distillation methods rely on deterministic mappings from noise to data, such as DDIM . The deterministic mappings can be efficiently learned by a student diffusion model to sample in fewer steps. We hypothesize that such deterministic map cannot exist for the diffusion language models studied in this work. Indeed, those models always initialize the denoising process with a sequence of masked token, hence a deterministic algorithm can only generate a single sample. As such, we devise a distillation method that does not does depend on deterministic maps. This is a significant finding because faster decoding mechanisms allow exploring a larger search space in applications that require search, planning, and reranking. In summary, our core contributions are as follows:

- •
We introduce Self-Distillation Through Time (SDTT), which allows generating at least 32 tokens at a time, while achieving better perplexity than GPT-2 with nucleus sampling for conditional and unconditional generation. Unlike many distillation methods for continuous diffusion models, SDTT does not rely on deterministic mappings such as DDIM . SDTT is very simple and easy to implement.
- •
We show that SDTT can generate tokens up to 8 times faster than AR models that use KV caching, for models with 1.3B parameters, in 16 decoding steps. Importantly, the discrete diffusion model does not rely on activation caching, suggesting that there is potential for even greater efficiency gains. The latency gains for smaller models are even greater.
- •
We demonstrate the effectiveness of SDTT for models with up to 860M parameters. To the best of our knowledge, this represents the largest publicly available discrete diffusion language model.
- •
We evaluate the distilled students on LAMBADA and 6 multiple-choice questions benchmarks from . We find that SDTT preserves the natural language understanding performance of the teacher.

## 2 Background

Figure: (a) Accuracy of the correct last word decoded from our model. Distillation with KLD loss leads the student model to outperform the teacher in terms of accuracy on LAMBADA.
Refer to caption: x2.png

### 2.1 Masked diffusion language modeling

Figure: (a) The distillation targets are the log probabilities that lead to a token being denoised, concatenated with log probabilities of the last step for tokens that remain masked.
Refer to caption: x4.png

We follow the notation of to introduce masked diffusion language modeling (MDLM). Language modeling can be framed as the sequential prediction task of discrete tokens ($x_{i}$) coming from a vocabulary $\mathcal{X}=\mathbb{Z}^{<N}=\left\{0,~{}...,~{}N-1\right\}$ that can take $N$ possible discrete values. A language model would predict sequences of length $L$, which can be defined as the sequences of $x_{i}$’s originating from $\mathcal{X}^{L}=\left\{\mathbf{x}^{(i)}=(x^{(i)}_{0},~{}\dots,~{}x^{(i)}_{L-1}
)\right\}_{i\in\mathbb{Z}^{<K}}$. Let $\mathcal{D}:=\left\{\mathbf{x}^{(0)},~{}\dots,~{}\mathbf{x}^{(K-1)}:\mathbf{x}
^{(i)}\in\mathcal{X}^{L}\right\}$ denote the training set. The goal of language modeling is to sample from the unknown distribution $p_{0}:\mathcal{X}^{L}\rightarrow[0,1]$ that generated the samples in $\mathcal{D}$.

Similarly to continuous diffusion, we sample from an approximation of $p_{0}$ by learning to denoise corrupted examples. One can sample from the model through ancestral sampling, starting from a stationary distribution. The stationary distribution of is such that all tokens of the sentence are replaced with a special MASK token like the MASK token used for pre-training BERT models. However, a key difference between BERT and MDLM is that MDLM is trained on sequences with varying levels of corruption, while BERT uses a fixed ratio.

#### Discrete absorbing diffusion process

MDLM defines a forward process to corrupt data and a backward process to learn to recover data. MDLM uses a continuous-time formulation, with the data distribution denoted as $p_{0}$ and the stationary noise distribution as $p_{1}=\bm{\pi}$. The forward process linearly interpolates between the one-hot distribution defined by the original document $\mathbf{x}$ and the stationary distribution $\bm{\pi}$, which places all mass on the MASK token. Mathematically,

$$ $q(\mathbf{z}_{t}|\mathbf{x}):=\text{Cat}(\mathbf{z}_{t};\alpha_{t}\mathbf{x}+( 1-\alpha_{t})\bm{\pi}),$ (1) $$

where the noise injection schedule is defined by $\alpha_{t}$, for $t\in[0,1]$. The constraints on $\alpha_{t}$ are that $\alpha_{t}\in[0,1]$, $\alpha_{t}$ should be a strictly decreasing function of $t$, and $\alpha_{0}\approx 1,\alpha_{1}\approx 0$. The forward process is called absorbing because once a token is assigned to a MASK token, it cannot be reverted to a real token.

We can derive the analytical form of the reverse process $q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x})$, with $t>s$ and $\alpha_{t|s}=\frac{\alpha_{t}}{\alpha_{s}}$ as

$$ $q({\mathbf{z}}_{s}|{\mathbf{z}}_{t},{\mathbf{x}})=\text{Cat}\left({\mathbf{z}} _{s};\frac{[\alpha_{t|s}{\mathbf{z}}_{t}+(1-\alpha_{t|s}){\bm{1}}\bm{\pi}^{ \top}{\mathbf{z}}_{t}]\odot[\alpha_{s}{\mathbf{x}}+(1-\alpha_{s})\bm{\pi}]}{ \alpha_{t}{\mathbf{z}}_{t}^{\top}{\mathbf{x}}+(1-\alpha_{t}){\mathbf{z}}_{t}^{ \top}\bm{\pi}}\right).$ (2) $$

#### Objective and parameterization

To generate new samples, we can simulate the reverse process from eq. 2. Since the ground-truth sample ${\mathbf{x}}$ is unknown, learn an approximation ${\mathbf{x}}_{\theta}$ using a neural network with parameters $\theta$. then use ${\mathbf{x}}_{\theta}$ instead of ${\mathbf{x}}$ to simulate the reverse process. The sampling distribution is denoted as ${p_{\theta}({\mathbf{z}}_{s}|{\mathbf{z}}_{t}):=q({\mathbf{z}}_{s}|{\mathbf{z}
}_{t},{\mathbf{x}}_{\theta}({\mathbf{z}}_{t},t)})$. optimize $\theta$ using a continuous version of the negative evidence lower bound (NELBO) of . Previous research has shown that continuous-time objectives optimize the data likelihood better . Due to the definition of the absorbing diffusion process, the NELBO simplifies to a weighted cross-entropy loss between the ground-truth ${\mathbf{x}}$ and the model predictions ${\mathbf{x}}_{\theta}$:

$$ ${\mathcal{L}^{\infty}_{\text{NELBO}}}=\mathbb{E}_{q}\int_{t=0}^{t=1}\frac{ \alpha^{\prime}_{t}}{1-\alpha_{t}}\log\langle\mathbf{x}_{\theta}({\mathbf{z}}_ {t},t),{\mathbf{x}}\rangle{\text{d}}t.$ (3) $$

To derive eq. 3, impose two properties on $p_{\theta}({\mathbf{z}}_{s}|{\mathbf{z}}_{t})$. First, denoised tokens are never re-masked during sampling. Practically, this is achieved by manipulating the output of the neural network ${\mathbf{x}}_{\theta}({\mathbf{z}}_{t},t)$ to ensure that no probability mass is assigned to the MASK token. Secondly, already-denoised tokens are carried-over to the next sampling step. showed that both constraints lead to improved likelihood.

### 2.2 Knowledge Distillation

Figure: Algorithm 1 Computing the Self-Distillation Through Time targets $\tilde{\mathbf{x}}_{\theta}^{\text{teacher}}({\mathbf{z}}_{t},t,\nicefrac{{m}} {{k}})$

Knowledge distillation is a technique where a student neural network is trained to imitate the predictions of a more complex teacher model. One of the main advantages of distillation is the ability to reduce the inference cost associated with sampling from large LLMs while surpassing the performance of smaller models trained without distillation . The most relevant to our work are the distillation methods that match the predictions of the teacher and the student using a divergence measure $\delta$:

$$ $\mathbb{E}_{{\mathbf{x}}\sim\mathcal{D}}\left[\delta(\mu_{s}({\mathbf{x}}_{t}| {\mathbf{x}}_{<t});\mu_{t}({\mathbf{x}}_{t}|{\mathbf{x}}_{<t}))\right],$ (4) $$

Where $\mu_{s},\mu_{t}$ are the AR distributions of the student and teacher, respectively, and $\mathcal{D}$ represent the training dataset. Common divergence measures include $f$-divergences such as the Kullback-Leibler divergence (KLD) or the total variation distance (TVD).

## 3 Method

### 3.1 Self-Distillation Through Time

As explained in fig. 3, discrete diffusion language models optimize the NELBO over the training examples. Fewer decoding steps typically lead to lower sample quality because the approximation of the reverse process is less accurate, as visible in the teacher curve in fig. 4.

To address the issue of low sample quality with fewer decoding steps, we propose Self-Distillation Through Time (SDTT). SDTT fine-tunes a pre-trained MDLM to allow decoding with significantly fewer steps. Interestingly, our final model decodes samples with lower generative perplexity in 32 steps than the teacher would with 1024 forward passes. In short, SDTT improves the sampling speed by distilling the inference time computation to sample multiple steps into the student.

Let $p_{\theta}^{(m)}$ be the distribution of samples generated with $m$ steps, using a denoiser with parameters $\theta$. SDTT trains a denoiser with parameters $\nu$ to minimize a divergence $d$ between $p_{\theta}^{(m)}$ and $p_{\nu}^{(k)}$. Here $k<m$, and $k$ divides $m$ (e.g., $m=1024$ and $k=512$):

$$ $\min_{\nu}~{}d\left(p_{\nu}^{(k)}||p_{\theta}^{(m)}\right).$ (5) $$

Since ${\mathbf{x}}_{\theta}$ and ${\mathbf{x}}_{\nu}$ are the only learnable elements of the sampling process, they completely determine the sampling distributions $p_{\theta}^{(m)}$ and $p_{\nu}^{(k)}$. As such, training ${\mathbf{x}}_{\nu}$ to match the predictions of ${\mathbf{x}}_{\theta}$ with fewer steps minimizes eq. 5. We now present a method for generating targets $\tilde{\mathbf{x}}_{\theta}^{\text{teacher}}({\mathbf{z}}_{t},t,\nicefrac{{m}}
{{k}})$ to train ${\mathbf{x}}_{\nu}$. Mathematically, we optimize the following objective:

$$ $\min_{\nu}~{}\mathbb{E}_{{\mathbf{z}}_{0}\sim\mathcal{D},{\mathbf{z}}_{t}\sim q _{t}({\mathbf{z}}_{t}|{\mathbf{z}}_{0})}\left[\delta({\mathbf{x}}_{\nu}({ \mathbf{z}}_{t},t)||\tilde{\mathbf{x}}_{\theta}^{\text{teacher}}({\mathbf{z}}_ {t},t,\nicefrac{{m}}{{k}}))\right],$ (6) $$

where $\delta$ a divergence measure between the student and the teacher targets $\tilde{\mathbf{x}}_{\theta}^{\text{teacher}}({\mathbf{z}}_{t},t,\nicefrac{{m}}
{{k}}))$. We consider the Kullback-Leibler divergence (KLD), Total Variation Distance (TVD), and Mean-Squared Error (MSE). See appendix B for details on those divergence measures.

Figure: Algorithm 2 One training round of Self-Distillation Through Time

#### Generating the Teacher Targets

Following the terminology of knowledge distillation, we call the denoiser ${\mathbf{x}}_{\theta}$ used for many steps decoding as the teacher and the denoiser ${\mathbf{x}}_{\nu}$ used for a few steps decoding as the student. To train ${\mathbf{x}}_{\nu}$ to match the predictions of ${\mathbf{x}}_{\theta}$, we sample from the teacher for $\nicefrac{{m}}{{k}}$ steps. Whenever a MASK token is denoised, we collect the log probabilities predicted by the teacher for this MASK token. These log-probabilities become the distillation targets $\tilde{\mathbf{x}}_{\theta}^{\text{teacher}}({\mathbf{z}}_{t},t,\nicefrac{{m}}
{{k}})$. Algorithm 1 outlines this process and fig. 3(a) presents it visually. While fig. 3(a) shows how to distill two decoding steps in one, the procedure can be extended to larger values of $\nicefrac{{m}}{{k}}$. The complete SDTT training loop is presented in algorithm 2.

#### Iterated SDTT

SDTT reduces the number of decoding steps by a factor $\nicefrac{{m}}{{k}}$. If we want to reduce the number of decoding steps further, we can apply SDTT with $k^{\prime}<k$, or alternatively apply SDTT $n$ times, using the newly distilled student as teacher for the next round, which we refer to as iterated SDTT. Instead of directly optimizing the divergence in eq. 5, we introduce $n$ intermediate distributions $p_{\nu_{i}}^{k_{i}}$ such that $\nicefrac{{m}}{{k_{i}}}$ is an increasing sequence as a function of $i$. In practice, we choose $m=2^{10}$ and $k_{i}=2^{10-i}$ with $0\leq i\leq 7$ and sequentially minimize the objective

$$ ${\min_{\nu}~{}d\left(p_{\nu_{j}+1}^{(k_{j+1})}||p_{\nu_{j}}^{(k_{j})}\right),}$ (7) $$

for $0\leq j<7$, where $\nu_{j}$ denotes the parameters of the $j$-th denoiser, with $\nu_{0}=\theta$ (teacher). If the minimization procedure was perfect, minimizing eq. 5 or eq. 7 should result in the same solution. However in practice, we observe that it is easier to minimize eq. 7 sequentially for increasing values of $i$, in a progressive fashion, similar to .

As an alternative to iterated SDTT, we tried using a single model and slowly growing the step size used to generate $\tilde{\mathbf{x}}_{\theta}^{\text{teacher}}({\mathbf{z}}_{t},t,\nicefrac{{m}}
{{k}})$. Unfortunately, this approach was unstable and the loss diverged after 30-50 steps, irrespective of how small the sampling step size was. Similar behavior was observed by .

## 4 Experiments

We distill MDLMs on the OpenWebText dataset as it was used to train recent discrete diffusion language models . We use the Adam optimizer with a learning rate of $6e-5$, a batch size of 128 and no weight decay. We linearly increase the learning rate for 500 training steps and keep it constant afterwards. As a base model, we reuse the checkpoint released by . See appendix C for more details.

In section 4.1, we evaluate 3 distillation divergences and show that
iterated SDTT can reduce the number of sampling steps by a factor 16-32. In section 4.2, we ablate on the importance of hyperparameters, including the duration of each round of iterated SDTT and the number of sampling steps to generate the targets $\tilde{\mathbf{x}}_{\theta}^{\text{teacher}}({\mathbf{z}}_{t},t,\nicefrac{{m}}
{{k}})$. In section 4.3, we scale SDTT to models with of up to 860M parameters. Finally, in section 4.4, we compare the latency of SDTT against autoregressive models that use KV caching.

#### Generative perplexity

Following prior work , we use a larger model to compute the generative perplexity of unconditional and conditional samples. We evaluate the smallest students using GPT-2 (large) . In the scaling experiments, we use Llama3 8B , since we compare models with up to 860M parameters. As noted by , the generative perplexity is sensitive to the floating-point precision. In this section, we sample using bfloat16, and report results using float64 in table 1. The conclusion are similar.

#### MAUVE

We evaluate conditional generation using the MAUVE score . MAUVE measures how well a model follows a prompt by comparing multiple generations with a reference continuation. We use the first 1024 samples with at least 1024 tokens from the WebText dataset , take the first 50 tokens as a prompt, and generate 50 tokens of continuation. For each prompt, we generate 5 continuations, as done in .

#### Sample diversity

Post-training can drastically reduce the diversity of language models . Hence, we measure the diversity of samples using the self-BLEU score with the same completions used to compute MAUVE.

#### Downstream performance

We measure the downstream performance using the LAMBADA dataset , as well as 6 multiple-choice question (MCQ) tasks from . On LAMBADA, we report an upper bound on the perplexity, computed using the ELBO (3). We also report the suffix accuracy by masking all tokens of the last word and predicting all of them in a single forward pass, using the argmax of the predictions. The diffusion model is correct only if all the masked tokens are decoded correctly in a single decoding step. The 6 other benchmarks from evaluate the MCQ accuracy.

Figure: (a) Diversity of conditional generation (small scale). We measure the trade-off between quality and diversity using self-BLEU . Deterministic sampling yields a score of 1. The diversity minimally decreases after distillation.
Refer to caption: x6.png

### 4.1 Ablation on the training divergence

SDTT requires choosing a divergence $\delta$ and we study the Mean-Squared Error (MSE), Total Variation Distance (TVD) and (reverse) Kullback-Leibler Divergence (KLD). We apply iterated SDTT for 7 rounds of $10k$ training iterations and generate $\tilde{\mathbf{x}}_{\theta}^{\text{teacher}}({\mathbf{z}}_{t},t,\nicefrac{{m}}
{{k}})$ with $2$ sampling steps from the teacher (algorithm 1). We use an exponential moving average (EMA) of the weights with a decay of $0.9999$ that we do not reset between rounds.

Figure 2 shows that students distilled with the KLD clearly outperform students trained using the MSE and TVD on LAMBADA. The LAMBADA accuracy of students tuned with the KLD slightly improves over the teacher, while the perplexity remains better or matches the AR baselines for all but the last round of SDTT. The improved accuracy on LAMBADA suggests that the model is better at predicting multiple tokens in parallel after distillation with SDTT, since we evaluates the accuracy by decoding all tokens of the last word simultaneously.

Figure 5 shows that the KLD seem to outperform the MSE and TVD objectives on MAUVE. Since we generate sequences of 100 tokens only for MAUVE, following , we sample with at most 128 steps, and use samples generated with 128 sampling steps from the teacher as a baseline. Note that as observed by , discrete diffusion models typically achieve slightly lower MAUVE scores than AR models. Nonetheless, distillation with the KLD objective improves the MAUVE score of the students. Similarly fig. 18 shows that continuations from the student distilled with the KLD reaches the lowest perplexity and match GPT-2 with nucleus sampling in 32 forward passes.

In table 1, we compare the downstream performance on the tasks of before and after distillation. We observe that SDTT minimally affects the results, and that student distilled with the KLD objective reaches higher accuracies than other students in all but one task

Figure 4(a) measures the diversity of samples using the self-BLEU score , for the students distilled with the KLD objective. See table 1 for results with the MSE and TVD. We find that SDTT minimally decreases the diversity. Compared to distilling autoregressive models , SDTT minimally reduces the diversity. For reference, routinely observes an increase of 15 in self-BLEU while we observe a change of at most 2 for the KLD student. See table 1 for more results and details on the self-BLEU score.

Figure 6 shows that students distilled with KLD have higher unconditional generative perplexity than those distilled with the MSE. However, KLD is the only objective that preserves performance in the LAMBADA data set while still significantly reducing the generative perplexity compared to the teacher. Therefore, in the remainder of this work, we focus on the KLD.

### 4.2 Additional ablations

Figure: Figure 5: MAUVE performance of the student after each round of SDTT. The teacher performance is computed using samples generated with 128 decoding steps.
Refer to caption: x8.png

#### Number of steps in each SDTT round

In section 4.1, each round of SDTT consists of $10k$ training iterations. Since the magnitude of the distillation loss does not reliably indicate convergence, we experiment with shorter rounds. We find that reducing the number of training iterations to $5k$ or $2.5k$ negatively impacted conditional generation performance, as shown in fig. 7. However, shorter rounds slightly improved the final generative perplexity (fig. 8) and resulted in marginally better LAMBADA perplexity (fig. 10). Since SDTT does not directly optimize the ELBO, an increase in perplexity is expected. Interestingly, the LAMBADA accuracy remains unchanged with shorter rounds.

#### Number of sampling steps to generate the targets

In section 4.1, the targets $\tilde{\mathbf{x}}_{\theta}^{\text{teacher}}({\mathbf{z}}_{t},t,\nicefrac{{m}}
{{k}})$ are generated using 2 sampling steps from the teacher. We explore distilling a larger number of sampling steps at once (4 or 8), since using more rounds of SDTT may induce more error accumulation in approximating the original teacher. Figure 13 shows that distilling more than two steps at a time is difficult and results in weaker results on LAMBADA. This suggests that the higher stochasticity of the targets generated with four or eight steps makes the task too difficult for the student.

#### Generating targets with the analytical sampler

observe that using an analytical sampler results in higher quality samples compared to ancestral sampling. However, when generating targets $\tilde{\mathbf{x}}_{\theta}^{\text{teacher}}({\mathbf{z}}_{t},t,\nicefrac{{m}}
{{k}})$ with analytical sampling, we observed minimal difference with ancestral sampling, as shown in fig. 11 and 12.

#### Resetting the optimizer and Exponential Moving Average between rounds

Using an Exponential Moving Average (EMA) of the weights is known to improve the quality of samples from diffusion models . However, when applying SDTT for multiple rounds, it is unclear whether the EMA or current weights should be used as the teacher for successive rounds. Additionally, it could be favorable to reset the optimizer state between rounds as we grow the decoding step size. We experiment with two approaches: either resetting the optimizer state only, or resetting both the EMA and optimizer state. Figure 14 shows the generative perplexity when resetting the optimizer state and using the EMA as the teacher instead of the current weights, while fig. 15 presents the corresponding results for MAUVE. When using the EMA as teacher, since we accumulate updates in the EMA over 10k training iterations only, we use a slightly lower decay rate of 0.999. We find that using the EMA of the weights as the teacher may slightly improve performance.

### 4.3 Scaling SDTT to 860M parameters

Figure: (a) KLD vs MSE
Refer to caption: x9.png

We apply SDTT to larger discrete diffusion models with up to 860M parameters. In this experiment, we train the models from scratch for 400k steps with a batch size of 512, a context length of 1024 and the Adam optimizer. We reuse the training configuration of and scale the models to larger sizes. We train 3 model sizes, small (169M), medium (424M) and large (863M). Details of the model architecture for each scale are shown in table 2. As for the other experiments, the models are diffusion transformers and we use an EMA with a decay of 0.9999. Although the results in section 4.2 suggest that short distillation rounds might be sufficient, it is unclear whether this result also holds on larger scales. Therefore, we use $10k$ steps per round of SDTT. For simplicity, we generate targets using 2 teacher ancestral decoding steps and do not reset the optimizer state or EMA between rounds.

Since we train larger models, we evaluate the generative perplexity using Llama3 8B . The generative perplexity over the 3 model sizes is shown in fig. 4(b). Interestingly, the smaller diffusion model (169M) sampled from with 64 steps or more after distillation achieves better generative perplexity than the largest model (863M) when sampling with 1024 steps. In fig. 16, we show that the MAUVE performance also improves after distillation for the medium and larger model. Finally, in fig. 17, we see that the LAMBADA accuracy improves after distillation, similar as in the smaller scale, when using the KLD objective.

### 4.4 Latency with SDTT

While SDTT allows sampling from discrete diffusion models with 32-64 times less decoding steps, a quantity of interest to practitioners is the actual latency of text generation. Indeed, while the reduction in the number of sampling steps is large, since discrete diffusion uses a non-causal architecture, we cannot use KV caching . KV caching improves the inference performance drastically for AR models, hence we compare the latency of SDTT with GPT-2 with KV caching. We successfully reproduce the results of , which showed a 4x improvement when sampling with 32 steps, and measure an 8x improvement with 16 decoding steps. We compute the latency using untrained models with around 1.3B parameters, using the same hyperparameters as . We use a batch size of 8 and time the sampling 10 times after one warm-up step on a single A100 GPU with 80 GiB of RAM. All models use FlashAttention . See Table 1 for additional experiments on the latency.

## 5 Related Work

#### Diffusion Models

Diffusion models are the basis of many state-of-the-art text-to-image models . After their introduction by , showed that diffusion models can achieve FID scores comparable to GANs .

#### Discrete Diffusion & Diffusion Language Models

Prior to , introduced a novel discrete diffusion language model called SEDD. When decoding with a large number of steps, SEDD can match or surpass GPT-2 in unconditional text generation. The model of learn a discrete generalization of the score of continuous diffusion models . developed the continuous-time discrete diffusion framework. extended Bernoulli diffusion to categorical distributions, and generalized the work of to more general corruption processes, including absorbing diffusion. develop a family of re-parameterized discrete diffusion models to enhance the training and decoding efficiency. In parallel, several studies have explored continuous diffusion for language modeling . Despite recent breakthroughs, diffusion language models still have some drawbacks . adapt Chain-of-Thought reasoning to diffusion models.

#### Distillation of Continuous Diffusion models

Distilling continuous diffusion models is a well-studied area. For a comprehensive survey, see . Many distillation methods rely on Denoising Diffusion Implicit Models (DDIM) , which showed that diffusion models can be sampled deterministically. unroll trajectories sampled with DDIM and train a student to map noise directly to images. pre-compute a dataset of noise-image pairs. Close to our work, teaches the student to match multiple sampling steps of the teacher, given corrupted training examples. However, unlike , we cannot rely on the existence of a deterministic map via DDIM. Consistency distillation fine-tunes a pre-trained diffusion model to predict the final sample from intermediate points of the sampling trajectory, which enable faster sampling. distills a pre-trained diffusion model into single-step generator through a novel loss, Integral Kullback-Leibler divergence. SD-XL Turbo uses an adversarial formulation to sample with 1-4 steps from a latent diffusion model .

#### Masked & Non Auto-Regressive Language Modeling

BERT introduced the masked language modeling objective. While BERT focuses on representation learning, discrete diffusion language models are generative. XLNet uses a generalized AR pretrtaining method to model the text distribution over all permutations of the training sequences, outperforming BERT on downstream tasks. adopt a similar objective to XLNet for generative modeling instead of natural language understanding.

## 6 Discussion

In this work, we introduce Self-Distillation Through Time (SDTT), a distillation method for discrete diffusion models. Recent works suggest that discrete diffusion models can match or outperform autoregressive models in text quality. However, those models require more inference resources than AR models to achieve good performance, because of the non-causal architecture of the neural network that prevents the use of KV caching. We show that SDTT can reduce the number of decoding steps while retaining performance. Our final student is up to 8x faster than AR models that use KV caching and we demonstrate that SDTT is applicable to larger models as well. In future work, we plan to evaluate SDTT on tasks that involve generating a large number of completions from a base language model.

## 7 Reproducibility Statement

We provide details on model architectures, hyperparameters, and provide pseudocode for our algorithm. We built on top of the open source model of , which makes it relatively easy for researchers to reproduce our results. Additionally, upon de-anonymization, we will release our code and artifacts.

## 8 Ethics Statement

Overall, language models are dual-use technologies, and thus, they can have unethical uses, such as fake content generation, and they can suffer from bias if applied to data sets that are not carefully curated. This paper focuses specifically on speeding up discrete diffusion language models at test time to reduce their computational demands; we do not have specific concerns with regard to this contribution.

## 9 Acknowledgements

We thank the ICLR’25 reviewers, area chairs, and organizers for their valuable feedback and support. We acknowledge the SCITAS team at EPFL for providing access to their beta cluster, and Karin Gétaz for her administrative assistance. This work was supported by the Swiss AI Initiative through a grant from the Swiss National Supercomputing Centre (CSCS), project ID a10 on Alps. Special thanks to Skander Moalla for providing a reproducible compute infrastructure code template.

## Appendix A Additional ablation results

**Table 1: Downstream evaluation results. We report the accuracy of GPT-2, the teacher and students after 7 rounds of SDTT. Distillation seems to minimally affect the downstream performance.**
| Task | GPT-2 | Teacher | KLD student | MSE student | TVD student |
| --- | --- | --- | --- | --- | --- |
| ARC-Easy | 43.81 | 40.91 | 40.57 | 40.45 | 40.32 |
| ARC-Challenge | 19.03 | 21.08 | 20.73 | 19.28 | 20.05 |
| HellaSwag | 28.92 | 30.50 | 29.65 | 29.10 | 29.18 |
| MathQA | 21.21 | 21.78 | 21.47 | 22.28 | 21.84 |
| PIQA | 62.89 | 59.74 | 59.85 | 58.11 | 58.16 |
| WinoGrande | 51.62 | 50.91 | 50.75 | 49.57 | 50.36 |

In this section, we show additional plots on the ablations we conducted. Because the KLD was best in retaining the performance on the LAMBADA dataset, we used it in most the ablations. Hence, unless specified, the following experiments distill using the KLD.

#### Generative perplexity and precision of the floating-point operations.

observed that low-precision sampling can be problematic in masked diffusion models, leading to reduced diversity and potentially misleading generative perplexity scores. As such, in addition to bfloat16, we try distilling (i.e. computing the backward KL) and sampling using 64 bits precision. Overall, it does lead to a higher generative perplexity, however the conclusions remain similar, as the final student achieves lower generative perplexity than GPT-2 with nucleus sampling (p=0.95) in 64 sampling steps, as shown in fig. 9.

#### Ablations on the number of steps per round of SDTT

In fig. 7 we show the MAUVE performance. In fig. 8 we show the generative perplexity, and in fig. 10, we show results on LAMBADA.

#### Ablation on the analytic sampler

In fig. 11 we show results on LAMBADA, and on fig. 12 the MAUVE score.

#### Distilling more than 2 steps at once

In fig. 13, we show the generative perplexity.

#### Ablation on the optimizer state and exponential moving average of the weights

In fig. 14 we show the generative perplexity when resetting the EMA and optimizer state. In fig. 14, we compare the generative perplexity when resetting the optimizer state only, and when resetting the EMA state. Finally, in fig. 15, we show the MAUVE score.

#### Plots for scaled SDTT

In fig. 16 we show the MAUVE score and in fig. 17, we show results on LAMBADA.

#### Conditional perplexity with TVD

In fig. 18(c), we show the conditional perplexity (prompt excluded) on the small scale, for models trained for 1M steps. Empirically, the TVD performs worse than the KLD and MSE.

#### Measuring the diversity

We evaluate the generation diversity using the self-BLEU score . The self-BLEU score averages the BLEU score between one completion and the others. Therefore, when the sampling algorithm is deterministic, the self-BLEU score is 1, and a lower self-BLEU score denotes a more diverse set of samples. Formally, let $X=\{x_{1},...,x_{n}\}$ be conditionally-generated sequences, starting with the same prompt. The self-BLEU score can be computed as

$$ $\text{self-BLEU :=}\frac{1}{n}\sum_{i}\text{BLEU}(x_{i},X\setminus\{x_{i}\}).$ (8) $$

We compute the self-BLEU score using 1000 prompts, as for MAUVE, and generate 5 continuations per prompt. Figure 4(a), fig. 19(a) and fig. 19(b) show the self-bleu score after distillation with the KLD, MSE and TVD objectives. Each objective only minimally decrease the diversity after distillation. Compared to on-policy distillation of autoregressive models , the decrease is marginal, as observe an increase of self-BLEU of the order of 10-20, demonstrating a more significant decrease in diversity.

#### Decoding latency

In addition to the results on the 1.3B scale, we report the latency for models with 169M, 424M, 863M, 3B and 8B parameters. We compute the latency with a batch size of 8 and 4. Figure 20 shows the latency with a batch size of 8 and fig. 21 using a batch size of 4. Figure 22 shows the trade-off between latency and perplexity. We measure the latency at the small model size and compare GPT-2 with the final students after 7 rounds of distillation.

#### Additional downstream evaluation results

We show the performance of GPT-2, the teacher and distilled students on additional downstream benchmarks from in table 1.

Figure: (a) 10k vs 5k iter/round.
Refer to caption: x11.png

Figure: (a) 10k vs 5k iter/round.
Refer to caption: x13.png

Figure: Figure 9: Generative perplexity when distilling and sampling with 64 bits precision. Namely, we sample from the teacher and students in float64, and compute the backward KL in float64.
Refer to caption: x15.png

Figure: Figure 10: Performance on LAMBADA when distilling with fewer steps per distillation round.
Refer to caption: x16.png

Figure: (a) Generative perplexity.
Refer to caption: x17.png

Figure: Figure 12: MAUVE performance when distilling using the ancestral sampler used by . We find no clear benefit over the ancestral sampler.
Refer to caption: x19.png

Figure: (a) 4 steps.
Refer to caption: x20.png

Figure: (a) Resetting the optimizer state between rounds.
Refer to caption: x23.png

Figure: (a) Resetting the optimizer state only.
Refer to caption: x25.png

Figure: Figure 16: MAUVE performance of medium and large models pretrained for 400k steps. This experiment supports our claims that SDTT helps the final models to approach the performance of the teacher with less sampling steps.
Refer to caption: x27.png

Figure: (a) Accuracy.
Refer to caption: x28.png

Figure: (a) Perplexity of completions when distilling with the KLD objective.
Refer to caption: x30.png

Figure: (a) Distillation with the MSE loss.
Refer to caption: x33.png

Figure: (a) Small (169M.
Refer to caption: x35.png

Figure: (a) Small (169M.
Refer to caption: x41.png

Figure: Figure 22: Perplexity vs wall-time latency (in seconds) for small models. We use 16, 32, 64, 128 ans 256 decoding step for the diffusion models.
Refer to caption: x47.png

## Appendix B Additional details on the divergence measures

In this work, we teach the student to match the teacher targets $\tilde{\mathbf{x}}_{\theta}^{\text{teacher}}({\mathbf{z}}_{t},t,\nicefrac{{m}}
{{k}})$ generated by algorithm 1. We penalize the student deviating from the targets using one of three divergence measure: the Kullback-Leibler Divergence (KLD), the Total Variation Distance (TVD), and the Mean-Squared Error (MSE). We now describe each of them.

**Table 2: Hyperparameters of the diffusion models at different scales. All models use RoPE positional encoding .**
| Model size | small | medium | large | 1.3B | 3B | 8B |
| --- | --- | --- | --- | --- | --- | --- |
| # params | 169M | 424M | 863M | 1.3B | 3B | 8B |
| Num Layers | 12 | 24 | 24 | 24 | 26 | 40 |
| Embedding dim. | 768 | 1024 | 1536 | 2048 | 3072 | 4096 |
| Num. heads | 12 | 16 | 16 | 32 | 32 | 32 |

### B.1 Kullback-Leibler Divergence

The Kullback-Leibler Divergence (KLD) between two discrete distributions $p$ and $q$ defined on the same finite sample space $\Omega$ is computed as

$$ $D_{\text{KL}}(p||q):=\sum_{x\in\Omega}p(x)\log\frac{p(x)}{q(x)}.$ (9) $$

The KLD has a unique minimum when $p$ and $q$ are equal, however the KLD is not symmetric, meaning that $D_{\text{KL}}(p||q)\neq D_{\text{KL}}(q||p)$ in general. In this work, we train the student with the reverse KLD $D_{KL}(p_{\theta}||p_{\text{teacher}})$. In the next paragraphs, we present differences between $D_{KL}(p_{\text{teacher}}||p_{\theta})$ (forward KLD) and $D_{KL}(p_{\theta}||p_{\text{teacher}})$ (reverse KLD).

#### The Forward KLD

The forward KLD is called zero-avoiding because if $p_{\text{target}}(x)$ is non-zero but $p_{\theta}(x)$ is close to zero, then $p_{\text{target}}(x)\frac{p_{\text{target}}(x)}{p_{\theta}(x)}$ will be large. To minimize the forward KLD, $p_{\theta}$ will try to assign non-zero probability to all points where $p_{\text{target}}$ is non-zero.

#### The Reverse KLD

The reverse KLD is called zero-forcing because if $p_{\text{target}}(x)$ is close to zero but $p_{\theta}(x)$ is not, $p_{\theta}(x)\frac{p_{\theta}(x)}{p_{\text{target}}}$ will be large. To minimize the reverse KLD, $p_{\theta}$ will try to assign zero probability to points where $p_{\text{target}}$ is close to zero.

### B.2 Total Variation Distance

The total variation distance (TVD) is a metric used to compare two probability distributions. For two discrete probability distributions $p$ and $q$ defined on the same finite sample space $\Omega$, the TVD is computed as:

$$ $d_{\text{TV}}(p,q)=\frac{1}{2}\sum_{x\in\Omega}|p(x)-q(x)|.$ (10) $$

The factor of $1/2$ ensures that the TVD ranges between 0 and 1, where $d_{\text{TV}}(p,q)=0$ if and only if $p=q$.

### B.3 Mean-Squared Error

Unlike the Kullback-Leibler divergence (KLD) and Total Variation Distance (TVD), the MSE can be used to compare any scalar quantities, not just probability distributions. For numerical stability, we compute the MSE in log space:

$$ $\text{MSE}(p,q)=\frac{1}{|\Omega|}\sum_{x\in\Omega}\left(\log p(x)-\log q(x) \right)^{2}.$ (11) $$

### B.4 χ 2 superscript 𝜒 2 \chi^{2} italic_χ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT divergence

The $\chi^{2}$ divergence can be used to compare two probability distributions. For two discrete probability distributions $p$ and $q$ defined on the same sample space $\Omega$m the $\chi^{2}$ divergence is computed as:

$$ ${d_{\chi^{2}}(p,q)=\sum_{x\in\Omega}q(x)\left(\frac{p(x)}{q(x)}-1\right)^{2}= \sum_{x\in\Omega}\frac{1}{q(x)}\left(p(x)-q(x)\right)^{2}.}$ (12) $$

As such, we see that the $\chi^{2}$ divergence is related to the MSE. Note that when using the MSE for distillation, we penalize the error in log space, while the $\chi^{2}$ penalizes error in probability space. Additionally, the MSE uses a uniform weight factor $\frac{1}{|\Omega|}$ for each term of the sum, while the $\chi^{2}$ divergence uses a weight of $\frac{1}{q(x)}$.

## Appendix C Implementation details

#### Architecture

To compare with , we trained the diffusion models using their code and pre-processing steps on the OpenWebText dataset . As , our models are not conditioned on the noise level. Nonetheless, kept the architecture of unchanged and makes the model unconditional by feeding it a zero tensor instead of the noise level. Removing the adaptive layers could improve the sampling speed further, but we avoided modifying the architecture to prevent potential problems. See table 2 for the hyperparameters of our models.

## Appendix D Text examples

We include non-cherry picked text generated from the small distilled model with KLD loss from the last round of distillation via unconditional sampling with varying number of steps. We show the first 512 tokens to so that the text fits on one page. Remember that those models are small and not fine-tuned for text quality. They can also start generating in the middle of sentences, since they are trained on a concatenated corpus of documents.