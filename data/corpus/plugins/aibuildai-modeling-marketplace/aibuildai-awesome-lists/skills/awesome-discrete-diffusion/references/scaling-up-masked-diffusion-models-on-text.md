---
arxiv_id: "2410.18514"
title: "Scaling up Masked Diffusion Models on Text"
year: 2024
source: arxiv2md
---

## Abstract

Abstract Masked diffusion models (MDMs) have shown promise in language modeling, yet their scalability and effectiveness in core language tasks, such as text generation and language understanding, remain underexplored. This paper establishes the first scaling law for MDMs, demonstrating a scaling rate comparable to autoregressive models (ARMs) and a relatively small compute gap. Motivated by their scalability, we train a family of MDMs with up to 1.1 billion (B) parameters to systematically evaluate their performance against ARMs of comparable or larger sizes. Fully leveraging the probabilistic formulation of MDMs, we propose a simple yet effective unsupervised classifier-free guidance that effectively exploits large-scale unpaired data, boosting performance for conditional inference. In language understanding, the 1.1B MDM outperforms the 1.1B TinyLlama model trained on the same data across four of eight zero-shot benchmarks. Notably, it achieves competitive math reasoning ability with the 7B Llama-2 model on the GSM8K dataset.
In text generation, MDMs with 16 times more pre-training time offer a flexible trade-off against ARMs with the accelerated sampling technique KV-Cache: MDMs match ARMs in performance while being 1.4 times faster during sampling.
Moreover, MDMs address challenging tasks for ARMs by effectively handling bidirectional reasoning and adapting to temporal shifts in data. Notably, a 1.1B MDM breaks the reverse curse encountered by much larger ARMs with significantly more data and computation, such as 13B Llama-2 and 175B GPT-3. Our code is available at https://github.com/ML-GSAI/SMDM .

## 1 Introduction

Autoregressive models (ARMs) have long been regarded as the gold standard in probabilistic language modeling. Their ability to predict the next token, grounded in the chain rule, naturally aligns with the sequential nature of language and scales effectively  when integrated with Transformers . However, ARMs exhibit inherent limitations, particularly in reasoning tasks that require bidirectional context understanding or handling temporal shifts in data. These shortcomings, widely recognized as the *reverse curse*  and *temporal quality degradation* , significantly hinder their applicability in complex language modeling scenarios. Additionally, their linear sampling time growth w.r.t. the output length poses practical challenges for long text generation.

The limitations of ARMs have sparked interest in an alternative approach: masked diffusion models (MDMs) . MDMs present a promising alternative due to their unique probabilistic framework, which enables flexible bidirectional context modeling by filling in masked positions across a sequence. Recent advances  have shown promise in unconditional text generation and zero-shot perplexity evaluation. Despite recent progress, the scalability of MDMs and their effectiveness in critical language tasks, such as conditional generation and language understanding, remain open questions. Furthermore, it is still unclear whether MDMs can address the inherent limitations of ARMs, such as improving bidirectional reasoning capabilities.

Given that scalability and generality across tasks are core attributes of large language models, advancing MDMs requires not only a focus on algorithm design  but also attention to an orthogonal dimension: the exploration of scalability and generality. From this perspective, this paper challenges the longstanding dominance of ARMs by presenting a comprehensive study of MDMs regarding key factors in language models: scalability, language understanding capabilities, and conditional generation performance.
To achieve this, we train a family of MDMs with up to 1.1 billion (B) parameters on a large-scale dataset and establish the first scaling law for MDMs.
Leveraging their unique probabilistic framework, we propose a simple yet effective *unsupervised classifier-free guidance (CFG)* mechanism to leverage unsupervised data to enhance inference performance in language tasks involving conditional distributions. Notably, unsupervised CFG does not rely on paired data as standard CFG  but can still benefit from paired data when available, achieving performance that surpasses standard CFG. Supported by the scaling law and unsupervised CFG, our extensive experiments yield the following key findings:

- •
Strong scalability. As the IsoFLOP analysis  scaling compute budgets from $6\times 10^{18}$ to $10^{20}$ FLOPs (see Fig. 1), the optimal validation loss of MDMs decreases according to a power law, with a rate matching that of ARMs (see Fig. 2). While MDMs maintain a constant computation gap of 16 times compared to ARMs, this gap is smaller than the factor of 64 observed in continuous diffusion models  and can be further minimized with future optimizations.
- •
Competitive in language understanding. Across eight standard zero-shot benchmarks, including tasks like *commonsense reasoning* and *reading comprehension*, our 1.1B MDM outperforms the larger 1.5B GPT-2 model on six tasks and the same-sized TinyLlama (with equivalent pre-training FLOPs) on four tasks. Moreover, the 1.1B MDM demonstrates competitive *math reasoning* performance compared to the 7B Llama-2 on the GSM8K dataset, while utilizing less than $5\%$ of its pre-training FLOPs.
- •
Flexible trade-off in conditional generation. On the standard MT-Bench, a 1.1B MDM matches the performance of a same-sized ARM while achieving a 1.4 times speedup in sampling time. By increasing sampling steps, MDMs can further improve generation quality at the cost of being 1.4 times slower. Notably, ARMs are equipped with KV-cache, a technique to speed up sequential sampling while MDMs exploit no system optimization but require 16 times pre-training time.
- •
Addressing challenging tasks for ARMs. MDMs effectively relieve *temporal quality degradation*  compared to a same-sized ARM and successfully overcome the *reverse curse*  encountered by much larger ARMs with significantly more data and computation, such as 13B Llama-2 and 175B GPT-3.

## 2 Masked Diffusion Models on Text

In analogy to continuous diffusion models , MDMs  also introduce a forward process that gradually adds noise to the data and learn a corresponding reverse process to generate samples. Our basic approach is built upon , an advanced MDM suitable for scaling.

Forward process. Let $K$ and $L$ denote the vocabulary size and sentence length respectively. Given a sentence ${\bm{x}}_{0}\in\{0,1,\dots,K-1\}^{L}$ and a noise level $t\in[0,1]$, the forward process in MDMs randomly and independently masks out tokens in the sentence, formulated as follows:

$$ $\displaystyle q_{t|0}({\bm{x}}_{t}|{\bm{x}}_{0})=\prod_{i=0}^{L-1}q_{t|0}({\bm {x}}_{t}^{i}|{\bm{x}}_{0}^{i})\quad\text{and}\quad q_{t|0}({\bm{x}}_{t}^{i}|{ \bm{x}}_{0}^{i})=\begin{cases}\alpha_{t},&{\bm{x}}_{t}^{i}={\bm{x}}_{0}^{i},\\ 1-\alpha_{t},&{\bm{x}}_{t}^{i}=m,\end{cases}$ (1) $$

where ${\bm{x}}^{i}$ denotes the $i$-th element of ${\bm{x}}$, $m$ denotes the mask token , ${\bm{x}}_{t}$ denotes the noisy data at time $t$ and $q_{0}(\cdot)$ is the data distribution $p_{\textrm{data}}(\cdot)$. We set the hyperparameter $\alpha_{t}$ as $1-t$ for the best empirical performance as suggested in previous work .

Reverse process. The reverse process in MDMs
iteratively recover values for masked tokens, starting from a mask sequence ${\bm{x}}_{1}$. Let $0\leq s<t\leq 1$, the reverse process is characterized by

$$ $\displaystyle q_{s|t}({\bm{x}}_{s}|{\bm{x}}_{t})=\prod_{i=0}^{L-1}q_{s|t}({\bm {x}}_{s}^{i}|{\bm{x}}_{t})~{}~{}\text{and}~{}~{}q_{s|t}({\bm{x}}_{s}^{i}|{\bm{ x}}_{t})=\begin{cases}1,&{\bm{x}}_{t}^{i}\neq m,{\bm{x}}_{s}^{i}={\bm{x}}_{t}^ {i},\\ \frac{s}{t},&{\bm{x}}_{t}^{i}=m,{\bm{x}}_{s}^{i}=m,\\ \frac{t-s}{t}q_{0|t}({\bm{x}}_{s}^{i}|{\bm{x}}_{t}),&{\bm{x}}_{t}^{i}=m,{\bm{x }}_{s}^{i}\neq m,\\ 0,&\textrm{otherwise}.\end{cases}$ (2) $$

Here $q_{0|t}(\cdot|\cdot)$ is the data prediction model  to be learned. Notably, revealed an intrinsic property of MDMs that $q_{0|t}(\cdot|\cdot)$ can be represented by conditional distributions on clean data $p_{\text{data}}(\cdot|\cdot)$ independently from the time $t$, distinct from other diffusion. Formally,

$$ $\displaystyle q_{0|t}({\bm{x}}_{0}^{i}|{\bm{x}}_{t})=p_{\text{data}}({\bm{x}}_ {0}^{i}|{\bm{x}}_{t}^{\text{UM}}),$ (3) $$

where ${\bm{x}}_{t}^{\text{UM}}$ collects all unmasked tokens in noisy data ${\bm{x}}_{t}$ and $p_{\text{data}}(\cdot|\cdot)$ is irrelevant to $t$.(^1^11For example, if ${\bm{x}}_{t}=[3,5,m,2]$, then ${\bm{x}}_{t}^{\text{UM}}=[3,5,\cdot,2]$ and $p_{\text{data}}(\cdot|[3,5,\cdot,2])$ is irrelevant to $t$.)

Training objective. A distribution $p_{{\bm{\theta}}}({\bm{x}}_{0}^{i}|{\bm{x}}_{t})$ parameterized by ${\bm{\theta}}$ is employed to approximate $p_{\text{data}}({\bm{x}}_{0}^{i}|{\bm{x}}_{t}^{\text{UM}})$, optimizing the following upper bound on negative log-likelihood :

$$ $\displaystyle-\log p_{{\bm{\theta}}}({\bm{x}}_{0})\leq\int_{0}^{1}\frac{1}{t} \mathbb{E}_{q({\bm{x}}_{t}|{\bm{x}}_{0})}\left[\sum_{\{i|{\bm{x}}_{t}^{i}=m\}} -\log p_{{\bm{\theta}}}({\bm{x}}_{0}^{i}|{\bm{x}}_{t})\right]dt\triangleq \mathcal{L}.$ (4) $$

We emphasize that the formulation is particularly suitable for scaling. First, it is among the best MDMs w.r.t. zero-shot perplexity . Second,
it removes the timestep from input and minimally modifies the original Transformers (see Sec. 3). Third, it enables unsupervised classifier-free guidance, which does not rely on paired data yet is effective in language tasks (see Sec. 4).

## 3 Scaling Laws for Masked Diffusion Models

Scaling laws  characterize the quantitative power-law relationship between model performance and computational resources under constraints, significantly influencing the progress of large ARMs. Previous work  fine-tunes pre-trained XLM-RoBERTa  models into MDMs and investigates scaling trends by varying the size of XLM-RoBERTa. However, a detailed exploration of scaling laws for MDMs, along with a fair comparison to ARMs in terms of scalability, remains absent. In this section, we address these two key questions. Our results reveal the strong scalability of MDMs, highlighting their potential as a competitive alternative to ARMs in language modeling.

Model. We employ a Transformer decoder for ARMs and the corresponding Transformer encoder for MDMs (note that it is unnecessary to input timestep $t$ according to Eq. (3)). The differences between these architectures are: (1) the encoder has an additional dimension in its embedding layer for the mask token, and (2) the encoder’s self-attention does not use a causal mask. All other architectural settings (e.g., depth, hidden size, and number of heads) remain consistent in both models.

We further enhance both models with several techniques inspired by advanced language models like Llama . Specifically, we adopt Pre-LayerNorm with RMSNorm  for better stability, use SwiGLU  as the activation function to enhance non-linearity, and implement RoPE  for more expressive positional encoding.

Data. The well-known Chinchilla scaling law  utilizes a large dataset with more data than the number of training tokens. Motivated by it, we employ the open-source SlimPajama dataset , a multi-corpora dataset comprising 627 billion tokens, which is sufficiently large for all of our experiments. For simplicity and fairness, we employ the Llama-2 tokenizer  for both ARMs and MDMs. Additionally, we set the context length to $2048$. Further implementation details are provided in Appendix B.2.

IsoFLOP analysis. We conduct a standard IsoFLOP analysis  to identify the optimal allocation between the non-embedding parameters $N$ and dataset size $D$. Specifically, building on prior studies , we scale the compute budget $C$ from $6\times 10^{18}$ to $10^{20}$ FLOPs. For a fixed $C$, we vary $N$ and $D$ such that $C=6ND$, a relationship valid for both ARMs and MDMs. We fit a quadratic function to capture the relationship between the validation loss $\mathcal{L}$ and the logarithm of the parameter size $\log N$. Specifically, the loss function $\mathcal{L}$ of MDMs is defined in Eq. (4). This regression allows us to determine the optimal parameter size $N^{*}_{C}$, which corresponds to the minimum validation loss $\mathcal{L}^{*}_{C}$ for a given compute budget $C$. The IsoFLOP analysis results are visualized in Fig. 1.

Scaling laws. After obtaining the optimal validation losses for the corresponding compute budget in $\{C_{0},C_{1},\dots,C_{n-1}\}$, we fit the following scaling law to model the relationship between them:

$$ $\displaystyle\min_{\alpha,\beta}\sum_{i=0}^{n-1}\left(\log\mathcal{L}^{*}_{C_{ i}}-\alpha\log C_{i}-\beta\right)^{2}.$ (5) $$

Let ${\alpha^{*}}$ and ${\beta^{*}}$ denote the solution of Eq. (5) and the validation loss empirically follows $\mathcal{L}=e^{\beta^{*}}C^{\alpha^{*}}$.

Figure: (a) Loss-Flops curve.
Refer to caption: x3.png

As illustrated in Fig. 2(a), the validation loss of MDMs decreases according to a power law as the compute budget increases, following a rate similar to that of ARMs. However, MDMs still require approximately 16 times more computational resources than ARMs to achieve comparable validation losses. Furthermore, the optimal model size also follows a power-law relationship with the compute budget, as shown in Fig. 2(b). The optimal size of MDMs is approximately half that of ARMs across different computations, reflecting a similar scaling behavior on utilizing the parameter capacity.

Despite the constant gap in computational resources (i.e., MDMs require 16 times more resources than ARMs), there is still potential to narrow this gap since optimizations for MDMs in model, data, and
system remain unexplored. Besides, for reference, reported that continuous diffusion models (CDMs) require 64 times more computational resources than ARMs. In Sec. 5, we further emphasize that MDMs achieve competitive results on commonsense reasoning and reading comprehension tasks compared to ARMs under equivalent computational resources. Additionally, our findings show that MDMs exhibit competitive mathematical reasoning capabilities (see Sec. 5) and superior bidirectional modeling ability (see Sec 7.1) when compared to significantly larger ARMs operating with far greater compute budgets.

In conclusion, the comparable scaling rates and the relatively small constant factors suggest that MDMs have strong scalability and promising potential as an alternative to ARMs on a large scale.

## 4 Unsupervised Classifier-free Guidance

We propose a surprisingly simple yet effective approach that leverages unlabeled data to boost performance in various language tasks, dubbed *unsupervised classifier-free guidance (CFG)*.

CFG. CFG  is an effective and versatile technique widely used in both continuous and discrete diffusion models, with applications spanning image  and text generation . Rooted in Bayes’ rule, CFG simultaneously trains a conditional and an unconditional diffusion model, introducing a rescaled distribution for inference. Specifically, at a given timestep $t\in[0,1]$, CFG  is defined as:

$$ $\displaystyle\tilde{p}_{{\bm{\theta}}}({\bm{x}}_{0}|{\bm{c}},{\bm{x}}_{t}) \propto\frac{p_{{\bm{\theta}}}({\bm{x}}_{0}|{\bm{c}},{\bm{x}}_{t})^{1+w}}{p_{{ \bm{\theta}}}({\bm{x}}_{0}|{\bm{x}}_{t})^{w}},$ (6) $$

where ${\bm{c}}$ is the condition, $w$ is a hyperparameter that flexibly controls the strength of ${\bm{c}}$, and $p_{{\bm{\theta}}}({\bm{x}}_{0}|{\bm{c}},{\bm{x}}_{t})$ and $p_{{\bm{\theta}}}({\bm{x}}_{0}|{\bm{x}}_{t})$ are the conditional and unconditional models respectively.

Notably, it seems that the conditional model must be trained on paired data before applying CFG. Consequently, to the best of our knowledge, all existing work  fall into supervised settings, where paired data are readily available.

Unsupervised CFG. We extend CFG to an unsupervised setting by introducing a new formulation:

$$ $\displaystyle\tilde{p}_{{\bm{\theta}}}({\bm{x}}_{0}|{\bm{c}},{\bm{x}}_{t}) \propto\frac{p_{{\bm{\theta}}}({\bm{x}}_{0}|{\bm{c}},{\bm{x}}_{t})^{1+w}}{p_{{ \bm{\theta}}}({\bm{x}}_{0}|{\bm{m}},{\bm{x}}_{t})^{w}},$ (7) $$

where ${\bm{m}}$ is a mask sequence of the same length as ${\bm{c}}$. Compared to Eq. (6), the dummy variable ${\bm{m}}$ translates the unconditional distribution to a conditional format without adding new information. For simplicity, we continue to refer to
$p_{{\bm{\theta}}}({\bm{x}}_{0}|{\bm{m}},{\bm{x}}_{t})$ as the unconditional distribution in unsupervised CFG throughout this paper.

The core insight is that an MDM already characterizes both distributions employed in Eq. (7) during unsupervised pretraining. Specifically, in language tasks, both ${\bm{c}}$ and ${\bm{x}}$ can be viewed as segments of a whole sequence, following the same distribution of unsupervised samples for pretraining.(^2^22E.g., the question “where does the sun rise?” and answer “from the east.” is a paired sample but their concatenation “where does the sun rise? from the east.” can be modeled by an MDM with unsupervised training.) After the pretraining on large-scale text data, MDMs can capture the joint distribution of the whole sequence, i.e., $p_{\textrm{data}}({\bm{c}},{\bm{x}})$.
Under the formulation, MDMs simultaneously learn all conditional distributions on clean data induced by $p_{\textrm{data}}({\bm{c}},{\bm{x}})$ according to Eq. (3). In particular, we have:

$$ $\displaystyle p_{{\bm{\theta}}}({\bm{x}}_{0}|{\bm{c}},{\bm{x}}_{t})\approx p_{ \text{data}}({\bm{x}}_{0}|{\bm{c}},{\bm{x}}_{t}^{\text{UM}})\quad\text{and} \quad p_{{\bm{\theta}}}({\bm{x}}_{0}|{\bm{m}},{\bm{x}}_{t})\approx p_{\text{ data}}({\bm{x}}_{0}|{\bm{x}}_{t}^{\textrm{UM}}),$ (8) $$

where both distributions are factorized as in Eq. (3), and the approximation error is due to the gap between the model distribution and the true data distribution. Notably, Eq. (8) also implies that the unconditional distribution $p_{{\bm{\theta}}}({\bm{x}}_{0}|{\bm{x}}_{t})$ used in standard CFG and the conditional distribution with a dummy variable $p_{{\bm{\theta}}}({\bm{x}}_{0}|{\bm{m}},{\bm{x}}_{t})$ share a similar role.

We have explained why unsupervised CFG works without paired data (see Sec. 5). Moreover, when paired data are available for downstream tasks, simply fine-tuning the conditional distribution in MDMs—similar to the classical approach used for ARMs—not only further improves the performance of unsupervised CFG but also outperforms the standard CFG trained on paired data, demonstrating its superior capability in leveraging large-scale unpaired data (see Sec. 6). While prior studies  emphasize the role of unconditional distributions in large language models, we show that unsupervised CFG employs a distinct motivation and formulation, as detailed in Appendix C.1.

## 5 Language Understanding

We investigate the capabilities of MDMs in language understanding, a critical skill for language models that has been largely overlooked in prior studies . Our results show that MDMs are highly competitive to ARMs of similar model sizes and computations.

Benchmarks. To provide a comprehensive evaluation, we assess MDMs on eight widely used benchmarks in the zero-shot setting, covering tasks in *commonsense reasoning* and *reading comprehension*: Hellaswag , ARC-e , BoolQ , PIQA , SIQA , Obqa , RACE , and LAMBADA . Additionally, we also assess the *math reasoning* ability on the GSM8K  dataset. Following previous works , we finetune MDM on the augmented training data  and test on GSM8K. For a detailed description of these benchmarks, please see Appendix D.

On certain challenging benchmarks such as ARC-c , WinoGrande , and MMLU , both ARMs and MDMs pre-trained in Sec. 3 perform similarly to random guessing. This is consistent with findings from , which showed that only ARMs with more than $10^{22}$ training FLOPs can surpass random guessing on MMLU, a phenomenon known as the emergence of new capabilities in large language models. We leave the exploration of their potential emergent abilities at a larger scale as future work.

Evaluation. We follow the widely used Language Model Evaluation Harness framework  to evaluate both ARMs and MDMs. For LAMBADA and GSM8K, given a prompt, we use greedy sampling to generate responses from each model and calculate matching accuracy against the ground truth (see Appendix A for details on the greedy sampling algorithm used for MDMs). For the other tasks, we report the accuracy of each model that selects the correct answer from the provided options based on the given context. Specifically, we compute the likelihood of each option given the prompt and choose the answer with the highest likelihood.

Table: Table 1: Ablation of unsupervised CFG without paired data. Unsupervised CFG significantly improves the zero-shot performance of MDMs across eight tasks.

Fixing the train-test discrepancy. Due to employing a bidirectional Transformer encoder, MDMs face a train-test discrepancy in context lengths, negatively impacting model performance. Specifically, the training context length is fixed at 2048 tokens, while the testing context length is variable and often shorter. To address this issue, we propose two mitigation strategies: (1) allocate a portion of training data with variable sequence lengths $L\sim\mathcal{U}[1,2048]$, where $\mathcal{U}[\cdot]$ denotes the uniform distribution; (2) pad sentences with mask tokens to reach 2048 tokens during evaluation.

As present in Appendix C.2, both strategies effectively reduce the train-test discrepancy, and only a small proportion (e.g., $1\%$) of variable-length training data is sufficient to activate the capability to handle variable length inputs. Given its superior inference efficiency (e.g., 20 times faster than method (2) on the Hellaswag dataset), we employ method (1) in subsequent experiments.

Flexible likelihood evaluation. As detailed in Sec. 2, the MDMs model the conditional distribution of clean data, which enables flexible likelihood evaluation. Given a prompt and a sentence ${\bm{x}}_{0}$ of length $L$, we can determine the conditional likelihood using the following methods: (1) employ Monte Carlo estimation to establish a lower bound of the log-likelihood based on Eq. (4); (2) utilize the chain rule to compute the likelihood as $\log p_{{\bm{\theta}}}({\bm{x}}_{0}|\text{prompt})=\sum_{i=0}^{L-1}\log p_{{
\bm{\theta}}}({\bm{x}}_{0}^{i}|\text{prompt},{\bm{x}}_{0}^{<i},m)$.

We observed that the chain rule for likelihood evaluation results in higher accuracy for Obqa and PIQA, while Monte Carlo estimation yields better accuracy for ARC-e, Hellaswag, RACE, and SIQA. Since the answer length of BoolQ consists of only one token (“Yes” or “No”), both methods produce identical results. We adopted this optimal configuration in subsequent experiments and please refer to Appendix C.2 for more details and an empirical explanation.

Effectivenes of unsupervised CFG without paired data. In this section, we use a default MDM model with 220M parameters and a training budget of $10^{20}$ FLOPs for efficiency. For likelihood evaluation, we use the rescaled conditional distribution defined in Eq. (7) of unsupervised CFG. Since no paired data is available, standard CFG cannot be applied in this scenario.
As shown in Table 1, unsupervised CFG significantly enhances the performance of MDMs across all eight widely used benchmarks, demonstrating its strong capability to leverage unpaired data effectively.

Competitive zero-shot language understanding performance.
As shown in Table 2, our 1.1B MDM outperforms the 1.1B TinyLlama  on four out of eight tasks when trained on the same SlimPajama dataset with an equivalent pre-training FLOPs(^3^33We used the intermediate checkpoint officially provided by TinyLlama. See Table 7 for download links.). Additionally, the 1.1B MDM surpasses the larger 1.5B GPT-2 model on six of the eight benchmarks.

Notably, Table 2 also highlights that our 1.1B MDM achieves performance comparable to the much larger 7B Llama-2  on mathematical reasoning tasks, as measured by GSM8K accuracy, while requiring less than 5% of the pre-training FLOPs. These findings emphasize the competitive performance of MDMs relative to ARMs.

We also analyze the scaling behavior of MDMs on the language understanding tasks and observe a clear trend: as validation loss decreases, performance on most tasks improves correspondingly. This indicates a positive scaling signal, suggesting that MDMs have the potential to achieve even stronger capabilities with further scaling. Detailed results and analyses are provided in Appendix C.2.

Table: Table 2: Evaluation of our 1.1B MDM. Both the MDM and Llama-2 models are fine-tuned for GSM8K, with all other benchmarks assessed in zero-shot settings. Result marked ^∗ is from . The pre-training datasets consist of approximately $540$B tokens for TinyLlama and MDM, compared to $2$T tokens for Llama-2. Our 1.1B MDM outperforms the same-size TinyLlama on four out of eight tasks and surpasses the larger GPT-2 on six tasks. Notably, MDM achieves GSM8K accuracy comparable to that of Llama-2, requiring less than $5\%$ of its pre-training FLOPs.

## 6 Conditional Language Generation

We investigate the capabilities of MDMs in conditional generation, another core language task largely unexplored previously. Our results show that a 1.1B MDM achieves a more flexible and effective quality-efficiency trade-off during inference than a same-sized ARM that utilizes KV cache.

Evaluation. Previous studies  have commonly employed generative perplexity as a metric to assess unconditional generation quality. However, recent work demonstrated that even low-quality samples can yield high generative perplexity scores, suggesting that this metric may not reliably reflect generative quality. Moreover, conditional generation is more widely applicable in real-world scenarios than unconditional generation. Therefore, this paper focuses on conditional generation.

In particular, we employ MT-Bench , which uses a strong language model (i.e., GPT-4o ) as a judge to score models on open-ended questions. This metric aligns well with human preferences and has become a standard for evaluating large language models.

Supervised fine-tuning. We employ an ARM and an MDM, both pre-trained as described in Sec. 3 with 1.1B parameters each.
For a meaningful comparison, we evaluate their inference performance and, guided by the scaling law, extend the MDM’s pre-training time by a factor of 16. Results using equal computation budgets are provided in Appendix C.3.
Following a standard process in language models, we fine-tune both models on the ShareGPT dataset(^4^44[https://sharegpt.com/](https://sharegpt.com/)), a high-quality dialogue corpus containing user prompts and corresponding ChatGPT responses .

Since ShareGPT samples vary in length, we pad each sample with the $|\text{EOS}|$ token to the maximum sequence length within a batch for the MDM. Following the same approach as for ARMs, we mask the loss on prompts, adding noise only to the response tokens (including the padding $|\text{EOS}|$), while keeping the prompts unchanged in the forward process. As a result, the MDM only tunes the conditional distribution of the response given prompt. We set the sequence length to 1024 and remove the $|\text{EOS}|$ token from the generated outputs during inference. For the ARM, generation stops when the $|\text{EOS}|$ token is produced, with a maximum sequence length set to 1024 . For a fair comparison, we use identical optimizer settings for both models and train for 3 epochs as specified in . Additional training details are provided in Appendix B.4.

Effectiveness of unsupervised CFG against standard CFG. As shown in Table 4, we evaluate the effectiveness of unsupervised CFG by comparing it against several baselines detailed in Appendix B.4. The first one fine-tunes only the conditional distribution of MDM on paired data and sampling without CFG. The second one fine-tunes both conditional and unconditional distributions on paired data and gets samples as in the standard CFG. Additionally, we enhance unsupervised CFG by fine-tuning its conditional distribution on paired data. This is because unsupervised CFG already leverages large-scale pre-trained data to obtain a strong unconditional model.
Notably, our unsupervised CFG outperforms the standard CFG, demonstrating its superior ability to leverage large-scale unpaired data considering the paired data for fine-tuning are often of a small scale. For a comprehensive comparison, we also demonstrate that unsupervised CFG outperforms sampling without CFG with half the sampling steps (i.e., equal sampling computation) in Appendix C.3.

Table: Table 3: Ablation of unsupervised CFG. The symbols ^∗ and ^† indicate the standard CFG and unsupervised CFG respectively. We report the results with the optimal scale searched in $\{0.4,0.6,0.8,1\}$ for both CFG approaches.

Better efficiency quality trade-off. We further compare MDMs and ARMs regarding sample quality and efficiency. Our study significantly extends prior work  in two key aspects: (1) we focus on the more practical and challenging task of conditional generation rather than unconditional generation, and (2) we measure the running time instead of the NFEs, even when ARMs are equipped with the KV-cache, a technique that accelerates sampling by caching intermediate features during sequential generation.

Built upon the unsupervised CFG, MDMs demonstrate a more flexible and effective trade-off between efficiency and quality in conditional generation compared to ARMs. As shown in Table 4, a 1.1B MDM matches the performance of a similarly sized ARM while achieving a 1.4 times speedup in sampling time. Conversely, by increasing the number of sampling steps (at the cost of being 1.4 times slower), MDMs can surpass ARMs in generation quality. All experiments in Table 4 are conducted on a single NVIDIA A100-40GB GPU. These results indicate that MDMs hold promise for conditional generation tasks, such as chat-based applications, where the ability to balance speed and quality is critical.

## 7 Challenging Tasks for ARMs

We demonstrate that MDMs exhibit distinct advantages over ARMs in tackling two critical challenges: *reverse curse*  and *temporal quality degradation* .

### 7.1 Breaking the Reverse Curse

introduced the concept of the reverse curse, which refers to the difficulty of ARMs in generalizing bidirectional relationships. Specifically, this occurs when a model is trained on information in the form “A is B” but fails to infer the reverse relationship “B is A.” For example, a model trained on the fact “Valentina Tereshkova was the first woman to travel to space” may not correctly answer the reverse question “Who was the first woman to travel to space?” This limitation raises concerns about whether large language models genuinely possess logical reasoning capabilities .

Table: Table 5: Results on breaking the reverse curse. The performance of GPT-3 and Llama-2 is sourced from and , respectively. All models are fine-tuned on the same dataset for 10 epochs. For MDM, we use a CFG scale of 0.8. While ARMs and T5 struggle to handle reverse queries, MDMs effectively overcome the reverse curse and maintain performance in the same direction.

Setup. We evaluate MDMs on the same reverse curse dataset used by , which consists of fictitious statements in the format “$\langle\text{name}\rangle$ is $\langle\text{description}\rangle$” and the reversals. We fine-tune MDMs on these statements and assess their performance using questions not seen during training. To ensure a comprehensive comparison, we additionally fine-tuned the T5  model using the same dataset (see Appendix B.5 for details). Following the same protocol as , we generate responses via greedy sampling and report the exact match accuracy. Additionally, we use the BLEU metric  to evaluate the quality of name-to-description generation, as suggested by .

Results. As shown in Table 5, both the T5  model and advanced ARMs achieve zero accuracy and low BLEU scores when prompted with reverse queries.
In contrast, MDMs achieve substantially higher scores across both metrics, despite using significantly fewer parameters, less computation, and a smaller training dataset. Specifically, our MDM uses only 10% parameters, 1% computation, and 10% data compared to 13B Llama-2.
Besides, MDMs perform similarly to ARMs with queries in the same direction.
These results indicate the power of MDMs in capturing bidirectional relationships and logical structures. This capability arises from the training objective of MDMs (i.e., Eq. (4)), designed to model all conditional distributions within the data. Further, we provide additional clarification on Table 5 (e.g., the lower accuracy of MDM in the reverse direction compared to the same direction) in Appendix C.4.

### 7.2 Relieving the Temporal Quality Degradation

highlight a common and challenging issue for modern AI models, including language models: model performance is sensitive to the temporal alignment between the training and test data, particularly when new data fall outside the temporal scope of the training set.

Setup. To evaluate the impact of temporal shifts, we train both ARMs and MDMs on the SlimPajama dataset  (see Sec. 3), released in 2023, and test them on the FineWeb dataset , which contains samples from February$\&$March, and April of 2024. We extract the first 0.5 billion tokens from each period for evaluation. We use models of equal size (220M parameters) that achieve similar validation losses on SlimPajama. However, it is worth noting that MDMs require 16 times more computation to reach this performance level.

Results. As shown in Table 6, although the MDM achieves slightly higher perplexity on the standard validation set (i.e., SlimPajama), it outperforms the ARM on the newer 2024 data. While the exact mechanism remains unclear, we hypothesize that this advantage arises from MDMs’ ability to simultaneously model all conditional distributions, making them less sensitive to distributional shifts compared to the unidirectional dependencies in ARMs. These results indicate that MDMs are inherently more robust to temporal shifts, making them better suited for evolving data distributions.

Table: Table 6: Perplexity ($\downarrow$) results on relieving temporal quality degradation. The symbol ^∗ indicates the training dataset. MDM demonstrates superior robustness to temporal shifts than ARM.

## 8 Conclusion

In this paper, we demonstrate the strong scalability of MDMs through a comprehensive scaling analysis. Our results show that MDMs can achieve comparable performance to ARMs in key tasks, such as language understanding, supported by the scaling law and the unsupervised classifier-free guidance. Furthermore, MDMs effectively address major limitations of ARMs, including the reverse curse and temporal quality degradation, even outperforming much larger models like Llama and GPT-3 in these aspects. These findings highlight MDMs as a promising alternative to ARMs for language modeling at scale.

We also observe that MDMs exhibit certain limitations, particularly in scaling laws and conditional generation, where a gap compared to ARMs persists. Our work provides a holistic view of the potential and limitations of MDMs, encouraging future research toward more efficient designs.

One of the most important future directions is to scale MDMs to larger sizes, potentially matching advanced ARMs . This would allow for a thorough investigation into the emergent behaviors  and long-range reasoning capabilities  of MDMs. By scaling up, we hope that MDMs can fully demonstrate their unique advantages over ARMs in real-world scenarios, offering a competitive alternative. Further, we believe the studies can deepen our understanding of large language models and the role of key factors such as autoregressive formulation in achieving such intelligence.

We also note another line of research focusing on continuous diffusion language models .
However, the experiments in this domain are relatively small in scale and lack evaluation on standard language benchmarks. We hypothesize that MDMs enjoy better scalability than these models due to their alignment with the inherent structure of language and ARMs.

## Ethics Statement

This paper focuses on the improvement of language models, which have vast potential to enhance communication, automate tasks, and facilitate access to information across languages. However, if misused, these models could be exploited to generate false information. Moreover, if trained on biased datasets, the generated text could perpetuate these biases. To mitigate these risks, we commit to transparency in our development processes and to continuously focus on research related to the safety and fairness of language models to further improve our models.

#### Acknowledgments

This work was supported by the National Natural Science Foundation of China (No. 92470118); Beijing Natural Science Foundation (No. L247030); Beijing Nova Program (No. 20220484044); Major Innovation & Planning Interdisciplinary Platform for the “Double-First Class” Initiative, Renmin University of China; the Fundamental Research Funds for the Central Universities, and the Research Funds of Renmin University of China (22XNKJ13). The work was partially done at the Engineering Research Center of Next-Generation Intelligent Search and Recommendation, Ministry of Education.

We thank Jingyang Ou for the insightful discussions on RADD. We also thank Ang Lv for valuable conversations about the reverse curse and Wenkai Yang for discussions on the supervised fine-tuning of ARMs. Additionally, we appreciate Siqi Kou for providing guidance on data processing and evaluation for the conditional generation experiments.

## Appendix A Greedy Sampling method of MDMs

We employ the sampling method of MaskGIT  as the greedy sampling strategy for MDMs. For completeness, we include the algorithm in Alg. 1 and provide the following intuitive explanation.

Let us first revisit the original sampling method for MDMs as described in Eq. (2). During each sampling step from time $t$ to $s$, if ${\bm{x}}_{t}^{i}\neq m$ it remains unchanged. Otherwise, it retains the masked state with a probability of $\frac{s}{t}$, or transitions to ${\bm{x}}_{0}^{i}\sim p_{{\bm{\theta}}}({\bm{x}}_{0}^{i}|{\bm{x}}_{t})$ with a probability of $1-\frac{s}{t}$. It is important to note that for all masked tokens ${\bm{x}}_{t}^{i}$, they transition to corresponding ${\bm{x}}_{0}^{i}$ with the same probability of $1-\frac{s}{t}$.

Different from the original sampling method, MaskGIT  does not transition all masked tokens to their corresponding ${\bm{x}}_{0}^{i}$ with the same probability of $1-\frac{s}{t}$. Instead, it specifically selects masked tokens that exhibit the highest conditional probability $p_{{\bm{\theta}}}({\bm{x}}_{0}^{i}|{\bm{x}}_{t})_{{\bm{x}}_{0}^{i}}$ for transition to ${\bm{x}}_{0}^{i}$.

Figure: Algorithm 1 Greedy sampling method of MDMs

## Appendix B Experimental details

### B.1 Reproducibility Statement

We implement our experiments based on the TinyLlama  codebase. We use the code provided by TinyLlama to preprocess the SlimPajama  dataset. Additionally, we use the code provided by CLLM  to preprocess the ShareGPT dataset. We employ the fictitious dataset provided by  and Fineweb dataset  for the reverse curse and temporal quality degradation experiments, respectively. Besides, when test math reasoning ability, we use the augmented data  for training and GSM8K  dataset for test. Because of their simplicity, we preprocess these four datasets by ourselves. We employ the lm-eval  and fast-chat  framework to evaluate language understanding tasks and conditional generation, respectively. In Sec. 5, the pre-trained GPT-2 and TinyLlama models are provided by HuggingaFace. The corresponding links are detailed in Tab. 7.

Table: Table 7: Links for code and checkpoints.

### B.2 Additional Experimental Details of IsoFLOP Analysis

Training details. We use identical optimizer settings for both MDMs and ARMs during pre-training. Consistency with TinyLLama , we utilize the AdamW optimizer , setting $\beta_{1}=0.9$, $\beta_{2}=0.95$, and a weight decay of $0.1$. Additionally, we apply a cosine learning rate schedule with a maximum learning rate of $4\times 10^{-4}$ and a minimum learning rate of $4\times 10^{-5}$ with $1\%$ of the tokens for linear warmup. Notably, if the number of warmup steps is less than $100$, it is set to $100$. The batch size is set to $256$.

Apart from the scaling law experiment, we also pre-train two 1.1B MDMs using $1.6\times 10^{21}$ and $3.3\times 10^{21}$ FLOPs for downstream tasks. Except for the batch size, we adopt the aforementioned pre-training settings for the 1.1B model. Specifically, we set the batch size to $384$ and $1024$ for the models trained with $1.6\times 10^{21}$ and $3.3\times 10^{21}$ FLOPs, respectively, due to the differing number of GPUs utilized. In the first version of this paper, we used the MDM with $1.6\times 10^{21}$ pre-training FLOPs in Sec. 5-6 and Sec 7.1. In the second version (this version), we replaced the MDM in Sec. 5 with the MDM pre-trained with $3.3\times 10^{21}$ FLOPs to more comprehensively demonstrate the scaling performance of the MDM.

Evaluation details. For MDMs, we found that using more Monte Carlo estimation samples (i.e., $128$) when computing the validation loss effectively reduces the number of outliers in Fig. 1(b). This is because increasing the number of Monte Carlo samples reduces the variance of the estimation, leading to a more precise estimation of the validation loss.

Model configs. We list all model configurations in Tab. 8.

Table: Table 8: Model configurations of MDMs and ARMs. ^∗ labels the non-embedding parameters.

### B.3 Additional Experiment Details of Language Understanding

We use the 1.1B MDM with $3.3\times 10^{21}$ pre-training FLOPs (see details in Appendex B.2) in Sec. 5. This model is pre-trained with $1\%$ data set to random length.

Additionally, we provide the experimental details for the GSM8K results. We fine-tune the MDM on the augmented training data  for 40 epochs, following prior works . The optimizer settings remain consistent with those described in Appendix B.2, and each data instance is padded with $|\text{EOS}|$ to a length of 256 tokens. For evaluation, we use greedy sampling, setting the sampling steps to $256$ and applying an unsupervised CFG scale of $0.1$.

Table: Table 9: Overview of different CFG strategies for conditional generation. The standard CFG fine-tunes both conditional and unconditional distributions on paired data, while unsupervised CFG is enhanced by fine-tuning only conditional distribution. Unsupervised CFG already leverages large-scale pre-trained data to obtain a strong unconditional model, resulting in improved performance compared to standard CFG.

### B.4 Additional Experimental Details of Conditional Generation

Setup. We use identical optimizer settings for both MDMs and ARMs during supervised fine-tuning. Similar to our pretraining process, we use the AdamW optimizer  with hyperparameters $\beta_{1}=0.9$, $\beta_{2}=0.95$, and a weight decay of $0.1$. We employ a cosine learning rate schedule starting from a maximum learning rate of $2\times 10^{-4}$ and decaying to a minimum of $2\times 10^{-5}$. Additionally, we apply linear warm-up over the first $200$ steps and set the batch size to $256$.

For the preprocessing of the ShareGPT dataset, we use the same method as described in . In addition, in line with , we fine-tune both ARMs and MDMs on the first-turn conversation from the ShareGPT dataset and report the first-turn conversation score. We do not use any annealing sampling method for ARMs and MDMs during generation. The MT-Bench score is obtained via the “gpt-4o-2024-05-13” API provided by OpenAI.

Different CFG strategies. We provide an overview of no CFG, standard CFG, and unsupervised CFG in Tab. 9.

During fine-tuning on labeled data, the standard CFG  replaces the label with a special token with a probability of $10\%$. This special token represents the unconditional distribution, thereby enabling the simultaneous training of both conditional and unconditional distributions. Specifically, for the implementation of standard CFG in our experiment, we randomly replace the prompt with the masked tokens with probability $10\%$.

In contrast to the standard CFG, unsupervised CFG already leverages large-scale pre-trained data to obtain a strong unconditional model, therefore we only enhance its conditional distribution during fine-tuning on paired data.

During inference, both standard CFG and unsupervised CFG employ the rescaled conditional distribution defined in Eq. (7).

### B.5 Additional Experimental Details of Reverse Curse

We use the same optimizer settings as Appendix B.4 except batch size when finetuning on the fictitious dataset provided by . As the fictitious dataset is smaller (i.e., only 3600 data), we use a batch size of 32 for fine-tuning. We train for 10 epochs following . We also pad each sample with the $|\text{EOS}|$ token to the maximum sequence length within a batch as detailed in Sec. 6. Following the same approach as , we do not mask the loss on prompts, adding noise to the prompt and response simultaneously as Eq. (4).

For the T5 model, we adopted the same settings as MDM, except for the learning rate. Initially, we tested maximum learning rates in {$10^{-5},10^{-4},10^{-3},10^{-2}$} and found that $10^{-4}$ yielded the best results. We further refined the learning rate by experimenting with $\{2\times 10^{-5},3\times 10^{-5},5\times 10^{-5},2\times 10^{-4},3\times 10^{
-4},5\times 10^{-4}\}$, identifying $2\times 10^{-4}$ as the optimal maximum learning rate. The minimum learning rate was set to one-tenth of the maximum.

Table: Table 10: Comparison of different methods to address train-test discrepancy. $1\%$ and $5\%$ denote that set $1\%$ and $5\%$ training data to random length, respectively. For simplicity, we employ the chain rule to calculate the conditional likelihood and do not use the unsupervised CFG. Both variable length training and padding mask tokens significantly improve the performance of MDMs in language understanding tasks.

## Appendix C Additional Results

### C.1 More Related Work About Unsupervised CFG.

Prior studies  emphasize the importance of unconditional distributions in large language models. By exploiting the distinctive properties of MDMs, unsupervised CFG introduces a novel approach to estimating the unconditional distribution (i.e., Eq. (8)) and incorporates this estimate through an alternative mechanism (i.e., Eq. (7)), inspired by the standard CFG framework (i.e., Eq. (6)).

### C.2 Additional Results of Language Understanding

Results of fixing traing-test discrepancy. For efficiency, we employ MDM with 220M parameters, pre-trained for $10^{20}$ FLOPs to experiment. Tab. 10 presents the ablation studies of variable length training and padding mask tokens, demonstrating that both methods significantly improve the performance of MDMs.

Results of different likelihood evaluation methods. For efficiency, we employ MDM with 220M parameters, pre-trained for $10^{20}$ FLOPs, and set $1\%$ training data to random length. Tab. 11 presents the ablation studies of different likelihood evaluation methods.

We empirically find that tasks requiring step-by-step reasoning tend to achieve higher accuracy when using the chain rule for likelihood evaluation. In contrast, tasks focused on contextual understanding perform better with Monte Carlo estimation. A comprehensive study is left for future work.

Table: Table 11: Comparison of different likelihood evaluation methods. We employed 1024 Monte Carlo samples for the Monte Carlo estimation. All results are reported with the corresponding optimal unsupervised CFG scale. The optimal likelihood evaluation method differs across tasks.

Scaling behavior of MDMs on language understanding tasks. As shown in Fig. 3, the performance of MDMs on the language understanding tasks shows a scaling behavior with respect to the validation loss, which is consistent with observations in ARMs . For efficiency and simplicity, methods for fixing train-test discrepancies and unsupervised CFG are not applied in this analysis.

### C.3 Additional Results of Conditional Generation

More MT-Bench results of MDM. In Sec. 6, we report the MT-Bench results of ARM and MDM with $10^{20}$ and $1.6\times 10^{21}$ pre-training FLOPs, respectively. Here, we present the MT-Bench result of MDM with $10^{20}$ pre-training FLOPs in Tab. 12.

Table: Table 12: MT-Bench results of MDM with $10^{20}$ pre-training FLOPs.

Additional ablation results on unsupervised CFG. Table 13 shows that unsupervised CFG outperforms sampling without CFG with half the sampling steps (i.e., equal sampling computation).

Table: Table 13: Additional ablation results on unsupervised CFG. The sampling computation in each column is the same.

Generated sentence of MDM on MT-Bench. We present some answers generated from MDM in Fig. (4-6).

### C.4 Additional Results of Reverse Curse

Complementary explanation.  highlight that models trained with less dependence on the precise sequence of tokens can successfully mitigate the reverse curse, which serves as a complementary explanation for the findings of Table 5.

In the NameToDescription test data for the same direction of Table 5, the T5 model outperforms MDM in BLEU scores but lags in exact match accuracy. This is because the T5 model tends to produce responses that are similar to the ground truth but differ slightly in a few words. It is worth noting that reverse question data shows a larger divergence between the training and testing sets, which accounts for the performance decline of MDM in the reverse task.

Additional results. Tab 14 shows the effectiveness of the unsupervised CFG on the reverse curse.

Figure: Figure 3: Scaling properties of MDMs on language understanding tasks. The x-axis represents the validation loss, while the y-axis indicates the accuracy.
Refer to caption: x5.png

Table: Table 14: Effectiveness of unsupervised CFG on reverse curse. The unsupervised CFG enhances the performance of MDM on the reverse queries.

## Appendix D Evaluation Metrics

In this section, we provide an overview of the benchmarks used in Sec. 5 and show some cases from these benchmarks in Tab. 15.

ARC-Easy. A subset of the AI2 Reasoning Challenge that focuses on elementary-level science questions to evaluate the model’s reasoning ability through basic scientific concepts.

BoolQ. A yes-or-no question-answering dataset designed to evaluate the model’s ability to answer questions based on a given passage.

HellaSwag. A metric assesses the model’s commonsense reasoning ability by completing a given sentence with one of four options.

OpenBookQA. A question-answering dataset, modeled after open-book exams, is designed to assess a model’s understanding of a subject by requiring multi-step reasoning and the integration of additional commonsense knowledge.

PIQA. Physical Interaction Question Answering is a metric that evaluates physical reasoning ability by asking models to select the best solution to a given problem involving everyday physical scenarios.

SIQA. Social Interaction Question Answering is a benchmark for commonsense reasoning and is established by presenting scenarios that require reasoning about social interactions and the motivations behind human behavior.

RACE. ReAding Comprehension Dataset From Examinations was designed to evaluate reading comprehension ability by understanding and interpreting text at a high school level.

LAMBADA. A dataset to evaluate models’ capabilities in text understanding through a final single-word prediction task based on a given context.

GSM8K. GSM8K (Grade School Math 8K) is a high-quality dataset of grade school math word problems, created to enable question answering involving multi-step mathematical reasoning.

Table: Table 15: Examples from language understanding benchmarks.

Figure: Figure 4: Generated sentence of MDM on the MT-Bench (case 1).

Figure: Figure 5: Generated sentence of MDM on the MT-Bench (case 2).

Figure: Figure 6: Generated sentence of MDM on the MT-Bench (case 3).