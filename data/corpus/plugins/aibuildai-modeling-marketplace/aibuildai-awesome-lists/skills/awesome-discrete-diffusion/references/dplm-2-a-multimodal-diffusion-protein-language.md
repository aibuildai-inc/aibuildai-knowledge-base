---
arxiv_id: "2410.13782"
title: "DPLM-2: A Multimodal Diffusion Protein Language Model"
year: 2024
source: arxiv2md
---

## Abstract

Abstract Proteins are essential macromolecules defined by their amino acid sequences, which determine their three-dimensional structures and, consequently, their functions in all living organisms. Therefore, generative protein modeling necessitates a multimodal approach to simultaneously model, understand, and generate both sequences and structures. However, existing methods typically use separate models for each modality, limiting their ability to capture the intricate relationships between sequence and structure. This results in suboptimal performance in tasks that requires joint understanding and generation of both modalities.
In this paper, we introduce DPLM-2, a multimodal protein foundation model that extends discrete diffusion protein language model (DPLM) to accommodate both sequences and structures.
To enable structural learning with the language model, 3D coordinates are converted to discrete tokens using a lookup-free quantization-based tokenizer.
By training on both experimental and high-quality synthetic structures, DPLM-2 learns the joint distribution of sequence and structure, as well as their marginals and conditionals.
We also implement an efficient warm-up strategy to exploit the connection between large-scale evolutionary data and structural inductive biases from pre-trained sequence-based protein language models.
Empirical evaluation shows that DPLM-2 can simultaneously generate highly compatible amino acid sequences and their corresponding 3D structures eliminating the need for a two-stage generation approach.
Moreover, DPLM-2 demonstrates competitive performance in various conditional generation tasks, including folding, inverse folding, and scaffolding with multimodal motif inputs, as well as providing structure-aware representations for predictive tasks.

## 1 Introduction

Proteins are macromolecules that execute crucial roles in every living organism.
They are characterized by their amino acid sequences and three-dimensional structure, where the sequence determines the structure, which in turn governs the
protein’s function.
Generative modeling for proteins has made significant strides in recent years.
Among them, diffusion models  exhibit great success in protein structure-based generative modeling .
Meanwhile, large-scale protein language models , trained on evolutionary-scale sequence database, have become one of the most important cornerstones in sequence-based foundation models for protein sequence representation learning and generation.
Remarkably, DPLM , a discrete diffusion  based protein language models, has exhibited the state-of-the-art performance in both sequence generation and understanding, addressing a wide range of sequence-oriented applications.

Many protein engineering applications, e.g., motif-scaffolding  and antibody design , require jointly determine both structure and sequence.
However, the aforementioned approaches mostly employ generative models for one modality (either sequence or structure) and resort to separate models  for the other.
This highlights the pressing need for multimodal protein generative models that can integrate both sequence and structure, enabling a more comprehensive understanding of protein behaviors and functions.
This, therefore, raises the following question:

Can we build a multimodal protein foundation model to simultaneously
model, understand, and generate both sequences and structures?

To pursue this goal, Multiflow  is a recent effort for structure-sequence co-generation that incorporates sequences into structure-based generative models using multimodal flow matching.
Despite its impressive structure generation capability, Multiflow exhibits suboptimal performance in co-generating structurally-compatible sequences and consequently resorts to instance-level knowledge distillation from ProteinMPNN .
Furthermore, it completely falls short in protein folding for given sequences, showing Mulitflow’s inadequacy in sequence understanding.
We argue that this bottleneck arises from the absence (co-)evolutionary inductive bias derived from massive pre-training from sequence database, as prior studies have demonstrated that the evolutionarily-informed representations learned by pre-trained protein language models implicitly capture structural information enables direct structure prediction .
As a consequence, the limitation in sequence understanding and generation renders Multiflow inadequate as a multimodal protein generative foundation.

Figure: Figure 1: Overall illustration of DPLM-2. (A) Structure tokenization consists of a GVP-based encoder to yield invariant backbone geometric features, a lookup-free quantizer (LFQ) to discretize encoded structural features into structure tokens within a codebook, and an IPA-based decoder as de-tokenizer to convert structure tokens back to backbone atomic coordinates. (B) Multimodal learning and generation of protein structure and sequence with DPLM-2. (C) Various applications of DPLM-2 as a protein foundation model: (1) unconditional protein sequence-structure mixed-modal co-generation; (2) protein sequence-structure joint representation for predictive tasks; (3) structure prediction; (4) fixed-backbone sequence generation; (5) conditional protein generation with structure-sequence mixed-modal input and output.
Refer to caption: x1.png

Inspired by the connection between evolutionary knowledge and spatial interactions, we deem that sequence-based generative language models like DPLM, with their strong sequence generation and predictive abilities, hold great promise as a foundation for multimodal learning for proteins.
Despite its exciting potential, this approach presents two key challenges: (1) language models cannot directly handle continuous data like structure; and (2) language models heavily necessitate sufficient scale of data and compute resources while structure data is much smaller compared to sequence databases.

In this paper, we address the aforementioned questions by introducing DPLM-2, a multimodal protein foundation model that advances the state-of-the-art discrete diffusion-based protein language model (i.e., DPLM) to accommodate both sequences and structures.
By training on both experimental and high-quality synthetic structures, DPLM-2 learns the joint distribution of sequence and structure, as well as their marginals and conditionals.
We present several key recipes to facilitate multimodal learning in DPLM-2:
(1) the core difficulty lies in enabling the language model to learn structural information, which is challenging and remains elusive, for which we develop a lookup-free quantization  structure tokenizer to convert 3D coordinates to discrete tokens and vice versa (Fig. 1A, §3.3);
(2) we implement an efficient warm-up strategy to exploit the connection between large-scale evolutionary data and structural inductive biases from pre-trained sequence-based DPLM (Fig. 1B, §3.2);
and (3) we also address the exposure bias problem in discrete diffusion for sequence learning  by a self-mixup training strategy that leads to enhanced generation quality and diversity.

We highlight our main contributions and findings as follows:

- (i)
We present DPLM-2, a multimodal protein generative language model that aims to simultaneously model, understand and generate protein structure and sequence.
We show that it can be fairly efficient and effective to obtain a mulitmodal protein model with moderate amount of high-quality data, a decent structure tokenizer and publicly-accessible sequence-only pre-trained language models.
- (ii)
As a mulitmodal generative model, DPLM-2 enables unconditional co-generation of designable and diverse proteins that guarantees consistency between structure and sequence (Fig. 1C(1)).
Our empirical evaluation shows that DPLM-2 attains competitive co-generation performance compared to structure-based generative approaches, while the proteins generated by DPLM-2 have a better alignment with the characteristics of natural proteins in secondary structure statistics (§4.1).
- (iii)
In addition, DPLM-2 supports various conditional generation tasks by its multimodal nature, ranging from (sequence-conditioned) folding (Fig. 1C(3), §4.2), (structure-conditioned) inverse-folding (Fig. 1C(4), §4.3), to more successful motif-scaffolding given multimodal motif conditioning (Fig. 1C(5), §4.4).
- (iv)
Last but not least, we demonstrate that the structure-aware protein representation learned by DPLM-2 brings additional benefit for a range of protein predictive tasks (Fig. 1C(2), §4.5).

##### Concurrent work.

During the development of DPLM-2, we became aware of the recently proposed multimodal generative protein language model, ESM3 , which also jointly models tokenized structure and sequence using a generative masked language model. While both models aim for similar goals, DPLM-2 differs from ESM3 in several key aspects:
(1) Multimodal protein generation: DPLM-2 treats structure and sequence modalities equally by design and emphasizes the simultaneous co-generation of compatible protein sequence and structure, whereas ESM3 is a sequence-first model (other modalities are subject to dropout during training) and generates in cascaded modality-by-modality manner.
(2) Data and compute efficiency:
ESM3 seeks to perform mulimodal pre-training from scratch using a huge amount of synthetic data, with modal size ranging from 1.4B to 98B.
With strict license and absence of training infrastructure, this prohibits community from replicating for customized purposes.
In contrast, DPLM-2 leverages much smaller datasets (PDB + SwissProt) and builds on open-source, pre-trained sequence-based DPLM (150M/650M/3B), which leverages DPLM’s learned evolutionary knowledge and inherits strong sequence understanding and generation capabilities.
We are also committed to open-source our models, training and inference code to democratize multimodal generative protein LMs to benefit the community.
Overall, we believe DPLM-2 provides unique contributions to the community.

## 2 Preliminaries

### 2.1 Generative Modeling for Protein

**Table 1: Generative tasks w.r.t. structure & sequence.**
| task | objective |
| --- | --- |
| folding | $p_{\theta}(\mathbf{x}|\mathbf{s})$ |
| inv-folding | $p_{\theta}(\mathbf{s}|\mathbf{x})$ |
| seq. gen. | $p_{\theta}(\mathbf{s})$ |
| struct. gen. | $p_{\theta}(\mathbf{x})$ |
| seq-struct co-gen. | $p_{\theta}(\mathbf{s},\mathbf{x})$ |

The aim of generative protein modeling is to estimate the underlying distribution $\mathrm{prot}\sim q(\mathrm{prot})$ of the protein data of our interest by learning a probabilistic model $p_{\theta}(\mathrm{prot})$.
Here $\mathcal{\mathrm{prot}}=(r_{1},r_{2},\dots,r_{L})$ denotes a protein with $L$ residues, where each residue $r_{i}=(s_{i},x_{i})$ is represented by two major modalities, i.e., $s_{i}\in\{0,1\}^{|\mathcal{S}|}$ is a categorical variable for its amino acid type in $\mathcal{S}=\{1,...,20\}$, and $x_{i}\in\mathbb{R}^{N_{\text{atoms}}\times 3}$ is the real-value Cartesian coordinates of its residue atoms (we only consider backbone atoms herein, i.e., $[\text{N},\text{C}_{\alpha},\text{C},\text{O}]$ with $N_{\text{atoms}}=4$).
Namely,

$$ $\displaystyle p_{\theta}(\mathrm{prot})=p_{\theta}(s_{1},s_{2},\dots,s_{L},~{} x_{1},x_{2},\dots,x_{L})=p_{\theta}(\mathbf{s},\mathbf{x})$ $$

As a result, most of protein tasks can be viewed as specifying their input conditioning and output between these two modalities (Tab. 1), including
(1) sequence-conditioned structure prediction ,
(2) structure-conditioned sequence generation ,
(3) sequence learning or generation  ,
(4) structure generation ,
and (5) sequence-structure co-generation .
These further enable various conditional applications by allowing single or mixed-modal conditioning for partial generation, e.g., motif-scaffolding and antibody design.

### 2.2 Diffusion Protein Language Model (DPLM)

Language models (LMs), typically parameterized by Transformers  have become the *de facto* choice dominating different domains with scalable and performing expressiveness .
Among them, protein LMs have been serving as one of the AI foundation for protein sequence learning  and generation .

Diffusion protein language model , in particular, shows excelling performance in both generation and representation learning of protein sequences.
DPLM is grounded in absorbing discrete diffusion framework , which is characterized by a forward and backward Markov process.
Let $\texttt{Cat}(\mathbf{x};\mathbf{p})$ be a categorical distribution on protein sequence $\mathbf{y}$ parameterized by a vector $\mathbf{p}$ on $(|\mathcal{V}|-1)$-dimensional probability simplex.
The forward process of discrete diffusion defines a Markov process governed by the transition kernel
$q(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)})=\texttt{Cat}\big{(}\mathbf{x}^{(t)};
\beta_{t}\mathbf{x}^{(t-1)}+(1-\beta_{t})\mathbf{q}_{\text{noise}}\big{)}$
that gradually perturb the data $\mathbf{x}^{(0)}\sim q(\mathbf{x}^{(0)})$ into a stationary distribution $\mathbf{x}^{(T)}\sim\mathbf{q}_{\text{noise}}$.
For absorbing diffusion, $\mathbf{q}_{\text{noise}}$ is the point mass with all of the probability on the absorbing (mask) state.
The learned backward process $p_{\mathbf{\theta}}(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)})$ reversely denoises the $\mathbf{x}^{(T)}$ towards the data distribution $\mathbf{x}^{(0)}$, which is typically optimized by the variational bound of the log-likelihood :

$$ $\displaystyle\mathbb{E}_{q(\mathbf{x}^{(0)})}\big{[}\log p_{\theta}(\mathbf{x} ^{(0)})\big{]}\geq\mathbb{E}_{q(\mathbf{x}^{(0:T)})}\bigg{[}\log\frac{p_{ \theta}(\mathbf{x}^{(0:T)})}{q(\mathbf{x}^{(1:T)}|\mathbf{x}^{(0)})}\bigg{]}$ $\displaystyle=\mathbb{E}_{q(\mathbf{x}^{(0)})}\Big{[}\log p_{\theta}(\mathbf{x }^{(0)}|\mathbf{x}^{(1)})+\textstyle{\sum_{t=2}^{T}}\underbrace{-\text{KL}\big {[}q(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)},\mathbf{x}^{(0)})\|p_{{\theta}}( \mathbf{x}^{(t-1)}|\mathbf{x}^{(t)})\big{]}\Big{]}}_{\mathcal{J}_{t}}+\text{ const.},$ $$

where $\mathcal{J}_{t}$ is the learning objective.
The learning objective of discrete diffusion can be further simplified into reweighted cross-entropies , resembling masked language modeling at arbitrary noise levels:

$$ $\displaystyle\mathcal{J}_{t}$ $\displaystyle=\mathbb{E}_{q(\mathbf{x}^{(0)})}-\text{KL}\big{[}q(\mathbf{x}^{( t-1)}|\mathbf{x}^{(t)},\mathbf{x}^{(0)})\|p_{{\theta}}(\mathbf{x}^{(t-1)}| \mathbf{x}^{(t)})\big{]}$ $\displaystyle=\mathbb{E}_{q(\mathbf{x}^{(0)})}\Big{[}\lambda^{(t)}\textstyle{ \sum_{1\leq i\leq L}}b_{i}(t)\cdot\log p_{\theta}(x^{(0)}_{i}|\mathbf{x}^{(t)} )\Big{]},$ (1) $$

where $\lambda^{(t)}$ is a weighting coefficient induced from the specific noising schedule.
For inference, DPLM is able to generate amino acid sequences by the reverse iterative denoising process of discrete diffusion  from the following distribution,

$$ $\displaystyle p_{\theta}(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)})=\textstyle\sum_{ \tilde{\mathbf{x}}^{(0)}}q(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)},\tilde{\mathbf{ x}}^{(0)}p_{\theta}(\tilde{\mathbf{x}}^{(0)}|\mathbf{x}^{(t)}).$ $$

Specifically, at time $t$, it first generates $\tilde{\mathbf{x}}^{(0)}$ from $p_{\theta}(\cdot|\mathbf{x}^{(t)})$, then a less noisy $\mathbf{x}^{(t-1)}$ is sampled by $q(\cdot|\mathbf{x}^{(t)},\mathbf{x}^{(0)}=\tilde{\mathbf{x}}^{(0)})$.
Within absorbing diffusion, the generation process can be viewed as an iterative mask-predict approach.
For sequence representation for predictive tasks, it can be obtained by simply letting DPLM take the sequence as input.

## 3 DPLM-2: A Multimodal Diffusion Protein Language Model

### 3.1 Overview

Fig. 1 illustrates DPLM-2’s overall architecture.
DPLM-2 is built on the state-of-the-art sequence-based generative protein LM, i.e., DPLM , using a discrete diffusion probabilistic framework to concurrently model both protein sequences and their corresponding structures.
To facilitate structure learning in language models, we introduce a token-based representation for protein structure via a tokenizer that converts $\mathbf{x}\in\mathbb{R}^{L\times N_{\text{backb}}\times 3}$, the 3D coordinates of the protein backbone into a discrete structure token sequence, denoted as $\mathbf{z}=(z_{1},z_{2},\dots,z_{L})\in\{0,1\}^{L\times|\mathcal{Z}|}$, where each token $z_{i}$ represents a local structural element of the $i$-th residue.
Given tokenized structure, DPLM-2 processes mulitmodal input by concatenating the structure token sequence $\mathbf{z}$ with the corresponding amino acid sequence $\mathbf{s}$ for the same protein.
Notably, there exists a position-by-position correspondence between $\mathbf{z}$ and $\mathbf{s}$, where $z_{i}$ and $s_{i}$ refer to the two modalities of the $i$-th residue, respectively.
To reinforce this correspondence, we assign identical position encodings to both $z_{i}$ and $s_{i}$, thereby ensuring that structural and sequence information is aligned at the residue level.

To train DPLM-2, we leverage a high-quality dataset comprising 20K clustered experimental structures from the Protein Data Bank (PDB)  and 200K predicted structures from the AFDB SwissProt split , with length $<512$.
During training, DPLM-2 is tasked with denoising the input sequence across a spectrum of noise levels, ranging from fully noisy to completely clean.
The multimodal training objective of DPLM-2 is derived from Eq. (1) as,

$$ $\displaystyle\mathcal{J}_{t}$ $\displaystyle=\mathbb{E}_{q(\mathbf{x}^{(0)},\mathbf{s}^{(0)}),\mathbf{z}^{(0) }\leftarrow\textit{tokenize}(\mathbf{x}^{(0)})}\Big{[}\lambda^{(t)}\textstyle{ \sum_{1\leq i\leq L}}b_{i}(t)\cdot\log p_{\theta}(z^{(0)}_{i},s^{(0)}_{i}| \mathbf{z}^{(t)},\mathbf{s}^{(t)})\Big{]},$ $$

where $\log p_{\theta}(z_{i},s_{i}|\cdot)=\log p_{\theta}(z_{i}|\cdot)+\log p_{\theta
}(s_{i}|\cdot)$ by assuming conditional independence. By learning $p_{\theta}(\mathbf{z}^{(t-1)},\mathbf{s}^{(t-1)}|\mathbf{z}^{(t)},\mathbf{s}^{
(t)})$, the model enables the simultaneous generation of highly correlated protein structures and sequences. This eliminates the need for a cascaded generation paradigm, allowing us to derive both the protein’s structure and sequence in a single step.

To further enhance DPLM-2’s ability to differentiate between structure and sequence, noising level for each modality is subjected to distinct scheduler, denoted as $t_{\mathbf{z}}$ and $t_{\mathbf{s}}$, respectively.
This facilitates a more comprehensive understanding of the relationships between protein sequences and their corresponding structures.
This design also allows us to explore arbitrary combinations of $(t_{\mathbf{z}},t_{\mathbf{s}})$, thus providing flexible sampling options, including sampling from the marginals of each modality and conditionals between them for various applications (Fig. 1C).
Furthermore, we also identify the exposure bias issue in discrete diffusion for sequence learning , and mitigate this by proposing a self-mixup strategy inspired by scheduled sampling, which improves both generation quality and diversity (see §A.1).

### 3.2 Efficient Warm-up from Pre-trained Sequence-based DPLM

Protein sequences encode critical evolutionary information, reflecting co-evolutionary processes where residue pairs mutate together and often interact in 3D space, offering insights for predicting protein folding . further showed that protein language models trained on large-scale evolutionary data implicitly capture this information, which can facilitate structure prediction.
Motivated by the link between evolutionary knowledge and structural interactions, we propose to built DPLM-2 with an efficient warmup from pre-trained sequence-based DPLM, to make the most of established evolutionary information for protein structure modeling,
Since our structure dataset is significantly smaller than UniRef50 sequence database (200K vs. 45M),
enabling efficient fine-tuning of the pre-trained model.
we want to keep the sequence knowledge intact and reduce the risk of catastrophic forgetting, we apply LoRA  to limit too much deviation to the original parameters.
This approach not only lowers training costs compared to starting from scratch but also effectively transfers valuable evolutionary information.

### 3.3 Learning Structure Tokenization

The core difficulty of achieving a mulimodal protein LM lies in enabling the language model to learn structural information, which is challenging and remains elusive,
Tokenizing continuous data modalities into discrete representations  has gained attraction across domains like image synthesis due to its ability to capture compact, meaningful information, enabling effective compression and efficient generation, especially with sequence-based models like Transformers.
Recent efforts have applied this approach to protein structure coordinates .
This allows language models to better learn the composition of local structural elements. However, how to learn an effective structure tokenizer remains an active research question.

Figure: Figure 2: Reconstruction and secondary structure correspondence of structure tokenizers.
Refer to caption: x2.png

Structure tokenization under a typical VQ-VAE  framework can be summarized as follows:

$$ $\mathbf{x}\xrightarrow{\text{encoder}}\mathbf{e}\xrightarrow{\text{quantizer}} \mathbf{z}\xrightarrow{\text{decoder}}\tilde{\mathbf{x}},$ $$

where (1) a structure encoder encodes backbone 3D coordinates $\mathbf{x}\in\mathbb{R}^{L\times N_{\text{backb}}\times 3}$ into invariant features $\mathbf{e}\in\mathbb{R}^{L\times d_{\text{quant}}}$, (2) a quantizer converts $\mathbf{e}$ into $\mathbf{z}$ of $L$ discrete tokens where $z_{i}\in\{0,1,\ldots,|\mathcal{Z}|\}$ given a finite-size codebook $\mathcal{Z}$; and (3) a structure decoder reconstructs 3D coordinates $\tilde{\mathbf{x}}$ from the discrete tokens.
We utilize a GVP-based  structure encoder from pre-trained GVP-Transformer  and a IPA-based  structure decoder.
In terms of quantizer, our preliminary experiment showed that conventional VQ-VAE pretty much struggles in training.
To mitigate this, we instead adopts Lookup-Free Quantizer (LFQ) from the currently best visual tokenizer  to protein structure tokenization.
Specifically, the latent space of LFQ is decomposed as the Cartesian product of single-dimensional binary variables, as $\mathbb{C}=\times_{k=1}^{\log_{2}|\mathcal{Z}|}\mathcal{C}_{k}$, where $\mathcal{C}_{k}=\{-1,1\}$.
Given the encoded feature $\mathbf{e}=\text{encoder}(\mathbf{x})\in\mathbb{R}^{L\times\log_{2}|\mathcal{Z
}|}$, each dimension (indexed by $k$) of the quantized representation $\mathtt{quant}(e_{i})$ is obtained from:

$$ $\mathtt{quant}(e_{i})[k]=\mathcal{C}_{i,k}=\mathtt{sign}(e_{i}[k])=-\mathbf{1} \{z_{i}[k]\leq 0\}+\mathbf{1}\{e_{i}[k]>0\}.$ $$

As such, with LFQ, the token indices for $\mathbf{z}=\{z_{1},z_{2},...,z_{i},...,z_{L}\}$ is given by:

$$ $z_{i}=\mathtt{index}(\mathtt{quant}(e_{i}))=\textstyle\sum_{k=1}^{\log_{2}| \mathcal{Z}|}2^{k-1}\mathbf{1}\{e_{i}[k]>0\},~{}\forall z_{i}\in\mathbf{z}.$ $$

The LFQ-based structure tokenizer is trained on the same structure dataset as mentioned before, using a combination of reconstruction, commitment, and entropy regularization losses, similar to standard VQ-VAE.
Here FAPE loss  is used as the primary reconstruction loss.

##### Evaluation.

As shown in Fig. 2A, LFQ significantly outperforms VQ-VAE regarding reconstruction accuracy while training of LFQ is much faster than VQ-VAE (2 vs. 15 days on 8 A100s).
Increasing codebook size leads to improved reconstruction while a codebook size of 8192 achieves the best compression-reconstruction trade-off.
Meanwhile in Fig. 2B, we observe a strong correlation between structure tokens and secondary structures. For instance, a lot of structure tokens concentrated at the alpha helix and beta sheet vertices, while some tokens lie between regions. This suggests that structure tokens the fine-grained structural elements in backbone local environment.

## 4 Experiments

In this section, we evaluate DPLM-2 on various generative and understanding scenarios, including unconditional protein generation (structure, sequence, and structure-sequence co-generation, §4.1), and a variety of conditional tasks, such as folding (§4.2), inverse folding (§4.3) and motif-scaffolding (§4.4), and a series of protein predictive tasks (§4.5).

Figure: Figure 3: Evaluation of DPLM-2 on unconditional structure-sequence co-generation. Here for designability of co-generated proteins, we use ESMFold to obtain refolded structure of DPLM-2-generated sequence and measure the structural similarity between DPLM-2-generated structure and the refolded structure, which aims to measure the compatibility of the co-generated structure and sequence pairs.
Refer to caption: x3.png

### 4.1 Unconditional Protein Generation

The goal of unconditional protein generation is to produce both the 3D structure and amino acid sequence. Typically, this is done using a cascaded approach: either generating the structure first and then use another model to predict the sequence, or vice versa. Here, we focus on generating structure and sequence simultaneously. We evaluate DPLM-2 on both cascaded and simultaneous generation across three tasks: unconditional structure generation, unconditional sequence generation, and structure-sequence co-generation.

Following Multiflow , we evaluate the generated proteins in terms of quality, novelty and diversity.
Quality is measured through designability (structure’s ability to fold into a valid sequence) and foldability (sequence’s ability to fold into a reasonable structure). Designability is assessed by folding the generated sequence with ESMFold , then using sc-TMscore and sc-RMSD with the co-generated structure to evaluate similarity.
Foldability is evaluated via ESMFold, with pLDDT $>$ 70 considered plausible.
Novelty is assessed by comparing generated structures to known ones in PDB using TMScore (pdb-TM), with lower values indicating greater novelty.
Diversity is measured by calculating pairwise TMscore (inner-TM), where lower scores indicate more dissimilarity.
The number of clusters identified by FoldSeek  also quantifies diversity, normalized by the total number of structures.

#### 4.1.1 DPLM-2 Enables High-quality, Diverse and Novel Protein Sequence and Structure Generation

Tab. 2 and Fig. 3
present the results of DPLM-2 for unconditional protein generation.
We highlight our key findings in the following aspects:

(1) DPLM-2 can generate diverse and highly-plausible protein with simultaneous structure-sequence co-generation.
We sampled 100 proteins for each length in 100, 200, 300, 400, and 500.
Fig. 3A/B demonstrates that DPLM-2 can sample sequence and structures with high designability across various lengths, with most sc-TM values exceeding 0.9, with diverse structure clusters.
Fig. 3D shows that the novelty of sampled proteins, measured by pdb-TM, generally increases with longer protein lengths.
In addition, DPLM-2 can generate with both modalities simultaneously or a modality-by-modality.
As shown in Tab. 2, the co-generation performance exhibit highest scTM, suggesting that co-modeling indeed benefits protein generation.

(2) DPLM-2 can attains competitive performance with strong baselines on co-generation, as well as backbone-only and sequence-only generation, respectively.
As shown in Tab. 2, DPLM-2 achieves the strong sc-TM compared to strong baselines, approaching the quality of native structures from PDB.
We notice that ESM3-Open , which runs in a sequence-then-structure order, fails short of unconditional generation.
Compared to MultiFlow , DPLM-2 achieves comparable co-generation quality.
Notably, as also reported in , Multiflow falls short of sequence generation when directly trained from structures with native sequences, resulting in greatly degraded co-generation performance without data distillation from external inverse folding models (ProteinMPNN).
For reference, we also provide the result of Multiflow retrained using our training data, where its co-generation performance remains unsatisfying and lags behind DPLM-2, which suggests that DPLM-2 has advantages of directly and effectively learning from complex structure-sequence joint distribution.
Moreover, DPLM-2 can also only produce single modality if needed, where it matches the best competitive models in these settings respectively.
These results demonstrate DPLM-2’s effectiveness as a mulitmodal generative model.

(3) DPLM-2 generates longer proteins beyond training data.
As DPLM-2 is trained with a $512$ length cutoff, we are curious about its length extrapolation, and evaluate sampled proteins at lengths of $[600,700,800,900,1000]$.
As shown in Fig. 3F, notably, for proteins exceeding the maximum training length of 512, the pLDDT scores of sequences sampled by DPLM-2 are close to those of DPLM.
This suggests that DPLM-2 largely retains its sequence generation capability inherited from sequence pre-training in DPLM, leading to its capability of length extrapolation.

(4) Case study.
Fig. 3H shows some generated samples of DPLM-2 up to 700 residues, while in Fig. 3I we showcase that we can manipulate DPLM-2 to design symmetric oligomers by forcing to duplicate the predicted tokens with repetitive structure and sequence patterns.

**Table 2: Benchmarking comparison of unconditional protein generation, in terms of structure-sequence co-generation, backbone-only generation, and sequence-only generation. For each method, we generate $100$ samples for lengths in $[100,200,300,400,500]$. * denotes Multiflow variants retrained by us using different dataset – native PDB data without ProteinMPNN distillation and the same training data as DPLM-2 (i.e., PDB+SwissProt), respectively.**
|  | Quality | Novelty | Diversity |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | scTM ($\uparrow$) | scRMSD ($\downarrow$) | pLDDT ($\uparrow$) | avg. pdb-TM ($\downarrow$) | avg. inner-TM ($\downarrow$) | MaxCluster ($\uparrow$) |
| Structure-sequence co-generation. |  |  |  |  |  |  |
| Native PDB protein | 4.623 $\pm$ 5.688 | 0.904 $\pm$ 0.129 | – | – | – | – |
| ESM3-Open (1.4B, seq $\rightarrow$ struct) | 0.624 $\pm$ 0.232 | 24.180 $\pm$ 24.109 | – | 0.660 $\pm$ 0.000 | 0.410 $\pm$ 0.167 | 0.540 |
| MultiFlow w/ distillation (official ckpt) | 0.930 $\pm$ 0.098 | 3.208 $\pm$ 4.741 | 79.447 | 0.704 $\pm$ 0.000 | 0.468 $\pm$ 0.152 | 0.500 |
| MultiFlow w/o distillation | 0.750 $\pm$ 0.163 | 9.306 $\pm$ 8.499 | 65.861 |  |  |  |
| MultiFlow (retrained on our training data) | 0.871 $\pm$ 0.934 | 6.580 $\pm$ 6.258 | 62.624 |  |  |  |
| DPLM-2 (650M, seq $\rightarrow$ struct) | 0.907 $\pm$ 0.117 | 6.337 $\pm$ 9.403 | 82.246 | 0.653 $\pm$ 0.195 | 0.594 $\pm$ 0.270 | 0.651 |
| DPLM-2 (650M, struct $\rightarrow$ seq) | 0.921 $\pm$ 0.098 | 4.969 $\pm$ 6.735 | 81.910 | 0.637 $\pm$ 0.195 | 0.679 $\pm$ 0.288 | 0.575 |
| DPLM-2 (650M, co-generation) | 0.925 $\pm$ 0.085 | 3.899 $\pm$ 3.723 | 82.686 | 0.640 $\pm$ 0.204 | 0.703 $\pm$ 0.279 | 0.545 |
| Unconditional backbone generation. (sequence predicted by ProteinMPNN) |  |  |  |  |  |  |
| Native PDB struct. (seq. from PMPNN) | 0.969 $\pm$ 0.000 | 0.864 $\pm$ 0.000 | – | – | 0.282 $\pm$ 0.000 | 0.782 |
| FrameDiff | 0.818 $\pm$ 0.000 | 3.919 $\pm$ 0.000 | – | 0.668 $\pm$ 0.000 | 0.465 $\pm$ 0.000 | 0.252 |
| FoldFlow | 0.540 $\pm$ 0.000 | 7.965 $\pm$ 0.000 | – | 0.566 $\pm$ 0.000 | 0.411 $\pm$ 0.000 | 0.762 |
| RFDiffusion | 0.914 $\pm$ 0.000 | 1.969 $\pm$ 0.000 | – | 0.657 $\pm$ 0.000 | 0.363 $\pm$ 0.000 | 0.598 |
| DPLM-2 (650M) | 0.945 $\pm$ 0.082 | 4.451 $\pm$ 5.261 | – | 0.637 $\pm$ 0.195 | 0.679 $\pm$ 0.288 | 0.575 |
| Unconditional sequence generation. (structures predicted by ESMFold) |  |  |  |  |  |  |
| EvoDiff | – | – | 35.846 | 0.432 $\pm$ 0.106 | 0.366 $\pm$ 0.070 | 0.990 |
| DPLM (650M) | – | – | 83.252 | 0.541 $\pm$ 0.187 | 0.515 $\pm$ 0.222 | 0.735 |
| DPLM-2 (650M) | – | – | 82.246 | 0.662 $\pm$ 0.199 | 0.589 $\pm$ 0.268 | 0.700 |

Figure: Figure 4: Analysis regarding secondary structure of generated proteins. (A) Statistics of averaged proportions of secondary structures for proteins from different methods and PDB; (B) Secondary structure vs. designability; (C) Samples of Multiflow, PDB and DPLM-2, as well as their secondary structure distributions.
Refer to caption: x4.png

#### 4.1.2 DPLM-2 Generates Proteins That Resembles Natural Proteins

To further analyze the properties of different model, we examine their secondary structure distribution against natural proteins from PDB.

##### Proteins sampled by DPLM-2 have secondary structures most similar to natural proteins.

As seen in Fig. 4A, structure-based models like RFDiffusion and MultiFlow generate proteins with more helices and fewer sheets and loops than natural proteins in PDB. Protein language models like ESM3 and DPLM-2 show no strong bias towards alpha helices, but ESM3 tends to generate more loops. Among the methods, DPLM-2 produces the most natural-like secondary structure proportions, closely matching PDB proteins.
In Fig. 4C, proteins generated by MultiFlow contain many helices and become more globular as length increases, exhibiting idealized secondary structures.
In contrast, proteins generated from DPLM-2 resembles natural ones have more balanced structures, with fewer helices and more beta sheets and loops.
On the other hands, simplex plots in Fig. 4C shows that while MultiFlow’s proteins are clustered in helix-rich regions, DPLM-2’s proteins span a wider area similar to natural proteins, while it rarely samples proteins composed mostly of sheets and loops, which do occur in nature.
Additionally, Fig. 4B shows that the loop ratio has a significant impact on designability, where a higher proportion of loops will increase scRMSD, as loops are highly flexible. Thus, proteins with long loops, which DPLM-2 often generates, tend to have relatively high scRMSD, aligning with the results in Tab. 2.

#### 4.1.3 Ablation Study

In DPLM-2 training, we start with a warmup from the sequence-based pre-trained DPLM to exploit established evolutionary information and augment the data with high-quality AlphaFold-predicted structures from SwissProt (around 200K) and clustered PDB structures. This section evaluates the effects of sequence pre-training and data augmentation on unconditional protein generation.

**Table 3: Ablation study on the sequence pre-training and training data augmentation.**
| sequence<br>pre-training | synthetic<br>structures | length 100 | length 200 | length 300 | length 400 | length 500 |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scTM | clusters | scTM | clusters | scTM | clusters | scTM | clusters | scTM | clusters |  |  |
| ✗ | ✗ | 0.9241 | 20 | 0.8674 | 34 | 0.7667 | 33 | 0.5016 | 25 | 0.4511 | 25 |
| ✓ | ✗ | 0.9610 | 26 | 0.9349 | 47 | 0.9169 | 38 | 0.8643 | 35 | 0.7673 | 52 |
| ✗ | ✓ | 0.8988 | 27 | 0.9182 | 15 | 0.9343 | 13 | 0.8518 | 21 | 0.8288 | 31 |
| ✓ | ✓ | 0.9348 | 35 | 0.9428 | 40 | 0.9232 | 48 | 0.9260 | 40 | 0.9012 | 32 |

Tab. 3 demonstrates that sequence pre-training and data augmentation can significantly improve the designability and diversity, especially in generating long proteins (length $>300$).
We hypothesize that the limited number of long proteins in PDB leads to insufficient training. In contrast, sequence pretraining, which includes evolutionary data, is essential and can be transferred to improve protein structure modeling and generation quality. Additionally, this evolutionary information boosts sampling diversity. While increasing the amount of training data improves designability, it is less effective in enhancing diversity compared to sequence pretraining. By combining both strategies, we achieve the best overall performance, which forms the core of our training strategy.

### 4.2 Forward Folding (Sequence-conditioned Structure Prediction)

**Table 4: Structure prediction performance comparison between DPLM-2 and different baseline approaches on CAMEO 2022 datasets. $\dagger$: PVQD results are quoted from .**
| Models | CAMEO 2022 | PDB date split |  |  |
| --- | --- | --- | --- | --- |
| RMSD | TMscore | RMSD | TMscore |  |
| ESMFold | 3.99/2.03 | 0.85/0.93 | 2.84/1.19 | 0.93/0.97 |
| ^†PVQD | 4.08/1.95 | 0.81/0.88 | – | – |
| MultiFlow | 17.84/17.96 | 0.50/0.46 | 15.64/16.08 | 0.53/0.49 |
| ESM3 | 6.33/2.98 | 0.85/0.92 | 4.94/2.28 | 0.87/0.93 |
| DPLM-2 (150M) | 9.22/7.64 | 0.75/0.81 | 8.35/5.60 | 0.76/0.82 |
| w/ folding SFT | 7.66/4.37 | 0.80/0.86 | 6.00/3.41 | 0.83/0.88 |
| DPLM-2 (650M) | 7.37/4.89 | 0.79/0.86 | 5.67/3.33 | 0.83/0.88 |
| w/ folding SFT | 6.21/3.78 | 0.84/0.89 | 3.40/1.78 | 0.89/0.94 |
| DPLM-2 (3B) | 6.34/3.65 | 0.83/0.89 | 4.54/2.54 | 0.86/0.92 |
| w/ folding SFT | 5.71/3.23 | 0.85/0.90 | 3.15/1.69 | 0.90/0.95 |

The goal of folding is to predict the 3D structure for the given amino acid sequence .
As a mulitmodal generative model, DPLM-2 spontaneously enables protein structure prediction task (see Fig. 1C-3) given sequence as conditioning.
We assess DPLM-2 on CAMEO 2022 and a PDB data split used by Multiflow .
We utilize RMSD and TMscore between predicted and ground truth structure for evaluation, while DPLM-2 adopts argmax decoding for 100 sampling iterations.

##### Tab. 4 indicates that DPLM-2 can perform sufficiently good folding in a zero-shot manner.

Performance can be improved after further supervised fine-tuning (SFT) using folding objective ($\max_{\theta}\log p_{\theta}(\mathbf{z}|\mathbf{s})$).
Overall, DPLM-2 can outperform or on par with the strong baselines, while achieving close performance with ESMFold.
Furthermore, We observe that DPLM-2 with larger model scales can attain better results than smaller ones.
We suggest that DPLM-2 benefits from the evolutionary information inherited from DPLM pre-trained on the vast number of protein sequences, which can be transferred and leveraged into structure modeling.

### 4.3 Inverse Folding (Structure-conditioned Sequence Generation)

The goal of inverse folding is to find an amino acid sequence that can fold to a given backbone structure.
For evaluation, we employ amino acid recovery (AAR) for sequence evaluation, and we also assess the structure by self-consistency TM-score (scTM) between the native structure and the ESMFold-predicted structure of the generated sequence.

**Table 5: Comparison on inverse folding task.**
| Models | CAMEO 2022 | PDB date split |  |  |
| --- | --- | --- | --- | --- |
| AAR | scTM | AAR | scTM |  |
| MultiFlow | 32.28/33.58 | 0.87/0.94 | 37.74/37.59 | 0.94/0.96 |
| ESM3 | 47.06/46.24 | 0.90/0.95 | 49.50/49.42 | 0.94/0.97 |
| DPLM-2 (150M) | 45.22/46.12 | 0.87/0.93 | 48.83/47.96 | 0.89/0.95 |
| DPLM-2 (650M) | 49.01/50.10 | 0.88/0.93 | 54.80/53/07 | 0.91/0.96 |
| DPLM-2 (3B) | 52.36/53.72 | 0.89/0.95 | 61.67/57.91 | 0.92/0.96 |

##### DPLM-2 can generate reasonable sequences that fold into the given structures.

Tab. 5 presents that DPLM-2 can outperform or be on par with other co-generation models (MultiFlow, ESM3).
As the model size increases, the performance in terms of sequence recovery (AAR) and structural consistency (scTM) improves, revealing the same scaling law observed in the folding task.
We suggest that multimodal training effectively aligns the structure and sequence into the same space, such that DPLM-2 can yield the corresponding sequence without additional training.

### 4.4 Scaffolding with Mixed-modal Motif Conditioning

The objective of motif-scaffolding is to generate a suitable scaffold to preserve the structure of the given motif and maintain its original function.
We follow the experimental setting of , with 24 motif-scaffolding problems and we sample 100 scaffolds for each motif, where we
(1) first determine the length of scaffold, and then
(2) keep the motif segment unchanged and sample the scaffold part conditioned on the motif.
The scaffold length is sampled from a range provided by , and when there are multiple motifs, the order of motif segments is consistent with .
We provide the 3D structure and sequence of motif as input of DPLM-2.
As a multimodal model, we evaluate DPLM-2 using sequence-based, structure-based, and co-generation approaches.
A scaffold is considered successful if it satisfies both criteria
(1) overall designablity, which is successful when pLDDT $>70$ (for sequence-based models) or scTM $>0.8$, and
(2) motif-preseving, which is deemed successful when the predicted motif structure matches the native one with motif-RMSD $<$1Å.

Figure: Figure 5: Evaluation of motif-scaffolding w.r.t. success rate and num. of solved problems.
Refer to caption: x5.png

##### Fig. 5 reveals that DPLM-2 is capable of generate reasonable scaffolds for the given functional motifs.

In sequence-based, structure-based and co-generation evaluation, DPLM-2 can outperform or be on par with the corresponding approaches in most cases, solving more motif problem and achieving higher average success rate.
We compared to sequence-based method, DPLM-2 shows better performance since it allows structural input of motif, which is important for preserving motif’s structure hence the functions.
Remarkably, DPLM-2 attains comparable performance with RFDiffusion when only generating scaffold structure, while achieve better performance when simultaneously designing scaffold sequence and structure, outperforming ESM3.
Despite not experimentally verified, these results suggest that with DPLM-2, mulitmodal conditioning and generation could lead to more successful conditional protein design.

**Table 6: Performance on various protein predictive downstream tasks. $\dagger$: benchmarked results are quoted from .**
| Models | Thermostability | HumanPPI | Metal Ion Binding | EC | GO | DeepLoc |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MF | BP | CC | Subcellular | Binary |  |  |  |  |  |
| Spearman’s $\rho$ | Acc ($\%$) | Acc ($\%$) | Fmax | Fmax | Fmax | Fmax | Acc ($\%$) | Acc ($\%$) |  |
| ^†SaProt (650M) | 0.724 | 86.41 | 75.75 | 0.884 | 0.678 | 0.356 | 0.414 | 85.57 | 93.55 |
| ^†MIF-ST | 0.694 | 75.54 | 75.08 | 0.803 | 0.627 | 0.239 | 0.248 | 78.96 | 91.76 |
| ESM2 (650M) | 0.691 | 84.78 | 71.88 | 0.866 | 0.676 | 0.344 | 0.402 | 83.68 | 92.28 |
| DPLM (650M) | 0.695 | 86.41 | 75.15 | 0.875 | 0.680 | 0.357 | 0.409 | 84.56 | 93.09 |
| DPLM-2 (650M) | 0.714 | 84.44 | 74.28 | 0.878 | 0.680 | 0.359 | 0.411 | 82.98 | 93.64 |

### 4.5 Evaluation of Protein Representation Learning

Directly access to structure information is supposed to benefit downstream protein predictive tasks.
To inspect this, we evaluate DPLM-2 on a variety of protein predictive tasks utilizing the dataset provided by SaProt , where we provide tokenized protein structure tokens along with the protein sequences to DPLM-2.

**Table 7: Performance without large-scale sequence pre-training.**
| Models | DeepLoc |
| --- | --- |
| Subcellular |  |
| Acc ($\%$) |  |
| DPLM (650M) | 63.49 |
| DPLM-2 (650M) | 66.77 |

##### DPLM-2 can perform multimodal representation learning by leveraging both structure and sequence information.

Tab. 6 presents that DPLM-2 shows further improvement compared to sequence-only methods (ESM2, DPLM) on some tasks, indicating that DPLM-2 can leverage protein structures to generate better representations containing multimodal information for downstream tasks.
However, we find that DPLM-2 falls behind the state-of-the-art structure-aware protein LM, i.e., SaProt, in most tasks and even lags behind DPLM in certain tasks.
We hypothesize this is because the strutcure training data of DPLM-2, consisting of PDB and SwissProt, is smaller and differs from UniRef50, which DPLM is pretrained on, potentially causing catastrophic forgetting and suboptimal representation. To test this, we conducted an experiment on the DeepLoc subcellular task, where DPLM-2 underperforms compared to DPLM. As shown in Tab. 7, without large-scale sequence pretraining, DPLM-2 outperforms DPLM significantly, suggesting that: (1) Incorporating structure information enhances performance over sequence-only models. (2) Smaller datasets can lead to catastrophic forgetting, diminishing the benefits of large-scale pretraining.
As result, to further improve the predictive performance, one deserving direction is to exploit larger-scale predicted structures in our future work.

## 5 Discussions

In this paper, we introduce DPLM-2, a multimodal diffusion protein language model that understands, generates and reasons over protein structure and sequence, aiming to severe as a mulimodal foundation for protein.
Despite promising performance spanning protein co-generation, folding, inverse folding and conditional motif-scaffolding with mulimodal input and output, there remains several limitations deserving to be addressed.
(1) Structure data: Our findings indicate that while structure awareness may help with predictive tasks, the limited structure data constrains DPLM-2’s ability to learn robust representations. It is also important to account for longer protein chains and multimers in future studies.
(2) Trade-off of discrete latent representation: Tokenizing structure into discrete symbols facilitates multimodal protein language models and co-generation but may come at the cost of losing fine-grained structural details and control, such as precise atomic positions and inter-atomic distances.
Future work should aim to also integrate the strengths of data-space structure-based generative models into sequence-based mulitimodal language models to maximize the best of both worlds.

## Acknowledgement

We would like to thank Dr. Hang Li for insightful discussions on the project and feedback on the manuscript that help shape this study.
We thank Yi Zhou, Jing Yuan, Yilai Li, Yuning Shen, Wesley Hsieh and Daiheng Zhang for their valuable comments.

## Appendix A DPLM-2 Training

### A.1 Tackling Exposure Bias in Discrete Diffusion with Self-mixup Training Strategy

We find that discrete diffusion training will face the exposure bias problem , which means mismatch between training and inference.
The model is trained to denoise given the ground-truth context during training.
However, during inference, the model needs to denoise based on the predicted tokens, which may not be correct and inconsistent with the always-accurate context during training.
This may lead to error accumulation and negatively impact the generation performance.

To address this issue, we propose a self-mixup training paradigm for discrete diffusion model, enhancing the consistency between training and inference.
During training, we perform an additional forward pass, allowing the model to first make predictions and then denoise based on those predictions.

Tab. 8 shows that the self-mixup training strategy effectively enhances the diversity of samples.
We attribute this to the model producing more accurate logits during inference, leading to more diverse reasonable sampling paths instead of converging on the sampling paths with the highest probability, which results in more diverse proteins.

**Table 8: Ablation study on the self-mixup training strategy.**
| Mixup<br>strategy | length 100 | length 200 | length 300 | length 400 | length 500 |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scTM | clusters | scTM | clusters | scTM | clusters | scTM | clusters | scTM | clusters |  |
| ✗ | 0.9237 | 44 | 0.9180 | 53 | 0.9147 | 48 | 0.9059 | 42 | 0.8896 | 33 |
| ✓ | 0.8812 | 62 | 0.8820 | 62 | 0.9172 | 59 | 0.9099 | 54 | 0.8845 | 38 |

### A.2 Dataset

The training set of DPLM-2 is composed by experimental data, i.e., PDB , and high quality synthetic data, i.e., SwissProt .
We filter the SwissProt data by pLDDT $>$ 85.
After filtering, the overall training set contains approximately 200,000 proteins.
We limit the maximum length of the training set to 512.
For proteins longer than 512, we randomly crop it to 512.
We crop the low pLDDT (pLDDT $<$ 50) segments located at the both ends of proteins in the SwissProt dataset.
These segments are typically non-structural and may negatively impact the training results.
Moreover, we find that the length distribution of the training set is not balanced, where the number of proteins with length less than 100 is relatively small, leading to a suboptimal diversity among the short proteins.
Therefore, during training, we randomly crop long proteins to short proteins with a probability of 50% for each batch to improve the diversity.

### A.3 Hyperparameter

We train all models using AdamW optimizer  with $\beta_{1}$ = 0.9 and $\beta_{2}$ = 0.95.
We use a weight decay of 0.01 and gradient clipping of 0.5.
We employ 2K warmup steps until reaching the maximum learning rate, and utilize a linear decay scheduler to decay LR to 10% of the maximum learning rate by the end of training.
The maximum learning rate is 1e-4, and the overall training step is 100,000.
We utilize the pretrained DPLM as the parameter initialization, and the diffusion timestep is set to 500.
We train 150M DPLM-2 with 8 A100 GPUs for 3 days, while 650M with 16 A100 GPUs for 3 days and 3B with 16 A100 GPUs for a week.

## Appendix B Structure Tokenizer

The core difficulty of achieving a mulimodal protein LM lies in enabling the language model to learn structural information, which is challenging and remains elusive,
Tokenizing continuous data modalities into discrete representations  has gained attraction across domains like image synthesis due to its ability to capture compact, meaningful information, enabling effective compression and efficient generation, especially with sequence-based models like Transformers.
Recent efforts have applied this approach to protein structure coordinates .

### B.1 Dataset

Our structure tokenizers are trained using the same structure data as our mulitmodal language model, containing both experimental and high-quality structures, totaling 200K proteins.

### B.2 Model Architecture

As shown in Fig. 1A, the structure tokenizer in this paper consists of a structure encoder, quantizer, and structure decoder. The encoder is based on a pre-trained GVP-Transformer , with its parameters frozen during training. It transforms backbone structures into geometric features, which are projected onto a latent embedding using an MLP layer. For the quantizer, we adopt a lookup-free quantizer from a state-of-the-art video tokenizer , where the latent dimension is set to $\log_{2}|\mathcal{Z}|$, with $|\mathcal{Z}|$ as the codebook size. The structure decoder follows the IPA-based modules from AlphaFold2 , using 4 EvoFormer layers without MSA row attention, following ESMFold , to generate atomic positions from the structure tokens.

### B.3 Training

The structure tokenizer is trained using a standard VQ-VAE framework, with the objective including reconstruction loss, codebook commitment loss, and entropy regularization loss to ensure effective codebook utilization. For the reconstruction loss, we adopt the FAPE loss, violation loss, and distogram loss from AlphaFold2, measuring the difference between predicted and native structures. To further enhance the training, we introduce a sequence prediction head on top of the structure decoder’s final representation and minimize the cross-entropy against the native sequence.

## Appendix C Motif Scaffolding

### C.1 Evaluation Pipeline

We evaluate DPLM-2 in sequence-based, structure-based and co-generation ways. The overall illustration is shown in Fig. 6.

We focus on the two aspects: overall quality and motif part consistency.
The assessment of overall quality varies across different approaches. Specifically,
(1) For sequence-based method, we only take the generated sequence and utilize ESMFold to obtain the predicted structure, and the pLDDT score provided by ESMFold is used to assess overall quality.
(2) For structure-based method, we only take the generated structure, and then leverage ProteinMPNN to predict the sequence, followed by ESMFold to predict the structure, where overall quality is assessed by scTM.
(3) For co-generation method, we take both the generated structure and sequence, and predict structure given generated sequence with ESMFold, where scTM is calculated between generated structure and ESMFold predicted structure to evaluate overall quality.
Considering that the ground truth motif structure is given, we only utilize the ESMFold predicted structure to calculate motif-RMSD.

Figure: Figure 6: Sequence-based, structure-based and co-generation evaluation pipeline of motif-scaffolding.
Refer to caption: x6.png

### C.2 Result of Each Problem

Tab. 9 presents the result of each motif-scaffolding problem.
DPLM-2 achieves the best average success rate in each evaluation.
Compared with ESM3, DPLM-2 shows better results in 12 problems in co-generation evaluation and 10 problems in sequence-based evaluation.
Meanwhile, DPLM-2 outperforms RFDiffusion in 14 problems in structure-based evaluation.
This demonstrates that DPLM-2 can achieve strong performance under various evaluation methods.

We also find that taking the best result from 8 samples can bring significant improvement compared to 1 sample, especially in terms of success rate.
In the co-generation evaluation, DPLM2 with sampling 8 times improves the success rate of most of the problems by a large margin.
We hypothesize that sampling eight times largely alleviates errors caused by randomness in the sampling process, thereby producing a more suitable scaffold for the given motif.

**Table 9: Motif-scaffolding results of each problem. * means best result from 8 samples.**
|  | sequence-based | structure-based | co-generation |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EvoDiff | DPLM | ESM3 | DPLM2 | *RFDiffusion | *DPLM2 | ESM3 | DPLM2 | *DPLM2 |  |
| 1BCF | 0.00 | 0.00 | 0.89 | 0.01 | 1.00 | 0.07 | 0.23 | 0.01 | 0.05 |
| 1PRW | 0.61 | 0.83 | 0.96 | 0.86 | 0.08 | 0.96 | 0.54 | 0.84 | 0.95 |
| 1QJG | 0.00 | 0.00 | 0.02 | 0.03 | 0.00 | 0.00 | 0.03 | 0.02 | 0.05 |
| 1YCR | 0.02 | 0.38 | 0.41 | 0.77 | 0.74 | 0.93 | 0.18 | 0.53 | 0.98 |
| 2KL8 | 0.04 | 0.08 | 0.11 | 0.47 | 0.88 | 0.94 | 0.11 | 0.57 | 1.00 |
| 3IXT | 0.06 | 0.17 | 0.18 | 0.67 | 0.25 | 0.77 | 0.02 | 0.41 | 0.73 |
| 4JHW | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4ZYP | 0.00 | 0.00 | 0.03 | 0.16 | 0.40 | 0.51 | 0.08 | 0.10 | 0.64 |
| 5IUS | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5TPN | 0.00 | 0.00 | 0.03 | 0.00 | 0.61 | 0.06 | 0.01 | 0.00 | 0.00 |
| 5TRV_long | 0.00 | 0.00 | 0.19 | 0.00 | 0.37 | 0.08 | 0.19 | 0.00 | 0.07 |
| 5TRV_med | 0.00 | 0.00 | 0.16 | 0.03 | 0.24 | 0.07 | 0.16 | 0.02 | 0.19 |
| 5TRV_short | 0.00 | 0.00 | 0.01 | 0.07 | 0.04 | 0.10 | 0.01 | 0.03 | 0.11 |
| 5WN9 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 |
| 5YUI | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6E6R_long | 0.01 | 0.65 | 0.07 | 0.91 | 0.86 | 0.92 | 0.04 | 0.78 | 1.00 |
| 6E6R_med | 0.03 | 0.94 | 0.24 | 0.93 | 0.89 | 0.88 | 0.14 | 0.77 | 0.97 |
| 6E6R_short | 0.07 | 0.87 | 0.09 | 0.86 | 0.39 | 0.78 | 0.06 | 0.64 | 0.99 |
| 6EXZ_long | 0.00 | 0.01 | 0.32 | 0.61 | 0.76 | 0.63 | 0.13 | 0.44 | 0.95 |
| 6EXZ_med | 0.00 | 0.00 | 0.31 | 0.66 | 0.49 | 0.63 | 0.31 | 0.55 | 0.96 |
| 6EXZ_short | 0.00 | 0.00 | 0.31 | 0.66 | 0.39 | 0.41 | 0.28 | 0.58 | 0.87 |
| 7MRX_long | 0.00 | 0.02 | 0.36 | 0.23 | 0.09 | 0.32 | 0.37 | 0.20 | 0.73 |
| 7MRX_med | 0.00 | 0.31 | 0.65 | 0.28 | 0.11 | 0.31 | 0.59 | 0.22 | 0.70 |
| 7MRX_short | 0.00 | 0.34 | 0.68 | 0.26 | 0.02 | 0.41 | 0.74 | 0.24 | 0.88 |
| pass rate | 7/24 | 11/24 | 21/24 | 18/24 | 20/24 | 20/24 | 20/24 | 18/24 | 19/24 |
| avg. success rate | 0.04 | 0.19 | 0.25 | 0.35 | 0.40 | 0.42 | 0.18 | 0.29 | 0.53 |

## Appendix D Related Work

### D.1 Protein Language Models

There is growing interest in developing protein LMs at the scale of evolution, such as the series of ESM , TAPE ,
ProtTrans , PRoBERTa , PMLM ,
ProteinLM ,
PLUS ,
Adversarial Masked LMs ,
ProteinBERT ,
CARP  in masked language modeling (MLM) paradigm,
ProtGPT2  in causal language modeling paradigm, and several others .
These protein language models exhibit remarkable generalization ability on various downstream tasks and be able to capture evolutionary information about secondary and tertiary structures from sequences alone.
Meanwhile, recent study shows these models’ potency in revealing protein structures , predicting the effect of sequence variation on function , antibody infilling  and many other general purposes .
Simultaneously, demonstrate that the large scale protein LMs can generate de novo proteins by generalizing beyond natural proteins, both theoretically and experimentally validating their hypothesis in exhaustive detail, in which protein LMs demonstrate competency in designing protein structure despite being exclusively trained on sequences.

### D.2 Protein Structure Generative Models

Diffusion models have become popular tools in structural biology for protein generation, and their utility has been demonstrated across a range of generative tasks in recent years. , along with others, have introduced several diffusion model variants, each with its unique approach. For instance, while some models focus on generating the protein backbone by diffusing over protein coordinates, others, such as those proposed by , target inter-residue angles. and have developed models that handle both the position and orientation of residue frames.
RFDiffusion  is a model that assists in designing protein structures for specific functions, such as enzymes. It is versatile in protein design and has been used to create therapeutic proteins, with some designs being confirmed in the laboratory.
ProteinSGM  is a model that uses 2D matrices, which represent the distances and angles between protein parts, to create 3D protein structures for novel protein designs.
FoldingDiff  is a model that generates protein sequences expected to fold into a specific structure. These sequences are verified with prediction tools, although they have not been experimentally confirmed yet.
Chroma  is a model designed for creating large proteins and protein complexes, considering various constraints like distances and symmetry. It transforms a collapsed polymer into protein backbone and sequence more quickly than older methods, thereby allowing for the efficient generation of large structures.
Multiflow  develop mulitmodal flow matching for protein structure-sequence co-generation .
ProtPardelle  propose an all-atom generative approach for co-design.