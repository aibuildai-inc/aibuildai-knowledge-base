---
arxiv_id: "2305.14671"
title: "A Survey of Diffusion Models in Natural Language Processing"
year: 2023
source: arxiv2md
---

## Abstract

Abstract This survey paper provides a comprehensive review of the use of diffusion models in natural language processing (NLP). Diffusion models are a class of mathematical models that aim to capture the diffusion of information or signals across a network or manifold. In NLP, diffusion models have been used in a variety of applications, such as natural language generation, sentiment analysis, topic modeling, and machine translation. This paper discusses the different formulations of diffusion models used in NLP, their strengths and limitations, and their applications. We also perform a thorough comparison between diffusion models and alternative generative models, specifically highlighting the autoregressive (AR) models, while also examining how diverse architectures incorporate the Transformer in conjunction with diffusion models.
Compared to AR models, diffusion models have significant advantages for parallel generation, text interpolation, token-level controls such as syntactic structures and semantic contents, and robustness. Exploring further permutations of integrating Transformers into diffusion models would be a valuable pursuit. Also, the development of multimodal diffusion models and large-scale diffusion language models with notable capabilities for few-shot learning would be important directions for the future advance of diffusion models in NLP.

## 1 Introduction

Figure: Figure 1: The yearly number of both published and preprinted papers on diffusion models for NLP. For year 2023, the blue bar shows the number collected until the end of April 2023, and the dashed gray bar shows the estimated number for the whole year.

Diffusion models have shown remarkable performance in image generation and attracted huge attention in the field of artificial intelligence. Researchers have also adopted the models to the field of natural language processing (NLP) and have just started to explore their generative capabilities in the domain (Fig. [1](#S1.F1)). To date, diffusion models have been applied to a wide range of generative NLP tasks, such as unconditional text generation, controllable text generation, machine translation, and text simplification.

The main challenge in incorporating diffusion models into NLP is the discreteness of texts, which contrasts with the continuous space in which diffusion is modeled.
To address this challenge, researchers have introduced modifications to the models, and we categorize them into two approaches:

- •
Discrete diffusion models built on categorical distributions. This method generalizes diffusion process to the discrete domain by corrupting and refining sentences at the token level.
- •
Embedding diffusion models encode discrete texts into continuous space and perform Gaussian noising. As part of this method, additional embedding and rounding steps can be used in the forward and reverse processes, respectively, to convert tokens into embeddings.

In the following sections, we first introduce the general framework of vanilla diffusion models and the modified architecture for discrete state spaces in Section [2](#S2).
In Section [3](#S3), we classify the surveyed architectures into two aforementioned approaches (discrete vs embedding diffusion models), using specific criteria that have been proposed.
In Section [4](#S4), we conduct a detailed comparative analysis of diffusion models against other generative models in NLP domain. Based on empirical evidence, we highlight the advantages of diffusion models over autoregressive (AR) models, specifically in terms of parallel generation, text interpolation, token-level control, and robustness.
In addition, we explore how various surveyed architectures have incorporated the Transformer with diffusion models for NLP.
We highlight algorithms and techniques proposed for diffusion models in NLP in Section [5](#S5).
Finally, we discuss potential future directions that are both timely and worthy of exploration in Section [6](#S6).

## 2 General Framework

Traditionally, diffusion models have focused on continuous state spaces, but recent advancements have expanded their application to discrete state spaces. Discrete diffusion models operate with discrete variables, such as text or categorical data, which present distinct characteristics and challenges.

A key distinction is the treatment of noise. Continuous diffusion models employ additive Gaussian noise, while discrete diffusion models introduce discrete perturbations or transformations to modify the discrete states. This enables exploration of different states and enhances sample diversity.

Transition probabilities also differ between continuous and discrete diffusion models. Continuous models utilize stochastic differential equations, whereas discrete models define transition probabilities using conditional distributions. These distributions capture dependencies between current and previous states, facilitating information propagation and guiding the diffusion process in discrete state spaces.

##### Diffusion Models

Denoising diffusion probabilistic models (DDPMs) were initially introduced by and enhanced by . DDPMs employ a two-step process: adding Gaussian noise and performing a reverse process to restore the original data. developed DDPMs with an embedding function that maps discrete text to a continuous space, achieving comparable results to state-of-the-art generative models like generative adversarial networks (GANs). Subsequent works have further improved the quality and efficiency of DDPMs.

The forward process generates $X_{t+1}$ by adding noise to $X_{t}$, creating a dependency solely on $X_{t}$. This categorizes the diffusion process as a Markov process, where the noise level is determined by the variance $\beta_{t}\in(0,1)_{t=1}^{T}$. The expression for $q(x_{t}|x_{t-1})$ can be written as follows:

$$ $q(x_{t}|x_{t-1})=N(x_{t};\sqrt{1-\beta_{t}}\cdot x_{t-1};\beta_{t}\mathbf{I})$ (1) $$

By applying the reparameterization approach to depict $X_{t}$, where $a_{t}=1-\beta_{t}$, $z_{t}\sim N(0,1)$, $t\leq 0$, the subsequent result can be obtained:

$$ $x_{t}=\sqrt{\alpha_{t}}x_{t-1}+\sqrt{1-\alpha_{t}}Z_{t-1}$ (2) $$

When computing $q(x_{t}|x_{0})$, the joint probability distribution of $(x_{1:T}|x_{0})$ can be determined because it is established as a Markov chain:

$$ $q(x_{1:T}|x_{0})=\sum_{t=1}^{T}q(x_{t}|x_{t-1})$ (3) $$

Then we can express $x_{t}$ at arbitrary time step $t$ with reference to $x_{0}$ in a closed form, where $\bar{\alpha}_{t}=\alpha_{1}\alpha_{2}...\alpha_{t}$:

$$ $q(x_{T}|x_{0})=N(x_{t};\sqrt{\bar{\alpha}_{t}}x_{0};(1-\bar{\alpha}_{t}) \mathbf{I})$ (4) $$

For the reverse process, if we can determine the probability distribution of $x_{t-1}$ based on the given condition of $x_{t}$, i.e., if $q(x_{t-1}|x_{t})$ can be known, then we can iteratively sample random noise to generate an image or sentence. The challenge is to obtain $q(x_{t-1}|x_{t})$. To approximate it, we utilize $p_{\theta}(x_{t-1}|x_{t})$. Given that the added noise at each step is relatively small, we assume that $p_{\theta}(x_{t-1}|x_{t})$ follows a Gaussian distribution that can be modeled using a neural network. The reverse process can be expressed as follows:

$$ $p_{\theta}(x_{t-1}|x_{t})=N(x_{t-1};\mu(x_{t},t),\sum_{\theta}(x_{t},t))$ (5) $$

$$ $p_{\theta}(x_{0:T})=p(x_{T})\sum_{t=1}^{T}p_{\theta}(x_{t-1}|x_{t})$ (6) $$

Applying Bayes’ rule, we can express $q(x_{t-1}|x_{t},x_{0})$ in terms of the known forward conditional probabilities $q(x_{t}|x_{t-1},x_{0})$, $q(x_{t-1}|x_{0})$, and $q(x_{t}|x_{0})$. Our objective is to minimize the mean square error (MSE) loss between the KL divergence of the model $p_{\theta}$ and the true distribution $q$.

Figure: (a) Discrete Diffusion Models
Refer to caption: extracted/2305.14671v2/figures/s3_discrete_model.png

##### Diffusion models for discrete state spaces

For scalar discrete random variables with $K$ categories, where $x_{t}$ and $x_{t-1}$ take values from 1 to $K$, the forward transition probabilities can be represented using matrices. Let $[Q_{t}]_{i,j}=q(x_{t}=j|x_{t-1}=i)$. We can denote the one-hot representation of $x$ using a row vector, which can be expressed as follows:

$$ $q(x_{t}|x_{t-1})=Cat(x_{t};p=x_{t-1}Q_{t})$ (7) $$

In this context, $Cat(x;p)$ represents a categorical distribution over the one-hot row vector $x$, where the probabilities are determined by the row vector $p$. The term $x_{t-1}Q_{t}$ corresponds to a row vector-matrix multiplication. An assumption is made that $Q_{t}$ is independently applied to each pixel of an image or token in a sequence, and that the distribution $q$ factorizes over these higher dimensions as well. Therefore, we can express $q(x_{t}|x_{t-1})$ in terms of a single element. Starting from $x_{0}$, we can derive the following $t$-step marginal and posterior at time $t-1$, where $\bar{Q}_{t}=Q1Q2...Q_{t}$:

$$ $q(x_{t}|x_{0})=Cat(x_{t};p=x_{0}\bar{Q}_{t})$ (8) $$

$$ $q(x_{t-1}|x_{t},x_{0})=\frac{q(x_{t}|x_{t-1},x_{0})q(x_{t-1}|x_{0})}{q(x_{t}|x _{0})}$ (9) $$

The Markov property of the forward process ensures that $q(x_{t}|x_{t-1},x_{0})$ can be simplified to $q(x_{t}|x_{t-1})$. Similarly, assuming the reverse process $p_{\theta}(x_{t}|x_{t-1})$ also exhibits a factorized structure, considering the conditional independence of the image or sequence elements, we can derive the KL divergence between $q$ and $p_{\theta}$ by aggregating the probabilities across all possible values of each random variable.

## 3 A Survey of Diffusion Models in NLP

**Table 1: Comparison of discrete and embedding diffusion models.**
| Model | Tasks | Schedule | Sampling |
| --- | --- | --- | --- |
| Discrete Diffusion Models |  |  |  |
| Multinomial Diffusion | unconditional text generation, unsupervised spell-checking | Transition matrices | — |
| D3PM (Discrete Denoising Diffusion Probabilistic Models) | char-level text and image generation | Uniform Transition Matrices | — |
| Zero-shot Diffusion | machine translation | Partial Noising | Classifier-free conditional denoising |
| SUNDAE (Step-unrolled Denoising Autoencoders) | machine translation and unconditional text generation | Uniform Transition Matrices | Low-temperature sampling, Argmax-unrolled decoding, fewer token update |
| DiffusionBERT | unconditional text generation | Spindle | x0-parameterization |
| SSD-LM (Semi-autoregressive Simplex-based Diffusion) | unconditional and controlled text generation | Logits generation | Greedy projection, Sampling, Multi-hot |
| Bit Diffusion (Generating Discrete Data using Diffusion Models with Self-Conditioning) | categorical image generation and image captioning | — | Self-Conditioning, Asymmetric Time Intervals |
| DiffusER (Discrete Diffusion via Edit-based Reconstruction) | machine translation, summarization, and style transfer | Edit-based Corruption | Beam Search, 2D Beam Search, Nucleus Sampling |
| Masked-Diffuse LM | controllable text generation | Mask with Entropy and Reluency | Minimum Bayes Risk |
| RDMs (Reparameterized Discrete Diffusion Model) | machine translation | — | Adaptive Routing Strategy |
| Embedding Diffusion Models |  |  |  |
| Diffusion-LM | controllable text generation | Cosine | Rounding Step and MBR |
| DiffuSeq | dialogue, question generation, simplification, paraphrasing | Partial Noising | Classifier-free Conditional Denoising, MBR |
| SED (Self-conditioned Embedding Diffusion) | conditional and unconditional text generation, text infilling | Cosine | Self-conditioning |
| CDCD (Continuous diffusion for categorical data) | prompt completion and infilling, machine translation | Partial Noising, Time warping | Self-conditioning, Time warping |
| Difformer | machine translation and abstractive text summarization | Noise Factor | 2D parallel decoding |
| SeqDiffuSeq | dialogue, question generation, simplification, paraphrasing, translation | Adaptive noise schedule | Self-conditioning |
| DiffuSum | extractive text summarization | — | — |
| GENIE (Diffusion Language Model Pre-training Framework for Text Generation) | text summarization, common sense generation | — | Continuous Paragraph Denoise |
| DiNoiSer (Diffused Conditional Sequence Learning by Manipulating Noises) | machine translation, text simplification, paraphrasing | Manipulated Noises | Self-conditioning, Condition-enhanced Denoiser, Beam Search, Minimum Bayes Risk |

We present several studies on diffusion models in NLP by grouping them based on their methods for adapting the diffusion process to the textual domain. Specifically, we have two groups: Discrete Diffusion Models and Embedding Diffusion Models (Figure [2](#S2.F2)). The former operates directly in the discrete input space, while the latter involves lifting discrete inputs into a continuous space.

For each category, we then categorize diffusion models into a multi-perspective taxonomy considering the following criteria: (1) the task they are applied to, (ii) schedule methods during the forward process and (iii) sampling methods used for the reverse process. We note that Reluency in “Schedule” column indicates a linguistic feature that measures the relevance of word $w$ in one sentence $d$ via tf-idf weights. Entropy is a measurement of the amount of information with entropy $H$ in the word $w$ to reflect the importance of that word.
Table [1](#S3.T1) shows the categorization.

### 3.1 Discrete Diffusion Models

In the discrete diffusion process, the data is corrupted by switching between discrete values. Discrete diffusion models extend diffusion models to discrete state spaces by corrupting and refining the sentences at the token level.

Multinomial Diffusion introduces a diffusion-based generative model specifically designed for non-ordinal discrete data. It achieves this by diffusing the data to a uniform categorical distribution, effectively capturing the underlying structure while maintaining controlled randomness. The model’s transition mechanism involves independent decisions to either resample or retain values, with resampling performed from a uniform categorical distribution.

D3PMs replaces Gaussian noise with Markov transition matrices to diffuse real-world data distribution. It incorporates various types of transition matrices, such as Gaussian kernels, nearest neighbors, and absorbing states, to extend corruption processes. Moreover, D3PMs introduces a novel loss function that combines the variational lower bound with an auxiliary cross-entropy loss. Unlike continuous diffusion, D3PMs allows precise control over the data corruption and denoising process by selecting $Q_{t}$ in Equation [7](#S2.E7), going beyond the use of additive Gaussian noise.

Zero-Shot Diffusion utilizes an encoder-decoder architecture with time-based positional encoding for neural machine translation. It employs a transformer encoder to process the source-language sentence and a transformer decoder to handle the noisy target sentence. Notably, this work pioneers conditional text generation using a diffusion model.

Bit Diffusion encodes discrete data as binary bits and trains a continuous diffusion model that treats these binary bits as real numbers. It firstly introduces the self-conditioning technique that greatly improves the sample quality and is widely applied to the following works .

SUNDAE proposes step-unrolled text generation and is the first non-AR method to show strong results in both machine translation and unconditional text generation.

DiffusER employs a 2-dimensional beam search and edit-based text generation. Instead of a pure end-to-end approach, the system divides the task into edit tagging and generation. It generates a sequence of edits to transform a random noise distribution into high-quality output.

DiffusionBERT combines diffusion models with Pre-trained Language Models (PLMs) by training BERT in reverse of a discrete diffusion process. It introduces a new noise schedule for the forward diffusion process and incorporates the time step into BERT . By including the time step, DiffusionBERT captures lost temporal information during diffusion, enhancing the accuracy of the reverse process.

SSD-LM stands out due to two key features. Firstly, it is semi-autoregressive, enabling iterative generation of text blocks and dynamic length adjustment during decoding. Secondly, it is simplex-based, directly applying diffusion on the natural vocabulary space instead of a learned latent space. This approach facilitates the incorporation of classifier guidance and modular control without the need for modifications to existing classifiers.

Masked-Diffuse LM employs strategic soft-masking, informed by linguistic features, to corrupt both discrete and continuous textual data. It iteratively denoises the data by predicting the categorical distribution. The gradual introduction of perturbations via soft-masking, following an easy-first-generation approach, enhances structural coherence, overall quality, and flexibility in text generation. This pioneering work utilizes linguistic features to effectively corrupt and recover input textual data, improving the generation process.

RDMs introduces a novel reparameterization technique for discrete diffusion models. It employs a stochastic routing mechanism to decide between denoising or noisy resetting for each token. The router ensures uniform processing by assigning equal probabilities to all tokens. This reparameterization simplifies training and enables flexible sampling.

### 3.2 Embedding Diffusion Models

Recent studies utilize diffusion processes to generate continuous representations (embeddings) for discrete tokens, known as embedding diffusion models.

Diffusion-LM constructs diffusion models on continuous word embedding space and incorporates auxiliary losses for joint learning of embedding and network parameters.

DiffuSeq focuses on sequence-to-sequence generation using encoder-only Transformers and partial noising to define the diffusion process and learn the denoising function.

SED builds upon the modeling and objectives of Diffusion-LM, introducing a self-conditioning mechanism that enhances baseline performance. Notably, it demonstrates successful scalability to large text datasets like C4 .

Difformer tackles challenges in applying continuous diffusion models to discrete text generation by addressing denoising objective collapse, imbalanced embedding scales, and inadequate noise during training. It introduces three crucial components: an anchor loss function, layer normalization for embeddings, and an increased noise factor to enhance the scale of added noise.

CDCD introduces a variance-exploding stochastic differential equations-based diffusion model tailored for text modeling and machine translation. It integrates time warping, an active learning strategy that dynamically adjusts the noise distribution during training to optimize efficiency.

SeqDiffuSeq incorporates self-conditioning and introduces a method to learn token-level noise schedules for text generation. By leveraging appropriate noise schedules, it aims to enhance the quality of generated samples and likelihood modeling . In contrast to DiffuSeq , SeqDiffuSeq explores different model structures and investigates the impact of noise scheduling in sequence-to-sequence tasks.

DiffuSum applies diffusion models to enhance extractive summarization. It generates summary sentence representations and extracts relevant sentences using representation matching. The model introduces a contrastive sentence encoding module that employs matching and multi-class contrastive losses to align and diversify representations. Significantly, DiffuSum represents the first known utilization of diffusion models in the field of extractive summarization.

GENIE is a large-scale diffusion-based language model consisting of an encoder and decoder. It enhances noise removal and paragraph-level coherence through continuous paragraph denoise (CPD) loss in pre-training. The CPD objective guides the diffusion-decoder to reconstruct a clean version of a corrupted text paragraph while preserving semantic and syntactic coherence.

DiNoiSer addresses small noise effects on "discrete" embeddings in a continuous space, improving diffusion models through noise manipulation in conditional sequence learning. It tackles the discreteness problem by excluding small-scale noises from diffused sequence learner training. For sampling, it introduces an effective method that consistently indicates large noise scales, enhancing the predictive capabilities by amplifying the influence of source conditions on predictions.

### 3.3 Discrete vs. Embedding Diffusion

**Table 2: Comparative Analysis of Discrete Diffusion Models and Embedding Diffusion Model. Refinements Adaptation column serves as an indicator of the system’s ability to incorporate refinements from continuous diffusion in the image domain.**
|  | Diffusion<br>Process | Classifier-based<br>Controls | Refinements<br>Adaptation |
| --- | --- | --- | --- |
| Discrete<br>Diffusion | token<br>level | ✗ | ✗ |
| Embedding<br>Diffusion | sequence<br>level | ✔ | ✔ |

In Table [2](#S3.T2), we summarize the advantages of embedding diffusion models over discrete diffusion models.

- •
Diffusion Process: embedding diffusion models transform discrete inputs into a continuous space, enabling representation of multiple outcomes at intermediate timesteps, particularly crucial in capturing token-level uncertainty in language modeling. In contrast, denoising models operating in the discrete input space lack this ability and are confined to specific tokens.
- •
Classifier-based Controls: embedding diffusion models can integrate classifier-based guidance, enhancing the quality of generated samples by leveraging additional information from a classifier to guide the sampling process. In contrast, discrete diffusion models lack this capability, thereby restricting their ability to generate high-quality samples.
- •
Refinements Adaptation: showed that discrete diffusion approaches do not reap the advantages derived from the advancements made in continuous diffusion methods within the domain of image processing. Conversely, embedding diffusion models exhibit the capacity to leverage these refinements, rendering them more advantageous and valuable in this context.

## 4 Diffusion vs. Other Generative Models

### 4.1 Comparison against Latent Variable Models

Unlike variational autoencoders (VAEs) or flow-based models , diffusion models are learned using a fixed procedure with the latent variable having a high dimensionality (same as the original data). GANs are known for potentially unstable training and less diverse generations due to their adversarial training nature. VAEs rely on a surrogate loss. Flow-based models require the construction of specialized architectures to construct reversible transforms.

**Table 3: Comparative Analysis of Autoregressive (AR) and Diffusion Models in NLP. Token-level Controls of diffusion models include syntactic structure, parse trees, semantic content, parts-of-speech, etc. In terms of training complexity, diffusion models employ multiple rounds of diffusion steps $T$ to generate the entire sequence. Each diffusion step involves optimizing the objective function to capture the denoising process. Specifically, Transformer models are utilized to model the denoising process within each diffusion step.**
|  | Advantages | Disdvantages |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Models | Parallel<br>Generation | Text<br>Interpolation | Token-level<br>Controls | Robustness to<br>Input Corruption | Training<br>Complexity | Model<br>Interpretability |
| AR Models | ✗ | ✗ | ✗ | ✗ | $\mathcal{O}(n)$ | \faSmileO |
| Diffusion Models | ✔ | ✔ | ✔ | ✔ | $\mathcal{O}(n^{T})$ | \faFrownO |

As notes, diffusion models have a distinct advantage over models like VAEs and GANs, which generate data in a single forward pass. Diffusion models instead focus on reconstructing a small amount of information that has been removed by the corruption process, making the task less challenging.

### 4.2 Comparison against Autoregressive Models

Figure: (a) Diffusion-based language model
Refer to caption: extracted/2305.14671v2/figures/diffusion-LM.png

Figure: Figure 4: Illustration for how to incorporate Transformers architecture with diffusion models in NLP.
Refer to caption: extracted/2305.14671v2/figures/connection_with_transformer.png

**Table 4: Training Corpus and connection with Transformer for Discrete and Embedding Diffusion Models. Parameter column refers to the size of used Transformer architecture specifically. Pre-trained column indicates whether the system uses the pre-trained word embedding or not.**
| Architecture | Training Corpus | Parameter Size | with Transformer | Pre-trained |
| --- | --- | --- | --- | --- |
| Discrete Diffusion Models |  |  |  |  |
| Multinomial Diffusion | text8, enwik8 | — | 12-layer Transformer | ✗ |
| D3PMs | text8, LM1B, CIFAR-10 | — | 12-layer Transformer | ✗ |
| Zero-shot Diffusion | WMT14 (DE-EN, FR-EN)<br>WMT19 (DE-FR) | — | 12-layer Transformer | ✗ |
| SUNDAE | WMT14 (EN-DE)<br>C4<br>Python code dataset | 63M | encoder-decoder Transformer<br>causality masking removed<br>in the decoder | ✗ |
| DiffusionBERT | LM1B | 110M | bert-base uncased is<br>trained for reverse process | ✔ |
| SSD-LM | OpenWebText | 0.4B | bi-directional<br>Transformer encoder<br>a timestep embedding added<br>before the first<br>Transformer block | ✗ |
| Bit Diffusion | CIFAR-10, ImageNET<br>MSCOCO 2017 | — | 6-layer Transformer decoder | ✗ |
| DiffusER | WMT’14<br>CNN/DailyMail, Yelp | — | 6-layer Transformer to predict<br>the edit operations<br>6-layer Transformer for generator | ✗ |
| Masked-Diffuse LM | E2E | 80M | BERT to encode the input text<br>Transformer to module<br>the reverse process | ✔ |
| RDMs | IWSLT14 (DE-EN)<br>WMT14 (EN-DE)<br>WMT16 (EN-RO) | — | Length prediction module<br>on top of Transformer encoder | ✔ |
| Embedding Diffusion Models |  |  |  |  |
| Diffusion-LM | E2E, ROCStories | 80M | 12-layer Transformer | ✗ |
| DiffuSeq | CCD, Quasar-T<br>Newsela-Auto<br>Wiki-Auto, QQP | 91M | 12-layer Transformer | ✗ |
| SED | C4 | 135M & 420M | 12-layer Transformer | ✔ |
| CDCD | MassiveText, C4<br>WMT2014, WMT2020 | 1.3B | Mask-conditional Transformer | ✔ |
| Difformer | IWSLT14, WMT14<br>WMT16, Gigaword | — | 6-layer Transformer | ✗ |
| SeqDiffuSeq | CCD, Quasar-T, Wiki-Auto<br>QQP, IWSLT14 | — | 12-layer Transformer | ✗ |
| DiffuSum | CNN/DailyMail,<br>XSum<br>PubMed | 13M | 8-layer Transformer as encoder<br>12-layer Transformer as generator | ✔ |
| GENIE | Gigaword, CNN/DailyMail<br>XSum, CommonGen | — | 6-layer Transformer as encoder<br>6-layer cross attention Transformer<br>as<br>denoising architecture | ✔ |
| DiNoiSer | IWSLT14 (DE-EN)<br>WMT14 (EN-DE, EN-RO)<br>Wiki-Auto, QQP | — | 12-layer Transformer | ✗ |

Autoregressive (AR) models currently dominate the field of language modeling. Also known as causal modeling or the next-token prediction task, AR modeling learns the joint distribution over a token sequence $p(x_{1},x_{2},...,x_{N})$ by factorizing it into sequential conditionals $p(x_{k}|x_{1},...,x_{k-1})$ and model them separately with shared parameters (see Figure [3](#S4.F3)). This means that sampling always proceeds along the left-to-right direction of the sequence. However, in many cases, the ability to go back and refine the earlier parts of the sequence should be useful. In Figure [3](#S4.F3), we illustrate the fundamental distinctions between AR and diffusion models, and highlight the distinctive features of the diffusion architecture that endow it with the ability to refine the previous generations, which has potentials to advance the state-of-the-art in the field.

Additionally, reveals that, compared to AR models , diffusion models can predict all tokens in a sequence at once, which increases interactions between tokens, potentially leading to more coherent samples. Similarly, and note that the fixed generation order (left-to-right) from AR models limits the model’s flexibility in many controllable generation settings. For example, infilling task, which imposes lexical control on the right contexts, and the syntactic structure control task, which controls global properties involving both left and right contexts. More importantly, this prohibits the iterative refinement of complete text drafts from making them more self-consistent, which is a common task for human writers.

In Table [3](#S4.T3), we summarize the empirical benefits of diffusion models over AR models. We categorize them into four aspects: parallel generation, sentence interpolation, token-level control, and robustness to input corruption.

- •
Parallel Generation: diffusion models exhibit a notable departure from the autoregressive nature of AR models. While AR models generate output tokens sequentially conditioned on preceding tokens, diffusion models adopt a parallel generation approach, enabling simultaneous generation of all output tokens. This characteristic enhances the speed and efficiency of text generation, rendering diffusion models particularly suitable for real-time applications.
- •
Text Interpolation: diffusion models demonstrate a superior capacity for text interpolation. Leveraging the denoising process inherent in their design, diffusion models can generate intermediate sentences between two given sentences, ensuring smooth transitions and coherent outputs. This capability enhances the overall fluency and cohesiveness of generated text.
- •
Token-level Controls: Diffusion models provide advanced Token-level Controls, facilitating fine-grained manipulation of generated outputs. This level of control enables precise modifications and interventions in the generated sequences, enhancing the interpretability and applicability of diffusion models in diverse downstream tasks.
- •
Robustness to Input Corruption: Diffusion models exhibit enhanced robustness due to their denoising mechanism that facilitates the reconstruction of the original input. This process aids in mitigating errors and noise present in the input sequence. Consequently, diffusion models are capable of capturing a broader spectrum of input variations by learning a more adaptable distribution over the input data.

In summary, diffusion models offer empirical advantages over AR models, encompassing parallel generation, text interpolation, and advanced token-level controls. These characteristics underscore the potential of diffusion models in various text generation scenarios, emphasizing their efficiency, coherency, and flexibility.
In addition to the advantages discussed in Table [3](#S4.T3), we also identify two significant disadvantages of diffusion models compared to AR models in terms of training complexity and interpretability.

- •
Training Complexity: Diffusion models are more difficult to train than AR models due to their more complex architecture and optimization objective. In a diffusion model, the entire sequence is generated simultaneously through multiple rounds of diffusion steps, which involve applying a non-linear function to a set of latent variables to obtain the next generation of the sequence. This requires optimizing a complex objective function that includes both the data likelihood and the distance between the generated and ground-truth sequences. On the other hand, AR models generate sequences sequentially by conditioning each time step on the previous ones. This allows for a simpler optimization objective and faster convergence during training.
- •
Model Interpretability: Diffusion models involve multiple non-linear transformations during the diffusion process, resulting in abstract representations in the latent space. These representations may not have a clear interpretation or meaning, and understanding how a specific output sequence is generated from the input can be challenging. This makes diffusion models less interpretable. In contrast, AR models generate sequences step by step, building on the previous steps. Each step is influenced by the preceding steps, making it easier to understand how the output sequence is generated based on the input. AR models are more interpretable due to this sequential nature.
These observations highlight the trade-offs associated with diffusion models, emphasizing the need to consider both their advantages and disadvantages in practical applications.

### 4.3 Transormers with diffusion models

Transformers architecture could be combined with diffusion models, as depicted in Figure [4](#S4.F4). Specifically, the Transformer models are used in the encoder-decoder layout to model the denoising function. During the reverse process, the input sequence $x$ therefore only requires one forward computation.

Furthermore, Table [4](#S4.T4) provides a comprehensive summary of the training corpus of surveyed systems, highlighting their associations with Transformers. This includes details such as the parameter size and the specific architectures employed by each system for modeling denoising functions, as well as their utilization of pre-trained representations from Transformers during the diffusion process. We hope that this summary can provide researchers with rapid insights into the interplay between Transformers and diffusion models in NLP.

Figure: Figure 5: Algorithms proposed to adapt the discrete data. Details of the proposed architectures are described in Section [3](#S3). Details of the algorithms are described in Section [5](#S5).
Refer to caption: extracted/2305.14671v2/figures/algorithms_and_techniques.png

## 5 Algorithms & Techniques

In this section, we highlight algorithms and techniques proposed for diffusion models in NLP. They are twofold: (1) adapting the models to discrete variables and (2) improving sampling procedures. Figure [5](#S4.F5) depicts the algorithms proposed from the surveyed papers.

### 5.1 Adapting Discrete Variables

#### 5.1.1 Diffusion Steps

To optimize the objective function, DDPM utilizes the property that the noise added at each time step in the diffusion process is Gaussian noise; hence the concrete expressions of the objective can be derived. However, the Gaussian distribution here is mainly for continuous domains such as image generations. Hence, D3PM proposed a new method for adding noises for discrete variables. D3PM defined a series of transition matrices that transformed the discrete tokens into [MASK] based on pre-defined probabilities at different time steps.

#### 5.1.2 Objective Functions

##### Predicting initial inputs directly

Traditionally, for the approximations of the mean values of each time step, DDPM predicts the noise at each time step directly, however, Diffusion-LM found that the model might fail to generate the initial input $x_{0}$ that commits to a single word as the denoising steps cannot ensure that $x_{0}$ lies precisely on the embedding of a word. To solve this problem, Diffusion-LM predicts the initial input $x_{0}$ directly in their objective functions.

##### Partial noising and conditional denoising

DiffuSeq connects the conditional text $c$ and the target text $x$, and adds noise only to the target text $x$ in forward process while denoising only $x$ in the denoising process. In contrast to Diffusion-LM’s approach of classifier-guided diffusion, DiffuSeq employs a method of classifier-free diffusion that is directed by spatial points. Thus, the system is capable of producing conditional generations in the absence of external classifiers.

### 5.2 Sampling from Latent Space

##### Asymmetric Time Intervals

Time step plays a critical role in diffusion models. During typical reverse diffusion, symmetric time intervals are often used for both state transition and time reduction, resulting in shared $t$ for $f(x_{t},t)$. However, shows experimentally that when taking a larger step, using asymmetric time intervals with $f(x_{t},t^{\prime})$, implemented via a simple manipulation of time scheduling at generation, can lead to improved sample quality.

##### Self-Conditioning

When estimating the data sample by the denoising network $f$ at a time step, conditioning the network directly on its previously estimated samples (as opposed to discarding them) can provide better sample quality .

##### Time Warping

introduces time warping, an active learning strategy that automatically adapts the distribution of noise levels sampled during training to maximize efficiency. The method alters the relative weighting of the noise levels corresponding to different time steps $t$.
To sample $t$ non-uniformly in practice, the inverse transform sampling can be used: first generate uniform samples $u$ $\in$ $[0,1]$ and then warp them using the inverse cumulative distribution function (CDF) of the distribution which corresponds to the desired weighting: $t$ = $F-(u)$. This time warping procedure is equivalent to time reweighting in expectation, but more statistically efficient.

## 6 Challenges & Future Directions

In this section, we advance potential lines of inquiry that are both contemporarily significant and intellectually deserving of investigation (Figure [6](#S6.F6)).

### 6.1 General Challenges

##### Latent Space Restriction

Diffusion models impose a restriction on the latent space representations, as the dimensions of latent vectors and inputs must be the same. This constraint limits the representational power of the latent vector.

##### Computational Cost

The convergence of diffusion models requires a large number of iterations, which can lead to significant computational costs, especially when dealing with large datasets.

Figure: Figure 6: Challenges and future directions we conclude based on the surveyed papers.
Refer to caption: extracted/2305.14671v2/figures/concept_fig_s6.png

##### Sensitivity

Diffusion models can be very sensitive to the choice of hyperparameters, such as diffusion coefficient, time step size, number of diffusion steps, etc., which can lead to suboptimal performance or even failure to converge.

##### Dependence on diffusion process assumptions

Diffusion models rely on the assumption that information diffuses smoothly and uniformly across the data, which may not always hold in practice. Given perfect mathematical formulation, the diffusion process itself might not be intuitive enough. For instance, optimizing from a totally noisy distribution is quite different to human mind.

##### Limited interpretability and explainabilities

The black-box nature of diffusion models makes it challenging to understand how they make decisions, limiting their interpretability. For instance, the latent vectors learned from diffusion models do not have any linguistic or structural explainabilities.

### 6.2 NLP-Specific Challenges

##### Token Rounding Errors

The learned embeddings through embedding diffusion models define a mapping from discrete text to the continuous $x_{0}$. We now describe the inverse process of rounding a predicted $x_{0}$ back to discrete text. Rounding is achieved by choosing the most probable word for each position. However, empirically, the model
fails to generate $x_{0}$ that commits to a single word .

##### High Perplexity

As stated in , the perplexity from diffusion models lags behind AR models. However, measuring perplexity with a pretrained AR model such as GPT-2 may bias the metric towards AR models. Besides, previous studies have demonstrated that generating text with low perplexity does not necessarily imply high quality, but rather suggests degenerate behavior. . Hence, better metrics which have a stronger correlation with human judgements of quality are needed. For this factor, proposed MAUVE Score, a metric for open-ended text generation that compares the distribution of generated text with that of reference text using divergence frontiers, to better correlate with human judgments.

### 6.3 Potential Future Directions

##### More Advanced Ways to connect Transformers

How to better combine the spatiality of Transformer and temporality of Diffusion is a tricky question since the ideologies for Transformer and Diffusion are from totally different perspectives. Common architectures from our surveyed paper make The time step $t$ included in the neural net through a Transformer sinusoidal position embedding in each block. And currently people just diffuse the whole sequence of the sentences, diffusion process on single token might be interesting to try on. More variations of injecting Transformers into Diffusion might be needed to explore and deeper analysis is needed with strong foundations.

##### Large Scaled Diffusion Language Models with impressive few-shot learning capabilities

Giant language modeling has made significant strides in recent years and has become a dominant area of research in artificial intelligence. With advances in deep learning and natural language processing, large language models like GPT-3 have shown impressive abilities in tasks such as language translation, text generation, question-answering, and even programming. Currently only SED has studied the scaling issues for diffusion models in NLP, the enormous potential of Large-Scale Diffusion Language Modeling in few-shot learning warrants further exploration.

##### Multimodal Diffusion Modeling

In recent years, there has been a growing interest in developing visual language models (VLMs), which are deep learning models that can understand the relationship between images and natural language. The amazing few-shot performance of VLMs shows great potential to transform how machines interact with the visual world and language, such as Vision-Language Pre-training (ViLBERT) model from Facebook AI Research (FAIR) and the Georgia Institute of Technology, and Flamingo from DeepMind. However, current VLMs are all based on Transformers, the incorporation of Diffusion Models presents vast potential for exploration and discovery.

## 7 Conclusion

This survey paper extensively discusses the formulations, strengths, limitations, and applications of diffusion models in NLP. We conduct a comprehensive comparison between diffusion models and alternative generative models, focusing on autoregressive (AR) models. Additionally, we explore the integration of the Transformer architecture with diffusion models across various architectures.

Our findings demonstrate the significant advantages of diffusion models over AR models. They excel in parallel generation, enabling faster and more efficient text generation. Diffusion models also demonstrate superior performance in sentence interpolation, token-level controls, and robustness to input corruption. Further research on integrating Transformers into diffusion models and developing multimodal and large-scale diffusion language models for few-shot learning is crucial.

In summary, this survey paper provides a comprehensive overview of diffusion models in NLP, highlighting their benefits, comparative analysis with AR models, and avenues for future research. We hope it can contribute to the understanding and advancement of diffusion models in the field of NLP.

## Limitations

The selection of diffusion models included in this paper may introduce a bias based on our knowledge and availability of resources. This could potentially exclude relevant diffusion models that were not considered or well-known at the time of the survey. It is crucial to acknowledge that the selection of specific models and the exclusion of others can impact the comprehensiveness and generalizability of the findings. Another limitation pertains to the understanding and interpretation of the inner workings and decision-making processes of the surveyed diffusion models. Diffusion models in NLP, particularly those employing deep learning techniques, are often regarded as black-box models with limited interpretability. The lack of interpretability can impede the trust and acceptance of diffusion models in practical applications.

## Ethics Statement

Diffusion models in NLP may be influenced by biases present in the training data, highlighting the need to consider the ethical implications of deploying biased models in real-world applications. Furthermore, the impact of diffusion models in NLP extends to shaping public opinion, influencing decision-making processes, and affecting social dynamics. Therefore, we prioritize responsible use and communication of the findings in this paper, avoiding sensationalism, misrepresentation, or overgeneralization of the capabilities and limitations of diffusion models in NLP to ensure a well-rounded understanding among the public.

## Acknowledgement

This project is supported in part by Sony Research Grant.