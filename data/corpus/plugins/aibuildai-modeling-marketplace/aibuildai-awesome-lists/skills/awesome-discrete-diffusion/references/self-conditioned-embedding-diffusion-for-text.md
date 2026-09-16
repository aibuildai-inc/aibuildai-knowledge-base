---
arxiv_id: "2211.04236"
title: "Self-conditioned Embedding Diffusion for Text Generation"
year: 2022
source: arxiv2md
---

## Abstract

Abstract Can continuous diffusion models bring the same performance breakthrough on natural language they did for image generation?
To circumvent the discrete nature of text data, we can simply project tokens in a continuous space of embeddings, as is standard in language modeling.
We propose Self-conditioned Embedding Diffusion ( Sed ), a continuous diffusion mechanism that operates on token embeddings and allows to learn flexible and scalable diffusion models for both conditional and unconditional text generation.
Through qualitative and quantitative evaluation, we show that our text diffusion models generate samples comparable with those produced by standard autoregressive language models — while being in theory more efficient on accelerator hardware at inference time.
Our work paves the way for scaling up diffusion models for text, similarly to autoregressive models, and for improving performance with recent refinements to continuous diffusion.

## 1 Introduction

Continuous diffusion models  have taken the world of image generation by storm, advancing the state of the art further than ever before .
Can the same framework encounter as much success on the text modality?
Diffusion for language is indeed an attractive prospect.
Compared to autoregressive (AR) models , diffusion models can predict all tokens in a sequence at once.
This allows for bidirectional, rather than causal attention—increasing interactions between tokens, potentially leading to more coherent samples.
Diffusion models can make a better usage of hardware accelerators during inference than AR models, since computations are parallelizable over the sequence axis.

Yet AR models remain the mainstream approach for modelling text.
A major obstacle to text diffusion is that diffusion processes typically operate in continuous space.
While this naturally handle images, text is inherently discrete.
Consequently, most previous attempts to apply diffusion to text have focused on *discrete* diffusion-like approaches.
These methods do not benefit from the refinements made to continuous diffusion in the image domain.
Crucially, they cannot make use of guidance , which drastically improves diffusion models sample quality.

We address this gap by making a simple observation: language models operate mostly in continuous space, with discrete tokens only as inputs and outputs.
A natural idea is then to conduct diffusion directly in a continuous token embedding space.
For simplicity, we use a fixed embedding space, either random or stemming from a trained language model.
Combined with the “self-conditioning” refinement, this forms the basis of the method we propose, Self-conditioned Embedding Diffusion (Sed).

Sed models rival mainstream AR models in both conditional and unconditional text generation. We make the following contributions:

- •
In section [3](#S3), we introduce Sed, the first continuous diffusion approach for text with good scaling properties (testing models up to 420M parameters). We analyze several continuous text diffusion settings, and identify self-conditioning and diffusion on small fixed embeddings as key factors to make continuous text diffusion work.
- •
In section [4](#S4), we apply classifier-free guidance  to text data—an original achievement. We show that Sed can rival AR models on generic language tasks, for
similar models sizes.
Sed samples achieve a better likelihood-entropy trade-off compared to these models, and are deemed comparable (if slightly worse) by human raters.

## 2 Related work

We provide an overview of diffusion models with a focus on modeling discrete data, as well as AR models and sample-based metrics for evaluating text generation.

#### Continuous diffusion on continuous image data.

Continuous diffusion has recently established itself as the method of choice for modeling continuous data such as images.
While our main focus in this paper is on discrete data, we review some key works in continuous data modeling as this literature was the major source of inspiration for Sed.
The first continuous diffusion formulation was introduced in the seminal work by .
improved and simplified this formulation, relating it to denoising score matching, and creating a new method called DDPM.
further improved upon DDPM, showcasing impressive diffusion results compared to GANs.
introduced diffusion in latent space.
Conceptually similar to Sed, it was specifically targeted at image modeling.
Classifier-free guidance was proposed by as a mean to improve image fidelity at the cost of reduced diversity.
GLIDE  scaled up the ideas of guided diffusion, while DALL-E 2  and Imagen  are the latest, most advanced image generation systems to date, combining most of the improvements proposed in previous works.

#### Discrete diffusion on discrete data.

One cannot simply reuse the methods that are successful on continuous image data in the discrete text domain.
A number of bespoke methods have been explored instead, forming the family of *discrete diffusion* approaches.
In discrete diffusion, the data is corrupted by switching from one discrete value to another.
This was first proposed in the seminal work by , where it was tested on simplistic binary heartbeat data.
It was extended to multinomial text modeling  and further scaled up in the D3PM work .
Most recently, a similar discrete diffusion approach was applied to image modeling in VQ-Diffusion .
In parallel, a few diffusion-like approaches were proposed in the denoising autoencoders literature.
CMLM  tackled machine translation. SUNDAE  was the first non-AR method to show strong results both in machine translation and unconditional text generation. MaskGIT  demonstrated excellent results in modeling VQ-discretized images. These approaches rely on training models to predict masked tokens from their context, and iterating this reconstruction step multiple times at sampling time. Despite those positive developments, the samples from discrete diffusion methods for text modeling remains less coherent than those produced by AR methods.

#### Continuous diffusion on discrete data.

Fewer works try to tackle diffusion on discrete data from the same angle as Sed – starting by turning the data into continuous representations before modeling it with continuous diffusion formulations.
used a VAE to generate such representations for discrete music modeling, with exciting results.
Closest to Sed, Diffusion-LM  trains a token embedding together with the diffusion model itself. Diffusion-LM meets success on specific language applications, in low data regime and on constrained, very formatted textual data.
Most recently, Analog Bits  introduced *self-conditioning*, closely related to step-unrolls in SUNDAE , together with bit-level modeling to improve the generation of discretized images.
While the qualitative results of those continuous methods on text modeling show promise, they have not been shown to scale to large realistic text datasets like C4  yet, or to compare with AR approaches on generic language tasks.

#### Auto-regressive modelling on discrete data.

AR models remain the method of choice for modeling discrete data.
In combination with neural networks, they were first explored by and later combined with RNNs .
Their breakthrough moment came with the advent of the Transformer architecture, introduced by for machine translation.
Even more impressive results were shown with GPT-3 , which trained a large AR language model unconditionally, and used few-shot prompting to adapt it to new tasks.
A few works later improved upon the results of GPT-3, including .

#### Sample-based evaluation of text generative models.

There are traditionally two classes of metrics for generative modeling: likelihood-based and sample-based.
While the likelihood-based way is mathematically appealing, its usefulness for measuring progress is reduced by the fact that not all models readily provide likelihood computation.
Just like the sampled-based FID metric was important for driving the progress of diffusion in image modeling, there is a need for a sample-based metric which would be universally accepted for text modeling.
investigated fidelity/variance metrics for evaluating text GANs.
suggested using FID for texts.
later used those previously proposed metrics to iterate on ScratchGAN but did not provide conclusive guidance on which metric a practitioner should choose – essentially finding serious vulnerabilities in all investigated metrics.
We opted for a middle ground, reporting both sample likelihood according to a strong AR model and human preferences.

## 3 Method

In this section, we outline the different components of Sed: continuous diffusion in the space of token embeddings and self-conditioning, which form the basis of our approach for unconditional text generation; span masking and guided diffusion to enable conditional generation.

### 3.1 Diffusion models for unconditional text generation

Diffusion models in continuous space.
We consider diffusion models as introduced by and improved by .
A diffusion model aims at modelling a data distribution ${\bm{x}}_{0}\in\mathbb{R}^{n}\sim q\in\mathcal{D}(\mathbb{R}^{n})$ by estimating a sequence of latent variables ${\bm{x}}_{T}$, …, ${\bm{x}}_{1}$ of the same dimensionality as the data ${\bm{x}}_{0}$.
Starting from ${\bm{x}}_{0}$, the latent variables are generated with a Markov chain called the *forward process*: ${\bm{x}}_{t}\sim q(\cdotp|{\bm{x}}_{t-1},t)$.
It is defined by gradually interpolating the iterate with Gaussian noise according to noise levels defined by a schedule $\beta_{1},...,\beta_{T}$:

$$ ${\bm{x}}_{t}\sim q(\cdotp|\,{\bm{x}}_{t-1},t)=\mathcal{N}(\sqrt{1-\beta_{t}}{\bm{x}}_{t-1},\beta_{t}{\bm{I}}).$ (1) $$

This parametrization gives us a closed form to sample ${\bm{x}}_{t}$ for any arbitrary $t\geq 1$, given ${\bm{x}}_{0}$:

$$ ${\bm{x}}_{t}=\sqrt{\alpha_{t}}{\bm{x}}_{t-1}+\sqrt{1-\alpha_{t}}\epsilon_{t}=\sqrt{\overline{\alpha}_{t}}{\bm{x}}_{0}+\sqrt{1-\overline{\alpha}_{t}}\epsilon,$ (2) $$

where $\alpha_{t}:=1-\beta_{t}$, $\overline{\alpha}_{t}:=\prod_{s=1}^{t}\alpha_{s}$, $\epsilon_{t}\sim\mathcal{N}\big{(}0,{\bm{I}}\big{)}$ and $\epsilon\sim\mathcal{N}\big{(}0,{\bm{I}}\big{)}$.

We define our generative model by approximately inverting the diffusion process of Eq. [1](#S3.E1) to obtain a *reverse process*.
The reverse process starts from ${\bm{x}}_{T}\sim\mathcal{N}(0,{\bm{I}})$ and is defined as a Markov chain with learned Gaussian transitions (parameterized by $\theta$, the weights of a neural network): ${\bm{x}}_{t-1}\sim p_{\theta}(\cdotp|{\bm{x}}_{t})=\mathcal{N}\big{(}{\bm{\mu}}_{\theta}({\bm{x}}_{t},t),\sigma(t)^{2}{\bm{I}}\big{)}$.
We train a neural network to predict an estimate $\hat{{\bm{x}}}_{0}({\bm{x}}_{t},t,\theta)$ of the data ${\bm{x}}_{0}$ and approximate the reverse process by using the following parametrization, with learnable means but fixed variances, and a fixed schedule $\beta_{1},...,\beta_{T}$:

$$ ${\bm{\mu}}_{\theta}({\bm{x}}_{t},t)=\frac{\sqrt{\overline{\alpha}_{t-1}}\beta_{t}}{1-\overline{\alpha}_{t}}\hat{{\bm{x}}}_{0}({\bm{x}}_{t},t,\theta)+\frac{\sqrt{\alpha_{t}}(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_{t}}{\bm{x}}_{t},\qquad\sigma(t)^{2}=\frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_{t}}\beta_{t}\cdot$ (3) $$

While there exists a tractable variational lower-bound (VLB) on $\log p_{\theta}({\bm{x}}_{0})$,
showed that better results are obtained by optimizing a simplified objective that re-weights the terms in the VLB.
We follow this approach, which simplifies the loss to a sum of mean-squared errors between the ground truth data ${\bm{x}}_{0}$ and its estimates $\hat{{\bm{x}}}_{0}({\bm{x}}_{t},t,\theta)$:

$$ $\mathcal{L}_{\text{diffusion}}=\mathbb{E}_{{\bm{x}}_{0}\sim q({\bm{x}}_{0}),\,t\sim\mathcal{U}(1,T)}\|{\bm{x}}_{0}-\hat{{\bm{x}}}_{0}({\bm{x}}_{t},t,\theta)\|^{2}\cdot$ (4) $$

Though this framework works out of the box on images, which are close to continuous, we cannot apply it directly to the discrete tokens of the text modality.
To resolve this issue, we perform continuous diffusion in a continuous space in which we embed text tokens.

Diffusion on word embeddings.
We consider textual data ${\bm{w}}=(w_{1},\ldots,w_{N})$, where each $w_{i}$ is a one-hot representation in $\mathbb{R}^{V}$ of a discrete token in $\left\{1,...,V\right\}$.
Each token $w$ has an associated embedding ${\bm{e}}_{w}\in\mathbb{R}^{D}$, with fixed norm $\sqrt{D}$ to match the norm of a random gaussian sample in dimension $D$ used to noise clean data.
We denote by ${\bm{E}}\in\mathbb{R}^{D\times V}$ the matrix of all embeddings.

We define our diffusion process in embedding space, rather than in token space. To that end, we define a forward *discrete-to-continuous* step $q_{{\bm{V}}}({\bm{x}}_{0}|{\bm{w}})=\mathcal{N}({\bm{E}}{\bm{w}},\sigma_{0}^{2}{\bm{I}})$, where $\sigma_{0}$ is a constant scale factor with a similar order of magnitude as $\beta_{1}$.
Conversely, we define a reverse *continuous-to-discrete* step $p_{\bm{R}}({\bm{w}}|{\bm{x}}_{0})=\prod_{k=1}^{N}\mathrm{Cat}(w_{k}|{\bm{E}}^{\prime}({\bm{x}}_{0})_{k})$, where ${\bm{R}}\in\mathbb{R}^{V\times D}$ is a learnable readout matrix initialized to ${\bm{E}}^{\top}$ and $\mathrm{Cat}(w_{k}|{\bm{l}})$ is the softmax probability of token $k$ with logits ${\bm{l}}\in\mathbb{R}^{V}$.

To train the readout step, we add a reconstruction loss to $\mathcal{L}_{\text{diffusion}}$ during training.
Conveniently, it naturally arises when deriving the VLB of $p_{\theta}({\bm{w}})$ with this discretization step , introducing a simple cross-entropy loss to maximise $p_{\theta}({\bm{w}}|{\bm{x}}_{0})$:

$$ $\mathcal{L}_{\text{recon}}=\mathbb{E}_{{\bm{w}}\sim\mathcal{D},{\bm{x}}_{0}\sim q_{{\bm{V}}}({\bm{w}})}[-\log p_{\bm{R}}({\bm{w}}|{\bm{x}}_{0})],\qquad\text{with}\qquad\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{diffusion}}+\mathcal{L}_{\text{recon}}.$ (5) $$

Contrary to what is done in , we do not learn the embedding matrix ${\bm{E}}$, as we identified that it was empirically unstable and could lead to drops in unigram entropy. The reconstruction loss $\mathcal{L}_{\mathrm{recon}}$ therefore only depends on the trainable readout weights ${\bm{R}}$.

At sampling time, we run the reverse process for $T=1000$ steps, ultimately yielding a continuous embedding $\overline{{\bm{x}}}_{0}$ of size $d_{\text{embed}}$.
We multiply it by ${\bm{R}}$ to obtain logits in $\mathbb{R}^{V}$, and then use the index of the maximum component to convert it to a token $w_{i}$, with $i=\operatorname*{arg\,max}_{1\leq j\leq V}({\bm{R}}\,\overline{{\bm{x}}}_{0})$.
This entails running $T$ full forward passes which is quite expensive compared to cached AR sampling; however each forward pass computes all timesteps at once which is naturally parallelisable.
Further, we hope to benefit from many diffusion sampling improvements to get $T$ down to low double-digits.

Self-conditioning .
In standard diffusion sampling, at each timestep $t$ the denoising network generates an estimate $\overline{{\bm{x}}}_{0}^{t}=\hat{{\bm{x}}}_{0}({\bm{x}}_{t},t,\theta)$ of ${\bm{x}}_{0}$ given only ${\bm{x}}_{t}$ as input.
Self-conditioning progressively refines ${\bm{x}}_{0}$ estimates by passing the estimate $\tilde{{\bm{x}}}_{0}^{t+1}$ obtained at the previous sampling step as input to the denoising network; the self-conditioned estimate is then defined as $\tilde{\bm{x}}_{0}^{t}=\hat{{\bm{x}}}_{0}({\bm{x}}_{t},\tilde{\bm{x}}_{0}^{t+1},t,\theta)$, and sets the diffusion direction. In practice conditioning is performed by concatenating ${\bm{x}}_{t}$ and $\tilde{{\bm{x}}}_{0}^{t+1}$ on the feature axis.
To approximate the inference behavior at
train time while remaining computationally efficient, we compute a first estimate $\overline{{\bm{x}}}_{0}^{t}=\hat{\bm{x}}_{0}({\bm{x}}_{t},0,t,\theta)$ with the self-conditioning set to zero, then perform a second forward pass using a stop gradient on $\overline{{\bm{x}}}_{0}^{t}$ to obtain $\tilde{{\bm{x}}}_{0}^{t}=\hat{{\bm{x}}}_{0}({\bm{x}}_{t},\overline{{\bm{x}}}_{0}^{t},t,\theta)$.
The denoising network is then optimized using the output from the two forward passes in order to estimate ${\bm{x}}_{0}$ accurately with and without self-conditioning.

Equipped with these 3 components we can train models to generate text, though only unconditionally.
To add conditional generation to our system’s capabilities, we use two additional methods.

### 3.2 Span masking and guidance for conditional text generation

By design diffusion models for text generation are flexible and can handle a wide variety of infilling tasks.
This is a key advantage over the predominant auto-regressive language models that typically generate text in a left-to-right fashion.

Span masking.
We train our model on a rich set of infilling tasks with the following method.
We split ${\bm{x}}_{0}$ between two set of tokens, diffusion tokens ${\bm{x}}$ over which we apply diffusion and optimize the diffusion loss from Eq. [4](#S3.E4), and conditioning tokens ${\bm{c}}$ that remain fixed. Conditioning tokens ${\bm{c}}$ are defined by a binary conditioning mask ${\bm{m}}$ set to one on conditioning positions and zero on positions to be infilled.

We sample conditioning mask ${\bm{m}}$ randomly as follows.
Given a sequence of length $L$ and a maximum number of spans $M$, we sample a number of spans $n$ uniformly in $[1,M]$.
Span starting positions are defined by $n-1$ integers $(i_{1},...,i_{n-1})$ sampled uniformly without replacement and sorted in increasing order to satisfy $0<i_{1}<...<i_{n-1}<L$.
The tuple $(i_{1},...,i_{n-1})$ partitions the sequence of tokens in $n$ spans satisfying $\mathbb{E}[i_{k}|n]=\frac{k}{n}L$.
The conditioning mask ${\bm{m}}$ is defined using even spans for conditioning and odd spans for infilling, and then ${\bm{m}}$ is flipped with a $50\%$ probability.
The case $n=1$ corresponds to unconditional generation; we then set ${\bm{m}}$ to 0 everywhere.

This span masking strategy defines a collection of text generation tasks with a large variety of conditioning which on average evenly splits the sequence between conditioning and infilling spans.
It enables conditional generation, and opens the door for additional diffusion improvements.

Guided diffusion.
Guidance  often improves the sample quality of conditional diffusion models.
We use *classifier-free* guidance , which alleviates the need for a separately-trained guide model. In the conditional case, our estimator $\tilde{{\bm{x}}}_{0}$ is now a function $\hat{\bm{x}}_{0}({\bm{x}}_{t},{\bm{c}},\tilde{\bm{x}}_{0}^{t+1},t,\theta)$, where ${\bm{c}}$ are fixed conditioning tokens.

During training, with fixed probability the conditioning tokens ${\bm{c}}$ used in the estimator $\hat{{\bm{x}}}_{0}$ are dropped and set to a null label $\emptyset$ equal to zero. During sampling, the model prediction is extrapolated in the direction of $\hat{{\bm{x}}}_{0}({\bm{x}}_{t},{\bm{c}},\tilde{{\bm{x}}}_{0}^{t+1},t,\theta)$ and away from $\hat{{\bm{x}}}_{0}({\bm{x}}_{t},0,0,t,\theta)$ as follows:

$$ $\tilde{{\bm{x}}}_{0,s}^{t}=\hat{{\bm{x}}}_{0}\big{(}{\bm{x}}_{t},0,0,t,\theta\big{)}+s\,\cdot\,\Big{(}\hat{{\bm{x}}}_{0}\big{(}{\bm{x}}_{t},{\bm{c}},\tilde{{\bm{x}}}_{0}^{t+1},t,\theta\big{)}-\hat{{\bm{x}}}_{0}\big{(}{\bm{x}}_{t},0,0,t,\theta\big{)}\Big{)},$ (6) $$

where $s\geq 1$ is the guidance scale.
Remark that we jointly drop conditioning and the self-conditioning $\tilde{{\bm{x}}}_{0}^{t+1}$, concretely setting both values to zero.
Classifier-free guidance allows leveraging both the unconditional and conditional abilities of a model to improve its conditional generations.

## 4 Experiments

**Table 1: Sed samples on unconditional generation, fill-in-the-middle and several spans in-filling.**
| Task | Samples |
| --- | --- |
| Unconditional | We make use of the very best supplies and solutions to ensure that the work is going to stand up to the test of time, and we help you save money with techniques that do not change the quality of your mission. We’ll achieve this by offering you the best deals in the field and avoiding pricey mistakes. If you want to spend less, Refrigerator Unit Repair Guys is the company to contact. |
| Fill-in-the-middle | A year ago in Paris, I had the opportunity to take a field trip to La Rite-en-Laurences International de France where I met David Nigel Johnson, a professor of social studies. What a great trip and what a great day! |
| Spans in-filling | There was no evidence, only fleeting glimpses of the killer and his fate. In fact, it seemed that there was no evidence. It was all guesswork, and one of the most unusual murder cases throughout history. |

### 4.1 Training details

We train all our models on the C4 dataset , using a SentencePiece tokenizer  composed of 32000 words.
We use a non-causal transformer model  as our diffusion model (see Appendix [A](#A1) for details).
Sed models are trained with sequence length 256, while for ablations models are trained with sequence length 128.
We insert uniformly, i.e. not necessarily at the end of the sequence, 10% of padding tokens in the training set to allow Sed models to generate samples of varying size and provide more flexibility.

To generate word embeddings, we train a BERT model of fixed size ($150$m parameters) and feature dimension $d_{\text{model}}=896$.
The diffusion space is defined by the initial lookup table of this BERT model.
We bottleneck the dimension of the word embeddings $d_{\text{embed}}$ and add a linear projection layer from $d_{\text{embed}}$ to $d_{\text{model}}$ at the beginning of the model.
We found this helped diffusion (see section [4.4](#S4.SS4)).

Sed models are trained with a cosine noise schedule , with $\beta_{1}=2.10^{-3}$, $\sigma_{0}=10^{-2}$ and $T=1000$.
We use batches of 65.536 tokens, thus for sequence length 256 the batch size is set to 256.
We use a maximum span count of 5 for all runs except for its specific ablation.
We train Sed models at two different scales: Sed-S ($135$m parameters, $10^{6}$ training steps) and
Sed-L ($420$m, $2.10^{6}$ steps).
Their detailed architectures can be found in Appendix [A](#A1).

### 4.2 Validation

While optimizing the perplexity of AR models for text leads to improved language models, directly optimizing the ELBO of diffusion models for images does not correlate strongly with sample quality as observed by
.
For images, the sample based metric FID has been introduced as a measure of sample quality and is now widely adopted.
Similarly, we need a sample-based metric for text generation that is reliable and allows comparison between a large variety of generative models. To provide a fair comparison to AR models, we rely on three metrics.

The first metric measures how likely the samples produced by a model are according to an AR language model with 70B parameters, trained on 1.4B tokens ; we denote this metric AR NLL for auto-regressive negative log-likelihood.
It provides a continuous measure of sample quality that has proven useful when combined with a measure of sample diversity, e.g. in the development of nucleus sampling for improved AR model decoding.

To measure diversity we rely on a second metric, the unigram entropy of samples, which helps balance the AR NLL that can be gamed by unnatural repetitive samples.
For both these metrics, our target is the score of the validation set data.
Deviating from the data unigram entropy in particular is a sign of degenerate modeling.

Though this initial combination has provided us with a reliable signal to iterate over our model design, it remains imperfect; it too can be gamed, though it is harder to do so.
To address this limitation, we also report human preferences.
We presented 6 colleagues with 20 pairs of samples for each comparison, asking them to pick the best one.

For all three metrics, we report results on two tasks: unconditional language modeling and suffix in-filling, the later a heavily conditioned task.

### 4.3 Results

Figure: Figure 1: Comparison of sample quality and diversity of Sed versus AR models on suffix in-filling. Sed uses guidance with scales in $\{1,2,4,8\}$ and AR uses nucleus sampling with a top-$p$ in $\{1.00,0.95,0.90,0.85\}$. Top-right points are Sed models with a guidance scale of 1 or AR models with a top-$p$ of 1.
Refer to caption: /html/2211.04236/assets/assets/diffusion_vs_ar_newer.png

Samples. We present samples generated with our Sed models in Table [10](#A3.T10).
We use a single model to perform a wide variety of text generation tasks, such as unconditional generation, filling-in-the-middle or filling several spans of text.
We show strong performance in the unconditional case, with samples that are syntactically correct and stay coherent on long sequences.
In the conditioned case, Sed models are able to infill spans with coherent transitions and links to the conditioning but also exhibit a rich diversity.
By design, Sed yields flexible bi-directional masking models that can perform text generation on a diverse set of conditioned task.
To compare Sed with AR baselines we next restrict conditioning to a prefix and consider a task of suffix in-filling.

Comparison to AR models. To assess the generation ability of Sed, we compare against AR baselines of similar capacity and trained following optimal scaling laws from on suffix in-filling.
We sample a batch of sequences from C4 and use the first 128 tokens as conditioning given to the model to generate a suffix of 128 tokens.
Figure [1](#S4.F1) reports AR NLL and unigram entropy of the generated suffixes for AR and Sed models.
As a reference point, we compute the AR NLL and unigram entropy of the ground truth C4 suffixes and report it on the plot.
Several methods can be used to improve sampling quality at the cost of samples diversity; we use nucleus sampling for AR models and guidance for Sed models.
We show the impact of guidance on samples quality in Table [3](#S4.T3).
To our knowledge, we are the first to show sample quality improvement when using guidance for text generation.

As shown in Figure [1](#S4.F1), both Sed-S and Sed-L perform strongly when compared against AR baselines – even though we report a metric favoring AR models on a task AR models have been designed to optimize.
Similar to nucleus sampling for AR models, guidance has a strong positive impact on sample quality that is both observed quantitatively with improved AR NLL in Figure [1](#S4.F1) and qualitatively in Table [3](#S4.T3).
We observe that using a top-$p$ nucleus sampling below $0.8$ for AR models or a guidance scale above $4$ for Sed models leads to samples exhibiting a lot of repetitions, a degenerate case reflected by a lower entropy of samples even though sample AR NLL improves.

Our human preference scores temper our observations in Table [4](#S4.T4).
They show that our NLL and entropy metrics do not tell the whole story, as humans still prefer AR models at equivalent size.
While Sed-L performs slightly worse than AR-L (38% preference in suffix in-filling, 44% on unconditional generation), its scores remain comparable. Sed-L is roughly on par with AR-S.

Finally, we compare Sed and AR models’ qualitative examples with short prompts in Table [2](#S4.T2).

**Table 2: We compare Sed-L (guidance scale 2.5) and AR-L (nucleus sampling, $p=0.95$) samples.**
| Sed L | AR-L |
| --- | --- |
| You’re going to love wearing this traditional tee from our latest Wilson collection. Designed in a scrapped floral styled knit with a sleeve of asymmetric lines across the round sole. Lightly fluffy, the square pleats will take you right to | A Koda Ram 25 is presented in sedan and a Maxima saloon. Based on the Acenta car, the powerful Koda 2014 hits Indian roads in the ”Maxima” body-con shape. Being powered by a Hyundai i20 1.4 litre diesel engine, the Koda 2014 is coupled |
| The beaver is an interesting animal that lives in rivers and lakes. He is not mainly concerned with finding wolves and dolphin but also has a great hunger for fish. The beaver has sharp legs, large eyes, and a black coat | The beaver is an interesting animal that lives in rivers and streams. It is usually seen in big numbers in the fields or upstream, and is quite docile. On cold days when its pattern is perfect, the beaver will have some interesting, and sometimes |
| Once upon a time in Spain, Leonardo Pueleva had the pleasure of meeting guests at Spanish restaurant, Buva Casinos. While driving, he got a chance to get to know the people behind the restaurant and, of course, how they made his experience very interesting. After his conversation, he got to | Once upon a time in Spain, which seems pretty much the same way now, the question that was posed to each of us at the end of our interview was ”would you like to see Froome one day?” In retrospect, after our interview, we have grown ever closer to that answer. As you will read in the article, I know that |

**Table 3: Impact of guidance on samples quality using our Sed model.**
| Guidance | 1.0 | 2.5 | 5.0 |
| --- | --- | --- | --- |
|  | In the cold, cold night sky, a fairy princess sits in a chair and surrounded by tea leaves in a pond. Meanwhile, she bies back into the cold, with bluish hair on her hips and elbows on her forehead - and her fingers numbed by the freezing temperature. | In the cold, cold night of November 2018, a little girl sits in a chair hidden under a light blanket on a patio. Meanwhile, she bends back into the chair with bluish hair on her forehead, her hands on her face, her fingers numbed by the freezing temperature. | In the cold, cold night of December, my oldest daughter sits in a chair accentuated in cotton fabrics and a pillow. Meanwhile, she yearns straight in the cold air, her wrists covering her neck, her eyes straight on her forehead, and her fingers numbed by the freezing temperature. |
|  | Barbara was one of our many wonderful women that really helped so I am so blown off by her purpose, civility; and adversity. Once I started interacting with her, it proved to me that no matter how hard this was, she always strove for excellence. | Barbara was one of the most gifted women in the world. She was creative and stood up by her integrity and civility; against adversity. Although she placed herself higher than her peers, it proved to me that no matter how hard this was, she always strove for excellence. | Barbara was one of the most brilliant women in the world. She was amazing in her heart, her spirit, her mind and in the soul. She never turned people off in her absence. It proved to me that no matter how hard this was, she always strove for excellence. |

**Table 4: Sed-L vs other models human preference scores on conditional and unconditional tasks.**
|  | Sed-S (cond) | AR-S (cond) | AR-L (uncond) | AR-L (cond) |
| --- | --- | --- | --- | --- |
| Sed-L | 63.4% $\pm$ 4.3% | 51.0% $\pm$ 5.0% | 43.8% $\pm$ 4.4% | 37.7% $\pm$ 4.4% |

### 4.4 Ablations

**Table 5: Ablation of the proposed Sed approach on unconditional generation. Both self-conditioning and embeddings pretraining play a key role in the model performance.**
| Diffusion space | Self-conditioning | Unigram entropy | AR NLL |
| --- | --- | --- | --- |
| Bits | ✗ | 6.97 | 7.01 |
|  | ✓ | 7.47 | 6.05 |
| Random embeddings | ✗ | 6.90 | 6.80 |
|  | ✓ | 6.86 | 5.31 |
| Pretrained embeddings | ✗ | 6.75 | 5.66 |
|  | ✓ | 6.77 | 4.57 |
| Data | – | $6.70\,\pm\,0.04$ | $1.81\,\pm\,0.15$ |

**Table 6: On unconditional generation, self-conditioning results in better sentence modelling, pretrained embeddings enhances topic modelling.**
| Random embeddings | Random embeddings | Pretrained embeddings |
| --- | --- | --- |
|  | with self-conditioning | with self-conditioning |
| did the buildingroom granted a lighter distance. On it though, salaries about clients that a child, which dispersed gluc so many events and certainly wanted the Mother’s project, discovered by their child would keep | Tree brings the sound, bearing features and capabilities that we are set in. For the first time, she uses a customizable framework to use that we help students publicly solve the weather conditions that we only offer students | almost six decades ago - 72 percent of Americans didn’t feel they’d actually rent their own cars this year. Conversely, 90 percent of Americans feel that the decision to rent a car is something they feel it’s impossible |

**Table 7: Word embeddings with small dimension have higher AR likelihood.**
| Embed. dim. | 16 | 32 | 64 | 128 | 256 | 896 |
| --- | --- | --- | --- | --- | --- | --- |
| AR NLL | 4.65 | 4.57 | 4.71 | 4.61 | 4.77 | 4.92 |

Self-conditioning and embedding pretraining.
Results from Table [5](#S4.T5) and samples from Table [6](#S4.T6) show the influence of both the diffusion space and self-conditioning.
AR NLL decreases very significantly when using self-conditioning, regardless of the rest of the setup.
Diffusing at the bit-level  yields very high NLLs.
While using random embeddings performs markedly better, using pretrained embeddings results in further improved numbers.

Samples from Table [6](#S4.T6) highlight that models trained on random word embeddings exhibit topic modelling abilities with the co-occurrence of words like child and mother even though the paragraph remains globally incoherent and meaningless tokens like gluc are generated.
Self-conditioning dramatically improves sample quality; the diffusion model gets the low-level structure right and generates syntactically correct sentences, even though the global text is not intelligible.
Combining self-conditioning and pretrained embeddings leads to globally coherent paragraphs that stay on topic with proper sentence structure.

Embedding dimension.
An important design choice for SED is the word embeddings space.
We study the influence of pretrained embedding size in Table [8](#S4.T8).
Surprisingly, there is a threshold after which performance degrades when increasing the dimension of embeddings.
We visualize the forward process for different embedding sizes by displaying the nearest neighbor of a noised token while running the forward process.
In high dimension we observe that the nearest neighbor of a noised token remains the starting token itself until it switches to a completely random, unrelated token.
In low dimension, we often observe that the closest neighbor of a noised token goes through several semantically related tokens (nearest neighbor of the starting token) before ultimately becoming random.
We hypothesize that the random walks defined by diffusion are more likely to drift towards neighbors of the starting token in low dimension.
As a result, when diffusing in low dimension information is destroyed in a more semantically meaningful fashion, which leads to an easier learning problem for the denoising function.

Number of spans.
In order to enable in-filling, we train the model not only to do unconditional generation but also to conditionally fill spans of tokens.
For each data point we sample a span number uniformly at random and span delimiters to generate the span mask.
Picking the maximum allowable number of spans has a significant effect on model performance, as we can see in Table [8](#S4.T8).
Somewhat counter-intuitively, adding span masking improves even unconditional generation NLLs.
It also appears that using a relatively high maximum span number is optimal.
We hypothesize that this results in a varied mix of task difficulty at training time, between ”easy”, very conditioned problems on the one hand and ”harder”, unconditional ones on the other.

Scaling. We show encouraging results when scaling from Sed-S (150m) to Sed-L ($420$m).
We train both models on sequences of $256$ tokens and report a AR NLL of $4.20$ for Sed-S compared to $3.68$ for Sed-L.
This improvement translates to improved sample quality, as is confirmed by our human preference scores, which are much higher for the larger model (63%, see Table [4](#S4.T4)).

## 5 Limitations

While our results are promising and show that continuous diffusion for text can be an exciting alternative to AR models, the current approach does present some significant limitations.

First, much more could be done in terms of model tuning, including scaling to much bigger models to better understand Sed’s limits, and to be able to compare it with state-of-the-art AR models.
Our training regime in particular would certainly benefit from more hyperparameter optimisation.

Second, one compelling reason we chose to explore continuous diffusion for text is to leverage the improvements produced by the literature on image generation.
While we have ported some (e.g. self-conditioning), a lot more remains unexplored.
The most obvious example is the sampling process itself, where the number of required steps has been considerably reduced for images (e.g. goes from 1000 to 35, and  all the way down to 4 on simple images).
Our current sampling is very inefficient, and this direction is one of the first improvements to make over Sed.

Third, Sed crucially relies on diffusing in a pretrained embedding space.
This means relying on a second model, and using embeddings that may not be optimal for diffusion.
Ideally, we’d train the full model end-to-end, which could yield even better results.
While  found some success with this approach, it was in a specific setting at a small scale; in practice we found it difficult to avoid competition between the diffusion and reconstruction loss.

Finally, our work would benefit from improved metrics in the experimental section.
Because the current state of the art involves AR models, the field lacks established benchmarks for tasks diffusion models are potentially better suited for, such as text in-filling.
We opted for a reasonable mix, evaluating the negative log-likelihood of generated samples according to a very strong AR model as well as their token entropy and complementing it with a human evaluation.
However, both NLL and unigram entropy are gameable (e.g. AR models assign very low NLL to repetitive snippets, and long enough repetitions can fool even entropy).
Further, our NLL is inherently tied to its AR model and could thus be providing an unfair advantage to AR models.
All told, we still found both metrics quite useful for measuring research progress, and our human evaluation confirmed our results.
Moving forward, defining a clean in-filling benchmark would help produce even more convincing results.

## 6 Conclusion

We propose Sed, the first generally-capable continuous diffusion model for text generation.
Sed models can perform both conditional and unconditional generation, and their performance rivals AR models while being more flexible in their use (e.g. enabling in-filling).
We demonstrate their performance and study the impact of the main design choices.

Despite its limitations, this work lays the foundation for more exciting research.
Promising directions include speeding up the sampling following the lessons learnt in the image domain, devising better embedding spaces for diffusion and investigating new in-filling capabilities.

## Appendix A Model architecture

For both the AR and the Sed models, we use the same transformer architecture, which are similar to those described in , with relative positional encoding as described in in the attention blocks, and with a 4x expansion and a Gelu non-linearity in the feed-forward blocks.
The architecture hyper-parameters are detailed in table [9](#A1.T9).
Noised word embeddings, ${\bm{x}}\in\mathbb{R}^{N\times D}$, are first passed through a linear projection that operates on each embedding independently to get a projected embedding whose feature dimension matches the width of the transformer, $d_{\text{model}}$.
At diffusion step $t$, we compute a time embedding as a sinusoidal position embedding of size $d_{\text{model}}$, which is then passed into a $d_{\text{model}}\times d_{\text{model}}$ linear layer and added to the projected embedding. We add a linear output projection layer ${\bm{E}}^{\prime}$ which takes the output of the transformer $y\in\mathbb{R}^{N\times d_{\text{model}}}$ and projects each element $(y_{i})_{1\leq i\leq N}$ back to the same size as the word embeddings. When using self-conditionning, we modify the input to the model by concatenating ${\bm{x}}$ and $\hat{{\bm{x}}}_{0}$ along the feature axis before passing them to the input projection layer.

**Table 9: Model hyper parameters.**
| Model | number of layers | $d_{\text{model}}$ | number of heads | head size |
| --- | --- | --- | --- | --- |
| S | 12 | 896 | 16 | 64 |
| L | 12 | 1536 | 16 | 128 |

## Appendix B Forward diffusion process visualization

To support the discussion on word embeddings dimension from Section [4.4](#S4.SS4), we present a visualization of the forward diffusion process.
Given starting tokens ${\bm{x}}_{0}$, we project the noised tokens ${\bm{x}}_{t}$ of the forward process at step $t$ to their nearest neighbor among word embeddings ${\bm{E}}$ to obtain ${\bm{w}}_{t}$.
We then store the 128 nearest neighbors $\mathcal{N}({\bm{w}}_{0})$ of starting tokens ${\bm{w}}_{0}={\bm{x}}_{0}$ and define the rank $r_{t}$ of ${\bm{w}}_{t}$ at its index in $\mathcal{N}({\bm{w}}_{0})$.
We display ${\bm{w}}_{t}$ and highlight it in green if $r_{t}$ is close to zero (meaning ${\bm{w}}_{t}$ is a close neighbor of ${\bm{w}}_{0}$) and in increasingly red colors otherwise. We present the first 16 nearest neighbors of ${\bm{w}}_{0}$ in Figure [2](#A2.F2) and provide an illustration of the color code used for highlighting.
Figure [3](#A2.F3) shows an instance of the forward diffusion process while diffusing on embeddings of with a high dimension of 896 and Figure [4](#A2.F4) shows diffusion on embeddings with a lower dimension of 32.

We observe that in high dimension, the noised token’s closest neighbour remains the original token up until the point where any token could be its closest neighbour.
The diffusion random walk does not seem to pass through the neighbourhoods of semantically-related tokens.
We hypothesize that the root cause of this issue is that the embedding space is mostly empty (with only 32000 points in $\mathbb{R}^{896}$, as $d_{\text{embed}}=896$); and that embeddings are potentially concentrating in a lower-dimensional space.

In contrast, in lower dimension we see meaningfully-related tokens appear as the corruption progresses (‘brown’ becomes ‘grey’, ‘quick’ becomes ‘swift’, ‘over’ becomes ‘underneath’ etc).
We believe this more gradual information destruction is beneficial for the diffusion model.

Figure: Figure 2: Nearest neighbors of tokens from the sentence ”the quick brown fox jumps over the lazy dog”
Refer to caption: /html/2211.04236/assets/assets/neigh_bert.png

Figure: Figure 3: Visualization of the forward diffusion process up to 300 steps when diffusing on embeddings with dimension 896.
Refer to caption: /html/2211.04236/assets/assets/neigh_forward_dim896.png

Figure: Figure 4: Visualization of the forward diffusion process up to 300 steps when diffusing on embeddings with dimension 32.
Refer to caption: /html/2211.04236/assets/assets/neigh_forward_dim32.png

Figure: Figure 5: Reverse diffusion process of Sed-L with guidance scale 2.5.
Refer to caption: /html/2211.04236/assets/assets/diffusion_reverse_process_3.png

## Appendix C Additional samples

**Table 10: Samples from Sed-L.**
| This course is essential for the environment in which students choose the curriculum option study at JHD. Geographical Integration is at the heart of this department. Along with the urban infrastructural innovations of the mid-1990’s and social issues in the 21st Century. People within the department regularly exemplify the concept of real integration. |
| --- |
| That did trigger some rewarding words or insights to begin preparing their kid for their future. Luckily, Mikaela has been nice enough to tolerate my questions about what parents can do to help, even during her summer vacation. |
| I went to college at Boston University. After getting my degree, I decided to make a change! I enrolled in outdoor schools. After getting my degree, I loved jet skiing, offshore fishing, and Kitesurfing. The list became growing. More importantly, I started a career by fishing at sea. Now, I can’t get enough of the Pacific Ocean! |
| The beaver is an interesting animal that lives in rivers and waterfalls across Puerto Rico. It loves kayaking, fishing, and swimming. But, you want to know what are the animals behind the beaver? |
| A year ago in Paris, I had the opportunity to take a field trip to La Rite-en-Laurences International de France where I met David Nigel Johnson, a professor of social studies. What a great trip and what a great day! |
| A year ago in Paris, my friends and I went on a dirt road trip to the city.<br>I remember walking through the church, its beautiful square, narrow streets lit with memorials and passing a terrible Catholic Bishop I am used to - what a sad day… |
| There was no evidence, only fleeting glimpses of the existence of cognitive disability. There was no luck and no solid science. It was all guesswork, the blinding prospect of pneumonia could spur imagination at the possibility of brain damage. |