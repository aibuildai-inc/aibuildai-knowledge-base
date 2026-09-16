---
arxiv_id: "2210.16886"
title: "DiffusER: Discrete Diffusion via Edit-based Reconstruction"
year: 2023
source: arxiv2md
---

## Abstract

Abstract In text generation, models that generate text from scratch one token at a time are currently the dominant paradigm. Despite being performant, these models lack the ability to revise existing text , which limits their usability in many practical scenarios. We look to address this, with \method ( Diffus ion via E dit-based R econstruction), a new edit-based generative model for text based on denoising diffusion models – a class of models that use a Markov chain of denoising steps to incrementally generate data. \method is not only a strong generative model in general, rivalling autoregressive models on several tasks spanning machine translation, summarization, and style transfer; it can also perform other varieties of generation that standard autoregressive models are not well-suited for. For instance, we demonstrate that \method makes it possible for a user to condition generation on a prototype, or an incomplete sequence, and continue revising based on previous edit steps. 1 1 1 Supplementary material will be released at https://github.com/machelreid/diffuser

## 1 Introduction

Revision and editing are central to how humans produce content; we write and revise emails and papers, gradually produce works of art, and iterate on plans for a project. Despite this, the most dominant paradigm in text generation is purely autoregressive, producing text left-to-right in a single pass . Although models employing this single-pass form of generation are highly performant, they are limited by the inability to refine existing text.

Figure: Figure 1: \method’s text generation process. Orange represents replacements, blue represents insertions, red represents deletions, and white represents keep operations. This process largely imitates a natural editing process .
Refer to caption: /html/2210.16886/assets/x1.png

To address this, we propose \method: Diffusion via Edit-based Reconstruction, a flexible method to apply edit-based generative processes to arbitrary text generation tasks. Specifically, we take inspiration from diffusion models , generative models that generate by way of incremental denoising steps,
and adapt this approach to the text generation paradigm with a formulation similar to natural editing processes.

The dominant approach for text generation is autoregressive (AR, “left-to-right”) models . These models have been greatly improved in recent years, primarily due to steep increases in the size of the models and datasets used for training .
Research into non-autoregressive approaches aims to enable more general modes of text generation , but has so far struggled to match the accuracy of models trained in an AR fashion. A thus far separate line of models has taken the perspective of modeling text edits for specific tasks: e.g. style transfer , sentence fusion , and grammatical error correction . \methodunifies these two perspectives by enabling edit processes to be applied to general purpose text generation without compromising performance or requiring external supervised data . This design enables it to both generate and edit text, including text produced by other models such as auto-regressive Transformers, making this a natural extension of the text generation paradigm.

models text generation as a series of diffusion steps at the token level. This form of generation allows us to develop a synthetic formulation of natural editing processes using edit-based corruption and reconstruction. Our method starts from an arbitrary sequence (either a prototype generation, randomly sampled tokens, or a null sequence) and progressively edits it into the final sequence guided by the Levenshtein edit operations of Insert, Delete, Keep, and Replace as shown in Figure [1](#S1.F1). This enables flexible editing in a range of contexts, including machine translation, summarization, style transfer, while also allowing for the possibility of taking outside input to guide and constrain generation.

Learning these edit-based diffusion processes required several innovations over standard autoregressive and MLM-style iterative generation approaches , including forming edit-based corruption and reconstruction processes for training (Sec [3](#S3)), as well as techniques to improve the quality of decoding sequences across both timesteps and token-level generations, for which we introduce a 2D beam search approach (Sec [3.6](#S3.SS6) & Sec [3.5](#S3.SS5)).

To demonstrate the effectiveness of \method, we test our method on three text generation tasks: machine translation, abstractive summarization, and text style transfer, and show on-par or improved performance compared to purely autoregressive, single-pass and non-autoregressive methods. We also provide qualitative samples of the edit processes learned by the models in different settings and analyses on training and inference speeds, as well as the relationship between edit steps and performance.

Overall, we demonstrate the potential of edit-based generative models to offer 1) more performant generation, 2) greater interactivity between different models (as we can now perform edits in the discrete space on model generated output), and 3) more flexible/controllable generation.

## 2 Background

operates at the intersection of text generation, editing processes, and diffusion models. We first provide the background and intuition of these three techniques.

### 2.1 Text Generation

Most text generation models used in NLP today are autoregressive in nature. In this paradigm, given a sequence ${\mathbf{s}}=[{\mathbf{s}}_{0},{\mathbf{s}}_{1},\dots,{\mathbf{s}}_{N}]$, one can model the likelihood of the entire sequence $P({\mathbf{s}})$ by modeling the probability of predicting each token in an autoregressive, often left-to-right, manner. This formulation, where the likelihood of a token $p({\mathbf{s}}_{t})$ is conditioned on its predecessors ${\mathbf{s}}_{<t}$, is shown below :

$$ $P({\mathbf{s}})=\prod_{i=0}^{N}p({\mathbf{s}}_{t}|{\mathbf{s}}_{t-1},{\mathbf{s}}_{t-2},\dots,{\mathbf{s}}_{0})$ (1) $$

Models trained with this objective can then be sampled from, or searched over (e.g. using beam search), to provide generations in downstream tasks such as machine translation or summarization.

Non-autoregressive models are a different variety of generative models, in which a sequence is generated in a single pass (removing the autoregressive conditioning on previously generated tokens) with multiple revision-level passes, often in the name of efficiency.

### 2.2 Editing Processes

Editing processes are a paradigm for modeling text by way of incremental revisions, taking inspiration from the the way humans generate text. Specifically, let $X=\{{\mathbf{x}}_{0},{\mathbf{x}}_{1},\dots,{\mathbf{x}}_{R}\}$ be a series of $R$ versions of a document, where ${\mathbf{x}}_{0},{\mathbf{x}}_{i},{\mathbf{x}}_{R}$ represents the initial, intermediate (at timestep $t$), and final/current state of a document, respectively. Using editing processes, we can model the probability of this series of documents versions occurring consecutively as follows:

$$ $\displaystyle p(X)=\prod^{R}_{i=0}p({\mathbf{x}}_{i}|{\mathbf{x}}_{0}^{i-1})$ (2) $$

With this formulation, editing processes can also be used to calculate the probability of only the final document while taking into account previous revisions, which is not possible in the traditional text generation setup as intermediate revisions are not explicitly known.

$$ $\displaystyle p({\mathbf{x}}_{R})=\sum_{\tilde{X}\in\{\tilde{{\mathbf{x}}}^{R}_{0}|\tilde{{\mathbf{x}}}_{R}={\mathbf{x}}_{R}\}}p(\tilde{X}).$ (3) $$

### 2.3 Diffusion Models

We now make the connection between editing processes and diffusion models . Continuous diffusion processes are commonly applied in computer vision tasks to iteratively convert a sample of noise into an image. This can be seen as an edit process in which the model iteratively *edits* a noisy image to bring it closer to a final, complete image.
These continuous diffusion models are often trained by modeling a Markov chain ${\mathbf{x}}_{T}\dots\mathbf{x}_{t}\dots\mathbf{x}_{0}$, where ${\mathbf{x}}_{0}$ represents the original image and ${\mathbf{x}}_{T}$ represents Gaussian noise. This chain is typically produced by incrementally adding Gaussian noise to ${\mathbf{x}}_{t}$ to form ${\mathbf{x}}_{t+1}$ (known as the *forward* or *corruption* process), wherein a model parameterised by $p_{\theta}$ is trained to *reverse* (or “*denoise*”) this process to form the chain $\sum^{T}_{i=1}p_{\theta}({\mathbf{x}}_{t-1}|{\mathbf{x}}_{t})$.

Analogized to text, this allows us to formulate natural edit processes as a discrete diffusion process in which a null string or a prototype is iteratively edited into free form text.
Our \methodmethod (Figure [1](#S1.F1)) takes inspiration from this process, but parameterises the corruption process by way of sampled discrete edit operations applied over a discrete sequence of tokens. The success of our method supports the findings in the vision domain , where it is found that diffusion models can learn to invert arbirtary transformations.

Previous work in diffusion models has largely focused on computer vision , in which the diffusion process is applied to raw image values. Within the context of natural language, both discrete diffusion models using only replacement operations (either applied to random tokens or masked tokens) , and continuous diffusion over word embeddings have been proposed. Our model is a more flexible approach, using all four edit operations, towards diffusion models when compared with this work owing to its edit process formulation, and is also more compatible with current models (e.g. AR bootstrapping).

## 3 \method

, being a diffusion-based method, has two main procedures: corruption and denoising. Unlike previous work in which this procedure is relatively inflexible (e.g., due to length restrictions and/or using continuous representations for the basis of the diffusion process), both our corruption process and denoising process are based on Levenshtein operations, allowing our model to learn to take advantage of the flexibility of text editing when generating.

### 3.1 Edit Operations

Given the central role of the Levenshtein edit operations in our models, we provide a brief overview of each operation and its role in the editing process. We use Figure [1](#S1.F1) as a guide when explaining each operation.
Insert: The insertion operation is used to add new text to a sequence. For example in Figure [1](#S1.F1), “*uses editing processes*” is added by *DiffusER* at timestep $x_{T-2}$.
Delete: The deletion operation erases existing text. In Figure [1](#S1.F1), this is shown when “*These*” gets deleted at timestep $x_{T-2}\rightarrow x_{T-3}$.
Replace: The replacement operation works overwriting existing text with new text. This is shown in Figure [1](#S1.F1) at step $x_{T}\rightarrow x_{T-1}$ where “*filter Toronto guilty trough feel*” is replaced by “*These model guilty named DiffusER*”.
Keep: The keep operation ensures that a portion of the text remains unchanged into the next iteration. This is illustrated in timestep $x_{T-2}\rightarrow x_{T-3}$ where “*model named DiffusER*” is kept.

### 3.2 Edit-based Corruption

The four Levenshtein edit operations described above allow us to transform any arbitrary sequence of tokens into another. This is in contrast to iterative mask replacement, which can only introduce new tokens .
For every timestep $i$, the corruption process $q({\mathbf{x}}_{i}|{\mathbf{x}}_{i-1};{\mathcal{E}}_{t},{\mathcal{E}}_{l})$ is parameterized by two distributions: the distribution over edit types ${\mathcal{E}}_{t}$ (e.g. 60% keep, 20% replace, 10% delete, 10% insert), and the distribution over edit length ${\mathcal{E}}_{l}$. The latter can be parameterized by any distribution over non-negative integers, such as a uniform distribution or a Poisson distribution. For instance, to learn a deletion operation in the reconstruction process, we insert randomly sampled distractor tokens, whereas, to learn an insertion operation we delete a subset of tokens contained in the sequence.

### 3.3 Edit-based Reconstruction

Our generative process is trained via the *Edit-based Reconstruction* (ER) process. ER can be thought of as the opposite of our corruption process, in which we need to find the appropriate edit operations to transform ${\mathbf{x}}_{T}$ to ${\mathbf{x}}_{0}$, by way of ${\mathbf{x}}_{T-1},\dots,{\mathbf{x}}_{1}$.

That is, given a corrupted sequence ${\mathbf{x}}_{T}$, we aim to learn the process by which we can reverse the corruption in the following form.

$$ $P_{\theta}({\mathbf{x}}_{0})=\prod^{T}_{t=0}p_{\theta}({\mathbf{x}}_{t-1}|{\mathbf{x}}_{t})$ (4) $$

Given that, we model the likelihood of each revision ${\mathbf{x}}_{t}$, this modeling procedure can be likened to modeling an edit process . As we include an edit process in our model and use Levenshtein tags for editing, one can think of ER as two distinct steps: identify which edits should take place (tagging process) and deciding which tokens should go in these positions (generative process). This decomposition is shown here:

$$ $p_{\theta}({\mathbf{x}}_{t-1}|{\mathbf{x}}_{t})=p_{\theta}^{\text{tag}}({\mathbf{e}}_{t}|{\mathbf{x}}_{t})p_{\theta}^{\text{gen}}({\mathbf{x}}_{t-1}|{\mathbf{x}}_{t}|{\mathbf{e}}_{t})$ (5) $$

where $p_{\theta}^{\text{tag}}$ parameterises the tagging model to estimate the likelihood of producing a given set of Levenshtein edit operations {Insert ,Delete ,Keep ,Replace } given ${\mathbf{x}}_{t}$, and $p_{\theta}^{\text{gen}}$ parametersies the generator model given sequence ${\mathbf{x}}_{t}$ and edit operations ${\mathbf{e}}_{t}$. This decomposition via edit-operations allows the generation process to be more controllable and more flexible as it allows up to explicitly specify edit types associated with tokens to be edited, rather than leaving both processes to be implicit. We also depict an example of this in Table [1](#S3.T1).

### 3.4 Implementing \method with Transformers

When implemented with Transformers , \methodconsists of two components: a tagger and generator. The tagger, a transformer network, is trained using cross-entropy loss over the ground-truth tag types to predict the edit operations that should be applied to the sequence, in preparation for the next generation step. Then, in the generation step, after removing tokens selected for deletion, we sum a learned embedding to insert and replace types and generate the inserted and replaced sequences autoregressively. Following this, we feed the output of this diffusion step into the tagger and perform another diffusion step. One step of this process can be compared to the reconstruction process used in .

**Table 1: Example diffusion process for machine translation from random tokens.**
| Step 1 | .elf meantime Nano (¡ Aden Prepare hue mere strictlyrights hueHeat Goalsgeordnet LewisSession beet remindersrights rézes redund boldWisconsinPort compl rocks@@actual Parish norm Lawyers Organisation deprecatedince eradicateewerkschaften oyleingebracht naked Lawyers Organisation von Gewerkschaften al contestants negligible GeneIZE etablieren.HT |
| --- | --- |
| Step 2 | .elf meantime Nano (¡ Aden Prepare hueHeat Goalsgeordnet aggravatedabgeordnet LewisSession beet remindersrights rézes redund boldWisconsinPort compl rocks@@oyleingebracht boldWisconsin eingebracht naked Lawyers Organisation von Gewerkschaften al contestants negligible 2400 CLR GeneIZE etablieren.HT isationatar ent |
| Step 3 | .elf meantime Nano (¡ aggravatedabgeordnet containing Tai Prison Kongressabgeordnet und John LewisSession beet remindersrights rézes redund boldWisconsin rézesvorschlag eingebracht naked Lawyers Organisation von Gewerkschaften al contestants negligible 2400 CLR GeneIZE als Bürgerrecht zu etablieren.isationatar ent |
| Step 4 | .containing Tai Prison Kongressabgeordnet und John LewisSession beet remindersrights rézesvorschlag eingebrachtnaked Lawyers Die Kongressabgeordneten Keith Ellison und John Lewis haben einen Gesetzesvorschlag eingebracht, um die Organisation von Gewerkschaften als Bürgerrecht zu etablieren. isationatar ent |
| Target | Die Kongressabgeordneten Keith Ellison und John Lewis haben einen Gesetzesvorschlag eingebracht, um die Organisation von Gewerkschaften als Bürgerrecht zu etablieren. |

### 3.5 Decoding Methods

has an inherently different generation process from a standard autoregressive language generation model—in addition to operating on a sequence/token level (in which generation is composed of generating individual tokens in a single-revision; *intra-revision*), we also operate on a *revision* level (in which the text is expanded across diffusion steps, *inter-revision*). This allows us to experiment with different methods for decoding on both the intra-revision (single sequence level) and inter-revision levels (multiple version level), which we explain below.

#### Beam Search

One method for decoding is to perform beam search over $b$ hypotheses
at every step on the output of our autoregressive generator (intra-revision level), while performing greedy decoding at the inter-revision level. Although being conceptually straightforward, this method has the limitation of not searching over the inter-revision space (despite revisions being a key component of our approach).

#### 2D Beam Search

We propose 2D beam search, in which we extend beam search as it is applied to token-level autoregressive generative models, and perform beam search using both an intra-revision width of $b$ and an inter-revision beam width of $r$. This allows us to perform search on the inter-revision level, which we find results in better downstream performance, but increases the beam count to $r\times b$ beams. Assuming a fixed sequence length and maximum number of diffusion steps, we would decode as follows: We first use beam search with width $b$ at the token level and take the $r$ most likely candidates (measured with log-likelihood). These $r$ candidates are then fed to the next step of the diffusion model, wherein for each of $r$ hypotheses the next diffusion step is performed with the token-level generator decoding with beam width of $b$. This leads us to have $r\times b$ candidate hypotheses, of which we take the top $r$. This process repeats for each diffusion step thereafter.

#### Nucleus Sampling

To improve the diversity of generations, we also consider a nucleus sampling based approach, where at every timestep ${\mathbf{x}}_{t}$, we use nucleus sampling with $p=0.6$ to sample each token autoregressively at the intra-revision level, and greedily decode at the inter-revision level (i.e. no search or sampling is performed over multiple diffusion steps).

### 3.6 Decoder Initialization Techniques

Figure: Figure 2: Figure illustrating bootstrapping methods for decoding.
Refer to caption: /html/2210.16886/assets/x2.png

Since our model is based on edit processes, it offers flexibility in terms of the discrete sequence from which to initialize the text generation. Previous work on non-autoregressive translation often starts with [MASK] tokens , a null string or random tokens . We include the latter two methods in our experiments, in addition to (1) experimenting with an AR Bootstrap, in which we learn to bootstrap from text generated by a purely autoregressive model, and (2) proposing to use the source-side text as an initial state for the \methoddecoder (as shown in Figure [2](#S3.F2)).
Null Sequence In this setting, we simply initialize \methodwith a null string, in which the first edit is constrained to be insertion.
Random Tokens In this setting, we initialize \methodwith a series of random tokens, following . The model then learns to edit this random sequence.
AR Bootstrap We bootstrap the reverse diffusion process by taking the output of \methodconstrained to generate autoregressively (essentially mimicking a standard autoregressive generator). We then use \methodto further edit the output of this operation.
Source Bootstrap In a sequence-to-sequence setting, we can also generate by bootstrapping using the source text, by setting ${\mathbf{x}}_{T}$ to be equivalent to ${\mathbf{s}}$. As we show in later sections, this is particularly useful in tasks such as summarization in which the output can be easily formulated as an edited version of the input.

## 4 Experiments

### 4.1 Models

#### \method

We instantiate \methodwith two separate Transformer models for the tagger and generator. We use the Transformer-base architecture, with 6 layers, for the a hidden dimension of 512, feedforward dimension of 2048, 8 attention heads, and dropout $p=0.3$.

#### Baselines (MT & Summ)

We use several Transformer baselines from previous literature for our various tasks. We include a conventional 6-layer encoder-decoder Transformer model from , as well as models proposed in related work from the non-autoregressive generation literature: Levensthein Transformer , CMLM , DisCo , Imputer , and SUNDAE .

### 4.2 Tasks

#### Machine Translation

We use the WMT’14 English-German dataset for our machine translation experiments. We use the same preprocessing and post-processing steps as . Unlike the standard in non-autoregressive translation work , we focus on using the gold machine translation data instead of distilled data. We use a Poisson distribution ${\mathcal{E}}_{l}(\lambda=3)$ over edit operation lengths in our corruption process. Note that we compute the edit operations over words rather than tokens. For this task, as well as the following ones, we use 12 diffusion steps, $b=5$, and $r=3$ for beam search, and ${\mathcal{E}}_{t}(60\%\ \texttt{{Keep}},20\%\ \texttt{{Replace}},10\%\ \texttt{{Insert}},10\%\ \texttt{{Delete}})$ based on numbers from preliminary experiments.

#### Summarization

We also benchmark on the CNN/DailyMail dataset for summarization . Summarization is different in nature from machine translation in that it can be described as more conducive to edits as a good summary tends to preserve many parts of the input. We use the same post-processing steps as . We use a Poisson distribution ${\mathcal{E}}_{l}(\lambda=8)$ over edit operation lengths in our corruption process (to roughly model sentence boundaries).

#### Text Style Transfer

We perform experiments using the Yelp dataset for the unsupervised text-style transfer task. We compare against methods such as Tag-and-Generate , Masker , and LEWIS . In contrast with machine translation and summarization, text style transfer datasets are often unaligned (i.e. without source-target pairs) leading to the prominence of unsupervised text style transfer methods. We propose a method of performing unsupervised text style transfer using \method, following the synthetic generation method in . We train two separate, style-specific (e.g. positive and negative) \methodmodels on the style-specific data. We then perform transfer at test time, feeding text from each style into the model trained to edit in the opposite style (e.g. positive text $\rightarrow$ negative \methodmodel; negative text $\rightarrow$ positive \methodmodel).

### 4.3 Results

**Table 2: Machine Translation (MT) and Summarization (Summ) results on WMT’14 En-De (gold) and CNN-DailyMail. Experiments on MT use BLEU while summarization uses ROUGE. \methodis compatible with a standard autoregressive model, while outperforming previous methods.**
| Model | En-De (MT) | CNN-DM (Summ) |
| --- | --- | --- |
| AR Transformer | 27.3 | 36.8 |
| SUNDAE | 26.3 | 37.0 |
| CMLM | 24.6 | — |
| Levenshtein Transformer^2 | 23.7 | — |
| DisCo | 24.7 | — |
| Imputer | 25.2 | — |
| \method | 27.2 | 37.8 |
| \method+ AR bootstrap | 28.8 | 38.4 |
| \method+ source bootstrap | 24.5 | 38.9 |

**Table 3: Results on Yelp dataset for text style transfer. Without task-specific training techniques, \methodperforms comparably to previous task-specific methods.**
| Model | Accuracy | BLEU |  |
| --- | --- | --- | --- |
| Masker | 40.9 | 14.5 |  |
| Tag and Generate | 86.2 | 19.8 |  |
| LEWIS | 93.1 | 24.0 |  |
| \method | 87.6 | 25.2 |  |

#### Main Results

We summarize our main results on both machine translation and summarization in Table [2](#S4.T2). As can be seen, for both machine translation and summarization tasks, \method, using 12 diffusion steps, outperforms all non-autoregressive baselines(^2^22We were not able to reproduce the published results of the Levenshtein Transformer using [their code](https://github.com/facebookresearch/fairseq/blob/main/examples/nonautoregressive_translation/), hence our reported BLEU score of 23.7 is slightly lower than that of 25.2 reported in) and rivals or outperforms the fully autoregressive model. Particularly interesting is how the various methods of initializing our model (i.e. AR Bootstrap and Source Bootstrap) can further improve performance well beyond the autoregressive baseline, depending on the task. We can enforce strong priors on generation depending on the task, while also remaining symbiotic with purely autoregressive transformers. We can see that for summarization, bootstrapping from the source input is more effective than bootstrapping from an abstractive autoregressive model. On both tasks, unlike many non-autoregressive methods, we show that \methodis complementary with token-level autoregressive methods and can be used naturally in conjunction with them.

#### Style Transfer Results

We also perform unsupervised text style transfer using our \methodmodels using the Yelp dataset. The results can be seen in Table [3](#S4.T3). We show that even without task-specific techniques (such as synthetic data generation and classifier based style-specific token identification), we still have competitive performance with state of the art methods.

### 4.4 Analysis

We perform additional analyses on \method, specifically focusing on the decoding method, the number of iterations versus the final BLEU score, and also a qualitative analysis of how text changes at every step.

#### Decoding Method Ablation

We perform an ablation of the decoding method, using \methodfor 12 steps (as used in our main results) and showing results when comparing greedy decoding, (1D) beam search, nucleus decoding, and 2D beam search.We show that 2D-beam search tends to perform the best, likely because it searches over multiple diffusion steps, while other methods (greedy, beam, nucleus) are still competitive.

**Table 4: Decoding method ablation on the MT test set.**
| Initialization | Decoding Method | BLEU |
| --- | --- | --- |
| Random Tokens | Greedy | 26.3 |
| Random Tokens | Beam $b=5$ | 26.7 |
| Random Tokens | Beam $b=15$ | 26.9 |
| Random Tokens | Nucleus | 26.8 |
| Random Tokens | 2D-Beam | 27.2 |

#### Number of Edit Steps versus Performance

We perform an analysis where we compare the number of timesteps in our denoising diffusion process and the final BLEU score on WMT’14 En-De when using 2D-Beam Search and random token initialization in Figure [4](#S4.F4). Here it can be seen that most performance gains are in the initial diffusion timesteps (0-10), with diminishing gains (for machine translation) or gradual losses (for summarization) between 10 and 30, after which performance marginally decreases towards 60 steps. Our model also continues improving for longer that SUNDAE, a possible benefit of using flexible edit operations.

#### How does text change every step?

We include a qualitative sample from our \methodsummarization model (Table [5](#S4.T5)). We find that \methodlearns edit processes intuitive to the task at hand: namely largely deleting portions and making minor edits to the remaining text (similar to how a human may perform summarization given a news article).

**Table 5: Example of our summarization \methodprocess on a test set example. Here we show that the majority of the summarization process is deletion coupled with minor edits. Despite this simplicity, we are able to improve over existing purely abstractive models.**
| Source Document | (CNN)They’re not gonna take it anymore. Really. Twisted Sister says that its 2016 tour will be its last, according to a press release. Next year marks the band’s 40th anniversary, and to celebrate, the tour is being titled ”Forty and F*ck It.” ”It’s official: Farewell,” Twisted Sister singer Dee Snider posted on Facebook. Snider also noted that the band will play with a new drummer, Mike Portnoy of Adrenaline Mob. Portnoy replaces A.J. Pero, who died March 20. The band will also perform two shows in Pero’s honor: one at Las Vegas’ Hard Rock Hotel and Casino, the other at the Starland Ballroom in Sayreville, New Jersey. The latter is in support of Pero’s family. Twisted Sister’s biggest hit, ”We’re Not Gonna Take It,” hit the Top Forty in 1984 and was featured in a popular video. |
| --- | --- |
| Step 1 | .(CNN)They’re not gonna take it anymore. Really. Twisted Sister says that its 2016 tour will be its last, according to a press release. Next year marks the band’s 40th anniversary, and to celebrate, the tour is being titled ”Forty and F*ck It.” ”It’s official: Farewell,” Twisted Sister singer Dee Snider posted on Facebook. Snider also noted that the band will play with a new drummer, Mike Portnoy of Adrenaline Mob. Portnoy replaces A.J. Pero, who died March 20. The band will also perform two shows in Pero’s honor: one at Las Vegas’ Hard Rock Hotel and Casino, the other at the Starland Ballroom in Sayreville, New Jersey. The latter is in support of Pero’s family. Twisted Sister’s biggest hit, ”We’re Not Gonna Take It,” hit the Top Forty in 1984 and was featured in a popular video. |
| Step 2 | Twisted Sister says that its 2016 tour will be its last, according to a press release. Next year marks the band’s 40th anniversary, and to celebrate, the tour is being titled ”Forty and F*ck It.” ”It’s official: Farewell,” Twisted Sister singer Dee Snider posted on Facebook. Snider also noted that the band will play with a new drummer, Mike Portnoy of Adrenaline Mob. Portnoy replaces A.J. Pero, who died March 20. The band will also perform two shows in Pero’s honor: one at Las Vegas’ Hard Rock Hotel and Casino, the other at the Starland Ballroom in Sayreville, New Jersey. The latter is in support of Pero’s family. Twisted Sister’s biggest hit, ”We’re Not Gonna Take It,” hit the Top Forty in 1984 and was featured in a popular video. |
| Step 3 | Twisted Sister says that its 2016 tour will be its last, according to a press release. Next year marks the band’s 40th anniversary, and to celebrate, the tour is being titled ”Forty and F*ck It.” Portnoy replaces A.J. Pero, who died March 20. The band will also perform two shows in Pero’s honor : one at Las Vegas ’ Hard Rock Hotel and Casino, the other at the Starland Ballroom in Sayreville, New Jersey. The latter is in support of Pero’s family. Twisted Sister’s biggest hit, ”We’re Not Gonna Take It,” hit the Top Forty in 1984 and was featured in a popular videoin Las Vegas and New Jersey. |
| Step 4 | Twisted Sister says that its 2016 tour will be its last, according to a press release. Next year marks the band’s 40th anniversary, and to celebrate, the tour is being titled ”Forty and F*ck It.” Portnoy replaces A.J. Pero, who died March 20. The band will perform two shows in Pero’s honor in Las Vegas and New Jersey. |
| Generated Summary | Twisted Sister says that its 2016 tour will be its last. Next year marks the band’s 40th anniversary, and to celebrate, the tour is being titled ”Forty and F*ck It.” A.J. Pero, died March 20. The band will perform two shows in Pero’s honor in Las Vegas and New Jersey. |

#### Time comparsion between decoding methods

We also measure the impact of the various decoding algorithms we used with results shown in Figure [4](#S4.F4). Beam search and 2D-Beam Search performs significantly slower than greedy and nucleus sampling, demonstrating the potential for improved decoding algorithms tailored for improving the trade-off between efficiency and accuracy in diffusion models. Despite this, diffusion models are still promising for constrained/editing-based generation.

Figure: Figure 3: Relative time (seconds) comparison between decoding methods, measured on a single V100 GPU. There is a trade-off between inference cost and performance. Faster well-performing decoding algorithms for diffusion models are an area for further work.
Refer to caption: /html/2210.16886/assets/speed.png

## 5 Related Work

#### Non-Autoregressive Generation

Work in machine translation has explored non/semi-autoregressive generation , which often includes an iterative refinement step . Previous methods in this space are often highly specialized underperform non-autoregressive methods due to the constraints imposed on generation for efficiency. This being said, demonstrated that non-autoregressive models are actually comparable in speed when using a larger batch size instead of $1$. Our method allows us to hone in on the notion of iterative refinement by way of editing processes, and is also relatively general, allowing us to combine \methodwith standard autoregressive models.

#### Learning Properties of Edits

Previous work has also looked at studying or exploiting the properties of edits. This was initially worked on in the context of vector representation learning of edits . Concurrently, a line of work has used edits for specific tasks such as sentence fusion, style transfer and grammatical error correction . Recent work has proposed *editing processes* , in which document generation is looked at through the lens of its revision history, rather than just at a token level. We take inspiration from this work and devise a process by which arbitrary text generation tasks can be fitted into this framework.

## 6 Conclusions

We proposed \method, an diffusion-based generative model for text using edits. \methodshows improvements across the tasks considered (machine translation, summarization, style transfer), with improved generative flexibility via incremental text improvement, and compatibility with standard autoregressive models. We hope that
\methodwith spur research on edit-based generative models, with further potentials including how we can leverage edits to ensemble models (regardless of parameter count) in the discrete space.

## Acknowledgements

We thank Armen Aghajanyan, Daniel Fried, Edison Marrese-Taylor, Eric Wallace, and Luke Zettlemoyer for their helpful comments in early discussions. We thank Ari Holtzman, Jungo Kasai, Aman Madaan, and Eric Wallace for feedback and proofreading the draft of this paper.