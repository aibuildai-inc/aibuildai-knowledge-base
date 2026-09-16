---
arxiv_id: "2412.17780"
title: "PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion"
year: 2024
source: arxiv2md
---

## Abstract

Abstract Peptide therapeutics, a major class of medicines, have achieved remarkable success across diseases such as diabetes and cancer, with landmark examples such as GLP-1 receptor agonists revolutionizing the treatment of type-2 diabetes and obesity. Despite their success, designing peptides that satisfy multiple conflicting objectives, such as target binding affinity, solubility, and membrane permeability, remains a major challenge. Classical drug development and target structure-based design are ineffective for such tasks, as they fail to optimize global functional properties critical for therapeutic efficacy. Existing generative frameworks are largely limited to continuous spaces, unconditioned outputs, or single-objective guidance, making them unsuitable for discrete sequence optimization across multiple properties. To address this, we present PepTune , a multi-objective discrete diffusion model for the simultaneous generation and optimization of therapeutic peptide SMILES. Built on the Masked Discrete Language Model (MDLM) framework, PepTune ensures valid peptide structures with bond-dependent masking schedules and penalty-based objectives. To guide the diffusion process, we propose a Monte Carlo Tree Search (MCTS)-based strategy that balances exploration and exploitation to iteratively refine Pareto-optimal sequences. MCTS integrates classifier-based rewards with search-tree expansion, overcoming gradient estimation challenges and data sparsity. Using PepTune, we generate diverse, chemically modified peptides optimized for multiple therapeutic properties, including target binding affinity, membrane permeability, solubility, hemolysis, and non-fouling for various disease-relevant targets. In total, our results demonstrate that MCTS-guided masked discrete diffusion is a powerful and modular approach for multi-objective sequence design in discrete state spaces.

## 1 Introduction

Peptides possess unique advantages as a therapeutic modality, including their ability to bind to a diverse set of binding motifs without requiring stable binding pockets, making them ideal for targeting structurally diverse protein surfaces . Their relatively large size and flexible backbone enable them to disrupt protein-protein interactions (PPIs) that are central to disease processes, particularly those requiring interactions with larger surface areas . These attributes have driven a surge in interest, with over 33 FDA-approved therapeutic peptides introduced since 2000 and more than 170 in clinical development for diseases ranging from diabetes to cancer . Specifically, peptides are behind one of the most landmark breakthroughs in recent clinical history: GLP-1 receptor agonists like semaglutide and liraglutide, which have transformed the treatment landscape for type-2 diabetes and weight loss . These engineered peptides have achieved remarkable efficacy, following years of meticulous structural and functional optimization . Their success highlights the potential of therapeutic peptides to address complex diseases where more traditional approaches, such as small molecules, often fall short .

As evidenced by the journey to generate semaglutide and liraglutide , peptides containing only the 20 wild-type amino acids have limitations including susceptibility to enzymatic degradation and low membrane permeability . To overcome these limitations, non-natural amino acids (nAAs) containing diverse chemical modifications to the peptide backbone and side chains have been integrated into peptides to enhance their therapeutic properties. For example, selespressin, which contains nAAs at its proteolytic site, has been shown to have a longer plasma half-life than its natural analog and GLP-1 analogs containing the nAA e-Nheptanoyl-l-lysine has demonstrated stronger binding affinity to their target . Furthermore, chemical modifications are commonly used to generate cyclic peptides, with over 99.6% of cyclic peptides containing nAAs . Due to their stable conformation, cyclic peptides have stronger binding affinity and specificity, enhanced membrane permeability, and low degradation . Despite this progress in peptide drug development, designing peptides that effectively engage new therapeutic targets remains a major limitation, with traditional methods involving screening large phage libraries of up to trillions of random amino acid permutations . This motivates the development of generative deep learning models that can effectively learn the space of clinically relevant peptides and sample de novo peptides conditioned with various therapeutic properties, including binding affinity, solubility, and membrane permeability.

Generative diffusion models are considered state-of-the-art for de novo binder design, with new models such as RFpeptides even enabling the design of high-resolution macrocyclic peptides given a target structure . However, structure-based models , or those that require an intermediate structure prediction step , rely on stable tertiary conformations of target proteins, precluding the design of peptide binders to disordered and dynamic targets, which drive a sizable portion of diseases . Generative peptide design language models relying only on the target sequence, such as PepPrCLIP and PepMLM , have demonstrated robust success on disordered and structurally diverse targets, but their utilization of only the 20 wild-type amino acids limits these models from sampling from the space of chemically-modified or cyclic peptides, precluding exploitation of the full therapeutic potential of synthetic peptides.

Recently, discrete generative models have shown incredible promise in areas such as text generation , image synthesis , executable code generation , DNA sequence optimization , and even protein design , but they still face significant limitations in multi-objective-guided generation and optimization. In our case, the challenge lies in simultaneously optimizing for conflicting therapeutic properties, a critical requirement for generating clinically viable peptides . Classifier-based and classifier-free guidance strategies have been explored to steer discrete diffusion objectives toward specific properties , yet these approaches often struggle with conflicting objectives, gradient estimation, and the sparsity of quality data.

In this work, we introduce PepTune, a multi-objective-guided discrete diffusion model for de novo peptide design. PepTune incorporates a novel Monte Carlo Tree Search (MCTS)-based framework for multi-objective guidance of a generative masked discrete diffusion model, pre-trained on a large dataset of chemically modified and cyclic peptides represented as Simplified Molecular Input Line Entry System (SMILES) strings . However, due to the granularity of SMILES representations, the vast majority of SMILES strings are neither chemically valid nor represent synthesizable peptides. By leveraging a bond-dependent masking schedule and invalid loss function, PepTune is capable of selectively sampling from the sub-space of valid peptide SMILES containing both non-natural amino acids (nAAs) and cyclic structures while maintaining a set of Pareto-optimal solutions across multiple therapeutic properties. Our results highlight PepTune’s unique capability to balance diverse objectives, setting a new standard for property-conditioned sequence generation.

Our contributions are as follows:

- 1.
Masked Diffusion Language Model for Discrete Sampling of Peptide SMILES. We introduce the first Masked Diffusion Language Model (MDLM) with a RoFormer backbone for de novo generation of peptide SMILES sequences that is capable of generating valid and diverse chemically modified and cyclic peptides that cannot be encoded with canonical amino acid sequences.
- 2.
Bond-Dependent Masking Schedule. Extending previous work in state-dependent masking , we devised a masking schedule where the probability of masking a token within a peptide bond increases at a slower rate in earlier times $t$ in the masking process compared to non-peptide bond tokens to enforce valid peptide structure. From our derivation of the reverse posterior and NELBO loss, we prove that this strategy increases the weight of the loss associated with peptide bond tokens in a time-dependent manner, resulting in earlier unmasking of peptide bond tokens in the reverse diffusion process.
- 3.
Global Sequence Invalid Loss. Since the vast majority of SMILES sequences do not translate into valid peptides, we introduce an invalid loss that penalizes predicted token probabilities where the argmax token sequence is not a valid peptide SMILES. We scale the penalty score by the softmax probability of the sampled token. This objective propagates the penalty from a single discrete sequence to the smooth token probability distributions, shifting the model parameters away from invalid predictions.
- 4.
MCTS for Robust Classifier-Based Multi-Objective Guidance. We propose a classifier-based multi-objective guidance strategy for discrete diffusion based on the Monte-Carlo Tree Search (MCTS) algorithm that selects a sequence of optimal unmasking steps based on preceding iterations, samples a new set of unmasked sequences to compute their objective scores, and finally back-propagates the scores in the path and updates a set of global Pareto-optimal sequences. This strategy balances exploration of the peptide space through the pre-trained MDLM and multi-objective guidance through the MCTS guidance to enable the discovery of a set of valid Pareto-optimal peptide sequences. We also provide a case study for a time-dependent multi-objective guidance strategy to enable prioritization of properties.
- 5.
Peptide SMILES Property Prediction Toolkit. To supplement our multi-objective guidance strategy, we train a suite of classification and regression models for property-prediction given peptide SMILES input. We train regression models to predict binding affinity to a protein target using cross-multi-head attention and cell membrane permeability using boosted trees. In addition, we train boosted trees for logistic regression of solubility, hemolysis, and non-fouling.

## 2 Results

### 2.1 Unconditional Generator of Peptide SMILES with Masked Discrete Diffusion

Given the immense success of masked language models (MLMs) in learning bidirectional relationships of sequential data , including peptide-protein interaction , we based our unconditional generator on the Masked Diffusion Language Model (MDLM) framework (Figure [1](#S2.F1)) . MDLM is a discrete diffusion architecture that leverages the MLM objective to effectively learn the clean distribution of sequences $p(\mathbf{x})$ by reconstructing clean sequences from sequences corrupted with [MASK] tokens.

Figure: Figure 1: PepMDLM. PepMDLM is a discrete masked diffusion model for unconditional de novo generation of peptide SMILES representations.
Refer to caption: /html/2412.17780/assets/Figures/mdlm.png

To train the MDLM, we randomly sample values of $t\in\text{Uniform}(0,1)$ such that each batch is masked for times $t$ ranging evenly between 0 (fully unmasked) and 1 (fully masked). For the forward masking at time $t$, the token at each position has a $\alpha_{t}$ probability remaining unchanged and a $(1-\alpha_{t})$ probability of transitioning to a [MASK] token. Therefore, the probability distribution of a single sample assigned time $t$ in the forward diffusion process is given by

$$ $\displaystyle q(\mathbf{z}_{t}|\mathbf{x}_{0})=\text{Cat}\left(\mathbf{z}_{t};\alpha_{t}\mathbf{x}_{0}+(1-\alpha_{t})\mathbf{m}\right)$ (1) $$

where $\mathbf{m}$ is a one-hot encoding vector with 1 at the index of the [MASK] token.

We enforce SUBS parametrization such that once a token is unmasked at time $t$, it remains unmasked at the same state for the remainder of the unmasking process for $t\to 0$. Since slight modifications to the peptide sequence can result in significant alterations to its properties , SUBS parametrization allows us to backpropagate rewards from unmasked sequences to earlier unmasking steps that accurately reflect the favorability of the particular unmasking step in generating optimal peptide SMILES sequences.

The backbone model used to generate the predicted probabilities $\mathbf{x}_{\theta}(\mathbf{z}_{t},t)$ of transitioning from a masked state to any token is predicted by a backbone RoFormer architecture (See Appendix [C.1](#A3.SS1)). RoFormer leverages rotary positional embeddings (RoPE) to capture long-range dependencies between tokens which effectively captures the relative inter-token interactions in peptide SMILES, especially for cyclic peptides.

After empirical training, we find that even after convergence to a training loss of 0.59 and validation loss of 1.41, the majority of generated SMILES were invalid peptides due to slight inaccuracies when generating the fundamental peptide backbone formed by peptide bonds between amino acid side chains. Motivated by these findings, we hypothesized that decreasing the masking rate for tokens forming peptide bonds at earlier time steps and masking them at later time steps in the training process would encourage the model to accurately unmask the peptide bonds first during generation before proceeding to fill in side-chain tokens. To this end, we conceptualized a bond-dependent masking schedule $\alpha_{t}(\mathbf{x}_{0})$ such that peptide bond tokens follow a polynomial masking schedule that increases slower for small $t$ and rapidly approaches $\infty$ for $t$ close to 1.

$$ $\displaystyle\alpha_{t}(\mathbf{x}_{0})=\begin{cases}1-t^{w}&\mathbf{x}_{0}=\mathbf{b}\\ 1-t&\mathbf{x}_{0}\neq\mathbf{b}\end{cases}$ (2) $$

With our bond-dependent masking scheme, we derive the parameterized reverse diffusion distribution $p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})$ that defines the transition from the partially masked sequence $\mathbf{z}_{t}$ at time $t$ to a slightly unmasked sequence $\mathbf{z}_{s}$ at time $s=t-\frac{1}{T}$ (Appendix [B.2](#A2.SS2)).

$$ $\displaystyle p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})=\begin{cases}\left(\left(\frac{s}{t}-\frac{s^{w}}{t^{w}}\right)\mathbf{b}+\frac{t-s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\mathbf{z}_{s}+\left(\left(\frac{s^{w}}{t^{w}}-\frac{s}{t}\right)\mathbf{b}+\frac{s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\mathbf{m}&\mathbf{z}_{t}=\mathbf{m}\\ \mathbf{z}_{t}&\mathbf{z}_{t}\neq\mathbf{m}\end{cases}$ (3) $$

Furthermore, we derive a continuous-time bond-dependent NELBO loss function $\mathcal{L}^{\infty}_{\text{NELBO}}$ (Methods [4.3](#S4.SS3), Appendix [B.3](#A2.SS3)) that scales the loss of incorrect predictions for peptide-bond tokens by the exponent $w$, encouraging the model to predict peptide bond tokens at high confidence.

In addition to enforcing peptide bond structure, we find that small syntactical errors during the sampling can result in completely invalid SMILES. Therefore, we introduce an additional invalid loss term $\mathcal{L}_{\text{invalid}}$ for when the sequence sampled from taking the argmax of the predicted probabilities for each training step corresponds to an invalid peptide SMILES based on our peptide SMILES validity filter (Methods [4.5](#S4.SS5)). We scale the loss with the softmax of each of the sampled token probabilities to bypass the vanishing gradient when taking the argmax.

With these enhancements, the loss converged after only two epochs of training on the dynamically-batched set of 11 million peptide SMILES (Methods [4.1](#S4.SS1)), with a per-token training loss of 0.832 and a validation loss of 0.880, but we trained for a total of 8 epochs (Table [1](#S2.T1)). Our optimized unconditional MDLM (PepMDLM) trained on 11 million peptide SMILES achieved a high valid peptide generation rate of 45% with a token length of 100 ($\sim$15 amino acids) and 36% with a token length of 200 ($\sim$30 amino acids) when passed through our peptide SMILES validity filter that evaluates the presence of peptide bonds and canonical and non-canonical side-chains (Methods
[4.5](#S4.SS5)).

Table: Table 1: Training and validation loss after convergence on 11 million peptide SMILES with bond-dependent masking and invalid loss.

### 2.2 PepMDLM Generates Diverse Chemically-Modified and Cyclic Peptides

To evaluate the unconditional generation quality as a benchmark for our guided generation model, we leverage the Moses metrics , including validity, uniqueness, diversity, and similarity to nearest neighbor (SNN) (Methods [4.5](#S4.SS5)), to compare our model with an autoregressive generator of macrocyclic peptides, HELM-GPT . In contrast to HELM-GPT, we use our peptide SMILES validity filter that checks that the SMILES string is a valid molecule and contains peptide bonds and valid natural or non-natural side chains. Overall, PepMDLM shows increased uniqueness and diversity with lower SNN, demonstrating our capability to comprehensively search the sub-space of valid peptide SMILES. Despite our lower validity, our model is unique to peptide SMILES, which has much higher granularity than HELM notation (Table [2](#S2.T2)).

**Table 2: Benchmark of PepMDLM unconditional model against HELM-GPT.**
| Model | Validity ($\uparrow$) | Uniqueness ($\uparrow$) | Diversity ($\uparrow$) | SNN ($\downarrow$) |
| --- | --- | --- | --- | --- |
| HELM-GPT | 0.839 | 0.913 | 0.595 | 0.975 |
| PepMDLM | 0.450 | 1.000 | 0.705 | 0.513 |

Since the advantage of SMILES-based representation of peptides lies in its ability to represent chemically modified and non-natural amino acids (nAAs), we compared the frequency of nAAs in the peptides generated by PepMDLM with the top 100 peptides with the highest $\log(P_{\text{exp}})$ scores measuring lipophilicity in the CycPeptMPDB database that contains a total of 7334 labeled peptide SMILES with permeability scores between -8.0 ($1.0\times 10^{-8}\text{cm/s}$) and -4.0 ($1.0\times 10^{-4}\text{cm/s}$) generated from the parallel artificial membrane permeability (PAMPA) assay. We also compared against the dataset of experimentally tested protein-binding peptides from the PepLand dataset . By feeding peptides from each dataset into our peptide sequence identification function, SMILES2PEPTIDE (Algorithm [8](#alg8)), we found that both datasets had an average nAA frequency greater than two per peptide which could be identified from a collection of over 200 nAAs from SwissSidechain , indicating the significance of nAAs in defining various peptide properties (Figure [2](#S2.F2)).

Similarly, PepMDLM generates valid peptides with an average slightly larger than both datasets, demonstrating our unique ability to design de novo peptides with cyclic and nAA modifications and expanding the search space of therapeutic peptides well beyond any generative model trained on canonical amino acid representations.

Figure: Figure 2: PepMDLM generates cyclic and modified peptides. (Above) Distribution comparison of non-natural amino acid frequency for 100 unconditionally-generated peptide SMILES with the peptide SMILES dataset of experimentally-validated peptides for membrane permeability (PAMPA) and binding affinity (Methods [4.1](#S4.SS1)). (Bottom) Per peptide frequency of non-natural amino acids (nAAs) and percentage of cyclic peptides in PepMDLM-generated sequences and experimentally-validated membrane-permeable peptides.
Refer to caption: /html/2412.17780/assets/Figures/nAA.png

### 2.3 Multi-Objective Guidance on Therapeutic Properties for Discrete Diffusion

To generate peptides with high clinical potential, they must achieve high binding affinity with a protein target while optimizing for an array of therapeutic properties like membrane permeability to reach intracellular targets, solubility to improve drug-loading, and non-fouling and non-hemolysis to mitigate negative side-effects. Guiding the discrete diffusion objective is challenging due to the lack of data on property-specific peptides to train large generative models and the lack of gradients of the discrete sequence space. Although existing strategies have been explored such as gradient estimation and converting to a continuous latent space , there has yet to be a robust strategy that operates directly in the discrete space and can scale for various distinct properties without sacrificing performance in any one property.

To address this gap, we introduce a classifier-based multi-objective guidance strategy to generate a set of non-dominated peptide SMILES $\mathcal{P}^{*}$ using Monte Carlo Tree Search (MCTS) (Figure [3](#S2.F3)). A sequence is non-dominated when no other sequence has a strictly larger score in one or more objectives while maintaining equal scores for the remaining objectives. We leverage the unique capability of our unconditional MDLM model to sample from the diverse space of peptide SMILES and our trained property classifiers to search the constrained space of peptides with therapeutically optimal properties through an iterative selection, expansion, rollout, and backpropagation loop (Methods [4.4](#S4.SS4)).

Figure: Figure 3: PepTune. PepTune is a multi-objective discrete diffusion model guided by Monte-Carlo Tree Search (MCTS). The full algorithm is detailed in Algorithm [3](#alg3).
Refer to caption: /html/2412.17780/assets/Figures/mcts-logo.png

The PepTune multi-objective guidance framework is defined as follows. We start each iteration from a fully masked sequence $\mathbf{z}_{t(T)}$, defining the root node of the MCTS tree. During the selection step, we traverse an optimal path through the MCTS tree defined as a series of partial unmasking steps that have been previously traversed based on the selection score vector $\mathbf{U}(\mathbf{z}_{t},\mathbf{z}_{s,i})$ (defined in Equation [19](#S4.E19)), indicating the likelihood that the given step generates a Pareto non-dominated sequence.

Upon reaching a terminal leaf node $\mathbf{z}_{t}$ defined as a partially masked sequence at time $t>0$ that has yet to be further unmasked, we expand the leaf node to explore $M$ different possible unmasking steps by batched Gumbel unmasking (Equation [21](#S4.E21)), which applies independently sampled Gumbel noise vectors to the token probabilities predicted by the MDLM backbone for sampling each child node $\mathbf{z}_{s,i}$ for $i=1\dots M$, enforcing diverse unmasking schemes while remaining consistent with the predicted token distribution.

Each expanded node is unmasked with greedy Gumbel-max sampling to obtain a fully unmasked sequence $\mathbf{x}_{s,i}$. For all $i$, we compute a $K$-dimensional score vector for $K$ objectives and compare it with the scores of the current pool of Pareto non-dominated sequences $\mathcal{P}^{*}$ to generate a reward vector $\mathbf{r}(\mathbf{x}_{s,i})\in\mathbb{R}^{K}$, where the entry $r_{k}(\mathbf{x}_{s,i})$ at index $k$ is a value in $[0,1]$ indicating the Pareto-optimality of the sequence for the $k$th objective compared to the current pool of Pareto non-dominated $\mathcal{P}^{*}$.

$$ $\displaystyle r_{k}(\mathbf{x}_{s,i})=\frac{1}{|\mathcal{P}^{*}|}\sum_{n=1}^{|\mathcal{P}^{*}|}\mathbb{I}\big{[}s_{k}(\mathbf{x}_{s,i})\geq s_{k}(\tilde{\mathbf{x}}_{n})\big{]}$ (4) $$

After computing the reward vector for all valid sequences generated during rollout, we take the sum of the reward vectors subtracted by a penalty score proportional to the fraction of sampled SMILES that are not peptides and add the resulting $K$-dimensional vector to the cumulative reward vectors $\mathbf{W}(\mathbf{z}_{t})$ of all predecessor nodes on the path to the root node $\mathbf{z}_{t(T)}$, which determines the selection steps of proceeding iterations.

Even though the selection process favors high-reward unmasking steps, we show that the resulting pool of generated peptides retains similar uniqueness and diversity scores to the peptides generated by our unconditional MDLM and in the training dataset (Table [3](#S2.T3)). In addition, the fraction of valid peptides consistently reaches 100% after only 20 iterations of the MCTS search algorithm, demonstrating the effectiveness of backpropagating the classifier-based rewards and invalidity penalty.

**Table 3: Evaluation metrics for generative quality of peptide SMILES sequences of max token length set to 200.^1**
| Model | Validity ($\uparrow$) | Uniqueness ($\uparrow$) | Diversity ($\uparrow$) | SNN ($\downarrow$) | Randomness ($\uparrow$) | KL-Divergence ($\uparrow$) |
| --- | --- | --- | --- | --- | --- | --- |
| Data | 1.000 | 1.000 | 0.885 | 1.000 | 4.55 | 0 (Reference) |
| PepMDLM | 0.450 | 1.000 | 0.705 | 0.513 | 4.11 | 0.174 |
| PepTune | 1.000 | 1.000 | 0.677 | 0.486 | 4.12 | 0.173 |

While MCTS has been integrated into autoregressive and inpainting generative models, it has yet to be effectively applied to diffusion-based models. Here, we have provided a robust framework for MCTS-guided discrete diffusion that can extend beyond de novo peptide generation to other multi-objective generative tasks.

### 2.4 Therapeutic Property Prediction for Peptide SMILES

While several classifiers exist for predicting properties of small-molecule SMILES sequences and amino-acid representations of peptides, there exists a gap in high-quality property models trained specifically on peptide SMILES data. To fill this gap, we train regression models for target-binding affinity and cell membrane permeability and binary classification models for solubility, hemolysis, and non-fouling specifically on peptide SMILES data (Table [4](#S2.T4)).

**Table 4: Benchmarks of solubility, hemolysis, and non-fouling prediction for PeptideCLM and PeptideBERT embeddings. Each was trained using XGBoost for classification.**
|  | Solubility | Hemolysis | Non-fouling |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Metric | Ours | PeptideBERT | Ours | PeptideBERT | Ours | PeptideBERT |
| F1 | 0.660 | 0.597 | 0.846 | 0.483 | 0.768 | 0.699 |
| Accuracy | 0.661 | 0.651 | 0.846 | 0.823 | 0.766 | 0.873 |

To guide the generation of peptides with high binding affinity to a given protein target, we trained a Transformer-based model with cross multi-head attention layers that learn the joint latent space of ESM-2-650M embeddings of the protein amino acid sequence and PeptideCLM embeddings of the peptide SMILES sequence (Figure [4](#S2.F4). See full architectural details in Appendix [C.3](#A3.SS3)). Given a peptide SMILES sequence and a protein amino acid sequence, the model was trained on natural and non-natural peptide SMILES $K_{d}/K_{i}/IC50$ binding affinity to predict a score that indicates weak binding (< 6.0), medium binding (6.0-7.5), and high binding ($\geq$ 7.5). Our regression model achieved a strong Spearman correlation coefficient of 0.869 on the training data and 0.633 on the held-out validation data.

Figure: Figure 4: Architecture of binding affinity regression model. Embeddings for the target protein sequence are generated with ESM-2 and embeddings for the peptide SMILES are generated using PeptideCLM. Cross multi-head attention layers combine the embeddings and predict a binding affinity score.
Refer to caption: /html/2412.17780/assets/Figures/binding-regression.png

For cell membrane permeability, we trained an XGBoost boosted tree regression model on PeptideCLM embeddings which returns the predicted PAMPA lipophilicity score ($\log P$) given a peptide SMILES sequence, where values $\geq-6.0$ indicate strong permeability and values $<6.0$ indicate weak permeability. We trained our model on 34,853 experimentally validated peptide SMILES (See Methods [4.1](#S4.SS1)) and achieved a strong Spearman correlation coefficient of 0.998 on the training dataset and 0.943 on the test dataset (Figure [5](#S2.F5), Table [5](#S2.T5)).

Figure: Figure 5: Correlation plots for binding affinity and membrane permeability classifiers. Plot of true permeability (logP) on the $x$-axis and predicted permeability on the $y$-axis for the (A) validation set and (B) training set. Plot of true binding affinity (log-scale) on the $x$-axis and predicted permeability on the $y$-axis for the (C) validation set and (D) training set.
Refer to caption: /html/2412.17780/assets/Figures/classifiers_new.png

**Table 5: Held-out validation performance of binding affinity and membrane permeability regression models trained on peptide SMILES.**
| Metric | Binding Affinity | Membrane Permeability |
| --- | --- | --- |
| Spearman Rank Correlation | 0.633 | 0.943 |
| MSE | 0.566 | 0.088 |

### 2.5 Case Studies for Multi-Objective Generation of Peptide Binders

With our trained property classifiers, we first conduct experiments for five diverse, therapeutically relevant protein targets to evaluate our multi-objective MCTS guidance strategy. To demonstrate generalizability, we include targets with known peptide binders such as GLP-1R and TfR, and proteins with no known binders, including GFAP, NCAM1, and AMHR2. These targets include both receptor proteins involved with active transport pathways as well as intracellular targets where cell membrane permeability is crucial to achieving therapeutic effects. For each target, we condition the generation on the binding affinity score given the target protein sequence along with solubility, hemolysis, non-fouling, and cell membrane permeability for intracellular targets. For external testing and validation, we use Autodock Vina to compute in silico binding affinities of our generated binders (See Methods [4.5](#S4.SS5)).

Targeting Receptors on the Blood-Brain Barrier. The Transferrin receptor (TfR) is a receptor protein abundant on the selectively permeable blood-brain barrier (BBB) that is responsible for transporting iron-binding transferrin (Tf) proteins into the brain parenchyma . Given its selective expression on brain endothelial cells and glioma cells and its ability to recycle back to the luminal surface after facilitating the internalization of cargo through the BBB , TfR has been extensively studied as a target for the intravenous delivery of various therapeutics and therapeutic nanocarriers through the BBB .

To generate relevant binders for TfR, we condition PepTune on binding affinity with the TfR sequence, in addition to solubility, hemolysis, and non-fouling. At each iteration, we measured the mean of the properties scores across all rolled-out sequences from the selected node to evaluate the effectiveness of the optimization strategy. We show that all properties, except solubility, exhibited an upward trend over iterations, with the average score of the binding affinity classifier exhibiting a significant increase in score to over 9.0 (Figure [6](#S2.F6)B). After plotting the distribution of 100 peptides generated from a single run of PepTune with the minimum number of sequences set to 100, we confirm that our multi-objective MCTS algorithm shifted the distribution to a higher predicted binding affinity than the unconditionally generated peptides (PepMDLM) and the data used to train the binding regression model (Figure [6](#S2.F6)A). Despite being conditioned on four distinct properties, PepTune is capable of generating higher affinity binders than the unconditional model, supporting the effectiveness of our multi-objective guidance strategy.

Encouraged by these results, we sampled the Pareto-optimal sequences from the generated peptides and used Vina docking to compute their optimized docking score. Notably, we observed that all of the generated binders that were selected for docking produced affinity scores below -6.0 kcal/mol, with our top-performing binder achieving a -8.4 kcal/mol binding affinity (Figure [6](#S2.F6)C). From the docking scores, we took the two binders with the best docking scores and visualized their binding conformation with TfR, showing that they bind to distinct motifs on the protein surface (Figure [6](#S2.F6)B, F, G).

Figure: Figure 6: PepTune-generated peptide binders to TfR. (A) Density plot depicting the frequency of predicted binding affinity scores from our trained regression model for the sequences in the data used to train the regression model, the generated peptides from our unconditional PepMDLM model, and our PepTune model conditioned on TfR binding affinity, solubility, hemolysis, and non-fouling. (B) Plots depicting the mean scores for each property over the number of iterations or traversals of the MCTS algorithm for 128 iterations and a maximum token length of 200. The shaded region represents the standard deviation. (C) Two-dimensional visualization of generated binders with token length 100, their corresponding docking scores ($\downarrow$) computed using Vina docking, and predicted classifier scores ($\uparrow$) from the trained classifiers. (D) Visualizations of generated binders with token length 200, their docking scores, and predicted classifier scores.
Refer to caption: /html/2412.17780/assets/Figures/tfr.png

To further confirm binding affinity to TfR, we compared our peptides to the well-established 7-amino acid peptide T7 (sequence: HAIYPRH) that selectively binds to an alternative site as compared to endogenous Tf on TfR . T7 has been extensively explored for targeted delivery of nanoparticles to the brain , and has demonstrated 7.89-fold enhanced brain penetration in in vivo mice models . After docking T7 with TfR, we obtained a docking score of -8.4 kcal/mol. Notably, our peptides optimized on all four therapeutic properties including TfR binding affinity show competitive docking scores to T7 (Figure [7](#S2.F7)A, C, E), suggesting that PepTune is capable of generating promising candidates for in vivo targeting and delivery across the BBB. Furthermore, after annotating polar contacts within 3.5 Å  we determine that both of the generated peptides with the best binding affinity scores have shared residue contacts when binding with TfR as T7 (Figure [7](#S2.F7)B, D, F), indicating that our generated peptides have similar binding properties to T7, enabling it to bind strongly to a shared binding site. Furthermore, our generated binders have diverse structural features, such as cycles in binder 1 and side-chain modifications in binder 2. Since T7 is known to bind to an alternative site than endogenous Tf , we show that PepTune can generate viable candidates for non-competitive binding to TfR for BBB-targeting applications.

Figure: Figure 7: Comparison of PepTune-generated peptides and established T7-peptide to TfR. Two-dimensional chemical structure of (A) PepTune-generated binder 1, (C) established T7 peptide, and (E) PepTune-generated TfR binder 2 and their Vina docking scores to TfR ($\downarrow$). Zoomed-in visualization of the docked binding positions of (A) binder 1, (B) T7, and (C) binder 2 with TfR. Polar contacts within 3.5 Å are annotated, and shared contacts between T7 and binder 1 (purple) and between T7 and binder 2 (blue) are highlighted. (C) Overlay of peptide binders on full TfR protein
Refer to caption: /html/2412.17780/assets/Figures/t7_control_new.png

Targeting GLP-1R. Given the significant development of glucagon-like peptide-1 (GLP-1R) peptide agonists for the treatment of type-2 diabetes and obesity , we compared GLP-1R binding affinity-conditioned peptides generated using PepTune with recent blockbuster GLP-1R agonists: semaglutide and liraglutide. Both semaglutide and liraglutide are over 30 amino acids in length and act by mimicking the binding of natural GLP-1 by binding to the activation pocket of GLP-1R with high precision (Figure [8](#S2.F8)) .

Shorter agonists or antagonists to GLP-1R would serve several benefits to the treatment of insulin-related disorders, including reduced cost and complexity of synthesis, lower immunogenicity, and faster tissue penetration. Therefore, we sought to generate shorter-chain peptides that are capable of binding to GLP-1R with comparable affinity to the existing agonists. We first generated a pool of peptide binders conditioned on binding affinity with the GLP-1R sequence, solubility, hemolysis, and non-fouling. After selecting the peptides with the highest predicted binding affinity scores from the Pareto non-dominated set, we performed docking and determined docking scores of -7.4 kcal/mol and -7.0 kcal/mol for the two best candidates. Our peptides show superior docking affinity to GLP-1R while interacting at overlapping binding motifs to semaglutide and liraglutide derived from the natural hormone ligand, GLP-1 (Figure [8](#S2.F8)). These results suggest that our PepTune-derived peptides can serve as potent agonists or antagonists of GLP-1R signaling.

Figure: Figure 8: Comparison of docked PepTune-generated peptides to existing GLP-1R agonists. (A, B) Docking images of semaglutide (score: -5.7 kcal/mol) and liraglutide (score: -5.1 kcal/mol) binding to GLP-1R. (C) Full view of the positive control GLP-1R agonists and the PepTune-generated binders on GLP-1R. (D, E) Docking images of binder 1 (score: -7.4 kcal/mol) and 2 (score: -7.0 kcal/mol) were generated using PepTune conditioned on predicted affinity to GLP-1R, solubility, hemolysis, and non-fouling. Shared polar contacts between binder 1 and either controls are highlighted in pink, shared polar contacts between binder 2 and either controls are highlighted in green, and the shared contacts across both binders are highlighted in purple.
Refer to caption: /html/2412.17780/assets/Figures/glp-control_new.png

Targeting Intracellular Proteins. Glial fibrillary acidic protein (GFAP) is an intracellular protein differentially expressed in astrocytes, a family of glial cells in the brain . Dysregulation of GFAP expression has been found to cause Rosenthal fibers, astrocytic cytoplasmic inclusions that are responsible for Alexander disease, a fatal neurodegenerative disease affecting infants . Discovering potent binders that inhibit or degrade GFAP proteins can have significant therapeutic implications. However, no established peptide binders exist to GFAP, which motivates their de novo design. In addition to achieving high binding affinity with GFAP, we posit that an optimal peptide binder must also cross the astrocyte cell membrane into the cytosol to access GFAP. Therefore, we condition the generation of GFAP binders on five properties: binding affinity to GFAP, solubility, hemolysis, non-fouling, and cell membrane permeability using our permeability regression model, demonstrating optimization across all of these properties (Figure [9](#S2.F9)). To confirm GFAP engagement, our docking peptides demonstrate strong affinities below -7 kcal/mol, motivating downstream experimental validation in astrocyte cultures (Figure [9](#S2.F9)B and D).

Figure: Figure 9: PepTune-generated peptide binders to intracellular protein GFAP. (A, C) Two-dimensional structures of GFAP binder 1 and 2 with predicted property scores, including cell membrane permeability. (B, D) GFAP binders 1 and 2 docked to GFAP with score of -8.5 kcal/mol and -7.1 kcal/mol respectively. (E) Full GFAP protein structure with docked binders 1 and 2. (F) The distribution of PAMPA membrane permeability scores from 34,853 experimentally-validated peptides compared to 100 peptides generated using our unconditional PepMDLM model, and 100 peptides generated with PepTune conditioned on both cell membrane permeability and affinity to GFAP. The permeability curve shifted towards higher permeability with a mean of -6.295. (G) Simultaneously, the distribution of predicted binding affinity scores to GFAP for the PepTune-generated peptides is shifted to higher scores with a mean of 8.053 compared to a set of experimentally-tested peptides and unconditional PepMDLM-generated peptides.
Refer to caption: /html/2412.17780/assets/Figures/gfap.png

Targeting Extracellular Proteins Without Existing Binders. To test the ability of our model to generate binders to challenging extracellular targets without existing binders, we evaluate PepTune-generated peptides for NCAM1 and AMHR2, two therapeutically relevant receptor proteins. Neural cell adhesion molecule 1 (NCAM1), is a transmembrane protein expressed on the surface of neurons and glial cells . Beyond its roles in neuronal migration and synaptogenesis, NCAM1 is also crucial for memory formation, highlighting its significance in brain development . As NCAM1 is an extracellular protein, we generated a library of peptides with PepTune-optimized NCAM1 binding affinity, solubility, hemolysis, and non-fouling (Figure [10](#S2.F10)F, G). All properties exhibited an upward trend across optimization iterations.

We selected two binders with the highest Vina docking scores for visualization (Figure [10](#S2.F10)A-E). Notably, in silico docking analysis revealed that binder 1 exhibits markedly high affinity binding (-8.6 kcal/mol) while binder 2 wraps around the NCAM1 structure via numerous polar contacts, suggesting extensive and specific interactions (Figure [10](#S2.F10)B and D).

Figure: Figure 10: PepTune-generated peptide binders to NCAM1. Two-dimensional structures of (A) binder 1 and (C) binder 2 genered with PepTune. Docking positions of (B) binder 1 and (C) binder 2 on NCAM1 with annotated polar contacts within 3.5 Å(̇G) Full NCAM1 protein structure with docked peptide binders 1 and 2. (H) (Top) Density plot of NCAM1 binding affinity scores for PepTune (mean: 6.708), PepMDLM (mean: 5.298), and peptides from a control set of experimentally-tested peptide SMILES (mean: 5.360). (Bottom) Plots depicting the average predicted score for NCAM1 binding affinity, solubility, hemolysis, and non-fouling over iterations of MCTS.
Refer to caption: /html/2412.17780/assets/Figures/ncam.png

Anti-Müllerian hormone type-2 receptor (AMHR2) is a transmembrane receptor involved in sex differentiation. Mutations in the AMHR2 gene are a leading cause of Persistent Müllerian duct syndrome (PMDS) in males, resulting in the retention of female gonads alongside male reproductive structures . In females, polymorphisms of AMHR2 have been associated with infertility . Most interestingly, antagonism of AMHR2 with therapeutic peptides can potentially serve as a specific therapy for polycystic ovarian syndrome (PCOS), which affects an estimated 4% to 10% of women globally , as AMHR2 signaling has been implicated in follicular arrest and dysregulated ovarian function .

Following similar computational set-ups as described previously, we generated in silico binders with high Vina predicted binding affinities (<-6 kcal/mol), despite observing a decrease in the predicted solubility along iterations (Figure [11](#S2.F11)). However, our observation of reduced solubility upon binder docking can be attributed to the presence of hydrophobic patches within the AMHR2 extracellular domain, particularly near the binding site to its ligand AMH . This phenomenon highlights the importance of balancing solubility and binding affinity in binder development. With further optimization of their therapeutic properties, we hope to demonstrate the potential of these binders for applications in fertility treatment in the future.

The examples above demonstrate the versatility of our method, which can be effectively applied to discover peptide binders for single target proteins lacking known ligands, thereby unlocking their therapeutic potential.

Figure: Figure 11: PepTune-generated peptides to AMHR2. Two-dimensional structures of (A) binder 1 and (B) binder 2 generated with PepTune. Docking positions of (A) binder 1 and (B) binder 2 on NCAM1 with annotated polar contacts. (G) Full AMHR2 protein structure with docked peptide binders 1 and 2. (H) (Top) Density plot of AMHR2 binding affinity scores for PepTune (mean: 8.212), PepMDLM (mean: 6.832), and peptides from a control set of experimentally-tested peptide SMILES (mean: 6.740). (Bottom) Plots depicting the average predicted score for AMHR2 binding affinity, solubility, hemolysis, and non-fouling over iterations of MCTS.
Refer to caption: /html/2412.17780/assets/Figures/amhr_new.png

### 2.6 Case Studies for Dual-Target Binding Peptides

Multi-target drug discovery is of significant interest in various fields including cancer therapeutics and drug delivery for neurological disorders given their ability to perform multiple different functions such as binding to biological barriers like the blood-brain barrier, penetrating target cells, and inhibiting protein-protein interactions , .

The design of dual-target drugs remains challenging across small-molecule, peptide, and protein domains due to the often contradictory structures and properties required for high affinity and specificity to multiple protein targets . Traditional techniques involve performing subsequent rounds of phage display to discover candidates that bind to both targets, which does not explore the full space of potential candidates and often results in peptides that bind to one target but fail to bind to the other. Guided diffusion presents a promising solution to de novo design of multi-target binding peptides; however, multi-target conditioning in the discrete sequence space remains under-explored.

PepTune is uniquely positioned to tackle the multi-target optimization task since it can explore several different unmasking pathways while maintaining a set of Pareto-optimal peptide sequences with non-dominated binding affinity scores with each of the protein targets. Our strategy enables conditioning on multiple target proteins to design binders with high affinity to both targets without sacrificing the discovery of peptides with superior binding affinity to only one of the targets since the model keeps track of all non-dominated peptides.

Figure: Figure 12: Property Scores Over Iteration for Dual-Target Conditioning on TfR and GLAST. (A) Plot of average predicted binding affinity score to GLAST over iterations. (B) Plot of average predicted binding affinity score to TfR over iterations. (C, D, E) Plot of average predicted solubility, hemolysis, and non-fouling scores over iterations.
Refer to caption: /html/2412.17780/assets/Figures/dual-curves.png

Targeting TfR and GLAST for Drug Delivery to Astrocytes. To evaluate PepTune’s capabilities in multi-target guidance, we generate bi-specific peptide binders to TfR and glutamate-aspartate transporter (GLAST) protein abundant on the surface of astrocytes, a type of glial cell in the brain. Successfully generating these peptides can facilitate BBB-crossing via TfR binding and uptake in astrocytes via GLAST binding for intravenous delivery of therapeutics for a multitude of neurological disorders where astrocytes are involved, including Alexander disease , Alzheimer’s disease , Parkinson’s disease , Huntington’s disease , multiple sclerosis , and several psychiatric disorders .

We generated a pool of 100 peptide binders conditioned on five properties: predicted binding affinity to TfR, predicted binding affinity to GLAST, solubility, hemolysis, and non-fouling. Notably, we observed an increase across all properties over iterations, with the final solubility, hemolysis, and non-fouling scores surpassing the binders conditioned only on TfR binding affinity (Figure [12](#S2.F12)). This suggests that our multi-target guidance strategy does not result in significant trade-offs in property scores.

To confirm that our generated binders indeed bind to both TfR and GLAST, we selected seven binders and conducted docking against TfR and GLAST separately for each binder. Incredibly, the docking scores across all seven binders were less than or equal to -7.5 kcal/mol for both targets, with the best-scoring binder simultaneously achieving a score of -10.5 kcal/mol for TfR and a score of -9.2 kcal/mol for GLAST (Table [6](#S2.T6)). In addition, the top-performing binders have diverse secondary structures (Figure [13](#S2.F13)) and have positive solubility, and hemolysis probabilities (Table [6](#S2.T6)). The binding positions and polar interactions vary greatly across the top-performing candidates, enabling the selection of binders that fit a range if binding site constraints. This indicates that PepTune can discover a wide subspace of optimal peptides with strong binding affinity to both TfR and GLAST that is not dependent on a specific binding motif. Our next steps consist of validating the dual-binding affinity of our top candidates in an in vitro BBB-transwell model and observing whether both BBB-crossing and uptake into basolateral astrocytes are enhanced for dual-target compared to single-target conditioned and control peptides.

**Table 6: PepTune-generated dual-target binders to TfR and GLAST.**
| Binder ID | TfR Docking Score (kcal/mol) ($\downarrow$) | GLAST Docking Score (kcal/mol) ($\downarrow$) | Solubility ($\uparrow$) | Hemolysis ($\uparrow$) | Non-fouling ($\uparrow$) |
| --- | --- | --- | --- | --- | --- |
| Binder 1 | -8.8 (8.800) | -8.9 (7.775) | 0.975 | 0.743 | 0.118 |
| Binder 2 | -8.0 (7.599) | -7.9 (6.751) | 0.938 | 0.835 | 0.309 |
| Binder 3 | -8.3 (7.537) | -8.2 (6.662) | 0.972 | 0.914 | 0.214 |
| Binder 4 | -7.6 (7.748) | -7.5 (6.946) | 0.959 | 0.902 | 0.290 |
| Binder 5 | -10.5 (8.714) | -8.5 (7.398) | 0.811 | 0.748 | 0.202 |
| Binder 6 | -8.4 (8.197) | -7.5 (7.076) | 0.971 | 0.855 | 0.165 |
| Binder 7 | -9.3 (8.321) | -9.2 (7.190) | 0.881 | 0.860 | 0.212 |

Figure: Figure 13: PepTune-generated bi-specific peptides to TfR and GLAST. Full protein binding location and close-up binding position for (A) dual binder 1, (B) dual binder 6, and (C) dual binder 8 with TfR (left) and GLAST (right). Polar contacts within 3.5 Å are highlighted.
Refer to caption: /html/2412.17780/assets/Figures/tfR-glast-new-2.png

Dual-Targeting of GFAP and an E3 Ubiquitin Ligase for Target Protein Degradation. As another dual-target case study, we used PepTune to generate peptides with high binding affinity to GFAP protein and an E3 ubiquitin ligase protein RBX1, a protein in the Skp1/Cullin-1/F-box (SCF) E3 ubiquitin ligase complex that recruits E2 to catalyze ubiquitination and subsequent degradation . A peptide generated for this task would be capable of binding to GFAP proteins overexpressed in Alexander disease and mediate their proteasomal degradation, which could alleviate the production of disease-causing Rosenthal fibers in astrocytes .

**Table 7: PepTune-generated dual-target binders to GFAP and RBX1.**
| Binder ID | GFAP Docking Score (kcal/mol) ($\downarrow$) | E3 Docking Score (kcal/mol) ($\downarrow$) | Solubility ($\uparrow$) | Hemolysis ($\uparrow$) | Non-fouling ($\uparrow$) |
| --- | --- | --- | --- | --- | --- |
| Binder 1 | -8.0 (8.384) | -8.4 (7.468) | 0.730 | 0.894 | 0.111 |
| Binder 2 | -8.3 (7.395) | -9.3 (7.089) | 0.972 | 0.869 | 0.134 |
| Binder 3 | -7.3 (7.925) | -8.7 (7.158) | 0.935 | 0.812 | 0.143 |
| Binder 4 | -8.8 (7.144) | -8.7 (7.000) | 0.897 | 0.807 | 0.158 |

After conditioning PepTune generation on binding affinity to GFAP, binding affinity to RBX1, solubility, hemolysis, and non-fouling (Table [7](#S2.T7)), we selected three non-dominated binders with predicted affinities greater than 7.0 for docking experiments. For these Pareto-optimal peptides, we indeed observed strong binding affinities for both GFAP and RBX1 post-docking, indicating their unique potential for multi-target interaction (Figure [14](#S2.F14)). GFAP is an intermediate filament protein and thus forms a unique rod-like structure with a head domain and a tail domain. The docking positions of all three candidates were along the rod domain, binding in the gap between adjacent rods in the filament. Contrarily, docked candidates to RBX1 consistently bound close to its interaction site of Cullin, rather than at the Skp2 F-box adaptor site (Figure [14](#S2.F14)), indicating that further motif conditioning, as done with recent peptide design language models , would benefit PepTune’s clinical potential.

Figure: Figure 14: PepTune-generated peptides with dual GFAP and RBX1 affinity. Full protein binding location and close-up binding position for (A) dual binder 3, (B) dual binder 2, and (C) dual binder 4 with GFAP (left) and RBX1 (right). Polar contacts within 3.5 Å are annotated and shared polar contacts between binders are highlighted.
Refer to caption: /html/2412.17780/assets/Figures/gfap-ligase_new.png

## 3 Discussion

In this work, we introduce PepTune, a generative framework that achieves multi-objective optimization directly in discrete sequence space. By leveraging MCTS for guidance, PepTune identifies Pareto-optimal peptide SMILES sequences conditioned on diverse therapeutic properties such as binding affinity, solubility, membrane permeability, and hemolysis. Unlike previous guidance methods, which struggle with gradient estimation or rely on projections between continuous and discrete spaces , PepTune operates natively in the discrete latent space. Our approach combines exploration through batched unmasking and reward-based exploitation of classifier predictions, ensuring valid peptide structures with a bond-dependent masking schedule and an invalid loss. Most importantly, unlike recent binder design methods , PepTune requires no obligate target three-dimensional structures or predicted structures (only the target sequence), enabling peptide design to conformationally diverse proteins, optimized for properties beyond local geometric interactions.

Despite its strengths, PepTune relies on synthetic peptide data (e.g., CycloPs ) and rare non-natural amino acids (nAAs) , which may increase synthesis complexity and costs. While we address this through feature-rich embeddings from a pre-trained chemical language model , improving high-quality labeled datasets remains critical for enhancing property prediction accuracy. Furthermore, while we evaluate binding using an external, state-of-the-art docking strategy via AutoDock Vina , there is a lack of biophysical models for other properties optimized via PepTune, including peptide solubility, hemolysis, and membrane permeability, outside of existing predictor algorithms. As such, we are currently conducting in vitro assays to confirm these properties of our generated peptides and will update the manuscript as these results are obtained.

Our next steps are to leverage PepTune for clinically-relevant peptide generation. As a concrete example in this manuscript, we have generated peptides to targets with functional relevance for Alexander disease . Building on our work developing peptide-guided degraders , we will extend PepTune to generate bi-specific peptides that both bind to dysregulated GFAP and recruit various other E3 ubiquitin ligases, especially those that are differentially expressed in astrocytes. Such a system would likely be superior to modalities like proteolysis-targeting chimeras (PROTACs), which are limited to only a minimal, general set of E3 ubiquitin ligases and require either putative or cryptic binding pockets, which do not exist on a large majority of disease-driving targets . Further, by optimizing sequences for BBB permeability (via TfR binding) and astrocyte-specific uptake (via GLAST binding) as we have done here, PepTune may enable the design of complete, specific therapies for this challenging disease. Overall, this work establishes and advances a new paradigm for peptide-based precision medicine, where multi-objective discrete optimization enables therapeutic peptide design with unprecedented control over functional properties.

## 4 Methods

### 4.1 Data Curation and Tokenization.

MDLM Training Data. To train the unconditional masked diffusion language model generator, we collected 11 million peptide SMILES consisting of 7451 sequences from the CycPeptMPDB database , 825,632 unique peptides from SmProt , and approximately 10 million modified peptides generated from CycloPs , which consists of 90% canonical amino acids, 10% unnatural amino acids from SwissSidechain , 10% dextro-chiral alpha carbons, 20% N-methylated amine backbone atoms, and 10% PEGylated peptides. All possible cyclization conformations were attempted on the peptides generated with CycloPs. We used SELFIES to check the integrity of the SMILES sequences.

We split our data by $k$-means clustering into 1000 groups of sequences with similar chemical properties based on their Morgan fingerprint , which is a bit-vector representation of the full peptide sequence where each bit encodes a feature relating to the SMILES atom types, connectivity, and bonding environment. The final dataset was a 0.8 to 0.2 split based on the clusters, maintaining similar diversities of the SMILES strings. Since the degree of masking is evenly spread between $t=0$ to $t=1$ within each training batch, grouping similar SMILES in the same batch ensures the model learns to reconstruct a diverse set of peptide SMILES from various degrees of masking.

Dynamic Batching. We applied dynamic batching to handle variable-length token sequences and increase computational efficiency. Inspired by ESM-2’s dynamic batching technique , input SMILES are sorted by length to maximize the utility of GPU memory. The maximum tokens per GPU was set to 16,000.

SMILES Tokenization. To enable the novel generation of non-natural amino acids containing cyclizations and diverse backbone and side-chain modifications, we trained our generative diffusion model on Simplified Molecular-Input Line-Entry System (SMILES) representations of peptides. We experimented with several tokenization schemes that capture common motifs in the training data to enhance the generation of valid peptide SMILES. We find that the SMILES Pair Encoding (SPE) tokenization scheme with the PeptideCLM vocabulary of 581 SMILES tokens and 5 special tokens with an average length of four characters per token, demonstrated superior performance, generating precise but valid peptides (Appendix [D.2](#A4.SS2)).

Classifier Data and Training. We trained our membrane permeability XGBoost regression model using experimentally-validated peptide SMILES consisting of 22,040 SMILES sequences obtained from the ChEMBL database and 7451 sequences from the CycPeptMPDB database .

We collected binding affinity, solubility, hemolysis, and non-fouling data for classifier training from the PepLand and PeptideBERT datasets . The binding affinity data was split within groups of peptides with weak, medium, and strong binding scores with a 0.8/0.2 ratio. The classifier was trained with cross-attention between ESM-2 protein embeddings and PeptideCLM SMILES representations . The hyperparameters were chosen with 50 trials of OPTUNA search . For other classifiers, data were randomly shuffled and split into 0.8/0.1/0.1 ratio for train, validation, and test. XGBoost classifiers were applied on PeptideCLM embeddings with 50 trials of OPTUNA search for the optimal boosted tree hyperparameters.

### 4.2 Unconditional Masked Discrete Diffusion Model

Notation. Let $\mathbf{x}_{0}\in\{0,1\}^{|\mathcal{V}|}$ represent the one-hot vector of a token in a sequence in the training data and $\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\in\Delta^{|\mathcal{V}|}$ be the vector of predicted token probabilities across the vocabulary $\mathcal{V}$ given the current state $\mathbf{z}_{t}$ at time $t$. In most contexts, $\mathbf{x}_{0}$ will be used to denote a single token, but when discussing the full sequence, $\mathbf{x}_{0}^{(\ell)}$ is used to denote the token at position $\ell$ in the sequence. Let $T$ denote the total number of time steps in the discrete forward and reverse diffusion processes. In the context of all time steps, we expand $t$ to $t(n)\in(0,1]$ when denoting a single time step in the forward and backward diffusion process with $n\in[1\dots T-1]$. Let $s(n)=t(n)-\frac{1}{T}$ denote the previous time step in the forward process. Then, let $\mathbf{z}_{t(n)}$ and $\mathbf{z}_{s(n)}$ denote the state of a specific token at time $t(n)$ and $s(n)$ in the diffusion process, respectively. Let $\alpha_{t}(\mathbf{x}_{0}):\mathbb{R}^{|\mathcal{V}|}\to\mathbb{R}$ denote a function that takes the unmasked token $\mathbf{x}_{0}$ and outputs a value in [0,1] representing the probability of remaining unmasked at time $t$ in the forward diffusion process. Let $\mathbf{b}\in\mathbb{R}^{|\mathcal{V}|}$ denote a vector with ones at indices of peptide bond tokens and zeroes at all remaining indices and let $\mathbf{x}_{0}=\mathbf{b}$ indicate that $\mathbf{x}_{0}$ is a peptide-bond token.

Bond-Dependent Masking Schedule. Since all peptides follow a distinct SMILES structure consisting of un-modified or modified peptide bonds before and after each central carbon atom with an amino acid side chain, we hypothesized that applying bond-dependent masking and unmasking schedules would allow the reverse diffusion process to learn to unmask the crucial structural components of a peptide SMILES that are common across all peptides before filling in the segments in-between with diverse amino acid side-chains.

Extending previous work in state-dependent masking , we devised a bond-dependent masking schedule where the probability of masking a token within a peptide bond increases at a slower rate in earlier times $t$ in the masking process compared to non-peptide bond tokens. To achieve this, we define the discrete-time log-linear masking schedule $\sigma(t)=-\log(1-t)$ for non-peptide bond tokens and the log-polynomial masking schedule $\sigma(t)=-\log(1-t^{w})$ for peptide-bond tokens. We show in Appendix [B.1](#A2.SS1) that the continuous-time probability of remaining unmasked at time $t$ in the forward diffusion process is given by the function $\alpha_{t}(\mathbf{x}_{0}):\mathbb{R}^{|\mathcal{V}|}\to\mathbb{R}$ that takes the vector encoding the token $\mathbf{x}_{0}$ and returns a probability.

$$ $\displaystyle\alpha_{t}(\mathbf{x}_{0})=\begin{cases}1-t^{w}&\mathbf{x}_{0}=\mathbf{b}\\ 1-t&\mathbf{x}_{0}\neq\mathbf{b}\end{cases}$ (5) $$

where $\mathbf{b}$ represents the with ones at indices of peptide bond tokens and zeroes at all remaining indices, including tokens within modified peptide-bonds identified with our BondMask function (Appendix [7](#alg7)). Since the probability of transitioning to a [MASK] token at time $t$ is given by $1-\alpha_{t}(\mathbf{x}_{0})$, there is a lower probability $t^{w}$ for $t\in(0,1]$ of masking a peptide bond token than the probability $t$ of masking a non-peptide bond token, especially in earlier time steps for smaller $t$ (Fig. [15](#A2.F15)A). As $t\to 1$, $\alpha_{t}(\mathbf{x}_{0})\to 0$ and the probability of masking for both peptide and non-peptide bond tokens approaches 1, ensuring that the model can learn to reconstruct the full token sequence during the reverse diffusion process.

With our bond-dependent masking rate $\alpha_{t}(\mathbf{x}_{0})$, we define the forward transition matrix as

$$ $\displaystyle q(\mathbf{z}_{t}|\mathbf{x}_{0})=\text{Cat}(\mathbf{z}_{t};\alpha_{t}(\mathbf{x}_{0})\mathbf{x}_{0}+(1-\alpha_{t}(\mathbf{x}_{0}))\mathbf{m})$ (6) $$

The reverse transition from state $t\to s$ given the bond-dependent forward masking schedule is derived in Appendix [B.2](#A2.SS2) as

$$ $\displaystyle q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})=\begin{cases}\left(\left(\frac{s}{t}-\frac{s^{w}}{t^{w}}\right)\mathbf{b}+\frac{t-s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{0}\mathbf{x}_{0}+\left(\left(\frac{s^{w}}{t^{w}}-\frac{s}{t}\right)\mathbf{b}+\frac{s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{0}\mathbf{m}&\mathbf{z}_{t}=\mathbf{m}\\ \mathbf{z}_{t}&\mathbf{z}_{t}\neq\mathbf{m}\end{cases}$ (7) $$

To estimate the reverse posterior, we define a parameterized RoFormer model $\mathbf{x}_{\theta}(\mathbf{z}_{t},t):\mathcal{V}^{L}\times[0,1]\to\Delta^{|\mathcal{V}|}$ that takes the partially masked sequence at time $t$ and predicts a vector of token probabilities over the $|\mathcal{V}|$-dimensional simplex for each position in the sequence. By substituting $\mathbf{x}_{0}=\mathbf{x}_{\theta}(\mathbf{z}_{t},t)$ into the true reverse transition, we get the predicted reverse transition distribution.

$$ $\displaystyle p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})=\begin{cases}\left(\left(\frac{s}{t}-\frac{s^{w}}{t^{w}}\right)\mathbf{b}+\frac{t-s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\mathbf{z}_{s}+\left(\left(\frac{s^{w}}{t^{w}}-\frac{s}{t}\right)\mathbf{b}+\frac{s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{0}\mathbf{m}&\mathbf{z}_{t}=\mathbf{m}\\ \mathbf{z}_{t}&\mathbf{z}_{t}\neq\mathbf{m}\end{cases}$ (8) $$

For larger $w$, peptide bonds are masked at later timesteps, encouraging earlier unmasking in the reverse diffusion process. However, setting $w$ too large can result in the model overfitting to the dataset . Empirically, we found that $w=3$ increased peptide validity while maintaining diversity across generated samples.

SUBS Parametrization. Following Sahoo et al. , we parameterize the reverse diffusion model using SUBS parametrization, which enforces zero-masking probability and carry-over unmasking. This strategy enforces the constraints applied in the forward diffusion process and has been shown to minimize perplexity .

- 1.
Zero Masking Probability. The forward process operates under the assumption that a token can only be masked once across times $t=0\to 1$. It follows that the probability of a sequence being masked in the reverse diffusion process is zero. To enforce this property, we explicitly set the log-probability of the sequence being masked in the reverse transition distribution to zero such that if a token is unmasked at time $t$, it remains unmasked for all timesteps from $t\to 0$.
$\displaystyle\langle\mathbf{x_{\theta}(\mathbf{z}_{t},t)},\mathbf{m}\rangle=0$
(9)
- 2.
Carry-Over Unmasking. For each transition in the forward pass, all tokens either remain unchanged or are masked. Therefore, in the reverse process, when a token is unmasked at time $t$, the unmasked token is copied over all time steps from $t\to 0$. We enforce this by setting the logits vector equal to the one-hot encoding for the unmasked token.
$\displaystyle\mathbf{x}_{\theta}(\mathbf{z}_{t},t)=\mathbf{z}_{t}\;\text{ if }\;\mathbf{z}_{t}\neq\mathbf{m}$
(10)

### 4.3 Loss Functions

Bond-Dependent Continuous-Time Diffusion Loss. To optimize the parameters $\theta$ of the reverse diffusion model, we maximize the evidence lower bound (ELBO) of the distribution $\log p(\mathbf{x}_{0})$, which is the log-probability distribution of generating the peptide sequences $\mathbf{x}_{0}$ present in the dataset. Therefore, we define our loss function as the negative ELBO (NELBO) given by

$$ $\displaystyle\mathcal{L}_{\text{NELBO}}$ $\displaystyle=\underbrace{\mathbb{E}_{q(\mathbf{z}_{t(1)}|\mathbf{x}_{0})}\bigg{[}-\log p_{\theta}(\mathbf{x}_{0}|\mathbf{z}_{t(1)})\bigg{]}}_{\text{reconstruction loss}}+\underbrace{\mathbb{E}_{q(\mathbf{z}_{t(T)},\mathbf{z}_{s(T)}|\mathbf{x}_{0})}\left[-\log\frac{p_{\theta}(\mathbf{z}_{t(T)})}{q(\mathbf{z}_{t(T)}|\mathbf{z}_{s(T)})}\right]}_{\text{prior loss}}$ $\displaystyle\quad+\underbrace{\sum_{n=1}^{T-1}\mathbb{E}_{q(\mathbf{z}_{s(n)},\mathbf{z}_{t(n)},\mathbf{z}_{t(n+1)}|\mathbf{x}_{0})}\bigg{[}-\log\frac{p_{\theta}(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)})}{q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)})}\bigg{]}}_{\text{diffusion loss}}$ (11) $$

Training on samples masked for continuous values of $t\sim\text{Uniform}(0,1)$ yields a tighter lower bound compared to discrete values of $t$ . When the predicted probability distribution $\mathbf{x}^{(\ell)}_{\theta}(\mathbf{z}_{t},t)$ is exactly the one-hot encoding vector $\mathbf{x}_{0}^{(\ell)}$ for each token at position $\ell$ in the true sequence, the loss reduces to 0, which supports our objective.

By our derivation in Appendix [A.2](#A1.SS2), our continuous-time bond-dependent NELBO separates into the sum of the negative log-losses (NLLs) for all non-peptide bond tokens that follow a log-linear masking schedule and the sum of the NLLs for all peptide bond tokens that follow a log-polynomial schedule.

$$ $\displaystyle\mathcal{L}^{\infty}_{\text{NELBO}}=\mathbb{E}_{t\sim\mathcal{U}(0,1)}\mathbb{E}_{q(\mathbf{z}_{t}|\mathbf{x}_{0})}\bigg{[}-\sum_{\ell:\mathbf{x}^{(\ell)}_{0}=\mathbf{b}}\frac{w}{t}\log\langle\mathbf{x}^{(\ell)}_{0},\mathbf{x}^{(\ell)}_{\theta}(\mathbf{z}_{t},t)\rangle-\sum_{\ell:\mathbf{x}^{(\ell)}_{0}\neq\mathbf{b}}\frac{1}{t}\log\langle\mathbf{x}^{(\ell)}_{0},\mathbf{x}^{(\ell)}_{\theta}(\mathbf{z}_{t},t)\rangle\bigg{]}$ (12) $$

Since the NLL term is minimized when the predicted probability of the ground truth token is close to 1, we show that applying the log-polynomial masking schedule for an exponent $w>1$ scales the diffusion loss NELBO by a factor of $w$ from the log-linear schedule. However, for earlier timesteps as $t\to 0$, both NLL weights increase to $\infty$, ensuring high precision in the final unmasking steps.

Given that peptide bonds form the fundamental backbone structure of a peptide, our bond-dependent masking strategy for peptide bonds acts as a weighted loss that introduces a higher penalty when the token predictions at positions of peptide bonds are inconsistent from the ground truth tokens during training, forcing the model to learn the specific structure of peptide SMILES strings in a vast space of SMILES strings that are not valid peptides.

Invalid Peptide Loss. To further discourage the generation from predicting token logits that produce invalid peptide SMILES, we incorporate a loss to penalize sampling of invalid peptide SMILES during training by taking the argmax of the predicted logits and assigning a penalty based on our peptide validity filter (Appendix [8](#alg8)). Given the peptide sequence $\tilde{\mathbf{x}}^{(\ell)}\in\{0,1\}^{K}$ generated from the argmax tokens with the highest probability from the predicted logits $\mathbf{x}^{(\ell)}_{\theta}(\mathbf{z}_{t},t)$, we minimize a penalty determined by our validity filter $\mathbf{1}[\tilde{\mathbf{x}}_{0}\text{ is Invalid}]$ which returns 0 when the sequence is a valid peptide SMILES and 1 when the sequence either is not a valid SMILES or does not correspond to a peptide sequence. Since the argmax function is not differentiable, we use the softmax probability of the sampled tokens to scale the penalty score for each token which acts as a scalar multiplier in the loss function.

$$ $\displaystyle\mathcal{L}_{\text{invalid}}$ $\displaystyle=\sum_{\ell=1}^{L}\tilde{\mathbf{x}}_{0}^{(\ell)\top}\text{SM}\big{(}\mathbf{x}_{\theta}^{(\ell)}(\mathbf{z}_{t},t)\big{)}\cdot\mathbf{1}[\tilde{\mathbf{x}}_{0}\text{ is Invalid}]$ $\displaystyle=\sum_{\ell=1}^{L}\frac{\exp(x^{(\ell)}_{\theta,k})}{\sum_{j=1}^{K}\exp(x_{\theta,j}^{(\ell)})}\cdot\mathbf{1}[\tilde{\mathbf{x}}_{0}\text{ is Invalid}]$ (13) $$

where $k=\arg\max_{j}(\mathbf{x}^{(\ell)}_{\theta}\big{(}\mathbf{z}_{t},t)\big{)}$ is the token with the highest predicted probability at position $\ell$ of the sequence.

Differentiating the invalid loss with respect to the probability vector $\mathbf{x}_{\theta}^{(\ell)}(\mathbf{z}_{t},t)$ for position $\ell$, we derive the gradient with respect to the predicted probability of the sampled token $j=k$ and all other tokens in the vocabulary $j\neq k$ in Appendix [B.1](#A2.SS1) as

$$ $\displaystyle\nabla\mathcal{L}_{\text{invalid}}=\begin{cases}\text{SM}(x^{(\ell)}_{\theta,k})\left(1-\text{SM}(x^{(\ell)}_{\theta.k})\right)&j=k\\ -\text{SM}(x^{(\ell)}_{\theta,j})\text{SM}(x^{(\ell)}_{\theta,k})&j\neq k\end{cases}$ (14) $$

Minimizing this objective function updates the parameters to lower the predicted probabilities for tokens that result in invalid peptide SMILES and increase the probabilities of the remaining tokens proportional to their original distribution such that the relative probability distribution of all other tokens $j\neq k$ is maintained.

Training. To train the MDLM to accurately approximate the true reverse transition distribution $q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})$ of a training sample $\mathbf{x}_{0}$ for all continuous timesteps $t=1\to 0$, we train a parameterized model $\mathbf{x}_{\theta}(\mathbf{z}_{t},t)$ that takes the current sequence $\mathbf{z}_{t}$ and returns a $|\mathcal{V}|$-dimensional vector of predicted probabilities of transitioning to each token at time $s<t$ (Algorithm [1](#alg1)). For each dynamic training batch $B$, we randomly sample $|B|$ values $t\in\text{Uniform}(0,1)$ and off-set each time $t$ by $\vec{\delta}=\left[0,\frac{1}{|B|},\frac{2}{|B|},\dots,\frac{|B|-1}{|B|},1\right]$ to get a vector $\vec{t}=(\vec{t}+\vec{\delta})\text{ mod }\mathbf{1}$ of evenly distributed time steps to ensure the model learns to regenerate the clean sample $\mathbf{z}_{t}$ for a continuous range of time steps. After applying bond-dependent masking to each training sequence, obtaining the predicted probabilities $\mathbf{x}_{\theta}(\mathbf{z}_{t},t)$ and sampling the discrete sequence $\tilde{\mathbf{x}}_{0}$ from greedy argmax sampling, we minimize the total loss function $\mathcal{L}$ given by

$$ $\displaystyle\mathcal{L}$ $\displaystyle=\mathcal{L}_{\text{NELBO}}^{\infty}+\mathcal{L}_{\text{invalid}}$ $\displaystyle=\frac{1}{|B|}\sum_{\mathbf{x}_{0}\in B}\bigg{(}-\sum_{\ell:\mathbf{x}^{(\ell)}_{0}=\mathbf{b}}\frac{w}{t}\log\langle\mathbf{x}^{(\ell)}_{0},\mathbf{x}^{(\ell)}_{\theta}(\mathbf{z}_{t},t)\rangle-\sum_{\ell:\mathbf{x}^{(\ell)}_{0}\neq\mathbf{b}}\frac{1}{t}\log\langle\mathbf{x}^{(\ell)}_{0},\mathbf{x}^{(\ell)}_{\theta}(\mathbf{z}_{t},t)\rangle$ (15) $\displaystyle+\sum_{\ell=1}^{L}\tilde{\mathbf{x}}_{0}^{(\ell)\top}\text{SM}\big{(}\mathbf{x}_{\theta}^{(\ell)}(\mathbf{z}_{t},t)\big{)}\cdot\mathbf{1}[\tilde{\mathbf{x}}_{0}\text{ is Invalid}]\bigg{)}$ $$

By increasing batch size and applying dynamic batching (Methods [4.1](#S4.SS1)), we obtain a tighter ELBO of the true distribution $\log p(\mathbf{x}_{0})$. The model used to generate the validation results in this manuscript is trained on our in-house 8$\times$A6000 Nvidia GPUs (50G memory) for 1600 GPU hours using the AdamW optimizer with a learning rate of 0.0003 and weight decay of 0.075.

Sampling. To sample from the unconditional PepMDLM model, we start with a sequence of length $L$ of only [MASK] tokens. We first compute the diffusion time steps $t\in\{\frac{1}{T},\frac{2}{T},\dots,1\}$ where $T=128$. From the predicted token probabilities $\mathbf{x}_{\theta}(\mathbf{z}_{t},t)$ generated by feeding $\mathbf{z}_{t}$ through the trained RoFormer backbone, we compute the reverse transition token distribution $p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})$ following Equation ([8](#S4.E8)) and perform Gumbel-max sampling to get the next token $\mathbf{z}_{s}$.

$$ $\displaystyle\mathbf{z}^{(\ell)}_{s}$ $\displaystyle\sim\arg\max\left(\log p_{\theta}(\mathbf{z}^{(\ell)}_{s}|\mathbf{z}^{(\ell)}_{t})+\mathbf{G^{(\ell)}}\right)$ ( $\mathbf{G}^{(\ell)}\in\mathbb{R}^{|\mathcal{V}|}$ ) $\displaystyle G^{(\ell)}_{i}$ $\displaystyle=-\log(-\log(u^{(\ell)}_{i}+\epsilon)+\epsilon)$ ( $u^{(\ell)}_{i}\sim\text{Uniform}(0,1)$ ) $$

where $G_{i}$ is the Gumbel noise applied to the token probability at index $i$ for position $\ell$ and $\epsilon=1e-10$. Then, we return the newly sampled tokens only when $\mathbf{z}^{(\ell)}_{t}=\mathbf{m}$, while keeping all unmasked tokens unchanged. After $T$ timesteps, we obtain a fully unmasked sequence $\mathbf{x}$.

### 4.4 Multi-Objective Guidance for Discrete Diffusion

In this section, we introduce the concept of Pareto dominance and non-dominance for multiple objectives and describe the Monte Carlo Tree Search (MCTS)-based algorithm for generating a set of Pareto-optimal sequences using our trained masked discrete diffusion model.

Pareto Optimization. When optimizing sequences for multiple objectives (e.g. affinity to multiple protein targets, membrane permeability, solubility, etc.), there is likely no single best sequence that achieves the highest score across all objectives. Optimizing one objective often leads to sacrificing performance on another objective.

Therefore, we focus on finding a set of Pareto optimal sequences that minimize the trade-offs between objectives to achieve overall optimal performance across all objectives. Formally, Pareto-optimal sequences (or non-dominated sequences) cannot be further optimized in any single objective without sacrificing performance in another objective.

Let $\mathbf{s}(\mathbf{x})=[s_{1}(\mathbf{x}),\dots,s_{K}(\mathbf{x})]\in\mathbb{R}^{K}$ be a vector of scores that measures the performance of a sequence $\mathbf{x}$ in $K$ different objectives, with higher scores indicating better performance. A sequence $\mathbf{x}^{*}$ is said to dominate another sequence $\mathbf{x}$ (denoted as $\mathbf{x}^{*}\succ\mathbf{x}$) if and only if it satisfies the following property. For all objectives $k\in[1\dots K]$, the score for the $k$th objective for $\mathbf{x}^{*}$ is greater than or equal to the score for the $k$th objective for $\mathbf{x}$, and for at least one objective $k\textquoteright$, the score for $\mathbf{x}^{*}$ is strictly greater than the score for $\mathbf{x}$.

$$ $\displaystyle\small\underbrace{\mathbf{s}(\mathbf{x}^{*})\succ\mathbf{s}(\mathbf{x)}}_{\mathbf{x}^{*}\text{ dominates }\mathbf{x}}\;\;\;\text{iff}\;\;\;\;\underbrace{\forall k\in[1,K]\;\;s_{k}(\mathbf{x}^{*})\geq s_{k}(\mathbf{x})}_{\mathbf{x}^{*}\text{ is no worse than }\mathbf{x}\text{ in any objective}}\;\land\underbrace{\;\exists k^{\prime}\in[1,K]\;\;s_{k^{\prime}}(\mathbf{x}^{*})\geq s_{k^{\prime}}(\mathbf{x})}_{\mathbf{x}^{*}\text{ is strictly better than }\mathbf{x}\text{ in at least one objective}}$ (16) $$

A Pareto-optimal sequence $\mathbf{x}$ is a sequence where there does not exist another sequence $\mathbf{x}^{*}$ in the current Pareto-optimal set $\mathcal{P}^{*}$ that dominates it. Since there are trade-offs between objectives, this does not mean that $\mathbf{x}$ is dominant to all other sequences.

$$ $\displaystyle\underbrace{\nexists\mathbf{x}^{*}\in\mathcal{P}^{*}\;\;\text{s.t.}\;\;\mathbf{s}(\mathbf{x}^{*})\succ\mathbf{s}(\mathbf{x})}_{\mathbf{x}\text{ is non-dominated}}$ (17) $$

We define the Pareto front as the set of non-dominated sequences $\mathbf{x}$ and their $K$-dimensional objective score vectors.

$$ $\displaystyle\mathcal{P}^{*}=\left\{\big{(}\mathbf{x},\mathbf{s}(\mathbf{x})\big{)}\;|\;\nexists\mathbf{x}^{*}\in\mathcal{P}^{*}\;\;\text{s.t.}\;\;\mathbf{s}(\mathbf{x}^{*})\succ\mathbf{s}(\mathbf{x})\right\}$ (18) $$

Since infinitely many trade-offs can exist between the $K$ objectives, there can be an infinite number of Pareto-optimal sequences. Therefore, multi-objective optimization aims to approximate a finite set of Pareto-optimal sequences with a reasonable number of iterations.

Notation. Let $\mathbf{z}_{t}$ denote the partially unmasked sequence at time $t$. $\mathbf{z}_{t}$ also corresponds to a node in the MCTS tree with a set of $M$ children nodes denoted as $\text{children}(\mathbf{z}_{t})=\{\mathbf{z}_{s,1},\dots,\mathbf{z}_{s,M}\}$. Each child node is itself a partially unmasked sequence at time $s$ derived from sampling the MDLM reverse posterior $p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})$. The children nodes at each iteration of MCTS are rolled out into a set of clean sequences denoted as $\{\mathbf{x}_{s,1},\dots,\mathbf{x}_{s,M}\}$, for each of which we compute a score vector $\mathbf{s}(\mathbf{x}_{s,i})\in\mathbb{R}^{K}$ and a rewards vector $\mathbf{r}(\mathbf{x}_{s,i})\in\mathbb{R}^{K}$, where $K$ is the number of objectives guiding the MCTS search.

Let $\mathcal{P}^{*}=\{\mathbf{x}^{*}_{n}\}$ be the set of $|\mathcal{P}^{*}|$ Pareto non-dominated sequences indexed $n\in[1\dots|\mathcal{P}^{*}|]$, which is updated at each iteration. At each node $\mathbf{z}_{t}$, we store a cumulative rewards vector $\mathbf{W}(\mathbf{z}_{t})$ and a counter for the number of times the node has been visited across all iterations $N_{\text{visit}}(\mathbf{z}_{t})$. Finally, we denote the iteration index as $i\in[1\dots N_{\text{iter}}]$, where $N_{\text{iter}}$ is the total number of search iterations.

Initialization. We initialize a sequence of length $L$ with only [MASK] tokens as the root node of the MCTS tree at $t=0$ and an empty set $\mathcal{P}^{*}$ that will maintain clean sequences with Pareto-optimal score vectors. We initialize a set of scoring functions $\mathbf{s}:\mathcal{V}^{L}\to\mathbb{R}^{K}$ that takes a clean sequence $\mathbf{x}_{s,i}\in\mathcal{V}^{L}$ generated from the partially masked sequence $\mathbf{z}_{s,i}$ and outputs a vector of real values $\mathbf{s}(\mathbf{x}_{s,i})\in\mathbb{R}^{K}$ that measures its performance in each of the $K$ objectives. We also set the hyperparameters, including the number of iterations $N_{\text{iter}}$, the number of children $M$, and the length of the token sequence $L$.

At each iteration, four steps are performed to update the set of Pareto optimal solutions: traversing the tree by selecting the best child node until reaching a leaf node (selection), expanding the leaf node into $M$ distinct partially unmasked sequences (expansion), fully unmasking each child node into a clean sequence and computing multi-objective score and reward vector (rollout), and finally backpropagating the total rewards to the predecessor nodes to guide the selection process at the next iteration (backpropagation).

Selection. At each iteration, we traverse the tree starting at the root node (fully masked sequence) $\mathbf{z}_{T}$ and selecting a child node based on the selection score vector $\mathbf{U}(\mathbf{z}_{t},\mathbf{z}_{s,i})$ that balances child nodes that generate high reward sequences from previous iterations and unexplored unmasking actions that could lead to a larger pool of diverse sequences.

$$ $\displaystyle\mathbf{U}(\mathbf{z}_{t},\mathbf{z}_{s,i})=\frac{\mathbf{W}(\mathbf{z}_{s,i})}{N_{\text{visit}}(\mathbf{z}_{s,i})}+c\cdot p_{\theta}(\mathbf{z}_{s,i}|\mathbf{z}_{t})\frac{\sqrt{N_{\text{visit}}(\mathbf{z}_{t})}}{1+N_{\text{visit}}(\mathbf{z}_{s,i})}$ (19) $$

The first term is the cumulative reward vector $\mathbf{W}(\mathbf{z}_{s,i})$ normalized by the number of times the node was previously visited. This guides the selection process towards the unmasking step that has resulted in fully unmasked sequences with optimal properties without biasing towards highly visited nodes. The second term is a scalar added element-wise to the normalized rewards. The scalar probability of the unmasking step based on the unconditional reverse posterior $p_{\theta}(\mathbf{z}_{s,i}|\mathbf{z}_{t})$ guides the selection towards the unmasking step with the highest probability to generate a valid peptide based on the pre-trained MDLM. When the number of times the parent node has been explored is high and the number of visits to a child node is low, the $\frac{\sqrt{N_{\text{visit}}(\mathbf{z}_{t})}}{1+N_{\text{visit}}(\mathbf{z}_{s,i})}$ term encourages exploration of the unexplored unmasking scheme given that $p_{\theta}(\mathbf{z}_{s,i}|\mathbf{z}_{t})$ is sufficiently high. However, as the number of visits to a child node increases, the impact of the second term decreases and the cumulative rewards dominate the selection score vector. $c$ is a scalar hyperparameter that determines the degree of exploration compared to exploiting high-reward nodes, which is selected to be $c=0.1$.

Then, we select uniformly at random from the pool of children nodes $\mathbf{z}_{s,i}\in\mathcal{P}^{*}_{\text{select}}$ whose selection score vectors are non-dominated, such that there doesn’t exist another child $\mathbf{z}_{s,j}$ where the selection score across each of the $K$-objectives is equal to the score of $\mathbf{z}_{s,i}$ and there exists a score strictly greater than $\mathbf{z}_{s,i}$.

$$ $\displaystyle\mathcal{P}^{*}_{\text{select}}=\left\{\mathbf{z}_{s,i}\;|\;\nexists\mathbf{z}_{s,j}\in\text{children}(\mathbf{z}_{t})\;\;\text{s.t.}\;\;\mathbf{U}(\mathbf{z}_{t},\mathbf{z}_{s,j})\succ\mathbf{U}(\mathbf{z}_{t},\mathbf{z}_{s,i})\right\}$ (20) $$

If the selected node is a non-leaf node, the loop repeats with the selected node $\mathbf{z}_{s,i}$ as the new parent node. If a fully unmasked node with $t=0$ is reached, we restart the selection process from the root node. Once a leaf node is reached, the loop ends and the next step executes.

Expansion. At the iteration at time $t$, we sample $M$ sequences from the reverse posterior $p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})$ defined in Equation ([8](#S4.E8)) to get a set of partially masked sequences which form the set of children nodes of $\mathbf{z}_{t}$: $\text{children}(\mathbf{z}_{t})=\{\mathbf{z}_{s,1},\dots,\mathbf{z}_{s,M}\}$. All the children nodes are added to the tree.

To ensure that the expansion step results in $M$ distinct unmasking steps, we experimented with two different batched unmasking techniques from the single partially masked sequence at a parent node. For the first method, we repeated the array corresponding to the parent node tokens over $M$ dimensions and added independently sampled Gumbel noise values $G^{(\ell)}_{i,j}$, where $i$ denotes the sequence in the batch, $\ell$ denotes the position in the sequence, and $j$ denotes the token index.

$$ $\displaystyle\tilde{p}_{\theta,i}\big{(}\mathbf{z}^{(\ell)}_{s,i}|\mathbf{z}^{(\ell)}_{t})$ $\displaystyle=\log p_{\theta}\big{(}\mathbf{z}^{(\ell)}_{s,i}|\mathbf{z}^{(\ell)}_{t}\big{)}+\mathbf{G}^{(\ell)}_{i}$ (21) $\displaystyle G^{(\ell)}_{i,j}$ $\displaystyle=-\log\left(-\log(u^{(\ell)}_{i,j}+\epsilon)+\epsilon\right)\;\;\;\;u^{(\ell)}_{i,j}\sim\text{Uniform}(0,1)$ (22) $$

where $\tilde{p}_{\theta,i}$ denotes the $i$th modified reverse transition distribution after applying Gumbel noise independently sampled for each token at each position for each sequence $i$ in the batch, where $i=[1\dots M]$. Then, for each position $\ell$ in the sequence, we sample $M$ distinct child sequences from the $M$ distinct distributions $\mathbf{z}^{(\ell)}_{s,i}\sim\tilde{p}_{\theta,i}\big{(}\mathbf{z}^{(\ell)}_{s,i}|\mathbf{z}^{(\ell)}_{t}\big{)}$.

The second method involves taking the softmax (denoted as SM) across the top $k$ probabilities after applying Gumbel noise and drawing random samples from the re-normalized softmax distribution over only the top $k$ most probable tokens.

$$ $\displaystyle\tilde{p}_{\theta,i}\big{(}\mathbf{z}^{(\ell)}_{s,i}|\mathbf{z}^{(\ell)}_{t})$ $\displaystyle=\text{SM}\bigg{(}\text{top}k\big{\{}\log p_{\theta}\big{(}\mathbf{z}^{(\ell)}_{s,i}|\mathbf{z}^{(\ell)}_{t}\big{)}+\mathbf{G}_{i}^{(\ell)}\big{\}}\bigg{)}$ (23) $$

After empirical experimentation, we found that the first method results in higher diversity across sequences whereas the second method prevents unlikely tokens. Since the reward generated by a sequence ultimately determines whether it is selected in subsequent iterations, we chose the first method to allow for greater exploration during the expansion step.

Rollout. From each child node generated at time $s$, we completely unmask the sequence by greedily sampling the argmax tokens from the predicted reverse transition distribution ${p}^{(\ell)}_{\theta,i}(\mathbf{z}_{s^{\prime}}|\mathbf{z}_{s})$ for all remaining time steps $s\to 0$ and $s^{\prime}=s-\frac{1}{T}$ to get a set of clean sequences $\{\mathbf{x}_{s,1},\dots,\mathbf{x}_{s,M}\}$ of SMILES tokens. Then, we feed each clean sequence $\mathbf{x}_{s,i}$ as input to the scoring functions for all of the $K$ objectives to generate the score vector $\mathbf{s}(\mathbf{x}_{s,i})=\big{[}s_{1}(\mathbf{x}_{s,i}),\dots,s_{K}(\mathbf{x}_{s,i})\big{]}\in\mathbb{R}^{K}$. Then, we use the score vector to compute a vector of rewards $\mathbf{r}(\mathbf{x}_{s,i})=\big{[}r_{1}(\mathbf{x}_{s,i}),\dots,r_{K}(\mathbf{x}_{s,i})\big{]}\in\mathbb{R}^{K}$.

The reward of a child node sequence for the $k$th objective is the fraction of the sequences $\mathbf{x}^{*}_{n}$ in the current set of Pareto-optimal sequences $\mathcal{P}^{*}$ where the child node has a higher classifier score in that objective. Specifically, the reward for the $i$th child node $\mathbf{z}_{s,i}$ and the resulting unmasked sequence $\mathbf{x}_{s,i}$ for the $k$th objective is given by

$$ $\displaystyle r_{k}(\mathbf{x}_{s,i})=\frac{1}{|\mathcal{P}^{*}|}\sum_{n=1}^{|\mathcal{P}^{*}|}\mathbf{1}\big{[}s_{k}(\mathbf{x}_{s,i})\geq s_{k}(\mathbf{x}^{*}_{n})\big{]}$ (24) $$

where $\mathbf{1}$ is an indicator function that returns 1 if the score for the $k$th objective of the $i$th child node is greater than or equal to the score of the $n$th sequence in the Pareto-optimal set $\mathcal{P}^{*}$. In parallel to computing the reward, we add all non-dominated children sequences to the set of Pareto optimal sequences $\mathcal{P}^{*}$ and remove all dominated sequences (Algorithm [6](#alg6)).

$$ $\displaystyle\mathcal{P}^{\prime*}$ $\displaystyle=\mathcal{P}^{*}\cup\big{\{}(\mathbf{z}_{s,i},\mathbf{s}(\mathbf{x}_{s,i}))\;|\;\forall\tilde{\mathbf{x}}\in\mathcal{P}^{*}\;\;\mathbf{s}(\mathbf{x}_{s,i})\succeq\mathbf{s}(\tilde{\mathbf{x}})\big{\}}$ (25) $\displaystyle\mathcal{P}^{\prime*}$ $\displaystyle=\mathcal{P}^{*}\setminus\big{\{}\tilde{\mathbf{x}}\;|\;\exists\mathbf{x}_{s,i}\in\text{children}(\mathbf{z}_{t})\;\text{s.t.}\;\mathbf{s}(\mathbf{x}_{s,i})\succ\mathbf{s}(\tilde{\mathbf{x}})\big{\}}$ (26) $$

In Appendix [D.1](#A4.SS1), we show a proof-of-concept for a time-dependent multi-objective guidance strategy where the update to the Pareto-optimal set $\mathcal{P}^{*}$ depends on the rewards for only a subset of the $K$ objectives that varies depending on the current iteration, enabling the prioritization of properties with larger influence on peptide structure and function in earlier iterations and fine-tuning on additional properties in later iterations.

Back-propagation. At each child node $\mathbf{z}_{s,i}$, the reward vector $\mathbf{r}(\mathbf{x}_{s,i})$ is used to initialize the cumulative reward vector $\mathbf{W}(\mathbf{z}_{s,i})$, and the number of visits $N_{\text{visits}}(\mathbf{z}_{s,i})$ is initialized to 1.

$$ $\displaystyle\mathbf{W}(\mathbf{z}_{s,i})$ $\displaystyle\leftarrow\mathbf{r}(\mathbf{x}_{s,i})$ (27) $\displaystyle N_{\text{visit}}(\mathbf{z}_{s,i})$ $\displaystyle\leftarrow 1$ (28) $$

Then, we backtrack through the predecessor nodes of $\mathbf{z}_{s,i}$ up to the root node $\mathbf{z}_{T}$, adding the child reward vector to the cumulative reward vector and incrementing the number of visits for each node in the path. For all nodes from $\mathbf{z}_{t}=\text{parent}(\mathbf{z}_{s,i})$ to $\mathbf{z}_{t}=\mathbf{z}_{T}$ we apply the following update

$$ $\displaystyle\mathbf{W}(\mathbf{z}_{t})$ $\displaystyle\leftarrow\mathbf{W}(\mathbf{z}_{t})+\sum_{i=1}^{M}\mathbf{r}(\mathbf{x}_{s,i})$ (29) $\displaystyle N_{\text{visit}}(\mathbf{z}_{t})$ $\displaystyle\leftarrow N(\mathbf{z}_{t})+1$ (30) $$

These updated scores are used to guide the selection process in the next iteration such that the unmasking paths that result in the highest reward sequences have a greater chance of being selected and explored further.

Penalizing Invalid Peptides. To discourage the selection process from choosing unmasking steps that result in invalid or unsynthesizable peptide SMILES, we subtract a penalty score calculated as the fraction of invalid SMILES sequences rolled out from the expanded children of a parent node determined based on our peptide SMILES validity filter. Since properties are irrelevant if the peptide SMILES is invalid, we subtract the penalty scaled by a constant $c$ element-wise from the cumulative reward vectors. This penalty-adjusted reward vector is backpropagated to all parent nodes to avoid paths leading to high invalidity rates in the next iteration. After empirical experimentation, we determined that setting the constant $c=0.5$ consistently results in 100% peptide validity rate after only $\sim$20 iterations of the MCTS search.

$$ $\displaystyle\mathbf{W}(\mathbf{z}_{t})$ $\displaystyle\leftarrow\mathbf{W}(\mathbf{z}_{t})+\sum_{i=1}^{M}\mathbf{r}(\mathbf{z}_{s,i})-c\cdot\left(\frac{N_{\text{invalid}}}{M}\right)$ (31) $$

Output. The output after $N_{\text{iter}}$ iterations is the set $\mathcal{P}^{*}$ of Pareto-optimal sequences across the $K$ objectives. Our strategy simultaneously guides the unmasking process towards optimality across multiple objectives directly in the discrete state space while exploring the diverse space of peptide sequences using the trained unconditional MDLM generator. Furthermore, we generate an set of Pareto-optimal sequences from a single run through the MCTS-search algorithm which are non-dominated from the total of $N_{\text{iter}}\cdot M$ total sequences sampled across all iterations.

### 4.5 Evaluation

Peptide Validity Filter. Among the sequential representations of peptides including amino acid sequences, HELM , and SMILES , SMILES is the most intricate representation of peptide sequences. Although this enables the representation of non-natural amino acids, diverse side-chain, and backbone modifications, and cyclic peptides, it also means that the vast majority of SMILES strings are not synthesizable peptides. Therefore, we devised an algorithm that determines whether a SMILES string is a valid peptide, characterized by peptide bonds and central carbon atoms. The filter first checks if the SMILES sequence is a valid molecule using RDKit .

Then, we use regular expressions to detect bond patterns for peptide bonds, N-methylated peptide bonds, reversed peptide bonds, and ester bonds, along the sequence to split the sequence into several segments with a bond before and after each segment. The filter checks each segment for chemical modifications based on their bond type, including N-methylation (N-Me) and O-linked glycosylation. The remaining segment is matched to the corresponding natural or non-natural amino acid side chains (Algorithm [8](#alg8)). Our filter is capable of detecting a library of over 200 nAAs from SwissSidechain and can classify a peptide SMILES as cyclic or non-cyclic. The tool is freely available on HuggingFace: [https://huggingface.co/spaces/ChatterjeeLab/SMILES2PEPTIDE](https://huggingface.co/spaces/ChatterjeeLab/SMILES2PEPTIDE).

Generation Quality Metrics. To evaluate the generation quality of our unconditional MDLM, PepMDLM, and our MCTS-guided MDLM, PepTune, we leverage the Moses metrics, including validity, uniqueness, diversity, similarity to nearest neighbor (SNN), and novelty . Validity is the percentage of generated sequences that are valid peptide SMILES based on our peptide validity filter. Uniqueness is the fraction of distinct peptide SMILES sequences among the valid peptide SMILES. Diversity is calculated as one minus the average Tanimoto similarity between the Morgan fingerprints for every pair of generated sequences. Similarity to nearest neighbor (SNN) takes the maximum Tanimoto similarity score for each generated sequence with the sequences in the training dataset and averages across all generated sequences to measure the average similarity of a generated sequence with its most similar neighbor.

Due to the limit of memory and CPU time required to load all the training dataset of 11 million peptide SMILES, we chose to sample a subset of 1000 batches randomly ($\sim$100k sequences) for novelty and SNN calculation. To assess the novelty of generated sequences, we employed Shannon entropy to quantify the SMILES token randomness between 100 PepTune-generated and 100 PepMDLM-generated sequences and the same randomly sampled 1000 subsets from the training set. Then Kullback-Leibler (KL) divergence was also used to evaluate divergence across token distributions from the generated peptides compared to the training data. The equations for all metrics are provided in Appendix [C.5](#A3.SS5).

Docking. For valid generated peptide SMILES with non-dominated scores across objectives, we used Autodock Vina (v 1.1.2) for in silico docking of the peptide binders to their target proteins (Appendix [9](#A3.T9)) to confirm binding affinity. Targets were preprocessed with MGITools (v 1.5.7) and the conformations of the SMILES were optimized by ETKDG from RDKit . The final results were visualized in PyMol (v 3.1), where the residues in the protein targets with polar contacts to the peptide binder with distances closer than 3.5 Å are annotated.

## 5 Declarations

Acknowledgments. We thank the Duke Compute Cluster, Pratt School of Engineering IT department, and Mark III Systems, for providing database and hardware support that has contributed to the research reported within this manuscript. We thank Alexander Tong for reviewing the theoretical formulations of PepTune. We also thank Sophia Vincoff and Lauren Hong for their assistance with figure generation.

Author Contributions. S.T. devised and developed PepTune architecture and theoretical formulations, and trained and benchmarked generation, prediction, and sampling models. Y.Z. advised on model design and theoretical framework, trained classifier models, and performed molecular docking. S.T. drafted the manuscript and S.T. and Y.Z. designed the figures. P.C. conceived, designed, supervised, and directed the study, and reviewed and finalized the manuscript.

Data and Materials Availability. Our peptide filtering, analysis, and visualization tool, SMILES2PEPTIDE,
is freely available on HuggingFace: [https://huggingface.co/spaces/ChatterjeeLab/SMILES2PEPTIDE](https://huggingface.co/spaces/ChatterjeeLab/SMILES2PEPTIDE). The PepTune codebase is freely accessible to the academic community via a non-commercial license at [https://huggingface.co/ChatterjeeLab/PepTune](https://huggingface.co/ChatterjeeLab/PepTune).

Funding Statement. This research was supported by NIH grant R35GM155282 as well as a gift from the EndAxD Foundation to the lab of P.C.

Competing Interests. P.C. is a co-founder of Gameto, Inc. and UbiquiTx, Inc. and advises companies involved in peptide therapeutics development. P.C., S.T., and Y.Z. have and are currently filing patent applications related to this work. P.C.’s interests are reviewed and managed by Duke University in accordance with their conflict-of-interest policies.

## Appendix A Extended Background

### A.1 Continuous-Time Discrete Diffusion

Discrete diffusion models are a subset of diffusion models where the forward corruption and reverse denoising processes operate in the discrete latent space via categorical probability distributions for discrete variables.

We denote a token in a sequence from the dataset as a one-hot vector $\mathbf{x}_{0}^{(\ell)}\in\{0,1\}^{|\mathcal{V}|}$. The discrete-time forward diffusion process involves applying categorical noise to $\mathbf{x}_{0}$ at varying degrees based on a noise schedule $\sigma(t)$ for a total of $T$ time steps ranging from no noise at $t=0$ to maximum noise at $t=1$. To clearly distinguish each step, we denote the $n$th transition in the forward pass as the transition from $s(n)$ to $t(n)$, where $s(n)=\frac{n-1}{T}$ and $t(n)=\frac{n}{T}$. The marginal noise that transforms the sequence $\mathbf{z}_{s(n)}$ at time $s(n)$ to a progressively noisier sequence $\mathbf{z}_{t(n)}$ at the next time step $t(n)=s(n)+\frac{1}{T}$ is given by a $|\mathcal{V}|\times|\mathcal{V}|$ marginal transition matrix $\mathbf{Q}_{t|s}=\sigma(t)\mathbf{I}_{|\mathcal{V}|}+(1-\sigma(t))\mathbf{1}\mathbf{\pi}^{\top}$, where the $(i,j)$th entry denotes the probability of transitioning from token $i$ to token $j$ at each position in the sequence.

Therefore, the marginal categorical distribution of $\mathbf{z}_{t(n)}$ in the discrete-time forward-pass diffusion process can be derived as

$$ $\displaystyle q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)})$ $\displaystyle=\text{Cat}(\mathbf{z}_{t(n)};\mathbf{Q}_{t|s}^{\top}\mathbf{z}_{s(n)})$ $\displaystyle=\text{Cat}(\mathbf{z}_{t(n)};\sigma(t(n))\mathbf{z}_{s(n)}+(1-\sigma(t(n)))\mathbf{\pi})$ (32) $$

where $\sigma(t(n))$ the marginal probability of a single position in the sequence remaining the same token during the transition $s(n)\to t(n)$ and $\big{(}1-\sigma(t(n)))$ is the marginal probability of transitioning to a different token based on the token probability distribution $\pi\in\mathbf{\Delta}^{|\mathcal{V}|}$. For simplicity, we denote $\sigma(t(n))=\sigma(n)$.

The cumulative transition from time 0 0 to time $t(t)$ is denoted as the product of the marginals $\mathbf{Q}_{t}=\prod_{n=0}^{t}\mathbf{Q}_{t|s}$, which is the product of marginal transitions $s(n)\to t(n)$ for $n$ ranging from 0 to $t$ applied to the clean input sequence $\mathbf{x}_{0}$.

$$ $\displaystyle\mathbf{Q}_{t}=\left(\prod_{n=0}^{t}(1-\sigma(n))\right)\mathbf{I}+\left(1-\prod_{n=0}^{t}(1-\sigma(n))\right)\mathbf{1}\pi^{\top}$ (33) $$

For the continuous-time forward pass diffusion process where $T\to\infty$ and $\frac{1}{T}\to 0$, we can take the limit as $T\to\infty$ to derive an expression for the continuous-time transition probability, $\alpha_{t}$.

$$ $\displaystyle\lim_{T\to\infty}\prod_{n=0}^{t}(1-\sigma(n))$ $\displaystyle=\lim_{T\to\infty}\exp\left(\log\prod_{n=0}^{t}(1-\sigma(n))\right)$ $\displaystyle=\lim_{T\to\infty}\exp\left(\sum_{n=0}^{t}\log(1-\sigma(n))\right)$ $\displaystyle\approx\lim_{T\to\infty}\exp\left(\sum_{n=0}^{t}-\sigma(n)\right)$ $\displaystyle=\exp\left(-\int_{n=0}^{t}\sigma(n)dn\right)$ $\displaystyle=\exp\left(-\int_{s=0}^{t(t)}\sigma(s)ds\right)$ (34) $$

We have shown that the continuous-time forward transition probability from $t=0$ to $t=t(t)$ is $\alpha_{t}=\exp\left(-\int_{s=0}^{t(t)}\sigma(s)ds\right)=\exp(-\bar{\sigma}(t))$ where $\bar{\sigma}(t)=\int_{s=0}^{t(t)}\sigma(s)ds$. Letting $t=t(t)$, we can define the continuous-time cumulative transition matrix $\mathbf{Q}_{t}$ at the limit where $T\to\infty$ and the continuous-time distribution $q(\mathbf{z}_{t}|\mathbf{x}_{0})$ as

$$ $\displaystyle\mathbf{Q}_{t}$ $\displaystyle=\alpha_{t}\mathbf{I}+\alpha_{t}\mathbf{1}\pi^{\top}$ (35) $\displaystyle q(\mathbf{z}_{t}|\mathbf{x}_{0})$ $\displaystyle=\text{Cat}(\mathbf{z}_{t};\mathbf{Q}_{t}\mathbf{x}_{0})$ $\displaystyle=\text{Cat}(\mathbf{z}_{t};\alpha_{t}\mathbf{x}_{0}+(1-\alpha_{t})\mathbf{\pi})$ (36) $$

It follows that the marginal transition $\mathbf{Q}_{s|t}$ is the cumulative transition $\mathbf{Q}_{t}$ divided by all previous transition probabilities, denoted as $\mathbf{Q}_{s}=\prod_{r=0}^{s}\mathbf{Q}_{s|r}$, so $\alpha_{s|t}=\frac{\alpha_{t}}{\alpha_{s}}$.

Following Austin et al. and substituting the marginal and cumulative probability distributions, we derive the true reverse transition from $t\to s$ conditioned on a clean sequence $\mathbf{x}_{0}$ as

$$ $\displaystyle q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})$ $\displaystyle=\frac{q(\mathbf{z}_{t}|\mathbf{z}_{s},\mathbf{x}_{0})q(\mathbf{z}_{s}|\mathbf{x}_{0})}{q(\mathbf{z}_{t}|\mathbf{x}_{0})}$ $\displaystyle=\text{Cat}\left(\mathbf{z}_{s};\frac{\mathbf{Q}_{t|s}\mathbf{z}_{t}\odot\mathbf{Q}_{s}^{\top}\mathbf{x}_{0}}{\mathbf{z}_{t}^{\top}\mathbf{Q}_{t}^{\top}\mathbf{x}_{0}}\right)$ $\displaystyle=\text{Cat}\left(\mathbf{z}_{s};\frac{[\alpha_{t|s}\mathbf{z}_{t}+(1-\alpha_{t|s})\mathbf{1}\mathbf{\pi}^{\top}\mathbf{z}_{t}]\odot[\alpha_{s}\mathbf{x}_{0}+(1-\alpha_{s})\mathbf{\pi}]}{\alpha_{t}\mathbf{z}_{t}^{\top}\mathbf{x}_{0}+(1-\alpha_{t})\mathbf{z}_{t}^{\top}\mathbf{\pi}}\right)$ (37) $$

where the numerator is the element-wise product of $|\mathcal{V}|$-dimensional vectors representing the marginal probability distribution of sampling $\mathbf{z}_{t}$ given $\mathbf{z}_{s}$ and the cumulative probability distribution for $\mathbf{z}_{s}$ from the original clean sequence $\mathbf{x}_{0}$. The denominator is a scalar probability of the specific token $\mathbf{z}_{t}$ being drawn from the noisy probability distribution at time $t$.

### A.2 Continuous-Time Negative Evidence Lower Bound (NELBO)

The objective of denoising diffusion probabilistic models (DDPMs) is to iteratively sample slightly less noisy intermediate sequences $\mathbf{z}_{t}$ until obtaining a clean sequence $\mathbf{x}$ that has a high probability of being drawn from the data distribution $p(\mathbf{x}_{0})$. To train a model that accurately samples from $p(\mathbf{x}_{0})$, we maximize the Evidence Lower Bound (ELBO) of $\log p_{\theta}(\mathbf{x}_{0})$ which measures how accurately the model parameterized by $\theta$ generates true samples $\mathbf{x}_{0}$ given a corrupted sequence $\mathbf{z}_{T}$ by iterative sampling from the reverse posterior $p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})$. The ELBO is maximized when $p_{\theta}(\mathbf{x}_{0})=1$ and $\log(p_{\theta}(\mathbf{x}_{0}))=0$ for all sequences $\mathbf{x}_{0}$ in the dataset, which supports the objective of accurately generating sequences from the data distribution. To convert this into a loss minimization objective, we define the loss function as the negative ELBO (NELBO). First, we compute $\log p_{\theta}(\mathbf{x}_{0})$ by integrating over the joint probability of all possible paths of intermediate states from the noisy state $\mathbf{z}_{T}$ at $t=T$ to the clean sample $x_{0}$ at $t=0$, denoted by $p_{\theta}(\mathbf{x}_{0:T})$. Since our goal is to reverse the forward masking of the clean sample $x_{0}$ from all time steps, we introduce an encoder term $q(\mathbf{z}_{1:T}|\mathbf{x}_{0})$ denoting the combined distribution of obtaining any noisy sequence between times $t=1$ to $t=T$ from the clean sequence $\mathbf{x}_{0}$.

$$ $\displaystyle\log p_{\theta}(\mathbf{x}_{0})$ $\displaystyle=\log\int p_{\theta}(\mathbf{z}_{0:T})d\mathbf{z}_{1:T}$ $\displaystyle=\log\int q(\mathbf{z}_{1:T}|\mathbf{x}_{0})\left[\frac{p_{\theta}(\mathbf{z}_{0:T})}{q(\mathbf{z}_{1:T}|\mathbf{x}_{0})}\right]d\mathbf{z}_{1:T}$ $\displaystyle=\log\left(\mathbb{E}_{q(\mathbf{z}_{1:T}|\mathbf{x}_{0})}\left[\frac{p_{\theta}(\mathbf{z}_{0:T})}{q(\mathbf{z}_{1:T}|\mathbf{x}_{0})}\right]\right)$ (38) $$

where $\mathbf{z}_{0:T}$ includes $\mathbf{x}_{0}$.

Using Jenson’s inequality, we move the logarithm inside the expectation and reverse the sign to get the NELBO for $\log p_{\theta}(\mathbf{x}_{0})$. We split the terms associated with the forward noising process $q(\mathbf{z}_{1:T}|\mathbf{x}_{0})$ and the reverse denoising model $p_{\theta}(\mathbf{z}_{0:T})$ into reconstruction term, the prior term, and the intermediate diffusion term.

$$ $\displaystyle\mathcal{L}_{\text{NELBO}}$ $\displaystyle=\mathbb{E}_{q(\mathbf{z}_{1:T}|\mathbf{x}_{0})}\left[-\log\frac{p_{\theta}(\mathbf{z}_{0:T})}{q(\mathbf{z}_{1:T}|\mathbf{x}_{0})}\right]$ $\displaystyle=\mathbb{E}_{q(\mathbf{z}_{1:T}|\mathbf{x}_{0})}\left[-\log\frac{p_{\theta}(\mathbf{x}_{0}|\mathbf{z}_{t(1)})p_{\theta}(\mathbf{z}_{t(T)})\prod_{n=1}^{T-1}p_{\theta}(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)})}{q(\mathbf{z}_{t(T)}|\mathbf{z}_{t(T-1)})\prod_{n=1}^{T-1}q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)})}\right]$ $\displaystyle=\mathbb{E}_{q(\mathbf{z}_{1:T}|\mathbf{x}_{0})}\left[-\log p_{\theta}(\mathbf{x}_{0}|\mathbf{z}_{t(1)})-\log\frac{p_{\theta}(\mathbf{z}_{t(T)})}{q(\mathbf{z}_{t(T)}|\mathbf{z}_{s(T)})}-\log\frac{\prod_{t=1}^{T-1}p_{\theta}(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)})}{\prod_{n=1}^{T-1}q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)})}\right]$ $\displaystyle=\mathbb{E}_{q(\mathbf{z}_{1:T}|\mathbf{x}_{0})}\left[-\log p_{\theta}(\mathbf{x}_{0}|\mathbf{z}_{t(1)})-\log\frac{p_{\theta}(\mathbf{z}_{t(T)})}{q(\mathbf{z}_{t(T)}|\mathbf{z}_{s(T)})}-\sum_{n=1}^{T-1}\log\frac{p_{\theta}(\mathbf{z}_{s(n)}|\mathbf{z}_{s(n)})}{q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)})}\right]$ $$

$$ $\displaystyle=\mathbb{E}_{q(\mathbf{z}_{1:T}|\mathbf{x}_{0})}\bigg{[}-\log p_{\theta}(\mathbf{x}_{0}|\mathbf{z}_{t(1)})\bigg{]}+\mathbb{E}_{q(\mathbf{z}_{1:T}|\mathbf{x}_{0})}\left[-\log\frac{p_{\theta}(\mathbf{z}_{t(T)})}{q(\mathbf{z}_{t(T)}|\mathbf{z}_{s(T)})}\right]$ $\displaystyle\quad+\sum_{n=1}^{T-1}\mathbb{E}_{q(\mathbf{z}_{1:T}|\mathbf{x}_{0})}\bigg{[}-\log\frac{p_{\theta}(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)})}{q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)})}\bigg{]}$ $\displaystyle=\underbrace{\mathbb{E}_{q(\mathbf{z}_{t(1)}|\mathbf{x}_{0})}\bigg{[}-\log p_{\theta}(\mathbf{x}_{0}|\mathbf{z}_{t(1)})\bigg{]}}_{\text{reconstruction loss}}+\underbrace{\mathbb{E}_{q(\mathbf{z}_{t(T)},\mathbf{z}_{s(T)}|\mathbf{x}_{0})}\left[-\log\frac{p_{\theta}(\mathbf{z}_{t(T)})}{q(\mathbf{z}_{t(T)}|\mathbf{z}_{s(T)})}\right]}_{\text{prior loss}}$ $\displaystyle\quad+\underbrace{\sum_{n=1}^{T-1}\mathbb{E}_{q(\mathbf{z}_{s(n)},\mathbf{z}_{t(n)},\mathbf{z}_{t(n+1)}|\mathbf{x}_{0})}\bigg{[}-\log\frac{p_{\theta}(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)})}{q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)})}\bigg{]}}_{\text{diffusion loss}}$ (39) $$

Now, we can take the limit for each of the loss terms as $T\to\infty$ to derive the continuous-time MDLM objective.

Reconstruction Loss $\mathcal{L}_{\text{reconst}}$. The reconstruction loss evaluates the final step of the reverse diffusion process that denoises the sequence at time $t(1)$ to the clean sequence at time $t=0$. Since $t(0)=\frac{1}{T}$, the distribution that the sequence $\mathbf{z}_{t(1)}$ is drawn from in the forward pass diffusion is given by

$$ $\displaystyle p(\mathbf{z}_{t(1)}|\mathbf{x}_{0})=\text{Cat}(\mathbf{z}_{t(1)};\alpha_{t(1)}(\mathbf{x}_{0})\mathbf{x}_{0}+(1-\alpha_{t(1)}(\mathbf{x}_{0}))\mathbf{m})$ (40) $$

Since we have $\alpha_{t}(\mathbf{x}_{0})=1-t^{w}$ for $\mathbf{x}_{0}=\mathbf{b}$ and $\alpha_{t}(\mathbf{x}_{0})=1-t$ for $\mathbf{x}_{0}\neq\mathbf{b}$, we can write

$$ $\displaystyle\alpha_{t(1)}(\mathbf{x}_{0})\mathbf{x}_{0}+(1-\alpha_{t(1)}(\mathbf{x}_{0}))\mathbf{m}$ $\displaystyle=\begin{cases}\left(1-\frac{1}{T^{w}}\right)\mathbf{x}_{0}+\frac{1}{T^{w}}\mathbf{m}&\mathbf{x}_{0}=\mathbf{b}\\ \left(1-\frac{1}{T}\right)\mathbf{x}_{0}+\frac{1}{T}\mathbf{m}&\mathbf{x}_{0}\neq\mathbf{b}\end{cases}$ (41) $$

In the limit as $T\to\infty$, both cases converge to $\mathbf{x}_{0}$, so we have $\mathbf{z}_{t(1)}\sim\text{Cat}(\mathbf{z}_{t(1)};\mathbf{x}_{0})$ and $\mathbf{z}_{t(1)}=\mathbf{x}_{0}$. Since $q(\mathbf{z}_{t(1)}|\mathbf{x}_{0})=\mathbf{x}_{0}$ in the forward pass, by parameterizing the reverse posterior to copy-over unmasked tokens, we get $p_{\theta}(\mathbf{x}_{0}|\mathbf{z}_{t(1)})=\mathbf{x}_{0}$. Therefore, the reconstruction loss reduces to 0.

$$ $\displaystyle\mathbb{E}_{q(\mathbf{z}_{t(1)}|\mathbf{x}_{0})}\bigg{[}-\log p_{\theta}(\mathbf{x}_{0}|\mathbf{z}_{t(1)})\bigg{]}$ $\displaystyle=\mathbb{E}_{q(\mathbf{z}_{t(1)}|\mathbf{x}_{0})}\bigg{[}-\log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{0})\bigg{]}$ $\displaystyle=0$ $$

Prior Loss $\mathcal{L}_{\text{prior}}$. The prior loss measures the first reverse transition from the fully masked sequence $\mathbf{z}_{t(T)}$ to a slightly unmasked sequence $\mathbf{z}_{s(T)}$.

$$ $\displaystyle\mathbb{E}_{q(\mathbf{z}_{t(T)},\mathbf{z}_{s(T)}|\mathbf{x}_{0})}\left[-\log\frac{p(\mathbf{z}_{t(T)})}{q(\mathbf{z}_{t(T)}|\mathbf{z}_{s(T)})}\right]$ $\displaystyle=-\mathbb{E}_{q(\mathbf{z}_{s(T)}|\mathbf{x}_{0})}\underbrace{\mathbb{E}_{q(\mathbf{z}_{t(T)}|\mathbf{z}_{s(T)})}\left[\log\frac{p(\mathbf{z}_{t(T)})}{q(\mathbf{z}_{t(T)}|\mathbf{z}_{s(T)})}\right]}_{\text{KL Divergence}}$ $\displaystyle=-\mathbb{E}_{q(\mathbf{z}_{s(T)}|\mathbf{x}_{0})}\bigg{[}\text{KL}\bigg{(}q(\mathbf{z}_{t(T)}|\mathbf{z}_{s(T)})||p_{\theta}(\mathbf{z}_{t(T)})\bigg{)}\bigg{]}$ (42) $$

Since $t(T)=1$, we have $\alpha_{t(T)}(\mathbf{x}_{0})=1-1=0$. Therefore, we derive

$$ $\displaystyle q(\mathbf{z}_{t(T)}|\mathbf{x}_{0})$ $\displaystyle=\text{Cat}\big{(}\mathbf{z}_{t(T)}\;;\;\alpha_{t(T)}(\mathbf{x}_{0})\mathbf{x}_{0}+(1-\alpha_{t(T)}(\mathbf{x}_{0}))\mathbf{m}\big{)}$ $\displaystyle=\text{Cat}\big{(}\mathbf{z}_{t(T)}\;;\;0\mathbf{x}_{0}+(1-0)\mathbf{m}\big{)}$ $\displaystyle=\text{Cat}\big{(}\mathbf{z}_{t(T)};\mathbf{m}\big{)}$ (43) $$

Since all sequences are completely masked at time $T$, it follows that the marginal distribution $q(\mathbf{z}_{t(T)}|\mathbf{z}_{s(T)})=\text{Cat}\big{(}\mathbf{z}_{t(T)};\mathbf{m}\big{)}$ and the prior distribution $p_{\theta}(\mathbf{z}_{t(T)})=\text{Cat}\big{(}\mathbf{z}_{t(T)};\mathbf{m}\big{)}$, so the KL divergence reduces to 0.

Diffusion Loss $\mathcal{L}_{T}$. The diffusion loss measures the consistency of each predicted reverse transition with the forward marginal transition for all $T$ time steps.

$$ $\displaystyle\sum_{n=1}^{T-1}\mathbb{E}_{q(\mathbf{z}_{s(n)},\mathbf{z}_{t(n)},\mathbf{z}_{t(n+1)}|\mathbf{x}_{0})}$ $\displaystyle\bigg{[}-\log\frac{p_{\theta}(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)})}{q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)})}\bigg{]}$ (44) $\displaystyle=-\sum_{n=1}^{T-1}\mathbb{E}_{q(\mathbf{z}_{s(n)},\mathbf{z}_{t(n+1)}|\mathbf{x}_{0})}\underbrace{\mathbb{E}_{q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)})}\bigg{[}\log\frac{p_{\theta}(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)})}{q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)})}\bigg{]}}_{\text{KL divergence}}$ $\displaystyle=-\sum_{n=1}^{T-1}\mathbb{E}_{q(\mathbf{z}_{s(n)},\mathbf{z}_{t(n+1)}|\mathbf{x}_{0})}\bigg{[}\text{KL}\bigg{(}q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)})||p_{\theta}(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)})\bigg{)}\bigg{]}$ $$

Since the objective is to accurate predict $\mathbf{z}_{s(n)}$ given $\mathbf{z}_{t(n)}$, we cannot condition on the term $\mathbf{z}_{s(n)}$. Instead, we can condition $q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)})$ on $\mathbf{x}_{0}$ and use Baye’s theorem to derive

$$ $\displaystyle q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)},\mathbf{x}_{0})$ $\displaystyle=\frac{q(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)},\mathbf{x}_{0})q(\mathbf{z}_{t(n)}|\mathbf{x}_{0})}{q(\mathbf{z}_{s(n)}|\mathbf{x}_{0})}$ $$

Rearranging the terms we get an expression for the true reverse transition $q(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)},\mathbf{x}_{0})$ conditioned on the clean data $\mathbf{x}_{0}$. By minimizing the KL divergence between the learned reverse transition $p_{\theta}(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)})$ and $q(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)},\mathbf{x}_{0})$, we can rewrite the diffusion loss as

$$ $\displaystyle\mathcal{L}_{T}=\sum_{n=1}^{T-1}\mathbb{E}_{q(\mathbf{z}_{t(n)}|\mathbf{x}_{0})}\bigg{[}\text{KL}\bigg{(}q(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)},\mathbf{x}_{0})||p_{\theta}(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)})\bigg{)}\bigg{]}$ (45) $$

In Appendix [B.3](#A2.SS3), we derive the bond-dependent continuous-time NELBO loss from its general form above.

### A.3 Guided Diffusion Models

Guided diffusion aims to sample from the data distribution conditioned on some property $y$, $\mathbf{x}\sim q(\mathbf{x}_{0},y)$, such that $q(y|\mathbf{x})$ is maximized. Therefore, the marginal reverse transition aims to sample $\mathbf{z}_{s}$ from a distribution $q(\mathbf{z}_{s}|\mathbf{z}_{t},y)$ conditioned on the current sequence $\mathbf{z}_{t}$ and a property $y$. Using Baye’s theorem, we can decompose the guided conditional distribution as

$$ $\displaystyle q(\mathbf{z}_{s}|\mathbf{z}_{t},y)$ $\displaystyle=\frac{q(y|\mathbf{z}_{s},\mathbf{z}_{t})q(\mathbf{z}_{s}|\mathbf{z}_{t})}{q(y|\mathbf{z}_{t})}$ (46) $$

There are two strategies to generate samples from this conditional distribution: classifier-free and classifier-based guidance.

Classifier-Free Guidance.
Classifier-free guidance strategies aim to model the conditional distribution $q(\mathbf{z}_{s}|\mathbf{z}_{t},y)$ by directly training the diffusion model on a subset of the unconditional data with property $y$, such that after training, the model samples from a learned distribution $p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t},y)$. However, classifier-free guidance fails at tasks where quality annotated data is scarce, including peptide sequences. Furthermore, this strategy cannot scale to multiple objectives which would require data conditioned on more than one property.

Classifier-Based Guidance.
Classifier-based guidance trains an unconditional diffusion model $p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})$ and a classifier model $p_{\phi}(y|\mathbf{z}_{s})$ with learned parameters $\phi$ that generates a score measuring the probability that the intermediate sequence $\mathbf{z}_{s}$ has property $y$. By Bayes’ theorem, we can model the conditional distribution as

$$ $\displaystyle p_{\theta,\phi}(\mathbf{z}_{s}|\mathbf{z}_{t},y)=\frac{p_{\phi}(y|\mathbf{z}_{s})p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})}{p_{\phi}(y|\mathbf{z}_{t})}$ (47) $$

Since the model parameters implicitly learn the normalized distribution, we can drop the $p_{\phi}(y|\mathbf{z}_{t})$ term. Then, at each iteration, we update the parameters $\theta,\phi$ in the direction of the gradient of $\log p_{\theta,\phi}(\mathbf{z}_{s}|\mathbf{z}_{t},y)$ obtained as the sum of the gradients of the unconditional distribution $\log p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})$ and classifier probability $p_{\phi}(y|\mathbf{z}_{s})$ with respect to the sampled sequence $\mathbf{z}_{s}$.

$$ $\displaystyle\nabla_{\mathbf{z}_{s}}\log p_{\theta,\phi}(\mathbf{z}_{s}|\mathbf{z}_{t},y)=\nabla_{\mathbf{z}_{s}}\log p_{\phi}(y|\mathbf{z}_{s})+\nabla_{\mathbf{z}_{s}}\log p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})$ (48) $$

After joint training with the classifier and unconditional data distribution, we can sample from the learned conditional distribution $p_{\theta,\phi}(\mathbf{z}_{s}|\mathbf{z}_{t},y)$.

Unlike classifier-free guidance, classifier-based guidance does not require training a generative model from a conditioned dataset. However, the gradient-based strategy for classifier-based guidance is not directly applicable to discrete state spaces due to the lack of a defined gradient. To mimic gradient-based updates to each sampling step, Gruver et al. leveraged iterative guidance steps on continuous latent embeddings for each token before decoding back to a discrete sequence at each time step. However, projecting to and from the continuous and discrete spaces results in inconsistencies in the guided diffusion process, where optimized hidden embeddings do not always map to optimal tokens.

Guidance in the Discrete State Space. To directly guide the diffusion objective in the discrete space, we must compute the optimality of a single discrete reverse transition $\mathbf{z}_{s}$ against all other possible transitions to maximize the conditional probability $p(y|\mathbf{z}_{s},\mathbf{z}_{t})$. That is, we need to compute Equation ([46](#A1.E46)) with the denominator expanded to represent all possible transitions from $\mathbf{z}_{t}$.

$$ $\displaystyle p_{\theta,\phi}(\mathbf{z}_{s}|\mathbf{z}_{t},y)=\frac{p_{\phi}(y|\mathbf{z}_{s})p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})}{\sum_{\mathbf{z}^{\prime}_{s}}p_{\phi}(y|\mathbf{z}^{\prime}_{s})p_{\theta}(\mathbf{z}^{\prime}_{s}|\mathbf{z}_{t})}$ (49) $$

However, computing $p_{\phi}(y|\mathbf{z}^{\prime}_{s})$ for all the possible transitions from state $\mathbf{z}_{t}$ is computationally intractable. Previous work has bypassed this limitation by approximation. For continuous and differentiable classifier functions $p(y|\mathbf{x}):\mathbb{R}^{L\times|\mathcal{V}|}\to\mathbb{R}$, we can approximate the denominator using the first-order Taylor expansion given by

$$ $\displaystyle\log p_{\phi}(y|\mathbf{z}_{s},\mathbf{z}_{t})$ $\displaystyle\approx\log p_{\phi}(y|\mathbf{z}_{t})+(\mathbf{z}_{s}-\mathbf{z}_{t})^{\top}\nabla_{\mathbf{z}}\log p(y|\mathbf{z})|_{\mathbf{z}=\mathbf{z}_{t}}$ (50) $$

which approximates the likelihood of observing property $y$ at the slightly denoised state $\mathbf{z}_{s}=\mathbf{z}_{t-\frac{1}{T}}$ given the known log-probability of observing $y$ for state $\mathbf{z}_{t}$. This eliminates the need to explicitly sample $\mathbf{z}_{s}$ for all possible state transitions and compute $\log p_{\phi}(y|\mathbf{z}_{s},\mathbf{z}_{t})$.

Digress has used this approximation strategy for guided discrete diffusion on categorical graph generation. Furthermore, Nisonoff et al. uses the Taylor-approximated conditional distribution $\log p_{\phi}(y|\mathbf{z}_{s})$ to adjust the unconditional transition rates $R_{t}(\mathbf{z}_{t},\mathbf{z}_{s}|y)$ given the unconditional rates $R_{t}(\mathbf{z}_{t},\mathbf{z}_{s})$ for predictor-guidance of Continuous-Time Markov Chains (CTMCs) in the discrete state space.

$$ $\displaystyle R_{t}(\mathbf{z}_{t},\mathbf{z}_{s}|y)$ $\displaystyle=R_{t}(\mathbf{z}_{t},\mathbf{z}_{s})\frac{\log p_{\phi}(y|\mathbf{z}_{s},\mathbf{z}_{t})}{\log p_{\phi}(y|\mathbf{z}_{t})}$ (51) $$

where $R_{t}(\mathbf{z}_{t},\mathbf{z}_{s}|y)$ is the predictor-adjusted rate of transitioning from state $\mathbf{z}_{t}$ to state $\mathbf{z}_{s}$

However, this strategy fails to scale to multiple objective guidance since it would require computing the joint probability over $K$ objectives $p_{\phi}(y_{1},y_{2},\dots,y_{K}|\mathbf{z}_{s},\mathbf{z}_{t})$ for some $K>1$. If all properties are mutually independent, we can factorize the distribution and compute the estimated probability of each objective and take their product $p_{\phi}(y_{1},y_{2},\dots,y_{K}|\mathbf{z}_{s},\mathbf{z}_{t})=\prod_{k=1}^{K}p_{\phi}(y_{k}|\mathbf{z}_{s},\mathbf{z}_{t})$. For the majority of multi-objective tasks including therapeutic peptide generation, independence across properties is not a reasonable assumption, and computing the joint distribution is required. Moreover, for objectives that guide toward contradictory optimal rates or transitions, training a model conditioned on these objectives could prevent the model from generating optimal sequences for either objective. Given these limitations, there remains a gap for efficient classifier-based conditioning for discrete diffusion that is robust to multi-objective tasks, which we address in this work.

## Appendix B Theoretical Details

### B.1 Bond-Dependent Masking Schedule

From Equation ([34](#A1.E34)), we define the continuous-time forward masking probability $1-\alpha_{t}$ at time $t$ with $\alpha_{t}=\exp(-\bar{\sigma}(t))$ , where $\bar{\sigma}:[0,1]\to\mathbb{R}^{+}$ is the cumulative discrete-time masking schedule. Following Lou et al. , we apply a log-linear masking schedule $\bar{\sigma}(t)=-\log(1-t)$ for the forward diffusion process which has shown to result in the lowest variance in the NELBO loss . Therefore, the continuous-time probability of remaining unmasked at time $t$ is equal to $\alpha_{t}=\exp\big{(}-(-\log(1-t))\big{)}=1-t$ and the weight that scales the negative log loss (NLL) is given by $\frac{1}{t}$ by our derivation in Appendix [B.3](#A2.SS3).

For peptide-bond tokens, we alter the masking schedule such that peptide-bonds are masked at a slower rate at earlier time steps by defining a log-polynomial masking schedule $\bar{\sigma}(t)=-\log(1-t^{w})$, for some constant exponent $w>1$. Note that when $w=1$, the log-polynomial schedule reduces to the log-linear schedule. Therefore, the probability of remaining unmasked becomes $\alpha_{t}=\big{(}-(-\log(1-t^{w}))\big{)}=1-t^{w}$ and the weight that scales the negative log loss (NLL) is given by $\frac{w}{t}$ by our derivation in Appendix [B.3](#A2.SS3).

Since $t\in(0,1]$, the probability that a peptide-bond token remains unmasked at time $t$ is equal to $\alpha_{t}=1-t^{w}$ which is larger than the log-linear schedule for $w>1$. Conversely, the probability that a peptide-bond token is masked before $t$ is $1-\alpha_{t}=t^{w}$ which is smaller than the log-linear schedule for $w>1$. As $t\to 1$, $\alpha_{t}\to 0$ for both the log-linear and log-polynomial time schedules, which means that both peptide-bond and non-peptide bond tokens will have a high probability of being masked in later times in the forward pass diffusion process.

The NLL of the peptide-bond tokens is weighted more heavily than non-peptide bond tokens for $t$ close to 1. As $t\to 0$, the NLL weight approaches $\infty$ for all tokens. This biases the reverse diffusion process to unmask peptide bond tokens earlier since it was trained to minimize the loss associated with each unmasking step. As $t\to 0$, the large NLL weight ensures that the final unmasking steps during the reverse diffusion process result in an unmasked sequence that lies within the space of valid peptide SMILES.

Figure: Figure 15: Plots of bond-dependent masking schedules. (A) The probability of remaining unmasked during the continuous-time forward diffusion process over time $t$ given different values of $w$ as the exponent of the masking schedule $\alpha_{t}=1-t^{w}$. We use $w=1$ for non-peptide bond tokens and $w=3$ for peptide bond tokens, resulting in slower masking of peptide-bond tokens. (B) The weight of the negative log-loss for different exponents $w$ in the log-polynomial masking schedule. The weight of the loss is higher for larger $w$ in earlier time steps, which results in a higher penalty for inaccurate predictions of peptide bond tokens compared to other tokens.
Refer to caption: /html/2412.17780/assets/Figures/masking.png

### B.2 Derivation of Bond-Dependent Reverse Posterior

For a single token, the bond-dependent forward diffusion process is defined by the probability distribution $q(\mathbf{z}_{t}|\mathbf{x}_{0})$ which transforms the clean inputs to sequences with varying degrees of masking based on a probability distribution $\alpha_{t}(\mathbf{x}_{0})$. We define $\alpha_{t}(\mathbf{x}_{0}):\mathbb{R}^{|\mathcal{V}|}\to\mathbb{R}$ as a function that takes the clean token encoding $\mathbf{x}_{0}$ and outputs the probability of remaining unmasked at time $t$ depending on whether $\mathbf{x}_{0}$ encodes a peptide bond token.

$$ $\displaystyle q(\mathbf{z}_{t}|\mathbf{x}_{0})=\text{Cat}(\mathbf{z}_{t};(\alpha_{t}(\mathbf{x}_{0}))\mathbf{x}_{0}+(1-\alpha_{t}(\mathbf{x}_{0}))\mathbf{m})$ (52) $$

Then, the marginal forward transition from time $s(n)\to t(n)$ is defined as

$$ $\displaystyle q(\mathbf{z}_{t(n)}|\mathbf{z}_{s(n)})=\text{Cat}\left(\mathbf{z}_{t(n)};\left(\frac{\alpha_{t}(\mathbf{x}_{0})}{\alpha_{s}(\mathbf{x}_{0})}\right)\mathbf{x}_{0}+\left(\mathbf{1}-\frac{\alpha_{t}(\mathbf{x}_{0})}{\alpha_{s}(\mathbf{x}_{0})}\right)\mathbf{m}\right)$ (53) $$

In this work, we classify each token into one of two states: peptide-bond tokens and non-peptide-bond tokens, which represent amino acid side chains and modifications. We define a function that generates a mask with values of 1 indicating tokens containing or contained within a peptide bond, and 0 otherwise (Algorithm [7](#alg7)). Let $\mathbf{b}\in\mathbb{R}^{|\mathcal{V}|}$ denote a vector with ones at indices of peptide-bond tokens. For derivation purposes, we let $\mathbf{b}^{\top}\mathbf{x}_{0}^{(\ell)}=1$ and $\mathbf{x}_{0}^{(\ell)}=\mathbf{b}$ when a token at position $\ell$ is a peptide bond token. Note that $\mathbf{b}$ is defined differently depending on the context of the token in the full sequence which is handled by the BondMask function. Then, we have $\alpha_{t}(\mathbf{x}_{0})=\big{(}\mathbf{1}-\mathbf{b}^{\top}\mathbf{x}_{0}\big{)}(1-t)+\mathbf{b}^{\top}\mathbf{x}_{0}(1-t^{w})$ or equivalently we can write

$$ $\displaystyle\alpha_{t}(\mathbf{x}_{0})=\begin{cases}1-t^{w}&\mathbf{x}_{0}=\mathbf{b}\\ 1-t&\mathbf{x}_{0}\neq\mathbf{b}\end{cases}$ (54) $$

By Bayes’ rule, the general state-independent form of the true reverse posterior is given by

$$ $\displaystyle q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})$ $\displaystyle=\frac{q(\mathbf{z}_{t}|\mathbf{z}_{s})q(\mathbf{z}_{s}|\mathbf{x}_{0})}{q(\mathbf{z}_{t}|\mathbf{x}_{0})}$ $\displaystyle=\frac{\left[\left(\frac{\alpha_{t}}{\alpha_{s}}\right)\mathbf{z}_{s}^{\top}\mathbf{z}_{t}+\left(1-\frac{\alpha_{t}}{\alpha_{s}}\right)\mathbf{m}^{\top}\mathbf{z}_{t}\right]\left[\alpha_{s}\mathbf{x}_{0}^{\top}\mathbf{z}_{t}+\left(1-\alpha_{s}\right)\mathbf{m}^{\top}\mathbf{z}_{t}\right]}{\left[\alpha_{t}\mathbf{x}_{0}^{\top}\mathbf{z}_{t}+\left(1-\alpha_{t}\right)\mathbf{m}^{\top}\mathbf{z}_{t}\right]}$ (55) $$

With bond-dependent masking, the value of $\alpha_{t}(\mathbf{x}_{0})$ and $\alpha_{s}(\mathbf{x}_{0})$ depend on the state of $\mathbf{x}_{0}$, so the bond-dependent reverse posterior becomes

$$ $\displaystyle q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})$ $\displaystyle=\frac{\left[\left(\frac{\alpha_{t}(\mathbf{z}_{s})}{\alpha_{s}(\mathbf{z}_{s})}\right)\mathbf{z}_{s}^{\top}\mathbf{z}_{t}+\left(1-\frac{\alpha_{t}(\mathbf{z}_{s})}{\alpha_{s}(\mathbf{z}_{s})}\right)\mathbf{m}^{\top}\mathbf{z}_{t}\right]\left[\alpha_{s}(\mathbf{x}_{0})\mathbf{x}_{0}^{\top}\mathbf{z}_{t}+\left(1-\alpha_{s}(\mathbf{x}_{0})\right)\mathbf{m}^{\top}\mathbf{z}_{t}\right]}{\left[\alpha_{t}(\mathbf{x}_{0})\mathbf{x}_{0}^{\top}\mathbf{z}_{t}+\left(1-\alpha_{t}(\mathbf{x}_{0})\right)\mathbf{m}^{\top}\mathbf{z}_{t}\right]}$ (56) $$

When $\mathbf{z}_{t}=\mathbf{x}_{0}$, the true reverse posterior simplifies to

$$ $\displaystyle q(\mathbf{z}_{s}|\mathbf{z}_{t}=\mathbf{x}_{0},\mathbf{x}_{0})$ $\displaystyle=\frac{\left[\left(\frac{\alpha_{t}(\mathbf{z}_{s})}{\alpha_{s}(\mathbf{z}_{s})}\right)\mathbf{z}_{s}^{\top}\mathbf{x}_{0}+\left(1-\frac{\alpha_{t}(\mathbf{z}_{s})}{\alpha_{s}(\mathbf{z}_{s})}\right)\mathbf{m}^{\top}\mathbf{x}_{0}\right]\left[\alpha_{s}(\mathbf{x}_{0})\mathbf{x}_{0}^{\top}\mathbf{x}_{0}+\left(1-\alpha_{s}(\mathbf{x}_{0})\right)\mathbf{m}^{\top}\mathbf{x}_{0}\right]}{\left[\alpha_{t}(\mathbf{x}_{0})\mathbf{x}_{0}^{\top}\mathbf{x}_{0}+\left(1-\alpha_{t}(\mathbf{x}_{0})\right)\mathbf{m}^{\top}\mathbf{x}_{0}\right]}$ $\displaystyle=\frac{\left[\left(\frac{\alpha_{t}(\mathbf{z}_{s})}{\alpha_{s}(\mathbf{z}_{s})}\right)\mathbf{z}_{s}^{\top}\mathbf{x}_{0}\right]\left[\alpha_{s}(\mathbf{x}_{0})\right]}{\alpha_{t}(\mathbf{x}_{0})}$ (57) $$

When $\mathbf{z}_{s}\neq\mathbf{x}_{0}$, $\mathbf{z}_{s}^{\top}\mathbf{x}_{0}=0$ so $q(\mathbf{z}_{s}\neq\mathbf{x}_{0}|\mathbf{z}_{t}=\mathbf{x}_{0},\mathbf{x}_{0})=0$. When $\mathbf{z}_{s}=\mathbf{x}_{0}$, we have

$$ $\displaystyle q(\mathbf{z}_{s}=\mathbf{x}_{0}|\mathbf{z}_{t}=\mathbf{x}_{0},\mathbf{x}_{0})$ $\displaystyle=\frac{\left[\left(\frac{\alpha_{t}(\mathbf{x}_{0})}{\alpha_{s}(\mathbf{x}_{0})}\right)\mathbf{x}_{0}^{\top}\mathbf{x}_{0}\right]\left[\alpha_{s}(\mathbf{x}_{0})\right]}{\alpha_{t}(\mathbf{x}_{0})}$ $\displaystyle=\left(\frac{\alpha_{t}(\mathbf{x}_{0})}{\alpha_{s}(\mathbf{x}_{0})}\right)\left(\frac{\alpha_{s}(\mathbf{x}_{0})}{\alpha_{t}(\mathbf{x}_{0})}\right)$ $\displaystyle=1$ $$

which means that $\mathbf{z}_{t}$ remains unchanged after unmasking. This supports the carry-over unmasking scheme which explicitly sets the probability of changing an unmasked token equal to $-\infty$.

In the forward diffusion process, a token either remains unchanged or is masked, so the only other case we need to consider is $\mathbf{z}_{t}=\mathbf{m}$. Since the masking schedule differs only when the ground truth token is a peptide bond token, or $\mathbf{x}_{0}=\mathbf{b}$, we can consider two cases: first, when $\mathbf{x}_{0}=\mathbf{b}$ and second, when $\mathbf{x}_{0}\neq\mathbf{b}$.

Case 1. Consider the case when $\mathbf{x}_{0}=\mathbf{b}$ or the ground truth token $\mathbf{x}_{0}$ is a peptide-bond token. From our modified masking schedule, we have $\alpha_{t}(\mathbf{b})=1-t^{w}$. Therefore, we can write the probability distribution for unmasking a peptide-bond token as

$$ $\displaystyle q(\mathbf{z}_{s}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0}=\mathbf{b})$ $\displaystyle=\frac{\left[\left(\frac{\alpha_{t}(\mathbf{z}_{s})}{\alpha_{s}(\mathbf{z}_{s})}\right)\mathbf{z}_{s}^{\top}\mathbf{m}+\left(1-\frac{\alpha_{t}(\mathbf{z}_{s})}{\alpha_{s}(\mathbf{z}_{s})}\right)\mathbf{m}^{\top}\mathbf{m}\right]\left[\alpha_{s}(\mathbf{b})\mathbf{b}^{\top}\mathbf{z}_{s}+\left(1-\alpha_{s}(\mathbf{b})\right)\mathbf{m}^{\top}\mathbf{z}_{s}\right]}{\left[\alpha_{t}(\mathbf{b})\mathbf{b}^{\top}\mathbf{m}+\left(1-\alpha_{t}(\mathbf{b})\right)\mathbf{m}^{\top}\mathbf{m}\right]}$ $\displaystyle=\frac{\left[\left(\frac{\alpha_{t}(\mathbf{z}_{s})}{\alpha_{s}(\mathbf{z}_{s})}\right)\mathbf{z}_{s}^{\top}\mathbf{m}+\left(1-\frac{\alpha_{t}(\mathbf{z}_{s})}{\alpha_{s}(\mathbf{z}_{s})}\right)\right]\left[\alpha_{s}(\mathbf{b})\mathbf{b}^{\top}\mathbf{z}_{s}+\left(1-\alpha_{s}(\mathbf{b})\right)\mathbf{m}^{\top}\mathbf{z}_{s}\right]}{\left(1-\alpha_{t}(\mathbf{b})\right)}$ (58) $$

The probability of transitioning from a masked state to a peptide-bond token is simplified to

$$ $\displaystyle q(\mathbf{z}_{s}=\mathbf{b}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0}=\mathbf{b})$ $\displaystyle=\frac{\left[\left(\frac{\alpha_{t}(\mathbf{b})}{\alpha_{s}(\mathbf{b})}\right)\mathbf{b}^{\top}\mathbf{m}+\left(1-\frac{\alpha_{t}(\mathbf{b})}{\alpha_{s}(\mathbf{b})}\right)\right]\left[\alpha_{s}(\mathbf{b})\mathbf{b}^{\top}\mathbf{b}+\left(1-\alpha_{s}(\mathbf{b})\right)\mathbf{m}^{\top}\mathbf{b}\right]}{\left(1-\alpha_{t}(\mathbf{b})\right)}$ $\displaystyle=\frac{\left(1-\frac{1-t^{w}}{1-s^{w}}\right)(1-s^{w})}{\left(1-(1-t^{w})\right)}$ $\displaystyle=\frac{\left(\frac{1-s^{w}-1+t^{w}}{1-s^{w}}\right)(1-s^{w})}{t^{w}}$ $\displaystyle=\frac{t^{w}-s^{w}}{t^{w}}$ $\displaystyle=1-\frac{s^{w}}{t^{w}}$ (59) $$

The probability of remaining in a masked state is

$$ $\displaystyle q(\mathbf{z}_{s}=\mathbf{m}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0}=\mathbf{b})$ $\displaystyle=\frac{\left[\left(\frac{\alpha_{t}(\mathbf{m})}{\alpha_{s}(\mathbf{m})}\right)\mathbf{m}^{\top}\mathbf{m}+\left(1-\frac{\alpha_{t}(\mathbf{m})}{\alpha_{s}(\mathbf{m})}\right)\right]\left[\alpha_{s}(\mathbf{b})\mathbf{b}^{\top}\mathbf{m}+\left(1-\alpha_{s}(\mathbf{b})\right)\mathbf{m}^{\top}\mathbf{m}\right]}{\left(1-\alpha_{t}(\mathbf{b})\right)}$ $\displaystyle=\frac{\left[\left(\frac{\alpha_{t}(\mathbf{m})}{\alpha_{s}(\mathbf{m})}\right)+\left(1-\frac{\alpha_{t}(\mathbf{m})}{\alpha_{s}(\mathbf{m})}\right)\right]\left(1-\alpha_{s}(\mathbf{b})\right)}{\left(1-\alpha_{t}(\mathbf{b})\right)}$ $\displaystyle=\frac{1-\alpha_{s}(\mathbf{b})}{1-\alpha_{t}(\mathbf{m})}$ $\displaystyle=\frac{1-(1-s^{w})}{1-(1-t^{w})}$ $\displaystyle=\frac{s^{w}}{t^{w}}$ (60) $$

which aligns with the constraint that $\mathbf{z}_{t}\in\{\mathbf{m},\mathbf{x}_{0}\}$ in the forward diffusion process.

Case 2: Consider the case when $\mathbf{x}_{0}\neq\mathbf{b}$ or the ground truth token $\mathbf{x}_{0}$ is not a peptide-bond token. From the baseline log-linear masking schedule, we have $\vec{\alpha}_{t}^{\top}\mathbf{x}_{0}=1-t$. Therefore, we can write the probability distribution for unmasking a peptide-bond token as

$$ $\displaystyle q(\mathbf{z}_{s}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0}\neq\mathbf{b})$ $\displaystyle=\frac{\left[\left(\frac{\alpha_{t}(\mathbf{z}_{s})}{\alpha_{s}(\mathbf{z}_{s})}\right)\mathbf{z}_{s}^{\top}\mathbf{m}+\left(1-\frac{\alpha_{t}(\mathbf{z}_{s})}{\alpha_{s}(\mathbf{z}_{s})}\right)\mathbf{m}^{\top}\mathbf{m}\right]\left[\alpha_{s}(\mathbf{x}_{0})\mathbf{x}_{0}^{\top}\mathbf{m}+\left(1-\alpha_{s}(\mathbf{x}_{0})\right)\mathbf{m}^{\top}\mathbf{m}\right]}{\left[\alpha_{t}(\mathbf{x}_{0})\mathbf{x}_{0}^{\top}\mathbf{m}+\left(1-\alpha_{t}(\mathbf{x}_{0})\right)\mathbf{m}^{\top}\mathbf{m}\right]}$ $\displaystyle=\frac{\left[\left(\frac{\alpha_{t}(\mathbf{z}_{s})}{\alpha_{s}(\mathbf{z}_{s})}\right)\mathbf{z}_{s}^{\top}\mathbf{m}+\left(1-\frac{\alpha_{t}(\mathbf{z}_{s})}{\alpha_{s}(\mathbf{z}_{s})}\right)\right]\left[\alpha_{s}(\mathbf{x}_{0})\mathbf{x}_{0}^{\top}\mathbf{m}+\left(1-\alpha_{s}(\mathbf{x}_{0})\right)\right]}{\left[\alpha_{t}(\mathbf{x}_{0})\mathbf{x}_{0}^{\top}\mathbf{m}+\left(1-\alpha_{t}(\mathbf{x}_{0})\right)\right]}$ (61) $$

With similar steps to Case 1, the probability of transitioning from a masked state to a non-peptide-bond token is given by

$$ $\displaystyle q(\mathbf{z}_{s}=\mathbf{x}_{0}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0}\neq\mathbf{b})$ $\displaystyle=\frac{\left(1-\frac{\alpha_{t}(\mathbf{x}_{0})}{\alpha_{s}(\mathbf{x}_{0})}\right)\left(1-\alpha_{s}(\mathbf{x}_{0})\right)}{\left(1-\alpha_{t}(\mathbf{x}_{0})\right)}$ $\displaystyle=\frac{\left(1-\frac{1-t}{1-s}\right)\left(1-(1-s)\right)}{\left(1-(1-t)\right)}$ $\displaystyle=\frac{t-s}{t}$ $\displaystyle=1-\frac{s}{t}$ (62) $$

It follows that the probability of remaining in a masked state in the reverse process is

$$ $\displaystyle q(\mathbf{z}_{s}=\mathbf{m}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0}\neq\mathbf{b})$ $\displaystyle=\frac{s}{t}$ (63) $$

This demonstrates that the probability of remaining in a masked state when $\mathbf{x}_{0}=\mathbf{b}$ is smaller than when $\mathbf{x}_{0}\neq\mathbf{b}$ since taking the exponent of a fraction results in a smaller value. So we have $\frac{s^{w}}{t^{w}}<\frac{s}{t}$ for $w>1$.

Combining Equations ([60](#A2.E60)) and ([63](#A2.E63)) we get the following distribution for the case when $\mathbf{z}_{t}=\mathbf{m}$ and $\mathbf{z}_{s}=\mathbf{m}$

$$ $\displaystyle q(\mathbf{z}_{s}=\mathbf{m}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0})$ $\displaystyle=\left(\frac{s^{w}}{t^{w}}-\frac{s}{t}\right)\mathbf{b}^{\top}\mathbf{x}_{0}+\frac{s}{t}$ $\displaystyle=\left(\frac{s^{w}}{t^{w}}\mathbf{b}-\frac{s}{t}\mathbf{b}+\frac{s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{0}$ $\displaystyle=\left(\left(\frac{s^{w}}{t^{w}}-\frac{s}{t}\right)\mathbf{b}+\frac{s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{0}$ (64) $$

Similarly, combining ([59](#A2.E59)) and ([62](#A2.E62)) we get the following distribution for the case when $\mathbf{z}_{t}=\mathbf{m}$ and $\mathbf{z}_{s}\neq\mathbf{m}$ or equivalently $\mathbf{z}_{s}=\mathbf{x}_{0}$.

$$ $\displaystyle q(\mathbf{z}_{s}=\mathbf{x}_{0}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0})$ $\displaystyle=\left(\frac{s}{t}-\frac{s^{w}}{t^{w}}\right)\mathbf{b}^{\top}\mathbf{x}_{0}+\left(1-\frac{s}{t}\right)$ $\displaystyle=\left(\frac{s}{t}\mathbf{b}-\frac{s^{w}}{t^{w}}\mathbf{b}+\mathbf{1}-\frac{s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{0}$ $\displaystyle=\left(\left(\frac{s}{t}-\frac{s^{w}}{t^{w}}\right)\mathbf{b}+\frac{t-s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{0}$ (65) $$

Now, we can write the true reverse posterior as

$$ $\displaystyle q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})=\begin{cases}\left(\left(\frac{s}{t}-\frac{s^{w}}{t^{w}}\right)\mathbf{b}+\frac{t-s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{0}\mathbf{x}_{0}+\left(\left(\frac{s^{w}}{t^{w}}-\frac{s}{t}\right)\mathbf{b}+\frac{s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{0}\mathbf{m}&\mathbf{z}_{t}=\mathbf{m}\\ \mathbf{z}_{t}&\mathbf{z}_{t}\neq\mathbf{m}\end{cases}$ (66) $$

Therefore, we get the following expression for the parameterized reverse posterior

$$ $\displaystyle p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})=\begin{cases}\left(\left(\frac{s}{t}-\frac{s^{w}}{t^{w}}\right)\mathbf{b}+\frac{t-s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\mathbf{z}_{s}+\left(\left(\frac{s^{w}}{t^{w}}-\frac{s}{t}\right)\mathbf{b}+\frac{s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\mathbf{m}&\mathbf{z}_{t}=\mathbf{m}\\ \mathbf{z}_{t}&\mathbf{z}_{t}\neq\mathbf{m}\end{cases}$ (67) $$

### B.3 Derivation of Bond-Dependent NELBO Loss

The diffusion objective in its general form is given by

$$ $\displaystyle\mathcal{L}_{\text{NELBO}}$ $\displaystyle=\sum_{n=1}^{T-1}\mathbb{E}_{q(\mathbf{z}_{t(n)}|\mathbf{x}_{0})}\bigg{[}\text{KL}\bigg{(}q(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)},\mathbf{x}_{0})||p_{\theta}(\mathbf{z}_{s(n)}|\mathbf{z}_{t(n)})\bigg{)}\bigg{]}$ $\displaystyle=\mathbb{E}_{t\in\{\frac{1}{T},\frac{2}{T},\dots,1\}}\mathbb{E}_{q(\mathbf{z}_{t}|\mathbf{x}_{0})}\bigg{[}T\cdot\text{KL}\bigg{(}q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})\big{|}\big{|}p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})\bigg{)}\bigg{]}$ (68) $$

First, we will derive an expression for the bond-dependent KL-divergence, which measures the difference between the learned reverse posterior $q\big{(}\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\big{)}$ and the true reverse posterior $q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})$ conditioned on the training distribution $\mathbf{x}_{0}$.

$$ $\displaystyle\text{KL}(q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})||p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t}))$ $\displaystyle=\sum_{\mathbf{z}_{s}=\mathbf{e}_{k}}q(\mathbf{z}_{s}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0})\log\frac{q(\mathbf{z}_{s}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0})}{p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t}=\mathbf{m})}$ $\displaystyle=\sum_{\mathbf{z}_{s}\in\{\mathbf{x}_{0},\mathbf{m}\}}q(\mathbf{z}_{s}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0})\log\frac{q(\mathbf{z}_{s}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0})}{p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t}=\mathbf{m})}$ $\displaystyle=q(\mathbf{z}_{s}=\mathbf{x}_{0}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0})\log\frac{q(\mathbf{z}_{s}=\mathbf{x}_{0}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0})}{p_{\theta}(\mathbf{z}_{s}=\mathbf{x}_{0}|\mathbf{z}_{t}=\mathbf{m})}$ $\displaystyle+q(\mathbf{z}_{s}=\mathbf{m}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0})\log\frac{q(\mathbf{z}_{s}=\mathbf{m}|\mathbf{z}_{t}=\mathbf{m},\mathbf{x}_{0})}{p_{\theta}(\mathbf{z}_{s}=\mathbf{m}|\mathbf{z}_{t}=\mathbf{m})}$ $\displaystyle=\left(\left(\frac{s}{t}-\frac{s^{w}}{t^{w}}\right)\mathbf{b}+\frac{t-s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{0}\log\frac{\left(\left(\frac{s}{t}-\frac{s^{w}}{t^{w}}\right)\mathbf{b}+\frac{t-s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{0}}{\left(\left(\frac{s}{t}-\frac{s^{w}}{t^{w}}\right)\mathbf{b}+\frac{t-s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{\theta}(\mathbf{z}_{t},t)}$ $\displaystyle+\left(\left(\frac{s^{w}}{t^{w}}-\frac{s}{t}\right)\mathbf{b}+\frac{s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{0}\log\frac{\left(\left(\frac{s^{w}}{t^{w}}-\frac{s}{t}\right)\mathbf{b}+\frac{s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{0}}{\left(\left(\frac{s^{w}}{t^{w}}-\frac{s}{t}\right)\mathbf{b}+\frac{s}{t}\mathbf{1}\right)^{\top}\mathbf{x}_{\theta}(\mathbf{z}_{t},t)}$ (69) $$

In the case where the true token $\mathbf{x}_{0}=\mathbf{b}$, we can simplify to

$$ $\displaystyle\text{KL}(q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})||p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t}))$ $\displaystyle=\left(\frac{s}{t}-\frac{s^{w}}{t^{w}}+1-\frac{s}{t}\right)\log\frac{\left(\frac{s^{w}}{t^{w}}-\frac{s}{t}+\frac{s}{t}\right)\mathbf{x}_{0}^{\top}\mathbf{x}_{0}}{\left(\frac{s^{w}}{t^{w}}-\frac{s}{t}+\frac{s}{t}\right)\mathbf{x}_{0}^{\top}\mathbf{x}_{\theta}(\mathbf{z}_{t},t)}$ $\displaystyle=-\left(1-\frac{s^{w}}{t^{w}}\right)\log\big{(}\mathbf{x}_{0}^{\top}\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\big{)}$ $\displaystyle=-\left(\frac{t^{w}-s^{w}}{t^{w}}\right)\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle$ (70) $$

Substituting $s=t-\frac{1}{T}$, we can simplify $s^{w}$ to

$$ $\displaystyle s^{w}$ $\displaystyle=\left(t-\frac{1}{T}\right)^{w}$ $\displaystyle=\left[t\left(1-\frac{1}{tT}\right)\right]^{w}$ $\displaystyle=t^{w}\left(1-\frac{1}{tT}\right)^{w}]$ $\displaystyle=t^{w}\left(1-\frac{w}{tT}+o\left(\frac{1}{T^{2}}\right)\right)$ $\displaystyle=t^{w}-\frac{wt^{w-1}}{T}+t^{w}o\left(\frac{1}{T^{2}}\right)$ (71) $$

where $o\left(\frac{1}{T^{2}}\right)$ denotes higher order terms that grow slower than $\frac{1}{T^{2}}$.

Now, we can write

$$ $\displaystyle\text{KL}(q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})||p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t}))$ $\displaystyle=-\left(\frac{t^{w}-\left(t^{w}-\frac{wt^{w-1}}{T}+t^{w}o\left(\frac{1}{T^{2}}\right)\right)}{t^{w}}\right)\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle$ $\displaystyle=-\left(\frac{\frac{wt^{w-1}}{T}-t^{w}o\left(\frac{1}{T^{2}}\right)}{t^{w}}\right)\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle$ $\displaystyle=-\left(\frac{w}{tT}-o\left(\frac{1}{T^{2}}\right)\right)\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle$ $$

In the case where the true token $\mathbf{x}_{0}\neq\mathbf{b}$, we can simplify to

$$ $\displaystyle\text{KL}(q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})||p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t}))$ $\displaystyle=\left(1-\frac{s}{t}\right)\log\frac{\left(1-\frac{s}{t}\right)\mathbf{x}_{0}^{\top}\mathbf{x}_{0}}{\left(1-\frac{s}{t}\right)\mathbf{x}_{0}^{\top}\mathbf{x}_{\theta}(\mathbf{z}_{t},t)}$ $\displaystyle=-\left(1-\frac{s}{t}\right)\log\big{(}\mathbf{x}_{0}^{\top}\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\big{)}$ $\displaystyle=-\left(\frac{t-s}{t}\right)\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle$ (73) $$

Similarly, substituting $s=t-\frac{1}{T}$, we have

$$ $\displaystyle\text{KL}(q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})||p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t}))$ $\displaystyle=-\left(\frac{t-\left(t-\frac{1}{T}\right)}{t}\right)\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle$ $\displaystyle=-\frac{1}{tT}\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle$ (74) $$

Now, we can combine the two cases using the indicator functions $\mathbf{1}[\mathbf{x}_{0}=\mathbf{b}]$ that evaluates to 1 when $\mathbf{x}_{0}=\mathbf{b}$ and 0 otherwise and $\mathbf{1}[\mathbf{x}_{0}\neq\mathbf{b}]$ that evaluates to 1 when $\mathbf{x}_{0}\neq\mathbf{b}$ and 0 otherwise. Since this definition of KL divergence is only applicable when $\mathbf{z}_{t}=\mathbf{m}$, we have

$$ $\displaystyle\text{KL}(q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})||p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t}))$ $\displaystyle=\bigg{[}-\mathbf{1}[\mathbf{x}_{0}=\mathbf{b}]\left(\frac{w}{tT}-o\left(\frac{1}{T^{2}}\right)\right)\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle-\mathbf{1}[\mathbf{x}_{0}\neq\mathbf{b}]\frac{1}{tT}\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle\bigg{]}$ (75) $$

Substituting this back into the equation for the discrete-time diffusion loss, we get

$$ $\displaystyle\mathcal{L}_{\text{NELBO}}=\mathbb{E}_{t\in\{\frac{1}{T},\frac{2}{T},\dots,1\}}\mathbb{E}_{q(\mathbf{z}_{t}|\mathbf{x}_{0})}\bigg{[}T\cdot\text{KL}\bigg{(}q(\mathbf{z}_{s}|\mathbf{z}_{t},\mathbf{x}_{0})\big{|}\big{|}p_{\theta}(\mathbf{z}_{s}|\mathbf{z}_{t})\bigg{)}\bigg{]}$ $\displaystyle=\mathbb{E}_{t\in\{\frac{1}{T},\frac{2}{T},\dots,1\}}\mathbb{E}_{q(\mathbf{z}_{t}|\mathbf{x}_{0})}T\cdot\bigg{[}-\mathbf{1}[\mathbf{x}_{0}=\mathbf{b}]\left(\frac{w}{tT}-o\left(\frac{1}{T^{2}}\right)\right)\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle$ $\displaystyle\hskip 30.0pt-\mathbf{1}[\mathbf{x}_{0}\neq\mathbf{b}]\frac{1}{tT}\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle\bigg{]}$ $\displaystyle=\mathbb{E}_{t\in\{\frac{1}{T},\frac{2}{T},\dots,1\}}\mathbb{E}_{q(\mathbf{z}_{t}|\mathbf{x}_{0})}\bigg{[}-\mathbf{1}[\mathbf{x}_{0}=\mathbf{b}]\left(\frac{wT}{tT}-To\left(\frac{1}{T^{2}}\right)\right)\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle$ $\displaystyle\hskip 30.0pt-\mathbf{1}[\mathbf{x}_{0}\neq\mathbf{b}]\frac{T}{tT}\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle\bigg{]}$ $\displaystyle=\mathbb{E}_{t\in\{\frac{1}{T},\frac{2}{T},\dots,1\}}\mathbb{E}_{q(\mathbf{z}_{t}|\mathbf{x}_{0})}\bigg{[}-\mathbf{1}[\mathbf{x}_{0}=\mathbf{b}]\left(\frac{w}{t}-To\left(\frac{1}{T^{2}}\right)\right)\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle$ $\displaystyle\hskip 30.0pt-\mathbf{1}[\mathbf{x}_{0}\neq\mathbf{b}]\frac{1}{t}\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle\bigg{]}$ (76) $$

Finally, taking the limit as $T\to\infty$, the higher-order term $\lim_{T\to\infty}To\left(\frac{1}{T^{2}}\right)=0$ and we get

$$ $\displaystyle\mathcal{L}^{\infty}_{\text{NELBO}}=\lim_{T\to\infty}\mathcal{L}_{\text{NELBO}}$ $\displaystyle=\lim_{T\to\infty}\mathbb{E}_{t\in\{\frac{1}{T},\frac{2}{T},\dots,1\}}\mathbb{E}_{q(\mathbf{x}_{t}|\mathbf{x}_{0})}\bigg{[}-\mathbf{1}[\mathbf{x}_{0}=\mathbf{b}]\left(\frac{w}{t}-To\left(\frac{1}{T^{2}}\right)\right)\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle$ $\displaystyle\hskip 30.0pt-\mathbf{1}[\mathbf{x}_{0}\neq\mathbf{b}]\frac{1}{t}\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle\bigg{]}$ $\displaystyle=\mathbb{E}_{t\sim\mathcal{U}(0,1]}\mathbb{E}_{q(\mathbf{x}_{t}|\mathbf{x}_{0})}\bigg{[}-\mathbf{1}[\mathbf{x}_{0}=\mathbf{b}]\frac{w}{t}\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle$ $\displaystyle\hskip 30.0pt-\mathbf{1}[\mathbf{x}_{0}\neq\mathbf{b}]\frac{1}{t}\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle\bigg{]}$ (77) $$

which is the continuous-time NELBO loss for a single token. Therefore, the loss across a sequence of $L$ tokens denoted as $\mathbf{x}^{(\ell)}_{0}$, we have

$$ $\displaystyle\mathcal{L}^{\infty}_{\text{NELBO}}=\mathbb{E}_{t\sim\mathcal{U}(0,1]}\mathbb{E}_{q(\mathbf{x}_{t}|\mathbf{x}_{0})}\bigg{[}-\sum_{\ell:\mathbf{x}^{(\ell)}_{0}=\mathbf{b}}\frac{w}{t}\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle-\sum_{\ell:\mathbf{x}^{(\ell)}_{0}\neq\mathbf{b}}\frac{1}{t}\log\langle\mathbf{x}_{0},\mathbf{x}_{\theta}(\mathbf{z}_{t},t)\rangle\bigg{]}$ (78) $$

which proves the loss defined in ([12](#S4.E12)).

### B.4 Gradient Flow of Invalid Loss

In this section, we show that the penalty for invalid token samples through the argmax function on predicted logits can be effectively backpropagated through the model parameters via our softmax scaling strategy. Here, we will denote the predicted probability for the token $k=\arg\max_{j}\big{(}\mathbf{x}^{(\ell)}_{\theta}(\mathbf{z}_{t},t)\big{)}$ with the highest probability as $x_{\theta,k}^{(\ell)}$ and all remaining token probabilities as $x_{\theta,j}^{(\ell)}$ for $j=[1\dots|\mathcal{V}|]$.

First, we define the softmax function as

$$ $\displaystyle\text{SM}\big{(}x^{(\ell)}_{\theta,k}\big{)}=\frac{\exp(x_{\theta,k}^{(\ell)})}{\sum_{j=1}^{|\mathcal{V}|}\exp(x_{\theta,j}^{(\ell)})}$ (79) $$

The partial derivative of the softmax probability $x^{j}_{\theta}$ for every token $j$ is given by equation

$$ $\displaystyle\frac{\partial}{\partial x_{\theta,j}^{(\ell)}}\left(\frac{\exp(x_{\theta,k}^{(\ell)})}{\sum_{j=1}^{|\mathcal{V}|}\exp(x_{\theta,j}^{(\ell)})}\right)=\frac{\left(\frac{\partial}{\partial x_{\theta,j}^{(\ell)}}\exp(x_{\theta,k}^{(\ell)})\right)\left(\sum_{j=1}^{|\mathcal{V}|}\exp(x_{\theta,j}^{(\ell)})\right)-\left(\frac{\partial}{\partial x_{\theta,j}^{(\ell)}}\sum_{j=1}^{|\mathcal{V}|}\exp(x_{\theta,j}^{(\ell)})\right)\left(\exp(x_{\theta,k}^{(\ell)})\right)}{\left(\sum_{j=1}^{|\mathcal{V}|}\exp(x_{\theta,j}^{(\ell)})\right)^{2}}$ (80) $$

Therefore, we have two cases for the derivative: first, the derivative with respect to $x^{(\ell)}_{\theta,k}$ which denotes the predicted probability for the token that was sampled, and second, the derivative with respect to $x^{(\ell)}_{\theta,j}$ for $j\neq k$ which denotes the predicted probabilities for all remaining tokens.

For the first case when $j=k$, the partial derivative simplifies to

$$ $\displaystyle\frac{\partial}{\partial x_{\theta,k}^{(\ell)}}\left(\frac{\exp(x_{\theta,k}^{(\ell)})}{\sum_{j=1}^{|\mathcal{V}|}\exp(x_{\theta,j}^{(\ell)})}\right)$ $\displaystyle=\frac{\exp(x_{\theta,k}^{(\ell)})\sum_{j=1}^{|\mathcal{V}|}\exp(x_{\theta,j}^{(\ell)})-\exp(x_{\theta,k}^{(\ell)})\exp(x_{\theta,k}^{(\ell)})}{\left(\sum_{j=1}^{|\mathcal{V}|}\exp(x_{\theta,j}^{(\ell)})\right)^{2}}$ $\displaystyle=\left(\frac{\exp(x_{\theta,k}^{(\ell)})}{\sum_{j=1}^{|\mathcal{V}|}\exp(x_{\theta,j}^{(\ell)})}\right)\left(\frac{\sum_{j=1}^{|\mathcal{V}|}\exp(x_{\theta,j}^{(\ell)})-\exp(x_{\theta,k}^{(\ell)})}{\sum_{j=1}^{|\mathcal{V}|}\exp(x_{\theta,j}^{(\ell)})}\right)$ $\displaystyle=\text{SM}(x^{(\ell)}_{\theta,k})\left(1-\text{SM}(x^{(\ell)}_{\theta,k})\right)$ (81) $$

For all $j\neq k$, the derivative simplifies to

$$ $\displaystyle\frac{\partial}{\partial x_{\theta,j}^{(\ell)}}\left(\frac{\exp(x_{\theta,k}^{(\ell)})}{\sum_{j=1}^{K}\exp(x_{\theta,j}^{(\ell)})}\right)$ $\displaystyle=\frac{0-\exp(x_{\theta,j}^{(\ell)})\exp(x_{\theta,k}^{(\ell)})}{\left(\sum_{j=1}^{K}\exp(x_{\theta,j}^{(\ell)})\right)^{2}}$ $\displaystyle=-\left(\frac{\exp(x_{\theta,j}^{(\ell)})}{\sum_{j=1}^{|\mathcal{V}|}\exp(x_{\theta,j}^{(\ell)})}\right)\left(\frac{\exp(x_{\theta,k}^{(\ell)})}{\sum_{j=1}^{|\mathcal{V}|}\exp(x_{\theta,j}^{(\ell)})}\right)$ $\displaystyle=-\text{SM}(x^{(\ell)}_{\theta,j})\text{SM}(x^{(\ell)}_{\theta,k})$ (82) $$

The parameters $\theta$ are updated such that the predicted probability of sampling the token $\ell$ with argmax probability $x^{(\ell)}_{\theta,k}$ which resulted in an invalid peptide SMILES sample is reduced. The gradient update is minimized for predicted probabilities near 0 and 1, suggesting that the loss function pushes the model towards higher confidence predictions from uncertain predictions to minimize invalid sampling.

$$ $\displaystyle x^{\prime(\ell)}_{\theta,k}\leftarrow x^{(\ell)}_{\theta,k}-\eta\cdot\text{SM}(x^{(\ell)}_{\theta,k})\left(1-\text{SM}(x^{(\ell)}_{\theta,k})\right)$ (83) $$

where $\eta$ is the learning rate.

In contrast, the parameters of the remaining tokens $x^{(\ell)}_{\theta,j}$ are updated so that the predicted probability of sampling the other tokens increases proportionally to their original softmax probabilities. This prevents extreme changes in the predicted probabilities of the remaining tokens and ensures that the token distribution remains relatively consistent with the previous iteration.

$$ $\displaystyle x^{\prime(\ell)}_{\theta,j}\leftarrow x^{(\ell)}_{\theta,j}+\eta\cdot\text{SM}(x^{(\ell)}_{\theta,j})\text{SM}(x^{(\ell)}_{\theta,k})$ (84) $$

Here, we show our invalid loss effectively updates parameters to reduce the position-specific token probabilities that result in invalid sequence samplings and push the model predictions toward other high-likelihood tokens.

## Appendix C Model Architectures and Training Details

### C.1 RoFormer Architecture Details

To predict the token probabilities at each reverse step $\mathbf{x}_{\theta}(\mathbf{z}_{t},t)$, we trained a RoFormer model that leverages rotary positional embeddings (RoPE) robust to varying input lengths and long-range dependencies between tokens. The specific hyperparameters of our model are given below.

**Table 8: Roformer Architecture Hyperparameters**
| Hyperparameter | PepTune |
| --- | --- |
| Input Dimension | 581 (vocab size) |
| Hidden Dimension | 768 |
| Intermediate Dimension | 3072 |
| Number of Layers | 8 |
| Attention Heads | 8 |
| Max Positional Embeddings | 1035 |
| Hidden and Attention Dropout Probability | 0.1 |

### C.2 PDB Docking Structures

**Table 9: PDB structures used for docking.**
| Protein | PDB |
| --- | --- |
| GFAP | 6A9P |
| TfR | 3KAS |
| GLP-1R | 3C5T |
| AMHR2 | 7L0J |
| GLAST | 5LM4 |
| NCAM1 | 2HAZ |
| RBX1 | 1LDJ |

### C.3 Target-Binding Prediction Model

We trained a multi-head cross-attention network with ESM-2 protein sequence embeddings and PeptideCLM peptide SMILES embeddings for the target binding affinity model. We trained on 1806 sequences from the PepLand canonical and non-canonical binding datasets containing the protein-target sequence, peptide SMILES sequence, and the experimentally-validated $K_{d}/K_{i}/IC50$ binding affinity score. After training for 50 epochs, the regression model achieved a strong Spearman correlation coefficient of 0.949 on the test dataset and 0.633 on the validation dataset.

**Table 10: Target-Binding Affinity Predictor**
| Layers | Protein Dimension | Peptide Dimension |
| --- | --- | --- |
| Embedding Module | 1280 | 768 |
| Linear Layer | 512 | 512 |
| Layer Norm | 512 | 512 |
| Cross-Attention $\times 3$ |  |  |
| Multi-Head Attention ($h=8$) | 512 | 512 |
| Linear Layer | 2048 | 2048 |
| ReLU | 2048 | 2048 |
| Dropout | 2048 | 2048 |
| Linear Layer | 512 | 512 |
| Shared Prediction Head |  |  |
| Linear Layer | 1024 |  |
| ReLU | 1024 |  |
| Dropout | 1024 |  |
| Regression Head | 1 |  |
| Classification Head | 3 |  |

### C.4 Boosted Trees for Peptide SMILES Property Prediction

Here, we present the details on the training and hyperparameters of our trained XGBoost boosted tree regression model for membrane permeability prediction and the boosted tree binary classification model for solubility, hemolysis, and non-fouling.

**Table 11: XGBoost Hyperparameters for Classification and Regression**
| Classification Hyperparameters | Regression Hyperparameters |  |  |
| --- | --- | --- | --- |
| Hyperparameter | Value/Range | Hyperparameter | Value/Range |
| Objective | binary:logistic | Objective | reg:squarederror |
| Lambda | $[1\text{e}{-8},10.0]$ | Lambda | $[0.1,10.0]$ (log scale) |
| Alpha | $[1\text{e}{-8},10.0]$ | Alpha | $[0.1,10.0]$ (log scale) |
| Colsample by Tree | $[0.1,1.0]$ | Gamma | $[0,5]$ |
| Subsample | $[0.1,1.0]$ | Colsample by Tree | $[0.5,1.0]$ |
| Learning Rate | $[0.01,0.3]$ | Subsample | $[0.6,0.9]$ |
| Max Depth | $[2,30]$ | Learning Rate | $[1\text{e}{-5},0.1]$ |
| Min Child Weight | $[1,20]$ | Max Depth | $[2,30]$ |
| Tree Method | hist | Min Child Weight | $[1,20]$ |
|  |  | Tree Method | hist |
|  |  | Scale Pos Weight | $[0.5,10.0]$ (log scale) |

### C.5 Evaluation Metrics

Validity is defined as the fraction of peptide SMILES that pass our SMILES2PEPTIDE filter (Algorithm [8](#alg8)), indicating that it translates to a synthesizable peptide.

Uniqueness is defined as the fraction of mutually distinct peptide SMILES.

Diversity is defined as one minus the average Tanimoto similarity between the Morgan fingerprints of every pair of generated sequences, which measures the similarity in structure across generated peptides.

$$ $\displaystyle\text{Diversity}=1-\frac{1}{\binom{N_{\text{generated}}}{2}}\sum_{i,j}\frac{\mathbf{f}(\mathbf{x}_{i})\cdot\mathbf{f}(\mathbf{x}_{j})}{|\mathbf{f}(\mathbf{x}_{i})|+|\mathbf{f}(\mathbf{x}_{j})|-\mathbf{f}(\mathbf{x}_{i})\cdot\mathbf{f}(\mathbf{x}_{j})}$ (85) $$

where $\mathbf{f}(\mathbf{x}_{i})$ and $\mathbf{f}(\mathbf{x}_{j})$ are the 2048-dimensional Morgan fingerprint with radius 3 for a pair of generated sequences $\mathbf{x}_{i}$ and $\mathbf{x}_{j}$.

Similarity to Nearest Neighbor (SNN) is defined as the maximum Tanimoto similarity between a generated sequence $\mathbf{x}_{i}$ with a sequence in the dataset $\tilde{\mathbf{x}}_{j}$.

$$ $\displaystyle\text{SNN}=\max_{j\in|\mathcal{D}|}\left(\frac{\mathbf{f}(\mathbf{x}_{i})\cdot\mathbf{f}(\tilde{\mathbf{x}}_{j})}{|\mathbf{f}(\mathbf{x}_{i})|+|\mathbf{f}(\tilde{\mathbf{x}}_{j})|-\mathbf{f}(\mathbf{x}_{i})\cdot\mathbf{f}(\tilde{\mathbf{x}}_{j})}\right)$ (86) $$

Randomness is defined as the Shannon Entropy on tokenized sequences given as:

$$ $\displaystyle E=-\sum_{i}^{L}p_{i}\log_{2}(p_{i})$ (87) $$

where $p_{i}$ is the probability of $i$-th unique token divided by the total number of tokens $L$ in the sequence.

KL-Divergence is defined as the divergence between the token distribution in the generated peptide SMILES $p_{i}$ and the token distribution in the training data.

$$ $\displaystyle\text{KL}(P\big{|}\big{|}Q)$ $\displaystyle=\sum_{i\in\mathcal{V}}\begin{cases}p_{i}\log_{2}(\frac{p_{i}}{q_{i}})&\text{if}\ q_{i}>0\\ p_{i}\log_{2}(\frac{p_{i}}{10_{-9}})&\text{if}\ q_{i}=0\end{cases}$ (88) $$

where $p_{i}$ is the probability of token $i$ in the training data, and $q_{i}$ is the probability of token $i$ in the generated data.

## Appendix D Further Experiments

### D.1 Case Study for Time-Dependent Multi-Objective Guidance

Some properties of peptides require more intense guidance towards specific structural or sub-structural features, while others may only require small changes in the side chain composition or non-natural modifications. To enable the prioritization of properties during guidance, we introduce a time-dependent multi-objective guidance strategy that guides the generation based on only a subset of properties depending on the current iteration number of the MCTS search. To achieve this, we define a $K$-dimensional vector $\mathbf{i}=[i_{1},i_{2},\dots,i_{K}]$ where each $i_{k}$ is the iteration number to begin guidance for the $k$th objective. Properties where $i_{k}=0$ are used to guide all iterations, whereas properties where $i_{k}>1$ are used to guide only the iterations from $i_{k}\to N_{\text{iter}}$.

Our time-dependent guidance operates as follows. During the expansion and rollout steps on iteration $i$, the rolled-out child sequences $\mathbf{x}_{s,i}$ that are non-dominated across the sub-vector of property scores $\mathbf{s}_{i}=[s_{k}\;|\;i_{k}\leq i\leq N_{\text{iter}}]$ dependent on the iteration $i$ is added to the Pareto-optimal set $\mathcal{P}^{*}$. Therefore, $\mathbf{x}_{s}$ does not need to be non-dominated in the properties $k$ where $i_{k}>i$. Similarly, only the sequences $\mathbf{x}^{*}\in\mathcal{P}^{*}$ that become dominated when adding $\mathbf{x}_{s}$ in the subset of properties represented in $\mathbf{s}_{i}(\mathbf{x}^{*})$ are removed from $\mathcal{P}^{*}$.

$$ $\displaystyle\mathcal{P}^{\prime*}$ $\displaystyle=\mathcal{P}^{*}\cup\big{\{}(\mathbf{z}_{s},\mathbf{s}(\mathbf{x}_{s}))\;|\;\forall\mathbf{x}^{*}\in\mathcal{P}^{*}\;\;\mathbf{s}_{i}(\mathbf{x}_{s})\succeq\mathbf{s}_{i}(\mathbf{x}^{*})\big{\}}$ (89) $\displaystyle\mathcal{P}^{\prime*}$ $\displaystyle=\mathcal{P}^{*}\setminus\big{\{}\mathbf{x}^{*}\;|\;\exists\mathbf{x}_{s}\in\text{children}(\mathbf{z}_{t})\;\text{s.t.}\;\mathbf{s}_{i}(\mathbf{x}_{s})\succ\mathbf{s}_{i}(\mathbf{x}^{*})\big{\}}$ (90) $$

Then, during selection, we only consider the cumulative rewards $\mathbf{W}_{i}=[W_{k}\;|\;i_{k}\leq i\leq N_{\text{iter}}]$ for the properties where $i_{k}$ s.t. $i_{k}\leq i\leq N_{\text{iter}}$ when computing the selection score vector $\mathbf{U}_{i}$ to form the Pareto-optimal selection set.

$$ $\displaystyle\mathcal{P}^{\prime*}_{\text{select}}$ $\displaystyle=\mathcal{P}^{*}_{\text{select}}\cup\big{\{}\mathbf{z}_{s}\;|\;\forall\mathbf{x}^{*}\in\mathcal{P}^{*}_{\text{select}}\;\;\mathbf{U}_{i}(\mathbf{z}_{t},\mathbf{z}_{s})\succeq\mathbf{U}_{i}(\mathbf{z}_{t},\mathbf{z}^{*})\big{\}}$ (91) $\displaystyle\mathcal{P}^{\prime*}_{\text{select}}$ $\displaystyle=\mathcal{P}^{*}_{\text{select}}\setminus\big{\{}\mathbf{z}^{*}\;|\;\exists\mathbf{z}_{s}\in\text{children}(\mathbf{z}_{t})\;\text{s.t.}\;\mathbf{U}_{i}(\mathbf{z}_{t},\mathbf{z}_{s})\succ\mathbf{U}_{i}(\mathbf{z}_{t},\mathbf{z}^{*})\big{\}}$ (92) $$

Finally, we select the next node $\mathbf{z}_{s}\sim\mathcal{P}^{\prime*}_{\text{select}}$ uniformly at random from the Pareto-optimal selection set.

To test this strategy, we generated 100 peptides conditioned only on membrane permeability for the first 50 iterations since we found it as the most challenging property to optimize. Then we conditioned all properties including membrane permeability, binding affinity to GFAP, solubility, hemolysis, and non-fouling. We show that during the first 50 iterations, all properties except membrane permeability show relatively constant average scores, whereas the permeability score increased (Figure [16](#A4.F16)). Then, after the 50 iteration mark, GFAP binding affinity and solubility curves increased significantly while the hemolysis and non-fouling curves increased slightly for the remainder of the iterations (Figure [16](#A4.F16)). Although all the results in this paper leverage peptides without time-dependent guidance, this serves as a proof of concept for future experiments varying the start times across properties to refine certain properties at later time steps where the generated sequences are already constrained to specific predefined substructures.

Figure: Figure 16: Time-Dependent Multi-Objective Guidance. (A) Plot of average membrane permeability score for 50 sampled sequences in the expansion and rollout step over iterations where the MCTS search is conditioned on permeability for all iterations. (B) Plot of average predicted binding affinity score to GFAP over iterations when conditioned starting from epoch 50. (C, D, E) Plot of average predicted solubility, hemolysis, and non-fouling scores over iterations when conditioned starting from epoch 50. Pink dotted lines denote the iteration where the MCTS search began conditioning on the property
Refer to caption: /html/2412.17780/assets/Figures/time-dependent.png

### D.2 Ablation Studies

In this section, we report the performance of our multi-objective MCTS-guided discrete diffusion model over different hyperparameter settings. Specifically, we discuss the effects of changing the number of expanded children nodes, the number of iterations, and the tokenization scheme.

Number of Children. The number of children $M$ is the hyperparameter that determines the batch size during the expansion step of MCTS. A small number of expanded nodes would limit the degree of exploration and number of generated sequences for evaluation at each iteration. If the initial iterations resulted in sub-optimal unmasking steps for all children, this could prevent the algorithm from discovering a local or global optimum across objectives. Suppose the number of children is too large. In that case, this can result in a lack of diversity if several sequences from the same expansion step are added to the Pareto-optimal set given their sequence similarity, leading to similar property scores. A large $M$ also slows down runtime significantly. To determine a value for $M$ within the two extremes, we evaluated the performance of the MCTS search for $M=10,50,70,100$. Overall, we found that $M=50$ yields consistently increasing scores across all properties, which we use for the remainder of the study.

Number of Iterations. The number of iterations $N_{\text{iter}}$ determines the number of selection, expansion, rollout, and backpropagation loops executed in a single MCTS run. In addition, $N_{\text{iter}}$ is the maximum value of $t$ or the number of unmasking steps that can be executed before the rollout begins, which corresponds to the maximum tree depth. We found that equating the number of diffusion steps $T$ to the number of MCTS iterations $N_{\text{iter}}$ results in convergence on the property prediction scores as the selection process becomes biased towards a single unmasking scheme. As shown in Figure [6](#S2.F6), all property scores converge for $N_{\text{iter}}=T=128$, which we use for the remainder of the study.

Tokenization Scheme. To evaluate the effect of different tokenization methods on the generation quality, we experimented with three different tokenization schemes: SMILES Pair Encoding (SPE) tokenization with the trained vocabulary used by PeptideCLM and Atom Pair Encoding (APE) tokenization for SMILES and SELFIES representations. Overall, we found that the SPE tokenization scheme decreased perplexity and maintained precision while capturing common peptide motifs like bonds and recurring side chains.

Table: Table 12: Effect of Tokenization on Sequence Length, Training, and Validation Loss after Convergence

## Appendix E Algorithms

Algorithm [1](#alg1) outlines the forward pass diffusion process (training) for PepMDLM, our unconditional peptide SMILES generator. Algorithms [2](#alg2), [3](#alg3), [4](#alg4), [5](#alg5) and [6](#alg6) describe PepTune, our MCTS-guided peptide SMILES generator. Algorithms [7](#alg7) and [8](#alg8) describe the bond mask function and peptide sequence decoder which can also act as a validity filter.

Figure: Algorithm 1 PepMDLM Training

Figure: Algorithm 2 Reverse Diffusion Step

Figure: Algorithm 3 PepTune: Multi-Objective Guided Discrete Diffusion with Monte-Carlo Tree Search (MCTS)

Figure: Algorithm 4 Batched Reverse Step

Figure: Algorithm 5 Selection

Figure: Algorithm 6 Update Pareto Front

Figure: Algorithm 7 Bond Mask

Figure: Algorithm 8 SMILES2PEPTIDE