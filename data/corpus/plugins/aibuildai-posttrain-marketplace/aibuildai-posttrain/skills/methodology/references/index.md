# Every post-training method in this skill

All 111 methods that earned a card: 104 read at 2026-08-10, plus seven distillation methods added by hand on 2026-09-03 (see How this list was built). The split below is the whole point: 33 have a working implementation in a library you can install today, and 78 do not, so running one of those means building it. Both tiers have a card; only the first tier is repeated in the SKILL.md table.

## Methods with an implementation (33)

| Method | Full name | Variant of | Shipped by | Card |
|---|---|---|---|---|
| `DAPO` | Dual Approximation Policy Optimization | GRPO | trl ... | `dapo.md` |
| `DoRA` | Weight-Decomposed Low-Rank Adaptation | LoRA | Hugging Face PEFT, via `LoraConfig(use_dora=True)` ... | `dora.md` |
| `DPO` | Direct Preference Optimization | the PPO-based RLHF pipeline of Ziegler et al. ... | trl (`DPOTrainer`). verl ships no standalone DPO trainer or config recipe: its documented algorithms list ... | `dpo.md` |
| `GDPO` | Group reward-Decoupled Normalization Policy Optimization | GRPO, itself a variant of PPO | verl, as an advantage estimator ... | `gdpo.md` |
| `GKD` | Generalized Knowledge Distillation | Supervised KD | trl, two entry points: `trl.experimental.gkd.GKDTrainer` for the full paper, and the top-level `trl.DistillationTrainer` for the fully on-policy case ... | `gkd.md` |
| `GPG` | Group Policy Gradient | GRPO, with the clipped surrogate objective, KL-vs-reference term ... | verl, via `algorithm.adv_estimator: gpg` plus `actor_rollout_ref.actor.policy_loss.loss_mode: "gpg"` ... | `gpg.md` |
| `GRPO` | Group Relative Policy Optimization | PPO | trl (`GRPOTrainer`), verl (`algorithm.adv_estimator: grpo`) | `grpo.md` |
| `GSPO` | Group Sequence Policy Optimization | GRPO, which is itself a variant of PPO | trl, via `GRPOTrainer` configured with `importance_sampling_level="sequence"` (there is no separate `GSPOTrainer` class) ... | `gspo.md` |
| `IPO` | Identity Preference Optimisation | DPO ... | trl, as `loss_type="ipo"` inside the top-level `DPOTrainer` class exported by `trl` ... | `ipo.md` |
| `KTO` | Kahneman-Tversky Optimization | DPO ... | trl ... | `kto.md` |
| `LAB` | Large-scale Alignment for chatBots | a taxonomy-guided variant of Self-Instruct-style synthetic data generation ... | instructlab-training (`from instructlab.training import run_training, TrainingArgs, TorchrunArgs, DeepSpeedOptions`) ... | `lab.md` |
| `LISA` | Layerwise Importance Sampling | not a formal variant of LoRA — the paper presents it as an alternative to LoRA — but ... | axolotl, config keys `lisa_n_layers` and `lisa_step_interval` (plus `lisa_layers_attribute`, default `model.layers`) ... | `lisa.md` |
| `LOMO` | LOw-Memory Optimization | SGD ... | `transformers` `Trainer`, entry point `TrainingArguments(optim="lomo")` / `optim="adalomo"` ... | `lomo.md` |
| `LORA` | Low-Rank Adaptation of Large Language Models | standard (full) fine-tuning, contrasted with adapter tuning | Hugging Face PEFT, whose top-level package exports `LoraConfig` and `get_peft_model` as the entry point ... | `lora.md` |
| `MiniLLM` |  | word-level (forward-KLD) knowledge distillation | trl, `trl.experimental.minillm.MiniLLMTrainer` with `MiniLLMConfig`; the experimental namespace may change or be removed without deprecation ... | `minillm.md` |
| `MPO` | Mixed Preference Optimization | DPO, with BCO's quality loss and an SFT-style generation loss added as extra terms | trl ... | `mpo.md` |
| `ORPO` | Optimization without Reference Model | not a strict derivative of one parent ... | LLaMA-Factory, as a stable, non-experimental training approach ... | `orpo.md` |
| `PPO` | proximal policy optimization | TRPO | trl ... | `ppo.md` |
| `PRIME` | Process Reinforcement through IMplicit rEwards | not a single-parent variant — it is a reward-shaping layer usable with several ... | verl, as an optional recipe ... | `prime.md` |
| `QLORA` | Quantized Low-Rank Adaptation | LoRA | `transformers` `BitsAndBytesConfig` ... | `qlora.md` |
| `RAFT` | Refinement and Adaptive Distillation | best-of-K (rejection) sampling, cited by the RAFT paper to WebGPT and Cobbe et al. ... | LMFlow (`lmflow.pipeline.raft_aligner.RaftAligner`, the toolkit built by the RAFT paper's own co-authors) ... | `raft.md` |
| `REMAX` |  | REINFORCE (Williams, 1987; 1992), as the paper itself states ... | verl, via `algorithm.adv_estimator=remax` ... | `remax.md` |
| `RLHF` | Reinforcement Learning from Human Feedback | not a variant of a single named post-training method ... | trl ships a `PPOTrainer`, but only under the experimental namespace `trl.experimental.ppo.PPOTrainer` ... | `rlhf.md` |
| `RPO` | Regularized Preference Optimization | DPO | trl, via ... | `rpo.md` |
| `SAPO` | Soft Adaptive Policy Optimization | GRPO | trl (`GRPOTrainer` with `loss_type="sapo"`), verl ... | `sapo.md` |
| `SDFT` | Self-Distillation Fine-Tuning | on-policy distillation | trl, under `trl.experimental.sdft` (`SDFTConfig`, `SDFTTrainer`) | `sdft.md` |
| `SeqKD` | Sequence-Level Knowledge Distillation | knowledge distillation (Hinton et al., 2015), specialized to sequence-level targets | trl, only under the experimental namespace as `trl.experimental.gkd.GKDTrainer` with `GKDConfig(seq_kd=True, lmbda=0.0)` ... | `seqkd.md` |
| `SFT` | supervised fine-tuning | none in the RL-method sense ... | trl, `SFTTrainer` / `SFTConfig`, contributed by Younes Belkada ... | `sft.md` |
| `SIMPO` | Simple Preference Optimization | DPO | trl ... | `simpo.md` |
| `SLIC` | show how the recently introduced Sequence Likelihood Calibration | SLiC | trl, as `loss_type="hinge"` inside the stable, top-level `DPOTrainer` (`from trl import DPOTrainer`) | `slic.md` |
| `TTRL` | Test-Time Reinforcement Learning | not a new RL algorithm but a new reward function layered on an existing one ... | the PRIME-RL/TTRL GitHub repository, the authors' own verl fork built for reproducing the paper's experiments ... | `ttrl.md` |
| `VERA` | Vector-based Random Matrix Adaptation | LoRA | `peft` (Hugging Face) exports `VeraConfig` and `VeraModel` from its top-level package ... | `vera.md` |
| `WPO` | Weighted Preference Optimization | DPO | trl, as `use_weighting` (bool, default `False`) on `DPOConfig` ... | `wpo.md` |

## Methods with no implementation found (78)

Each card states where the search looked and what it found. Running one of these means writing the trainer yourself, usually from the paper plus the authors' own research code.

| Method | Full name | Paper | Variant of | What is missing | Card |
|---|---|---|---|---|---|
| `ACT` | Action-Based Contrastive Self-Training | arXiv:2406.00222 | DPO | no mainstream RLHF/post-training library ships an ACT trainer as a top-level export ... | `act.md` |
| `ALORA` | an innovative approach we call allocating low-rank adaptation | arXiv:2403.16187 | LoRA | no library implements ALoRA. As of this check ... | `alora.md` |
| `APA` | Alignment with Perceived Ambiguity | arXiv:2404.11972 | supervised fine-tuning / instruction alignment ... | no training-framework library implements APA. The paper's own code, heyjoonkim/APA on GitHub ... | `apa.md` |
| `ARPO` | Agentic Reinforced Policy Optimization | arXiv:2507.19849 | GRPO, restated inside the ARPO paper with the same clipped surrogate ... | no mainline framework ships ARPO — a case-insensitive text search for "arpo" in trl's GRPOTrainer documentation page and verl's ... | `arpo.md` |
| `BLOB` | beyond post-training Bayesianization and propose Bayesian Low-Rank Adaptation by Backpropagation | arXiv:2406.11675 | LoRA, with a Bayesian variational posterior added only on the $A$ matrix | no PEFT-standard library (trl, peft) ships BLoB as of this reading | `blob.md` |
| `BOND` | Best-of-N Distillation | arXiv:2407.14622 | distills Best-of-N sampling into policy weights ... | no open-source RL library implements BOND. trl's `docs/source/_toctree.yml` toctree has no BOND entry anywhere in it ... | `bond.md` |
| `BPO` | Bootstrapped Preference Optimization | arXiv:2403.08730 | DPO | a GitHub repository-name/description search for "bootstrapped preference optimization" and "BPO" returns exactly one result ... | `bpo.md` |
| `BREAD` | Branched Rollouts from Expert Anchors Bridge SFT & RL | arXiv:2506.17211 | GRPO, itself a variant of PPO | no library implements BREAD as a named trainer at the time of this card ... | `bread.md` |
| `CFT` | Critique Fine-Tuning | arXiv:2501.17703 | SFT | no library ships a distinct "CFT trainer." The method's own released code trains it as a standard supervised-fine-tuning run ... | `cft.md` |
| `CODI` | Continuous Chain-of-Thought via Self-Distillation | arXiv:2502.21074 | Coconut's autoregressive continuous-thought mechanism ... | no general-purpose post-training library ships a CODI trainer. trl's `trl/trainer` directory ... | `codi.md` |
| `Context Distillation` |  | arXiv:2209.15189 | knowledge distillation, specialized to a same-weights, prompt-difference setting | no framework implements context distillation under this name; trl's `DistillationTrainer` is GKD, a different method ... | `context-distillation.md` |
| `DART` | Difficulty-Aware Rejection Tuning | arXiv:2407.13690 | rejection sampling fine-tuning (RFT) ... | no RL or SFT training library (trl, verl, axolotl) exposes a `DARS`/DART trainer | `dart.md` |
| `DICE` | DPO ImpliCit rEwards | arXiv:2406.09760 | DPO, used twice - once to produce the starting policy, once per bootstrapping round | no library implementation was found in the sources read for this card ... | `dice.md` |
| `Distilling Step-by-Step` |  | arXiv:2305.02301 | knowledge distillation, using chain-of-thought prompting for the extra supervision | no post-training library ships it as a named trainer; the authors' own repository builds a custom `TaskPrefixTrainer` ... | `distilling-step-by-step.md` |
| `DistiLLM` |  | arXiv:2402.03898 | KD with KLD, skewing the divergence and replacing continual on-policy generation | no framework in this corpus ships a DistiLLM trainer; trl's `GKDTrainer` uses generalized JSD, not skew KLD ... | `distillm.md` |
| `DNO` | Direct Nash Optimization | arXiv:2404.03715 | DPO ... | no library implements DNO. Checked at trl commit `dd7cbaf` (2026-07-31): the top-level `trl/__init__.py` exports only ... | `dno.md` |
| `DPA` | Directional Preference Alignment | arXiv:2402.18571 | Rejection Sampling Fine-tuning / RAFT, made preference-conditional and multi-objective | no library ships DPA as a named trainer | `dpa.md` |
| `DPG` | Distributional Policy Gradient | arXiv:2302.08215 | Policy Gradient / REINFORCE (Williams, 1992), as cited by the DPG paper | no packaged training library implements DPG. Checked trl's trainer directory listing ... | `dpg.md` |
| `DYLORA` | dynamic low-rank adaptation | arXiv:2210.07558 | LoRA | no maintained library implements DyLoRA under a top-level export ... | `dylora.md` |
| `FEDIT` | Federated Instruction Tuning | arXiv:2305.05644 | FedAvg, specialized to LoRA-adapter-only local updates for LLM instruction tuning | this card did not fetch or check trl, verl, OpenRLHF, or any other post-training library's docs or source for a FedIT trainer ... | `fedit.md` |
| `GFPO` | Group Filtered Policy Optimization | arXiv:2508.09726 | GRPO | no library implements GFPO as of this check. `huggingface/trl`'s top-level package export list ... | `gfpo.md` |
| `GPO` | generalized preference optimization | arXiv:2402.05749 | none in the sense of a single parent ... | no framework ships a trainer literally named "GPO." trl's `DPOTrainer` implements three of the paper's named instances as a ... | `gpo.md` |
| `GRM` | generative reward modeling | arXiv:2504.02495 | as an RM output format, a departure from the scalar/Bradley-Terry reward model ... | no library implements SPCT's training pipeline (rejective-sampling cold start plus rule-based RL) as of this check — trl's GRPO ... | `grm.md` |
| `HIR` | Hindsight Instruction Relabeling | arXiv:2302.05206 | HER, adapted from goal-conditioned robotic RL to language-model instruction following | no library implementation was found in the searches this card ran | `hir.md` |
| `INSCL` | Instruction-based Continual Learning | arXiv:2403.11435 | Continual-T0 / random replay for continual instruction tuning | no post-training library in this survey packages InsCL as an importable trainer | `inscl.md` |
| `JMLR` | Joint Medical LLM and Retrieval Training | arXiv:2402.17887 | RAG, with the retriever initialized from and structured as ColBERT | no packaged training library implements JMLR - it is not available as an importable trainer in trl or verl | `jmlr.md` |
| `LAT` | latent adversarial training | arXiv:2407.15549 | untargeted LAT (Casper et al., 2024) ... | trl's documented trainer index ... | `lat.md` |
| `LCPO` | Length Controlled Policy Optimization | arXiv:2503.04697 | GRPO, via a reward-function change, not a change to GRPO's policy-gradient update | no framework ships LCPO as a built-in trainer or reward recipe as of this check ... | `lcpo.md` |
| `LIPO` | Listwise Preference Optimization | arXiv:2402.01878 | DPO ... | no library implements the general LiPO-λ listwise objective (Eq. 10, K responses, the $\Delta_{i,j}$ Lambda weight) | `lipo.md` |
| `LOFIT` | Localized Fine-Tuning on LLM Representations | arXiv:2406.01563 | the learning-free localized representation-intervention framework instantiated by ITI and ... | no PEFT or RL-training library implements LoFiT. Checked by reading the top-level export lists of Hugging Face `peft` ... | `lofit.md` |
| `MALT` | Multi-Agent LLM Training | arXiv:2412.01928 | DPO, applied per-role on top of an SFT stage the paper describes as analogous to STaR | no framework this card checked implements MALT as a named trainer or recipe | `malt.md` |
| `MAPORL` | Multi-Agent Post-co-training for collaborative LLMs with Reinforcement Learning | arXiv:2502.18439 | multi-agent PPO, itself an extension of PPO | no established RL post-training library (TRL, verl, or similar) ships MAPoRL as a built-in trainer | `maporl.md` |
| `OLORA` | orthogonal low-rank adaptation | arXiv:2310.14152 | LoRA | no established training library was found to implement O-LoRA (continual learning) as a trainer | `olora.md` |
| `OPSD` | On-Policy Self-Distillation | arXiv:2601.18734 | on-policy distillation (GKD) ... | no library ships an `OPSDTrainer` | `opsd.md` |
| `PAL` | Pluralistic Alignment Framework | arXiv:2406.08469 | the Bradley-Terry-Luce reward-modeling approach used across RLHF alignment ... | PAL's own README describes no trl or verl integration ... | `pal.md` |
| `PM` | preference matching | arXiv:2405.16455 | standard KL-regularized RLHF, which the paper attributes to InstructGPT ... | no library implementation was found | `pm.md` |
| `PRORL` | Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models | arXiv:2505.24864 | GRPO, with DAPO's decoupled clipping and dynamic sampling folded in | no library ships a "ProRL" trainer | `prorl.md` |
| `PUMA` | propose a Personalized User Memory-enhanced Alignment | arXiv:2410.17236 | DPO ... | no general-purpose RL/alignment library ships PUMA as a named trainer ... | `puma.md` |
| `RAFE` | Ranking Feedback Improves Query Rewriting | arXiv:2405.14431 | DPO, KTO ... | no library ships "RaFe" as a named trainer; it is a training recipe (reranker score -> DPO/KTO/PPO) built on existing trainers | `rafe.md` |
| `REFT` | Reinforced Fine-Tuning | arXiv:2401.08967 | PPO | neither trl nor verl implements a trainer named for ReFT ... | `reft.md` |
| `REINFORCE` |  | arXiv:2312.08935 | none — REINFORCE is the foundational stochastic policy-gradient algorithm in this corpus | no framework surveyed here ships a trainer literally implementing Williams's REINFORCE with a moving-average baseline | `reinforce.md` |
| `ReMA` | Reinforced Meta-thinking Agents | arXiv:2503.09501 | single-agent RL for LLM reasoning (VRP-RL / MRP-RL) ... | trl's and verl's documentation carry no "ReMA" entry point as of this card ... | `rema.md` |
| `REST` | Reinforced Self-Training | arXiv:2406.03816 | ReST$^{EM}$, which is itself a simplified ... | no training-library trainer implements ReST-MCTS\* under this name | `rest.md` |
| `RFT` | Rejection sampling Fine-Tuning | arXiv:2308.01825 | SFT ... | no training library ships a dedicated "RFT" trainer as of this reading ... | `rft.md` |
| `RLAIF` | RL from AI Feedback | arXiv:2309.00267 | RLHF, with the AI-feedback substitution introduced as RLAIF by Bai et al | no library ships a trainer named "RLAIF" ... | `rlaif.md` |
| `RLCS` | Reinforcement Learning with Curriculum Sampling | arXiv:2507.01006 | GRPO, itself a variant of PPO | no released training framework implements RLCS's difficulty-tiered curriculum sampler as of this check ... | `rlcs.md` |
| `RLIF` | Reinforcement Learning from Internal Feedback | arXiv:2505.19590 | RLIF is presented as a paradigm parallel to RLHF and RLVR (exemplified by DeepSeek-R1) ... | no production trainer library ships RLIF or Intuitor as of this reading ... | `rlif.md` |
| `RLSQM` | Reinforcement Learning from Static Quality Metrics | arXiv:2310.02368 | PPO, via the InstructGPT-style RLHF recipe | no library implements RLSQM. A Hugging Face Hub model search for "RLSQM" returned zero results, checked 2026-08-10 ... | `rlsqm.md` |
| `RLVR` | Reinforcement Learning with Verifiable Rewards | arXiv:2411.15124 | the KL-constrained RLHF objective with a learned reward model ... | no framework ships a trainer literally named "RLVR" — it is a reward-function convention ... | `rlvr.md` |
| `RRHF` | Rank Responses to Align Language Models | arXiv:2304.05302 | PPO, per the paper's own framing of RRHF as a simplification of the PPO stage of RLHF | no major RL-post-training library was found to implement RRHF. Neither trl nor verl lists an RRHF trainer or advantage estimator ... | `rrhf.md` |
| `RRM` | robust reward model | arXiv:2409.13156 | pairwise/Bradley-Terry reward-model training | no library implements RRM as a named trainer or off-the-shelf recipe as of this check ... | `rrm.md` |
| `RSO` | Rejection Sampling Optimization | arXiv:2312.11456 | DPO and SLiC ... | no framework I checked implements RSO's rejection-sampling data-construction step. trl's `DPOTrainer` exposes ... | `rso.md` |
| `SCOTT` | Self-Consistent Chain-of-Thought Distillation | arXiv:2305.01879 | rationale-distillation knowledge distillation in the self-rationalization paradigm ... | no post-training library implements SCOTT as a ready trainer ... | `scott.md` |
| `SDPO` | Self-Distillation Policy Optimization | arXiv:2601.20802 | GRPO, which is itself a variant of PPO | only the paper's own code, `lasgroup/SDPO` ... | `sdpo.md` |
| `SEAL` | Safety-enhanced Aligned LLM Fine-tuning | arXiv:2410.07471 | not a variant of PPO/DPO-style post-training methods ... | no general post-training library ... | `seal.md` |
| `SGRPO` | Serial-Group Decaying-Reward Policy Optimization | arXiv:2505.07686 | GRPO | no framework implementation found | `sgrpo.md` |
| `SKD` | Speculative Knowledge Distillation | arXiv:2410.11325 | On-Policy KD / GKD | no widely-used library implements the token-level accept/resample loop; only the authors' `speculative_kd` research code ... | `skd.md` |
| `SKTUNING` | Self-Knowledge Tuning | arXiv:2402.09267 | a reference-free special case of DPO's pairwise loss | no packaged training library implements SK-Tuning — a search of trl's live `main`-branch documentation index page ... | `sktuning.md` |
| `SPIN` | Self-Play fIne-tuNing | arXiv:2401.01335 | SFT ... | no framework ships the paper's original algorithm as a top-level trainer. verl provides an experimental recipe ... | `spin.md` |
| `SPIRAL` | Self-Play on Zero-Sum Games Incentivizes Reasoning | arXiv:2506.24119 | REINFORCE, adapted to a shared-policy two-player self-play setting with a ... | only the paper's own repository, spiral-rl/spiral, run via `bash run.sh` ... | `spiral.md` |
| `SPO` | Self-Play Preference Optimization | arXiv:2401.04056 | the two-policy dueling technique for computing Minimax Winners ... | no library implements it. `trl`'s trainer directory (as of the 2026-08-09 fetch) exports `dpo`, `grpo`, `kto`, `reward`, `rloo` ... | `spo.md` |
| `SPPO` | Self-Play Preference Optimization | arXiv:2405.00675 | DPO (same reference-model log-ratio parameterization; different loss target) | the paper's own reference implementation, `uclaml/SPPO`'s `SPPOTrainer` ... | `sppo.md` |
| `SPT` | supervised pinpoint tuning | arXiv:2409.01658 | structured selective PEFT, per Ding et al.'s PEFT taxonomy ... | no packaged library implements SPT. The paper's own repository ... | `spt.md` |
| `SRFT` | Supervised Reinforcement Fine-Tuning | arXiv:2506.19767 | GRPO, via an off-policy demonstration-augmentation design similar to LUFFY | no mainline framework (trl, upstream verl) implements SRFT as of this check | `srft.md` |
| `SRPO` | we present two-Staged history-Resampling Policy Optimization | arXiv:2504.14286 | GRPO | no framework implements SRPO's two-stage curriculum or History Resampling as a named trainer or config | `srpo.md` |
| `STAR` | Similarity-guided Teacher-Assisted Refinement | arXiv:2403.09629 | STaR (Self-Taught Reasoner) | no RL or post-training library was found to ship a Quiet-STaR trainer ... | `star.md` |
| `STARPO` | State-Thinking-Actions-Reward Policy Optimization | arXiv:2504.20073 | generalizes PPO and GRPO from single-turn to trajectory-level optimization ... | RAGEN ... | `starpo.md` |
| `STEPGRPO` | Step-wise Group Relative Policy Optimization | arXiv:2503.12937 | GRPO | no library implements StepGRPO - it does not appear in trl's or verl's trainer lists as of this reading | `stepgrpo.md` |
| `TDPO` | Token-level Direct Preference Optimization | arXiv:2404.11999 | DPO | no established framework | `tdpo.md` |
| `TORL` | Tool-Integrated Reinforcement Learning | arXiv:2503.23383 | GRPO, with the policy-update math untouched and the rollout/reward changed | no framework ships a trainer named ToRL. The paper's own reference implementation is the GAIR-NLP/ToRL GitHub repository ... | `torl.md` |
| `TREEGRPO` | Tree-based Group Relative Policy Optimization | arXiv:2509.21240 | GRPO | only the authors' own research repository, AMAP-ML/Tree-GRPO on GitHub (Apache-2.0, 393 stars as of this reading) ... | `treegrpo.md` |
| `UFT` | Unified Fine-Tuning | arXiv:2505.16984 | GRPO, extended with the hint mechanism introduced by R3 | no library ships UFT as a named trainer | `uft.md` |
| `URM` | Uncertainty-aware Reward Model | arXiv:2410.00847 | multi-attribute reward models, e.g. the reward model in Nemotron-4 340B ... | no RL/alignment framework trainer implements URM. The authors released trained checkpoints on the Hugging Face Hub ... | `urm.md` |
| `VDPO` | Vision-guided Direct Preference Optimization | arXiv:2411.02712 | DPO | no library implements V-DPO. The paper's own reference code is a standalone research repository ... | `vdpo.md` |
| `VPD` | Visual Program Distillation | arXiv:2312.03052 | ViperGPT's visual-program framework ... | no library implements VPD. A GitHub repository search for "visual program distillation" returns only the paper's own ... | `vpd.md` |
| `VPO` | value-incentivized preference optimization | arXiv:2405.19320 | DPO | no framework implements it. trl's top-level package exports no VPO-named trainer class ... | `vpo.md` |
| `VRPO` | Variance-Reduced Preference Optimization | arXiv:2505.19223 | DPO | no library implements VRPO training | `vrpo.md` |
| `WARM` | we propose Weight Averaged Reward Models | arXiv:2401.12187 | prediction ensembling of reward models ... | no WARM implementation was found in trl or verl, the two libraries checked for this card ... | `warm.md` |

## How this list was built

34 arXiv phrase queries ran at 2026-08-10, plus the algorithm sources of the libraries in the `framework` skill and a bulk ACL pass. They produced 3,984 distinct method names. A name earned a card only when the paper that introduced it could be picked by comparing candidates or by matching a title, when its citation count cleared the tail, and when a person read it and confirmed it is a post-training method for a language model: the same arXiv words also return quantization, unlearning, prompt tuning, benchmarks, and robot control, and no query separates them.

**This list is not saturated.** Capture-recapture across the two channels (3,547 arXiv names, 841 ACL names, 511 in both) estimates about 5,838 names exist, so the arXiv side reached about 60.8% of them. Queries were still adding names that nobody had seen when gathering stopped. A method that is not here may still exist.

**Seven distillation methods were added by hand on 2026-09-03.** The harvest missed them for two separate reasons, both recorded in issue #325: GKD was found with 621 citations but its paper attribution was a single-candidate guess, which the picker refuses before the keep list is consulted; the other six carry names that are not all-capital acronyms, so the name extractor never saw them at all. Each was given the same card treatment as a harvested method, and their arXiv identity was checked by hand. Four citation counts are blank because the Semantic Scholar rate limit refused the read, and a guess would have been worse than a blank.
