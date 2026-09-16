---
description: >-
  Methods that post-train a base language model. 111 method cards carrying the paper's
  math, the cost in theory and in practice, the knobs, and the training signals to
  watch. The table in this skill holds the 33 methods a library ships today, each with a
  one-sentence when-to-pick; references/index.md carries the other 78, which nobody has
  implemented yet. references/inference.md is the generation-settings fix that changes
  scores without training. Use it when choosing how to train, and again while a run is
  live to read its signals.
---

# Post-training methods

A post-training method is the rule that turns data into a weight update: which loss is computed, what the data must look like, and which extra models have to be in memory while it runs. This skill serves one decision - which method to train with - and one habit while the run is live. The thing that decides the choice in practice is not novelty: a method with an efficient implementation in a library you already run beats a newer method you would have to build, unless the newer one buys something the task actually needs.

## When to consult this skill

- The designer, when the plan has data and a goal but no training rule yet.
- The implementer, before writing the training config: the card names the knobs and the defaults each library really uses.
- The implementer again while a run is live: the chosen method's card says which signals to read and what their shapes mean.

## General training craft

This part belongs to no single method and no single library, so it lives here once instead of on a hundred cards.

- Watch the gradient norm, the loss shape, and the learning rate against the stage you are in. A loss that falls to zero in the first hundred steps is usually a data bug, not a fast learner.
- Tune one thing at a time and keep the pair of numbers that the change produced. A setting adopted without a before-and-after pair is a guess.
- Fix the generation settings before you fix the training: `references/inference.md` is the hour-zero change that moves scores without touching weights, and the documented jumps there are tens of points. Read it first, then train.
- Method-specific signals do not live here. DPO watches its reward margins, GRPO watches entropy and the spread of group rewards; each method's card carries its own Watch section.

## Method index

**This table holds only the methods a library ships today.** 33 of the 111 methods with a card have an implementation you can install; the other 78 are in `references/index.md`, one hop away, and running one of those means writing the trainer yourself.

| Method | When to pick it | Variant of | Data it needs | Extra models | Shipped by | File |
|---|---|---|---|---|---|---|
| `DAPO` | online RL with a rule-based or verifiable reward (not a learned reward model) on long chain-of-thought tasks ... | GRPO | prompts each paired with a checkable ground-truth answer ... | none beyond the policy | trl ... | `dapo.md` |
| `DoRA` | pick DoRA over plain LoRA when the low-rank adapter is capacity-limited ... | LoRA | whatever the underlying supervised fine-tuning task needs ... | none beyond whatever the base fine-tuning objective ... | Hugging Face PEFT ... | `dora.md` |
| `DPO` | pick DPO when you already have (or can cheaply construct) a static dataset of paired preferences and do not want to run generation inside the ... | the PPO-based RLHF pipeline of ... | a static dataset of ... | one frozen reference model ... | trl (`DPOTrainer`). verl ships no standalone ... | `dpo.md` |
| `GDPO` | pick GDPO over plain GRPO specifically when training against two or more reward components at once ... | GRPO, itself a variant of PPO | prompts plus, for each rollout ... | none required by the method's own definition beyond ... | verl ... | `gdpo.md` |
| `GKD` | pick GKD to compress a larger teacher into a smaller student while correcting the exposure-bias gap that fixed-sequence imitation leaves behind, when you can afford to run the student's decoder during training ... | Supervised KD ... | input prompts, optionally paired with fixed target sequences when the student-data fraction is below 1; no reward signal and no preference pairs ... | one frozen teacher, forward passes only, at every training step ... | trl, `trl.experimental.gkd.GKDTrainer` for the full paper and the top-level `trl.DistillationTrainer` for the fully on-policy case ... | `gkd.md` |
| `GPG` | on-policy RL for LLM reasoning when every sampled completion gets a single scalar outcome reward and you want to avoid GRPO's extra reward-std ... | GRPO ... | prompts ... | none required by the method's own definition — Table 2 ... | verl ... | `gpg.md` |
| `GRPO` | online RL when every sampled completion can be scored with a scalar reward (a programmable function or a reward model) and memory is tight ... | PPO | prompts from the distribution you care about ... | never a value network | trl (`GRPOTrainer`) ... | `grpo.md` |
| `GSPO` | choose GSPO over its parent GRPO when training runs slightly off-policy ... | GRPO ... | prompts from the target distribution plus a per-response ... | none required by the objective itself — like GRPO ... | trl ... | `gspo.md` |
| `IPO` | pick IPO over plain DPO when preference labels are likely to be sparse, imbalanced across actions ... | DPO ... | a dataset of prompts with a preferred and dispreferred ... | one frozen reference policy $\pi_{ref}$ ... | trl ... | `ipo.md` |
| `KTO` | pick KTO when feedback arrives as a per-output binary label (thumbs-up/down, pass/fail) rather than as pairs ... | DPO ... | one prompt-completion pair per example plus a boolean ... | one frozen reference model, forward passes only ... | trl ... | `kto.md` |
| `LAB` | pick LAB when you need to instruction-tune a base model at scale from a defined taxonomy of tasks, without human annotators or a proprietary teacher ... | a taxonomy-guided variant of ... | a taxonomy whose leaf nodes each carry 1-3 manually written ... | no value network, no reward model ... | instructlab-training ... | `lab.md` |
| `LISA` | pick LISA for supervised or continued-pretraining fine-tuning of a single model on labeled or plain-text data when you want full-parameter-level ... | not a formal variant of LoRA — ... | any labeled fine-tuning set ... | none | axolotl ... | `lisa.md` |
| `LOMO` | pick LOMO when the constraint is GPU memory for full-parameter fine-tuning of an LLM and Adam-style optimizer state ... | SGD ... | whatever labeled or instruction-formatted fine-tuning data ... | none | `transformers` `Trainer` ... | `lomo.md` |
| `LORA` | pick LoRA whenever full fine-tuning's memory or storage cost is the blocker and the task can tolerate a bounded, low-rank weight update ... | standard (full) fine-tuning ... | whatever the wrapped objective needs ... | none required by LoRA's own definition ... | Hugging Face PEFT ... | `lora.md` |
| `MiniLLM` | pick MiniLLM when the student is too small to cover the teacher's whole distribution and forward-KLD distillation makes it put weight on regions the teacher never visits ... | word-level (forward-KLD) knowledge distillation ... | prompts plus a fixed instruction-following corpus, with the student sampling its own responses during training ... | one frozen teacher, plus the student's own earlier policy for the policy-gradient correction ... | trl, `trl.experimental.minillm.MiniLLMTrainer` with `MiniLLMConfig`; the experimental namespace may change without deprecation ... | `minillm.md` |
| `MPO` | offline preference tuning for a multimodal (image-conditioned) reasoning model when you have chosen/rejected response pairs and plain DPO on that ... | DPO ... | preference pairs with a fixed schema — image, question ... | one frozen reference model $\pi_0$ ... | trl ... | `mpo.md` |
| `ORPO` | preference pairs are available but you want to skip both the reward-model stage of RLHF/PPO and the frozen reference-model pass of DPO ... | not a strict derivative of one ... | preference pairs ... | none | LLaMA-Factory, as a stable ... | `orpo.md` |
| `PPO` | online RL when you can afford to train a full second network (the critic) alongside the policy and want a per-token learned baseline rather than a ... | TRPO | prompts (or environment states) plus a way to score each ... | always a value network (critic) — the objective's ... | trl ... | `ppo.md` |
| `PRIME` | pick PRIME over plain outcome-only online RL when you have a cheap, per-response outcome verifier ... | not a single-parent variant — it ... | prompts with a machine-checkable outcome verifier — ... | no value network | verl, as an optional recipe ... | `prime.md` |
| `QLORA` | pick QLoRA over plain LoRA when the base model itself does not fit in available GPU memory at 16-bit ... | LoRA | whatever the outer training objective needs ... | none beyond the base model itself | `transformers` `BitsAndBytesConfig` ... | `qlora.md` |
| `RAFT` | when a reward model (or any scalar scorer) can score fresh completions and you want an alignment loop that only ever runs plain SFT ... | best-of-K (rejection) sampling ... | a prompt set only ... | a reward model to score completions (required) ... | LMFlow ... | `raft.md` |
| `REMAX` | pick ReMax when responses can be scored with a scalar reward (a reward model or a programmatic scorer) and you want an online ... | REINFORCE ... | a prompt-only dataset from the target distribution ... | no value network ... | verl, via `algorithm.adv_estimator=remax` ... | `remax.md` |
| `RLHF` | pick RLHF when you can collect human comparisons over model outputs (or already have a trained reward model) and can afford an online RL loop that ... | not a variant of a single named ... | three separate datasets in the paper ... | three beyond the policy ... | trl ships a `PPOTrainer` ... | `rlhf.md` |
| `RPO` | offline preference optimization when you already have static chosen/rejected pairs and specifically want a cheap defense against DPO's ... | DPO | an offline preference dataset of prompt, chosen-response ... | one frozen reference model ... | trl, via ... | `rpo.md` |
| `SAPO` | online RL with a per-response scalar reward, in the same setting as GRPO ... | GRPO | the same as GRPO ... | none required by the method's own definition ... | trl (`GRPOTrainer` with `loss_type="sapo"`) ... | `sapo.md` |
| `SDFT` | expert demonstrations exist but no scalar reward does ... | on-policy distillation | per example ... | the definition always needs a second scored ... | trl ... | `sdft.md` |
| `SeqKD` | pick SeqKD when you want the cheapest useful distillation: run the teacher once to generate sequences, then train the student on them with ordinary supervised fine-tuning ... | knowledge distillation (Hinton et al., 2015), specialized to sequence-level targets ... | prompts, plus the sequences the teacher generated for them; no reward signal and no preference pairs ... | one teacher, used offline for generation only, so it is not held in memory during training ... | trl, only as `trl.experimental.gkd.GKDTrainer` with `GKDConfig(seq_kd=True, lmbda=0.0)` ... | `seqkd.md` |
| `SFT` | pick SFT whenever you have (or can curate) demonstrations of the exact behavior you want and no further preference or reward signal is available or ... | none in the RL-method sense ... | (instruction, optional input, output) pairs ... | none | trl, `SFTTrainer` / `SFTConfig` ... | `sft.md` |
| `SIMPO` | offline preference optimization on chosen/rejected pairs when you want to drop the reference model entirely for memory and compute savings ... | DPO | a preference dataset of ... | none by design — no reference model, no value network | trl ... | `simpo.md` |
| `SLIC` | pick SLiC-HF when you already have, or can build, preference triples ... | SLiC | preference triples (x, y+, y-), plus SFT pairs ... | none required by the method's own objective ... | trl ... | `slic.md` |
| `TTRL` | pick TTRL when you have a batch of unlabeled prompts (often literally a benchmark's test split) that you can sample repeatedly ... | not a new RL algorithm but a new ... | prompts only ... | none beyond whatever the underlying RL algorithm ... | the PRIME-RL/TTRL GitHub repository ... | `ttrl.md` |
| `VERA` | pick VeRA over its parent LoRA when many task- or user-specific adapters must coexist in memory or be stored cheaply ... | LoRA | whatever supervised data the underlying fine-tuning task ... | none | `peft` (Hugging Face) exports `VeraConfig` ... | `vera.md` |
| `WPO` | pick WPO over plain DPO when you must train on off-policy preference pairs (data labeled on outputs from other models, e.g | DPO | a preference dataset of prompt, chosen-response ... | none required by the method's own definition ... | trl, as `use_weighting` ... | `wpo.md` |

Rules for this table:

- **The when-to-pick cell is one sentence in colon form** - what it is, then when to pick it. It is compressed from the card's own verdict, and every cell in this row is a pointer, not the answer: the card carries the math, the real defaults, and the failure modes.
- **`Variant of` is a column, never a directory.** Grouping methods into families is a judgement, and judgements change. A column changes in one line; a directory change moves files and breaks every path that pointed at them.
- **`Shipped by` is what earns a place in this table.** A method nobody implements is not dropped and not hidden - it keeps a card and a row in `references/index.md`, saying what building it would take. It is kept out of this table only because a reader choosing a method to run today cannot use it.
- A cell here can be too short to act on. Two of these methods ship as a setting on another method's trainer, not as a trainer of their own, and the card is the only place that says so.

## Everything else that was found

`references/index.md` carries all 111 methods in two groups: the 33 implemented ones repeated in full, and the 78 with no implementation found. Read it when the table above has no fit, or to check whether a name that came up elsewhere is known. `references/index.json` holds the same rows for matching by field.

## The cards

`references/<method>.md`, one card per method. A card is a report: numbered markers in the text resolved by a Sources list at the end, every named method citing its defining paper, every claim carrying a URL and a commit or a fetch date. Read the card before you write a config - the paper's value and the library's default are usually not the same number, and the card prints both.

## Scope

- Methods whose title or abstract carries post-training wording, plus every method cited in the algorithm source of a library in the `framework` skill. The second route is what catches a method introduced inside a model report, whose abstract may never use the field's words at all.
- A name earned a card only when the paper that introduced it could be picked by comparing candidates or by matching a title, and when a person then read it and confirmed it is a post-training method for a language model. The same search words also return quantization, unlearning, prompt tuning, benchmarks, and robot control.
- **This list is not saturated.** The search produced 3,984 distinct names; capture-recapture across two channels estimates about 5,838 exist, so about 61 percent of the field was reached. A method that is not here may still exist.
- Nothing comes from our own runs.
