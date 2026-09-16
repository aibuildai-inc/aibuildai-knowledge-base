# DART

https://arxiv.org/abs/2407.13690

Rejection-sampling data synthesis where each math query keeps getting sampled, not for a fixed count, but until it accumulates a target number of correct responses set by its own difficulty, so hard queries end up with as many or more training examples than easy ones.

**DART** (Difficulty-Aware Rejection Tuning) is a data-synthesis method for supervised fine-tuning on mathematical problem-solving, introduced by Tong et al., whose own analysis of existing rejection-sampling datasets found that most correct-response coverage concentrates on easier queries, so much so that the hardest queries frequently get no correct response at all under a fixed per-query sampling budget [1]. Its parent is rejection sampling fine-tuning (RFT), the method Yuan et al. propose for building augmented SFT data by sampling a model's own outputs and keeping only the ones whose final answers check out [2]; DART's paper calls the standard, fixed-trials-per-query version of this "vanilla rejection tuning" (VRT) [1]. Vanilla rejection sampling draws a fixed number of candidate responses per query and filters by final-answer correctness [1]. DART instead uses two difficulty-aware sampling strategies, DARS-Uniform and DARS-Prop2Diff, that keep sampling per query until a target number of *correct* responses is reached, where that target is either constant (Uniform) or scaled to the query's measured difficulty (Prop2Diff) [1]. The paper gives two reasons: difficult queries are hypothesized to be crucial for learning complex reasoning, and prior work suggests difficult training samples are more effective for building model capability, citing Sorscher et al. and a 2024 ICLR submission by Liu et al. [1].

The method has no adopting downstream framework beyond its own release: the paper's own DART-Math datasets (DART-Math-Uniform, DART-Math-Hard) were built with it and published to a Hugging Face collection alongside the code [1][3]. On the paper's own comparison, DART-Math-Llama3-8B (Prop2Diff) beats the VRT baseline trained on the same 590k-query pool by 4.5 average points across 6 benchmarks, and DART-Math-DSMath-7B (Uniform and Prop2Diff) each land within 0.1 average points of DeepSeekMath-7B-RL, a GRPO-trained RL model, using SFT alone (Table 7: 49.2 and 49.4 respectively versus 49.3) [1]. Lineage: rejection sampling fine-tuning, RFT (2023) [2] -> DART (DARS-Uniform / DARS-Prop2Diff, 2024) [1] -> named successors checked for in the sources read for this card: none found (see While it runs for the search performed). The paper was confirmed via its own arXiv abs page for 2407.13690, cross-checked against the GitHub repo's identical title and abstract [1][3].

**When to pick it**: pick DART over plain rejection sampling / RFT [2] whenever the correct-response yield differs a lot by query difficulty, since a fixed per-query trial budget under-samples exactly the hard queries needed for reasoning gains, and DART's own ablation shows this data-distribution fix outperforming a same-size VRT baseline by up to ~7 points on MATH [1]. It is an offline, SFT-only method: all training data is generated once, up front, from a fixed policy snapshot (DeepSeekMath-7B-RL in the paper) and then used for standard supervised fine-tuning, unlike online RL training, which resamples from the current policy during the run [1]. The paper's own nearest-online-neighbor comparison is against GRPO [4], the RL method it states was used to train DeepSeekMath-7B-RL [1]: sole SFT with DART-Math reaches accuracy within 0.1 average points of that GRPO-trained model on the same base (Table 7: 49.2/49.4 vs. 49.3) [1].

**Variant of**: rejection sampling fine-tuning (RFT) [2], specifically the "vanilla rejection tuning" baseline defined and used for comparison in the DART paper [1].

**Data it needs**: a set of math queries with ground-truth final answers (the paper fixes GSM8K and MATH training queries) [1], plus a generator model to sample candidate responses (DeepSeekMath-7B-RL in the paper, an open-weight 7B model, explicitly avoiding GPT-4) [1]. No preference pairs and no reward model beyond exact final-answer matching against ground truth [1]. Training scale in the paper: the two released datasets are about 591k (DART-Math-Uniform) and about 585k (DART-Math-Hard, from DARS-Prop2Diff) query-response pairs [1][3]. Offline: all responses are synthesized before any fine-tuning starts; the sampling policy is not updated during data generation or training [1].

**Extra models**: none beyond the response generator used at data-synthesis time (frozen, inference-only, no gradients) [1]. No value network, no separate reward model (correctness is checked by answer matching), and no reference model, since the fine-tuning step afterward is ordinary supervised learning [1].

**Shipped by**: no RL or SFT training library (trl, verl, axolotl) exposes a `DARS`/DART trainer. The reference implementation is the authors' own `dart-math` package on GitHub (`hkust-nlp/dart-math`, pip-installable via `pip install -e "."`), whose `dart_math.gen` module implements the DARS stopping rule as `is_dp_dars_finished`, checked against per-query `max_n_trials` and `min_n_corrects` counters [3][5]. The repo is not archived but its last push was 2024-12-10 [3]. Building DART on top of another framework means writing this per-query stopping loop around a generation-and-grading pipeline plus a subsequent ordinary SFT job - not a loss added to an existing RL trainer.

## How it works

The loop, per query: keep sampling responses from a fixed generator model and grading each by final-answer match against ground truth, until a per-query target number of *correct* responses is hit or a trial cap is reached; the accumulated correct responses become that query's SFT training examples [1].

**Difficulty measure - fail rate** [1]: for a query, sample $n_d$ responses from the generator and compute

$$ \text{fail rate} = \frac{\#\text{ of wrong responses}}{\#\text{ of all raw responses}} $$

harder queries have a higher fail rate because the generator less often produces a correct answer [1]. The paper reuses the responses gathered while computing fail rate as synthetic training data, merging DARS-Uniform synthesis (at $k_u=192$) with fail-rate calculation for efficiency [1].

**DARS-Uniform** [1]: for every query, keep sampling until $k_u$ correct responses have been collected, where $k_u$ is a fixed hyperparameter set by the desired final dataset size (the paper uses $k_u=40$ to build DART-Math-Uniform) [1].

**DARS-Prop2Diff** [1]: for every query, keep sampling until the number of correct responses is (linearly) proportional to its difficulty score (fail rate), with the hardest queries receiving $k_p$ responses ($k_p=192$ in the paper, to build DART-Math-Hard) [1]. This is a deliberate bias in the opposite direction from vanilla rejection sampling, towards harder queries [1]. Because Prop2Diff can assign zero target responses to easy queries when the dataset is small, the paper enforces at least one correct response per easy query in practice ("+Cover") [1].

Worked example (Uniform, $k_u=3$): a query has fail rate 0.75 under the generator, so on average 1 in 4 sampled responses is correct. Sampling continues, drawing responses one at a time and grading each against the ground-truth answer, until 3 correct ones have been collected - on average this takes about 12 raw samples for this query, versus far fewer for an easy query with fail rate near 0, but under DARS-Uniform both queries still end up contributing exactly 3 correct training examples each, unlike vanilla rejection sampling with a fixed trial count, where the easy query would typically end up with more correct responses than the hard one [1].

Both DART-Math datasets additionally include the original GSM8K and MATH training queries and answers alongside the synthetic responses [1].

## Cost

**Theory, from the method's own definition:**

- Time: sampling cost per query is driven directly by its fail rate - a query needing $k$ correct responses at fail rate $f$ needs on the order of $k/(1-f)$ raw generation calls, so DARS-Prop2Diff, which assigns the largest $k_p$ to the hardest (highest-$f$) queries, concentrates generation cost precisely where it is most expensive per correct sample [1]. The paper reports that reaching a 90% per-query coverage target needs $n_{\max}\ge 2048$ trials/query and ~5M raw samples under DARS-Uniform, versus $n_{\max}\ge 8000$ and >15M raw samples under DARS-Prop2Diff, for the same target dataset [1].
- Memory: none beyond running one generator model at inference (no gradients, no optimizer state) during synthesis, and one model in ordinary SFT afterward [1]. There is no extra trained model versus plain SFT - no value network, reward model, or reference model are part of the method's definition [1].

**In practice, per implementation:**

- `dart-math` reference repo [3][5]: response generation is done with vLLM (`dart_math.gen.Generator`, `pipeline/gen.py`), and per-query stopping is tracked with `max_n_trials` and `min_n_corrects` counters checked by `is_dp_dars_finished` [5]. The subsequent SFT step uses the HuggingFace `Trainer` with sequence packing, which the README reports gives a 6-8x training speedup in the authors' setting versus unpacked sequences [3]. No other shipping framework's cost behavior applies here since none implement DARS.

## How to use it

- Data preparation: fix a query set with verifiable ground-truth final answers (GSM8K + MATH in the paper) [1]; pick a generator model capable of producing some correct responses on most queries - the paper's ablation shows even a 7B open-weight model (DeepSeekMath-7B-RL) can synthesize correct responses for most queries, avoiding reliance on GPT-4 [1].
- Correctness/reward convention: a response counts as correct only if its final answer string-matches the ground-truth answer; no partial credit and no learned reward model [1]. The repo's evaluation grading uses the same final-answer matching, with a stricter/looser answer-extraction mode toggle [1][5].
- Key knobs, paper's own values as anchors (no other implementation ships different defaults; "not checked" = this card did not verify that setting elsewhere):

| knob | paper (DART-Math datasets) [1] |
| --- | --- |
| target correct/query, Uniform ($k_u$) | 40 |
| target correct/query at hardest, Prop2Diff ($k_p$) | 192 |
| DARS-Uniform target used for the merged fail-rate computation ($k_u$) | 192 (not-stated: the paper never gives a numeric value for the fail-rate metric's own $n_d$; in practice it reuses a DARS-Uniform run at $k_u=192$ for efficiency) |
| trial cap ($n_{\max}$) for ~90% coverage, Uniform | 2048 |
| trial cap ($n_{\max}$) for ~90% coverage, Prop2Diff | 8000 |
| SFT epochs (default / Llama3 / MMIQC baseline) | 3 / 1 / 1 |
| SFT learning rate (Mistral-7B / DSMath-7B & Llama3-8B / Llama3-70B) | 1e-5 / 5e-5 / 2e-5 |
| SFT batch size (packed sequences, length 4096) | 64 |

- Trade-off: raising $k_u$ or $k_p$ raises both per-query training signal and synthesis cost, and for Prop2Diff the cost is concentrated on the hardest queries, so scaling $k_p$ up scales the "trial cap needed for coverage" the fastest for exactly those queries [1]. Uniform and Prop2Diff datasets in the paper end up close in overall size (~591k vs ~585k) despite very different per-query allocation, because Prop2Diff spends far more trials per hard query but fewer per easy query [1][3].

## While it runs

- Signals and their healthy shapes: the paper's own diagnostic is the per-query response-count distribution versus difficulty (fail rate) - a healthy DARS-Prop2Diff run produces monotonically more responses for harder queries with no bias toward easy ones, in contrast to VRT's flat allocation [1]. Fail rate itself is the difficulty signal to track during generation: a query stuck near fail rate 1.0 after the trial cap contributes zero or one correct response even under Prop2Diff's Cover rule, indicating the trial cap is too low for that difficulty tier [1].
- Published reference runs: the paper's own Table 2 is the reference comparison - DART-Math-Llama3-8B (Uniform) beats the VRT baseline on all 6 benchmarks by 3.5 points on average, and (Prop2Diff) by 4.5, both trained on the same 590k-query pool [1]; a Prop2Diff run that does not clear the VRT baseline by a comparable margin under the same base model and dataset size indicates the synthesis or training setup diverged from the paper's.
- Degeneracies and defaults: without the "+Cover" rule enforcing at least one correct response per easy query, DARS-Prop2Diff can leave some easy queries with zero synthetic responses when the target dataset is small, which the paper's own ablation (Figure 4) shows hurts performance relative to the covered version [1]; this is a documented default divergence to check for, not an inferred one.
- Named successors: checked only the paper's own GitHub repo (README and issues) for mentions of methods building on DARS; none found there as of 2026-08-09 [3]. No broader citation search was performed for this card, so this is not a claim that no successor exists elsewhere.
- Known failure modes: the paper's own limitations section states fail rate may be a sub-optimal difficulty metric, and names direct scoring, Elo ratings, or minimum-pretraining-compute estimates as unexplored alternatives, without saying the scoring alternative is LLM-based [1]. That same section separately notes DART-Math is limited to natural-language reasoning and does not generate or execute code, unlike the methods it cites for code-assisted math [1]; the paper's Appendix A, not the limitations section, is where it specifically names ToRA and MARIO and excludes them from direct comparison on the grounds that they solve math problems with code in addition to natural language, which the paper treats as a different problem setting from DART-Math's [1]. Searching both open and closed `hkust-nlp/dart-math` GitHub issues (5 open, 5 closed as of 2026-08-09) surfaces one maintainer-reported numerical trap: in closed issue #3, maintainer tongyx361 explains that an earlier commit set `ignore_eos=True` because Llama-3-8B(-Base) tends to decode an EOS token prematurely on some inputs, and forcing that flag on broke generation for normal instruct models, so a later commit exposed `ignore_eos` as a CLI option defaulting to `False` [6].
- What the gain is - and is not: the paper's own headline result is that difficulty-aware allocation of synthetic correct responses closes most of the gap to RL (DeepSeekMath-7B-RL) using SFT alone on a smaller, open-weight-only dataset (Table 7) [1]; it does not claim DART adds reasoning capability beyond what rejection sampling from the generator model can already produce - it only fixes which queries the correct samples come from, and the paper explicitly finds a limited effect on the easy, in-domain GSM8K benchmark, where vanilla rejection tuning was not badly biased to begin with [1].

## Sources

[1] Tong, Zhang, Wang, Wu, He, "DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving", 2024. https://arxiv.org/abs/2407.13690 - defines DART/DARS-Uniform/DARS-Prop2Diff, fail-rate metric, main results (Table 2), RL comparison (Table 7), coverage ablation, limitations. The unversioned URL resolves to v2 (23 Dec 2024 UTC, 234 KB per the page's own submission history, superseding v1 of 18 Jun 2024, 134 KB); the HTML full text fetched for this card is v2. Fetched 2026-08-09 (HTML full text and abstract page).

[2] Yuan, Yuan, Li, Dong, Tan, Zhou, "Scaling Relationship on Learning Mathematical Reasoning with Large Language Models", 2023. https://arxiv.org/abs/2308.01825 - defines Rejection sampling Fine-Tuning (RFT), the parent method. Fetched 2026-08-09 (abstract page).

[3] hkust-nlp/dart-math GitHub repository README and repo metadata. https://github.com/hkust-nlp/dart-math - dataset sizes, NeurIPS 2024 acceptance, install instructions, training pipeline, repo activity. Fetched 2026-08-09 (raw README at `main`, and GitHub API repo metadata at `main`, pushed_at 2024-12-10).

[4] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. https://arxiv.org/abs/2402.03300 - defines GRPO, the RL method used to train DeepSeekMath-7B-RL, the model DART is compared against in [1] Table 7 and used as DART's own data-synthesis generator. Not independently fetched for this card; GRPO's identity and DeepSeekMath-7B-RL's training method are as stated in [1].

[5] `dart_math/gen.py` in hkust-nlp/dart-math. https://raw.githubusercontent.com/hkust-nlp/dart-math/main/dart_math/gen.py - `is_dp_dars_finished` stopping-rule implementation (`max_n_trials`, `min_n_corrects`), `Generator` class using vLLM. Fetched 2026-08-09 at commit reachable via the `main` branch (unpinned, mutable ref).

[6] hkust-nlp/dart-math GitHub Issues (open and closed), via the GitHub REST API. https://github.com/hkust-nlp/dart-math/issues - issue #3 and maintainer tongyx361's comment explaining the `ignore_eos` bug and its fix. Fetched 2026-08-09 (`state=closed` and `state=open` issue lists, and comments on issue #3; unpinned, mutable endpoint - reflects the issue tracker's state at fetch time).
