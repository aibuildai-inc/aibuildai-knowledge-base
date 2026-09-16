# HIR

A two-phase offline supervised method: sample outputs from the model at a high temperature, then relabel each prompt in hindsight to match whatever the model actually produced, and fine-tune on the relabeled pairs with an added contrastive loss and entropy bonus - no reward model, no value network, no PPO loop.

**HIR** (Hindsight Instruction Relabeling) is a method for aligning language models with instructions, introduced by Zhang, Liu, Wong, Abbeel and Gonzalez as an alternative to RLHF that turns feedback into a rewritten instruction and trains on the rewritten pair with ordinary supervised learning, using only the parameters of the base model that is being tuned [1]. Its parent is Hindsight Experience Replay (HER), which trains on a failed trajectory by swapping in, as the replayed goal, the state the trajectory actually reached rather than the goal it originally aimed for, turning what looked like a reward-free failure into a labeled success for a different goal [2]; HIR adopts the relabeling strategy in HER [1]. HIR casts instruction-following as goal-conditioned RL, with the instruction as the goal, the query and generated tokens as state and action, and the language model doubling as both policy and world model [1]. It alternates an online sampling phase, which generates instruction-output pairs at temperature $\tau=1$, with an offline phase that rewrites the instruction to match each output and applies standard sequence-to-sequence training, plus a contrastive instruction-following loss and an entropy-regularization term [1]. The paper gives two reasons beyond simplicity: PPO-based RLHF is complex, sensitive to hyperparameters, and needs an extra reward-and-value-network training pipeline [1]; and Final-Answer RL (FARL), which imitation-learns only on correct outputs, throws away every failed sample, while HIR's relabeling keeps and uses the failures too [1].

This card checked only one implementation, the paper's own GitHub repository (`tianjunz/HIR`, commit `b5985874`), which is a research-code release for reproducing the paper's BigBench experiments, not a packaged trainer [3]; no search for other adopters was carried out, so no claim is made here about whether any other system uses HIR. On 12 BigBench reasoning tasks with FLAN-T5-large, the paper's introduction reports HIR outperforming its PPO and FARL baselines by 11.2 and 32.6 points respectively [1, Sec. 1]; the paper's Table 2 breaks this down per task and its own discussion of that table highlights Tracking Shuffled Objects (5) and (7), where HIR surpasses the best baseline by 41.2 and 29.2 points respectively [1, Sec. 6.1, Table 2]. Lineage in one line: HER (2017 [2]) -> HIR (2023 [1]); of the 72 papers the Semantic Scholar API lists as citing arXiv:2302.05206, none has a title naming HIR or claiming a fix to a documented HIR bias [4]. This card covers arXiv:2302.05206; a distinct, later paper also uses the name "Hindsight instruction Replay" (arXiv:2512.23457) but does not cite or mention this paper, its authors, or its title anywhere in its text [5], and is not covered here.

**When to pick it**: offline supervised fine-tuning when you can score generations for correctness (a scripted checker, not necessarily a trained reward model) and want to reuse failed generations instead of discarding them, without standing up a value network or reward model [1]. Prefer PPO/RLHF [6] when a learned reward model captures preferences a scripted checker cannot, and you can afford the extra reward-and-value training pipeline [1]. Prefer FARL [7] - imitation learning on only the correct outputs - when hindsight-relabeling failures is undesirable or infeasible; HIR's own comparison shows FARL is the closer offline neighbor, differing from HIR only in whether failure data is relabeled and used rather than discarded [1].

**Variant of**: HER [2], adapted from goal-conditioned robotic RL to language-model instruction following [1].

**Data it needs**: a set of queries (no instruction needed at start - the paper initializes the instruction to a fixed placeholder like "Generate a correct answer to this problem" and lets the offline phase relabel it [1, Sec. 4.2, A.2]); a scripted or learned feedback function `R(p,q,o)` that scores an output as correct/incorrect (the paper's BigBench experiments use a binary final-answer checker [1, Sec. 4.2, A.2]); and an instruction-generation function `φ` that maps a query, output, and score to a new instruction (scripted in the paper's experiments as swapping between two fixed instruction strings [1, Sec. A.2]). No chosen/rejected pairs and no pre-collected offline dataset are needed at the start of training. The method is semi-online: each of the paper's N episodes alternates T rounds of fresh online sampling from the current policy with K rounds of offline supervised training on the resulting relabeled data (Algorithm 1) [1] - it is not a single offline pass over a static dataset, but training itself is the ordinary supervised seq2seq loss, not a policy-gradient update. Training scale in the paper: FLAN-T5-base and FLAN-T5-large on 12 individual BigBench reasoning tasks, each split 80/20 train/test [1, Sec. 6].

**Extra models**: none. The only trained parameters are the base language model's own weights [1] - no value network, no reward model beyond the scripted checker, and no separate reference model for a KL term: the paper contrasts HIR with standard PPO by noting that PPO adds a KL-divergence penalty against a reference policy, which HIR's supervised objective does not include [1, Sec. 4.3]. No shipping framework was found to load any HIR-specific model at all (see Shipped by).

**Shipped by**: no library implementation was found in the searches this card ran. A keyword search of trl's and verl's documentation index pages for "hindsight" returned no matches (trl `main` docs index, verl `latest` docs index, checked 2026-08-09) [8][9]; this checked the docs index pages only, not either project's source tree or full trainer/config list, so it rules out a documented "hindsight" trainer but not an unindexed one. The paper's own repository, `tianjunz/HIR`, is unpackaged research code built around BigBench task scripts and a `run.sh` training loop, not a reusable trainer [3]. Building HIR on top of an existing trainer would mean writing a custom training loop (the online-sampling/offline-relabeling alternation of Algorithm 1) plus the relabeling and contrastive-loss logic, since that alternation is not what a standard PPO- or DPO-style trainer's loss function provides.

## How it works

Each episode: sample T rounds of instruction-output pairs from the current policy at temperature $\tau$ to build an online replay set $D_{online}$; then run K rounds of offline training that relabel a sampled batch from $D_{online}$ and update the model with a combined seq2seq, contrastive, and entropy loss; repeat for N episodes (Algorithm 1) [1].

**Relabeling.** For a pair $(p, q, o)$ that may not be aligned, HIR rewrites the instruction to $p^*$ using $\phi(p, q, o, R(p,q,o))$, so the pair $(p^*, q, o)$ is aligned by construction [1, Sec. 4.2]. HIR additionally applies sub-output relabeling: at an intermediate timestep $i$, the instruction is edited as a future goal based on the future portion of the output, $p^* = \phi\big(p, q, \{e_i,\dots,e_L\}, R(p,q,\{e_i,\dots,e_L\})\big)$, and the model is trained to predict $\{e_i,\dots,e_L\}$ given $(p^*, q, \{e_0,\dots,e_{i-1}\})$ with the standard seq2seq loss $\mathcal{L}_{supervise}$ [1, Sec. 4.2]. On the BigBench tasks the paper's scripted feedback and relabeling functions are binary: $R(o,p,q)=1$ if $o$ is correct and $p=p_{correct}$, $=1$ if $o$ is wrong and $p=p_{wrong}$, $=0$ otherwise; and $\phi(o,p,q,r)=p$ if $r=1$ else $\lnot p$, where $p_{correct}$ and $p_{wrong}$ are two fixed instruction strings and $\lnot p_{correct}=p_{wrong}$ [1, Sec. A.2].

**Contrastive instruction loss.** With $P_{ik} = \log P_M(o_i \mid q_k, p_k)$ the log-probability of output $i$ conditioned on the $k$-th query/instruction pair in the batch, the loss is

$$ \mathcal{L}_{contrastive} = -\sum_{i=1}^{n} \log \frac{\exp(P_{ii})}{\sum_{k=1}^{n} \exp(P_{ik})} $$

which pushes up the probability of output $i$ under its own (relabeled) instruction and pushes it down under every other instruction in the batch, discouraging the model from mapping the same output to different instructions [1, Eq. 4].

**Entropy regularization.** $\mathcal{L}_{entropy} = \sum_{k=1}^{n} P_k \log P_k$ is applied to the output distribution given an instruction, to keep the online sampling phase from converging too early and losing exploration [1, Eq. 5].

**Final loss:** $\mathcal{L}_{final} = \mathcal{L}_{supervise} + \alpha\, \mathcal{L}_{contrastive} + \beta\, \mathcal{L}_{entropy}$, with $\alpha, \beta$ as tunable coefficients [1, Eq. 6]; the paper's run uses $\alpha=1$ (contrastive) and $\beta=0.001$ (entropy) [1, Table 7].

**Worked micro-example.** Query $q$ = "I have a blackberry, a clarinet, a nectarine..."; the placeholder instruction starts as $p_{correct}$ = "Generate a correct answer to this problem." The model samples an output $o$ = "6", which the scripted checker finds correct, so $R=1$ and $\phi$ leaves the instruction as $p_{correct}$ - the pair $(p_{correct}, q, o=\text{``6''})$ is used as a positive supervised example. Had the model instead sampled a wrong count, $R=0$ under $p_{correct}$ would relabel the pair to $(p_{wrong}, q, o)$ - the same failed generation, now supervised as a correct example of "generate a wrong answer," so no data is discarded [1, Sec. A.2].

## Cost

**Theory, from the method's own math:** no value network and no reward-model forward pass are required beyond a scripted checker, so per the paper's own comparison HIR trains and holds only the policy model, versus PPO/RLHF's policy plus value network plus (typically) a learned reward model [1]. Online sampling still requires generating T rounds of completions per episode from the current policy before every K rounds of offline training (Algorithm 1) [1] - a naive reading says generation cost scales with $T \times$ online-samples-per-round $\times$ completion length, same shape as any on-policy RL method's rollout cost, even though the training update itself is ordinary supervised backprop rather than a policy-gradient step.

**In practice, per framework:** no shipping framework was found (see Shipped by), so no framework-level memory or throughput figures exist to report. The paper's own experiments (FLAN-T5-base/large, single-task BigBench runs) do not report wall-clock or per-run memory figures [1].

## How to use it

- Prompts/queries: from the target task's training split; the paper used an 80/20 train/test split per BigBench task and drew random batches of queries as prompts at each online-sampling round [1, Sec. 6].
- Instruction and feedback: the paper's own convention on BigBench is a binary scripted checker plus two fixed instruction strings ($p_{correct}$, $p_{wrong}$) that the relabeling function swaps between based on correctness [1, Sec. A.2] - a different task would need its own feedback function $R$ and instruction-generation function $\phi$, which the paper allows to be learned or scripted [1, Sec. 4.2].
- Key knobs, from the paper's own hyperparameter table (no framework ships this method, so there is no separate framework-default column):

| knob | paper value (Table 7) [1] |
| --- | --- |
| online samples per iteration | 4 |
| sampling temperature $\tau$ | 1.0 |
| learning rate | 0.0005 |
| train batch size | 64 |
| train epochs per iteration | 10 |
| label smoothing | 0.2 |
| entropy coefficient $\beta$ | 0.001 |
| contrastive loss coefficient $\alpha$ | 1 |

  The paper notes Final-Answer RL was run with the exact same hyperparameters as HIR to isolate the effect of relabeling versus filtering [1, Sec. A.1].
- Trade-off: sub-output relabeling (relabeling at intermediate timesteps, not just the full output) gives denser feedback but is itself an ablated component - the paper's ablation (Table 4) shows removing sub-output sampling lowers Logical Deduction (3 Objects) from 91.7 to 75.0 while leaving Tracking Shuffled Objects (3) unchanged at 100.0, so its effect on the target task is not uniform across BigBench tasks [1, Sec. 6.3, Table 4].

## While it runs

- Signals and their healthy shapes: no first-hand monitoring guidance (metric names, logged quantities, or "healthy shape" language analogous to trl's GRPO logging guide) was found in the paper or its repository; the paper reports only final task accuracy per BigBench task, not training curves [1].
- Published reference runs: Table 2 (FLAN-T5-large, single-task) and Table 3 (FLAN-T5-base vs. large) in the paper give final BigBench accuracies for HIR against No-Training, Fine-Tuning, PPO, and FARL baselines, and are the only published reference numbers found [1, Tables 2-3]; no raw training logs were found published alongside the repository.
- Degeneracies and defaults: the paper's own ablation (Table 4) shows all three components it studies - sub-output sampling, entropy regularization, and label smoothing - are individually load-bearing to some degree, with the largest single drop being sub-output removal on Logical Deduction (3 Objects), 91.7 -> 75.0 [1, Sec. 6.3, Table 4]. No config default was found to diverge between the paper and a framework, because no framework ships this method.
- Named successors: none found. The 72 papers the Semantic Scholar citations API returns for arXiv:2302.05206 (fetched in full in one query, titles and IDs only) include no title naming HIR or claiming to fix a documented bias in it [4] - contrast with GRPO's DAPO/Dr. GRPO successors, which do make that claim against GRPO.
- Known failure modes: the paper itself reports that on direct-generation tasks (Object Counting, Word Sorting) HIR's performance is not comparable to fine-tuning, because fine-tuning has access to the correct answer directly while HIR only performs final-answer checking [1, Sec. 6.1]. This card did not fetch the repository's issue tracker, so no claim is made here about maintainer-reported failure modes; the README and repository homepage that were fetched contain no discussion of failure modes at all [3].
- What the gain is - and is not: the paper reports HIR reaching accuracy on par with, and on several BigBench tasks above, models fine-tuned on the ground-truth answer, while HIR itself only ever sees a binary correct/incorrect signal [1] - the gain is closing most of the fine-tuning gap under much weaker supervision, not a claim that HIR adds reasoning capability beyond what a correctness label can teach; the paper explicitly notes direct-generation tasks remain harder for HIR than for full fine-tuning [1, Sec. 6.1].

## Sources

In-text markers [n] refer to this list, numbered by first appearance.

[1] Zhang, Liu, Wong, Abbeel, Gonzalez, "The Wisdom of Hindsight Makes Language Models Better Instruction Followers", 2023. https://arxiv.org/abs/2302.05206 - defines HIR: goal-conditioned RL formulation, Algorithm 1, relabeling, contrastive and entropy losses, hyperparameters, BigBench results and ablations. Fetched 2026-08-09 (PDF, converted to text with pdftotext).

[2] Andrychowicz et al., "Hindsight Experience Replay", 2017. https://arxiv.org/abs/1707.01495 - HER, the parent method: the goal-relabeling mechanism (replaying a failed trajectory with the state it actually reached in place of its original goal). Abstract page fetched 2026-08-09; full text fetched 2026-08-09 (PDF, converted to text with pdftotext) to ground the relabeling-mechanism claim, since the abstract alone does not describe it.

[3] tianjunz/HIR GitHub repository (the paper's own implementation). https://github.com/tianjunz/HIR - unpackaged research code for BigBench task reproduction; README fetched via raw.githubusercontent.com does not carry a commit pin, but the repository homepage fetched the same day shows HEAD at commit `b5985874c24ac0dfd2d24a62192daffd11dd3731`, and all claims about the repository's contents on this card are pinned to that commit. No issue-tracker discussion of failure modes was fetched or checked. Fetched 2026-08-09.

[4] Semantic Scholar Graph API, citations endpoint for paper arXiv:2302.05206 (`https://api.semanticscholar.org/graph/v1/paper/arXiv:2302.05206/citations?fields=title,year,externalIds&limit=100`), a live, offset/limit-based endpoint, not a paginated search UI. Fetched 2026-08-09; returned all 72 citing papers known to the endpoint at that time in a single call (`next` field empty), titles and external IDs only, no abstracts read.

[5] Title and full text of arXiv:2512.23457, "Replay Failures as Successes: Sample-Efficient Reinforcement Learning for Instruction Following" - checked for any mention of "Wisdom of Hindsight", "Gonzalez", "Abbeel", or "2302.05206"; none found. Fetched 2026-08-09 (PDF, converted to text with pdftotext).

[6] Schulman, Wolski, Dhariwal, Radford, Klimov, "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347 - PPO, the RLHF-side baseline HIR is compared against in [1]. Cited via [1]; not separately fetched.

[7] Uesato et al., "Solving math word problems with process- and outcome-based feedback", 2022. https://arxiv.org/abs/2211.14275 - Final-Answer RL (FARL), the offline baseline HIR is compared against in [1]. Title confirmed by fetching the arXiv abstract page 2026-08-09; full text not fetched, cited via [1]'s description of FARL.

[8] trl documentation index (`main` build). https://huggingface.co/docs/trl/main/en/index - checked for "hindsight"; no match. Fetched 2026-08-09.

[9] verl documentation index (`latest` build). https://verl.readthedocs.io/en/latest/index.html - checked for "hindsight"; no match. Fetched 2026-08-09.
