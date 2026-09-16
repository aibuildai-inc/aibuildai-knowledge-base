---
description: >-
  The research flow of one post-training run: thirteen ordered research
  actions, each with the judgment it settles, a worked case, the
  candidate directions open at that point, and the boundary where the
  case must not be copied. 34 reference cards of depth sit behind the
  steps. Consult it when planning a run before any data is chosen,
  again while the run is live to keep the budget honest, and again in
  the final hours before saving. Dataset choice, method choice, and
  trainer choice stay delegated to the dataset, methodology, and
  framework skills; this skill owns the order, the judgments, and the
  budget between them.
---

# Post-training workflow

Read this file whole before the first decision of a run. Return to the step you are in while the run is live. The steps are research actions, not a fixed recipe. Each one names the judgment it settles, walks one concrete case, lists the directions open at that point, and says when that case does not transfer. The actions are not optional: work through every step, and skip one, reorder one, or leave one of its directions untried only for a reason you can state at that decision. Reading a step and then doing none of it is a step done wrong.

The catalog of datasets is the **dataset** skill. The training rule itself, its knobs, and its per-method signals are the **methodology** skill. The trainer that produces the folder is the **framework** skill; its reference file `loading-the-result.md` carries the exact loading recipe. This skill also carries 34 reference cards under `references/`, one per recurring move or failure family. `references/index.md` maps every card to the step that reads it. `references/33-decision-order.md` condenses the same decisions into a one-page priority order.

**Insurance, from the first minutes.** The run can die at any moment, and whatever sits in the delivery folder at that moment is what gets graded. So the folder must hold a model that is proven to load and answer, at all times, from minutes after launch. Before training starts, copy the base model's weight files, tokenizer files, and a deliberate `generation_config.json` into the folder as real files. Replace that model only with a checkpoint you evaluated and found better. Write the replacement in the foreground. After every replacement, start a fresh process, load the folder by path alone, and generate one answer, so the folder stays proven.

## Step 1: understand what the scoring rewards -- and price it

"This is a code task" or "this is a math task" is still too coarse. Settle these before any data is chosen: does the score read the final answer or the whole reply; does a format error score zero on its own; does reasoning earn anything or only add truncation risk; can an answer be checked automatically by execution, by an answer key, or by a schema; is there one score or several dimensions; and do inference-time sampling settings affect the official number.

Then price it. A number you did not produce yourself is not a baseline. Run the shipped evaluator once on the untouched base model, on the full set or on a fixed slice if the full set is too slow. The number it returns is the baseline, and it prices the floor under every later decision. When a baseline is published for the task, reproduce it; that also proves your loop matches the grader's before any later result depends on it.

| Task type | What actually needs attention |
| --- | --- |
| Exact answer | Is the answer correct, and can it be extracted |
| Structured output | Grammar, schema, termination |
| Executable output | Syntax, runtime, tests |
| Rubric-based answer | The balance between several quality dimensions |
| Preference-based output | Overall quality, style, following constraints |

**Case:** suppose the scorer of a code-generation task takes only the first Python fence and cuts it down to the function body, so module-level imports are stripped in the process. The answer is to move the missing imports inside the function body and re-run the training samples the way the real scorer would. Or suppose a function-calling task turns out not to be "can the model use tools": the model already produces correct JSON, but a wrong `<tool_call>` wrapper token still makes the parser record a failure. Or suppose a health-advice task is not exact match at all. It scores accuracy, completeness, communication, and context awareness together, so training medical facts alone need not raise the total. And a run can score almost nothing on a function-calling task after SFT with the shipped sampling defaults, then win nearly the whole task back without touching weights, because the failure lives in the evaluator's own generation path rather than in the model. The same baseline measurement also gives the per-evaluation price the budget uses later: how long one full evaluation takes on this model size, on this hardware.

**Directions:**

- Read the shipped evaluation code, not any prose description of the task.
- Write down the metric, the extraction rule, the token cap, and any retry ladder.
- If several dimensions are scored, write down which ones the base model is already good at.
- If extraction is strict, write down the exact final form a generation must end on.
- Record what the evaluator does to a generation: which span it extracts, what it does on a parse failure, whether it retries with a smaller token cap.
- Record the sampling settings it uses; shipped defaults are tuned for chat variety.
- Time the baseline run, so the price of one full evaluation is a measured number, not a guess.
- Keep a fixed slice aside as the cheap proxy you will use at every checkpoint.

**Do not copy when:** the table is not a permanent label. One task can be several types at once -- a code task can also be judged on style, and a rubric task can also have an extractable field -- so read the scorer instead of picking a row. And a proxy slice built here steers training but never replaces the full number. If the full evaluation later disagrees with the proxy trend, trust the full number, and check the slice and the extraction rule for drift.

Cards: `references/31-per-task-map.md`, `references/02-evaluator-recon.md`, `references/04-chat-template-alignment.md`, `references/05-output-grammar.md`

## Step 2: read the generations and break down the loss

A score hides the failure. Read the generations themselves and answer: did the model understand the task; is the output well formed; where exactly is the answer wrong; does it stop normally; is it truncated; does it fall into a repetition loop; does the same question swing widely across samples; and does the error come from knowledge, from reasoning, from format, or from the inference settings.

Then count. "The model is not capable enough" is not a diagnosis. Estimate what share each kind of error owns, for example 40 percent invalid output, 30 percent hitting the length limit, 10 percent repetition loops, 20 percent finished normally but answered wrong. A model with that profile and a model that is 100 percent well-formed but wrong need completely different training plans.

| Main failure | What to consider first |
| --- | --- |
| Output invalid | Template, format data, decoding |
| Finishes but answers wrong | Knowledge or reasoning data |
| Heavy truncation | Reasoning length and termination |
| Heavy repetition | Concise data, stopping behavior, sampling |
| Execution failures | Execution-filtered data |
| Right tool name, wrong arguments | Argument and schema coverage |
| Target skill up, general ability down | Data mixture and overfitting |

**Case:** picture a run reading function-calling generations that score almost nothing and finding the JSON content mostly correct already. The real failure is the opening special token being sampled into some other reserved token. Reading only the score leads to "not enough training data, keep doing SFT"; reading the text leads to "the model learned the behavior, but the inference settings do not call it out reliably". A run reading generations on a rubric-scored health-advice task might find the base model already has some accuracy but is very weak on completeness and communication, and so train complete, safe, clear answering behavior instead of more medical facts. And a run that counts might find, on a math contest task, that only a minority of items finish normally, most of those are correct, and the rest are truncated. The model is usually right whenever it finishes, so the main loss is not arithmetic -- it is not finishing inside the token budget. The reasonable directions there are shorter correct reasoning, an explicit final-answer line, better termination, and less repeated self-checking -- not more knowledge and not longer chains of thought.

**Directions:**

- Sample a few dozen generations greedily and read them end to end. This reading is not skippable, and no training data is chosen before it is done.
- Separate invalid output from wrong answers before counting anything.
- Generate the same item several times, to see how much of the failure is variance.
- Count the categories on a fixed sample, so the same count can be repeated after training.
- Put the two largest shares in writing.
- If truncation dominates, measure the length distribution against the scoring token cap before choosing anything.
- Keep a handful of these transcripts as the before picture you will compare against after training.

**Do not copy when:** a clean-looking sample set does not mean the model is fine. If the score is low and the text looks reasonable, the extraction rule from step 1 is the next suspect, not the weights. And "consider first" is not "only this": the table orders suspects; it does not decide the fix, and one run's percentages say nothing about your model.

Cards: `references/03-baseline-failure-taxonomy.md`, `references/06-eos-and-length.md`

## Step 3: state a training hypothesis you can prove wrong

Do not write "I will fine-tune the model". Write something a single experiment can refute: the model lacks the target output grammar rather than tool-choice ability; the model can solve the problem but reasons too long; the model knows the medical facts but cannot build a complete answer; the model already produces correct code sometimes, so execution filtering can give on-policy data; the model is simply underfit, so more of the same training may beat new data.

**Case:** when data, template, adapter rank, learning rate, temperature, and an RL stage all change together, a higher score afterwards tells you nothing about which change earned it. The cheap experiments are the ones that hold everything else still. Keep the weights fixed and change only decoding, and see whether the errors disappear. Keep the data fixed and only train longer, and see whether the model really was underfit.

**Directions:**

- Write the hypothesis as a sentence naming one layer: data, supervision target, capacity, schedule, or inference settings.
- Name the single measurement that would refute it.
- Make the next action the cheapest experiment that separates it from its nearest rival explanation.

**Do not copy when:** a hypothesis that no measurement can refute is not a hypothesis. If you cannot say what result would make you drop it, go back to step 2 and get a sharper breakdown first.

Cards: `references/32-conditional-rules.md`

## Step 4: survey data by the job each source does

Do not shop for "the one best dataset for the task". Sort candidate sources by the job they do in the mixture. Two sources with the same topic can play completely different roles.

| Data job | What it fixes |
| --- | --- |
| Domain data | Knowledge gaps and problem-type coverage |
| Format data | Learning the target output structure |
| Concise correct data | Less over-reasoning and truncation |
| Hard data | Raising the ceiling |
| General instruction data | Preventing forgetting after narrow training |
| Multi-turn data | Using context and earlier constraints |
| Verifiable data | Supporting rejection sampling and RL |
| Preference data | Optimizing open-ended quality |

**Case:** a code-generation run might combine three sources with three different jobs: problems that ship reference solutions and tests, code instruction data that has already passed execution filtering, and completions the model itself later generates that pass the tests. A run on a rubric-scored health-advice task might add general chat data, not because it is medically relevant, but to hold instruction following in place. A later version that cuts the general chat for more targeted, more verbose data can watch the total fall.

**Directions:**

- For every source you keep, write the job it does, its licence, where its rows came from, and its contamination risk.
- Inventory whatever data the environment already gives you before choosing, rather than choosing first.
- Treat a checkpoint you trained as a source too; it can both continue training and generate data.

**Do not copy when:** any rules your setting imposes override general practice. An older run's data source can be forbidden even when its experiment logic still holds.

Cards: `references/08-data-portfolio.md`, `references/14-data-mixture.md`, `references/34-distillation.md`

## Step 5: clean, unify, and verify the data

Never hand a whole public dataset to the trainer unread. Open random rows and check: do prompt and response match; is the answer actually correct; is the reasoning complete; what does the length distribution look like; how often does the same problem repeat; are tool schemas uniform; does the code run; do the multi-turn rows really depend on earlier turns; and is the source allowed under any rules your setting imposes.

**Case:** a function-calling corpus can arrive with bare function schemas, an OpenAI-style wrapper, arguments as a string, arguments as an object, single calls, and multiple calls, all mixed. The move is to unify the schema, then check per row that the called function exists, that the arguments parse into an object, that required fields are present, and that only single-call rows matching the task shape survive. A math-contest corpus can repeat the same problems across many reasoning traces. Keeping the shorter correct trace per problem instead of everything can shrink the corpus several times over, with length control and stratified sampling on top.

**Directions:**

- Filter on the per-row quality flags large corpora already carry; keep rows marked correct with a complete reasoning chain.
- Normalize whitespace and case, fingerprint a prefix of every training problem, and drop anything matching a test item -- per row, for every source, late additions included.
- Cap target length to what the scoring token budget affords, and keep a share of short targets.
- End every training target on the answer, in exactly the form step 1 recorded.
- Strip the source's own markup, so the model never learns to emit it.

**Do not copy when:** the trade between more traces and more unique problems is a comparison, not a rule -- one run's compression ratio is worth nothing on your corpus. Run the comparison when a fixed token budget meets heavy per-problem duplication, and skip it when problems barely repeat. Contamination is the one place with no trade: a handful of residual overlapping rows can cost the whole run.

Cards: `references/09-schema-unification.md`, `references/10-deduplication.md`, `references/11-decontamination.md`, `references/12-verifier-filtering.md`, `references/29-sandbox-and-recovery.md`, `references/05-output-grammar.md`, `references/06-eos-and-length.md`, `references/04-chat-template-alignment.md`

## Step 6: decide which part of the output to supervise

The loss does not have to cover the whole sequence. Choose deliberately between: supervising only the assistant completion; supervising reasoning and final answer together; supervising only the final answer; supervising the whole tool call; supervising the code block or the function body; upweighting particular structure tokens; and using different data ratios for reasoning and for answers.

**Case:** a QLoRA run may have to refuse the tokenizer's default chat template, because the default injects an unwanted empty `<think>` block into the assistant turn. The answer is to hand-match the ChatML framing the evaluator uses and set the prompt tokens' labels to `-100`, so only the assistant completion is trained. On a tool-use task the opposite risk appears: if training rows are full of "Let me think about which tool to use...", the model learns to emit prose before the tool call, and the parser failure rate goes up.

**Directions:**

- Render a few fully tokenized training rows and read them, template and label mask included, before launching anything.
- Check that the framing at training time is byte-for-byte what the evaluator produces at scoring time.
- If a structure token is what fails, consider supervising or weighting it directly instead of adding more rows.

**Do not copy when:** turning reasoning off is not general advice. On a mathematics contest task reasoning itself earns the score; the problem there is controlling its length and making sure the final answer still appears.

Cards: `references/15-completion-only-loss.md`, `references/22-custom-loss.md`, `references/04-chat-template-alignment.md`

## Step 7: choose the training setup for the experiment you need

Do not reduce the choice to "full fine-tuning below some parameter count, LoRA above it". It also depends on: how far the behavior has to move; whether you need to try several data plans quickly; where the memory is actually going; how long the context is; whether an RL stage is planned; how easily this model forgets its original ability; and how many intermediate candidates you want to keep.

System optimizations split in two. The ones that cost nothing when they work -- fused attention such as FlashAttention, fused kernels such as Liger, packing or length grouping -- are on by default. Genuinely try each one before launch, and leave one off only when it will not install after real attempts or this model cannot use it, with that reason written down; "it was a bother" is not a reason. The ones that trade something away -- gradient checkpointing, an 8-bit optimizer, QLoRA -- follow the opposite rule: use one against a bottleneck you measured, never as a fixed full set.

| Bottleneck | Candidates |
| --- | --- |
| Attention activations large | FlashAttention |
| Activations large overall | Gradient checkpointing |
| Heavy padding waste | Packing or length grouping |
| Optimizer state large | 8-bit optimizer |
| Base weights dominate memory | QLoRA |
| Need many fast experiments | LoRA |
| Common kernels inefficient | Liger or other fused kernels |
| Data loading cannot keep up | Preprocessing, parallel loading, length bucketing |

**Case:** public training-run records do not share one preference. One set of runs chose LoRA on almost every task, another leaned on full fine-tuning, and a third used QLoRA in more than half of them. That says the trade between experiment speed, capacity, and memory was resolved differently -- not that one answer is right. A run might stack 4-bit NF4 storage for base weights, BF16 compute, LoRA, FlashAttention 2, gradient checkpointing, an 8-bit optimizer, and length grouping. That is one deliberate choice to adapt domain and format at low memory cost, not a generally optimal configuration. Another run might want Liger, find the system's disk conditions will not allow the install, use gradient checkpointing instead, and handle a large-vocabulary logits OOM by adjusting the batch.

**Directions:**

- The methods themselves live next door: the methodology skill's cards carry full fine-tuning's alternatives such as LoRA and QLoRA, and the framework skill's library cards carry each kernel and trainer. Read the method there, and keep only the experiment plan here.
- Fix the training stop time before launch, working backward from the deadline: measure once how long a full save plus a fresh-process load check takes at this model size, double it for the overwrite of the delivery folder, take the per-evaluation price measured in step 1, subtract both from the budget, and cut the schedule to whatever window remains.
- Prefer the arrangement whose first scored checkpoint arrives early; an early bad number redirects the run while budget still exists.
- When an optimization will not install, exhaust the cheap retries first -- build settings, a matching prebuilt wheel, a nearby version -- and only then ask whether the alternative already removes the measured bottleneck, before spending hours forcing it.

**Do not copy when:** another task that must move the base model's behavior much further may be better served by full fine-tuning than by the low-memory stack above. And a reserve is not optional in either direction: runs have trained to the wire and lost the save, and runs have stopped hours early and wasted the window. So the stop time is written down before launch.

Cards: `references/16-precision.md`, `references/17-flash-attention.md`, `references/18-gradient-checkpointing.md`, `references/19-packing.md`, `references/20-optimizer-scheduler.md`, `references/01-terms-apart.md`, `references/28-time-budget.md`

## Step 8: pilot, then run the first plain SFT

A pilot is not "did the loss go down". Watch: does the model start producing the correct structure; does output length grow abnormally; does it terminate properly; does new repetition appear; does a gain in one ability cost another; are generation speed and memory acceptable; and does the model rapidly amplify an error that was in the data.

The main run the pilot feeds exists to produce one diagnosable candidate. It does not need a multi-stage curriculum, self-training, GRPO, DPO, weight averaging, and a custom loss at the same time. The more it carries at once, the less you can tell where the gain came from.

**Case:** a math-contest pilot can raise the quick evaluation while mean output length creeps toward the generation cap. The right conclusion is not "score up, scale it up" but "reasoning SFT works, and the main run must handle length and termination". The counter-case is just as common: when the loss falls normally but the output still does not satisfy the parser, the template and the label mask are the first thing to re-read, not the data volume. For the main run itself, across public training-run records every run starts from SFT, and the difference between stronger and weaker runs comes from how they iterate on data, parameters, and stages afterwards. A code-generation run can raise its baseline most of the way with execution-filtered SFT alone, and only then add STaR and GRPO on top.

**Directions:**

- The pilot is the default. Going straight to the full schedule needs a stated reason, such as a window too short to fit one.
- Run the pilot long enough to produce one checkpoint you can evaluate, not just a loss curve.
- Measure throughput here, and recompute the stop time from that measured number rather than the pre-launch estimate.
- Save checkpoints at a cadence that writes about ten across a full run. Keep only the last few, and never delete the one the delivery folder was copied from.
- If the pilot checkpoint beats the model now in the delivery folder on the proxy, overwrite the folder now and re-run the fresh-process load check.
- Then train the simplest schedule that tests step 3's hypothesis, on the corpus step 5 produced, with the supervision target step 6 chose.
- Keep the schedule inside the window step 7 fixed, and recut it if the measured throughput says it will not fit.
- Evaluate on the fixed proxy slice at each checkpoint, and let that drive the delivery-folder overwrite.
- Keep the full evaluation for the candidate you intend to deliver.

**Do not copy when:** a pilot that improves the proxy and breaks a behavior you were not watching is not a green light. Read samples beside the numbers: a model whose samples wander into repetition or never land on the answer form is failing even while its loss falls. And "SFT first" is not a law for every task. It reads as: while the model cannot yet produce verifiable answers reliably, SFT usually gives a stronger signal than going straight to RL. If the model already produces valid answers reliably, that argument does not apply.

Cards: `references/23-checkpoint-cadence.md`, `references/30-pipeline-shapes.md`, `references/28-time-budget.md`

## Step 9: re-diagnose after training

Do not compare "baseline 10 percent, SFT 20 percent" and stop. Ask how the 20 percent arrived: did the valid rate rise; did the normal-termination rate rise; did outputs get longer; did knowledge errors fall; did only the easy items improve; has a new mode collapse appeared; and has ordinary instruction following been damaged.

**Case:** picture a rubric-scored health-advice run whose four versions separate the explanations cleanly.

| Version | Change | Result |
| --- | --- | ---: |
| v1 | Fairly balanced data | the reference |
| v2 | More verbose, more targeted, less general chat | below v1 |
| v3 | Added targeted synthetic data on top | below v2 |
| v4 | Restored v1's data, only trained longer | best, above v1 |

The conclusion is not "targeted data is always bad". It is that this model was mainly underfit at the time, that longer answers did not improve completeness, and that removing the general data hurt instruction following. On a math contest task, a second stage that emphasizes longer complete reasoning can never beat the better checkpoint from the first stage -- which says the problem there was not a shortage of long reasoning.

**Directions:**

- Re-run the step 2 breakdown on the same fixed sample, and compare the shares, not just the totals.
- Check the abilities you were not training as well as the one you were.
- When a fancier version loses to a simpler one, accept that result and go back to the version that won.

**Do not copy when:** these numbers belong to one model, one corpus, and one budget. What transfers is the shape of the experiment -- change one thing, re-measure the breakdown -- not the verdict that verbose data or targeted data is bad.

Cards: `references/03-baseline-failure-taxonomy.md`, `references/14-data-mixture.md`

## Step 10: try cheap inference fixes before retraining

Before spending the window on more training, sweep the settings that cost minutes: greedy against sampling, temperature, top-p and top-k, the EOS token, the assistant-turn stop, maximum generation length, repetition penalty, whether thinking mode is on, and whether the model emits a second answer or a second tool call after the first. This sweep comes before any retraining decision, every time. It costs minutes against hours, so skipping it needs a stated reason.

**Case:** two cases point in opposite directions. On a function-calling task, the same weights can move from almost no correct items to most items correct when the change is greedy decoding plus the correct assistant-turn EOS. On a math contest task, lowering the temperature below the model card's recommended value can fail to help and make things worse. More deterministic decoding does not suit every long-reasoning task.

**Directions:**

- The methodology skill's `inference.md` reference carries the full inference-settings method; this step owns only the search order.
- If the errors concentrate in structure tokens, JSON wrappers, or random format drift, compare greedy with the shipped sampling defaults.
- If the same weights improve sharply under greedy, fix decoding before touching weights.
- If reasoning accuracy drops under greedy, keep a moderate-sampling candidate in the comparison.
- Decide every field of `generation_config.json` on purpose, and keep exactly those fields in the delivery folder.

**Do not copy when:** never write the rule as "structured tasks use temperature 0, math tasks use temperature 0.6". The condition is what the errors look like, not what the task is called.

Cards: `references/07-decoding-search.md`

## Step 11: pick the second stage from evidence

There are at least five second stages, and entering RL by default is a choice made without evidence. Choose the one whose entry conditions your step 9 breakdown actually met.

**Case:** a rubric-scored run's best version might simply restore the earlier, better mixture and train more, instead of inventing new data. A math-contest run might take the concise route: shortest-correct traces and an explicit answer format. A code-generation run might take the self-training route: sample repeatedly over problems that ship tests, keep the answers that pass, add those verified on-policy rows, and gain a little. The same run can then continue with a test-pass reward under GRPO, pick the best of several checkpoints, and gain more. Public records also show small models collapsing under GRPO without KL, with a KL anchor added afterwards. An open-ended writing run might take the preference route: broad writing and instruction SFT first, then DPO on preference pairs.

**Directions:**

- The methodology skill's cards on RFT, STaR, GRPO, and DPO carry each method in full; this step owns only the entry conditions.
- Continue the same SFT when the data direction is already right, the model is still underfit, and there is no clear forgetting or overfitting.
- Take concise or efficiency annealing when accuracy on finished answers is high, many outputs are truncated, reasoning repeats its own checks, and the model does not commit to a final answer in time.
- Take self-training, RFT, or STaR when a reliable verifier exists, the current model already succeeds sometimes, you can sample several answers per legal training problem, and the correct completions vary.
- Take GRPO when the reward computes reliably, the model already emits valid completions steadily, the several completions of one problem contain both right and wrong, the reward is hard to game with formatting alone, and you can watch for collapse.
- Take DPO when there is no exact answer but there are high-quality chosen and rejected pairs, and the difference is overall writing, style, or instruction following.

**Do not copy when:** each route has an edge it falls off. A verifier alone does not justify GRPO: if every completion fails, improve SFT first, and if every completion succeeds, find more discriminating training problems. RFT creates no ability from nothing -- if the model fails on everything, there is nothing to filter. Over-shortening every trace can lower the ceiling when the real gap is reasoning ability rather than truncation. And preference pairs whose only difference is that the chosen answer is longer teach length, not quality.

Cards: `references/12-verifier-filtering.md`, `references/13-curriculum.md`, `references/21-multi-stage-sft.md`, `references/22-custom-loss.md`, `references/01-terms-apart.md`, `references/29-sandbox-and-recovery.md`, `references/30-pipeline-shapes.md`, `references/34-distillation.md`

## Step 12: compare checkpoints, decoding, and an averaged candidate

The last checkpoint is a candidate, not the answer. Compare candidates on: the total score; the valid rate; the truncation rate; output length; performance across problem types; the swing across repeated evaluations; whether individual items are stably correct or coin flips; whether a dtype conversion degraded anything; and whether the inference configuration was identical across the comparison.

**Case:** a run evaluating several GRPO checkpoints can see the curve rise and then fall, and choose a middle checkpoint rather than the last step. A math-contest run comparing checkpoints might average the weights of three neighboring ones, and pick that average not only for the mean score but because more items are stably solved, fewer items are coin flips, and a single official evaluation would be more predictable -- while the wider averages it tries do not help.

**Directions:**

- When several nearby checkpoints each look good but solve different stable sets, test a weight average and re-evaluate it as an ordinary candidate.
- Repeat the evaluation of the top two candidates, to see how much of the gap is noise.
- Hold decoding fixed while comparing weights, and hold weights fixed while comparing decoding.
- Every time a candidate wins, overwrite the delivery folder with it in the foreground, and re-run the fresh-process load check.

**Do not copy when:** do not average checkpoints from clearly different training stages, or ones that scored poorly on their own. Do not treat averaging as a mandatory end-of-training step. And do not let a comparison run past the reserve: a scored earlier checkpoint beats an unloadable later one. How much of the evaluation a comparison may read is set by any rules your setting imposes, not by this card.

Cards: `references/24-checkpoint-tournament.md`, `references/25-repeated-evaluation.md`, `references/26-checkpoint-averaging.md`, `references/07-decoding-search.md`

## Step 13: verify the final candidate's real behavior -- and deliver it

The final check is about how the model actually behaves, not whether the training process looked successful. Confirm: the folder loads; the tokenizer and template match the evaluator; the key special tokens agree; EOS is right; merging the adapter did not change behavior; a precision conversion did not cost score; the generation config takes effect; the final weights really came from the base model you started from; and the data and training method obey any rules your setting imposes. That last check is worth more than any of the numbers you are comparing.

**Case:** loading in the training process proves nothing, because that process already holds the classes and paths that wrote the files. A run's whole delivery risk can sit in the last write. The environment stops the run once the budget ends, background jobs die unsaved, and a save interrupted mid-write can leave the folder broken -- when a moment earlier it held a model that was proven to load.

**Directions:**

- Merge any adapter into the base weights, so the folder loads as a plain model with no adapter files beside it. The framework skill's `loading-the-result.md` carries each trainer's exact recipe.
- Save the tokenizer with library versions matching what the grading side will load with, when you know them. A tokenizer written under a newer major can crash grading.
- Deliver exactly the `generation_config.json` fields step 10 decided.
- Run the final save in the foreground, inside the reserve step 7 fixed.
- Then start a fresh process, load the folder by path alone, generate one answer on a held-out item, and check it ends on the extractable form inside the scoring token cap, including any shrink retries.
- Confirm the fingerprint check of step 5 ran on every corpus that reached the trainer, late additions included.
- Leave the evaluator, templates, and everything beside them exactly as shipped.

**Do not copy when:** when the reserve is threatened, do not attempt one last risky overwrite. The folder already holds a model that loaded and answered when you last checked it; that model is the submission. If any check above fails on the newest candidate, put the last model that passed the load check back into the folder -- shipping a folder that fails a check is worse.

Cards: `references/27-artifact-verification.md`, `references/16-precision.md`

# What every decision must answer

The point is not to march through thirteen actions in order. It is that at every important decision you can answer four questions: what did I observe; which layer does that blame; which experiment separates the competing explanations at the lowest cost; and does the result support continuing in this direction.
