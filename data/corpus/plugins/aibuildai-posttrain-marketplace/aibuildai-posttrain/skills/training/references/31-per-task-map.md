# Per-task map

Read in step 1 to see the task family's usual failure and the moves that pay.

| Task family | Key failure | High-value trick | Easy way to fail |
|---|---|---|---|
| **Math contest, exact integer answer** | reasoning too long, truncation, repetition, integer answer format | shortest correct trace, length stratification, concise anneal, an answer line, repeated evaluation, soup of nearby checkpoints | adding longer CoT with no condition; believing a low temperature is always better |
| **Math word problems, numeric answer** | format, number extraction, default sampling | concise CoT, exact numeric reward, format reward, greedy decoding, GRPO | a high temperature that makes simple arithmetic drift |
| **Code generation graded by tests** | parser, imports, syntax and runtime errors, tests, dangerous code | SFT in the harness format, execution filtering, STaR, unit-test GRPO, greedy, sandbox | not executing the training code; running candidate code in the task directory |
| **Function calling** | tool schemas not uniform, wrong argument types, extra text, no stopping | schema canonicalization, single tool-call SFT, assistant-only loss, stop-token weighting, greedy, a correct EOS | learning only the function name and never checking the arguments |
| **Multiple-choice science questions** | chat template, choice output, science domain coverage | exact evaluator framing, balanced science data, response-only QLoRA, choice-format training | the default Qwen template injecting a wrong thinking structure |
| **Health advice scored on a rubric** | completeness, safety, context awareness, instruction retention | a mixture of medical, multi-turn, and general chat, completion-only fine-tuning, rubric-axis diagnosis, more epochs to fix underfitting | only adding verbosity; deleting general chat; targeted data that is too narrow |
| **Open-ended writing judged by preference** | no exact verifier, length and style limits, judge variance | a broad writing mixture, preference data, DPO, a length and repetition proxy, sampling | picking answers by length alone; tuning against the judge's own quirks instead of writing quality |

The dataset skill holds the data sources named here, and the methodology skill holds the methods; this card keeps only the per-task-family picture the workflow needs.
