# Generation settings: the hour-zero fix

Generation settings change scores without changing weights. Fix them before
the first training decision, and check them again when the final model is
saved. This file covers only how the model speaks at scoring time. How to save a loadable model is the framework
skill's `references/loading-the-result.md`.

## The one big fact

Base models ship a `generation_config.json` tuned for chat variety, not for
single-sample scoring. When a grader samples each answer once, greedy
decoding (`temperature=0.0`) usually beats those shipped defaults, and the
gap can be larger than a whole training run's gain. In published
post-training runs, forcing greedy decoding (`temperature=0.0` in
`generation_config.json`) beat the base models' shipped sampling defaults:
one GSM8K run jumped from 42.7% to 78%, and one function-calling (BFCL) run
went from 17% to 91%. Both agents treated the sampling defaults as a
first-order bug to fix before training.

So the first action of a run is to read the shipped
`generation_config.json` and decide every field on purpose. The fields that
matter: `do_sample`, `temperature`, `top_p`, `top_k`, `repetition_penalty`,
`max_new_tokens`, and the `eos_token_id` list.

## What you control, and what the grader controls

The model's delivery folder is the only channel. A grader that loads the
folder reads `generation_config.json` and the tokenizer from it; it does
not read a script. Everything else is the grader's: the chat template it
applies, the prompt, the token budget, and how it extracts the answer.
Verified on one benchmark harness's evaluator (other graders may differ; check yours):

- The evaluator forces its own chat templates and ignores the tokenizer's.
  Train inside the template the grader will apply, not inside one you wrote.
- Failed evaluation attempts retry with SMALLER token caps (down to 2000
  on GSM8K). Long reasoning that scores in the
  first attempt can truncate to zero in a retry. Short answers are a
  robustness feature, not a style choice.
- A leading think block is stripped before judging, and text repeated more
  than 5 times is cut (the judged long-form tasks' `evaluate.py`).
  Greedy decoding repeats more, so on judged long-form tasks check output
  for loops before choosing it.
- A model whose reasoning PRINTS a special token like `<|im_end|>` stops
  itself at that point under the forced template.
  Keep special tokens out of training text.

## Greedy is a default, not a law

Pick per task, and measure. Exact-answer tasks (math, multiple choice,
code, tool calls) reward the single most likely token path, so greedy wins
there in the documented runs. Judge-scored long-form tasks reward variety
and can punish the loops greedy produces. When the shipped default already
sets a low temperature, forcing 0.0 may change little; the point is that
the value is CHOSEN, never inherited unread.

## The check recipe

One setting change is one comparison. Run the task's own evaluator on a
small sample (its `--limit` flag) with the old and the new
`generation_config.json`, same model, and keep the two numbers next to the
change. A setting adopted without that pair is a guess, and the
documented jumps above show the stakes run to tens of points.
