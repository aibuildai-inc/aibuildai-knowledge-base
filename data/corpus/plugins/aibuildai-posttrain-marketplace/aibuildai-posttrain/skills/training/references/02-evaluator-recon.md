# Evaluator reconnaissance: understand the scoring before you train

Read in step 1, before any data is chosen.

This is the most common move, and often the one that pays best.

A run usually reads these first:

- how the prompt is put together;
- which chat template is used;
- which part of the output the evaluator cuts out;
- whether it takes only the first code block;
- what form the final answer must have;
- whether reasoning is allowed;
- the maximum generation length;
- EOS and stop sequences;
- how an invalid output is scored;
- whether the evaluator uses the `generation_config.json` in the model folder.

## Why it matters

A low zero-shot score from the base model may not mean weak ability. It may mean:

- it never enters the assistant role;
- it prints an extra prefix;
- the answer never appears where the parser looks;
- it keeps reasoning and never hands in a final answer;
- the tool call JSON is wrapped in explanation;
- the Python code sits in the wrong place;
- it generates an empty `<think></think>` block;
- EOS is wrong, so it keeps going and writes a second answer.

A QLoRA run may refuse the tokenizer's default template, because that template injects an empty `<think>` block into the assistant turn, and rebuild the evaluator's Qwen ChatML framing by hand. A code-generation run can build its data to follow the evaluator's "first Python code block" parsing exactly, and put the missing imports inside the function body.

## The rule a run should form

Before training starts, write one fixed evaluator contract:

```text
prompt renderer
assistant prefix
expected response grammar
answer extractor
EOS / stop tokens
maximum prompt length
maximum completion length
invalid-output conditions
generation_config loading behavior
```

Note one limit here: **the evaluator is a fixed quantity. Fit the training data and the final model config to it, and read it without editing it** -- an evaluator that moved during the run measures nothing comparable.
