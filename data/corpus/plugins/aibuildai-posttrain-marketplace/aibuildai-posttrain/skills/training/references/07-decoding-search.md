# Decoding and sampling parameter search

Read in step 10, before blaming weights for a bad number, and in step 12 for the final decoding choice.

The official evaluator reads `generation_config.json`, so the inference settings are not a side matter. They are part of the final model.

Parameters that need testing:

- `do_sample`
- `temperature`
- `top_p`
- `top_k`
- `repetition_penalty`
- `max_new_tokens`
- `eos_token_id`
- `min_new_tokens`
- whether thinking mode is on

## One important pattern

**A structured task with one right answer usually leans greedy; open writing or hard reasoning does not always.**

From public training-run records:

- a numeric-answer math run can set temperature to 0 and gain a large share of its score;
- a function-calling run can go from almost no correct items to most items correct once a wrong stop token is fixed alongside greedy decoding.

This shows that a so-called "training failure" is sometimes just the base model's default sampling config being wrong for the task.

But do not turn every task greedy by reflex. A math-contest run that lowers `temperature` below the model card's recommended value can watch its internal result fall and have to go back to the recommended sampling settings.

## The search order to use

```text
1. verify EOS and parser compatibility
2. greedy baseline
3. low-temperature sampling
4. model-recommended sampling
5. repetition penalty
6. full-score confirmation
```

Do not settle the final decoding settings on only 5 to 10 samples.

The methodology skill's `references/inference.md` carries the full inference-settings method; this card keeps only the search the workflow needs.

The next layer is data engineering, covered by the card that follows.
