# Completion-only / assistant-only loss

Read in step 6: where the loss is computed.

Many runs compute no loss on the prompt tokens and train only the assistant response.

The usual implementation:

```text
labels = [-100] * prompt_length + response_token_ids
```

Or use:

- `completion_only_loss=True`
- `assistant_only_loss=True`

## Why it helps

- no gradient is wasted on learning to repeat the user prompt;
- it matters most with long tool schemas;
- the model is less likely to treat system/user text as something it must generate;
- the training signal lands on the answer, the tool call, or the code.

A math-contest SFTConfig can use packing, BFD, and `completion_only_loss=True` together. A multi-turn rubric-scored run can compute loss only on the final assistant turn. A QLoRA run can mask the prompt labels by hand.

## Easy mistakes

After packing, the response mask boundary must still be right. If it is not, you will:

- compute loss on the previous sample's padding or separator;
- treat the next sample's prompt as an answer;
- mask out a short completion completely;
- be left with only the prompt after truncation, so there is no usable label.

The framework skill's TRL card carries the full library settings; this card keeps only what the workflow needs.
