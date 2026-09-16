# Packing, BFD, and length bucketing

Read in step 7 when short samples waste the batch.

## Packing

Put several short samples into one fixed-length sequence, so there is less padding.

## BFD packing

`packing_strategy="bfd"` usually means best-fit-decreasing: it orders the samples by length first, so the leftover space is used better.

## `group_by_length`

This does not really join samples together. It puts samples of similar length into the same batch, so there is less padding inside the batch.

## The combinations in the records

A math-contest SFT might use:

```text
packing=True
packing_strategy="bfd"
dataset_num_proc=16
dataloader_num_workers=2
completion_only_loss=True
```

A function-calling full fine-tuning run might use `group_by_length=True` with a length column instead.

Under a corpus of short samples, turn packing or length grouping on before launch; the risks below are things to check, not reasons to leave it off.

## The risks

- Whether the attention boundary is right after packing.
- Whether EOS separates the different samples.
- Whether the response-only mask is still correct.
- Packing still cannot solve one single sample that is too long.
- Packing can change the effective token count per optimizer step.

So it is best to record all of these:

```text
tokens / second
non-padding tokens / batch
examples / second
peak memory
effective tokens / optimizer step
```

and not the batch size alone.

The framework skill's card on TRL carries the full library detail; this card keeps only what the workflow needs.
