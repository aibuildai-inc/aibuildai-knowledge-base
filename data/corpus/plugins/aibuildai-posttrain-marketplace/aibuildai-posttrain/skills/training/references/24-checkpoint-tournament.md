# Checkpoint tournament and layered evaluation

Read in step 12 when more than one checkpoint competes.

Better runs use a layered tournament:

```text
all checkpoints
    ↓ cheap proxy / validation loss
a few candidates
    ↓ fast evaluation on 20-50 samples
2-4 candidates
    ↓ full benchmark
the best 1-2
    ↓ repeated evaluation / dtype / decoding checks
final model
```

A strong code-generation run might compare its GRPO checkpoints at the end and pick the one that scores best on the full evaluation set.

A math-contest run should compare more than the mean number correct. It can also compare:

- always-solved problems;
- flaky problems;
- stability across different runs.

It might find that the best single checkpoint solves a few problems every time and many only by chance, while the soup of the last few checkpoints solves more problems every time and swings less, and so pick the soup.

## Checkpoint statistics worth saving

```text
benchmark score
invalid rate
truncation rate
mean output length
per-item pass bitset
pairwise disagreement
generation seed
decoding config
dtype
checkpoint step
validation loss
```
