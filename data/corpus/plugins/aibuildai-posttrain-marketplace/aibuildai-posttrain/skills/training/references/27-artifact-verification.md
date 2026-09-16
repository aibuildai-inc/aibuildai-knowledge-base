# Artifact precision, completeness, and smoke verification

Read in step 13: what the delivered folder must prove.

The final model must be loaded again from the **delivery folder**. Do not assume that a training checkpoint is good enough.

Public training-run records show these failures:

- the score dropped after the BF16 conversion;
- `temperature=0.0` failed some `save_pretrained` checks, so the JSON had to be written by hand;
- the tokenizer template was not copied;
- the EOS config was lost;
- a shard was missing;
- the model directory pointed at an adapter;
- vLLM could not recognize the architecture;
- regex or tokenizer warnings;
- files went missing after a copy.

Compare the delivery folder with the folder already evaluated, file by file and size by size, and check the safetensors index; or run the shipped evaluator once more on the final folder.

A solid finalization checklist should hold:

```text
model identity
architecture
parameter count
dtype
all shard presence
tokenizer load
chat template render
generation config
single-sample generation
small official-eval smoke test
full candidate score provenance
```

The framework skill's `references/loading-the-result.md` carries the exact loading recipe for each trainer; this card keeps only what the workflow must prove.
