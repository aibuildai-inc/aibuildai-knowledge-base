# The decision order worth keeping

Read once before step 1: the priority order the whole workflow follows.

The priority these records point to is not "start with the most complex algorithm". It is:

```text
0. rules audit
   ↓
1. read the evaluator contract
   ↓
2. run the baseline + build a failure taxonomy
   ↓
3. first fix chat template / output grammar / EOS / decoding
   ↓
4. build a clean data portfolio
   ↓
5. canonicalize + deduplicate + decontaminate
   ↓
6. choose full FT / LoRA / QLoRA
   ↓
7. pilot to check throughput, OOM, loss, and generation behavior
   ↓
8. main SFT
   ↓
9. analyze the real failures instead of blindly adding data
   ↓
10. do a concise anneal / RFT / STaR when it is needed
   ↓
11. do GRPO only when a verifier exists and SFT is already stable
   ↓
12. consider DPO when there is no exact verifier but legal preference data exists
   ↓
13. checkpoint tournament + decoding sweep
   ↓
14. try a soup of nearby checkpoints if needed
   ↓
15. merge, dtype check, artifact smoke test
```

## The core conclusion

What paid again and again in these records is not one magic optimizer. It is this combination:

1. **Evaluator-aware formatting**
2. **Verifier-aware data filtering**
3. **Length and termination engineering**
4. **A high-quality data mixture, deduplicated and decontaminated**
5. **Choosing full FT / LoRA / QLoRA by the resources you have**
6. **System optimizations such as FlashAttention, Liger, checkpointing, and packing**
7. **Doing RFT/STaR or GRPO after SFT, not before**
8. **Treating the generation config as part of the model**
9. **Choosing the checkpoint from real benchmark behavior**
10. **Strict artifact verification**
11. **Keeping the test set out of the training loop, so the score still measures generalization**

Public training-run records support this too: the main gap between runs does not come from whether they know SFT. It comes from data, format, iterative diagnosis, inference configuration, and whether they can move safely into the harder self-training and RL stages. The dataset, methodology, and framework skills carry the data sources, the methods, and the libraries named here; this card keeps only the order in which the workflow decides.
