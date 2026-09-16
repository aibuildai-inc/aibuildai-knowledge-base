# Verifier-based filtering and rejection sampling

Read in step 5 when a rule can check each row, and in step 11 for self-generated data.

This is the strongest family of data tricks for math, code, and tool use.

The basic form:

$$
D_{\text{accepted}}
=
\{(x,y): V(x,y)=1\}
$$

The verifier can be:

- an exact final answer;
- unit tests;
- Python execution;
- a JSON parse;
- a function name match;
- argument schema validation;
- an XML or format check;
- a timeout check.

## Two ways to use it

### A. Clean outside training data

A code-generation run can re-run all of its training code through an execution path close to the evaluator's, so that only samples whose tests pass go into SFT. An SFT v1 built this way holds only decontaminated samples in the evaluator's own format, every one really run and checked.

### B. Filter the model's own samples

Sample several candidates per prompt:

```text
prompt
  ├─ completion 1 → fail
  ├─ completion 2 → pass → keep
  ├─ completion 3 → timeout
  └─ completion 4 → pass → keep/deduplicate
```

This turns the model's own lucky wins into new supervised data.

## Risks

- a verifier that is too loose keeps reward-hacking output;
- a verifier that is too strict throws away answers that are just as good;
- code must run in a sandbox;
- many alike candidates must be deduped again;
- a hard prompt where every candidate fails makes no training row at all.

The methodology skill's card on RFT carries the full method; this card keeps only what the workflow needs.
