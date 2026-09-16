# Deduplication and picking representatives

Read in step 5: dedup beyond exact strings.

Dedup is not only about dropping the exact same string. It also covers:

- many reasoning traces for the same problem;
- many near-alike answers to the same prompt;
- repeats across data sources;
- tool calls that become the same after unification;
- code solutions that differ only in comments or variable names;
- large piles of synthetic examples from one template.

## The shortest-correct trick

A large reasoning corpus can hold many correct traces for the same problem. Do not keep them all. Instead:

- squeeze the traces down to one per unique problem;
- keep the shortest correct trace per problem;
- then sample by length band.

This gets four things at once:

- a higher unique-problem per token ratio;
- less over-long reasoning;
- a positive signal for stopping (termination);
- more problems covered inside a fixed token budget.

## Three things to keep apart

- **Dedup**: stop repeats from eating the training budget;
- **Keeping variety**: in some RFT settings, keeping different correct reasoning paths for the same problem can still help;
- **Blind shortest-only**: it can throw away the long reasoning you need, so look at the benchmark generation budget and at the model's failure profile first.

The dataset skill carries the catalog of sources; this card keeps only the dedup work the workflow needs.
