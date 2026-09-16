# Decontamination and source tracking

Read in step 5, on every corpus that reaches the trainer.

Training data may match the overall shape of the task without harming the measurement:

- style;
- format;
- domain;
- difficulty.

What destroys the measurement is building training data out of specific test items. That covers:

- copying;
- rewording;
- changing the numbers;
- making same-template variants from items the model failed;
- using a test item as a seed for synthetic generation;
- hand-covering the pattern of specific wrong answers.

A graded setting may ship an n-gram decontamination tool, and may also judge contamination semantically, because n-grams alone cannot catch "different words, same problem" targeting.

## Data facts worth recording

- source name;
- source revision;
- source license;
- row count before and after filtering;
- exact duplicate count;
- n-gram overlap count;
- blocked function names and answer strings;
- synthetic generator seed;
- self-generated checkpoint identity;
- verifier version;
- contamination report;
- source mixture weights.

## The line that matters

Using test data **as a filter that only removes and never adds** is common in public training-run records. But looking at why one test item failed and then making a training row for it is targeted derivation: the score stops measuring the model and starts measuring how much of the test set the corpus absorbed.

The dataset skill carries the catalog of sources; this card keeps only the cleaning work the workflow needs.
