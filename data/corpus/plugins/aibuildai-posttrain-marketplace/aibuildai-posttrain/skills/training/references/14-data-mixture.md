# Data mixture, oversampling, and anti-forgetting

Read in step 4 when setting source weights, and in step 9 when specialization hurts general ability.

A training set is usually not a plain concat. You set a weight per source on purpose.

Common moves:

- repeat scarce high-value data N times;
- give verified self-generated rows a higher weight;
- cap how much a large but noisy source may contribute;
- keep some share of general instruction data;
- set a floor quota for multi-turn, format, and hard examples;
- change the mixture between stages.

## The counter-example that matters most

Suppose v2 uses more verbose, more targeted health answers and cuts general chat, and the score falls below v1. The reading is:

- length is not completeness;
- cutting general-chat data badly hurts instruction following.

v3 adds targeted synthetic data on top and falls further. Only going back to the original balanced recipe and training longer lifts it above v1.

So the skill must keep these apart:

```text
task specialization
general instruction retention
format coverage
multi-turn coverage
hard-example coverage
```

Do not give the whole budget to benchmark-like data.

The dataset skill carries the catalog of sources; this card keeps only the weighting work the workflow needs.
