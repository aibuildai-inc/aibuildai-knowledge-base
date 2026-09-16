# Checkpoint averaging and model soup

Read in step 12 when nearby checkpoints are each strong.

The definition:

$$
\theta_{\text{soup}}
=
\sum_{i=1}^{K}w_i\theta_i,
\qquad
\sum_i w_i=1
$$

The simplest form is an equal-weight mean.

## When it applies

- the same base model;
- the same architecture;
- the same tokenizer and vocabulary;
- parameter names exactly the same;
- best if the checkpoints are neighbors from the same training run;
- the checkpoints sit in the same loss basin;
- each one already performs well on its own.

## Possible gains

- it smooths the noise of a single checkpoint;
- it cuts overfitting along some parameter directions;
- it makes a single sampling pass more stable;
- it acts like a simplified stochastic weight averaging.

## A worked math-contest case

Suppose the final soup averages three late checkpoints, picked for its pooled score and its stable solved core.

But a wider soup and a stage-2 soup can both come out worse.

## So a soup is not the default move

The right flow is:

```text
select individually strong, nearby checkpoints
average in FP32
save one model
re-run full evaluation
compare both mean and failure overlap
```

Do not submit a soup just because "an average is usually steadier".
