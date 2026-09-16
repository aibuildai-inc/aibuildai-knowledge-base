# A portfolio of data sources, not one dataset

Read in step 4 when choosing sources.

A run often builds several data sources into one portfolio, where each source does a different job:

- ability in the task's own domain;
- general instruction following;
- multi-turn;
- imitation of the output format;
- hard examples;
- concise examples;
- variety of tool schemas;
- correctness of code execution;
- safety and the style of communication.

## Typical mixes

**For a code-generation task graded by tests:**

- MBPP;
- code instruction data filtered by execution;
- later, self-generated STaR samples;
- oversample the high-value MBPP data.

**For a function-calling task:**

- xLAM;
- Hermes;
- ToolACE;
- the function-calling subset of SmolTalk;
- locally generated, deterministic data with different schemas and argument types.

**For a rubric-scored health-advice task:**

- real medical question-answering prompts;
- multi-turn health conversations;
- general chat data, so the model does not learn only long medical answers and lose plain instruction following.

## The core principle

The quality of training data is not one number. "Closer to the scored task" may raise task fit, but it may also:

- cause data contamination;
- lead to fitting that is too narrow;
- break plain instruction following;
- add verbosity instead of completeness;
- lower the variety of the output.

The dataset skill's catalog carries the full detail of the sources it lists; this card keeps only the portfolio thinking the workflow needs.
