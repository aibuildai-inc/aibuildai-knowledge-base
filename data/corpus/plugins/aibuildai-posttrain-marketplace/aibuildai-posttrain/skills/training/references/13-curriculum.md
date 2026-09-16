# Difficulty, length, and curriculum

Read in step 11 when ordering stages by difficulty or length.

Public training-run records show several kinds of curriculum:

- easy → hard;
- long → concise;
- broad SFT → targeted anneal;
- general ability → task format;
- standard data → hard data the model can currently solve;
- main training → a short stage at a low learning rate;
- difficulty split by teacher pass rate.

## The most common curriculum for reasoning tasks

```text
high-quality reasoning SFT
        ↓
look at truncation / repetition
        ↓
pick shorter but still correct traces
        ↓
concise annealing at a lower learning rate
```

## But not every second stage works

Suppose a math-contest run adds a second stage of "long but complete traces", hoping to lift ability on hard problems. It can score lower, with the stage-2 soup worse than the soup of stage-one checkpoints.

So the curriculum must follow the failure mode:

- if the problem is that the model cannot do complex reasoning, long traces may help;
- if the problem is that it already reasons but cannot stop, long traces make it worse;
- if it is overfitting, more targeted annealing will also make it worse;
- if it is underfitting, more epochs may beat changing the data.

The dataset skill carries the catalog of sources; this card keeps only the ordering work the workflow needs.
