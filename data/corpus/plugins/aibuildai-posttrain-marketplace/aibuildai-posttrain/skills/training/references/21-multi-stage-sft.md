# Multi-stage SFT and annealing

Read in step 11 when planning a second stage.

Several pipelines come back again and again.

## A. Broad SFT then targeted SFT

The first stage learns basic instruction following and format. The second stage raises the share of task-specific data.

## B. Reasoning SFT then concise anneal

First learn to solve the problem, then learn to finish it on a shorter path.

## C. SFT then one more epoch

When the diagnosis is underfit, and not bad data, keep the data recipe the same and only add training.

## D. SFT then RFT data then a second SFT

Add the model's own verified completions.

## E. SFT then a preference or RL stage

Run DPO or GRPO on top of an output style that is already stable. The methodology skill's cards on DPO and GRPO carry the full methods; this card keeps only what the workflow needs.

On a rubric-scored task, the best gain can come from "the same v1 data, trained longer", not from more complex targeted data.

A math-contest run can go the other way: a second stage of long-trace continuation gives no gain. More training is not better by nature.

## Keep the incumbent

Before you try a second stage, copy the current best model whole into a safe folder. Keep the verified v1 in the delivery folder and treat v4 as a pure-upside experiment: replace v1 only after v4 wins a full evaluation.
