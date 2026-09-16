# Moves that must not become unconditional rules

Read in step 3 and in every step after it: rules that must stay conditional.

| Wrong assumption | Counter-example |
|---|---|
| **"Longer output is more complete"** | on a rubric-scored task, a more verbose version of the data can score below the balanced version |
| **"The more targeted the synthetic data, the better"** | adding targeted synthetic data on top can lower the score further; generating data aimed at the actual test items can also count as contamination |
| **"Hard reasoning needs longer traces"** | a long-trace second stage on a math contest task can score below the first stage |
| **"A lower temperature is always more stable and scores higher"** | the same math-contest weights can score lower below the model card's recommended temperature than at it |
| **"A soup always raises the score"** | a wider soup and a stage-2 soup can both lose to the selected soup |
| **"The lowest validation loss is the best on the score"** | a tournament on the real score often picks a checkpoint that is not the last one |
| **"The BF16 conversion does not change the result"** | an FP32 to BF16 conversion on a code-generation task can lose correct items |
| **"Use only hard items for GRPO"** | when every completion gets reward zero, there is no advantage signal inside the group |
| **"Deleting general data raises specialization"** | on a rubric-scored task, instruction following can go clearly backwards |
| **"Code that parses is good enough as training data"** | strong code-generation runs actually execute and test every piece of code |
| **"GRPO without KL saves resources, so it is better"** | small models can collapse without it, and a KL anchor is the repair |
| **"Installing more kernels always makes it faster"** | the Liger install is limited by disk and environment, so a fallback is required |

The methodology skill's cards on GRPO, DPO, and the rest carry the full methods, and the framework skill's card on Liger carries that library; this card keeps only the rule that none of these moves is unconditional.
