# EOS, stopping, and length engineering

Read in steps 2 and 5 when anchoring answers and capping lengths.

This is one big class of problems that the task tends to leave underrated.

The concrete moves are:

- set `eos_token_id` correctly;
- allow both the architecture EOS and the assistant-turn EOS;
- set `pad_token_id`;
- add EOS at the end of the training target, in plain sight;
- control `max_new_tokens`;
- use the shortest correct trace;
- anneal toward concise answers;
- give the closing token a larger loss weight;
- use a format reward for closing the output properly;
- drop training samples that were truncated;
- count whether a generation stopped on EOS, on max length, or on the parser.

A function-calling run can write its own `WeightedTrainer` and give `</tool_call>` and `<|im_end|>` a larger cross-entropy weight, which trains the model directly to "stop after one call".

A math-contest run can use the shortest correct traces, a training length cap set below the evaluator's generation budget, and a final answer line, to handle truncation.

The methodology skill's cards carry the full method behind concise annealing and reward shaping; the framework skill's cards carry the trainer detail behind a custom weighted loss. This card keeps only what the workflow needs.
