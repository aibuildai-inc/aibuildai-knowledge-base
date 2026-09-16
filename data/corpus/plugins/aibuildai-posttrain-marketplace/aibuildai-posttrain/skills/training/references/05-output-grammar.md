# Output grammar and structure engineering

Read in steps 1 and 5: the exact answer shape each task family needs.

Different task families really do need different "languages".

| Task family | Usual target grammar |
|---|---|
| Math contest, exact integer answer | `<think>…</think>` plus `ANSWER: 123`, or the integer answer format the evaluator names |
| Math word problems, numeric answer | Reasoning plus a clear final numeric answer; some runs use XML |
| Multiple-choice questions | The named option letter, or the answer structure the evaluator asks for |
| Code generation graded by tests | The function body inside the first Python code block |
| Function calling | A strict `<tool_call>{JSON}</tool_call>` |
| Health advice scored on a rubric | Plain natural language, but it must be safe, complete, and sensitive to context |
| Open-ended writing judged by preference | Natural language writing that obeys the length, the style, and the user's constraints |

Structure engineering covers:

- giving a clear training signal on the final-answer token;
- forcing a single tool call, not an explanation plus a tool call;
- making the JSON arguments an object;
- printing no Markdown fence unless the evaluator asks for one;
- XML opening and closing tags;
- where the code imports sit;
- keeping the final answer apart from the reasoning.

On a numeric-answer task, a GRPO reward can split into three parts:

1. a correctness reward;
2. an XML format reward;
3. a numeric-answer reward.

So even when the answer is still wrong, the model can get a denser training signal from "the format is legal" and "the answer can be parsed".

The methodology skill's card on GRPO carries the full method and its reward design; this card keeps only the output grammar the workflow needs.
