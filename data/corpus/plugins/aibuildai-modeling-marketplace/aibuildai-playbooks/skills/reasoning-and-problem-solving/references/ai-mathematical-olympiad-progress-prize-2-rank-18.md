# [Private27,Public28] Solution for 17th: To get code & fix it with DeepSeek

Competition: ai-mathematical-olympiad-progress-prize-2
Rank: #18
Source: https://www.kaggle.com/c/ai-mathematical-olympiad-progress-prize-2/discussion/573071

LLM used:
 [deepseek-r1-distill-qwen-14b-awq-casperhansen](https://www.kaggle.com/models/huikang/deepseek-r1/Transformers/deepseek-r1-distill-qwen-14b-awq-casperhansen/1)

I thought some of the questions can be solved much more efficiently by running Python code— so I explored generating code with DeepSeek R1. However, simply adding "Make python code." to the prompt did not work well.
Here is what I found that worked well with DeepSeek and QWQ,
#To get code and fix it with LLM:
Step 1: Initial Prompting

Provide the question text as a prompt to the model.
Set output length between 2000 to 4000 tokens. 
With this, get output in 1-2 min.
Output text are always truncated but no problem for further use for code generation.

Step 2: Prompt for Code Generation

Construct a new retry_prompt using the model's earlier output:
Structure:

        content = <Instruction text to make code from previous thought>
         (e.g."Please make a short summary of your approach, including python code.")
        content += question
        content += output

        prompt = [{"role": "user", "content": content}]
        prompt = tokenizer.apply_chat_template(
                            conversation=retry_prompt,
                            tokenize=False,
                            add_generation_prompt=True)

Then, add either of the following:

For better code (more verbose, with reasoning):

        prompt += '\n</think>\n'
        prompt += "Here's a structured approach to solve the problem:\n### Approach\n"

For fast (but sometimes lower-quality) code:

        prompt += '\n</think>\n'
        prompt += '```python\n'

With this, get code within another ~1 min.

Step 3: Error Fixing Prompt

If generated code fails when run via subprocess, build a new error-fixing prompt:

        content = <Some instructions to fix the code>
         (e.g."Please correct python code below and put in ```python\n{{corrected code here}}```.">
        content +=  question
        content += <Code with error>
        content += <Error message>

Retry generation with this new prompt.
Sometime works, but sometimes encounters another error and repeat this fixing process.

Notebook demonstrating code generation and error fixing using the reference set:
[https://www.kaggle.com/code/ippeiogawa/generate-code-from-deepseekr1](https://www.kaggle.com/code/ippeiogawa/generate-code-from-deepseekr1)



#Summary of strategy
- Samples:
Total of 16 samples —
→ 4 used to get code (as mentioned above)
→ 12 used to get boxed answers
- Chunked Reasoning:
Output every 2000 tokens → combine 3–4 outputs → prompt with:
"Please think again from combined previous thought."
- vLLM.LLMEngine:
Used to stream results as soon as a sample finishes.
LLMEngine Code Snippet:

        from vllm import EngineArgs, LLMEngine, SamplingParams

        # Initialize engine
        engine_args = EngineArgs(
            llm_model_path,
            max_model_len=max_len,
            trust_remote_code=True,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.96,
            seed=2025,
        )
        engine = LLMEngine.from_engine_args(engine_args)

        # Main loop
        while <Some conditions to continue generation or to stop it>:
            if prompt:
                engine.add_request(str(request_id), prompt, sampling_params)
                request_id +=1
        #
            request_outputs = engine.step()
        #
            for request_output in request_outputs:
                if request_output.finished:
                    output_text = request_output[0].text
                    # Extract an answer
                    # Process output
                    if Code in output_text:
                        run via subprocess
        #
                    # Build new prompts for generating or fixing code or to think again with combined thought.
                    new_prompt = <construct new prompt>
                    engine.add_request(str(request_id), new_prompt, new_sampling_params)
                    request_id +=1

- Early stopping: 
If 3 identical answers are produced, I treat that as the final answer and move on to the next question.

Thank you for reading!
