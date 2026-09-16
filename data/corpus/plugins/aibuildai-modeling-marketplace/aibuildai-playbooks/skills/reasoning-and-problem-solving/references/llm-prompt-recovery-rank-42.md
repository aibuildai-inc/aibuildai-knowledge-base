# 0.66+ solution - mean prompt + SFT + text_type

Competition: llm-prompt-recovery
Rank: #42
Source: https://www.kaggle.com/c/llm-prompt-recovery/discussion/494641

First of all, I'd like to thank the organizers and congratulations to all the winners!
Special thanks to:

- @richolson for sharing great ideas
- @seifachour12 for the mean prompt "Please improve this text using the writing style with maintaining the original meaning but altering the tone." which scores 0.63. Despite many attempts, I didn't improve it :)

# Datasets

I crafted two tiny datasets similar to what organizers shared:
- original_text, rewrite_prompt, rewritten_text
- original_text, text_type (text, poem, story, memo, email, etc)

# Models
- SFT Mistral-7B-Instruct-v0.2 to predict rewrite_prompt:
```
instuction = "I will provide you two texts - original text and rewritten text"
text = f"""[INST] {instuction} [/INST] Sure. Write the original text</s> [INST] {sample['original_text']} [/INST] Write the rewritten text</s> [INST]{sample['rewritten_text']}[/INST] The following prompt could be used to transform the original text into the rewritten text: Rewrite the original text """
```

- SFT Mistral-7B-Instruct-v0.2 to predict original text_type (overhead, but training pipeline was ready)
```
instuction = "Provide a text and I'll tell you it's type. I'll output only answer without explanation."
text = f"""[INST] {instuction} [/INST] Sure. Write the text</s> [INST] {sample['original_text']} [/INST] The text is"""
```

# Inference 
```
REWRITE_PROMPT = "Please improve this {text_type} using the writing style {rewrite_prompt} with maintaining the original meaning but altering the tone."
```

At the end of the competition, I tried to combine rewrite prompts predictions from 3 different models which gave a small boost.


# Didn't work well
- Extending text_types with (slogan, haiku, tongue twister, etc) and generating a prompt `Convert this {text_type_1} to {text_type_2}` if text_type_1 != text_type_2
- Larger models like Mixtral-8x7B-Instruct-v0.1 performed worse
- PPO
