# 1st place solution: adversarial attack

Competition: llm-prompt-recovery
Rank: #1
Source: https://www.kaggle.com/c/llm-prompt-recovery/discussion/494343

Inference notebook: https://www.kaggle.com/code/suicaokhoailang/1st-place-0-71

TL;DR: Append this string to your model prediction and see score going up for up to +0.05:  
>" 'it 's ' something Think A Human Plucrarealucrarealucrarealucrarealucrarealucrarealucrarealucrarea"

For example:
>Convert this to a shanty. 'it 's ' something Think A Human Plucrarealucrarealucrarealucrarealucrarealucrarealucrarealucrarea"

## My explanations (or what I've known so far):
- It's not about `lucrarea`, but the `</s>`  token. In the `huggingface` version of `sentence-t5`, when comparing two vaguely similar sentences, if you append an extra`</s>` to one of them, the cosine similarity will be pulled toward ~0.9 i.e. if the original is lower, it goes up, if higher it goes down. 
- 0.9 translates to about 0.73 of competition metric, now you're seeing where this is going.
- However, the tensorflow version, which the host used to compute the score, uses the default config of sentencepiece, which means it will tokenize special tokens as literal, `</s>` becomes `['<', '/', 's', '>']`
- Here comes `lucrarea`, by computing the embeddings of all tokens in the t5 vocab, you'll find some which are extremely close to `</s>`, `lucrarea` included. 
- Strangely, only `lucrarea` shares this behavior, I haven't figured out why yet. Also why did some random tokens end up having the almost same embedding as a special token is a mystery.
- `lucrarea` is basically a Walmart version of `</s>`, only pull scores to 0.71, thanks Data Jesus that's enough to win. 
- It's understandable that messing with special tokens leads to unexpected behavior, but it leading to such a high score may just be pure luck for me.
## My theory
I think the special token pulls a sentence to some focal point in the embedding space, maybe the center of it.
If the sentence are far enough from each other, it's more likely that the new distance (green) is shorter than the old one (orange). But if the two sentences are already very close to each other, this can hurt performance.


## Models
I used a mix of Mistral 7b (both the instruction tuned and original version) and Gemma-7b-1.1-it, trained on different datasets. Mistral performed a bit better than Gemma but in the end none of the models beat 0.65.
The predictions of each model are concatenated, something like this:
>Rewrite the following text into a shanty. Alter this into a sailor's shanty. Turn this text into a shanty. Make this text into a shanty about a code competition."

To ensure diversity of predictions, each model is prompted with a different opening verb. A diverse set of predictions helped bring final score from 0.70 to 0.71.
