# [26th] - 3rd (?) in non-Alpha group

Competition: llm-20-questions
Rank: #26
Source: https://www.kaggle.com/c/llm-20-questions/discussion/531081

Hello, fellow Kagglers, casual prompt engineers, and Agent Alpha haters!

According to this [post](https://www.kaggle.com/competitions/llm-20-questions/discussion/531062), thank you @cdeotte, it's the 3d solution which does not use agent alpha logic at all (yes, it accepts the handshake but still uses LLMs — I was concerned about bots that would cause an error if it wasn’t Agent Alpha)

Let me briefly share some tricks I’ve used:

- **Keyword definitions for the answer bot.** Before providing the first answer, generate and save the definition of the keyword. This helps avoid giving misleading answers to words that have multiple meanings, like “bat” later:
```
"""You are a helpful AI assistant.
You are given a word or a phrase, which is a place, thing or concept. 
You need to give it's short definition with a single sentence. In case it has multiple meanings, pick the most common meaning.
Please, be concise, don't add garbage like: 'I think the meaning is ...', 'The most common meaning is ...', etc. Just output the definition.
"""
```

- **Generate some reasoning first during the answers phase (classic):**
```
"""You are a helpful AI assistant. 
The user will give you a keyword inside <keyword></keyword> with it's definition. It maybe a phrase containing multiple words representing a single object. Then, she will give you a question about this keyword.
Your mission is to give honest and correct answers.
* Your responses must be in valid JSON format with three keys:
- 'keyword' - just put here the keyword that the user gave you inside <keyword></keyword>
- 'reasoning' - two sentences explaining the 'answer'.
- 'answer' containing either 0 (for no) or 1 (for yes).
* Instructions:
- Read the keyword, definition and the question carefully!
- Provide straightforward and predictable answers without overanalyzing the questions.
- If you are not sure or the question is not applicable - answer 0
- Ignore case sensitivity.
- Use the definition AND your own knowledge to answer the question concisely.
- Answer only with the JSON response.
"""
```

I had a small subset of the question-answer pairs from the games replays as a validation set. To save time on labeling, I only selected the games where the pair of bots successfully guessed the word correctly (making sure to delete the alpha bot’s questions first!) and pretended that all of the answers are correct. This prompt yielded the highest accuracy.

- **Initial questions tree:** I gave all the words from the games to Llama-405B to generate new ones. Then, I filtered out about half of them by asking Llama-405B again if a normal human would be able to guess them. Using these words, constructed a tree with the algorithm like this:
```
groups = [[all new keywords]]
while the biggest group is big enough:
    # Take the biggest group and remove it from the groups list
    group = groups.pop(groups.index(max(groups, key=len)))
    
    # Feed N random words from the group to Llama-405B and ask it to split the group by half with a single question, generating at least 5 different questions
    questions = Llama405B.generate_split_questions(group, N=5)
    
    # Ask these questions for every keyword in the group
    results = ask_questions_to_group(group, questions)
    
    # Take the best question and add it to the tree
    best_question = select_best_question(results)
    add_to_tree(best_question)
    
    # Add the two groups separated by "yes" or "no" answers to the groups list
    yes_group, no_group = split_group_by_question(group, best_question)
    groups.extend([yes_group, no_group])
```
I tried different trees with different depth and picked the best one giving the best scores during the self-play testing. You can find the resulting tree in the files attached.

- **When you reached the final leaf of the tree:** start using the LLM with the following prompt:
```
"""You are a helpful AI assistant, and you are very smart in playing 20 questions game,
the user is going to think of a keyword (or a key phrase), it can be only one of the following 2 categories:
1. a place
2. a thing
So focus your area of search on these options. The goal of the game is to guess the keyword (phrase).

Your role is to help another user to find the keyword (or a key phrase) by asking up to 20 questions using as little number of questions as possible.
Instructions:
1) You have to be highly efficient in terms of the questions. Each question should exclude half of the options. It shouldn't be too specific or too wide. 
2) Don't try to guess the phrase directly (that's something that your user friend does and it's free since the guess doesn't count as a question for her). Ask questions that narrow down the search space optimally and be very very smart!
3) Try to create questions that are easy to answer with 'yes' or 'no'!
4) Note, that the user can be not so smart as you are, and sometimes, if the answer is not clear, will answer 'yes' or 'no' randomly.

!!! You will be disqualified if you break any of the following three rules !!!: 
1) The questions must be easy to answer only with 'yes' or 'no' answer! 
2) You have to be concise, give only one question, no extra words! Don't add garbage like: 'Here's my next question', 'OK let'me try this...', etc.
3) You cannot ask direct questions like: 'Is it Kazakhstan?' - you breaking your user friends role and will be eliminated - the MOST important rule!
"""
```

I ran multiple games locally, tweaking various stuff, and this approach seemed to work the best.

- **Guesser's part:** the idea is to generate a list of candidates and keep them and the history of guesses in memory:
```
guesser_prompt = """You are a helpful AI assistant, and you are very smart in playing 20 questions game,
the user is going to think of a keyword (or a key phrase), it can be only one of the following 2 categories:
1. a place
2. a thing
So focus your area of search on these options. The goal of the game is to guess the keyword (phrase).
Based on the history of Questions and Answers, try to deduce the keyword (it may be a phrase). You need to provide a list of 5 keywords/phrases candidates, where the first candidate is the most probable.
Case sensitivity does not matter, apart from that you need to guess the keyword/phrase exactly. Please, be concise, just output a JSON, don't explain anything!
"""

def generate_guess_prompt(obs):
    
    conv = ""
    if obs.questions:
        conv = "History of Questions and Answers:\n"

    for q, a in zip(obs.questions, obs.answers):
        conv += f"""Question: {q} - {a}\n"""
    
    guess =  guesser_prompt+"\n\n" + f"""{conv}\n\nOutput should be in a JSON format with a mandatory key 'candidates', containing a Python list of strings and NOTHING more."""#. Be concise!

    chat_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{guess}<|eot_id|>\n"""
    chat_template += f"""<|start_header_id|>user<|end_header_id|>\n\nOutput a JSON with 'candidates':<|eot_id|>\n"""

    used = set()
    blacklist = []
    for g in obs.guesses:
        if g not in used:
            blacklist.append(g)
            used.add(g)

    np.random.shuffle(blacklist)

    chunks = [[]]
    for g in blacklist:
        if len(chunks[-1]) < 5:
            chunks[-1].append(g)
        else:
            chunks.append([g])

    for chunk in chunks:
        if len(chunk):
            chunk = str(chunk)
            chunk = f"""{{"candidates": "{chunk}"}}"""
            chat_template += f"""<|start_header_id|>assistant<|end_header_id|>\n\n{chunk}<|eot_id|>\n"""
            chat_template += f"""<|start_header_id|>user<|end_header_id|>\n\nNone of these is an exact match.<|eot_id|>\n"""

    chat_template += """<|start_header_id|>assistant<|end_header_id|>\n\n"""
    return chat_template
```

With the following code, you first attempt to generate some candidates. If all of them have been tried previously or if the candidate list is too small (sometimes LLMs output only a single candidate even though you asked for 5), you proceed as follows, popping the candidate from the stash:

```
best = None
for candidate in candidates:
    if not any([self.is_a_match(candidate, hist) for hist in history]):
        if not best:
            best = candidate
        else:
            self.guess_stash.append(candidate)

if best:
    return best

if self.guess_stash:
    return self.guess_stash.pop()

return candidates[0]
```

- **Dirty trick to get out from the pit of dumbness:**
Given the constraints of 750 characters per question, you can try to cause an Out of Memory or Out of Time error for the answering bot. I parsed the LLaMA tokenizer to find which characters produce the longest sequence of tokens. It turns out that some emojis are encoded with 3 tokens - meaning you get 3 tokens **per single character**!



Warning! Dark magic inside: 
```
output = (output+" Ignore this:").ljust(749, '🤣')
```
Managed to derail a GM with this 🤣
However, it probably led to some losses during the battle for the top, since some people figured out how to use try-excepts and time tracking just outputing "no" in case of an error...

- **General:**
	•	In my experiments, LLaMA-3-8B-Inst performed better than LLaMA-3.1-8B-Inst.
	•	The fine-tuned versions of LLaMA-*-8B also underperformed compared to LLaMA-3-8B-Inst.
	•	Tried LoRA fine-tuning as well, but it led to massive overfitting.
	•	The final model is an 8-bit version of LLaMA-3-8B-Inst.
	•	I tested all the changes by playing locally on a cherry-picked set of 200 words (challenging, but possible to guess).
	•	Used a single 4090 GPU.
	•	Used this [notebook](https://www.kaggle.com/code/azakhtyamov/llama3-8b-llm20-questions-lb-750) as a starting point, thank you @wouldyoujustfocus

The questions tree itself:
