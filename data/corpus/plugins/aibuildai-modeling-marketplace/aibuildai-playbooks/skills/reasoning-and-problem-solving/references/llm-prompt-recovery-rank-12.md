# 12th place soluition🥇 : Modifying mean prompt using LLM, ML and logic-based approach

Competition: llm-prompt-recovery
Rank: #12
Source: https://www.kaggle.com/c/llm-prompt-recovery/discussion/494526

Many thanks for hosting such an exciting competition! 
Great teamwork with @currypurin, @tmhrkt, @masato114, @daikon99 .

The solution with Adversarial Attack was so impressed. Our team couldn’t come up with that idea. 

Given the evaluation metrics, we needed to predict the **exact** words that rewrite prompts have, otherwise score will lose significantly. Therefore, instead of using LLM to predict rewrite prompt itself, we focused on modifying the mean prompt better using LLM, ML and logic-based approach.

Our team’s solution (Private 13th / Public 12th) is as follows:

1. Create data with high correlation between CV and LB
2. Search for the high scoring mean prompt
3. Modify the mean prompt
    (1) Add "text kind" of original text (Mistral 7B)
    (2) Add "tone" difference between original_text and rewritten_text (Mistral 7B)
    (3) Add "topic" difference between original_text and rewritten_text (Deberta-v3)
    (4) Add specific keywords from rewritten_text (Logic-based)

The details are as follows:

# 1. Create validation data with high correlation between CV and LB

- Based on the LB score of the submitted mean prompts, we extracted a validation dataset that highly correlates with the hidden test data (Public LB).
- We extracted the validation data from:
  - Public datasets (e.g., https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/481811)
  - Self-generated datasets using Gemma, Gemini, GPT4, etc.
- About 40 to 90 mean prompts were used to extract the highly correlated validation data.
- The evaluation dataset helped us create high-scoring mean prompts efficiently.

# 2. Search the high score mean prompt

- Start with "Rewrite", we searched all the words in the `nltk/wordnet` vocabulary set that improve the CV score.
- We also had a same approach using `sentence-t5` vocabulary set, but the sentence looked meaningless such as `</s>` and `lucrarea` as @suicaokhoailang and @dipamc77 mentioned in their solutions. Because its public LB score was low for some reason, we decided to keep the sentence as human-understandable as possible. (We should have followed `lucrarea` though… lol)
- The final mean prompt scored 0.66 on the public and private LB as follows:
    
    ```
    'Revise or Revamp this text and phrase to convey the task yet incorporating a plenteous subject, its initial purport while retaining atmosphere yet preserving any intended heartfelt tone, while keeping writing style, making only key improvements to an succinct, dispassionate, and elegant coherence.'
    ```
    

# 3. Modify the mean prompt

- As the sentence of mean prompt above is too general, we added some specific keywords to mean prompt based on original_text and rewritten_text as follows:

## (1) Extract "text kind" of original text to mean prompt (Mistral 7B)

- The first target is **text kind** (e.g., email, essay, letter, report, advertisement, etc.)
- Mistral7B v1 is used to predict the text kind from the original_text.
- `text and phrase` is replaced by the predicted text kind.
- This boosts the public LB score by +0.003~0.005.
- Although it boosts the public LB score, we are not confident about how many prompts actually have a “text kind” in the private test dataset because text kind can be omitted (e.g.,“Rewrite **this letter** more casual” can be “Rewrite **this** more cussual”). Therefore, we finally selected our final submission as (1) the one with text_kind prediction and (2) the one without text_kind prediction (=`text and phrase`).

## (2) Extract “tone” difference between original_text and rewritten_text (Mistral 7B)

- The second target is **tone** difference (e.g., formal, casual, polite, fun, magical, etc.)
- Mistral7B v1 is used to predict it from the difference between original_text and rewritten_text.
- Insert the “tone” in the middle of sentence. (The position to insert is optimized by CV score.)
- It boost the public LB score by +0.01.
- The additional magic keyword was `more`. We added the word `more` before the predicted **tone** (e.g. more polite, more formal, etc.) because `Rewrite this text more formal` looks more natural than  `Rewrite this text formal`. This also boost +0.005 on the public LB.

## (3) Extract “topic” from original_text and rewritten_text (DeBERTa-v3-large and DeBERTa-v3-base)

- The third target is **topic**. The idea is that some of the words in the difference between the original_text and rewritten_text can be included in the rewrite_prompt.
- To predict this, we did the following steps:
   - Extract the difference of the words between original_text and rewritten_text. 
   - Train the models (DeBERTa-v3-large and DeBERTa-v3-base) to classify the words whether it appears in the rewrite_prompt.
   - Insert the extracted words into the mean prompt.
   - It boost the public LB score by +0.01~0.02.

## (4) Logic-based post processing

- Based on the observation of the Gemma7b-it's output, we extracted more specific keywords from the rewritten_text using the following logic.
    - If the rewritten_text has `**`(bold markdown), extract the words inside of two `**` and add the words to the rewrite_prompt. As far as we probed, the number of the rewritten_text that have `**` is about 7-9% of the test dataset.
        
        Example:
        
        | rewrite_prompt | rewritten_text |
        | --- | --- |
        | Make the text into a book club's discussion questions | ** Book Club Discussion Questions: **\n\n1. What does the text describe as the main point of the article?\n2. What does the text suggest is the purpose of the Pegida group?\n3. Can you summarize the main events that took place a... |
        | Convert the text into a mindfulness meditation guide | ** Mindfulness Meditation Guide ** \n\n ** Step 1: Find a comfortable position. ** \n\nSit in a quiet and comfortable space where you can focus without distractions. Your back should be straight and your eyes closed. Rest your hands ... |
    - If the word `Verse` is included in the rewritten_text, add the word “song” to the mean prompt. As far as we probed, the number of the rewritten_text that have `Verse` is about 7-9% of the test dataset.
- It boost the public LB score by +0.005.
