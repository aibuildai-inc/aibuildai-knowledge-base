# 7th place solution: T5 decoding, iterative mean refinement, LLMs and a gold medal mean string

Competition: llm-prompt-recovery
Rank: #6
Source: https://www.kaggle.com/c/llm-prompt-recovery/discussion/494755

First, thanks to the organizers for running this competition!

My solution consists of four parts:
1. Iterative mean prompt optimization (0.69, gold medal string).
2. Two LLM predictions.
3. Finding an optimal point (in T5-space) between mean prompt and the two  predictions based on the predictions' agreement with one another.
4. Decoding the point to a string.

The first one is done offline as a preparation, the other three are done online.

## 1. Iterative mean prompt optimization

People quickly realized a well-constructed constant prediction can score highly on the LB. To find an optimal string, I started from a known, good string ("Improve this text ..."), made a guess to where the mean of the (public) LB is, moved the string towards that mean:

### Guessing the mean of the LB
When you submit a constant prediction to the competition, it returns the mean distance of that string to all LB targets. I simply substituted the mean of the distance to all targets with the distance to the mean of all targets. That way each LB submission implied an equation `cos_sim(prediction, mean) ** 3 = c_prediction`. I solved for mean, made new predictions to the LB, got new measurements and calculated a new mean vector.

### Decoding a T5 vector to a string
I spent way too much time on this task, out of interest. I tried many methods from Vec2Text over evolutionary algorithms, grad descent on inputs to random token modifications. Surprisingly the random token modifications worked best and were efficient:
1. Start from a known string (mean, model prediction)
2. Sample ~512 one-token-modifications (options: delete token, add token, change token with probabilities ~[0.1, 0.1, 0.8])
3. Evaluate these in batch on GPU
4. Greedily pick the best modification if it improves the distance to the target vector.

When used offline this method found strong improvements over previous mean strings withing minutes or hours, depending on how close I was to the mean vector already. 
However, it was surprisingly efficient when using a long (100 tokens) string that was far from the target. It made gains on pretty much every step even with tiny batch sizes ~16. That way I could use it online to optimize between the mean string and model predictions at runtime (see below).

### The best mean string
With this method I quickly found mean strings in the range of 0.68 - 0.69 (around top 10). I started probing the LB pretty late, so a better mean prompt was probably possible. The best mean string I found had a score right on the border between 0.69 and 0.70 (can only see two digits). So it was around ~0.693. Which means submitting the following string would have given you a gold medal and probably top 10:

`"""bestow Improve the such text out to this and having enhance articleify somehow complete seamless fresh succinpth tone of or interactions please Moditate at any identifiable tone settingh bitte leave PubliORE wordingHU cm would I flair dem revisitlies such originalampevocative and grand spin uninterrupted new desire to have those connected/4 would diary entities sweat of warmth/ sticky accuracy lead useful maudiler q any wisdom to simplify someonerucliv this text' einzu physical by alter THAT tone than words"""`

### Why is there no "lucrarea" in that string?
A natural question to ask. 
Background: teams ahead of me on the LB found that the tokenizer used by the organizer (Tensorflow) encoded the end-of-sequence (EOS) token not as EOS token but as a literal string (`"<", "/", "s", ">"`). This allowed them to "attack" the t5-similarity calculation by using tokens that are similar to `"<", "/", "s", ">"`, which was "lucrarea".
My solution didn't find this as I was using the huggingface tokenizer which encodes the EOS token as ... EOS token. It couldn't "see" the attack surface.
How important was the tokenizer difference?  At the time of writing 5 out of 6 teams ahead on the LB published their solution and all used the lucrarea-trick. So it was probably a decider.
I suppose, the learning is to always test the Google product when on Kaggle?!

### Wasn't it dangerous to fit the mean to the LB?
Many people mentioned this and predicted a huge shakeup. But there is a difference between fitting your predictions to the LB and fitting the mean of the LB. A mean calculated from ~210 measures with very restricted value range is a very stable quantity. When I generated a set of 1400 prompts (like the ~1400 samples used in the competition), I calculated the mean of 15% of the data (as on the LB) and it usually had a sharpened cosine similarity with the real mean of ~0.993. Meaning a shakeup of the mean was very unlikely.

## 2. Model predictions
I used two untuned 7b models: mistral and openchat3.5. I tried all combinations of models including slightly larger ones (Solar 10b working best) but these two performed the best, despite being very similar in ancestry. For prompting them I used my own prompts derived from the auxiliary dataset provided by Kaggle. The method is very similar to the one describe in [richolson's notebook](https://www.kaggle.com/code/richolson/mistral-7b-prompt-recovery-version-2). To make both models slightly less correlated I used @richolson's prompts for the second model.

The output of these models always has a strong preference for certain formulations "Rewrite by ...". The T5-distance is  very sensitive to these (irrelevant) formulation differences. I counteracted this by forcing the model to begin with "Rewrite this text", replace that prefix with ~60 variants and taking their (T5-) mean. The resulting two vectors plus the mean string above were inputs to the final part:

## 3./4. Finding an optimal string between predictions and mean
I calculated the agreement between the two model predictions via sharpened cosine similarity and chose a weighting between mean and the predictions as `[1-agreement, agreement/2, agreement/2]`. The exact agreement calculation varied a bit over time, but it didn't matter that much since when the agreement is low, the model can't find a much better point than the mean anyway (as the two measures are in "opposite" directions).

The resulting T5 vector was "decoded" using the iterative method described in 1.. The model started from `mean_str + prediction_model_0` and iterated for 200 epochs with a batch_size of 16 at runtime. This way I spend around ~3hours on generating the predictions from two models and 5h-6h on positioning the final string.
This method moved smoothly between mean and predictions while being very safe: it usually moved closer to all three points at once (768-dim space!).
Examples:


