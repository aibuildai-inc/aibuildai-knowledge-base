# 49th place solution

Competition: kaggle-llm-science-exam
Rank: #49
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446355

I had not had much experience using language models before, so I entered this competition to learn about them. As a result, I was able to learn a lot about working with language models, and in addition, I was able to win my first silver medal!

However, this result wasn't achieved through my own abilities, but through the collective knowledge of many Kagglers, especially the following people.
- @radek1 for pointing me in the initial direction of Multiple Choice using Deberta.
- @cdeotte for publishing the RAG approach and providing a large amount of well-maintained training data.
- @yalickj for providing a validation dataset that correlates to LB and help model training.
- @mbanaei for providing an improved context retrieving method and pushing the competition to the next stage.

What I have done is merely a combination of these things. I did not have time and abilities to make the fundamental and cool improvements that were the major issues in this competition, such as an efficient and accurate context retrieval method, or advanced computer engineering to run huge models on limited compute resources(like @cpmpml shown).

Anyway, the approach I took was the following points.

- Multiple Choice model for DeBERTa-v3-large and Sequential Classification model for Mistral-7B are trained with 60k+40k+99k dataset with tfidf based 270k Wiki Contexts attached.
    - For DeBERTa-v3-large, I freezed the weights of the upper layers to improve the training time.
    - I originally used LLaMa2-7B as the LLM, but Mistral-7B, which was just released, performed wonderfully. I trained Mistral-7B with 4-bit quantize on sequential classification LoRA and obtained a clear improvement in accuracy.
    - Actually, a submission done with a model using only the 60k dataset as training data achieved Public LB 0.9, Private LB 0.911, which was my best, although I could not choose😅. In my case, the addition of 40k+99k training data improved the CV and Public score(0.910) but didn't improve the Private score.
- Using the DeBERTa-v3-large(Multiple choice) and Mistral-7B(Sequential classification) trained with the above methods, I performed inference separately for the two contexts, parsed and un-parsed tfidf based 270k Wiki context , and then used a simple soft-voting for ensembling.
    - In using multiple models, I took care to avoid OOM by writing all code in `.py` files and having it run as `!python xxx.py`.

Here is the notebook: https://www.kaggle.com/code/bobfromjapan/49th-solution-deberta-and-mistral-7b-w-270k-cont

Once again,  I would like to thank all the Kagglers and staff(, and my RTX4090 that kept computing for almost a whole month😀)
