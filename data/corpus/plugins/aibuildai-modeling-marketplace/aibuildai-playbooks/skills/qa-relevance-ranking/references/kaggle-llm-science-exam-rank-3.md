# 3rd place solution [Update + Code links]

Competition: kaggle-llm-science-exam
Rank: #3
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446358

First of all, I want to thank organizers for amazing competition which provided a nice chance to dive deep into LLMs and thanks to co-kagglers who shared their datasets and approaches. Specifically, thanks to @radek1 and @cdeotte for high quality datasets, which helped massively in training good question answering models. Thanks to @simjeg for showing how to do the impossible and to @cpmpml for developing these ideas and adding Xwin model weights. Special thanks to @sugupoko for sharing 70k dataset, which helped me to tune re-ranker model (the dataset contributed to qa models as well), which in turn gave huge boost to my scores. To contribute to knowledge sharing, I will publish my code, but it is going to take a few days.

Important note before going into details about the solution: I used Platypus2, but still not sure if it is allowed or not due to its license restrictions. But the main reason why I kept it in final submissions is that organizers stated that winning models are not used for any business purpose, which, I guess, implies that commercial usage restrictions in license should not be a problem.

**[Update]** Turns out it was fine to use Platypus2, but surprisingly additional ablation studies showed that slightly better scores could have been achieved without it! More details are in a table below.

# Diagram of the solution


# Some notes about the solution
## Wikipedia processing
It starts with Wikipedia dump processing, because, as many noted before, available processed wikipedia data have numerous flaws from missing numbers to missing articles. I used https://github.com/attardi/wikiextractor and modified it to solve problem of deleted numerical values. Most likely some problems still remained, but some were solved.

Search for context is performed in one stage on passage level as opposed to the search for articles first and then for sentences. The reason is that in some cases two stage search just has no chance to find necessary context. For example, in training data I saw a question about actions of a character from some play. But the title or beginning of the article about this play does not mention this character, so encoding of the article does not contain necessary information. Not sure how frequent such cases, but probably not negligible.

## Reranker training
I saw some comments in other discussions that people tried to train reranker model, but without success. From what I understood in the process of tuning my reranker, there are two key ingredients here:
1. Having candidates in pairs of (question, candidate), both positive and negative, from the same distribution as they appear during inference. In my case it was important to use wikipedia passages, that I got from wikipedia processing, and not the context from original 70k dataset.
2. Train with hard negatives. Though there is a caveat here: a raw model, that was not pretrained for reranking task, will fail miserably on hard negatives. So it is important to take some pretrained one (ibm/re2g-reranker-nq in my case) or train it in two stages.

## Ablation studies
It is interesting to see how each part of the solution contributes to overall score, but I have not done thorough ablation studies, because assembled all parts together very late. But there are some hints from earlier submissions:
| Pipeline configuration | Private  | Public |
| --- | --- | ---- |
| Inference only with 5 debertas (not final models), not tuned reranker | 0.894 | 0.893 |
| +answering 500 hard questions with Platypus | 0.91 | 0.909 |
| +replacing debertas with new ones and adding 3 electras and 5 robertas | 0.912 | 0.914 |
| +replacing reranker with tuned one | 0.927 | 0.922 |
| +adding Xwin to ensemble | 0.928 | 0.926 |

I think such results show that question answering part does its job very well if provided with correct context (boost from tuning reranker) and the weakest part of the solution is the initial retrieval. Which makes sense, since it is the part that is not tuned for competition data and also it consists of two very small models with very limited capacity, which are tasked to index huge dataset. Probably much higher score could have been achieved with tuned retrieval models.

**[Update]**
Table with measurements of each part's contribution to the final score
| Pipeline configuration | Private  | Public |
| --- | --- | ---- |
| Final submit | 0.9284 | 0.9286 |
| Without Xwin | 0.9272 | 0.9257 |
| Without Platypus2 | 0.9288 | 0.9309 |
| Without both LLMs | 0.9165 | 0.9201 |
| Without bge-small in retrieval | 0.9272 | 0.9261 |
| Without MiniLM-L6-v2 in retrieval | 0.9258 | 0.9259 |
| Without Electra and Roberta in MLMs ensemble | 0.9246 | 0.9268 |
| Without reranker | 0.9113 | 0.9130 |

<br>
Code:
[Reranker training](https://www.kaggle.com/code/podpall/3rd-place-reranker-training)
[Full pipeline inference](https://www.kaggle.com/code/podpall/3rd-place-full-inference)
