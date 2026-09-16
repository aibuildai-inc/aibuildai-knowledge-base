# Public Top3 -> Private Top15 Solution

Competition: kaggle-llm-science-exam
Rank: #15
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446816

What a big surprise！ But we still lost the $10000 prize. It looks like overfit. From Top3 shake to Top15.
    To be honest, we focus our all attention on RAG，and just use 2 deberta to do MutilChoice. So the overfitting problem may be in the deberta(not like other's LLM-7B).
    My English is not very good, so some of the content is translated from Chinese using a translator。🤣🤣🤣🤣
    We have a 5-way recall strategy. It can be divided into 2 parts, Sentence Model or TFIDF.
**Here's a flowchart of our overall program.**

**Part1. Sentence Model:**
    We use simcse to train our Sentence Model with SFT, it improves us LB 0.015. **And specially, We use a trick called Difficult Sample Comparison Learning.** We train our Simcse firstly, and use this model to inference our train dataset. This will produce Top5 recall, then we put Top4(except the ground truth) to Simcse Loss as negative label. This will give our model a stronger textual representation. And this gives our model an additional 0.005 LB lift over normal Simcse.
**Note that all the Sentence Models we use are trained with simcse**
**Part2. TFIDF**
    Just like open-source notebook. **And specially, we did a speedup on TFIDF and it only took 20min.**

**In particular, since we found that the sentence-model cannot encode all the text of a wiki article, TFIDF or BM25 or LGBRanker are several effective complementary schemes to it, since they can all encode to all lengths of text.**

Then we have 5-way recall strategy.

1.Dataset 6800K wikipedia. We use sentence model to recall Top1000 wiki articles. And we use Bm25(or a LightGBM Ranker) to reorder the Top1000 articles, only remain Top30 articles. Then split the articles and use our sentence model to get Top20 sentence. This way's LB score is 0.885.

2.Dataset 270K wikipedia(2100K paragraphs). We use sentence model to recall Top5 paragraphs. 

3.Dataset 270K wikipedia(2100K paragraphs). We use TFIDF to recall Top8 paragraphs. 

4.Dataset 270K wikipedia(2800K paragraphs). We use sentence model to recall Top5 paragraphs. 

5.Dataset 270K wikipedia(2800K paragraphs). We use TFIDF to recall Top8 paragraphs.

    Interestingly, we slice the dataset into multiple indexes and then recall them individually and then sort them according to search_score to get Top5. this solves the problem of faiss taking up too much memory. 🤣🤣🤣🤣🤣🤣


**Some miscellany**
    **This is a Discussion Competition!**
    It was a very competitive competition. But there was also a lot of great open source work that completely changed the competition. MB's 270K dataset is undoubtedly excellent work, but it also completely disrupted the game in the last half. After that, all the trick can be found in DISCUSSION.
   Thx to [MB's notebook](https://www.kaggle.com/code/mbanaei/86-2-with-only-270k-articles), this leads us to fusion on multiple datasets.
   Thx to [william.wu](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/442595#2462052), this leads us to fusion on Faiss and TFIDF.
   Thx to [MGÖKSU](https://www.kaggle.com/code/mgoksu/0-807-sharing-my-trained-with-context-model), it is a wonderful baseline!

    This is the first kaggle competition I've been seriously involved in, thanks to my teammates and the organizers, and thanks to everyone for the open source work!
**Life is full of surprise!**
