# [1st Place Efficiency Prize] Scientific Journal

Competition: llm-detect-ai-generated-text
Rank: #40
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/471898

My efficiency solution is based on two wonderful notebooks created here
- https://www.kaggle.com/code/datafan07/train-your-own-tokenizer by @datafan07. 
- https://www.kaggle.com/code/siddhvr/llm-daigt-sub by @siddhvr 

**Quick rundown**
  - A custom Byte-pair encoding tokenizer on the “public + private” test dataset.
  - Train a TFIDFVectorizer on the tokenized test set
  - Train three classifier models “MultinomialNB, SGDClassifier, and LGBMClassifier” on top of the TFIDF vectors and
  - Perform an ensemble using a VotingClassifier of the above-mentioned 3 classifiers.
  - External data: kids-frontier + daigt-v2-train-dataset by @thedrcat

My unique solution for this competition is to curate a good dataset for this competition, I have tried very advanced NLP/LLM models, techniques/tricks that u can think of, but none works on public LB. My hypothesis is that this competition needs a dataset of similar distribution e.g. the one shared by the community and especially daigt-v2-train-dataset. My final CPU submission was just a simple baseline based on @datafan07 <a href="https://www.kaggle.com/code/datafan07/train-your-own-tokenizer">work</a>, and just incorporate my carefully curated data. Adding other data causes severe data drift which further increases my gap between CV/LB.

I felt that having a strong educational dataset of a similar distribution is crucial, hence I invest most of my time to find a really good educational dataset especially written by students of various grades with a focus from grade 1 - grade 13. A collective effort has been started from this competition <a href='https://www.kaggle.com/competitions/commonlitreadabilityprize'>CommonLit Readability Prize competition</a>. In that competition, as part of the training data, the host uses some excerpts from “kids.frontier.org”, hence I started exploring what kind of dataset these are, to my surprise this is a very good essay dataset that has yet to show its true potential. I did not share these datasets prior to this competition, as I believe this is somewhat a very strong dataset in future NLP competition from The Learning Agency Lab..

**External data:**
1. <a href="https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset/data">daigt-v2-train-dataset</a> by @thedrcat 
2. <a href="https://www.kaggle.com/datasets/xyzdivergence/kf-dataset">Kids Frontier scientific journal</a> 
3. <a href="https://www.kaggle.com/datasets/xyzdivergence/kf-data-source">Kids Frontier source</a>

**What is <a href="https://kids.frontiersin.org/articles/">Kids Frontier</a>?**
Frontiers for Young Minds" is an open-access scientific journal platform uniquely designed for children. It features articles written by scientists and is reviewed by a broad range of young people before publication. 

Distinguished scientists are invited to write about their discoveries in a <u><b>language that is accessible for young readers</b></u>, and it is then up to the kids themselves – with the help of a science mentor – to provide feedback and explain to the authors how to best improve the articles before publication. 

It covers scientific journals from the following domains 1. Astronomy and Physics, 2. Biodiversity, 3. Chemistry and Materials, 4. Earth Sciences, 5. Engineering and Technology, 6. Human Health, 7. Mathematics and Economics, 8. Neuroscience and Psychology.

I have collected around 1k articles/journals raw texts from “kids.frontiersin.org". I have used the final generated article/essays/journal from the following **prompt2** only. The original article/journal is only used as a reference and possess a “Creative Commons Attribution License (CC BY)” 

-	**gpt-3.5-turbo-1106 API** is used for generating the final text, using the following two prompts.
-      **prompt1**: "Given the following text what are some questions to ask? Generate at least 40 different questions. \nText: {text}"
-	**prompt2**: "Given the following article, Summarize the question '{question}' as a {grade_level} grade student in less than 500 words. The summarization can be either complete, incomplete or partially complete. \nArticle: {article}" (generated 60k samples, only 30k used for training)


A sample of a journal article can be found at: https://kids.frontiersin.org/articles/10.3389/frym.2023.1215124

<b>Public LB</b>: 0.947061, <b>Private LB</b>: 0.91967, <b>CPU Inference time</b>: 17mins

**References**
- https://www.kaggle.com/code/datafan07/train-your-own-tokenizer 
- https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
- https://www.kaggle.com/code/siddhvr/llm-daigt-sub
- https://kids.frontiersin.org/
- https://www.frontiersin.org/news/2017/07/05/frontiers-for-young-minds-using-frontiers-for-young-minds-articles-in-your-classroom/

**Code**: https://www.kaggle.com/code/xyzdivergence/llm-daigt-sub/notebook?scriptVersionId=153307051
