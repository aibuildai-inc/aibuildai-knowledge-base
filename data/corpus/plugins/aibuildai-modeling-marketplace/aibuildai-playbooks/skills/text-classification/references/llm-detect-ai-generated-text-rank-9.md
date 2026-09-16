# [1st Public/9th Private] LLMLab - Solution Summary

Competition: llm-detect-ai-generated-text
Rank: #9
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470255

Thanks to the Kaggle and the Learning Agency Lab for hosting this challenge. We really enjoyed this and learned a lot over the past 3 months. Thanks also to all who contributed with ideas and datasets. 

Our solution is composed of two main parts, a TFIDF based pipeline using several classifiers sitting on top of the TFIDF features, and a BERT based pipeline based on classifying human and generated texts. 

We believe the novel components to our solution include:
1. Carefully curated datasets with diversity in prompts and models varying in size from ~200K samples to ~700K samples.
2. Statistical based and reverse engineered deobfuscation pipeline. (+0.006 public/-0.001 private)
3. Cluster based post processing (~0.001)

The TFIDF pipeline is similar to the ones used in many public notebooks with the notable exception that we used a reverse engineered based approach to correct systematic spelling mistakes. A separate notebook will be published on how we handled deobfuscation. Our TFIDF also had a post processing element where we noted that pairs of texts of a high enough similarity are almost always both LLM generated.

Our BERT based pipeline consists of 2 deberta_v3_large and 1 roberta model. Each was trained on a different but overlapping variant of highly diverse datasets. The datasets were generated from a large number of open source LLM’s of varying sizes between 7-70B parameters, as well as commercial LLM offerings, and a wide variation of temperature and top_p.  Generated texts were derived from essay generation, paraphrasing and text completion prompts.  Human texts were curated from student essays and open source web text.  250k-700k total samples were in each dataset version. The best single BERT based classifier achieved 0.96 public / 0.915 private leaderboard.
