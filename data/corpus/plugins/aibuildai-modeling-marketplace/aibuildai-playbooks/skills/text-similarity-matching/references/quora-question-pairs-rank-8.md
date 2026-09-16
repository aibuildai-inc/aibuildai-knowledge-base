# 8th solution with part of source code(under construction)

Competition: quora-question-pairs
Rank: #8
Source: https://www.kaggle.com/c/quora-question-pairs/discussion/34371

Finally we have more than 1000 features in all.

Ok lets get a little bit more technical:
# Overview of solution
## Preprocessing
1.remove punctuations

2.using porter stemmer

3.generate unigram bigram phrases of stemed courpus

4.generate distinct unigram bigram phrases of stemed courpus


## qian's features
1.count/ratio of words/char of questions 

2.count/ratio of common words

3.jaccard/dice distiance

4.count/ratio of digits or punctuations in questions

5.tfidf of raw corupus with nrange=(1,2)

6.tfidf of unigram/bigram

7.tfidf of distinct words' unigram/bigram

8.tfidf of cooccurence of (distinct) words unigram/bigram

9.gensim tfidf similarity

10.similarity of self/pre-trained wored2vec weighted average embedding vectors(idf as weight)

11.similarity of self/pre-trained glove weighted average embedding vectors(idf as weight)

12.tfidf decomposition by NMF,SVD,LDA using sklearn

13.similarity of distinct word pairs in q1 and q2 using self/pre-trained 
wored2vec/glove,aggregated

14.num of nodes belong to cliques

15.sklearn tfidf similarity

16.deepwalk embedding of question as nodes

17.using label to encode cooccurence distinct words and aggregation by mean max min std

18.fuzz_feature

19.NER by spacy

20.simhash of unigram/bigram

21.decomposition of adjacency matrix

22.glove weighted average embedding vectors(idf as weight)

23.aggregation of size of cliques of each node

24.average neighbour degree

25.aggregated distinct words by wordnet

26.(distinct words)entropy based question representations


## fengari's features
1.decomposition features of ngrams : nmf + svd + lsi +lda

2.decomposition featues of diff ngrams :nmf + svd +lsi + lda

3.similarities and distances of decomposition features above

4.maxclique features of edges

5.maxclique features of nodes

6.bfs (depth =2) cnts of graph

7.duplicated features ( with ranking )

8.number diff feature among question pairs

9.pagerank (directed/undirected)

10.tsne of all leak features

11.doc2vec and doc2vec sim features

## hhy's features
1.similarity and distance of pre-trained glove weighted average embedding vectors

2.decomposition features of w2v

3.duplicate feature

4.nlp stats features(contains token log prob,brown cluster,pos tag,dependency,entity,subject,verb,object)using spacy

5.dependency tree feature using stanford nlp utils

6.wordnet similarity feature

7.stop words basic stats feature and char distribution (with tf)

8.word move distance

9.ngram extra features(contains BLEU metric,indicator,pos_link,postion change,and pos tag compare)

10.decomposition features of ngram extra features : nmf + svd

11.neighbor basic feature

12.neighbor semantics similarity

13.neighbor distance compare :long match + edit + jaccard + dice + word move distance

14.neighrbor combined with nlp basic features compare

15.deeplearning model: siamse + siamse_match + bimpm

## qian's base model and metafeature
base model type 1: lgb,xgb,et,rf,mlp with basic feature + decomposition features

base model type 2: lr,linear svc with clique weighted basic feature + tfidf features

base model type 3: lstm,attentive lstm siamese


## final stacking
We using the same 5 fold stacking

stack level 1 : lgb、xgb、mlp with dense feature、mlp with sparse tfidf weighted feature、et、rf and so on.

stack level 2 : we use lgb、xgb、mlp、rf and et, and our final submit was a simple avg of those models.




PS. We have included our features and stacking models.
More deep models will be released.


https://github.com/qqgeogor/kaggle-quora-solution-8th
