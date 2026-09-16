# 4th Place Solution

Competition: kaggle-llm-science-exam
Rank: #4
Source: https://www.kaggle.com/c/kaggle-llm-science-exam/discussion/446307

First and foremost, we would like to express our gratitude to the hosts and the Kaggle team for arranging this wonderful competition. Also, a huge shout-out to all the participants for fighting through this intense competition. It was an extremely educational and exciting competition.
Below, we present our solutions.

Our solution is based on the high-quality combination of retrieval + deberta v3 models. The implementation of high-quality retrieval and how it can be incorporated as context was critically important.

# Retrieval Part
Example code: https://www.kaggle.com/code/linshokaku/4th-elasticsearch-retrieval-example

## Retrieval Workflow
- Stage 1 
   - Sentence-wise keyword retrieval
   - Using Elasticsearch  
- Stage 2
   - Version 3: Elasticsearch score
   - Version 5: Edit distance score
   - Version 7: Semantic search score 
## Validation (Retrieval)
Implemented multiple retrieval rules and evaluated scores in a zero-shot manner without intervening model training. Initially, evaluations were carried out with llama2 7b and train.csv. After the performance of the pre-trained model using context significantly improved, this model was used to evaluate using a validation dataset drawn from the 60k dataset at https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/436383, taking out 2000 instances. As no model training was inserted, it was possible to execute improvement cycles at a relatively high speed.

## Details
We believed the following elements to be necessary for high-quality retrieval:
- As much as possible, clean Wikipedia data
- Search relating to all elements of the prompt, options
- Comprehensive search  

Acquiring clean Wikipedia data is possible through the cirrussearch dump, which can be referenced from the Wikiextractor document at https://github.com/attardi/wikiextractor#wikipedia-cirrus-extractor. We built an Elasticsearch server with the texts contained in the cirrussearch dump divided into sentences and ran it on the Kaggle kernel.

The searches are always performed sentence by sentence. For each prompt or option divided into sentences, we extract all the words, eliminate duplicates and stop words, access the Elasticsearch server, and extract similar sentences. Based on the Wikipedia sentences fetched in this way, we generate three types of contexts: v3, v5, and v7. Each context involves basic processing such as concatenating surrounding sentences of the target sentence and concatenating consecutive sentences. The differences are as follows:

- v3: Sorts context based on the score at the time of Elasticsearch search.
- v5: Sorts context based on the edit distance with the question sentence.
- v7: Sorts context using semantic search, implemented using sentence-transformers.
For the embedding between the prompt and Wikipedia, we used msmarco-bert-base-dot-v5. For the embedding between the option and Wikipedia, we used all-mpnet-base-v2.

### Tips:
Elasticsearch requires writing at startup, so it needs to be copied and run on /kaggle/temp. As discussed in the discussion at https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/444742, the I/O speed of /kaggle/temp became unstable, so we added an innovation to directly perform data I/O from the file system on /kaggle/input by linking files in the Elasticsearch database that only perform reading with a symbolic link.

# Modeling Part
## Validation
Since it was important to train a model that could effectively reference context irrespective of the domain, we randomly extracted 2,000 instances as validation data from the open dataset without limiting to a specific domain. The raw train.csv was too easy for validation and was hardly useful; therefore, we did not use it much in the later stages.We were able to avoid overfitting by trusting the obtained correlation between validation scores  (2,000 samples) and LB.
Here is an illustration of the correlation between validation scores and LB: 



## training 
Model: Deberta v3 Large 
Datasets: Our main training datasets were 60k and 40k. 
Improvement in accuracy was noted by extending the number of tokens from 512 to 768 and 1280.

### Method to include context:
We realized that the performance improved as more tokens were included as context. Thus, we implemented a method where we first positioned the prompt & choices and then packed the remaining portions with as much context as possible.

## Context Ensemble
There were questions that matched the retrieval and some that did not. Ensembling the results inferred by each of the three types of contexts for a single model improved the score (Even during training, v7 only was good, but ensembling with multiple contexts during inference improved performance). This likely indicates that the diversity of the retrievers ensures the search results.
