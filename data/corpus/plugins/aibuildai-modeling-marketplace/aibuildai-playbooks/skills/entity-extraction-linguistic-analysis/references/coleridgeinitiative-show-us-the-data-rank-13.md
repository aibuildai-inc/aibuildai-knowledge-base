# 13th place solution

Competition: coleridgeinitiative-show-us-the-data
Rank: #13
Source: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/248541

#### **Most Important Key**
For me the most important key to this competition was Khoi Nguyen’s (@suicaokhoailang) [discussion topic](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/232964) that encouraged the removal of predictions similar to the provided training labels before submission in order to better assess the ability of a solution to generalize.

#### **General Thoughts on Approach**
Since we were provided with noisy, incomplete labels for training, I leaned away from methods that required labels for training. Actually I leaned away from training altogether except for playing around a bit with [skweak](https://github.com/NorskRegnesentral/skweak), which I probably should have stuck with in retrospect.

#### **Solution**
[Notebook](https://www.kaggle.com/trentb/coleridge-dataset-extractor?scriptVersionId=66770265) (runs in 20 - 25 minutes)
##### **Extractor Functions**
Through some exploratory methods, it appeared that words like Study, Survey, Database, Dataset, Archive, Assessment, Catalog, Collection, Registry, and Initiative appeared frequently in presumed dataset mentions. For each word, I created a function to examine words around each of these words to extract dataset mentions. While I initially considered doing the extraction with regular expressions, I discovered the versatility of [SpaCy’s sentencizer](https://spacy.io/api/sentencizer/) and leveraged it. The sentencizer permitted an easy, intuitive, pythonic way to examine tokens appearing before and after words, including built-in functions for checking tokens for case combinations and sentence positions. I added a good bit of nested, complex conditional statements, but it was debuggable and maintainable because the code was much more readable than regular expressions. And it was still really fast, which allowed for a lot of developmental iterations.

I iteratively developed these extractor functions by running them on the training data, ranking the extracted dataset mentions by their document frequency, and modifying the functions to make cleaner extractions with more focus put on the dataset mentions with higher document frequency.

##### **Frequency Filtering**
After combining the outputs of all extractor functions, I ranked them by their document frequency and retained only the top 95 percentile. The rationale behind retaining the top 95 percentile was that infrequent dataset mentions were less likely to be important to the sponsor’s objective and, hence, less likely to have been labeled in the testing data.

##### **Final Dataset Search**
The last step was to search through the documents again but with a slight twist. The list of high-frequency dataset mentions was sorted by decreasing length. For each document, the sorted list was iterated through, replacing matches with Xs. For example, in theory, if “ADNI” and “Alzheimer’s Disease Neuroimaging Initiative (ADNI)” were both in the list and only “Alzheimer’s Disease Neuroimaging Initiative (ADNI)” was in the document, this method prevents predicting both “ADNI” and “Alzheimer’s Disease Neuroimaging Initiative (ADNI)”. To explain further, because it’s longer, “Alzheimer’s Disease Neuroimaging Initiative (ADNI)” would be found first, saved as a prediction, and Xed out. When the shorter “ADNI” is subsequently searched for, it will not be found. Only “Alzheimer’s Disease Neuroimaging Initiative (ADNI)” will be predicted as desired.
