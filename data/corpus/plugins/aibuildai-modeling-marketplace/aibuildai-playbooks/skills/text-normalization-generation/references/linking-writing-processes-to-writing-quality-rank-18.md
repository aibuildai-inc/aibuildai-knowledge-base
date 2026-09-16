# Private 20th Solution

Competition: linking-writing-processes-to-writing-quality
Rank: #18
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/466789

The result was 20th place with a solo silver medal. Thank you to the organizers. My solution consists of two parts: a decision tree-based solution with feature engineering (almost identical to the public notebook) and a BERT model that predicts from reconstructed text.
Feature Engineering: (I will omit this part as it is similar to the public notebook.)
I used features similar to the public notebook, which include:
Text Features:
char_num
Sentence statistics
Paragraph statistics
Punctuation statistics
\n statistics
\n\n statistics
etc.
Event Log Features:
Count and nunique max min fe
Statistical aggregations for all numerical columns
P and R burst fe
Pause-time agg
BERT Part:
Before using this model, I referred to the following notebook for preprocessing (https://www.kaggle.com/code/kawaiicoderuwu/essay-contructor) and reconstructed the text. Thank you for the excellent notebook.
Before handling the BERT model, text preprocessing is performed. For example:
qqqq qq qqqq (qq qqqqq qq). q'q q qqqqqq qqqqqq.
4 2 4 (2 5 2). 1'1 1 6 6.
Continuous 'q's are converted to consecutive numbers.
Then, it is trained. This allows prediction based on the sequence of consecutive word counts and features of various symbols (.,;: () '! \n \n\n, etc.).
Even if you use the public notebook as it is, the Private LB goes from 0.568 to 0.565.
model:
 deberta-v3-base
 deberta-xlarge
The predictions of these models were ensembled with weights.
