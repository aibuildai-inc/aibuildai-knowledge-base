# [1st place solution] Last Query Transformer RNN

Competition: riiid-test-answer-prediction
Rank: #1
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/218318

Hi Kagglers, here is the paper link.
Paper Link : https://arxiv.org/abs/2102.05038
Please refer the paper for details.
# Summary
I wanted to use Transformer but I could not input long history to the model because QK matrix multiplication in Transformer has O(L^2) time complexity when input length is L. 
# 
My approach is to use only last input as Query, because I only predict last question's answer correctness per history inputs. It means I will only compare between last question(query) and other questions(key), and not between other questions. It makes QK matrix multiplication in Transformer to have O(L) time complexity(because len(Q)=1, len(K)=L), which allows me to input much longer history. 
# 
In final submission, I ensemble 5 models with 1728 length history inputs.
I didn't do feature engineering much, since I can use extremely long history, I wanted the model to learn it by itself. 5 input features I used are question id, question part, answer correctness, current question elapsed time, and timestamp difference.
# Acknowledgement
Thank you @limerobot, to share amazing transformer model to approach table data problem in the 3rd place solution of 2019 Data Science Bowl. It was a good motivation to work on transformer encoder. https://www.kaggle.com/c/data-science-bowl-2019/discussion/127891 
# 
Thanks to competition sponsers and organizers for hosting this fun competition.

# 
Thank you for reading.
