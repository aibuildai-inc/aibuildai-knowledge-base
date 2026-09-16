# 3rd Place Solution Summary and Q&A

Competition: santander-customer-transaction-prediction
Rank: #3
Source: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88902

First of all, we would like to congratulate everyone who put in the time and effort. Kaggle isnt about the LB it's more about learnings, that's what I have learnt.


As requested in my [previous discussion](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88861#512622), I'm going to share how our team acheieved our results. To be honest, I'm still flabbergasted on how we even achieved our ranking. 


Detailed kernel and explanation will be posted by Nawid (He's the real wizard and he's also looking for a job XD). Kernel should also contain out thought process as well as discussion that somehow took place in the kernel itself instead.


So I was asked
A) how you teamed up, did you reach out, where you approached, did you already know your peoples, etc.,

To be honest, I dont know any of my teammates previously. I had this question on to merge or not to merge and I even posted [this](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/87193) to ask for opinions. In the end I took the leap of faith and merged with interneuron. Basically all the discussions were on the forum and then only thing I did was to validate that these are people who took the time and effort to figure the data and not just those who forked kernels. This isnt much but it's what I did.


B) what your thought process was like that led to your various incremental discoveries, i.e. not just what worked but how you guided yourself.
Inital breakthrough to 0.901 was that we figured out counts improved CV however LB did not changed and after rereading forums we realised that we had to remove fake test when doing so.
Initally we thought that this isnt the magic and didnt pay much attention to it and continue to try and find the magic. It was only after Nawid joining us that we realise that the actualy magic about this dataset is that each column is independent and not correlated. Our hypothesis was that data was already preshuffled within 1 and 0 which was why shuffling it again still works. We went back to discussions again and we found another interesting post about that fact that everyone jumped after the NB kernel was shared. So we tried training models on each feature independly however without much improvement. This was where we started using LGBM instead of NB on each feature where we broke the 0.92 mark.

Our last idea that gave us the push was to use a NN to blend instead of a simple linear blending as it will be able to factor in feature importance as well. This worked so well that it made us jump to 5th on the first try, however we found this too late and didnt have much time to improve it further.


C) What features did you use for the blend? also original features? or only OOF predictions? What architecture?
OOF, count, density, deviation and original input. 4 layers of Conv1D. Strides of first layer to match number of features. 4 in this case
