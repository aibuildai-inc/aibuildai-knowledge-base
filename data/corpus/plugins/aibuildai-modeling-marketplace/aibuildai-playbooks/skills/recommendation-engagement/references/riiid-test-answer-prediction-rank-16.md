# 16th Place - Single Model

Competition: riiid-test-answer-prediction
Rank: #16
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209798

Hi all,


Here is an insight of our 2 solutions that both scored public 0.812/private 0.815 and that reached 16th gold place.


This competition was both ML and engineering optimization to make everything work in 9h with 13GB RAM/16GB GPU. We spent almost 30% of time on optimization to keep the last 512 interactions per users +  per content attempts in memory + required for our features.

We would like to thank Kaggle and RIIID organizers for this great competition! Congratulations to the top teams and all competitors for their motivation all along the challenge.
I would like to thank my teammates @rafiko1, @cdeotte @titericz and @matthiasanderer. You've been amazing, I've learnt a lot from you. I really enjoyed this competition.


## Solution 1: Single transformer model

The SAINT+ model is described here https://arxiv.org/pdf/2010.12042.pdf
The code for our SAINT+ adaptation is available here https://github.com/rafiko1/Riiid-sharing.






Our single model SAINT++ achieved, CV: 0.812 Public LB: 0.812, Private LB: 0.815

We trained with 95% of users first then fine tuned with all data using smart window technique (see diagram below for SAKT). The model is simple in terms of features. It only contains the four features of SAINT+ (pictured above), with one additional feature - the number of attempts of a user for specific content (hence SAINT++):



*   Content id
*   Lag time
*   Prior question elapsed time
*   Previous responses
*   Number of attempts 

The greatest improvement in features compared to SAINT+ came from grouping lag time into seconds, unlike minutes as done in the paper. 
Then, we went bigger and bigger on the architecture and burned some GPU power 🔥. We increased on parameters of the model, most importantly the sequence length and number of layers. Final parameters of the model are as follows: 




1. Input Sequence length: 512
2. Encoding layers: 4
3. Decoding Layers: 4
4. Embedding size: 288
5. Dense Layer: 768
6. heads: 8
7. Dropout: 0.20

We used the Noam learning rate scheduler: with initial warmup and exponential decrease down to 2e-5. 
 
Final improvement came from our **_recursive trick_** during inference. Here, we rounded predictions that came from the same bundle to **_0 or 1 _**- as their true response is unknown in time yet. The rounded predictions are then fed back to the model to predict the next response within the same bundle. This trick boosts CV LB +0.0025, but requires a batch size of 1, so we couldn’t ensemble multiple transformer models.


### Solution 2: Ensemble of transformer, modified SAKT and LGB



*   LightGBM model scored CV=0.793, public LB=0.792 with 44 features.  
Our main features: 
Question correctness per content and per user, tags 1 and 2, part, elapsed time, had explanation, number of attempts, multiple lags, running average (answer), multiple rolling means/median (answers, lags, elapsed time) + weighted mean, mean after/before 30 interactions, multiple momentums (lag, answers), per part correctness, per session (8 hours split) running average. Only 3 categories: part, tags1, tags2. Train/valid split from [Tito](https://www.kaggle.com/its7171/cv-strategy). 


*   Pytorch SAKT modified scored CV=0.786, LB=0.789 with additional features. Training procedure with a smart window. 
*   When a user’s sequence length is larger than model input, i.e. N>W, then using random crops gives +0.002 CV versus tiled crops. And using smart random crops gives +0.003 CV versus tiled crops. Basic random crops have a low probability of selecting the early or late questions from a user’s sequence whereas smart random crops have an equally likely probability of selecting all questions from a user.

    


 


*   TensorFlow Transformer model alone scored CV=0.811, LB=0.811

    Same as solution#1 but with sequence length = 256



### What did not work:



*   TabNet
*   Features with lectures for LGB. It worked on CV but not on LB (might be an issue in inference).
*   Post processing using absolute position of question aka. question sequence number. Plotting mean(answered_correctly) vs question number looked like the image below. We can see that the 30 first questions have a different distribution compared with the rest. Also looks like there are subsequently batches of 30 questions (becomes visible if we zoom in the plot below). PP using that information worked in CV improving by around 0.0009, but didn’t work on LB.  

    





### What worked partially:

 
But was not applicable for us within the 9h runtime limit:



*   More than 3 models ensemble
*   Level 2 model (XGB) could boost by +0.001


### Lessons learnt:



*   Start inference Kernel as soon as possible when you need to deal with an API.
*   Try to simulate API locally to understand how data will be handled. [Tito](https://www.kaggle.com/its7171/time-series-api-iter-test-emulator)’s simulator was perfect for that purpose.
*   Push your inference (with the simulator) to the limits to debug it, it will avoid the frustrating “submission scoring error”. 
*   Team-up at some point, your teammates always have good ideas.

One additional word to Kaggle @sohier I loved your API and the way it hides private data, it’s more realistic as in real world usage/production and it avoided chaotic blending. Congratulations for that, however, even if I guess you want to prevent probing, you should find a solution to provide better error feedback. If it is not possible (the more error codes the more probing) then you need to provide a simulator and guidelines allowing competitors to troubleshoot locally.
