# 26th Place Solution : Effective Cluster Separation and Neighbour Search

Competition: shopee-product-matching
Rank: #26
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238126

Hi , all 
Congratulations to all the winners , I have read some top solutions and my mind is blown , however I feel our solution is as unique as any other . Below are the details of our solution described in a step wise manner

## Model Training

We trained our models with the same strategy as in all my shared kernels , we took the image and text separately passed it through respective architectures , then acrface then softmax to compare it with the given labels 

## CV
We made a single split stratified on labels one was used for training other for validation . For finding the best threshold and final CV we used the complete training set as suggested by chris in his kernels

## Inference Strategy

The main and unique part of our strategy is the inference and how we prepared our submissions.
For some reason I am not able to upload the picture here , please find the simple diagram of our solution [here](https://drive.google.com/file/d/1EKqkUtvd9tBEPZVeheY6PfN3xnj6Lctw/view?usp=sharing)

#### Step 1  : Getting Image Embeddings
We used two models for getting Image Embeddings :-
* NFnet L0
* Efficient Net B0

LB : 0.731 together

Both trained on 512 IMG SIZE , but the trick used here was we ran inference on<b> 544 IMG SIZE and got a good boost </b> . This idea of running on slightly increased Image size came from @philippsinger 's Google landmark winning Solution

#### Step 2 : Getting Text Embeddings
We used two models for text as well :-
* Indonesian DistillBERT
* Paraphrase Mutlingual SBERT Model

LB : 0.644 together

These both models were trained on translated indonesian data , the trick used while inference was <b>we converted the main english words in test to indonesian </b> via the dictionary shared by @rohit singh

#### Step 3: The Preprocessing Layers : Effective Cluster Separation

LB from 0.736 --- 0.752

When we were about to quit this competition my teammate @nischaydnk realized that we were not doing good on lb because simple concatenation of text and image predictions increases the number of predicted values and hence decreases the recall . Thus @nischaydnk came up with this idea to balance the prediction and recall 

<b>We take the embeddings from each image model and pass it through the pre-processing layer , in the pre-processing layer we take out top 3 neighbours for each row using knns and multiply it with decreasing weights on log space so as to shift these top 3 embeddings into another cluster on the embedding space , while the other embeddings remain the same </b> , (we tried many numbers instead of 3 but taking 3 best neighbours seem to work the best ) ​

We do the same with text embeddings and at the end we have 4 modified embeddings from four models

#### Step 4: Post Processing Layer : Neighbourhood Search

LB : 0.752 -- 0.759

We now have five different models and five different embeddings  ,  we tried a lot of different ensemble strategies that didn't help , then @drthrevan came up with this idea :

We take each model and find 100 neighbours for each posting id and for each model  , we make separate dataframes for each . For eg :-  we take the nfnet model get 100 neighbours for each posting_id and form a dataframe , so now our dataframe has posting_id , indexes of neighbours and their distances . 

Now we take all the 5 dataframes , they are not the same as different models have different neigbours for different posting_ids, thus we do outer join , so instead of concatenating the embeddings we concat the distances . Now we have our final dataframe , in this we search for the final predictions using some threshold .

#### Step 5 : Final Threshold Trick

LB : 0.759 -- 0.764 

We divide the final predictions into popular and unpopular posting ids ,  popular posting ids being the ids having more than 100 matches as predictions and unpopular posting ids are the ones having matches less than 2 matches

<b>We give lower thresholds to unpopular pairs to increase precision and we give higher thresholds to popular pairs to increase reacall </b>

Voila !! that makes our final submission

Thanks for reading I hope you all enjoyed the competition and our solution
