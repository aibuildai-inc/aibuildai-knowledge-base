# 13th place solution,  Imitation learning with language modeling

Competition: kore-2022
Rank: #13
Source: https://www.kaggle.com/c/kore-2022/discussion/337476

Thanks for organizers hosting a nice competition.
I was really enjoined it!

<!--  -->

My initial plan was imitation learning first and then moving on to reinforcement learning. But due to the complicated action space of kore_2022, it took really long time for me to establish a successful model of imitation learning. Although my final solution was still on imitation learning phase,  I think this model could be applied to reinforcement learning.

## Overview

* Defining action space as causal language modeling(next word prediction) with "action strings tokenization".
* Model: Use encoder-decoder architecture.
* Need long training for getting imitation learning worked.

## Action strings tokenization
It was really hard to define action space for neural network at this competition. I treated this problem as NLP, causal language modeling(next word prediction) with  the following "action strings tokenization".
<!-- [action_strings_tokenization] -->
[action_strings_tokenization]
Basically I used original characters at the action strings as token, but I added some modifications like below.

* Adding fleet destination tokens,  "**HOME**" (back to launched shipyard), "**FRIEND**"(go to a friend shipyard), "**ENEMY**"(go to a enemy shipyard),"**CONSTRUCTION**"constructing  shipyard),  and "**DRIFT**"( no shipyard destination).
At this competition, these destinations are fundamental choice for each launch action. And  I think a successful model can explicitly learn such a strategy, that's why I added this destination token on the *beginning* of the launch sequence  under causal language modeling.

* Splitting num_ships into  2 token positions. This increases the total sequence length, but reduce total token dimension or ids. By reducing the token dimension we could expect faster training convergence.

* "**DO_NOTHING**" token was introduced  for separating each shipyard action.

<!--  -->

This tokenization is inspired from the object detection method, "[pix2seq](https://arxiv.org/abs/2109.10852)". At that study the coordinate of  object bounding box is discretized and then tokenized along with class label for solving object detection with language model.

## Model
* Based on the tokenization above, I can formulate the imitation learning of this competition as [image captioning task](https://paperswithcode.com/task/image-captioning), which is the mixture of  computer vision and NLP task.
<!-- [kore_2022_model] -->
[kore_2022_model]

* I basically used  the architecture of "[pix2seq](https://arxiv.org/abs/2109.10852)" . 
    * pix2seq architecture: Vision-Encoder + Transformer-Encoder + Transformer-Decoder.
    
    I modified the positional encoding from this architecture. At the vanilla transformer, positional encoding is added on only its input.  I followed the positional encoding usage of [detr](https://arxiv.org/abs/2005.12872),  at which positional encoding is added for each attention layer input(query and value).  By this modification, training convergence was improved.

* Before transformer layers, encoding kore board feature with custom [ConvNeXt Block](https://arxiv.org/abs/2201.03545) with kernel size(3, 3) and TorusConv2d. TorusConv2d was used at hungry geese [1st place solution](https://www.kaggle.com/competitions/hungry-geese/discussion/263279).

* Unlike typical image captioning,  we have positional information for the target object, "shipyard" beforehand. So I added shipyard position (x, y) and shipyard internal sequence index on decoder input.

<!--  -->
