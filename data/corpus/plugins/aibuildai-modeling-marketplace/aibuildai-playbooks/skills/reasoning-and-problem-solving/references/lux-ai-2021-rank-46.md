# A (hopefully) silver solution?

Competition: lux-ai-2021
Rank: #46
Source: https://www.kaggle.com/c/lux-ai-2021/discussion/294015

# Introduction
[full code](https://www.kaggle.com/bachngoh/ensemble-of-il-agents-hopefully-silver)
Hi. First of all, I want to thank the competition organizer, this has been a really fun journey for me. I have learned a lot about RL through competition (although I failed to apply those knowledges)

Since the competition is now over, I want to share my (hopefully **silver**, or bronze) solution. My agent is an ensemble of 3 IL agents, I borrowed that idea from the [lux-ai-with-il-ensemble-of-models](https://www.kaggle.com/realneuralnetwork/lux-ai-with-il-ensemble-of-models) notebook but instead of choosing the most common action, I take the softmax function and choose the action with highest probability. 3 IL models are:
- [lux-ai-with-il-decreasing-learning-rate](https://www.kaggle.com/realneuralnetwork/lux-ai-with-il-decreasing-learning-rate) 
- [toad model from the orginal ensemble notebook](https://www.kaggle.com/realneuralnetwork/lux-ai-with-il-ensemble-of-models) I tried to use many different models but somehow this one gave the best performance
- [unet immitation learning](https://www.kaggle.com/bachngoh/luxai-unet-immitationlearning-lb-1100) I was inspired by [this](https://www.kaggle.com/c/lux-ai-2021/discussion/289540) amazing post by nosound @zaharch 

I also used unet to train the city tile actions:
- [unet for ctiles](https://www.kaggle.com/bachngoh/luxai-unet-for-ctiles)

Advantages:
- Simple and quite beginner friendly
- Fast to train: the unet mode only takes 5 minutes per epoch to train

I think my solution is still quite basic. I definitely gonna try some of the wonderful ideas shared in the discussion section in future competitions (maybe LuxAI ss2 :) ).
