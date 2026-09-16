# 7th Place Solution

Competition: stable-diffusion-image-to-prompts
Rank: #7
Source: https://www.kaggle.com/c/stable-diffusion-image-to-prompts/discussion/410618

Thanks to all for the great competition, and especially to my teammate @evilpsycho42. Don't let his "evilpsycho" username fool you--he was a fantastic teammate, and we had a great Slack conversation throughout the competition as we discussed our strategy and commiserated together when many of our best ideas failed to work as well as we hoped.

Here are some key elements to our 7th place solution:

**Data Generation**

We assumed the training data would play a major role in this competition, so we started generating custom images using Stable Diffusion as quickly as possible. We tried to gather a diverse set of prompts, generally limiting them to have a cosine similarity of less than 0.7 using the competition metric. We filtered what we thought we the highest quality, low correlation prompts from the following sources:
* Diffusion DB
* Conceptual Captions
* COCO image captions
* Flickr Image Captions
* The ChatGPT prompts from @jeinsong (https://www.kaggle.com/datasets/jeinsong/chatgpt-images-w-prompts)

Most importantly, we designed a custom prompt generator that we found to be very effective. It formed prompts from random combinations of subjects (40k people, places, and things), artistic media, adjectives, and verbs.

**Validation Set**

I'm sure most of us struggled at first to find a good validation set. We found it was important to ensure that our validation prompts came from a variety of sources and had a low correlation with our training data. 

**Models**

Our best models were CLIP vision towers, especially `'convnext_xxlarge.clip_laion2b_soup'`  and `'eva02_large_patch14_clip_336.merged2b'`, and we also included a few of the smaller ViT and convnext models from open_clip/timm in our final ensemble.

While these CLIP backbones had the best single model scores, we found a robustness benefit from also including a small weight on a zero-shot image captioning model in our final ensemble, though not a large difference which one we used. Our top model uses the GIT model captioning model (https://arxiv.org/abs/2205.14100)

**Head Pretraining Trick**

Since we were using CLIP models, we found a pretraining trick that allowed us to initialize the model's projection head that maps the CLIP backbone output to the 384 embedding used by the competition. We used our custom prompt generator to generate a million random prompts, and we created their text embeddings with the CLIP model's text tower as well as the text embeddings using the competition metric (`all-MiniLM-L6-v2`). Then we trained a simple linear layer that mapped the CLIP embedding to the competition embedding (e.g. `nn.Linear(in_features=clip_embeddings_size,out_features=384,bias=True)` )

We used this as the pretrained weights for our model's projection head, which added information and performed better than using random weights to initialize the head.

**Training**

We tried a few different variations of training techniques, and our favorite runs generally included:
* 2 epochs
* layer rate decay
* very little image augmentation (small amount of random/resize crop)
* large amounts of gradient accumulation to simulate large batch sizes

We found the WiSE-FT technique was very effective in improving single model scores, but not helpful in our ensemble. 

**Single Model LB Scores**

In case you are curious, here are the LB scores of our top models:

[private_lb_score]

**Conclusions**

Thanks to the organizers, and congrats to the other contestants!

**Edit May 18**
Based on comments below, we shared the following, which I will add to the original post.

- The prompt generator is here [https://www.kaggle.com/datasets/rturley/custom-prompt](https://www.kaggle.com/datasets/rturley/custom-prompt) with sample code in this notebook [https://www.kaggle.com/code/rturley/custom-prompt-generator-for-image-generation](https://www.kaggle.com/code/rturley/custom-prompt-generator-for-image-generation)

- The first 100k images that came from this prompt generator is in this dataset [https://www.kaggle.com/datasets/rturley/stable-diffusion-100k-custom-prompts-and-images](https://www.kaggle.com/datasets/rturley/stable-diffusion-100k-custom-prompts-and-images) with sample code in this notebook (https://www.kaggle.com/rturley/tour-of-stable-diffusion-custom-prompt-dataset)[https://www.kaggle.com/rturley/tour-of-stable-diffusion-custom-prompt-dataset]
