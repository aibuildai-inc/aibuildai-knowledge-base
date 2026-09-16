# 1st place solution

Competition: google-universal-image-embedding
Rank: #1
Source: https://www.kaggle.com/c/google-universal-image-embedding/discussion/359316

So the competition is finalized now, and we are so glad to get the 1st place in both public LB and private one. We really appreciate the amazing competition held by the host @andrefaraujo and the support provided by Kaggle staffs @maggiemd Contestants who share their wonderful ideas and resources also help me a lot. I may forget to tag some persons but I want to say you guys did really a wonderful job! 

## TL;DR

Here I will not only share how we get the final performance, but I also go through the whole competition, for what difficulties we faced and what approaches we chose to overcome, in the order of time. I hope it will help people better learn the practical things one may face and alternative ways to get over them in future competitions.

##Get started - The first month of the competition

Honestly We felt doubtful about joining this competition or not. It is so rare for both of us to see a competition w/o a provided datasets. But we finally took it as a challenge for us to get over this task, and decided to join in (Luckily we did so!). 

At the beginning, @khabel and @skyyy93 were way ahead of the rest of people in LB, making us eagerly want to know what general orientation they were walking toward to lead to such a giant score gap. We decided to go through the discussion carefully to find some tips or hints to reach a superior score. https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/340043 and a deleted thread claims high score of only using pre-trained model, even like 0.47+, but not claiming the actual name of the pre-trained weights. At the same time we found some thread like [https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/338505], indicating their bad performances after training or fine-tuning. So the first idea was assuming maybe @khabel and other contestants with higher ranking just uses pre-trained weights w/o training or fine-tuning. So we decided to collect powerful pretrained weights first. From the https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/340043, we can tell that larger datasets result in better score, since weights trained on ImageNet-22K is better than 1K ones. The best weights of ImageNet-22K gives 0.405, so the 0.47+ one should be trained on much more larger datasets. Searching pre-trained weights on super large datasets, we found a wonderful work CLIP[https://arxiv.org/abs/2103.00020]. For the reproductive code, [https://github.com/mlfoundations/open_clip] was found to be a good choice to begin with. We tried almost every available weights for VIT-L, realizing the best performance come from Laion-400M 31ep, with a score of 0.499. We thought we almost find the most powerful pretrained weight, which still have a gap of like .050 to @khabel Considering notebooks and discussion like https://www.kaggle.com/code/moeinshariatnia/random-vs-avg-max-dim-reduction, we want to find whether some tricks may help. We found that random projection works better for weak weights like efficientnet-b0 trained on ImageNet-1K, but harm the performance of strong weights like VIT-L Laion-400M 31ep. The rest methods like PCA, t-SNE is not working in our experiments. We did find some seemingly robust methods to improve the score w/o training, simply using less values from embedding vectors to calculate the mean will boost ~0.010 on both public LB and private LB. The reason we guess is that calculating average from a large amount of values in embedding will let the real feature get vanished. However It seems that is the best we can get, to reach ~0.510 w/o training and fine-tuning. We need to go further, and since that time we believe @khabel must train on something. 

## Let’s dive into training

Training is not that easy, not only because many contestants claimed they get much lower score by training, but also the strict rule of this competition forbidding people from using datasets with no commercial use guarantee. We tried our best, and went through https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/337384, but it is still rather hard to find datasets perfectly meet the requirements, and also be believed to help in this competition. Realizing this competition is an extend of the previous series of competitions Google Landmark. We decided to try to first train on GLDv2-clean, unsure if it legal though. Intuitively, the best way may be simply adding a projection layer to downsize the embedding to 64-d, and only fine-tuning the last layer. Since that we used GLDv2, it was reasonable to get to know how the winners of that competition did to reach the best performance. From [https://www.kaggle.com/competitions/landmark-retrieval-2020/discussion/176037] and [https://arxiv.org/abs/2009.05132], we supposed arcface may play a vital role here. So we just went simple first, fine-tuning the last layer using arcface with a margin of 0.5 and s of 30. It was surprising that we reached 0.560 after 6 epochs of fine-tuning, pointing out that we are in the right direction.

But we realized it is hard to move on, if we supposed GLD-v2 is on the boarder line, the rest of datasets is just obvious illegal from our view. We strongly hesitate that most teams, including us, may be disqualified because of datasets issue. We opened a thread here [https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/344643], and we glad to see @maggiemd and @andrefaraujo response in time to make any public datasets available as long as they are mentioned in the forum. 

Getting free from the previous rules, we can try a lot of stuffs to get improved. We always think we should follow this order: Chose datasets -> Decide model, training-related things. So we decided to test various datasets. To maintain a good performance and save time, we added datasets to out training list iteratively instead of training on each datasets from very beginning. We then added Products-10k, Shopee, MET Artwork Dataset, Alibaba goods, H&M Personalized Fashion, GPR1200, GLDv2-Full, DeepFashion - Consumer-to-shop Clothes Retrieval Benchmark part, boosting the score above 0.610. I felt like the score most likely converged since more epochs of fine-tuning did not help improve then. So it is time to do something about the model and training part. I felt like it is hard to do some stuffs about model structure, because a pre-trained weights is very vulnerable to any extra structure added, and even if we come up with a way of that, we don’t think it is a good trade-off regarding computing resources and time. 

So moving to the training tricks, we suppose leaving the backbone weights as it is, which was definitely not a good idea. Since now the linear head is well trained, we did not need to worry about the random linear head would affect the backbone weights. So we unfreeze the backbone part, using 10 times lower initial learning rate training the whole part. Just as @anonamename mentioned in [https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/359161], we found it is very likely to cause over-fitting, if we also trained on 6 or more epochs as we did on the linear head. Investigating on the phenomenons, we found that when we train on the whole models, the weights of linear projection has jumped sharply. It is weird since linear projection has already converged when we fine-tune on it individually, reminding us of thinking about what is actually the weights of linear projection is. 

Given a set of weights of a projection layer F(C, X), 
where C denotes the central embedding of each class from class pool X.

From above, the shaking of the F(C, X) reveals that the central embedding of each classes changed rapidly, which results in the Euclidean distance between classes has been largely changed. The Euclidean distance between each class is meaningful indeed, since it indicates the relationships between each classes. The rapid changing of central embedding may be a regardless-of-the-cost adaption to fit for the training loss. We supposed this was one of the root cause of the over-fitting. So we decided to freeze the final fc layer when train on backbone.

Adding dropout to the full connection layer is already a well-known trick to avoid over-fitting of training CNNs or vision transformers. It is not always work in most time, but here we supposed it is worthy to try, because 1) Our model have 2 fc layers, 1 to get embedding in the length of 512, provided originally by VIT, and we added another to downsize it to 64. More fc layers tend not to lose that much information because of dropout. 2) The over-fitting is rather serious here.

We combined the two methods above, with only 3 epochs, and get the model boost to 0.650-0.660.

In the previous training, we found that products-10k is most likely give the most improvement, so we extracted products-10k to fine-tune individually followed the order of “first fc, then backbone”. It reached 0.671. 

It seems that it is hard to continue improving, what should kagglers do now:) 

## Yes, ENSEMBLE!

But ensemble seems not working in this competition, as mentioned in [https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/337360] and [https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/339554]. We need to find the reason if we want to solve an issue. Why did ensemble not work here? I fully agree with @manwithaflower ‘s point of view indicated in [https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/340546#1876558]. Recalling the stuffs we found about the F(C, X), if we got similar F(C, X) in different models, then the ensemble of those models should work. 

How to achieve this? Basically we have two ways:

First, if we fine-tune models in different hyper-parameters, like model soups [https://arxiv.org/abs/2203.05482], but in the models ensemble style, it will likely work since the F(C, X) won’t change a lot. That is what we tried, and we got boost to 0.680 with ensemble of the resolution 224 and 280. 

Second, if we tried to keep F(C, X) the same with a maintaining in performance, we can also get models ensemble work, and even achieve better performance than the first way, because the variance of the models can be huge which help ensemble. The way to did it is based on an  
assumption that the different spaces represented by F(C, X) of different models can be aligned through a linear operation. Like below:

If we have,
 y^ = F1(C, X) * B1(W, x), and,
 y^ = F2(C, X) * B2(W, x),
Then ∃ G(C1, C2), such that,
 y^ = F1(C, X) * G(C1,C2) * B2(W, x)

So all we need is to train a G(*) on the models to do the ensemble. This is what we want to further investigate after this competition.

Things is really interesting till here, then a good catch by @dschettler8845 in [https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/353768] breaks that. LOL (but it is really an informative catch!)

I need to redo EVERY THING MENTIONED ABOVE to the new laion-2b VIT-H model thanks to this weight:(, except several changes: 1) drop model ensemble, VIT-H is really a huge guy 2) train on all the datasets at the same time, drop products-10k, leave products-10k as the final fine-tuning datasets.

Then it reaches a score of 0.703. After fine-tuning on 280, it reaches 0.705. Here are some tricks, training again through all the process from last layer to backbone in 280 with less epochs and lower learning rate, then it gives 0.723. Here is an alternative way not to turn higher the resolution, but gives some overlap to the patch. It gives the best performance in the last day to reach public LB with 0.732 and private LB 0.728 with 290 resolution and 4 pixels overlapping. 

Hope this will help someone to better perform in Kaggle Competitions. See you guys in the next comp, Happy Kaggling!

Github repository: https://github.com/LouieShao/1st-Place-Solution-in-Google-Universal-Image-Embedding
Paper: https://arxiv.org/abs/2210.08473
