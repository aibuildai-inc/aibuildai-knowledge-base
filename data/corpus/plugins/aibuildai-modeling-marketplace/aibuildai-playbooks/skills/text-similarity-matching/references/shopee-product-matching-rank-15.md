# 15th place solution

Competition: shopee-product-matching
Rank: #15
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238029

## Summary
* Image : CNN+GeM+Arcface
* Title : TF-IDF and finetuned LaBSE
* Post processing: Cross-mean and DBA/QE
* Merge Image/Title embeddings as graph



## Image
* resnest101
* resnest200
* eca_nfnet_l1

Trained with Arcface and GeM.  Input size is 512x512.  
Usual augmentation(LR flip, cutout, bright, shiftscalerotate, etc) by albumentations.  
512 dimension for each model, concatenate them to produce 1536 dimensional vector.  

## Title
* Preprocessing  
 * unicode handling
 * tolower
 * Unit cleaning  
250 gram, 250gr, 250gram, 250g, means almost same,  
so I change them to "250gram" as preprocessing.  
I did same other dozens of units(meter, yard, ml, kg, pcs, lembar, ...)  

* TF-IDF  
Fit and transform to test dataset.  

* LaBSE  
Finetuned with Arcface and GeM.  
Augmentation by EDA(Easy Data Augmentation)-like method.  

Finally concat TF-IDF and BERT vector for title embedding.  

## Post Processing
* Cross-mean embedding  
(I don't know there is proper term for this)  
If there are products(e.g. A,B and C) which have identical image, they are same product. (precision > 99.9%)  
I replace each \*title\* embedding to mean of title embedding: (A+B+C)/3  
Same procedure for image embedding, mean by same title.

* DBA/QE
For image X, I got 3 nearest neighbor(e.g. X,Y,Z) by image embedding, then replace X's image embedding with weighted(logspace) sum of (X,Y,Z).  
Same for title.

## Graph merge
Now we have two embeddings: image and title.  
I Merge them as graph.  

For each image or title embedding, I got 50 nearest neighbors and euclidean distance.  
One node represents one embedding. Edge weight is their euclidean distance.  
Then We can merge image and title graph. If there is duplicated edge, pick small(near) one.  

I employed some kind of [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) to form clusters.  
I chose final threshold from LB score.  




## Things that didn't work
* Pseudo label: works pretty well in CV, but I can't do this within 2 hours.  
* uSIF, fasttext, glove
* PorterStemmer, LancasterStemmer
* Spell correction
* CosFace, AdaCos
* OCR: produced Title quality is not good
