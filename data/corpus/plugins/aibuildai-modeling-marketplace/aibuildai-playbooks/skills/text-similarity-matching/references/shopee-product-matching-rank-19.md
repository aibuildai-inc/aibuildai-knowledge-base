# 19 place. Voting and similarity chain

Competition: shopee-product-matching
Rank: #19
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238146

# Models
img: eca_nfnet_l0 + eff_b5
txt: tfidf + bert(phrase multilang)

# Tricks and ideas:  

1. Voting --  Image_score and text_score averaging **+0.11** (.739->.750) i
For each product we create null array (dictionary), and add to cells results NN (nearest neighbor) search 1-cosine_nn_dist. for tfidf cosine_dist. Normalize by the number of models for text and for images
2. Always search for pair **+0.05** (.734->.739)
3. Own models for images **+0.02**. Other loss (r focal loss). More augmentations
4. Text cleaning **+0.01**
5. Subtraction -1 in voting if postfixes do not match. gr, ml, pa. **+0.02** 
5.1. spf type prefixes
5.2. Synonyms of postfixes (picture 1). And creating glued pairs of number-postfixes.
5.3. Checking if postfixes have splitting power
The postfixes and prefixes were collected from training sample, all text sequences which are written together with numbers 30meter spf50

6. Similarity chain (picture 2) **+0.02**
If a product is selected with high accuracy then we are looking for a product similar to the query product and a new product  


# Not worked
1. Learning TFIDF coefficients with cosine loss
I wanted to use parametrization training on a test sample, but didn't have time to
2. Segmentation/detection. Segmented the image with silence detection/backround removal/ matting.
2.1 I multiplied the obtained mask by the last feature map.
Found a rectangular area, global pooling applied to it and got a lot of scene fragments, that I merged with the tagged image into NearestNeigbor
3. Extra Attention after the last layer. ECA, self-supervised

## Visualisation
first -- query, green -- matched, red -- miss matched, blue -- not found/matched

pictire 1. Check postfix like gr, ml, pa


picture 2. Similarity chain. Two last images not found, but images is similar to matched product
