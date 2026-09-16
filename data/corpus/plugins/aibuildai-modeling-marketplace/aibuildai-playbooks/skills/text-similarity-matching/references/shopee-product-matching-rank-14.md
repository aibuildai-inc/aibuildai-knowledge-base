# 14th Place Gold - Image Text Decision Boundary

Competition: shopee-product-matching
Rank: #14
Source: https://www.kaggle.com/c/shopee-product-matching/discussion/238033

Thank you Kaggle and Shopee for hosting a wonderful competition. This was one of my favorite competitions. This competition challenged us to build image models, text models, and combine everything in ways never done before.

# RAPIDS TfidfVectorizer
My solution began with RAPIDS TfidfVectorizer and EfficientNetB0 384x384 images as presented in my public notebook [here][1]. Next I ensembled with XLM-RoBERTa which has multilingual pretraining and I ensembled with EfficientNetB3 512x512 image.

# Best Public Notebook LB 0.730
My [public notebook][1] scored LB 0.700. Next @ragnar123 increased it to LB 0.730 adding ArcFace and cosine distance thresholds [here][2]. Public notebooks make decisions as follows. If two products have image cosine distance less than 0.3 or text cosine distance less than 0.17, they are a match:

[image]

# Better Decision Boundary LB  0.750

Note how the plot above shows a simple decision boundary (of constant functions). By using our CV, we can calculate the probably that two products are a match based on their image cosine distance and text cosine distance and make a decision boundary using piecewise linear functions:

[image]

[image]

## Add Matches With Middle Image and Text Similarity

From the plot above, we see that we should add more matches than the public notebooks. For example if two products have `cosine image similarity = 0.4` and `cosine text similarity = 0.4` then we should consider them a match. These two products were not considered a match with the simple rule "image < 0.3 or text < 0.17".

## Sample Code
In order to find these matches, we must compute image and text cosine distances simultaneously.

    # NORMALIZE IMAGE EMBEDDINGS
    s = np.sqrt(np.sum(np.multiply(image_embeddings,image_embeddings),axis=1))
    image_embeddings = image_embeddings / np.expand_dims(s,axis=-1)

    # NORMALIZE TEXT EMBEDDINGS
    s = np.sqrt(np.sum(np.multiply(text_embeddings,text_embeddings),axis=1))
    text_embeddings = text_embeddings / np.expand_dims(s,axis=-1)
    
    pids = []
    all_id = df.posting_id.values

    # COMPUTE DISTANCES IN BATCHES
    CT = int(np.ceil(len(df)/BATCH))
    for k in range(CT):
        
        a = k*BATCH
        b = (k+1)*BATCH
        b = min(len(df),b)

        for j in range(b-a):

            # COMPUTE CHUNK OF DISTANCES
            img = 1-image_embeddings.dot(image_embeddings[a:b,].T).T 
            txt = 1-text_embeddings.dot(text_embeddings[a:b,].T).T 
        
            # DECISION BOUNDARY
            idx1 = np.where(img[j,] <0.3)[0]
            idx2 = np.where(txt[j,]<0.17)[0]
            idx3 = np.where( 0.62*img[j,] +txt[j,] <0.73)[0]
            idx = np.concatenate([idx1,idx2,idx3])
            idx = np.unique(idx)
            pids.append( all_id[idx] )

    # PREDICTIONS
    df['matches'] = pids

# Remove False Negative and False Postives LB 0.770
The above decision boundary increases our CV LB to 0.750. To increase our LB to 0.770, we must do five additional techniques to remove false negatives and false positives (which increase metric F1 score). I will explain these further in part 2 of my solution.
* If a product has 0 matches, slowly increase boundary to find 1 match. (Remove false negative).
* If a product has matches with a product of same image and/or same text, analyze all of those products' matches and then remove matches from current product if they don't appear often in the neighbors' neighbors. (Remove false positive via Graph analysis).
* If a product has a match and that match is part of a group with same image and/or same text, then decide to include all of that cluster or remove all of that cluster. (Remove false positive and false negative via Graph analysis).
* Build a stage 2 model that determines if we should increase threshold or decrease threshold of current boundary based on what product type we are looking at. (Stacking).
* If a match has a different unit (50 ml) than the current product (100 ml), remove that match. (Remove false positive).


[1]: https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700
[2]: https://www.kaggle.com/ragnar123/shopee-inference-efficientnetb1-tfidfvectorizer
