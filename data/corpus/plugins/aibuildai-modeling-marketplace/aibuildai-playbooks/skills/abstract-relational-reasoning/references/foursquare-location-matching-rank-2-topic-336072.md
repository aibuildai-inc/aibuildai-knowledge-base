# 2nd place solution - T0m Part

Competition: foursquare-location-matching
Rank: #2
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/336072

This post is part of the 2nd place solution.

Brief Summary
- https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336062

Other part
- https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336090

***

# Candidate Generation
We picked up 128 candidates id for each of the following two ways.
- lat, lon base distance nearest ids
- name tfidf vector base cosine similarity nearest ids

and then, select a total of 128 from both with country-specific optimized ratios.
Max IoU 0.9895

***

# Transformer Candidate Blocking


#### Absolute Features
- Lat, Lon
- Country embedding
- Categories embedding
- xlm-romerta-base name embedding
- tfidf-svd vector (name, all)

#### Relative Features
- edit-distance (name, address, zipcode, phone, url, categories)
- tfidf-svd vector (name, all) cosine similarity

single LB: 0.918

#### Blocking
threshold=0.005, and union find post process
average candidates per id 4.1 

Max IoU 0.986
