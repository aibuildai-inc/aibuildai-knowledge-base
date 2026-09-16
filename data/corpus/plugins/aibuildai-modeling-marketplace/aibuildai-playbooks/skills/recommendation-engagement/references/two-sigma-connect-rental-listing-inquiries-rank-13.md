# Solution Share(#13)

Competition: two-sigma-connect-rental-listing-inquiries
Rank: #13
Source: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32156

It is great competition and let me first congrats to all winners, especially plantsgo as a first-time player in tier counted competition! Many thanks to KazAnova for sharing the magic feature that's the spirit which makes kaggle great!

Now let's get technical.

# basic features
    bedroom
    bathroom
    price
    longitude
    latitude
    manager_id
    building_id
    street_address
    display_address
    listing_id
    created_month
    created_day
    created_hour

# simple constructed features
    manager_id onehot
    building_id onehot
    street_address onehot
    display_address onehot
    price_t = price/bedrooms
    num_photos
    num_features
    num_description_words
    room_dif = bedroom-bathroom
    room_sum = bedroom+bathroom
    price_t1 = price/room_sum
    bedroom_count
    bathroom_count
    manager_id_count
    building_id_count
    street_address_count
    display_address_count
    bow of features limit 400/100

# constructed features
    manager_id grouping building_id tfidf to nmf
    manager_id grouping building_id tfidf deepwalkembedding
    manager_id grouping street_address tfidf to nmf
    manager_id grouping display_address tfidf to nmf
    building_id grouping manager_id tfidf to nmf
    building_id grouping street_address tfidf to nmf
    manager_id grouping price_t1 mean
    manager_id grouping room_sum mean
    manager_id grouping created difference mean
    manager_id grouping created count mean
    manager_id grouping created 24-hour mean(how often manager post during each hour)
    manager_id grouping building_id zero count
    manager_id grouping building_id zero ratio
    manager_id grouping latitude median
    manager_id grouping longitude median
    latitude,longitude grouping distance to clustering center
    manager_id latitude,longitude median grouping distance to clustering center
    manager_id grouping label encoding

# magic features

My final solution is an average of two different types of stacking procedure:
1.create 5-fold oof metafeature on two features set for each base model(xgb et rf nn lr lsvc gblinear)
2.for each xgb et rf, extracting 75% of original feature set and using subset to train metafeatures, each based model will generate 50 metafeatures and using all outputs as new features for the second layer.

The final submission is an weighted average of the two stacking result.
