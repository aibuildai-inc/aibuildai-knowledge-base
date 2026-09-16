# 3rd place solution (with code)

Competition: santander-product-recommendation
Rank: #3
Source: https://www.kaggle.com/c/santander-product-recommendation/discussion/26899

Sorry for my late post, and congratulations to idle_speculation, Tom and everyone!

There were many twists and turns, but I would like to explain my final approach here.
I think that my approach is not so complicated, and it was performed by my 8GB RAM laptop (but a little more RAM is recommended).

The overview is illustrated in attached figure, and attached codes make 'prediction based on 2016-05-28 data' in the figure, which scores 0.0308403 in public LB and 0.0312074 in private LB (6th).

**Models**

 - The probability of new purchase of each product (denoted by Pr) was calculated by xgboost model (for 20 products except for ahor_fin, aval_fin, deco_fin and deme_fin).
 - Pr of cco_fin was predicted only by 2015-12-28 data.
 - Pr of reca_fin was predicted only by 2015-06-28 data.
 - Pr of other 18 products were predicted by 2016-05-28, 2016-04-28, 2016-03-28, 2016-02-28, 2016-01-28 and 2015-12-28 respectively.
 - Pr of other 18 products were also predicted by data from 2015-12-28 to 2016-04-28 using only records which have new purchased products.
 - Pr of ahor_fin, aval_fin, deco_fin and deme_fin were fixed to 1^-10.

**Features**

The number of used features was 142. Details are as given below.

 - original features except for 'fecha_dato', 'ncodpers', 'fecha_alta', 'ult_fec_cli_1t', 'tipodom' and 'cod_prov' (18 features)
 - concatenation of 'ind_actividad_cliente' and last month value of that (1 feature)
 - concatenation of 'tiprel_1mes' and last month value of that (1 feature)
 - last month values of 20 products (20 features)
 - concatenation of above 20 features as a character (1 feature)
 - the number of purchased products in the last month (1 feature)
 - count of index change pattern (0 to 0, 0 to 1, 1 to 0 and 1 to 1) until last month of 20 products (80 features)
 - length of continuous 0 index until last month of 20 products (20 features)

Character (factor) variables are replaced with target mean in each modeling process.

**Other remarks**

 - I basically used '2016-05-28' data as a validation dataset, but when I used '2016-05-28' data as a train dataset, I used '2016-04-28' data as a validation dataset.


Thank you for reading my clumsy post!

**Edit: Replaced the code (twice) which makes training data because there were severe mistakes (now v3). Sorry!**
