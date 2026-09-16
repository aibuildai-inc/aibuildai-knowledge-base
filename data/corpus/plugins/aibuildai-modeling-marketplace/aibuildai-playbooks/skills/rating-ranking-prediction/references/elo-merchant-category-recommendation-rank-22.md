# 22nd (some 10k feats)shareing some engineering and some trick(not used)

Competition: elo-merchant-category-recommendation
Rank: #22
Source: https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82057

Fisrt of all congratulations to the winner, sec, thanks for kagglers sharing in the compete,I have learned a lot

law: trust your local cv

here something I want to share
trick!!!
final_pred = binary_prediprob * -33 + pred_without_outlier * ( 1- binary_prediprob), as I see in previos post the 1st used, We use this trick too,but did not select as our final sub
eng!!!

 1.  choose a good baseline
      I take this kernel as my baseline (here --&gt; https://www.kaggle.com/roydatascience/elo-merchant-recommendation-fathers-day-specials),tks to Ashish Gupta
 2. repreoduce feature with baseline feats( except holidays )
      select subsets to reprodeuce feats mainly through → lag &gt; threadhold (-3 ,-6) as subset condiction, reproduce feats we did once such as month_diff ---&gt; lag_3_month_diff,lag_6_month_diff, this give me at least 10thousand improvements
 3. abount merchants.csv,
       [user preference], find user preference merchant(id) , and use it, use its features which is calcaulate from merchants.csv
 4. ratios
       basically new/his, feature calcu by new_merchant div by same feature calcu in his
  
 5. word2vec
       through what-and-when-and-where format to build senctence
 6. timestamp
     use feats from use timestamp

apart from modeling, I think most improtant of all, do more eda, undersand well feats , and What is the company's business? 
10k feats
 1.
new_mount_sum/his
new_date_max_tonow/his
new_date_max_tonow/his
 2.
pref_merchant_purchase_amount_percard

pref_merchants_avg_sales_lag3_ratio

new_pref_merchants_avg_sales_lag3_ratio

pref_merchants_avg_sales_lag3_div_6

new_pref_merchants_avg_sales_lag3_div_6

new_pref_merchants_avg_sales_lag3

pref_merchants_avg_sales_lag3

new_pref_merchants_numerical_1

new_pref_merchants_numerical_1_ratio

pref_merchants_numerical_1

pref_merchants_numerical_1_ratio

pref_merchants_most_recent_sales_range_ratio

new_pref_merchants_most_recent_sales_range

new_pref_merchants_most_recent_sales_range_ratio

pref_merchants_most_recent_sales_range

new_pref_merchants_category_4

pref_merchants_category_4
