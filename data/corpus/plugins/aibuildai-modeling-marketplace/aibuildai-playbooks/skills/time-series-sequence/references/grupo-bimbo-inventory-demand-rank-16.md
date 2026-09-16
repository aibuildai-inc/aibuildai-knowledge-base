# #19 place solution

Competition: grupo-bimbo-inventory-demand
Rank: #16
Source: https://www.kaggle.com/c/grupo-bimbo-inventory-demand/discussion/23208

I only trained on the rows from week 9. I used Apache Spark to create 12 features for those rows and XGBoost to train the model

 1. Avg demand for this client
 2. Avg demand for thhis product
 3. Weighted average demand for this product, weighted by total demand for each client
 4. Avg demand for this product in this city
 5. Avg demand for this product at this client
 6. Avg demand one weeks ago for this product at this client
 7. Avg demand two weeks ago for this product at this client
 8. Avg demand three weeks ago for this product at this client
 9. Avg demand four weeks ago for this product at this client
 10. Avg demand five weeks ago for this product at this client
 11. Predicted demand for this week by linear regression FTRL model using the following features:

	For every row in week 9, I calculated the historic average demand (before week 9) that that client
	had sold. Then I made tuples of the type (This_product_id,previously_sold_product_id) and hashed them. 
	The feature value was the average demand of the previously sold product. I also hashed the ID's in the row and gave them the feature value 1.0
	The output value was the demand for the given row.
 12. Predicted demand for this week by linear regression FTRL model using the following features:

	For all rows before week 9, I hashed all IDs and gave them the value 1.0.
	I also used poly4 interaction of the IDs and gave them the value 1.0
	The output value was the demand for the given row.
