# 14 place write-up

Competition: santander-value-prediction-challenge
Rank: #14
Source: https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/63799

After Giba published the leak, we observed that these rows also form a sequence in other columns, prompting us to search for other column groups of 40. This allowed us to identify more accurate leaks. Through an iterative and partially manual process, we discovered 7,866 leaks in the test set and 3,882 in the train set, achieving a perfect accuracy of 1.0.

1. The first and most important thing that I want to mention - the construction of a good CV was the key moment. The CV should be strictly correlated with LB.
2.  Adversarial validation was everywhere.
3. After detecting all leaks, we created a new train/test split and rebuilt all our and public models.
4. And finally - more blends for the god of blends ;)
![enter image description here][1]
Special thanks to everyone who shared their insights about leaks and models.


  [1]: https://storage.googleapis.com/kaggle-forum-message-attachments/373437/10136/santander_2018.jpg
