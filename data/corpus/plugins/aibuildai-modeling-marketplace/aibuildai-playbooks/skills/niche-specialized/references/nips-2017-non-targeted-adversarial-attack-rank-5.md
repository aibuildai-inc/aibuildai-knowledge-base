# 5th place solution

Competition: nips-2017-non-targeted-adversarial-attack
Rank: #5
Source: https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack/discussion/43413

Hi Everyone,

It is seemed that final results are uploaded to leaderboad.<br>
I make a conceptual diagram for my solution. My solution is based on basic iterative method. The key point is applying blur effects to gradient by convolving gaussian filter. This works as regularization and enable to attack many types of model simultaneously (This ability is called *transferability* in adversarial-example research).

Code: https://github.com/toshi-k/kaggle-nips-2017-adversarial-attack
![enter image description here][1]

  [1]: https://raw.githubusercontent.com/toshi-k/kaggle-nips-2017-adversarial-attack/master/img/solution.png
