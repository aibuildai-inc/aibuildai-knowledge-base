# 26th place solution

Competition: landmark-retrieval-2019
Rank: #26
Source: https://www.kaggle.com/c/landmark-retrieval-2019/discussion/97040#latest-560400

Hi Everyone,

Ending final validation, I share my solution.<br>
I trained two models. One uses raw images as input. The other uses DELF as input. I concatenated two vectors from two models for indexing. I also used query expansion and geometric verification.

Code: https://github.com/toshi-k/kaggle-google-landmark-retrieval-2019

![conceptual diagram][1]

[1]: https://raw.githubusercontent.com/toshi-k/kaggle-google-landmark-retrieval-2019/master/img/solution.png
