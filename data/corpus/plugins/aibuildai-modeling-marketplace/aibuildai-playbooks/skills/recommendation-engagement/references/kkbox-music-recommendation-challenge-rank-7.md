# make use of representation learning methods

Competition: kkbox-music-recommendation-challenge
Rank: #7
Source: https://www.kaggle.com/c/kkbox-music-recommendation-challenge/discussion/45614

Greeting!
Just want to share the idea of my solution to you. Recently, I developed a network representation [toolkit][1] which is able to learn the representation of vertices in a given network. I think this is useful for estimating the similarities among different objects such as the songs, artists, etc. Concretely, this toolkit contains the so-called network/graph embedding and matrix factorization methods. You are allowed to use them to get the vector representations as long as giving a pre-constructed network. For instance, we can feed a user-song network/graph to the toolkit, and then to compute similarity between a user and a song afterward. Note that there are several ways to construct the network/graph and several ways to make use of the obtained vector representations. Here is the toolkit link: [https://github.com/cnclabs/proNet-core][1]

My current submission is done by a single gradient boosting machine model with 1) raw features, 2) statistics of features 3) user-item interaction features and 4) score features computed via the learned representations. Enjoy:) 

  [1]: https://github.com/cnclabs/proNet-core
