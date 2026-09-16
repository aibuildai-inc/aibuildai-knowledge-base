# 2nd Place Solution (LSTM-Encoded SAKT-like TransformerEncoder)

Competition: riiid-test-answer-prediction
Rank: #2
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/210113

Thank you all teams who competed with me, all the people who participated in this competition, and the organizer that hosts such a great competition with the well-designed API!
Congrats @keetar, who defeats me and becomes the winner in this competition.

I'm happy because it's my first time I get solo prize!

It's my 4th kaggle competition and it was fun to compete with my past teammates ( @nyanpn, @pocketsuteado) and people who I competed with past competitions (e.g. @aerdem4, @its7171). 
Here, I will explain the summary of my model and my features. 

I uploaded 2 kaggle notebooks for the explanation. As the ensemble is not so important in my solution, I will only explain my single model. 
1. [6 similar models weighted average](https://www.kaggle.com/mamasinkgs/public-private-2nd-place-solution) : 0.817 public/0.818 private
2. [single model](https://www.kaggle.com/mamasinkgs/public-private-2nd-place-solution-single-fold) : 0.814 public/0.816 private

#Models 


##Overview 
my model is similar to SAKT, with 400 sequence length, 512 dimension, 4 nheads. 
I don't use lecture information for the input of transformer model, which I guess is why I lost in this competition. For the query and key/value of SAKT-like model, I used LSTM-encoded features, whose input is as follows.
- **"Query" features**
content_id
part
tags
normalized timedelta
normalized log timestamp 
correct answer
task_container_id delta
content_type_id delta
normalized absolute position 

- **"Memory" features**
explanation
correctness
normalized elapsed time
user_answer

## Detailed Explanation of Training/Inference Process
I tried a very precise indexing/masking technique to avoid data leakage in training process, I guess which is partly why I became 2nd place. OK, suppose a very simple model of the sequence length = 5, and a task_container_id history of a specific user (without lecture) is like this.
`[0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 8, 7, 7, 6]`
As there is 15 measurements in this user, I made 15/5 = 3 training samples and loss mask.

(1) input task_container_id: `[pad, pad, pad, pad, pad, 0, 0, 0, 1, 1]` 
(1) loss_mask: `[False, False, False, False, False, True, True, True, True, True]`
(2) input task_container_id: `[0, 0, 0, 1, 1, 2, 2, 3, 3, 4]` 
(2) loss_mask: `[False, False, False, False, False, True, True, True, True, True]`
(3) input task_container_id: `[2, 2, 3, 3, 4, 5, 8, 7, 7, 6]` 
(3) loss_mask: `[False, False, False, False, False, True, True, True, True, True]`

In this competition, the handling of the task_container_id is very important, as **it is not allowed to use "memory" features of the same task container id** to avoid leakage.
So, after applying LSTM to the features, such fancy indexing is required to avoid the leakage for 3 training samples, where -1 means this position can't attend any position.

(1) indices: `[-1, -1, -1, -1, -1, 4, 4, 4, 7, 7]`
(2) indices: `[-1, -1, -1, 2, 2, 4, 4, 6, 6, 8]`
(3) indices: `[-1, -1, 1, 1, 3, 4, 5, 6, 6, 8]`

To get this indices very fast in the training/inference phase, I wrote such cython (#1) function (vectorized implementation):

```
%%cython
import numpy as np
cimport numpy as np
cpdef np.ndarray[int] cget_memory_indices(np.ndarray task):
    cdef Py_ssize_t n = task.shape[1]
    cdef np.ndarray[int, ndim = 2] res = np.zeros_like(task, dtype = np.int32)
    cdef np.ndarray[int] tmp_counter = np.full(task.shape[0], -1, dtype = np.int32)
    cdef np.ndarray[int] u_counter = np.full(task.shape[0], task.shape[1] - 1, dtype = np.int32)
    for i in range(n):
        res[:, i] = u_counter
        tmp_counter += 1
        if i != n - 1:
            mask = (task[:, i] != task[:, i + 1])
            u_counter[mask] = tmp_counter[mask]
    return res
```
After applying such fancy indexing to the output of LSTM features, I concatenated them with "Query" features and apply MLP, then we can get query for SAKT model. Then, I obtained key/value for SAKT model by applying MLP to query concatenated with "Memory" features. 

To train SAKT-like model, precise memory masking is also required to avoid leakage. I used [3D attention mask](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) of torch.nn.MultiheadAttention. It should be noted [torch.repeat_interleave](https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html) must be leveraged to make 3D mask (batchsize * nhead, sequence length, sequence length). 
The memory mask for these 3 samples are like this. Here, 1 means True and 0 means False. Please remember the [documentation](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) says 
``` 
attn_mask ensure that position i is allowed to attend the unmasked positions. If a BoolTensor is provided, positions with True is not allowed to attend while False values will be unchanged.
```
(1) memory_mask:
```
array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
       [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]])
```
(2) memory_mask:
```
array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
       [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
       [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
       [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]])
```
(3)  memory_mask:
```
array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
       [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
       [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
       [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
       [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
       [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]])
```

To make this memory mask very fast in the training phase, I wrote such cython (#2) function (not vectorized implementation):
```
%%cython
import numpy as np
cimport numpy as np
cpdef np.ndarray[int] cget_memory_mask(np.ndarray task, int n_length):
    cdef Py_ssize_t n = task.shape[0]
    cdef np.ndarray[int, ndim = 2] res = np.full((n, n), 1, dtype = np.int32)
    cdef int tmp_counter = 0
    cdef int u_counter = 0
    for i in range(n):
        tmp_counter += 1
        if i == n - 1 or task[i] != task[i + 1]:
            res[i - tmp_counter + 1 : i + 1, :u_counter] = 0
            if u_counter == 0:
                res[i - tmp_counter + 1 : i + 1, n - 1] = 0
            if u_counter > n_length:
                res[i - tmp_counter + 1: i + 1, :(u_counter - n_length)] = 1
            u_counter += tmp_counter
            tmp_counter = 0
    return res
```

In the inference phase, memory mask is not required. For more details, please look at my [single model notebook](https://www.kaggle.com/mamasinkgs/public-private-2nd-place-solution-single-fold).

#My Features
Transformer is great, but it suffers from a problem that the sequence length cannot be infinite. I mean, it cannot consider the information of very old samples. To tackle this problem and to leverage the lecture information, I made simple features and concatenated them with the output features of SAKT-like model, and applied MLP and sigmoid.
I uploaded the names of 90 features as attachments. For the implementation of these features, I didn't use any pd.merge or df.join and most of the implementation are done using numpy. To make user-content features, I used scipy.sparse.lil_matrix to spare memory usage. I think my feature is not so good as other competitors (about 0.795 when using GBDT), but still improved score 0.001 ~ 0.002.
I made some tricky features (e.g. obtained by SVD), but it did not improve the score of NN model (improved GBDT model, though.), probably because the information of NN features includes that of such tricky features.

#CV Strategy
My CV strategy is completely different from tito( @its7171 )'s one. First, I probed the number of new users in test set and I found there are about 7000 new users. as we know there is 2.5M rows in test set and we know the average length of the history of all users, we can estimate the number of rows of the new users and the number of rows of the existing users. After the calculation, I found
```
the number of rows of new users (i.e. user split): the number of rows of existing users (i.e. timeseries split) = 2 : 1.
```
So, For validation, I decided to use 1M rows for timeseries split and 2M rows for user split.
When making validation set of timeseries split, I was so careful that leakage can't happen. I mean, my training dataset and validation dataset never shares same task_container_id for a given user.

#What worked 
1. increase length from 100 to 400 worked.
2. using normalized timedelta is better than digitized timedelta (mentioned in SAINT+ paper).
3. concatenating embeddings is better than adding embedding, as mentioned in https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/201798. 
4. dropout = 0.2 is very important.
5. StepLR with Adam is good. I trained my model for about 35 epoch with lr = 2e-3, then trained it for 1 epoch with lr = 2e-4.

#What didn't work
1. Random masking of sequences didn't improve the score.
2. bundle_id and normalized task_container_id is not needed for the input of LSTM.
3. As I didn't make diverse models, weighted averaging is enough and blending using GBDT didn't work well.
4. Full data training (without any validation data) didn't improve the score.
5. In my implementation, SAINT didn't work well. In my opinion, it's logically very hard to implement SAINT without data leakage in training due to task_container_id.

#Comments
I was very sad to see the private score bug problem of kaggle. Of course this competition is very great, but I think this competition could have been one of the most successful competition in kaggle without this problem.
Anyway, I really enjoyed my 4th kaggle competition. I'll continue kaggle and will surely become the winner in the next competition!

<h1>Thanks all, see you again!</h1>


P.S. The word `memory_mask` may be confusing because it is different from pytorch [Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer)'s `memory_mask`.
The word `memory_mask` is more like `mask` in pytorch's [TransformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html).
The reason my word is confusing is, I used Encoder-Decoder model at first, then gave up it and started using Encoder-only model, but I didn't change the function names. :(
