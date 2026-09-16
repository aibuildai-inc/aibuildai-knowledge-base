# 10th place write up (simple)

Competition: santander-value-prediction-challenge
Rank: #10
Source: https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/63812

Congrats to everyone who tried really hard to find the leaks &amp; build ML models! I myself also had a hard time trying to push my place up during the last few weeks, but the outcome showed that it was definitely worth it. Here I want to briefly describe what I have done, mainly because I saw a lot of people asking how simple the solution can be. Here's what I did for the final submission:

## Find The Leaks ##

I tried several methods to find the leaks. By using the rows found in the first few kernels I was able to find roughly 80+ sets. After some futile works, I did what @Paradox said in the forum to find a total of 100 good (40) column sets. I also used the rest of the column sets but they don't seem to help too much. This gives 7841 leaky rows in the test set.

    def get_order(data, feats, extra_feats, offset = 2):
        f1 = feats[:(offset * -1)]
        f2 = feats[offset:]
        for ef in extra_feats:
            f1 += ef[:(offset * -1)]
            f2 += ef[offset:]
        d1 = data[f1].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
        d2 = data[f2].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
        d2['pred'] = data[feats[offset-2]]
        d3 = d2[~d2.duplicated(['key'], keep=False)]
        d3['i3'] = d3.index
        d4 = d1[~d1.duplicated(['key'], keep=False)]
        d4['i4'] = d4.index
        d5 = d4.merge(d3, how='inner', on='key')
        return d5
    
    def get_order_T(data, feats, offset = 2):
        f1 = []
        f2 = []
        for ef in feats:
            f1 += ef[:(offset * -1)]
            f2 += ef[offset:]
        d1 = data[f1].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
        d2 = data[f2].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
        d3 = d2[~d2.duplicated(['key'], keep=False)]
        d3['i3'] = d3.index
        d4 = d1[~d1.duplicated(['key'], keep=False)]
        d4['i4'] = d4.index
        d5 = d4.merge(d3, how='inner', on='key')
        return d5
    
    def get_sets(cor):
        c1 = cor.iloc[:, 0]
        c2 = cor.iloc[:, 1]
        d = dict((x1, x2) for x1, x2 in zip(c1, c2))
        sets = []
        for k in d.keys():
            set_k = []
            k_left = k
            while d[k_left] in d.keys():
                set_k.append(k_left)
                k_left = d[k_left]
            set_k.append(k_left)
            set_k.append(d[k_left])
            EXIST = False
            for i in range(len(sets)):
                for item in set_k:
                    if item in sets[i]:
                        EXIST = True
                        break
                if EXIST:
                    break
            if EXIST:
                if len(set_k) &gt; len(sets[i]):
                    sets[i] = set_k
                continue
            sets.append(set_k)
        return sets
    
    def merge_two_lists(list1, list2):
    '''
    useful if want to merge two lists
    '''
        if list1[0] in list2 and list1[-1] in list2:
            return list2
        elif list2[0] in list1 and list2[-1] in list1:
            return list1
        elif list1[0] in list2 and list1[-1] not in list2:
            start = 0
            for i in range(len(list2)):
                if list2[i] == list1[0]:
                    break
                else:
                    start += 1
            return list2 + list1[len(list2) - start:]
        elif list2[0] in list1 and list2[-1] not in list1:
            start = 0
            for i in range(len(list1)):
                if list1[i] == list2[0]:
                    break
                else:
                    start += 1
            return list1 + list2[len(list1) - start:]

    def merge_two_sets(sets1, sets2):
    '''
    useful if want to merge two sets (e.g., from train and test)
    '''
        heads = []
        tails = []
        joints = []
        for s in sets1:
            heads.append(s[0])
            tails.append(s[-1])
        for h in tqdm(range(len(heads))):
            for t in range(len(tails)):
                head = heads[h]
                tail = tails[t]
                for joint in sets2:
                    for val1, val2 in zip(joint[:-1], joint[1:]):
                        if tail == val1 and head == val2:
                            joints.append((t, h))
        return joints
    
    d5 = get_order(train, cols, cols_extra, 1)
    cor = d5[['i4','i3']]
    sets = get_sets(cor) # the sets of rows
    d5_T = get_order_T(train.T, sets, 1)
    cor_T = d5_T[['i4','i3']]
    sets_T = get_sets(cor_T) # the sets of columns

##The ML Model##

I simply used the 1.37 LB kernel by @Emmanuel Perry. Of course, I tried a lot of other things but I decided not to use them.

So, here we are, a simple solution. It is actually a little weird that this can end up in top 10. If anyone is interested I can post my code used to find the leaky rows and columns :)
