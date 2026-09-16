# 10th Place Solution

Competition: hpa-single-cell-image-classification
Rank: #10
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238364

## Pipeline
[pipeline]

## Filtering
```
    # relabel and remove unmatched cell
    ## single
    gr = train_single.groupby("image_id")
    cell_size = gr.cell_id.transform(max)
    for c1, c2, c3 in zip(COLS_TARGET, COLS_RELABEL, COLS_PRED):
        # 1
        re1 = (
                train_single[c1] * ((gr[c3].rank() / cell_size) > 0.5)
                |
                train_single[c1] * (train_single[c3] >= 0.5)
        )
        re1.loc[re1==0] = np.nan
        # 0
        re0 = (
                train_single[c1] * ((gr[c3].rank() / cell_size) < 0.1)
                &
                train_single[c1] * (train_single[c3] < 0.1)
        )
        re0.loc[re0==0] = np.nan
        re0.loc[re0==1] = 0
        train_single[c2] = pd.concat([re0, re1], axis=1).max(1)
        train_single.loc[train_single[c1]==0, c2] = 0
    train_single = train_single.dropna().reset_index(drop=True)

    ## multi
    import cudf
    cdf = cudf.DataFrame(train_multi[['image_id'] + COLS_PRED])
    pred_quantile1 = cdf.groupby("image_id").agg(lambda x: x.quantile(0.50)).to_pandas()
    pred_quantile1 = train_multi[['image_id']].merge(pred_quantile1, on='image_id', how='left')
    pred_quantile0 = cdf.groupby("image_id").agg(lambda x: x.quantile(0.50)).to_pandas()
    pred_quantile0 = train_multi[['image_id']].merge(pred_quantile0, on='image_id', how='left')
    for c1, c2, c3 in zip(COLS_TARGET, COLS_RELABEL, COLS_PRED):
        # 1
        re1 = (
                train_multi[c1] * (train_multi[c3] >= pred_quantile1[c3])
                |
                train_multi[c1] * (train_multi[c3] >= 0.5)
        )
        re1.loc[re1==0] = np.nan
        # 0
        re0 = (
                train_multi[c1] * (train_multi[c3] < pred_quantile0[c3])
                &
                train_multi[c1] * (train_multi[c3] < 0.1)
        )
        re0.loc[re0==0] = np.nan
        re0.loc[re0==1] = 0
        train_multi[c2] = pd.concat([re0, re1], axis=1).max(1)
        train_multi.loc[train_multi[c1]==0, c2] = 0
    train_multi = train_multi.dropna().reset_index(drop=True)

```

## Final Submission
https://www.kaggle.com/inoueu1/hpa2021-p-all-1stx8-2ndx16-imgx16/notebook?scriptVersionId=62576779
