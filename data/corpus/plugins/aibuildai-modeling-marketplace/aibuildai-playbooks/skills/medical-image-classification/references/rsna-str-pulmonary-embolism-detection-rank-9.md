# 9th place solution ( + github code)

Competition: rsna-str-pulmonary-embolism-detection
Rank: #9
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193417

Congratulations to all winners !
This competition was hard on me in many ways.

# Solution Overview

.png?generation=1603760751579416&alt=media)

# Preprocess

- In the train data, no CT image have PE after 400th image. So, we used only images before 400th image.


- For stage 1 training, we preprocessed image-level labels like following image.
  - Before

  - After


# Stage 1 training

- We used 512 x 512 image + efficientnet-b5 and 384 x 384 image + efficientnet-b3 and using preprocessed labels.

# Stage 2 training

- Inference time was so severe because we used 512 x 512 image + efficientnet-b5. So, we subsampled 400 sequences to 200 sequences and used Deconvolution module.
  - We got the same CV score when using 400 sequences.
- We was not able to use various models in stage 1 because of resource. Therefore, we trained various models in stage 2.
  - Input: b5-feature only, b3-feature only, b5-feature + b3-feature
  - model: Conv1D, LSTM, GRU, Conv1D + LSTM
  - output: 3 x 4 = 12 predictions

# Stacking

- We trained LGBM, Conv1D and GRU.
  - We used only PE-exam when training pe_present_on_image by lgbm because image from negative PE doesn't affect competition metric.

# Postprocess

- We've implemented a heuristic post process.
  - This postprocess increased CV and public score 0.002 (Private 0.160 -> 0.162)
  - Main idea
      - replace  `pe_present_on_image` with `1 - negative_exam_for_pe` when `1 - negative_exam_for_pe <= pe_present_on_image`
      - repeat sigmoid -> logit -> logit += s -> sigmoid until satisfying label consistency
```python
label_cols = [
      "pe_present_on_image",
      "negative_exam_for_pe",
      "indeterminate",
      "chronic_pe",
      "acute_and_chronic_pe",
      "central_pe",
      "leftsided_pe",
      "rightsided_pe",
      "rv_lv_ratio_gte_1",
      "rv_lv_ratio_lt_1",
    ]

def postprocess(x, s=2.0):
    logit = np.log(x/(1 - x))
    logit = logit + s
    sigmoid = 1 / (1 + np.exp(-logit))
    return sigmoid

def satisfy_label_consistency(df):
    rule_breaks = consistency_check(df).index
    print(rule_breaks)
    if len(rule_breaks) > 0:
        df["positive_exam_for_pe"] = 1 - df["negative_exam_for_pe"]
        df.loc[
            df.query("positive_exam_for_pe <= pe_present_on_image").index,
            "pe_present_on_image",
        ] = df.loc[
            df.query("positive_exam_for_pe <= pe_present_on_image").index,
            "positive_exam_for_pe",
        ]
        rule_breaks = consistency_check(df).index
        df["positive_images_in_exam"] = df["StudyInstanceUID"].map(
            df.groupby(["StudyInstanceUID"])["pe_present_on_image"].max()
        )
        df_pos = df.query("positive_images_in_exam > 0.5")
        df_neg = df.query("positive_images_in_exam <= 0.5")
        if "1a" in rule_breaks:
            rv_filter = "rv_lv_ratio_gte_1 > 0.5 & rv_lv_ratio_lt_1 > 0.5"
            while len(df_pos.query(rv_filter)) > 0:
                df_pos.loc[df_pos.query(rv_filter).index, "rv_min"] = df_pos.query(
                    rv_filter
                )[label_cols[8:]].min(1)
                for rv_col in label_cols[8:]:
                    df_pos.loc[
                        df_pos.query(rv_filter + f" & {rv_col} == rv_min").index, rv_col
                    ] = postprocess(
                        df_pos.query(rv_filter + f" & {rv_col} == rv_min")[
                            rv_col
                        ].values,
                        s=-0.1,
                    )
            rv_filter = "rv_lv_ratio_gte_1 <= 0.5 & rv_lv_ratio_lt_1 <= 0.5"
            while len(df_pos.query(rv_filter)) > 0:
                df_pos.loc[df_pos.query(rv_filter).index, "rv_max"] = df_pos.query(
                    rv_filter
                )[label_cols[8:]].max(1)
                for rv_col in label_cols[8:]:
                    df_pos.loc[
                        df_pos.query(rv_filter + f" & {rv_col} == rv_max").index, rv_col
                    ] = postprocess(
                        df_pos.query(rv_filter + f" & {rv_col} == rv_max")[
                            rv_col
                        ].values,
                        s=0.1,
                    )
            df.loc[df_pos.index, label_cols[8:]] = df_pos[label_cols[8:]]
        if "1b" in rule_breaks:
            pe_filter = " & ".join([f"{col} <= 0.5" for col in label_cols[5:8]])
            while "1b" in consistency_check(df).index:
                for col in label_cols[5:8]:
                    df_pos.loc[df_pos.query(pe_filter).index, col] = postprocess(
                        df_pos.loc[df_pos.query(pe_filter).index, col], s=0.1
                    )
                df.loc[df_pos.index, label_cols[5:8]] = df_pos[label_cols[5:8]].values
        if "1c" in rule_breaks:
            chronic_filter = "chronic_pe > 0.5 & acute_and_chronic_pe > 0.5"
            df_pos.loc[df_pos.query(chronic_filter).index, label_cols[3:5]] = softmax(
                df_pos.query(chronic_filter)[label_cols[3:5]].values, axis=1
            )
            df.loc[df_pos.index, label_cols[3:5]] = df_pos[label_cols[3:5]]
        if "1d" in rule_breaks:
            neg_filter = "negative_exam_for_pe > 0.5 | indeterminate > 0.5"
            while "1d" in consistency_check(df).index:
                for col in label_cols[1:3]:
                    df_pos.loc[df_pos.query(neg_filter).index, col] = postprocess(
                        df_pos.loc[df_pos.query(neg_filter).index, col], s=-0.1
                    )
                df.loc[df_pos.index, label_cols[1:3]] = df_pos[label_cols[1:3]].values
        if "2a" in rule_breaks:
            neg_filter = "negative_exam_for_pe > 0.5 & indeterminate > 0.5"
            while len(df_neg.query(neg_filter)) > 0:
                df_neg.loc[df_neg.query(neg_filter).index, "neg_min"] = df_neg.query(
                    neg_filter
                )[label_cols[1:3]].min(1)
                for neg_col in label_cols[1:3]:
                    df_neg.loc[
                        df_neg.query(neg_filter + f" & {neg_col} == neg_min").index,
                        neg_col,
                    ] = postprocess(
                        df_neg.query(neg_filter + f" & {neg_col} == neg_min")[
                            neg_col
                        ].values,
                        s=-0.1,
                    )
            neg_filter = "negative_exam_for_pe <= 0.5 & indeterminate <= 0.5"
            while len(df_neg.query(neg_filter)) > 0:
                df_neg.loc[df_neg.query(neg_filter).index, "neg_max"] = df_neg.query(
                    neg_filter
                )[label_cols[1:3]].max(1)
                for neg_col in label_cols[1:3]:
                    df_neg.loc[
                        df_neg.query(neg_filter + f" & {neg_col} == neg_max").index,
                        neg_col,
                    ] = postprocess(
                        df_neg.query(neg_filter + f" & {neg_col} == neg_max")[
                            neg_col
                        ].values,
                        s=0.1,
                    )
            df.loc[df_neg.index, label_cols[1:3]] = df_neg[label_cols[1:3]]
        if "2b" in rule_breaks:
            while "2b" in consistency_check(df).index:
                for col in label_cols[3:]:
                    df_neg.loc[df_neg.query(f"{col} > 0.5").index, col] = postprocess(
                        df_neg.loc[df_neg.query(f"{col} > 0.5").index, col], s=-0.1
                    )
                df.loc[df_neg.index, label_cols[3:]] = df_neg[label_cols[3:]].values
    return df

```

Updated: 
We uploaded code on github (https://github.com/shimacos37/kaggle-rsna-2020-9th-solution).
