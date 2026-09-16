# 6th place solution ,  yolov5m6/s6,  trust cv, post classification.

Competition: tensorflow-great-barrier-reef
Rank: #6
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307619

Congratulations to all team which trust your CV ,  the Private board shake huge. Thanks to my teammates @khyeh0719 @trushk  I have a greate journey in this competition.

## Pipeline

[pipeline]

## Trust your CV

We split data by video_id, and calculate the f2-score on the whole oof data, and fund something interesting.

1. CV shows model is work great when infer img resolution = train img resolution,  but LB need X2~X2.5 infer resolution to get nice score.
2. Track improve the score about 0.002 on both CV and LB .
3. Post classification improve the score about 0.006~0.01 on both CV and LB.
4. Infer with low conf and use Wbf ensemble give use about 0.02 boost.

finally we choose our 4 sub on this strategy:
1. best cv sub w/ normal resolution     ->LB 0.616 / Private 0.723
2. best lb sub                                        ->LB 0.716 / Private  0.692 (big shake)
3. best cv X1.5 resolution.                    ->LB 0.668/ Private 0.727 
4. best cv X2.0 resolution                    ->Lb 0.700/ Private 0.711

## Best CV Model

1. post cls use swin-transformer
2. track use nofair public setting
3. wbf ensemble
4. single model as follow show:

| name | train config | infer config |
| --- | --- |--- |
| s1 |  yolov5m6_e5_bs1_lr01_img3000_public | sz3600_nmsconf0.01_nmsiou0.2|
| s3 |  yolov5m6_e5_bs1_lr01_img3000_public | sz2412_nmsconf0.01_nmsiou0.2|
|s15| yolov5s6_e5_bs2_lr01_img3000_public | sz3000_nmsconf0.01_nmsiou0.2|
|s61|yolov5m6_e10_bs8_lr01_img3600_cp_nb8|sz3600_conf0.01_iou0.2|
|s62|yolov5s6_e10_bs8_lr01_img3000_cp_nb8|sz3000_conf0.01_iou0.2|
|s63|yolov5m6_e10_bs8_lr01_img3600_cp_nb8|sz2412_conf0.01_iou0.2|
