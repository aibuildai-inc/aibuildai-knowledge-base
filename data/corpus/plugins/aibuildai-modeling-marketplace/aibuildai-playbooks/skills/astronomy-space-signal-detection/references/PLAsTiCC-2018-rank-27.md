# 27 Test Set Basic Stats in 8 minutes

Competition: PLAsTiCC-2018
Rank: #27
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/71827

It's time to put skills learnt from a certain botched competition to use! The following BigQuery script extracts the same features as the kernel https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135 minus the tsfresh features, but in just 8 minutes without having to manually split time series by objects.

```
WITH  
  aux AS(
  SELECT
    object_id,
    POW(flux/flux_err, 2) AS flux_ratio_sq,
    flux * POW(flux/flux_err, 2) AS flux_by_flux_ratio_sq
  FROM
    `project.astro.series` ),  
  simple AS(
  SELECT
    src.object_id,
    AVG(flux) AS flux_mean,
    MAX(flux) AS flux_max,
    MIN(flux) AS flux_min,
    APPROX_QUANTILES(flux, 2)[ORDINAL(1)] AS flux_median,
    STDDEV(flux) AS flux_std,
    AVG(flux_err) AS flux_err_mean,
    MAX(flux_err) AS flux_err_max,
    MIN(flux_err) AS flux_err_min,
    APPROX_QUANTILES(flux_err, 2)[ORDINAL(1)] AS flux_err_median,
    STDDEV(flux_err) AS flux_err_std,
    AVG(detected) AS detected_mean,
    AVG(flux_ratio_sq) AS flux_ratio_sq_mean,
    STDDEV(flux_ratio_sq) AS flux_ratio_sq_std,
    SUM(flux_ratio_sq) AS flux_ratio_sq_sum,
    AVG(flux_by_flux_ratio_sq) AS flux_by_flux_ratio_sq_mean,
    STDDEV(flux_by_flux_ratio_sq) AS flux_by_flux_ratio_sq_std,
    SUM(flux_by_flux_ratio_sq) AS flux_by_flux_ratio_sq_sum
  FROM
    `project.astro.series` src,
    aux
  WHERE
    src.object_id = aux.object_id
  GROUP BY
    object_id ),  
  skews AS (
  SELECT
    src.object_id,
    AVG(POW((flux - flux_mean) / flux_std, 3)) AS flux_skew,
    AVG(POW((flux_err - flux_mean) / flux_std, 3)) AS flux_err_skew,
    AVG(POW((flux_ratio_sq - flux_ratio_sq_mean) / flux_ratio_sq_std, 3)) AS flux_ratio_sq_skew,
    AVG(POW((flux_by_flux_ratio_sq - flux_by_flux_ratio_sq_mean) / flux_by_flux_ratio_sq_std, 3)) AS flux_by_flux_ratio_sq_skew
  FROM
    aux,
    simple,
    `project.astro.series` src
  WHERE
    aux.object_id = simple.object_id
    AND aux.object_id = src.object_id
  GROUP BY
    object_id),  
  det_mjd AS (
  SELECT
    object_id,
    MAX(mjd) - MIN(mjd) AS det_mjd_diff
  FROM
    `project.astro.series` src
  WHERE
    detected = 1
  GROUP BY
    object_id )  
SELECT
  simple.*,
  flux_max - flux_min AS flux_diff,
  (flux_max - flux_min) / flux_mean AS flux_diff2,
  flux_by_flux_ratio_sq_sum / flux_ratio_sq_sum AS flux_w_mean,
  (flux_max - flux_min) / (flux_by_flux_ratio_sq_sum / flux_ratio_sq_sum) AS flux_dif3,
  flux_skew,
  flux_err_skew,
  flux_ratio_sq_skew,
  flux_by_flux_ratio_sq_skew,
  det_mjd_diff  
FROM
  simple, skews, det_mjd  
WHERE
  simple.object_id = skews.object_id
  AND simple.object_id = det_mjd.object_id
```

```
Query complete (7 min 59.079 sec elapsed, 16.9 GB processed)
```
