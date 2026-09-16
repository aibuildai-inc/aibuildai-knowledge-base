---
title: "MALI: A memory efficient and reverse accurate integrator for Neural ODEs"
entry_type: paper
source: "https://openreview.net/pdf?id=blfSjHeFM_e"
upstream_list: "Zymrael/awesome-neural-ode"
category: "Differential Equations in Deep Learning"
subcategory: "Training of Neural ODEs"
description: "Existing implementations of the adjoint method suffer from inaccuracy in reverse-time trajectory, while the naive method and the adaptive checkpoint adjoint method (ACA) have a memory cost that grows with integration time. In this project, based on the asynchronous leapfrog (ALF) solver, we propose the Memory-efficient ALF Integrator (MALI), which has a constant memory cost w.r.t number of solver steps in integration similar to the adjoint method, and guarantees accuracy in reverse-time trajectory (hence accuracy in gradient estimation)."
---

# MALI: A memory efficient and reverse accurate integrator for Neural ODEs

**Source**: [https://openreview.net/pdf?id=blfSjHeFM_e](https://openreview.net/pdf?id=blfSjHeFM_e)

**Category**: Differential Equations in Deep Learning | **Subcategory**: Training of Neural ODEs

## Description

Existing implementations of the adjoint method suffer from inaccuracy in reverse-time trajectory, while the naive method and the adaptive checkpoint adjoint method (ACA) have a memory cost that grows with integration time. In this project, based on the asynchronous leapfrog (ALF) solver, we propose the Memory-efficient ALF Integrator (MALI), which has a constant memory cost w.r.t number of solver steps in integration similar to the adjoint method, and guarantees accuracy in reverse-time trajectory (hence accuracy in gradient estimation).
