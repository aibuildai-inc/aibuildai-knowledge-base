# Exploiting Homogeneity in Feasibility-Driven Polygon Packing

Competition: santa-2025
Rank: #9
Source: https://www.kaggle.com/c/santa-2025/writeups/exploiting-homogeneity-in-feasibility-driven-polyg

First I want to thank Kaggle for hosting such a challenging yet wonderful competition, and kudos to [Jeroen Gardeyn](https://www.kaggle.com/jern97) (author of [**sparrow**](https://github.com/JeroenGar/sparrow), who incidentally also participated in this competition and is ranked 6th) for developing such a delicate yet powerful solver package.

The solution proposed here is largely an extension of the vanilla **sparrow** solver, specialized for the *homogeneous* setting in this competition (all items are the same simple polygon). The main theme is: **exploit homogeneity aggressively**—in collision evaluation, in seeding, and in candidate generation.

Highlights in this solver (that complements vanilla sparrow):

1. A fast **collision severity kernel** using an SAT LUT (plus AABB early rejects).
2. **Low-discrepancy + hard-core thinning** to generate better feasible seeds (fast early progress).
3. **Motif mining**: explicitly learn and reuse repeating local patterns (structure reuse).

---

## Problem recap and metric

For each group size \\(N\\), you must place \\(N\\) identical “Christmas tree” polygons (translation + rotation) with **no overlaps** (touching is allowed), and you are scored by the area of the *smallest axis-aligned bounding square* of the union divided by \\(N\\). The total score is the sum over all groups. Concretely, for a group \\(N\\):

- Compute the union bounds \\([x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]\\)
- Let \\(S_N = \max(x_{\max}-x_{\min},\, y_{\max}-y_{\min})\\)
- Final: \\(\text{score} = \sum_N \text{score}_N\\), where \\(\text{score}_N = S_N^2 / N\\)

The official metric also enforces: no overlaps (but touching is OK), and \\(x,y \in [-100,100]\\).

---

## Baseline: Sparrow framing and why it’s a great fit for the task

The baseline framework is inherited from Sparrow’s central idea: rather than directly optimizing a combinatorial/continuous objective under a hard non‑overlap constraint, one alternates between (a) shrinking the container (or otherwise tightening the objective) and (b) solving the induced feasibility problem by temporarily allowing overlap and iteratively “separating” items until feasibility is recovered. This is effective because the feasible region is extremely restrictive: purely feasible‑to‑feasible local moves often cannot make meaningful progress, whereas controlled excursions into infeasibility allow traversal between distant feasible configurations. Sparrow explicitly motivates this relaxation strategy and organizes it into exploration and compression phases.

Within each shrink step, the core subroutine is a separation procedure:

- Maintain a scalar “infeasibility” measure \\(e(\mathbf{s})\\) for the current configuration \\(\mathbf{s}\\), formed as a sum of collision severities (and possibly wall violations).
- Repeatedly select colliding items and search for improved positions by sampling candidate transforms around/within the container.
- Use guided local search (GLS): persistent or severe collisions receive increased weights, which gradually forces the search to resolve them or to perform disruptive moves that escape local minima. Sparrow describes this mechanism in detail and highlights why a *continuous* collision severity is essential for guidance.

This baseline already works remarkably well on many irregular strip‑packing problems, but homogeneous instances expose two bottlenecks:

1. **Geometry cost dominates**: repeatedly evaluating collision severity between identical shapes becomes the main runtime.
2. **Lack of “regularity reuse”** : in homogeneous settings, good solutions often consist of small clusters repeated many times, and Sparrow “lacks a mechanism to repeat them,” limiting its ability to exploit such structure.

The remainder of this writeup focuses on three additions designed to address these bottlenecks directly.

---

## 1) Collision (severity) kernel: SAT LUT specialized for one polygon

### Motivation

Sparrow’s separation loop needs a collision **severity** signal—continuous-ish, cheap, and “good enough” to guide search. It doesn’t need to be the true signed distance. The sparrow paper explicitly argues many effective collision quantifiers are essentially **1D metrics** (“penetration-depth derivatives”). And it also notes taking a square root can turn an “area-like” overlap proxy into a **1D-like metric**. So the goal here was: build an extremely fast 1D-ish severity metric *for this one tree shape*.

### Key idea: precompute everything you can (homogeneity!)

Because every item is the same polygon:

- The set of SAT axes is fixed (up to rotation).
- Relative orientation between two trees is discrete (we use a fixed number of angle bins, e.g. 4096).
- So we can build a **lookup table of projections** indexed by relative angle.

### What the SAT LUT contains (implementation ingredients)

1. **4-part convex decomposition**
   The original tree polygon is concave, but it can be decomposed into 4 convex parts (tiers + trunk). This makes SAT cheap and robust because:

   - convex SAT is straightforward,
   - and we can early reject part pairs using part AABBs.
2. **Unique normal set (≈ 8 axes)** 
   Extract unique edge normals from the convex parts, normalized + deduped.
3. **Projection LUT by relative angle**
   For each relative angle bin \\(\Delta a\\), and for each part and axis, precompute the interval:

   \\[
[\min_{v \in \text{part}} u\cdot v,\; \max_{v \in \text{part}} u\cdot v]
\\]

   so at runtime, we only apply a shift from translation.

- **AABB early rejects**

  - tree-level AABB reject
  - per-part AABB reject before doing SAT on that part pair

### Runtime severity: “SAT violation” as a smooth-ish 1D penalty

For a part-pair and a tested axis \\(u\\), you compute the overlap of the projected intervals after applying the translation shift; if any axis separates, there’s no collision for that part pair.

In code (simplified from `convex_parts_violation_sat` + `trees_severity`), the key is:

- compute overlap \\(= \min(\text{hi}) - \max(\text{lo})\\)
- add `eps_gap` so you can require a small separation margin
- if `v <= 0`: separated
- else `v` is the “penetration-style deficit” for that part pair

Then aggregate all violated part pairs using a selectable norm:

- L1 (sum), L∞ (max), L2 (RSS), L4 (smooth max)

In practice, L1 is found to be more effective than others in our settings.

### Code excerpt (SAT LUT build + severity)

```rust
// (1) Precompute projections for all relative angles, per part-pair and axis.
for rel in 0..sat.k {
    let d = (rel as Real) * sat.step;
    let (cd, sd) = d.cos_sin();

    for pb in 0..K_PARTS {
        for pa in 0..K_PARTS {
            for k in 0..sat.axes {
                let ax = sat.axis_local[k as usize];
                let wx = cd.mul_add(ax.x, -sd * ax.y);
                let wy = sd.mul_add(ax.x, cd * ax.y);

                // project B (rotated by rel) onto A's axis k
                let (mn, mx) = interval_for_axis(&sat.part_verts[pb], wx, wy);
                sat.proj[row + pb * axes + k] = Interval { mn, mx };
            }
        }
    }
}

// (2) Runtime: for each overlapping part-pair, find min axis-violation (SAT deficit)
if let Some(v) = convex_parts_violation_sat(...) {
    match sat.severity_agg {
        SatSeverityAgg::L1 => acc += v,
        SatSeverityAgg::Linf => acc = acc.max(v),
        // ...
    }
}
```

---

## 2) Low-discrepancy seeding for initial placements (QMC + hard-core thinning)

### Motivation: make the first shrink “nice”

Intuitively, sparrow works best when after shrinking the container, the induced infeasibility is:

- spread out (many small overlaps),
- not too deep (avoid extreme entanglement),
- locally resolvable.

Random seeds tend to create:

- clumps + big holes,
- deep interlocks after the first shrink,
- and a lot of wasted separation effort.

So the idea is to use a **low-discrepancy stream** to get even coverage quickly, but **hard feasibility** (no overlaps) is preferred because the solver assumes feasible starts.

### The practical “hybrid” idea that is mathematically coherent:

1. Generate a low-discrepancy stream \\(u_k \in [0,1]^2\\).
2. Map to a square:

   \\[
p_k \in [-L/2, L/2]^2
\\]

3. Accept \\(p_k\\) only if it is at least \\((2R+\eta)\\) away from all previously accepted centers:

\\[
\|p_k - p_i\|_2 \ge 2R + \eta
\\]

This is Poisson-disc / hard-core thinning.

If you also need orientations, treat the transform as 3D:

\\[
(x,y,\theta)\in [-L/2,L/2]^2 \times \Theta
\\]

and use a low‑discrepancy set in \\([0,1]^3\\) where the third coordinate selects a discrete angle bucket. One idea that stems from here is that enforcing homogeneous angles from the start mimicks "lattice" pattern that is often favored by large \\(N\\). But the pseudo-random angles initiation is good enough for now. Empirically, this improves early convergence: the solver spends less time “unwinding chaos” and more time tightening. I also analyzed “discrepancy vs space-filling” tradeoffs separately (main idea: the hard-core filter breaks perfect QMC structure, but you keep much of the uniformity while improving coverage and feasibility).

### Implementation: Owen-scrambled Sobol + distance filter

In code, this is implemented as CPU method `LdsHybrid`:

- Use **Owen-scrambled Sobol** points (`sobol_burley`) as the low-discrepancy generator.
- Compute a conservative “bounding circle radius” \\(R\\) from the tree vertices.
- Enforce minimum spacing:

  \\[
\text{min\_sep} = 2R + \text{eps\_gap} + \eta
\\]
- If we fail to place all trees after `max_samples`, we restart with a slightly larger side (`side_growth`), up to `max_restarts`.

### Code excerpt (core seeding loop)

```rust
let r = tree_bounding_radius(sat);
let gap = solver.search.eps_gap.max(0.0);
let eta = p.eta.max(0.0);
let min_sep = 2.0 * r + gap + eta;
let min_sep2 = min_sep * min_sep;

for _ in 0..p.max_samples {
    let [u0, u1, u2] = seq.next3();
    let x = (u0 * 2.0 - 1.0) * half;
    let y = (u1 * 2.0 - 1.0) * half;

    // Hard-core filter
    if poses.iter().any(|q| (x-q.x).powi(2) + (y-q.y).powi(2) < min_sep2) {
        continue;
    }

    // Discrete angle from third coordinate
    let ai = ((u2 * (m as Real)) as usize).min(m - 1);
    set_angle_idx(&mut pose, allowed_angles[ai], sat);

    poses.push(pose);
    if poses.len() == n { return poses; }
}
```

---

## 3) Motif mining: explicitly learn and reuse repeating local structure

### Motivation (this is *the* homogeneous trick)

Sparrow’s own paper calls out a “blind spot” on homogeneous instances:

> “it lacks a mechanism to repeat them”

In homogeneous packing, good solutions often consist of:

- a small number of tight local clusters,
- repeated many times with the same relative transforms.

So instead of hoping the solver “accidentally” recreates the same micro-structure everywhere, I added a mechanism to **mine repeating local transforms** and **inject them as candidate proposals** during sampling.

### Motif representation (portable building blocks)

A motif is a relative transform from an **anchor tree** **\\(j\\)** to a neighbor \\(i\\), expressed in the anchor’s local frame:

- relative translation:

  \\[
\begin{aligned}
  \Delta x &= \cos\theta_j (x_i-x_j) + \sin\theta_j (y_i-y_j) \\
  \Delta y &= -\sin\theta_j (x_i-x_j) + \cos\theta_j (y_i-y_j)
  \end{aligned}
\\]
- relative angle (discrete bins):

  \\[
\Delta a = a_i - a_j \quad \text{(wrapped)}
\\]

This “anchor-local” choice is important:

- motifs are not tied to global axes,
- they’re rotation-invariant at the pattern level,
- they transfer cleanly to any other anchor.

### Mining motifs (cheap histogram of local patterns)

In `solver/sampling.rs::mine_motifs`:

- For each anchor \\(j\\), query nearby candidates from the **broadphase grid** (not \\(O(n^2)\\)).
- Keep the `motif_neighbors` nearest neighbors (approximate KNN).
- Quantize \\((\Delta x,\Delta y)\\) to bins of size `motif_bin_xy`, and optionally quantize \\(\Delta a\\) by `motif_bin_da`.
- Build a histogram keyed by `(qx, qy, da)`.
- Keep the top `motif_max` bins (prefer those with count ≥ `motif_min_count`).
- Store the *mean* dx/dy per bin (reduces quantization bias).

### Using motifs during sampling (structured candidates)

Motifs get injected as a separate candidate family (`t_motif` samples per move), alongside:

- diverse (uniform),
- focused (Gaussian around current),
- contact-based proposals,
- angle-only and micro-translate escapes.

During a move of tree `idx`:

- choose an anchor neighbor `j` (biased toward colliding anchors)
- either:

  - “match” the current relative transform to the best motif (`motif_p_match`), or
  - sample motifs weighted by frequency
- instantiate:

  \\[
p_i \leftarrow p_j \oplus (\Delta x,\Delta y,\Delta a)
\\]
- optionally jitter and reject if container clamping would distort it too much (`motif_clamp_reject_eps`)
- evaluate energy, keep top candidates, then refine

### Code excerpt (mining + proposing)

Mining key loop (trimmed):

```rust
// For each anchor j, keep k nearest neighbors from grid query,
// and histogram relative transforms in anchor-local coords.
for j in 0..n {
    // query grid in expanded AABB, keep k nearest by r^2
    // ...
    for &(_, i) in knn.iter() {
        let dxl = pj.cosv.mul_add(dxw, pj.sinv * dyw);
        let dyl = (-pj.sinv).mul_add(dxw, pj.cosv * dyw);
        let qx = (dxl / bin_xy).round() as i32;
        let qy = (dyl / bin_xy).round() as i32;
        let da = quantized_angle_delta(pi.a, pj.a, motif_bin_da);

        let e = hist.entry((qx, qy, da)).or_default();
        e.cnt += 1;
        e.sum_dx += dxl;
        e.sum_dy += dyl;
    }
}
```

Proposal injection (trimmed):

```rust
let anchor_idx = pick_anchor(&anchors, &colliding_anchors, rng, prefer_colliding_p);
let pj = sol.poses[anchor_idx];

// pick motif: match vs weighted-random
let m = if rng.uniform01() < motif_p_match { best_match(...) }
        else { extras.pick_motif_weighted(rng) };

let new_a = pj.a.wrapping_add(m.da as u32);
set_angle_idx(&mut p, new_a, sat);

// anchor-local (dx,dy) -> world
let dxw = pj.cosv.mul_add(m.dx, -pj.sinv * m.dy);
let dyw = pj.sinv.mul_add(m.dx, pj.cosv * m.dy);
p.x = pj.x + dxw;
p.y = pj.y + dyw;
```

---

## How the three ideas fit together in the actual solver loop

A practical way to view the final solver is:

- **Bootstrap** a feasible seed (LDS hybrid).
- Repeat:

  - **shrink** the container side a bit
  - run **separation**: move colliding items until feasible (or stall)

    - each move uses candidate sampling:

      - random (diverse/focused)
      - **structured** (motif/contact)
      - escape moves (angle-only / micro-translate)
    - each candidate is scored by a fast weighted energy:

      \\[
E = \sum_{\text{walls}} w_{\text{wall}} \cdot v_{\text{wall}} \;+\; \sum_{j \in \text{neighbors}} w_{ij} \cdot \text{sev}(i,j)
\\]

      where `sev(i,j)` is the SAT-LUT severity.

In conclusion, the SAT LUT is the “engine” that makes this affordable; the LDS seed makes the first few shrinks behave nicely; and motif mining helps maintain / reproduce structure across the layout instead of rediscovering it from scratch.
