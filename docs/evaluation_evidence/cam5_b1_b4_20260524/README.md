# Cam 5 B1-B4 Evaluation Evidence

This folder contains small, trackable evidence files copied from the local B1-B4 experiment output:

```text
dtc_counting/outputs/final_cam5_b1_b4_20260524_v4/
```

Large videos and overlay media are intentionally not committed. The committed files are enough to verify the reported quantitative metrics and the MOI strategy used by B2/B3/B4.

## Method

| Baseline | ROI source | MOI source |
|---|---|---|
| B1 | Manual official ROI | Official MOI |
| B2 | Manual official ROI | Track-mined MOI aligned to official MOI IDs |
| B3 | SAM Automatic ROI | Track-mined MOI aligned to official MOI IDs |
| B4 | Grounding DINO + SAM ROI | Track-mined MOI aligned to official MOI IDs |

The key alignment step is:

```text
generated trajectory vectors -> vector matching -> official MOI IDs
```

This matches the method now used by the web demo auto path: SAM/Grounding-SAM bootstraps ROI, while MOI used for counting is generated from trajectories and aligned when a reference MOI file is available.

## Files

| File | Purpose |
|---|---|
| `comparison_summary.json` | Machine-readable B1-B4 metric summary |
| `comparison_summary.csv` | Table-friendly metric summary |
| `b2_moi_from_tracks_aligned.txt` | Track-mined MOI after official-ID alignment |
| `b3_bootstrap_decision.json` | Evidence that B3 used track-mined MOI fallback |
| `b4_bootstrap_decision.json` | Evidence that B4 used track-mined MOI fallback |

