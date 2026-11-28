# TH08 matched resolution and player position experiment

One frozen camera controller played 32 complete Ghost-team credits on the official TH08 0.02 trial on Normal, without continues. Twenty-seven met the original input-integrity gate and supplied 26,354 decisions and 81 confirmed fatal hits. Five original runs with a stale input window were retained in the collection record and excluded from the learner. None of the 32 reached Stage 2.

The existing 96×112 camera and the new 192×224 camera came from the same source images. Grayscale conversion and rounding differ slightly between the two paths. The independent source audit found four pixels with a difference of two levels after area reprojection among over 331 million logged policy pixels. All other pixels were within one level. The first learner rejected the fifth credit before any optimization because its reprojection bound was too strict. The registered correction matched the source audit's bound and preserved the entire admitted cohort.

The four camera models each ran five 540-update fits. Within each fold, all models used the same original images, qualifying time windows, physical crop lattice spacing and independent calibration partition. The comparison used a logistic model informed by native position, resources and elapsed time. In the two diagnostic arms, actual native player position provided the crop center. The deployable camera arms inferred position from pixels.

| Camera | Crop center | Mean fold AP | Fatal hits alerted ≥20 ticks early | Combined gate |
|---|---|---:|---:|---:|
| 96×112 | Learned | 0.1051 | 20/81 | 0/5 |
| 96×112 | Native coordinates | 0.0863 | 18/81 | 0/5 |
| 192×224 | Learned | 0.0825 | 21/81 | 0/5 |
| 192×224 | Native coordinates | 0.1010 | 19/81 | 0/5 |

The logistic mean AP was 0.0848. Even the branches given exact coordinates had insufficient early coverage. The observed results do not support deploying this hit predictor as PPO shaping on this cohort. The study leaves open other sensors, objectives and data scales.

[Fold figure](matched-resolution.png) · [Held folds and optimizer logs](report.json) · [Sealed input hashes and groups](admitted.json) · [Collection evidence](../th08-highres-native)
