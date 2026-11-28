# TH08 original Normal native color-motion probe

Claude Opus 5.5 proposed investigating how much local motion remains usable after the playfield is reduced to 96×112 grayscale. Steam Deck replayed two original 0.02 Normal Ghost Stage 1 human sources in isolated processes. The collector saved 512 coordinate-aligned pairs per source. Each pair contains two 64×64 native RGB crops four original ticks apart. All 2,048 luma tiles were reconstructed and matched the recorded low-resolution frames exactly, along with 20 independently stored full-color witnesses. The previous frame uses the same crop center as the current frame.

The inexpensive probe used 509 chronological training windows from one author and 512 from a separate author. Both paths received the same native-coordinate crop centers. The fitted two-convolution decoder took four low-resolution camera pairs. A bilinearly enlarged grayscale difference is its reference. Each held label scores the absolute RGB difference between the paired original crops.

| Held mean absolute error, lower is better | Bilinear | Fitted | Relative reduction |
|---|---:|---:|---:|
| All pixels | 0.05630 | 0.05472 | 2.8% |
| Bright saturated motion pixels | 0.17002 | 0.14006 | 17.6% |

This is a sensor diagnostic with privileged crop centers. The labels include moving sprites and effects, so the experiment cannot measure bullet recall, avoidance, or original Stage 2 outcomes. There is one author per split. This measured improvement leaves headroom in the reconstruction task and supplies no policy promotion decision.

[Train camera audit](train-original-motion-audit.json) · [Held camera audit](held-original-motion-audit.json) · [Probe protocol](protocol.json) · [Probe training log and metrics](report.json)
