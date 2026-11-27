# Measured throughput, transfer and storage

Gamer streamed an 11,865,273-byte sealed game archive three times each to Momentum and Steam Deck over SSH with compression disabled. The receivers hashed and discarded the bytes. Median end-to-end time, including authentication and checksum, was 1.35 seconds to Momentum (70.5 Mbit/s) and 1.58 seconds to Steam Deck (60.0 Mbit/s). One transfer includes thousands of actions. These measurements quantify batched transport only. [Durations and verified digest](verified-lan-transfers.json)

The 128 PNG evidence images from 32 held TH17 games had 128 distinct checksums. Re-encoding the existing PNGs with lossless WebP saved 16.0% at method 0 and 25.4% at method 2. Their serial break-even link rates were 21.5 and 4.2 Mbit/s, below both measured transfer rates. On these routes we retain the original evidence and batch it after each credit. Re-encoding on the rollout path would add CPU time. [All pixel-equivalence checks and encode timings](lossless-evidence-encoding.md)

| Original game | Original credits | PPO decisions | Exact compressed learner input (MiB) | Archive bytes per decision |
|---|---:|---:|---:|---:|
| TH08 | 32 | 22,962 | 163.5 | 7,464 |
| TH10 | 32 | 6,932 | 62.9 | 9,510 |
| TH11 | 32 | 7,630 | 47.7 | 6,553 |
| TH12 | 32 | 18,259 | 145.1 | 8,332 |
| TH13 | 32 | 19,408 | 120.6 | 6,518 |
| TH17 | 32 | 31,072 | 213.5 | 7,205 |

The complete six-game input occupies 753.3 MiB for 106,263 unique decisions. The exact archives include the per-game compressed trajectory payload and a copy of the full manifest. Original recordings and checkpoint downloads are accounted for separately. These numbers bound archive transport for one transfer per batch; they do not measure protocol round trips or per-step network traffic.

| Four-instance TH17 worker measurement | Common-window length | Mean decisions/s with four active | Mean decisions/s including gaps | Median interval between credits |
|---|---:|---:|---:|---:|
| Training | 455.1 s | 29.6 | 24.7 | 32.6 s |
| Held evaluation | 3788.7 s | 29.6 | 12.8 | 33.1 s |

On Momentum, the RTX 5090 traversed the 257,145-window human camera corpus twenty times. The 5,142,900 training presentations took 189.0 seconds summed over the twenty training loops (27,217 presentations/s). That timer excludes staging, replay decoding, held inference, checkpoint writing and original gameplay. A new physical transition and a repeated optimization sample carry different information.

The TH17 gap runs from the last applied action in one credit to the next first action on that instance. It includes menu reset, evidence collection, packing, model readiness and scheduling. The figures use the interval when four slots were allocated to this study. Separate native inputs retain stage, stock and actual key evidence.

[Native training intervals](../th17-train-actual-rollout-throughput.json) · [Native held intervals](../th17-held-actual-rollout-throughput.json) · [Optimizer trace](../six-gae985/policy/report.json) · [Twenty-epoch learner trace](../../four-title-complete-epochs/epoch-20/report.json)
