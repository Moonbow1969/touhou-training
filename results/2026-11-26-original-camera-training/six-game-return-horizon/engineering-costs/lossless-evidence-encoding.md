# Lossless transport of original TH17 evidence images

Across 32 completed Hard and Lunatic test credits, the four per-game PNGs occupied 61.48 MiB. There were 128 distinct full-file checksums among 128 PNGs. Each tested encoding preserved every decoded RGBA byte and image dimension. Existing source archives retain the original PNG file bytes.

| WebP method | Output MiB | Saved MiB | Reduction | Additional encode time | Serial break-even link |
|---|---:|---:|---:|---:|---:|
| webp0 | 51.67 | 9.81 | 16.0% | 3.84 s | 21.5 Mbit/s |
| webp2 | 45.89 | 15.59 | 25.4% | 31.04 s | 4.2 Mbit/s |

Break-even divides saved wire bits by measured additional encoding seconds, assuming serial re-encoding of a PNG already produced by the worker. The timer covers encoding on the benchmark host. PNG decoding, output verification, network latency, protocol overhead and network transfer are outside it. Native direct encoding, work overlapped with gameplay, or a filesystem archive may have different costs. No step-time game traffic was rerouted in this measurement.

[Per-image byte counts, timings and checksums](lossless-evidence-encoding.json) · [Original source bundle](../th17-held-native-records.release.json)
