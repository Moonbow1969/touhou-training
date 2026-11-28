# Synchronous TH08 control qualification

The two registered engineering trials ended before a full original stock. The changed controller remains ineligible for training or evaluation. Both used the official TH08 0.02 Normal trial, Ghost Team, and the same four-head image-only policy as the 32 preceding training credits.

| Isolated lane | Witnessed decisions | Last native Stage 1 tick | Result |
|---|---:|---:|---|
| A | 0 | 36 | The post-input frozen-state check rejected the transition during stock entry. |
| B | 4 | 92 | The pre-input frozen-state check rejected the next policy action. |

On lane B the Linux game PID was stopped at original ticks 76, 80, 84 and 88. Each paused interval included an independent clock-reader heartbeat, the RGB capture, learned action, X11 acknowledgement, and resumption of the same PID. The native saved input word matched the requested action on all 16 following original ticks. The small number of windows does not measure the tail or throughput of a completed stock. No Stage 2 or terminal stock outcome is available. The precise failed predicate within the combined Linux stopped-state and original stage/tick test was unobserved. The interaction between the background Wine memory reader and Linux stop-state reporting needs direct instrumentation.

The watchdog resumed a separate test process after an 801 ms stall and immediately after its owner exited. The original game is the only PID to receive SIGSTOP. The Wine server, 32-bit read-only observer, Xvfb and CPU inference service continue running. To build the opt-in variant supply `-DTH08_SYNCHRONOUS_ENGINEERING` when compiling the TH08 0.02 `tick4/stream.c`, and bind the resulting binary at the precise `/clock/stream` entry point consumed by `postcredit-ranking.sh`. The ordinary rollout does not define this option. Two preceding deployment attempts are recorded separately. Their entry point used the ordinary operator, and both sources remain preserved outside the synchronous denominator.

[Original registration](registration.json) · [Entry-point correction before synchronous input](correct-runtime-mount/registration.json) · [Watchdog and deployed source hashes](watchdog.json) · [Native words, raw downloads and endpoint accounting](engineering-results.json)
