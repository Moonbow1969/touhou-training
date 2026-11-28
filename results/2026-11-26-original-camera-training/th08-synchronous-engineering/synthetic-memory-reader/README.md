# Why the original frozen-state check can fail

A synthetic Windows counter and a second Windows process ran under the same Wine image as the TH08 original. A Linux monitor stopped the counter process through 100 intervals per condition. It sampled Linux process state 45 times per interval and independently read the memory-mapped counter before and after each interval.

| Windows reader | Intervals | Linux T | Linux tracing stop (t) | Linux runnable (R) | Intervals advancing counter |
|---|---:|---:|---:|---:|---:|
| Absent | 100 | 4,500 | 0 | 0 | 0 |
| Repeated ReadProcessMemory | 100 | 3,829 | 117 | 554 | 0 |
| Reader idle acknowledgement around the sample | 100 | 4,500 | 0 | 0 | 0 |

The treatment waited for the broker to complete its Windows memory reads and acknowledge an odd-numbered request epoch. After the frozen interval, the monitor briefly resumed reads and verified a fresh source-counter sample at the same address, then reestablished observer quiescence before resuming the process. A preliminary fixture compared two independently written counters and could freeze in the single-instruction gap between stores. The final fixture uses a single shared address for both readings.

The game-operator candidate enables the new handshake with `-DTH08_QUIET_READS` for both `bracket.c` and `stream.c`, in addition to their existing engineering options. The candidate compiled on the target Linux architecture. The original-game run remains unqualified. These fixture outcomes require no game assets or policy and account for one way the original proof could have failed, while the exact predicate in the prior original failures was never recorded.

[Raw count and source SHA256](qualification.json) · [Final quiescence fixture source](fixture/quiescence-v2) · [Matched preceding probe](fixture/native-proof-v2)
