# Four complete TH07 retail observer qualifications

The new original-game worker passed **4 of 4** registered engineering credits on Momentum. The prespecified qualification threshold was 3 of 4. All four used the same frozen parent policy on the owner's unmodified retail 1.00b build, Normal Reimu A, two spare lives, three initial Bombs, and zero continues. Each credit ran from Game Start through the original Game Over. Stage 2 arrivals were 0/4.

| Seed | Stage 1 ticks | Exact full four-tick windows | Actual policy decisions | Native-read p99.9 upper bound | Maximum native read | Official score |
|---|---:|---:|---:|---:|---:|---:|
| 2173701 | 7,129 | 1,781 / 1,781 | 1,782 | 0.09 ms | 4.05 ms | 929,970 |
| 2173702 | 10,081 | 2,519 / 2,519 | 2,520 | 0.09 ms | 9.84 ms | 704,500 |
| 2173703 | 6,347 | 1,585 / 1,585 | 1,586 | 0.08 ms | 33.71 ms | 695,280 |
| 2173704 | 2,934 | 732 / 732 | 733 | 0.09 ms | 76.47 ms | 583,110 |

The official input recordings and an independent native witness confirmed all 6,621 applied actions, including the four short terminal windows. There were zero missed scheduled decisions. Two separate scorecard captures agreed with the officially saved replay for all four games. Each game container used an average of 2.05–2.06 CPU cores with zero throttled periods. The CPU calculation excludes the separate host policy process.

The async writer sealed 39,735 sequenced records in 3,377,143 journal bytes. Of 764,110 measured native reads, 2,727 were rejected before entering the verified state stream. The latency distribution includes these retries. Each of the registered CPU sets ran two successive credits. The longest read was 76.47 ms, below the 100 ms cutoff. Four accepted starts establish that this build passes the small engineering gate; more independent sessions are needed to estimate a sustained acceptance rate of 90%.

The regression suite accepted consistent redundant reads, rejected a torn duplicate even when its checksum was recomputed, and rejected a corrupted checksum. Removing the entire independent native witness for one four-tick command caused the action verifier to fail despite an intact official replay. A prior case with a last live score of 625,510 and a final score of 625,830 now obtains its official score from corroborating result-screen digits and the replay.

The [per-credit metrics](summary.json), [regression results](fixtures.json), [preregistered protocol](protocol.json), [build and controller hashes](source-sha256.txt), and [raw source receipt](raw.release.json) are included. The latter indexes official input recordings, raw camera histories, independent timing and action evidence, terminal screenshots, and the sealed journals. No policy was fitted or promoted in these engineering runs.
