# TH08 human action pretraining

We decoded 64 complete six-stage Royalflare replays from 64 credited players across four team shots. Each difficulty contributed 24 Hard or Lunatic training players and eight held players. The native key stream yielded 1,405,879 training decisions and 487,927 held decisions at a four-tick stride. After tiling within each stage, 1,396,800 training decisions per pass and 484,736 held decisions entered the model. Each replay passed the native checksum, stage and endpoint-score checks. [Human sources and split](source-manifest.json) · [Replay archive](https://maribelhearn.com/royalflare/th08)

The causal action model started from four-title human pretraining and trained on TH08 for four passes on an RTX 4070. Its input was previous human keys and replay metadata. The comparison measures action prediction with human history. A closed-loop TH08 visual policy requires a separate original-game evaluation. [Training protocol](action-prior/protocol.json) · [Training trace](action-prior/training.json) · [Final weights](action-prior/epoch-4.pt)

| Author-macro metric | Repeat last | Frozen four-title prior, mean metadata | Four-epoch TH08 model |
|---|---:|---:|---:|
| Next direction | 76.31% | 77.05% | 77.15% |
| New direction at an actual turn | 0% | 10.83% | 10.61% |

The paired gain in next-direction accuracy over the frozen mean-metadata control was 0.10 percentage points (95% author-bootstrap interval −0.006 to +0.208). The turning difference was −0.216 points (−0.981 to +0.492). The new model predicted the correct turn on 9.31% of Hard turns and 11.91% of Lunatic turns. The frozen model with unused unknown-title and unknown-shot embeddings scored only 40.60% on next direction. Averaging the embeddings of its trained titles and shots supplied a stronger post-hoc transfer control. Fourteen of these 16 names were absent from the historical parent training roster. On those 14 players, the frozen mean-metadata control scored 76.99% and the fine-tune scored 77.13%. [Paired controls and credited-author audit](action-prior/transfer-controls.json) · [Rank and author results](action-prior/report.json) · [Figure](held-direction.png)
