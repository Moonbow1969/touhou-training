# Human replay sources

The Gensokyo Replay Archive lists 10,981 downloadable replay entries across TH10, TH12 and TH13 in the November 26 inventory. TH10 contributes 4,065 entries, TH12 contributes 3,933 and TH13 contributes 2,983. The archive retains the player spelling, shot, replay type, upload date and source link. The accompanying compressed JSON Lines file contains one record per listing. [TH10 archive](https://maribelhearn.com/gensokyo?game=MoF) · [TH12 archive](https://maribelhearn.com/gensokyo?game=UFO) · [TH13 archive](https://maribelhearn.com/gensokyo?game=TD)

Each listing counts one source record. A single player may submit many replays, one binary may appear in multiple archives, and a practice replay may cover one stage. Training sample counts come from separate capture receipts. Each admitted example pairs a contemporaneous stock-game camera image with the four future input ticks decoded from that exact replay. The receipts record the source checksum, captured tick span, eligible control frames and native input agreement. Group complete source players before producing training and held examples.

The TH13 archive starts on August 13, 2011. The verified TH13 camera captures use retail version 1.00c. Replay compatibility with an earlier trial build requires independent playback and an input-alignment check. The metadata index includes public archive entries and source links. See [machine-readable counts](external-replay-source-counts.json) and [compressed index](external-replay-candidates.jsonl.gz).

TH12 also has an external C++ implementation with a retail replay comparison tool. Its documented validation coverage and proposed use for accelerated training are described in the [TH12 rollout resource](th12-headless-candidate.md).
