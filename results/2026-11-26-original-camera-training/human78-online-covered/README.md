# Three-game PPO from the expanded human SFT

The spatial-attention SFT trained on 235,457 human camera decisions from TH10, TH12 and TH13. Its deployed actor collected 48 original Hard and Lunatic training games, eight per title and difficulty. The new PPO completed four epochs over 22,857 playable decisions. The parent controls movement, focus, shot and cast directly from the camera. [PPO checkpoint and optimizer history](policy/report.json) · [Original counts by game and rank](verified-training-by-rank.json) · [Human SFT fit](../scaled-human-sft/new-64/policy/report.json)

The historical run name counts 78 training replay-header aliases. Public scoreboard credits identify 77 distinct training authors and six held authors. Both aliases for the overlapping author appeared on the training side. [Source-by-source author grouping](../th11-expansion/retail-camera-author-groups-v2.json)

The original controller had already collected the 32 TH10 and TH13 credits. All 16 TH12 credits came from new starts under the independently validated dark-scene controller. On Hard seed 1421203, the original game consumed the previous input at fatal stock loss, then consumed the last requested input after death. All 500 camera observations, four-head action likelihoods and values were checked. The optimizer uses the 499 playable actions, assigns the life-loss reward to zero-based action 498 and ends the return there with zero bootstrap. [Original timing and consumed keys](native-boundaries/train-human78-sft-1421203-beta.json) · [Exact optimizer boundary](native-boundaries/1421203-optimizer-audit.json)

The raw camera release preserves the sampled inputs. The small correction file carries the TH13 native-clock reward arrays. Reconstruct the learner's exact input files using the following command from the repository root with the downloaded assets. Every reconstructed file is checked against its training SHA-256. [Raw release asset and checksum](release-archive.json) · [Optimizer reward corrections](optimizer-reward-corrections.zip) · [Exact training manifest](corrected/credits.json)

```sh
python -m scripts.rebuild_original_training_view restore --archive three-game-hard-lunatic-human78-ppo-native-coverage-original-behavior.zip --patch optimizer-reward-corrections.zip --out training-shards
```
