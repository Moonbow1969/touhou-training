# 261123 TH14 policy drift and replay coverage

The 8,192-step PPO policy assigned more probability to standing still on recorded TH14 expert states than its SFT parent. Its first-miss median improved in an earlier three-pair native test, with zero Stage 1 clears in either learned arm. The training task ended after 512 decisions, before the native Stage 1 clear observed with the hybrid baseline. [Native pilot](261123-th14-full-policy-ppo.md)

## Offline drift

Both actors received the same recorded observations, using the SFT observation transforms. Idle is movement index 4. Sampled idle is the mean categorical probability of that action, computed without drawing samples. Greedy movement accuracy compares the actor’s highest-probability direction with the replay’s recorded direction. The TH14 replay participated in SFT validation. TH17 test replays are from different players on a title present in SFT training.

| Replay slice | Decisions | Expert idle | Sampled idle, SFT → PPO | Greedy idle, SFT → PPO | Greedy movement accuracy, SFT → PPO |
| --- | ---: | ---: | ---: | ---: | ---: |
| TH14 Normal Reimu A Stage 1 validation | 2,288 | 42.1% | 40.8% → 50.9% | 49.0% → 65.7% | 72.8% → 49.0% |
| First 500 stage frames of the same replay | 124 | 42.7% | 40.0% → 57.0% | 50.0% → 72.6% | 70.2% → 43.5% |
| TH17 held-out players, every fourth paired decision | 12,860 | 36.7% | 31.9% → 35.3% | 38.6% → 43.7% | 64.9% → 45.0% |

The TH14 differences are largest in the opening slice. These figures measure imitation and action distributions on human trajectories. The native sampled comparison measures performance on states reached by the learned policies. [Machine-readable audit](../../results/2026-11-23-th14-full-policy/sft-ppo8192-offline-drift.json)

Reproduce on CPU with the archived title-equal seed-61 SFT checkpoint, 8,192-step PPO archive and aligned replay snapshot. Keep SFT `metrics.json` and the PPO archive’s original training `report.json` alongside the weights. The audit verifies parent lineage, loads tensor weights, and records the SHA-256 of each evaluated shard without writing replay observations to its report.

```powershell
python -m scripts.audit_full_policy_drift --base models-local/overnight-title-equal-61/sft/policy.pt --ppo models-local/th14-full-policy-v2-8192/policy.zip --data replays/data/paired-overnight-youmu-otter --out runs/th14-offline-drift.json
```

## Native horizon and SFT coverage

The 512-decision horizon is terminal at roughly 2,066 native stage frames. The last 4,096-step segment observed nine first misses and one horizon terminal. The hybrid cleared Stage 1 at roughly 9,900 frames. Preserve the 512-decision recipe when repeating the early-survival comparison. A separate Stage 1 clear experiment can initialize from the same SFT checkpoint and use a horizon long enough to reach the boss and stage transition. The horizon also changes the per-step survival reward divisor. Record the new recipe and avoid treating its returns as a continuation of the early-survival task. [Original training log](../../results/2026-11-23-th14-full-policy/survival-8192-train.json)

The SFT checkpoint used 147,560 decisions across 38 stages from eight replays. TH14 contributes 2,415 decisions from one Normal Reimu A Stage 1 and 18,440 from one six-stage Lunatic Sakuya A replay. The held-out TH14 validation set is the one 2,288-decision Stage 1 in the table. Three new complete Normal Reimu A training replays provide about 67,600 four-frame windows before alignment and eligibility filtering. The reserved full Normal replay `th14_ud0559.rpy` and its player group are excluded from the audit and the SFT input and selection sets. Its paired native test still needs capture. [Capture split and hashes](../../results/2026-11-23-th14-full-policy/replay-capture-split.json)

## Isolated Wine replay capture benchmark

Qualify the privately installed Steam 1.00b build as a game worker before collecting. Use one Wine prefix, virtual display and audio sink per job. Place the selected replay in that prefix’s own ShanghaiAlice TH14 profile under a single replay slot. Verify the replay file fingerprint against the frozen plan and the executable fingerprint against the native pilot. Launch Replay in the game’s own menu and verify the start scene and decoded input signature before accepting rows. [Isolated worker setup](../../tools/rollout-worker-probe/README.md)

The current capture path uses Win32 process reads and pause/resume in `ReplayProcess`, the common 96-row structured reader, and live input alignment against the decoded action stream. A Windows Python runtime inside the isolated Wine prefix can first test that path with matching Win32 dependencies. A small Windows capture helper compiled with the existing worker toolchain is an alternative if the Python runtime or debugger bridge proves unstable. A helper must emit the same player and projectile fields, controllability, native input word and stage frame, then reuse the Python alignment and shard writers. [Capture code](../../replays/collect_stage.py), [Structured reader](../../environment/structured.py)

Benchmark Stage 1 of the first training replay in isolation before a six-stage run. Record startup, active capture time, accepted decisions per second, stage-frame delta histogram, alignment offset and coverage. The existing gate is at least 95% moving-input agreement and 90% frame coverage. Repeat the same replay through the stock Windows capture path and compare paired player positions, inputs and projectile samples on common native stage frames. Once parity is measured, run two isolated full replays concurrently and report per-worker and aggregate accepted rates. The audit opened no Wine game and collected no new replay states.
