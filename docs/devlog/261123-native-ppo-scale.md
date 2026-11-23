# Native TH14 PPO at rollout scale

## Runs and measurements

The distributed learner uses the unmodified TH14 0.01b trial executable. The actor receives relative projectile positions and velocities plus a 10-field context, and selects movement, focus, shoot and Bomb. The observation space differs from the Steam 1.00b policies. The first distributed run starts from random initialization, and subsequent runs restore both the policy and its optimizer. Each survival episode terminates at the first miss, stage advance, game end, or 3,300 decisions. The survival reward uses native progress and no geometric controller.

| Run | Workers | Accepted decisions | Collection and optimization |
| --- | ---: | ---: | --- |
| Gate | 4 across 2 hosts | 4,096 | 52.6–57.9 decisions/s after the opening round. The 5090 optimizer took about 0.21 s per established round. |
| Expansion from the gate checkpoint | 6 across 3 hosts | 12,288 additional, 16,384 total | 88.8–108.0 decisions/s after the opening round. Momentum's CPU optimizer median was 0.277 s per round, including the opening update. All 12,288 accepted decisions advanced four native ticks. |
| Scale from the expansion checkpoint | 11 across 6 hosts | 248,064 additional requested, 264,448 target | 57 rounds of 4,352 decisions. The report is a running snapshot. |

The scale configuration gives two 256-decision fragments to each of the six faster actors and one to each of the other five. The seven-lane scripted endurance run exercised Endeavour, Entropy and Steam Deck for 13,056 decisions, with 95 successful Retry sequences and one five-tick decision. The PPO actor discards an invalid fragment and resamples. The learner retains the latest two checkpoints and rejects snapshots that exceed the configured disk budget or free-space reserve. [Gate](../../results/2026-11-23-distributed-rollouts/trial-ppo-4096.json), [expansion](../../results/2026-11-23-distributed-rollouts/trial-ppo-16384.json), [scale progress](../../results/2026-11-23-distributed-rollouts/trial-ppo-scale-running.json), [endurance](../../results/2026-11-23-distributed-rollouts/th14-trial-seven-lane-endurance.json)

In a separate native comparison, the 4,096-decision checkpoint increased median first miss from **1,118 to 1,250 frames** and went farther in four of six paired policy-seed groups. Both arms cleared Stage 1 in **0/6**. Native RNG was not controlled. This is a one-training-seed pilot. [Evaluation traces](../../results/2026-11-23-distributed-rollouts/trial-ppo-4096-eval.json)

## Steam 1.00b continuation

The Steam test uses the original SFT parent and resumes the 8,192-step full-policy PPO archive. It retains the 512-decision survival horizon for lineage and schedules a sampled, eight-block SFT/PPO/hybrid evaluation after 65,536 total decisions. An earlier segment reached 13,312 durable steps before losing Windows foreground. The interrupted game had reached its Game Over menu when the runner was started again. The runner now supplies an opt-in native Retry for that verified setup. The first recovered update accepted 256 four-frame decisions, and all 52 eligible horizontal commands moved in the requested direction. [Interrupted segment](../../results/2026-11-23-th14-full-policy/survival-65536-initial-segment.json), [recovered run snapshot](../../results/2026-11-23-th14-full-policy/survival-65536-retry-running.json)

The standalone 1.00b launch in an isolated Wine prefix logged `SteamAPI_Init() failed`. Both the playable original and multi-instance trial remain in the benchmark. The trial result does not measure Steam full-game transfer. The full-horizon experiment will require a separate task configuration. The [offline policy drift audit](261123-th14-policy-drift.md) documents why the short baseline cannot train Stage 1 clears.

## Reproduce the two-worker gate

Install PyTorch, Gymnasium, Stable Baselines3 and `requirements-rollout.txt` in the learner and in each actor environment. Build the [verified trial worker](../../tools/rollout-worker-probe/README.md) for each host. Start the learner with a private endpoint and use SSH forwarding for remote actors.

```sh
python -m scripts.train_distributed_native_ppo learner --run th14-demo --workers actor-1 actor-2 --horizon 3300 --rounds 4 --out runs/trial-demo
python -m scripts.train_distributed_native_ppo actor --run th14-demo --worker actor-1 --horizon 3300 --seed 101 --native ./launch-native-worker
python -m scripts.train_distributed_native_ppo actor --run th14-demo --worker actor-2 --horizon 3300 --seed 102 --native ./launch-native-worker
```

Each actor command needs its own instance-scoped launcher. The processes must use the same run name and horizon. Pass `--resume` and a fresh output directory to continue. Worker quotas can change while minibatch settings stay compatible. An independent `evaluate` invocation can compare `--policy initial=random` with `--policy candidate=PATH` using the same task configuration and a separate native worker. The public runner now scopes each routing ID to its run and logs missing actor names at startup. The benchmark jobs used the earlier worker-only routing. Two stale sessions briefly delayed the scale launch, so the stale sessions were closed and the affected actors reconnected before rollout.
