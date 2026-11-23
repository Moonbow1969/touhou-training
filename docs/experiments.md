# Recording experiments

Run `scripts.record_run` for research training, replay capture, and native evaluation. Each invocation creates a unique directory and records the command, environment, source commit and working-tree status, input and artifact hashes, wall time, exit code, and combined console stream. With uncommitted changes it also saves the working Python sources. Paths supplied to the recorder are relative to the shell’s working directory. A fresh run directory allows repeats and concurrent jobs without replacing earlier logs.

Keep the run directory with the corresponding artifacts. For a remote worker, run the recorder on that worker or preserve its training console and environment along with the locally captured launcher log. Collect results and logs before cleaning temporary remote storage. Choose stable inputs while a job is running. Record the game build and any prepared practice state in `--note` for a native job.

```powershell
python -m scripts.record_run --name sft-demo --kind sft --input replays/data/paired-ringfix --artifact runs/sft-demo-model --note 'Four-frame decisions' -- python -m scripts.pretrain_policy --data replays/data/paired-ringfix --out runs/sft-demo-model --epochs 20 --hidden 512 --batch-size 256 --motion-history --soft-movement
```

```powershell
python -m scripts.record_run --name world-demo --kind world --input replays/data/paired-ringfix --artifact runs/world-demo-model -- python -m scripts.train_world --data replays/data/paired-ringfix --out runs/world-demo-model --epochs 20 --width 96
```

```powershell
python -m scripts.record_run --name ppo-dodge-demo --kind rl --input runs/sft-demo-model --input runs/dodge-imitation --artifact runs/dodge-ppo --note 'TH14 1.00b, Normal, Reimu A, Stage 1' -- python -m scripts.train_stage_dodge_ppo --base runs/sft-demo-model/policy.pt --head runs/dodge-imitation/head.pt --out runs/dodge-ppo --steps 1024
```

SFT and world-model jobs write `epochs.jsonl` after every epoch. PPO writes Stable Baselines3 `progress.csv`, intermediate training progress and checkpoint files, and JSON evaluation results. The recorder retains stdout and stderr even when the child exits with an error. A run killed by a process or machine failure may keep its last recorded `starting` or `running` status and a partial log. Preserve that evidence and start a fresh run on retry.

Use `replays.snapshot` to freeze existing and newly aligned shard directories into a new versioned directory. It checks sample counts, duplicate replay stages, and player split consistency. The adjacent manifest hashes the copied data. Keep world-model snapshots under the same replay data directory as the source action caches.

During a native replay capture, `--progress-every` controls interim logging of the current stage frame and sampled states. A stage enters the paired dataset only after it meets the alignment and coverage checks. When continuing a replay from a later selected stage, use `--all-stages --from-stage` with that stage number.

For a TH10 replay format check, run `python -m replays.inspect_rpy --title 10 <replay-file>.rpy`. The decoder was checked on six local original-game replays and six fixture replays. The separate `scripts.probe_th10` command reads the supported Steam build during its native demo. It records player position, active bullet count and the native replay frame counter. These are runtime observations and action-stream checks. Replay-state alignment is still pending for this title. [TH10 replay format reference](https://github.com/Fluorohydride/threp/blob/master/th10.cpp) and [game layout reference](https://github.com/N0zoM1z0/th10).

Use `replays.action_prior` to train on decoded button sequences and replay metadata. Its inputs can be packed directly from the action caches with `python -m replays.action_prior pack --corpus replays/data/expanded-corpus.json --out runs/action-tokens`. Train with `python -m replays.action_prior train --data runs/action-tokens --out runs/action-prior --epochs 10`. The trainer records each epoch and selects by validation direction-change accuracy.

For native PPO, supply `--analytic-control` to compare the frozen imitation head, the trained PPO head, and rule-based selection. `--body-risk` and `--laser-risk` apply consistently to all arms. Use `--resume-ppo` with a saved training checkpoint to continue a run after an interruption. `--initial-temperature` applies only at initialization and rescales the final actor layer for exploration.

Use `--evaluation-aim` to apply geometric aiming to each arm after training, and `--evaluation-reference` to include a previous PPO checkpoint. The selected rank of the foreground original game must match `--difficulty` if it is changed from Normal.

For SFT use `--patience 6` with a maximum epoch budget to stop after six rounds without an improvement in validation loss. The policy checkpoint is always the selected validation epoch. Preserve replay and player splits while fitting and selecting weights.

For a corpus expansion from an existing SFT checkpoint, use `--resume` with the same observation and network configuration. `--learning-rate 0.00004` sets the warm-start step size. Add `--retain-resume` to include epoch zero in checkpoint selection. `--anchor-movement 4` adds a KL loss against the frozen initial movement distribution during warm-start training.

The expanded SFT comparison kept four-frame decision targets, recent motion history, and stage-clock context. It used a pooled projectile encoder, hidden width 512, batch size 256, and seeds 61, 97, and 137. The 20-epoch matched comparison leaves early stopping disabled, while the follow-up stopping experiment uses the same seed and switches on `--patience 6`. Set `--title-balance 0.5` to reproduce the inverse-square-root title weighting.

```powershell
python -m scripts.pretrain_policy --data replays/data/paired-expanded --out runs/sft-balanced-61 --seed 61 --epochs 20 --hidden 512 --batch-size 256 --motion-history --stage-clock --soft-movement --title-balance 0.5
```

To measure a short-horizon world checkpoint on native rollouts, supply `--world-shadow` to `scripts.evaluate_boss_aim`. The report logs coarse occupancy AP near the actual future player for exact four-frame transitions. Keep the world checkpoint’s metrics sidecar alongside the weights.

World-model training can use `--patience 5` with a maximum epoch budget. With this flag it selects the best mean validation near-player occupancy AP across forecast horizons and records the per-horizon scores in `epochs.jsonl`. It reports the selected epoch and the actual epochs completed. With patience disabled the terminal epoch supplies the checkpoint.

To reproduce the Stage 4 diagnostic, select TH14 Normal Reimu A Stage 4 in Practice Start and launch the recorded evaluator. `--prepared` adopts the active stage on its first trial, so the report includes the actual starting frame.

```powershell
python -m scripts.record_run --name th14-stage4-practice --kind evaluation --input models-local --artifact runs/stage4-practice.json --note 'TH14 1.00b, Normal, Reimu A, Stage 4 Practice' -- python -m scripts.evaluate_boss_aim --base models-local/sft/policy.pt --ppo models-local/ppo/ppo.zip --student models-local/aim/aim.pt --first-stage 4 --last-stage 4 --prepared --stage-practice --candidate-only --blocks 1 --body-risk --laser-risk --emergency-bomb --out runs/stage4-practice.json
```

For a single-credit story test, select Start and use Reimu A with the same rank as the command. The native evaluator restarts once before recording, then plays the six-stage story without issuing a Continue. It records the native screen and continue count, with an immediate update at stage transitions. The story run used the frozen motion-history SFT base, the masked aiming checkpoint, and the four-frame world checkpoint for prediction telemetry.

```powershell
python -m scripts.record_run --name th14-story --kind evaluation --input models-local --artifact runs/th14-story.json --note 'TH14 1.00b, Normal, Reimu A, six-stage story' -- python -m scripts.evaluate_boss_aim --base models-local/sft/policy.pt --analytic-dodge --student models-local/aim/aim.pt --body-risk --laser-risk --emergency-bomb --world-shadow models-local/world/world.pt --max-misses 12 --last-stage 6 --blocks 1 --limit 30000 --candidate-only --out runs/th14-story.json
```

On a shared Windows desktop, enable `--focus-wait` for native TH14 trials. The controller preserves a paused process while the game is in the background. Foreground input remains exclusive to the active game.

The experimental `--adaptive-motion` option estimates player speed from prior exact four-frame moves. A run using the flag reports how often it applied the slow prediction and switched speed estimates. Keep the flag disabled when reproducing the original one-credit controls.

For native transfer, enter Normal with Reimu and launch the evaluator as Stage 1 begins. The verified TH17 pilot used Reimu Wolf. TH15 support expects Legacy Mode and still needs a successful native trial. `--wait-start` lets the evaluator load the checkpoint before the stage starts. Provide the checkpoint and its metrics sidecar to verify the training and selection titles. Run the same procedure without `--base` for stationary shooting.

```powershell
python -m scripts.record_run --name th17-zero-shot-sft --kind evaluation --input models-local/sft --artifact runs/th17-zero-shot-sft.json -- python -m scripts.evaluate_title_transfer --title 17 --base models-local/sft/policy.pt --wait-start 45 --out runs/th17-zero-shot-sft.json
```

Use `--analytic-dodge --body-risk` for geometric selection, or `--ppo` with a frozen dodge checkpoint plus `--body-risk` for the learned selector. Keep the first-miss limit, native frame allowance, shot, difficulty, and decision cutoff identical across trials.

For a leave-one-title-out replay test, keep the original player splits and prepare a training shard directory with only the selected training titles. Leave the action caches alongside its parent data directory as usual. Train with the same architecture, then evaluate the resulting checkpoint on the reserved title using `scripts.evaluate_replay_policies --holdout-title 17`. The evaluator checks the candidate’s training and validation manifests. Record which checkpoint used the title at training time for any reference comparison.

For source and data changes, add a dated devlog entry with the hypothesis, affected code or data, exact comparison, result, and decision. Include failed attempts and specify when comparison groups are missing. Refer to run IDs or checkpoint hashes. Before publishing a report, review raw logs and manifests for local paths, account details, and third-party data. The public record contains reviewed aggregates and source code. Local raw evidence includes full command lines and tracebacks.
