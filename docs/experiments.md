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

To reproduce the Stage 4 diagnostic, select TH14 Normal Reimu A Stage 4 in Practice Start and launch the recorded evaluator. `--prepared` adopts the active stage on its first trial, so the report includes the actual starting frame.

```powershell
python -m scripts.record_run --name th14-stage4-practice --kind evaluation --input models-local --artifact runs/stage4-practice.json --note 'TH14 1.00b, Normal, Reimu A, Stage 4 Practice' -- python -m scripts.evaluate_boss_aim --base models-local/sft/policy.pt --ppo models-local/ppo/ppo.zip --student models-local/aim/aim.pt --first-stage 4 --last-stage 4 --prepared --stage-practice --candidate-only --blocks 1 --body-risk --laser-risk --emergency-bomb --out runs/stage4-practice.json
```

On a shared Windows desktop, enable `--focus-wait` for native TH14 trials. The controller preserves a paused process while the game is in the background. Foreground input remains exclusive to the active game.

For native transfer, enter Normal with Reimu and launch the evaluator as Stage 1 begins. The verified TH17 pilot used Reimu Wolf. TH15 support expects Legacy Mode and still needs a successful native trial. `--wait-start` lets the evaluator load the checkpoint before the stage starts. Provide the checkpoint and its metrics sidecar to verify the training and selection titles. Run the same procedure without `--base` for stationary shooting.

```powershell
python -m scripts.record_run --name th17-zero-shot-sft --kind evaluation --input models-local/sft --artifact runs/th17-zero-shot-sft.json -- python -m scripts.evaluate_title_transfer --title 17 --base models-local/sft/policy.pt --wait-start 45 --out runs/th17-zero-shot-sft.json
```

Use `--analytic-dodge --body-risk` for geometric selection, or `--ppo` with a frozen dodge checkpoint plus `--body-risk` for the learned selector. Keep the first-miss limit, native frame allowance, shot, difficulty, and decision cutoff identical across trials.

For a leave-one-title-out replay test, keep the original player splits and prepare a training shard directory with only the selected training titles. Leave the action caches alongside its parent data directory as usual. Train with the same architecture, then evaluate the resulting checkpoint on the reserved title using `scripts.evaluate_replay_policies --holdout-title 17`. The evaluator checks the candidate’s training and validation manifests. Record which checkpoint used the title at training time for any reference comparison.

For source and data changes, add a dated devlog entry with the hypothesis, affected code or data, exact comparison, result, and decision. Include failed attempts and specify when comparison groups are missing. Refer to run IDs or checkpoint hashes. Before publishing a report, review raw logs and manifests for local paths, account details, and third-party data. The public record contains reviewed aggregates and source code. Local raw evidence includes full command lines and tracebacks.
