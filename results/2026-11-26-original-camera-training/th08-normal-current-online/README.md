# TH08 Normal Ghost original-game PPO pilot

Four fresh, original build 0.02 Normal Ghost credits on Entropy start with two spare lives and three Bombs. The model controls direction, Focus, shot and Bomb from the current camera and its local difference channel. The original Continue prompt ends the credit. Native stock changes supply rewards and the final loss has zero bootstrap. Bomb changes within six decisions of a life loss are excluded from the resource reward. The local learner replays every original camera and verifies the chosen-action log probabilities and zero starting critic.

| Independent training seed | Decisions | Reward | Final spare lives | Last playable Bombs |
|---|---:|---:|---:|---:|
| 80960101 | 370 | −3.00 | 0 | 3 |
| 80960102 | 600 | −3.06 | 0 | 1 |
| 80960103 | 623 | −3.03 | 0 | 3 |
| 80960104 | 771 | −3.03 | 0 | 3 |

The four-head update uses 2,364 decisions, two epochs and 50 accepted updates. Final joint KL against the original policy is 0.001967. Critic RMSE across the same training data moves from 0.4336 after epoch one to 0.4261. Four complete credits suffice to test the rollout and learner interface. A separate original-game comparison supplies any evidence of policy benefit.

The optimizer uses the files named by `qualified-train.json` relative to that manifest. Unpack the small release below before rerunning. The update expects the recorded [current-camera SFT parent](../th08-normal-temporal/change-free/policy.ts) and release original footage is available separately.

[Frozen training credits](qualified-train.json) · [Learner protocol](policy/protocol.json) · [Optimizer log and native re-forward](policy/report.json) · [PPO TorchScript](policy/policy.ts) · [Exact PPO input](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th08-normal-current-online-optimizer-input.zip) · [Four original training recordings](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th08-normal-current-online-four-on-policy-normal-originals.zip)
