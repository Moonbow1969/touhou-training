# PPO initialization on stock TH14

The existing stock-game evaluations compare frozen SFT with PPO initialized from that same SFT. The random-initialization distributed pilot uses the separate 0.01b trial and a different observation contract. Its curve belongs to the trial experiment. [Stock PPO record](../sft-to-ppo-pipeline.md), [trial pilot](261123-native-ppo-scale.md)

## Matched pilot

- Use stock Steam TH14 1.00b Normal Reimu A Stage 1. Stop at the first miss, native clear, or 3,300 decisions. Hold each action for four nominal native ticks.
- Use the same width-512 actor and critic architecture, motion history, stage clock, action heads, survival-time reward, PPO hyperparameters, training seed 211 and CPU learner. Initialize the learned arm from the existing SFT encoder and actor with a fresh critic. Initialize all trainable parameters in the random arm with the native PPO constructor. The SFT reference supplies only the observation transformation and architecture for that arm.
- Measure at 0, 8,192 and 16,384 online decisions. Use the archived verified SFT and warm-start checkpoint lineage. Train the random arm in two 8,192-decision segments and preserve its exact initial actor, both endpoint archives, and the per-update logs.
- Re-evaluate all six frozen policies with eight common sampling seeds, 901–908, under a rotated arm order. Disable geometry, aim correction and automatic Bomb in every trial. The sampling seeds control the policy draw. The original game's RNG remains unmanaged.
- Use median furthest Stage 1 frame as the principal outcome. Report change from each arm's own step-0 median and the change per 1,000 decisions. Report per-seed differences, clears, and Bomb use. Keep on-policy training episodes and frozen-policy evaluation in separate figures.

The pilot tests one training seed and one PPO recipe on one original stage. The decision budgets count online interactions. The SFT arm also used offline demonstrations before PPO. Evaluation seeds measure rollout variation for the frozen checkpoints. Additional training seeds are needed to measure variation from retraining.

## Figures and source tables

The archive separates the simultaneous PPO training trace from the frozen evaluations. Each panel in the archived evaluation figure has its own matched sampling-seed cohort. The matched initialization figure uses a new common cohort across all budgets.

The figure directory keeps PNG previews, editable SVG, and the per-episode and per-update CSV inputs. Run the plotting script from the repository root with Matplotlib installed.

```text
python -m scripts.plot_th14_ppo_curves
```

[Archived figures and CSV data](../../results/2026-11-23-th14-full-policy/figures), [plotting code](../../scripts/plot_th14_ppo_curves.py)

The random arm completed both training segments. Its two reports include each PPO update, each completed training episode, checkpoint fingerprints and native timing. The combined on-policy figure shows the policies changing through training. The frozen-policy figure supplies the comparison at fixed budgets.

[Combined training rollout figure](../../results/2026-11-23-th14-full-policy/figures/ppo-matched-initialization-on-policy.svg), [first random training segment](../../results/2026-11-23-th14-full-policy/ppo-random-training-8192.json), [second random training segment](../../results/2026-11-23-th14-full-policy/ppo-random-training-16384.json)
