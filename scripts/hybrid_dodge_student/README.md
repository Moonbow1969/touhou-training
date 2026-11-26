# Privileged hybrid movement distillation

The policy consumes stock TH14 1.00b Normal Reimu A state. A frozen relative-clock student proposes movement, focus, Shoot and Bomb. A neural movement policy uses the proposal, all 96 projectile records through a learned encoder, the closest six raw projectiles, current context and stage. The actor emits four simultaneous controls. Geometric labels and demonstrated trajectories are used only during training. The native evaluator records supplementary body and laser sensors outside the policy input.

Run the commands from the repository root with its normal Python environment. The private replay, observer, model and game artifacts follow this repository's existing sibling work and outputs layout. Source file names and checksums appear in the aggregate results. Public Git contains neither game/replay content nor neural weights. Repository code carries its MIT license.

## Reproducing the offline policy

Point TH14_HYBRID_STUDENT_ARTIFACT_DIR to a fresh private directory when reproducing. Training, calibration and parity checks use separate entry points.

```powershell
$env:TH14_HYBRID_STUDENT_ARTIFACT_DIR = 'C:/private/experiments/hybrid-dodge-reproduction'
.venv/Scripts/python.exe -m scripts.hybrid_dodge_student.train
.venv/Scripts/python.exe -m scripts.hybrid_dodge_student.calibrate
.venv/Scripts/python.exe -m scripts.hybrid_dodge_student.evidence
.venv/Scripts/python.exe -m scripts.hybrid_dodge_student.evaluate --arm challenger --seed 4001 --verify-only
```

In a clean artifact directory the first and conservative checkpoints reproduced the original SHA-256 values fb13c2f48d7501752b6cfa1d2417a61d3f8fd775d50de392a027828c6f5100d4 and 3cdd6a1501196dec9a46f07c5d708a6c0130438c7a24ed99c2e67585ee68934b. The fixture and labels require the private native/source archives. The evaluator pins the checkpoint, actor code, shared modules and menu references. Standalone inference needs only the packed conservative checkpoint and the repository's ordinary observation interface. The evaluate command accepts --arm anchor to score the underlying frozen relative policy.

## Prospective joint-movement PPO

The following offline command tests the warm start and a clipped PPO update on synthetic inputs.

```powershell
.venv/Scripts/python.exe -m scripts.hybrid_dodge_student.ppo --seed 4101 --credits 2 --temperature .55 --verify-only
```

The following command is for an explicitly scheduled native development slot. It starts fresh, runs at most two full credits from Stage 1 to the Stage 3 entry objective, then performs one PPO update. The collector samples the learned joint movement/focus distribution. Original Shoot and Bomb stay greedy. A new value head starts at zero, and the optimizer starts without previous history. A native hit incurs −0.015, Stage 2 entry earns 0.5 plus 0.03 per spare life and 0.01 per Bomb, first vulnerable appearance of each Stage 2 spell earns 0.06, and Stage 3 entry earns 1.5 plus 0.10 per spare life and 0.02 per Bomb. The report separates these progress rewards from actual resource consumption.

```powershell
.venv/Scripts/python.exe -m scripts.hybrid_dodge_student.ppo --seed 4101 --credits 2 --temperature .55 --out ../../outputs/touhou-sft-world/th14-hybrid-joint-ppo-4101-4102-261125
```

The warm start agrees exactly with the conservative policy's movement logits at temperature 1. Greedy decisions are unchanged at .55. On old training credit 2202 the mean probability of the greedy joint action at .55 was 96.38% overall, 98.13% for quiet decisions and 65.95% for decisions whose chosen projected clearance was below four units. These estimates describe the archived states.

The collector writes native transition/sensor archives, per-action behavior log probabilities, observed rewards, value/advantage targets, model signatures, and stage entry/exit resources. The PPO update consumes only trajectories freshly sampled under this run's proposal. Both 3601 credits, 4001 and the historical development-check seeds are excluded from future PPO collection. Use new seeds for an independent evaluation.

```powershell
.venv/Scripts/python.exe -m scripts.hybrid_dodge_student.evaluate --arm challenger --seed 4201 --ppo-checkpoint ../../outputs/touhou-sft-world/th14-hybrid-joint-ppo-4101-4102-261125/candidate.pt --verify-only
.venv/Scripts/python.exe -m scripts.hybrid_dodge_student.evaluate --arm challenger --seed 4201 --ppo-checkpoint ../../outputs/touhou-sft-world/th14-hybrid-joint-ppo-4101-4102-261125/candidate.pt --out ../../outputs/touhou-sft-world/th14-hybrid-joint-ppo-4201-261125
```

For a separately designated supervised data-aggregation experiment, derive action-conditioned labels from the prospective credit archives. The command checks their pre-collection role, policy fingerprints, seed isolation, native Continue counter and original behavior buffer. The output stores the original executed movement separately from the retrospective teacher action and projected margin for every proposal.

```powershell
.venv/Scripts/python.exe -m scripts.hybrid_dodge_student.aggregate --development-report ../../outputs/touhou-sft-world/th14-hybrid-joint-ppo-4101-4102-261125/report.json --out ../../outputs/touhou-sft-world/th14-hybrid-labeled-development-4101-4102-261125
.venv/Scripts/python.exe -m scripts.hybrid_dodge_student.check_aggregation
```

The conservative actor and holdouts remain frozen. Inspect the ending result, stage resources, actual hit onsets and actual Bomb consumption to decide whether a PPO descendant advances. Teacher-projected hazard improvements alone cannot establish a native gain.
