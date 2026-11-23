# Touhou Training

Open research for training agents across the original Touhou Project games. The project is building a reproducible pipeline from expert replays to policy imitation, dynamics prediction, and native reinforcement learning. The long-term benchmark is one-credit clears across titles and difficulties, including Lunatic.

## Current result

Two TH14 1.00b Normal Reimu A story trials cleared all six stages with zero continues. The first scored 296,778,700 with six misses and 23 bombs. Replacing the base SFT checkpoint with the expanded, title-balanced policy scored 253,575,700 with five misses and 22 bombs. Both runs used geometric dodging, learned boss aim, and emergency bombing. The world model scored forecasts in shadow mode. Separate native PPO comparisons measured Stage 1 performance.

The fixed-speed Hard pilot with the expanded checkpoint cleared four stages and ended in Stage 5 at frame 16,912 with zero continues. The speed-calibration follow-up ended in Stage 4. Fixed speed remains the default.

- [Initial benchmark and devlog](docs/devlog/261123-initial-benchmark.md)
- [Late-stage practice and title transfer](docs/devlog/261123-transfer-and-late-stages.md)
- [Overnight training recipes](docs/devlog/261123-overnight-recipes.md)
- [Native environment, models, and evaluation design](docs/design.md)
- [Training logs and reproduction commands](docs/experiments.md)
- [Machine-readable results](results/2026-11-23.json)
- [Overnight machine-readable results](results/2026-11-23-overnight.json)

## Set up

Native capture and play require Windows and a legally obtained compatible copy of the game. For the story benchmark use TH14 1.00b in a 640×480 window. Start a Normal story run with Reimu A before launching the controller. Clone [ParaKit](https://github.com/Guy-L/parakit) beside this checkout. Its offsets are loaded at runtime. The experiments began with ParaKit revision `2f2db40`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

Offline SFT and world-model training run independently of the original game once replay state pairs have been captured. On a Linux worker install PyTorch compatible with the worker’s CUDA runtime, NumPy, Gymnasium and Stable Baselines3. The Windows capture dependencies apply only to native tools.

## Prepare demonstrator data

Set `TOUHOU_REPLAY_DIR` to your local data directory. The seed script reads `royalflare-index.html` saved from the [TH14 replay index](https://maribelhearn.com/royalflare/th14). The expanded corpus scripts verify replay bytes and construct per-stage action streams. Keep the corpus metadata and action caches alongside the raw replays. For a single recorded replay already selected in the game’s menu, capture with the corresponding manifest and `--start-selected`.

```powershell
$env:TOUHOU_REPLAY_DIR = Join-Path $PWD 'replays/data'
python -m replays.build_corpus
python -m replays.build_multigame_corpus
python -m replays.expand_corpus --target 2000
python -m replays.collect_stage <replay-file>.rpy --manifest replays/data/expanded-corpus.json --all-stages --start-selected
```

The repository includes the acquisition and alignment code. Local game files, downloaded replay files, paired states, model checkpoints, and raw logs are stored outside Git. The benchmark release includes aggregated metrics and protocol details.

## Credits

Based on [Tianxiu (Tyson) Zhou’s `th14-rl`](https://github.com/choutianxius/th14-rl) at `6de2bc04d2cef164e70e0c9e75e2517f9d337c1e`. The original MIT license and Git history are retained. Native structured observations use offsets from [Guy-L’s ParaKit](https://github.com/Guy-L/parakit). The replay corpus uses the Royalflare mirror maintained by [Maribel Hearn](https://maribelhearn.com/royalflare/).
