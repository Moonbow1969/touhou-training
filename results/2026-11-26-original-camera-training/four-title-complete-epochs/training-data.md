# Reproducing the four-title camera training run

Extract the archives under the same directory. The resulting `bundle` contains the 74 training stages, 18 development stages, original stage metadata, the registered initialization and the protocol. The trainer verifies both data and metadata hashes before fitting.

```sh
python -m scripts.train_camera_full_epochs --bundle bundle --out fit
```

| Inputs | Bytes | SHA-256 |
|---|---:|---|
| [train-th10](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-title-complete-epochs-train-th10.zip) | 412,783,137 | `cff625e372cd63499092671660d6930aad3377546c9e3920a3d68dff6da7a36a` |
| [train-th12](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-title-complete-epochs-train-th12.zip) | 561,021,472 | `9ad9666d3439fd2e9aa6a7c7cdec7b888cca9f07a5558effbb5ce480117799d3` |
| [train-th13](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-title-complete-epochs-train-th13.zip) | 488,036,544 | `e4170917598db359c3ace20712fcce61dd59972a9ad22b57e86f9d1a79cde715` |
| [train-th17](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-title-complete-epochs-train-th17.zip) | 681,037,051 | `0d7a70e60902b8e6dedf2a82f421f2150bbdce67765bd27e964b9ac2d85dc4b8` |
| [development-th10](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-title-complete-epochs-development-th10.zip) | 58,543,727 | `33a6806c1a1b81210604c1c438909124df53e4a189099864b183303edd719c2c` |
| [development-th12](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-title-complete-epochs-development-th12.zip) | 81,129,924 | `41988d65de9e3f0559681561794797b56b23449ac1742ca56c50a29be0fcc29c` |
| [development-th13](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-title-complete-epochs-development-th13.zip) | 88,703,487 | `e7074906af16cdf3353e19c74f51bb3f08b51744d627624f5384d99b586e8644` |
| [development-th17](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-title-complete-epochs-development-th17.zip) | 203,135,728 | `503e65bbb16a9c8b5ddcb0cc8ff9c37a2a6c33392ac0aca229abf095997743b4` |
| [encoder-and-registration](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-title-complete-epochs-encoder-and-registration.zip) | 710,407 | `35c99802b6496596a8a45627b563a97f34a8ecb600ef0a6c746183387e76dadc` |

The [exact Python sources](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-title-complete-epochs-exact-learner-code.zip) contain the fifteen modules from the registered learner, preserving their original bytes. The protocol records the SHA-256 of each module. Extract them alongside `bundle` before invoking the trainer.
