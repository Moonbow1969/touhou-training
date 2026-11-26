# Two TH13 trial workers

The private containers use the project's existing Wine, Xvfb, PulseAudio and software-rendering trial image. Each has an independent virtual display, prefix, keyboard and capture directory. The program reads pixels through X11 and applies keyboard events through XTest. Both the game directory and the collector binary are mounted read-only.

On Endeavour, download the public trial installer linked from the developer's release page, verify its size and digest, and extract the Inno Setup application directory. The measured installer was 178,328,922 bytes with SHA-256 c7a26c96e54f62711e7dc75ed82be7269a627ce1fa904a8cce4084c44995c942. The executable digest is checked at launch. [Developer release announcement](https://kourindou.exblog.jp/14614181/) · [Developer download page](https://www16.big.or.jp/~zun/html/th13dl.html)

Run this on the Linux host, with the capture directory available to the SSH account.

```sh
sh tools/rollout-worker-probe/th13-pixels/launch.sh /path/to/extracted/app /tmp/th13-run toho-trial:endeavor-native-v6 th13-probe-
```

Confirm the saved title, difficulty, character and Stage 1 screens for both displays. The pilot used the default Normal rank and Hakurei Reimu. The script starts each lane from the game's pause screen. On the controller machine, install NumPy and Pillow and run the collector with a previously authenticated SSH host.

```sh
python tools/rollout-worker-probe/th13-pixels/collect.py --ssh endeavor --remote-root /tmp/th13-run --container-prefix th13-probe- --output pixel-episodes
```

The input vocabulary is horizontal left, neutral and right with independently balanced shooting. The file stores an additional zero vertical and focus column for shared tooling. Each episode saves 81 grayscale observations and 80 actions, monotonic capture timestamps, applied-event synchronization timestamps, independent seed, split, and visually checked beginning and ending stocks. Episode numbers 1–8 are training, 9–10 development, and 11–12 final test per lane. Simultaneous capture does not imply an identical game RNG or synchronized native ticks.

The HUD gate is specific to the original 640×480 trial client at root display coordinates +3,+22 on a 1024×768 private Xvfb. It checks the single zero score glyph, two spare lives, two Bombs, and the dark opening background. A source-visible screenshot from a qualified credit is retained at each endpoint. The tracker leaves a timed-out credit paused for native Retry. After the run, stop only the two named containers if the pool has no additional work.

## Online horizontal policy

The online actor samples left, neutral or right every 340 milliseconds of requested wall time and holds Shoot. Its input consists of the current 96×112 grayscale observation and its predecessor. Both become a 6×7 grid, first of mean brightness and then of mean absolute per-pixel change. A bias and the 84 grid features feed three linear logits. Input sampling, action sampling and image capture occur inside the lane. A separate host process checks the probabilities and log probabilities against the saved frames and frozen 3×85 weights. Final test credits use the same stochastic categorical decoder as training.

Compile the actor in the Linux host directory, targeting the Wine image's glibc, and copy the executable to each previously started private container. The example prefix identifies only these two workers.

```sh
gcc -std=c99 -O2 -Wall -Wextra -Werror tools/rollout-worker-probe/th13-pixels/pixel-policy.c -o /tmp/th13-pixel-policy -l:libX11.so.6 -l:libXtst.so.6 -lm
for lane in a b; do docker cp /tmp/th13-pixel-policy "th13-probe-$lane:/tmp/pixel-policy"; done
```

Initialize the pilot's direct stochastic actor and collect a two-credit block. Each worker must be on its Stage 1 pause screen. The game RNG is uncontrolled. The policy-credit runner uses fresh, unique controller seeds for every round and lane.

```sh
python -c "import numpy as n; w=n.zeros((3,85)); w[:,0]=n.log([.15,.70,.15]); n.savez('initial.npz',weights=w)"
python tools/rollout-worker-probe/th13-pixels/policy_credit.py --ssh endeavor --known-hosts /path/to/pinned-hosts --prefix th13-probe- --remote-root /tmp/th13-run --output pg --round 1 --model-a initial.npz --model-b initial.npz
```

Record each credit's independently verified life return in its own `label.json`. Use `status` equal to `active`, `resolved_trance`, `game_over`, or `censored`, with a numeric `reward` or null and a short `note` naming the images supporting it. Active return is visible spares at the cutoff minus starting spares. A confirmed Game Over uses −3 from the standard starting credit. Where automatic death trance is already active at the cutoff, only its validated debit counts. A fresh contact during follow-up belongs outside the horizon. The runner saves the horizon and 3, 7 and 12 second follow-ups for this adjudication.

The update sums score functions within each complete credit and averages across credits, using other credits' mean return as a baseline, a gradient norm bound of 2 and empirical forward KL bound of 0.01. A constant-return batch has exactly zero gradient. Pass only independently classified training credits collected under the checkpoint provided to `--initial`.

```sh
python tools/rollout-worker-probe/th13-pixels/update_policy.py --initial initial.npz --credit pg/pg-01-a --credit pg/pg-01-b --credit pg/pg-02-a --credit pg/pg-02-b --output pg/update1.npz
```
