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
