# TH13 original full-credit adapter

This worker runs the developer's Trial 0.01a executable in an isolated Wine/Xvfb display. All recorded benchmark credits start on Normal with Reimu, two spare lives and two Bombs. The actor receives a grayscale playfield and uses the same portable human-replay SFT or PPO weights as the other original-game workers. One process controls one display. Use a unique capture directory, container name and Wine prefix per instance.

The movement head samples nine positions in a row-major 3×3 grid. Index 0 is up-left, 4 is neutral and 8 is down-right. Three independent binary heads control focus, shoot and Bomb. Simultaneous keys go through ordinary X11 keyboard events. TH13's manual Trance button is outside this shared vocabulary. The actor releases all keys during terminal transitions and records spare-life and Bomb changes through automatic death trance.

## Launch and capture

Use the extracted official trial, the private Wine image described in the adjacent TH13 pixel worker, GCC, MinGW and Python with NumPy. The expected SHA-256 for `th13.exe` is `09818929cceea76ff4c65630e649bbccdb85d92b2e28bb4dfe6526749e6f4fb6`. The launcher checks it and starts the original at the Stage 1 pause menu. The example below allocates one private lane. Run it from the repository root on Linux.

```sh
run_dir=$(mktemp -d)
lane=th13-eval-a
TH13_LANES=a sh tools/rollout-worker-probe/th13-pixels/launch.sh   /path/to/extracted/app "$run_dir" toho-trial:endeavor-native-v6 th13-eval-
src=tools/rollout-worker-probe/th13-full
mkdir -p "$run_dir/a/lib"
gcc -std=c11 -O2 -Itools/full_action -Itools/rollout-worker-probe/th13-pixels   "$src/full-trial-actor.c" -o "$run_dir/a/lib/actor" -l:libX11.so.6 -l:libXtst.so.6 -lm
cp "$src/read-th13-trial.c" "$src/collect-credit.sh" "$run_dir/a/lib/"
cp /path/to/frozen/policy.bin "$run_dir/a/lib/policy.bin"
docker exec "$lane" i686-w64-mingw32-gcc-posix -O2 /capture/lib/read-th13-trial.c -o /capture/lib/observe.exe
docker exec -d "$lane" sh -c 'wine /capture/lib/observe.exe > /capture/observer.log 2>&1'
docker exec -e DISPLAY=:66 "$lane" sh /capture/lib/collect-credit.sh held-sft-731001 731001 /capture/lib/policy.bin
credit="$run_dir/a/held-sft-731001"
python "$src/pack_credit.py" "$credit" "$run_dir/a/lib/policy.bin" --role held --seed 731001   --out "$credit/rollout.npz" > "$credit/receipt.json"
python "$src/summarize.py" "$run_dir/a" "$run_dir/summary.json"
```

One policy step normally spans four observed native stage ticks. A 350 ms guard permits another decision when Bomb or trance animation stops the stage clock. Software rendering and CPU contention affect elapsed wall time. The playfield crop is 384×448 at pixel (35,38) of the private root display. A 4×4 average produces 96×112 grayscale samples in width-by-height order. The actor keeps action history across life losses.

The trial exposes Stages 1–3. This cohort reached an original game-over through exhaustion of its initial credit. A negative spare-life count denotes zero remaining active lives. `last_playable_bombs` is the inventory immediately before the final debit. `remaining_bombs` is the terminal stock field, including stock observed during the ranking transition. Stage 2 arrival uses the first native transition into Stage 2, preserves the entry stock, and does not reset the actor. A nonzero Continue counter invalidates a credit. A timeout is an infrastructure failure.

## Trace and evidence

`rollout.npz` contains the policy features [T,110], four categorical/ordinal action labels and four behavior log-probabilities, scalar critic and reward, terminal markers, original control word, original resource counts and stage ticks. Reward components appear in this order: life losses, earned lives, stage advances, effective Bombs and action hold seconds. The scalar reward is `-loss + earned + 2*advance - .03*effective_bombs + .001*hold_seconds`. Keys sampled while Bomb is unavailable carry no resource penalty. The reported native-key match compares the observed original control word with the requested seven keys on each decision.

The optional passive player observer and camera pair are for offline visual supervision. For the verified build, the player object pointer is at module RVA `0xbcb20`; `+0x550` and `+0x554` are its x and y values. Bounds at `+0x5a0` and `+0x5ac` give an independent center check. The camera brackets every screenshot with two native reads and records their game tick and position. The benchmark actions receive no coordinates. The two diagnostic captures came from separate PPO1 training seeds, 730107 and 730108. Frozen benchmark seeds 731001–731012 are held out.

The current collector validates genuine stock exhaustion and ranking transition. A full retail-story ending requires separate executable calibration and ending recognition. The supplied trial and its assets are installed by the user.
