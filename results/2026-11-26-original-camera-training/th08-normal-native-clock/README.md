# TH08 0.02 native decision clock

The stock Normal Ghost trial on Accelerator advanced **5.187–5.209 original ticks per applied action** in a 135-action engineering measurement of the earlier wall-timed runner. The measured median action interval was 99.72 ms. Each timing bound uses a native memory read bracketed by two Linux monotonic timestamps. In the recorded game, the two engineering X presses reached the game's consumed input word at ticks 226 and 564. The HUD changed from three Bombs to two, then one, with two spare lives throughout. The native input, original recording buffer and physical keymap agreed. [Measurements](validated-live-clock.json) · [Original recording, inputs and observer source](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th08-normal-native-clock-one-stock-native-clock.zip)

| Original engineering credit | Decisions | Observed native timing | Endpoint |
|---|---:|---|---|
| Alternating focus input used to identify each boundary | 135 | 134/134 successive input changes exactly four ticks apart | Cap with 2 spares, 1 Bomb. The final hold lasted three ticks before release. |
| Unmodified human SFT, four sampled heads | 131 | 524/524 target native input slots matched. All 49 distinguishable edges and all 35 adjacent distinguishable four-tick intervals matched. | First spare lost, 1 spare and 3 Bombs remaining. |

The unmodified run had one camera gate late by one native tick and one late X11 application on a repeated input. The application interval ranged from 41.46 to 100.97 ms with a median of 83.44 ms. Identical neighboring commands cannot yield two distinguishable transitions in the original recording. Each command still has an X11 receipt and four individually checked original input words. Both short credits ran entirely in Stage 1. [Marked timing and video](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th08-normal-native-clock-tick4-qualification-original-evidence.zip) · [Unmodified SFT timing and video](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th08-normal-native-clock-tick4-unforced-original-evidence.zip)

## Running the native scheduler

Use a prepared local Linux TH08 **0.02** study with a previously checked original title, difficulty and team menu, original trial files, the existing FIFO camera runner, the preconfigured Wine and CPU container images, and its local inference dependencies. The source study's four-head policy is retained by default. Supplying a different `--policy` binds that TorchScript actor into the new study. Installation verifies the 0.02 executable digest, compiles the read-only clock observer and leaves runtime hashes in `app/tick4-deployment.sha256`.

```bash
python3 -m scripts.prepare_th08_native_tick4 \
    --source-study "$QUALIFIED_002_STUDY" --destination "$NEW_STUDY" \
    --wine-image "$WINE_IMAGE" --history-states 1
bash "$NEW_STUDY/app/credit-tick4.sh" "$SEED" Normal Ghost engineering-zero-shot-original 135
python3 -m scripts.audit_th08_live_tick4 --raw "$NEW_STUDY/games/$SEED" --out "$AUDIT_DIR"
```

Omit the final cap argument to continue to the game's original endpoint. The optional cap accepts the engineering role only. Each stage has a separate original tick index and native recording buffer. Run qualification uses the returned stage, tick, stock and original terminal screenshots. Each CSV row includes the requested X11 mask, the four native input slots, current and next stocks, and whether this boundary is individually observable. Whole-game eligibility requires independent native terminal and coverage checks.
