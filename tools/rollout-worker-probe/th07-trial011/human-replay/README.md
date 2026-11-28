# TH07 trial 0.11 original human playback

The collector opens the original Replay menu, selects a registered source and stage and uses normal-speed playback. The original program supplies the player's inputs throughout each segment. Captures run in two isolated Xvfb/Wine instances on Momentum. A sensor reads the original replay clock, current and previous input words, player center and two native rectangles. Original playfield pixels and native positions are separated in the archive. The native values provide supervision and validation for a visual actor.

## Verify a published stage

Download the stage ZIP, original replay and archive SHA from the corresponding index entry. Python 3.11+, Pillow and NumPy suffice for offline verification in a checkout containing the project's replay decoder.

```
python tools/rollout-worker-probe/th07-trial011/human-replay/verify_dataset.py STAGE.zip --replay SOURCE.rpy --sha256 ARCHIVE_SHA256
```

The checker verifies every signed member, decodes the public source replay, compares both observed input words at each live clock, reconstructs all four action heads, and regenerates the grayscale policy camera from every available original color witness byte for byte. It consumes one archive at a time. The published Pykrete Reimu B Stage 1 archive was freshly downloaded and verified with its original source after upload.

## Observation and learning arrays

* `gray.raw.gz` is a consecutive array of `uint8[T,112,96]`. The sensor brackets the screenshot/position reads with the exact same native tick and both input words, dropping captures that cross the observed boundary. Only positive original ticks divisible by four are considered. Some nominal ticks are missed and the native tick vector preserves every gap.
* `positions.npz` contains observed native clock, input words, original and downsampled XY, stage-source words, validity masks and causal action indices. A training row at `j=causal_pair_rows[i]` uses `gray[j]` and, if required, the difference from `gray[j-1]`. Both native centers must separately match both body-rectangle midpoints to less than 0.02 native pixel, and the two images must be exactly four native ticks apart. ZIP rows retain rejected XY as raw evidence. Use the published indices for supervision.
* Direction, focus and fire use the first future input `source[t+1]`. Direction class is row-major from up-left=0 through neutral=4 to down-right=8. Focus uses 0x4 and shoot uses 0x1. Bomb is a rising edge of 0x2 anywhere in the next four source ticks. Original current and previous input use `source[t]` and `source[t-1]`. Native coordinates are playfield-local 384×448 with the full-color crop starting at (32,16).
* `full-color.bgra.gz` contains repeated uint32 little-endian original tick followed by 640×480 BGRA pixels. All eight archives contain native high-resolution witnesses at a sparse cadence and complete screenshots of their original stage selection and exit. Their full policy-resolution images are stored at every retained capture.
* Sprite-template registration is a diagnostic. For new shots the bank comes from that shot's Stage 1 TRAIN pixels through tick 3600. Its Stage 1 error excludes that initial interval. Stage 2 uses the unchanged bank. For the existing Reimu A source pair the bank is built from Boredperson Stage 2, and zylo stays held out. The search receives no target coordinates.

## Reproduce live collection on the recorded setup

The original game image, authorized trial installation, private source paths and Wine/Docker launcher must be available on the host. The runner expects the private base declared at the top of the scripts. Two concurrent replay processes were measured. Each instance uses up to two CPU cores and no GPU. No game files are included in the stage archives.

Build on a system compatible with the image's glibc. Put `native_input.c` from the parent directory on the C include path. `sensor.c` records the Stage 2 Reimu A high-resolution cadence. `sensor_shots.c` retains a sparser color cadence for the other shots. Both collect every successful native four-tick grayscale sample.

```
cc -O2 -I.. sensor_shots.c -ldl -lm -o sensor
python3 capture_shots.py 0
python3 capture_shots.py 1
python3 verify_shots.py /var/tmp/moonbow-th07-trial-261127/human-normal-shots/pykrete-reimub/s2
```

The original selector uses Down before confirming the replay stage with Z. A rejected engineering attempt applied Down on the later playback-mode screen and loaded Stage 1. It was excluded. Source input words distinguish the selected stage independently from its menu screenshot.

Team Shanghai Alice retains rights to the game. Replay input attribution and original download URLs are in the index. Published observation archives contain gameplay captures and derived numeric measurements. The code license covers the collection and checking code.
