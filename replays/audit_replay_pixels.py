"""Validate causal alignment independently with original player displacement and camera template registration."""

from pathlib import Path
import argparse
import json
import numpy as np, cv2
from .inspect_rpy import action_streams


def fit(path, replay_dir, audit):
    meta = json.loads(path.with_suffix(".json").read_bytes())
    title = int(str(meta["title"]).upper().removeprefix("TH"))
    focus_bit = 4 if title == 10 else 8
    control_bits = 0xF3 | focus_bit
    with np.load(path) as z:
        f, raw, x, obs = z["native_frame"], z["native_input"], z["xy"], z["obs"]
        actions = z["action"]
        ix = f + meta["alignment_offset"]
        rpy = replay_dir / meta["name"]
        words = next(
            np.frombuffer(d, "<u2").reshape(-1, 3)[:, 0]
            for i, d in action_streams(rpy, title)
            if i == meta["stage"]
        )
        assert np.array_equal(
            z["input_chunk"], words[ix[:, None] + np.arange(1, 5)]
        ) and np.array_equal(
            z["past_input_chunk"], words[ix[:, None] + np.arange(-7, 1)]
        )
        assert np.all((raw & control_bits) == (words[ix] & control_bits))
        # Full pixel camera using physical centers, including the sprite and bottom of the hitbox.
        pos = x / 4 + np.array([48, 0])
        d = np.diff(x, axis=0)
        dt = np.diff(f)
        eligible = (
            (dt == 4)
            & (np.max(np.abs(d), axis=1) <= 20)
            & (np.max(np.abs(d), axis=1) > 0.1)
        )
        lags = {}
        for shift in range(-8, 9):
            j = ix[1:, None] + shift + np.arange(-3, 1)
            keep = eligible & np.all((j >= 0) & (j < len(words)), axis=1)
            q = words[j[keep]]
            dx = ((q & 128) > 0).astype(float) - ((q & 64) > 0)
            dy = ((q & 32) > 0).astype(float) - ((q & 16) > 0)
            gain = np.where((q & 8) > 0, 0.45, 1.0) / np.maximum(1, np.hypot(dx, dy))
            e = np.column_stack(((gain * dx).sum(1), (gain * dy).sum(1)))
            y = d[keep]
            s = float((e * y).sum() / np.square(e).sum())
            err = np.linalg.norm(s * e - y, axis=1)
            lags[shift] = dict(
                rows=len(y),
                mean_abs_displacement_error_native_px=float(err.mean()),
                fitted_unfocused_px_per_tick=s,
            )
        # Normalized template registration compares consecutive actual images. Stable good matches required.
        # Estimate offset of rendered sprite from native position independently of controller input.
        correct = []
        still = []
        match = []
        reg = []
        for i in np.flatnonzero(eligible):
            cx, cy = np.rint(pos[i]).astype(int)
            a, b = np.rint(pos[i + 1]).astype(int)
            if not (9 <= cx < 87 and 11 <= cy < 102 and 9 <= a < 87 and 11 <= b < 102):
                continue
            template = obs[i, cy - 4 : cy + 4, cx - 3 : cx + 4]
            space = obs[i + 1, b - 9 : b + 10, a - 9 : a + 10]
            if template.std() < 10:
                continue
            rr = cv2.matchTemplate(space, template, cv2.TM_CCOEFF_NORMED)
            _, v, _, (rx, ry) = cv2.minMaxLoc(rr)
            if v < 0.77:
                continue
            best = np.array([a - 9 + rx + 3, b - 9 + ry + 4])
            dv = best - np.array([cx, cy])
            if np.linalg.norm(pos[i + 1] - pos[i]) < 1:
                continue
            correct.append(np.linalg.norm(best - pos[i + 1]))
            still.append(np.linalg.norm(best - pos[i]))
            match.append(v)
            reg.append(float(np.dot(dv, (pos[i + 1] - pos[i])) > 0))
        # Twenty movement transitions before/after, conditioned on exact, genuine human changes.
        starts = (actions[:, 0] != 4) & (np.r_[4, actions[:-1, 0]] == 4)
        bomb_starts = ((words[1:] & 2) != 0) & ((words[:-1] & 2) == 0)
        ret = dict(
            schema_version=1,
            replay_sha256=meta["replay_sha256"],
            player=meta["player"],
            split=meta["split"],
            verified_future_and_past_chunks=True,
            all_rows_native_word_match=True,
            alignment_native_word_offset=meta["alignment_offset"],
            eligible_moving_displacement_rows=int(eligible.sum()),
            observed_motion_starts=int(starts.sum()),
            expert_bomb_true_onset_native_ticks=(
                np.flatnonzero(bomb_starts) + 1
            ).tolist(),
            covered_future_bomb_onsets=sum(
                bool(np.any((ix < b) & (ix + 4 >= b)))
                for b in np.flatnonzero(bomb_starts) + 1
            ),
            displacement_lag_test=lags,
            lowest_displacement_error_shift_ticks=min(
                lags, key=lambda i: lags[i]["mean_abs_displacement_error_native_px"]
            ),
            camera_registration=dict(
                accepted_movement_pairs=len(correct),
                median_match_correlation=float(np.median(match)),
                median_native_xy_error_camera_px=float(np.median(correct)),
                stationary_prediction_median_error_camera_px=float(np.median(still)),
                motion_direction_agrees=float(np.mean(reg)),
            ),
        )
        audit.mkdir(parents=True, exist_ok=True)
        (audit / (path.stem + ".json")).write_text(json.dumps(ret, indent=2) + "\n")
        print(
            path.stem,
            len(f),
            ret["lowest_displacement_error_shift_ticks"],
            ret["camera_registration"],
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for archive in args.archives:
        fit(archive, args.replay_dir, args.out)


if __name__ == "__main__":
    main()
