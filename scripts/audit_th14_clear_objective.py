"""Check the proposed pure-policy clear objective against archived TH14 results."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class NativeState:
    stage: int
    screen: int
    pause: int
    lives: int
    continues: int
    practice_clear_verified: bool = False


def classify(goal, stage, state, training_limit_reached=False):
    if goal not in ("stage", "practice", "story"):
        raise ValueError("Unknown native clear goal.")
    if state.continues:
        raise ValueError("A continue invalidates a one-credit trajectory.")
    if goal == "story" and state.stage == 6 and state.screen == 15 and state.lives >= 0:
        return "clear"
    if state.screen != 7:
        raise ValueError("The native screen cannot be scored as gameplay.")
    if goal == "stage" and state.stage == stage + 1 and state.pause in (0, 2) and state.lives >= 0:
        return "clear"
    if (goal == "practice" and state.stage == stage and state.pause == 1
            and state.lives >= 0 and state.practice_clear_verified):
        return "clear"
    if state.stage != stage and not (goal == "story" and stage <= state.stage <= 6):
        raise ValueError("The native stage is outside the task.")
    if state.pause == 1 and state.lives < 0:
        return "game_over"
    if state.pause != 2 or state.lives < 0:
        raise ValueError("The native transition is unresolved.")
    return "time_limit" if training_limit_reached else "ongoing"


def score(outcome, previous_potential=0., next_potential=0.):
    """Return reward, terminated, truncated for the undiscounted clear task."""
    if outcome not in ("clear", "game_over", "time_limit", "ongoing"):
        raise ValueError("Only a validated native outcome can receive a reward.")
    terminated = outcome in ("clear", "game_over")
    shaping = (0. if terminated else next_potential) - previous_potential
    return float(outcome == "clear") + shaping, terminated, outcome == "time_limit"


def audit(results):
    stock = results / "2026-11-23-th14-full-policy"
    training = json.loads((stock / "stock-ppo-warm-65536-train.json").read_bytes())
    trials = training["training_episodes"]
    midboss_frame = 2422
    hybrid = json.loads((stock / "ppo-long-8192-sft-paired.json").read_bytes())["controllers"]["hybrid"]
    stage_clears = [
        classify("stage", 1, NativeState(row["native_stage"], 7, row["native_pause_state"],
                                        row["lives_remaining"], 0))
        for row in hybrid
    ]
    if not stage_clears or any(outcome != "clear" for outcome in stage_clears):
        raise ValueError("The known native Stage 1 advances did not pass the clear contract.")

    overnight = json.loads((results / "2026-11-23-overnight.json").read_bytes())["native_th14_story"]
    one_credit = NativeState(overnight["native_stage"], overnight["native_game_screen"], 0, 0,
                             overnight["native_continues"])
    if classify("story", 1, one_credit) != "clear":
        raise ValueError("A verified one-credit ending did not pass the clear contract.")
    video = json.loads((results / "2026-11-24-th14-normal-story-capture.json").read_bytes())
    continued = NativeState(6, video["native_ending_screen"], 0, 0, video["continues"])
    try:
        classify("story", 1, continued)
    except ValueError:
        pass
    else:
        raise ValueError("A continued ending was admitted as a one-credit clear.")

    alive_after_miss = NativeState(1, 7, 2, 0, 0)
    game_over = NativeState(1, 7, 1, -1, 0)
    practice_win = NativeState(1, 7, 1, 0, 0, practice_clear_verified=True)
    if classify("stage", 1, alive_after_miss) != "ongoing":
        raise ValueError("A recoverable miss became a terminal failure.")
    if classify("stage", 1, game_over) != "game_over" or classify("practice", 1, practice_win) != "clear":
        raise ValueError("Synthetic practice or Game Over boundaries failed their contract.")
    at_limit = classify("stage", 1, alive_after_miss, training_limit_reached=True)
    if score(at_limit) != (0., False, True) or score("clear") != (1., True, False):
        raise ValueError("A collection limit was confused with a clear.")
    if score(at_limit, .125, .25) != (.125, False, True):
        raise ValueError("A collection limit dropped the live successor potential.")
    for terminal, expected in (("clear", 1.), ("game_over", 0.)):
        measured = sum(score(outcome, before, after)[0] for outcome, before, after in
                       (("ongoing", 0., .125), ("ongoing", .125, .25), (terminal, .25, .25)))
        if measured != expected:
            raise ValueError("Potential shaping changed the return of a completed trajectory.")
    return {
        "stock_survival_training": {
            "completed_episodes": len(trials),
            "source_decisions": training["trained_steps"] - training["starting_steps"],
            "clears": sum(row["stages_cleared"] > 0 for row in trials),
            "episodes_reaching_frame_2000": sum(row["stage_frame"] >= 2000 for row in trials),
            "episodes_reaching_midboss_frame": sum(row["stage_frame"] >= midboss_frame for row in trials),
            "midboss_frame_reference": midboss_frame,
            "aggregate_stage_frames_after_midboss_reference": sum(max(0, row["stage_frame"] - midboss_frame)
                                                                  for row in trials),
        },
        "archived_result_checks": {
            "stage_1_advances": len(stage_clears), "one_credit_ending_accepted": True,
            "continued_ending_rejected": True,
            "stage_advance_screen": "stock gameplay screen inferred from the recorded Stage 2 transition",
        },
        "synthetic_contract_checks": {
            "recoverable_miss": True, "game_over": True, "practice_clear": True, "collector_timeout": True,
            "shaped_and_unshaped_completed_returns_equal": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    print(json.dumps(audit(args.results), indent=2))


if __name__ == "__main__":
    main()
