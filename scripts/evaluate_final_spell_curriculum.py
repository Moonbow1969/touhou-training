"""Score a stock TH14 reverse-curriculum PPO against SFT and a passive suffix."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

from scripts import evaluate_credit_story as story, evaluate_demo_route_encounters as encounters
from scripts import evaluate_demo_route_phase, native_story_restart, native_story_transitions as transitions
from scripts import train_credit_boss_ppo as training, train_final_spell_curriculum as curriculum
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import DATA, TEMPLATE
from scripts.evaluate_full_credit import continues, resources
from scripts.evaluate_sft_native import Arm, load_arm, load_ppo_arm, output_lock
from scripts.evaluate_stage_policies import policy_observation
from scripts.pretrain_policy import ACTION_NAMES
from scripts.train_full_policy import publish_json


ARM_NAMES = ("passive_neutral", "original_sft", "curriculum_ppo256")
ORIGINAL_SFT_SHA256 = "d62aea41897019df9bec90f3f3288a350a0dfbd1bd471fbee0377400cdc1e2bc"
HANDOFF = dict(stage=1, spell_id=8, native_max_hp=11400, remaining_hp_limit=600,
               must_be_alive_vulnerable=True, player_state=1, bomb_active=False)
DEFAULT_CHECKPOINT = DATA / "th14-native-final-spell-sft-ppo-256-261125/policy.zip"
NEUTRAL = (4, 0, 0, 0)
JOURNAL_SCHEMA = "th14-final-spell-natural-handoff-v1"


class SpellSuffix:
    # This adapter routes the exact shared observation into each source's own transform.
    motion_spans = None
    retain_recent_motion = False
    stage_clock = False

    def __init__(self, prefix, suffix, *, passive=False):
        self.prefix, self.suffix, self.passive = prefix, suffix, passive
        self.environment = None
        self.entries = None

    def begin(self, trial):
        seed = trial["policy_seed"]
        self.entries = dict(reached=False, prefix_seed=seed, suffix_seed=seed ^ (1 << 62), spell_start_frame=None)
        self.handoff_ns = None
        return self.entries

    def try_handoff(self):
        env, entry = self.environment, self.entries
        snapshot, threshold = env.snapshot, HANDOFF["remaining_hp_limit"]
        if snapshot.stage != HANDOFF["stage"] or snapshot.spell_id != HANDOFF["spell_id"] or snapshot.boss is None:
            return False
        if entry["spell_start_frame"] is None:
            entry["spell_start_frame"] = env.last_frame
        entry["min_spell_hp_seen_in_prefix"] = min(snapshot.boss.hp,
                                                  entry.get("min_spell_hp_seen_in_prefix", snapshot.boss.hp))
        if not curriculum.eligible(snapshot, threshold, bomb_active=False):
            return False
        offsets = env.reader.offsets.bomb
        pointer = env.reader.read_uint(offsets.bomb_pointer, relative=True)
        if not pointer:
            raise RuntimeError("Missing native Bomb manager at the final-spell eligibility check.")
        if not curriculum.eligible(snapshot, threshold, bomb_active=bool(env.reader.read_uint(pointer + offsets.zBomb_state))):
            entry["eligible_hp_rows_blocked_by_active_bomb"] = entry.get("eligible_hp_rows_blocked_by_active_bomb", 0) + 1
            return False
        if continues(env) or env.current_stage() != 1 or env.game.read_game_val("game_state") != 2:
            raise RuntimeError("Suffix handoff requires active Stage 1 and the same zero-Continue credit.")
        player = snapshot.observation["context"][:2] * np.array((192, 448), dtype=np.float32)
        entry.update(reached=True, prefix_decisions=env.steps, first_suffix_decision=env.steps + 1,
                     stage_frame=env.last_frame, native_stage_frame=env.stage_frame(), spell_id=snapshot.spell_id,
                     boss_hp=snapshot.boss.hp, boss_max_hp=snapshot.boss.max_hp, boss_vulnerable=True,
                     player_state=snapshot.player_state, player_xy=player.tolist(), active_bomb=False,
                     resources=resources(env), prefix_lives_lost=env.hits, prefix_bombs_consumed=env.bombs_used,
                     held_keys_before_release=sorted(env.held))
        self.handoff_ns = perf_counter_ns()
        env.release()
        torch.manual_seed(entry["suffix_seed"])
        return True

    def predict(self, observation, deterministic=False):
        env = self.environment
        if env is None or self.entries is None:
            raise RuntimeError("Bind and reset a fresh native credit before running the final-spell comparator.")
        first_suffix = not self.entries["reached"] and self.try_handoff()
        if not self.entries["reached"]:
            with story.bound_policy(self.prefix.policy, env) as policy:
                return policy.predict(policy_observation(policy, env, observation, self.prefix.history),
                                      deterministic=deterministic)
        if self.passive:
            action, state = np.asarray(NEUTRAL, dtype=np.int64), None
        elif self.suffix is None:
            raise RuntimeError("The trained suffix is awaiting a completed checkpoint.")
        else:
            policy = self.suffix.policy
            action, state = policy.predict(policy_observation(policy, env, observation, self.suffix.history),
                                           deterministic=deterministic)
        if first_suffix:
            self.entries["first_suffix_action"] = dict(zip(ACTION_NAMES, map(int, action)))
        return action, state


def load_curriculum(path, template, prefix, allow_pending):
    report_path = path.with_name("report.json")
    prior = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    if allow_pending and (prior is None and not path.is_file() or prior and prior["status"] == "started"):
        return None
    if not prior or prior["status"] != "complete":
        raise ValueError("Wait for the completed native curriculum report and its committed policy.zip.")
    arm = load_ppo_arm(path, template)
    c = arm.provenance["training_configuration"]
    expected_prefix = json.loads(json.dumps(prefix.provenance))
    if (prior["specification"] != c or prior["checkpoint_steps"] != 256 or prior["starting_steps"] != 0
            or prior["requested_additional_steps"] != 256 or prior["resume_checkpoint_sha256"] is not None
            or prior["checkpoints"]["ppo_sha256"] != arm.provenance["weights_sha256"]
            or not arm.provenance["warm_start_actor_exact"] or arm.provenance["native_training_decisions"] != 256
            or c["initialization_sha256"] != ORIGINAL_SFT_SHA256 or c["prefix"]["provenance"] != expected_prefix
            or c["prefix"]["privileged_training_curriculum"] is not True or c["task"]["handoff"] != HANDOFF
            or c["task"]["start_frame"] != 30 or c["curriculum_source_sha256"] != digest(Path(curriculum.__file__))
            or not training.comparable_observations(arm, template)):
        raise ValueError("The PPO archive must be the exact 256-decision native reverse curriculum from original SFT.")
    arm.provenance["training_report_sha256"] = digest(report_path)
    return arm


def load_arms(args):
    original = load_arm(args.shoot_policy)
    if original.provenance["weights_sha256"] != ORIGINAL_SFT_SHA256:
        raise ValueError("Use the unchanged original SFT actor and its native transforms.")
    arms = {}
    for name in ARM_NAMES:
        prefix = encounters.make_arms(args)["macro_encounter_clock"]
        passive = name == ARM_NAMES[0]
        if passive:
            suffix = None
            source = dict(kind="neutral", action=dict(zip(ACTION_NAMES, NEUTRAL)))
        elif name == ARM_NAMES[1]:
            suffix, source = original, original.provenance
        else:
            suffix = load_curriculum(args.checkpoint, original, prefix, args.verify_only)
            source = suffix.provenance if suffix else dict(status="awaiting_curriculum_checkpoint")
        policy = SpellSuffix(prefix, suffix, passive=passive)
        arms[name] = Arm(policy, prefix.history or original.history, dict(prefix=prefix.provenance, suffix=source))
    return arms


def suffix_outcome(result, trace, handoff_ns):
    entry = result["reset_info"]["curriculum_handoff"]
    arrival = next((row["entry"] for row in result["stage_records"] if row["stage"] == 2), None)
    metrics = dict(handoff_reached=entry["reached"], native_stage2_entry=arrival,
                   native_stage2_reached=arrival is not None, suffix_stage2_success=entry["reached"] and arrival is not None)
    if not entry["reached"]:
        metrics.update(no_handoff=True, terminal_outcome=result["outcome"])
        return metrics

    fields = {name: i for i, name in enumerate(transitions.NATIVE_FIELDS)}
    read = lambda row, name: row[fields[name]]
    rows = trace.rows[entry["prefix_decisions"]:]
    if not rows or handoff_ns is None:
        raise RuntimeError("A reached handoff must have at least one journaled suffix action.")
    first, final = rows[0][1][0], rows[-1][2][0]
    key = tuple(read(first, field) for field in ("stage", "boss_pointer", "boss_max_hp", "spell_id"))
    if (key[0], key[2], key[3], read(first, "boss_hp"), read(first, "stage_frame")) != (
            1, 11400, 8, entry["boss_hp"], entry["native_stage_frame"]):
        raise RuntimeError("The journal does not link to the verified natural handoff.")

    def matched(row):
        return read(row, "snapshot_valid") and key == tuple(
            read(row, field) for field in ("stage", "boss_pointer", "boss_max_hp", "spell_id"))

    measured = 0
    positives, increases, eligible, sampled = 0, 0, 0, 0
    observed_hp = [entry["boss_hp"]]
    last_spell_frame = entry["native_stage_frame"]
    for _, (before, _), (after, _), *_ in rows:
        if matched(before):
            sampled += 1
            last_spell_frame = read(before, "stage_frame")
            observed_hp.append(read(before, "boss_hp"))
        if not (matched(before) and matched(after)):
            continue
        eligible += 1
        observed_hp.append(read(after, "boss_hp"))
        last_spell_frame = read(after, "stage_frame")
        delta = read(before, "boss_hp") - read(after, "boss_hp")
        if delta > 0:
            measured += delta
            positives += 1
        elif delta < 0:
            increases -= delta
    native_elapsed = read(final, "global_tick") - read(first, "global_tick")
    if native_elapsed < 0:
        raise RuntimeError("Native clock ran backwards within a fresh credit.")
    metrics.update(
        terminal_outcome=result["outcome"], suffix_decisions=len(rows), suffix_native_ticks=native_elapsed,
        suffix_native_seconds=native_elapsed / 60., suffix_wall_seconds=(trace.finished_ns - handoff_ns) / 1e9,
        entry_hp=entry["boss_hp"], entry_stock=entry["resources"],
        same_spell_samples=sampled, linked_hp_transitions=eligible, raw_linked_hp_decrease=measured,
        positive_hp_decrement_transitions=positives, raw_linked_hp_increase=increases,
        lowest_observed_hp=min(observed_hp), last_observed_hp=observed_hp[-1], last_observed_hp_frame=last_spell_frame,
        terminal_hp_unobserved=not bool(matched(final)),
        suffix_lives_lost=result["deaths"] - entry["prefix_lives_lost"],
        suffix_actual_bombs_consumed=result["actual_bombs_consumed"] - entry["prefix_bombs_consumed"],
    )
    return metrics


class Journal(transitions.StoryTransitions):
    def finish(self, outcome, final_observation):
        self.finished_ns = perf_counter_ns()
        super().finish(outcome, final_observation)


class JournalArchive(transitions.StoryArchive):
    def __init__(self, args, specification, arms):
        super().__init__(args, specification)
        self.arms = arms

    def start(self, env):
        transitions.archive.storage_remaining(self.out, self.args)
        return Journal(env)

    def metadata(self, trial, offsets, rows):
        return dict(super().metadata(trial, offsets, rows), final_spell_journal=JOURNAL_SCHEMA)

    def publish(self, trial, result, trace):
        policy = self.arms[trial["arm"]].policy
        result["curriculum_suffix"] = suffix_outcome(result, trace, policy.handoff_ns)
        return super().publish(trial, result, trace)


def summarize_report(report):
    groups = {}
    for row in report["completed"]:
        trial, result = row["trial"], row["result"]
        group = groups.setdefault(trial["block"], dict(seed=trial["policy_seed"], arms={}))
        group["arms"][trial["arm"]] = dict(outcome=result["outcome"], handoff=result["reset_info"]["curriculum_handoff"],
                                           suffix=result["curriculum_suffix"])
    matched = [group["arms"] for group in groups.values() if len(group["arms"]) == len(ARM_NAMES)]
    arms = {}
    for name in ARM_NAMES:
        rows = [group[name] for group in matched]
        all_rows = [group["arms"][name] for group in groups.values() if name in group["arms"]]
        arms[name] = dict(credit_denominator=len(rows), handoffs=sum(row["suffix"]["handoff_reached"] for row in rows),
                          no_handoff_outcomes=dict(Counter(row["outcome"] for row in rows if
                                                          not row["suffix"]["handoff_reached"])),
                          suffix_stage2_successes=sum(row["suffix"]["suffix_stage2_success"] for row in rows),
                          all_stage2_entries=sum(row["suffix"]["native_stage2_reached"] for row in rows),
                          completed_credit_denominator=len(all_rows),
                          completed_no_handoff_outcomes=dict(Counter(row["outcome"] for row in all_rows if
                                                                    not row["suffix"]["handoff_reached"])),
                          completed_suffix_stage2_successes=sum(row["suffix"]["suffix_stage2_success"] for row in all_rows))
    return dict(schema=JOURNAL_SCHEMA, status=report["status"], planned_credits=len(report["specification"]["schedule"]),
                completed_credits=len(report["completed"]), matched_blocks=len(matched), arms=arms, by_block=groups)


def specification(args, arms):
    spec = story.run_specification(args, arms)
    spec["controller"].update(kind="matched_natural_final_spell_suffix", natural_handoff=HANDOFF,
                              prefix_failure_in_denominator=True, policy_unchanged_between_stages=False,
                              suffix_decode="sampled full four-head SFT/PPO or all-neutral", release_at_handoff=True)
    spec["runtime"]["runner_sha256"] = digest(Path(__file__))
    sources = (story, encounters, evaluate_demo_route_phase, curriculum, training, transitions)
    spec["runtime"]["source_sha256"].update({m.__name__: digest(Path(m.__file__)) for m in sources})
    spec["final_spell_journal"] = dict(schema=JOURNAL_SCHEMA,
                                       dmg="Positive raw observed decrements when same boss and native spell persist.",
                                       missing_terminal_hp="Reported separately; no last HP is imputed.")
    transitions.configure(args, spec)
    return spec


def run(args, arms, spec, menu):
    archive = JournalArchive(args, spec, arms)
    path = args.out / "report.json"

    def reset(env, trial):
        observation, info = native_story_restart.reset(env, trial, start_frame=args.start_frame, menu=menu)
        return observation, dict(info, curriculum_handoff=arms[trial["arm"]].policy.begin(trial))

    try:
        return story.evaluate(args, arms, spec, path, menu, None, reset_callback=reset, transition_archive=archive)
    finally:
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("specification") == spec:
                publish_json(args.out / "matched-suffix.json", summarize_report(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--artifact", type=Path, default=DATA / "th14-demo-alice-route-261125")
    parser.add_argument("--data", type=Path, default=DATA / "sft-native-data-261123/frozen-training-snapshot")
    parser.add_argument("--bomb-model", type=Path, default=DATA.parent / "th14-randomized-bomb-value-261125/bomb_value.json")
    parser.add_argument("--shoot-policy", type=Path, default=TEMPLATE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3201, 3202, 3203, 3204])
    parser.add_argument("--limit", type=int, default=5500)
    parser.add_argument("--focus-wait", type=int, default=90)
    parser.add_argument("--menu-references", type=Path, nargs=2, metavar=("DEFAULT", "RETRY"))
    parser.add_argument("--menu-crop", action="append", type=int, nargs=4)
    parser.add_argument("--transition-budget-mib", type=int, default=64)
    parser.add_argument("--transition-reserve-mib", type=int, default=20480)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.set_defaults(target="stage2", start_frame=30, compare_reference=False, menu_max_error=.035,
                        crossover=[], timing_policy=[], restart_hook=None)
    args = parser.parse_args()
    if (min(args.limit, args.transition_budget_mib) < 1 or args.focus_wait < 0 or args.transition_reserve_mib < 0
            or bool(args.menu_references) != bool(args.menu_crop) or args.menu_crop and len(args.menu_crop) < 2
            or len(args.seeds) != len(set(args.seeds))
            or any(not 0 <= seed < 2**63 or seed in range(3101, 3105) for seed in args.seeds)):
        parser.error("Use valid native limits, distinct seeds outside the training rotation, and complete menu references.")
    if not args.verify_only and (os.name != "nt" or not args.out or not args.menu_references):
        parser.error("Native evaluation requires Windows, --out, and the verified Game Over menu references/crops.")
    args.game_over_default_reference, args.game_over_retry_reference = args.menu_references or (None, None)
    torch.set_num_threads(1)
    try:
        menu = (native_story_restart.GameOverMenu(*args.menu_references, args.menu_crop, args.menu_max_error)
                if args.menu_references else None)
        arms = load_arms(args)
        spec = specification(args, arms)
        if args.verify_only:
            print(json.dumps(dict(status="verified" if arms[ARM_NAMES[2]].policy.suffix else "pending_training_checkpoint",
                                  ready_for_native=bool(arms[ARM_NAMES[2]].policy.suffix and args.menu_references),
                                  specification=spec), indent=2), flush=True)
            return
        if not args.resume:
            args.out.mkdir(parents=True, exist_ok=False)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    with output_lock(args.out):
        report = run(args, arms, spec, menu)
    print(json.dumps(summarize_report(report)["arms"]), flush=True)


if __name__ == "__main__":
    main()
