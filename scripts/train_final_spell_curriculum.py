"""Train pure SFT-initialized PPO after a successful demonstration prefix in stock TH14."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

import torch

from scripts import evaluate_credit_story as story, evaluate_demo_route_encounters as encounters
from scripts import evaluate_demo_route_phase, train_credit_boss_ppo as ppo
from scripts.checkpoint_storage import checkpoint_manifest, digest, read_resume
from scripts.evaluate_bomb_crossover import DATA, TEMPLATE
from scripts.evaluate_sft_native import load_arm, output_lock
from scripts.native_story_restart import GameOverMenu


def eligible(snapshot, hp_limit, *, bomb_active):
    boss = snapshot.boss
    return (snapshot.stage == 1 and snapshot.spell_id == 8 and boss is not None and boss.max_hp == 11400
            and boss.vulnerable and 0 < boss.hp <= hp_limit and snapshot.player_state == 1 and not bomb_active)


class FinalSpellRollout(ppo.BossRollout):
    def advance_prefix(self, observation, policy_seed):
        stage = self.stage
        start = None
        bomb = stage.reader.offsets.bomb
        with story.bound_policy(self.prefix.policy, stage), torch.random.fork_rng(devices=[]):
            torch.manual_seed(policy_seed)
            while True:
                snap = stage.snapshot
                if snap.boss and snap.spell_id == 8:
                    if start is None:
                        start = stage.last_frame
                    ptr = stage.reader.read_uint(bomb.bomb_pointer, relative=True)
                    if not ptr:
                        raise RuntimeError("Missing native Bomb manager during the final spell.")
                    active = bool(stage.reader.read_uint(ptr + bomb.zBomb_state))
                    if eligible(snap, self.args.handoff_hp, bomb_active=active):
                        return dict(outcome="handoff", stage_frame=stage.last_frame, stage_decisions=stage.steps,
                                    deaths=stage.hits, bombs_consumed=stage.bombs_used, player_state=snap.player_state,
                                    spell_id=snap.spell_id, boss_hp=snap.boss.hp, boss_max_hp=snap.boss.max_hp,
                                    vulnerable=True, previous_native_bomb_active=active, spell_start_frame=start,
                                    player_xy=observation["context"][:2].tolist())
                action, _ = self.prefix.policy.predict(ppo.action_observation(self.prefix, stage, observation))
                observation, _, terminated, truncated, info = stage.step(action)
                if ppo.continues(stage):
                    raise RuntimeError("The prefix entered a continued credit.")
                if terminated or truncated:
                    outcome = ppo.native_outcome(stage, info, truncated)
                    if outcome == "ongoing":
                        raise RuntimeError("Invalid native prefix terminal.")
                    return dict(outcome=outcome, stage_frame=stage.last_frame, stage_decisions=stage.steps,
                                deaths=stage.hits, bombs_consumed=stage.bombs_used,
                                reason="no eligible live final spell handoff", spell_start_frame=start)


@contextmanager
def native_session(args, arms):
    menu = GameOverMenu(*args.menu_references, args.menu_crop, .035) if args.menu_references else None
    with story.native_session(args, arms, menu, None) as (stage, _):
        yield stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DATA / "th14-demo-alice-route-261125")
    parser.add_argument("--data", type=Path, default=DATA / "sft-native-data-261123/frozen-training-snapshot")
    parser.add_argument("--bomb-model", type=Path, default=DATA.parent / "th14-randomized-bomb-value-261125/bomb_value.json")
    parser.add_argument("--init-policy", type=Path, default=TEMPLATE)
    parser.add_argument("--handoff-hp", type=int, default=600)
    parser.add_argument("--prefix-seeds", type=int, nargs="+", default=[3101, 3102, 3103, 3104])
    parser.add_argument("--seed", type=int, default=947)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--section-limit", type=int, default=1500)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--menu-references", type=Path, nargs=2, metavar=("DEFAULT", "RETRY"))
    parser.add_argument("--menu-crop", action="append", type=int, nargs=4)
    parser.add_argument("--verify-only", action="store_true")
    parser.set_defaults(shoot_policy=TEMPLATE, start_frame=30, damage_potential=.15, compare_reference=False,
                        target="stage2", focus_wait=90, limit=5500, handoff_frame=None,
                        checkpoint_budget_mib=64, disk_reserve_mib=20480)
    args = parser.parse_args()
    if (not 1 <= args.handoff_hp <= 1900 or args.steps <= 0 or args.steps % ppo.PPO_SETTINGS["n_steps"]
            or not 1 <= args.section_limit <= 2200 or not args.prefix_seeds
            or len(set(args.prefix_seeds)) != len(args.prefix_seeds) or min(args.prefix_seeds) < 0
            or args.seed < 0 or bool(args.menu_references) != bool(args.menu_crop)
            or args.menu_crop and len(args.menu_crop) < 2):
        parser.error("Use valid stock episode limits, unique seeds and complete optional menu references.")
    torch.set_num_threads(1)
    prefix = encounters.make_arms(args)["macro_encounter_clock"]
    student, template = load_arm(args.init_policy), load_arm(TEMPLATE)
    if not ppo.comparable_observations(student, template):
        raise ValueError("The pure student must use the original SFT observation contract.")
    spec = ppo.specification(args, prefix, student, template)
    spec["task"].pop("handoff_frame")
    spec["task"].pop("allowed_handoff_overshoot")
    spec["task"]["handoff"] = dict(stage=1, spell_id=8, native_max_hp=11400, remaining_hp_limit=args.handoff_hp,
                                   must_be_alive_vulnerable=True, player_state=1, bomb_active=False)
    spec["prefix"]["privileged_training_curriculum"] = True
    spec["learner"] = "Pure four-head SFT initialization; no demo, geometry, aim override or Bomb override after handoff."
    spec["curriculum_source_sha256"] = digest(Path(__file__))
    if args.verify_only:
        print(json.dumps(spec, indent=2))
        return
    if os.name != "nt" or not args.out:
        parser.error("Supply a fresh output folder for the stock Windows game.")
    resume = None
    if args.resume:
        manifest = checkpoint_manifest(args.resume)
        if not manifest or "prefix_attempt_cursor" not in manifest:
            raise ValueError("The supplied native PPO checkpoint lacks its rollout manifest.")
        fingerprint, exact, _ = read_resume(args.resume, spec)
        if not exact:
            raise ValueError("The continuation failed the original actor initialization check.")
        resume = manifest, fingerprint
    args.out.mkdir(parents=True, exist_ok=False)
    with output_lock(args.out):
        ppo.train(args, prefix, student, spec, resume, rollout_class=FinalSpellRollout, stage_session=native_session,
                  extra_sources=(sys.modules[__name__], encounters, evaluate_demo_route_phase, story))


if __name__ == "__main__":
    main()
