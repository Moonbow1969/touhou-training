"""Compare randomized risk-ranked movement to random movement in the original TH14 trial."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from random import Random
from shutil import disk_usage
from threading import Lock
from time import perf_counter

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts import collect_trial_contact_windows as trial
from scripts.score_trial_contact_windows import observations
from scripts.train_anticipatory_contact import AnticipatoryContact
from scripts.train_full_policy import publish_json
from scripts.train_local_avoidance import local_context


MODEL_SHA256 = "7700eb95c641a1c95bdf9c4a7d51d4137a17755fe70ab673e99fff8a3301d124"
ARMS = ("uniform", "lowest_risk_four")
LIMIT = 1700
SEED = 4_102_719
ATTEMPTS = 3


@dataclass(frozen=True)
class Assignment:
    worker: str
    host: str
    pair: int
    order: int
    arm: str
    seed: int

    @property
    def key(self):
        return f"{self.worker}/{self.pair}/{self.arm}"


def schedule(worker_host, lanes, blocks):
    return [
        Assignment(lane, worker_host[lane], block + 1, order + 1,
                   ARMS[(lane_index + block + order) % 2],
                   SEED + 100_003 * lane_index + block * 103 + order * 101)
        for lane_index,lane in enumerate(lanes) for block in range(blocks) for order in range(2)
    ]


def choose(rng, risk=None):
    focus = rng.randrange(2)
    if risk is None:
        return rng.randrange(9), focus
    if risk.shape != (2,9) or not np.isfinite(risk).all():
        raise ValueError("The frozen risk head did not return its 18 finite action scores.")
    rank = np.lexsort(([rng.random() for _ in range(9)], risk[focus]))
    return int(rank[rng.randrange(4)]), focus


@torch.inference_mode()
def risk_scores(model, state):
    row = dict(start=dict(player=state["player"], prior_focus=bool(state["held"] & 16),
                          bullets=state["bullets"], bullet_count=state["bullet_count"]))
    bullets, context = observations([row])
    return model(bullets, local_context(context, 0))[0, :18].reshape(2,9).numpy()


def load_model(path):
    report = json.loads(path.with_name("report.json").read_text())
    if digest(path) != MODEL_SHA256 or report["model_sha256"] != MODEL_SHA256 or report["status"] != "complete":
        raise ValueError("The randomized trial is locked to the original replay-trained model.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema"] != "th14-learned-contact-12tick-v1":
        raise ValueError("Unexpected held-12-tick risk model schema.")
    model = AnticipatoryContact().eval()
    model.load_state_dict(payload["state_dict"])
    return model


def preflight(model, spec):
    tied = np.zeros((2,9), dtype=np.float32)
    rng = Random(1949)
    frequencies = Counter(choose(rng, tied) for _ in range(9000))
    levels = np.arange(18).reshape(2,9).astype(np.float32)
    assert len(frequencies) == 18 and all(415 < n < 585 for n in frequencies.values()), frequencies
    assert all(move < 4 for move, focus in [choose(rng, levels) for _ in range(900)])
    assert len(spec["assignments"]) == len({row["seed"] for row in spec["assignments"]})
    toy = dict(player=[0.,350.], held=0, bullet_count=1, bullets=[[0.,-20.,0.,2.,5.,1]])
    assert risk_scores(model, toy).shape == (2,9)
    return dict(tie_frequency_min=min(frequencies.values()), tie_frequency_max=max(frequencies.values()), finite_shape=[2,9])


def opening(pipe, prior=None):
    state = pipe.read(timeout=90) if prior is None else pipe.request("R")
    trial.check_state(state)
    if (not 0 <= state["frame"] < 90 or state["delta"] or state["lives"] != 2 or state["held"]
            or state["player_state"] == trial.HIT or prior is not None and state["frame"] >= prior["frame"]):
        raise RuntimeError("The official trial did not start an untouched two-stock credit.")
    return state


def run_credit(pipe, state, assignment, model, radius):
    rng = Random(assignment.seed)
    start = dict(frame=state["frame"], lives=state["lives"], bombs=state["bombs"], player_state=state["player_state"])
    assigned = np.zeros((2,9), dtype=int)
    achieved, decision_count, observed = state["frame"], 0, None
    masks = Counter()

    def step(mask):
        nonlocal state, achieved, decision_count, observed
        previous = state
        state = pipe.request(f"S {mask}")
        trial.check_state(state)
        exact = (state["ticks"] - previous["ticks"]) % (2**32) == state["frame"] - previous["frame"] == state["delta"] == 4
        if not exact or state["held"] != mask:
            raise RuntimeError("The closed-loop credit lost exact native timing or its held command.")
        masks[mask] += 1
        decision_count += 1
        if observed is None:
            achieved = state["frame"]
            if state["player_state"] == trial.HIT:
                observed = [previous["frame"] + 1, state["frame"]]
            elif state["lives"] != 2:
                raise RuntimeError("Native inventory changed before a readable hit onset.")

    while observed is None and state["frame"] < LIMIT:
        eligible = (state["frame"] >= trial.FIRST_FRAME and state["player_state"] == trial.ACTIVE
                    and trial.in_range(state, radius))
        if eligible:
            risk = risk_scores(model, state) if assignment.arm == ARMS[1] else None
            move, focus = choose(rng, risk)
            assigned[focus,move] += 1
            mask = trial.action_mask(move, focus)
            for _ in range(3):
                step(mask)
        else:
            mask = trial.SHOOT
            step(mask)
    if observed is not None:
        for _ in range(trial.INJURY_CONFIRM_STEPS):
            if state["lives"] < 2:
                break
            step(mask)
        if state["lives"] != 1:
            raise RuntimeError("Native injury did not corroborate the observed hit.")
        if observed[1] < trial.FIRST_FRAME:
            raise RuntimeError("The native opening registered a startup hit.")
    event = observed is not None and observed[1] <= LIMIT
    result = dict(assignment=asdict(assignment), start=start, injury_by_1700=event, injury_interval=observed,
                  cutoff_straddles=bool(observed and observed[0] <= LIMIT < observed[1]),
                  restricted_survival_frame=min(observed[1], LIMIT) if observed else LIMIT, achieved_frame=achieved,
                  last_qualified_native_frame=state["frame"], confirmed_lives=state["lives"], exact_native_decisions=decision_count,
                  held_action_counts=assigned.tolist(), command_counts=dict(masks))
    return state, result


def paired_summary(completed):
    by_arm = {arm: [row for row in completed.values() if row["assignment"]["arm"] == arm] for arm in ARMS}
    summary = {}
    for arm, rows in by_arm.items():
        summary[arm] = dict(credits=len(rows), injuries=sum(row["injury_by_1700"] for row in rows),
                            cutoff_straddles=sum(row["cutoff_straddles"] for row in rows),
                            restricted_mean_survival_frames=float(np.mean([row["restricted_survival_frame"] for row in rows])),
                            median_achieved_frame=float(np.median([row["achieved_frame"] for row in rows])))
    pairs = defaultdict(dict)
    for row in completed.values():
        a = row["assignment"]
        pairs[(a["host"],a["worker"],a["pair"])][a["arm"]] = row
    by_host = defaultdict(list)
    for key, rows in sorted(pairs.items()):
        c,b = (rows[a] for a in ARMS)
        by_host[key[0]].append((int(c["injury_by_1700"]) - int(b["injury_by_1700"]),
                                b["restricted_survival_frame"] - c["restricted_survival_frame"]))
    host_arrays = [np.asarray(group) for group in by_host.values()]
    effect = np.concatenate(host_arrays).mean(axis=0)
    rng = np.random.default_rng(20723)
    simulations = [np.concatenate([group[rng.integers(len(group), size=len(group))] for group in host_arrays]).mean(axis=0)
                   for _ in range(3000)]
    interval = np.quantile(simulations,[.025,.975],axis=0)
    return dict(arms=summary, pairs=len(pairs), control_minus_risk_injury_rate=float(effect[0]),
                risk_minus_control_restricted_survival_frames=float(effect[1]),
                host_stratified_paired_bootstrap_95=dict(injury_rate=interval[:,0].tolist(), survival_frames=interval[:,1].tolist()),
                by_host={k:dict(pairs=len(g),control_minus_risk_injury_rate=float(np.mean(g,axis=0)[0]),
                                risk_minus_control_restricted_survival_frames=float(np.mean(g,axis=0)[1])) for k,g in by_host.items()})


def run(args, plan, training, workers, model, assignments):
    if training["status"] != "complete" or training["totals"]["windows"] < 20000:
        raise RuntimeError("The separate training corpus still owns the six trial lanes.")
    if disk_usage(args.out).free < 8*trial.GIB:
        raise OSError("The evidence volume has less than 8 GiB free.")
    if {a.seed for a in assignments} & {row["seed"] for row in training["shards"].values()}:
        raise ValueError("A closed-loop seed overlaps the training corpus.")
    path = args.out / "report.json"
    training_hash = digest(args.training / "manifest.json")
    report = dict(plan_sha256=digest(args.out / "plan.json"), training_manifest_sha256=training_hash,
                  status="evaluating", completed={}, rejected_attempts={})
    if args.resume:
        prior = json.loads(path.read_text())
        if any(report[key]!=prior[key] for key in ("plan_sha256", "training_manifest_sha256")):
            raise ValueError("The resumed fixed trial has different lineage.")
        expected = {a.key: asdict(a) for a in assignments}
        if any(expected.get(key) != row["assignment"] for key,row in prior["completed"].items()):
            raise ValueError("A completed result does not belong to the fixed assignment schedule.")
        report = prior
    elif path.exists():
        raise FileExistsError("Use the explicit fixed-plan resume for an existing report.")
    publish_json(path, report)
    lock = Lock()

    def play(lane):
        pipeline = None
        state = None
        try:
            for assignment in assignments:
                if assignment.worker != lane or assignment.key in report["completed"]:
                    continue
                for _ in range(ATTEMPTS):
                    try:
                        if pipeline is None:
                            pipeline = trial.TrialWorker(workers[lane]["command"]).__enter__()
                        state = opening(pipeline, state)
                        state, result = run_credit(pipeline, state, assignment, model, plan["near_radius"])
                    except (RuntimeError, ValueError, OSError) as error:
                        with lock:
                            report["rejected_attempts"].setdefault(assignment.key,[]).append(f"{type(error).__name__}: {error}")
                            publish_json(path, report)
                        if pipeline is not None:
                            pipeline.__exit__(None,None,None)
                        pipeline, state = None, None
                    else:
                        with lock:
                            report["completed"][assignment.key] = result
                            publish_json(path, report)
                        print(json.dumps(dict(event="qualified_credit", **result)), flush=True)
                        break
        finally:
            if pipeline is not None:
                pipeline.__exit__(None,None,None)

    with ThreadPoolExecutor(max_workers=len(plan["worker_host"])) as pool:
        list(pool.map(play, plan["worker_host"]))
    complete = len(report["completed"]) == len(assignments)
    report["status"] = "complete" if complete else "partial"
    if complete:
        report["fixed_cohort"] = paired_summary(report["completed"])
    publish_json(path, report)
    if not complete:
        raise RuntimeError("Fixed cohort incomplete. Every qualified assigned credit is preserved.")
    print(json.dumps(report["fixed_cohort"],indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers-config",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--training",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--blocks",type=int,default=6)
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    training=json.loads((args.training/"manifest.json").read_text())
    workers=json.loads(args.workers_config.read_text())
    spec=training["specification"]
    host, lanes = training["worker_host"], spec["lane_order"]
    if spec["purpose"]!="training_only" or set(host)!=set(lanes) or set(host)!=set(workers):
        parser.error("Use the known corpus lanes and separate training manifest.")
    if args.blocks<5:
        parser.error("The fixed cohort needs at least five paired blocks on each lane.")
    fingerprints = ("executable_sha256", "worker_source_sha256", "readout_sha256", "image_id")
    qualified = {}
    for lane in lanes:
        worker = workers[lane]
        if worker["status"] != "ready" or worker["executable_sha256"] != trial.TRIAL_SHA256:
            parser.error("Every lane needs the verified original trial executable.")
        qualified[lane] = {key: worker[key] for key in fingerprints}
    if any(qualified[row["worker"]] != row["launch"] for row in training["shards"].values()):
        parser.error("A trial worker's immutable version has changed since the training acquisition.")
    assignments=schedule(host,lanes,args.blocks)
    model=load_model(args.model)
    plan=dict(schema=1,setting="Official TH14 trial 0.01b Normal Reimu A Stage 1",
              model_sha256=MODEL_SHA256, reserve_collection_sha256=spec["reserved_eval_collection_sha256"],
              runner_sha256=digest(Path(__file__)), worker_host=host, qualified_workers=qualified,
              near_radius=spec["sampling"]["near_projectile_radius"], eligible_start_frame=trial.FIRST_FRAME,
              endpoint_native_frame=LIMIT, frames_per_action=12, four_tick_checks=3, shoot=1,bomb=0,initial_spare_lives=2,
              attempts_per_assignment=ATTEMPTS, assignments=[asdict(a) for a in assignments],
              primary="First observable native hit checkpoint at/before frame 1700 corroborated by life stock",
              secondary="Restricted mean of hit upper frame and 1700, plus achieved native frame",
              sampling=dict(uniform="uniform Focus and movement",lowest_risk_four="uniform Focus; random tie-break; uniform among lowest 4/9 movement risks",
                            between_windows="Shoot only outside active bullet-bearing eligibility",
                            native_rng_controlled=False, no_checkpoint_updates=True, fixed_cohort=True))
    plan["preflight"]=preflight(model,plan)
    if args.plan_only:
        args.out.mkdir(parents=True,exist_ok=False)
        publish_json(args.out/"plan.json",plan)
        print(json.dumps(dict(status="planned",credits=len(assignments),checks=plan["preflight"])))
        return
    if json.loads((args.out/"plan.json").read_text()) != plan:
        raise ValueError("Collect only the previously qualified frozen schedule.")
    run(args,plan,training,workers,model,assignments)


if __name__=="__main__":
    main()
