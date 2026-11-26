"""Fit and evaluate a regularized, two-action Bomb value model from randomized native interventions."""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import torch

from scripts import evaluate_credit_story as story
from scripts import evaluate_full_credit as credit
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import HASHES, PPO397, REPO, TEMPLATE
from scripts.evaluate_sft_native import Arm, load_arms, output_lock
from scripts.train_native_bomb_timing import BASE, OBSERVATION, SCHEMA, native_bomb_features


MODEL_SCHEMA = "th14-bomb-16frame-randomized-linear-value-v1"
ARM = "learned_bomb_value"
COST = .05
STATE_BINS = 9  # Native states 0 through 7 plus one other-state indicator.


def features(raw, contextual):
    raw = np.atleast_2d(np.asarray(raw, dtype=np.int64))
    if raw.shape[1] != 3 or np.any((raw[:, 1] < 0) | (raw[:, 1] > 1)) or np.any(raw[:, 2] < 0):
        raise ValueError("Expected native state, held Bomb bit and nonnegative native inventory.")
    if not contextual:
        return np.ones((len(raw), 1))
    state = np.where((raw[:, 0] >= 0) & (raw[:, 0] < 8), raw[:, 0], 8)
    one_hot = np.eye(STATE_BINS)[state]
    return np.hstack((one_hot, raw[:, 1:2] * one_hot, ((raw[:, 2:3] - 2) / 3) * one_hot))


def fit(rows, contextual=True):
    x = features([r["features"] for r in rows], contextual)
    action = np.asarray([r["bomb"] for r in rows])
    y = np.asarray([[r["native_life_losses"],r["actual_bombs_consumed"]] for r in rows], dtype=float)
    # Separate a shared conditional mean from a signed action contrast. Symmetric zero-centered
    # regularization makes unseen effects shrink to zero; there is no special deathbomb coefficient.
    design = np.hstack((x, (2*action[:,None] - 1) * x))
    penalty = np.full(x.shape[1], 8.) if contextual else np.ones(1)
    penalty[:STATE_BINS] = 1.
    ridge = np.concatenate((penalty, 2*penalty))
    coefficients = np.linalg.solve(design.T @ design + np.diag(ridge), design.T @ y)
    return coefficients


def action_values(coefs, raw, contextual=True):
    x = features(raw, contextual)
    mean, diff = np.split(coefs, 2)
    # Return N x action x two outcome predictions. Values are linear expected event counts.
    outcomes = np.stack((x@(mean-diff), x@(mean+diff)), axis=1)
    return -(outcomes[:,:,0] + COST * outcomes[:,:,1]), outcomes


def dataset(reports):
    credits, rejected, observed = {}, Counter(), []
    for path in reports:
        report = json.loads(path.read_bytes())
        spec = report["specification"]
        meta = spec["intervention"]
        if (report["status"] != "complete" or [g["trial"] for g in report["completed"]] != spec["schedule"]
                or spec["policies"][BASE]["weights_sha256"] != HASHES[BASE]
                or meta["schema"] != SCHEMA or tuple(meta["observation"]) != OBSERVATION
                or meta["assignment_probability"] != .5 or meta["label_native_frames"] != 16
                or meta["followup_bomb_bit"] != 0 or not meta["requires_positive_native_bomb_stock"]):
            raise ValueError("Fit requires complete 16-frame, 50/50 PPO397 randomized scout reports.")
        observed.append({"file":path.name,"sha256":digest(path),"experiment":meta})
        for trial in report["completed"]:
            ident = f'{meta["assignment_seed"]}/{trial["trial"]["policy_seed"]}'
            if ident in credits:
                raise ValueError(f"Duplicate credit {ident}.")
            rows = []
            for row in trial["result"]["windows"]:
                if not row["label_valid"]:
                    rejected[row["label_rejection"]] += 1
                    continue
                samples = row["transitions"]
                if (row["assignment_probability"] != .5 or row["bomb"] not in (0,1) or row["features"][2] <= 0
                        or not samples or samples[0]["action"][3] != row["bomb"]
                        or any(s["action"][3] for s in samples[1:]) or any(s["global_ticks"]!=4 for s in samples)
                        or not (row["elapsed_native_ticks"]==16 or 0<row["elapsed_native_ticks"]<16
                                and row["terminal_outcome"] in ("clear","game_over"))):
                    raise ValueError("A reported accepted intervention violates the randomized label contract.")
                rows.append(dict(row, credit=ident))
            credits[ident] = rows
    if len(credits) < 3 or any(not v for v in credits.values()):
        raise ValueError("Need at least three complete credits with labeled randomized decisions.")
    return credits, dict(rejected), observed


def heldout_evaluation(credits):
    rows, scores = [], {}
    for ident, valid in credits.items():
        train = [r for k,g in credits.items() if k != ident for r in g]
        if len({r["bomb"] for r in train}) != 2:
            raise ValueError("A training fold lacks one of the two assigned actions.")
        raw = [r["features"] for r in valid]
        q, outcomes = action_values(fit(train),raw)
        control, _ = action_values(fit(train,False),raw,False)
        folds = []
        for i,r in enumerate(valid):
            a, y = r["bomb"], -r["native_life_losses"]-COST*r["actual_bombs_consumed"]
            choices = {"value_head":int(q[i].argmax()),"action_only":int(control[i].argmax()),
                       "release":0,"press":1,"frozen_ppo397":r["base_bomb"]}
            estimators = {}
            for name,decision in choices.items():
                match = a == decision
                estimators[name] = dict(action=decision, matched=bool(match),ips=2*float(match)*y,
                                        dr=float(q[i,decision]+2*float(match)*(y-q[i,a])))
            folds.append(dict(credit=ident,features=r["features"],rare=r["rare"],assigned=a,utility=y,
                              observed_life_losses=r["native_life_losses"],observed_consumption=r["actual_bombs_consumed"],
                              predicted_action_values=q[i].tolist(),actual_action_prediction=outcomes[i,a].tolist(),
                              factual_squared_error=float((q[i,a]-y)**2),
                              action_only_squared_error=float((control[i,a]-y)**2),policies=estimators))
        rows.extend(folds)
        scores[ident] = summarize(folds)
    return rows,scores


def summarize(rows):
    if not rows:
        return {"n":0}
    result = {"n":len(rows),"assigned":[sum(r["assigned"]==k for r in rows) for k in (0,1)],
              "factual_rmse":float(np.sqrt(np.mean([r["factual_squared_error"] for r in rows]))),
              "action_only_rmse":float(np.sqrt(np.mean([r["action_only_squared_error"] for r in rows]))),"policies":{}}
    for p in rows[0]["policies"]:
        chosen = [r["policies"][p] for r in rows]
        result["policies"][p] = dict(ips=float(np.mean([x["ips"] for x in chosen])),dr=float(np.mean([x["dr"] for x in chosen])),
                                     presses=sum(x["action"] for x in chosen),matched=sum(x["matched"] for x in chosen))
    return result


def train(args):
    credits,rejected,source = dataset(args.reports)
    rows,folds = heldout_evaluation(credits)
    training = [r for group in credits.values() for r in group]
    weights = fit(training)
    support = {}
    for r in training:
        a = r["bomb"]
        key = ",".join(map(str,r["features"]))
        sub = support.setdefault(key,{"n":[0,0],"life_losses":[0,0],"bombs":[0,0]})
        sub["n"][a]+=1
        sub["life_losses"][a]+=r["native_life_losses"]
        sub["bombs"][a]+=r["actual_bombs_consumed"]
    for key,sub in support.items():
        value,_ = action_values(weights,[int(k) for k in key.split(",")])
        sub["fitted_q"] = value[0].tolist()
        sub["learned_action"] = int(value[0].argmax())
    unsupported = []
    for state in sorted({r["features"][0] for r in training}):
        for held in (0,1):
            for stock in range(4):
                raw = [state,held,stock]
                counts = support.get(",".join(map(str,raw)),{}).get("n",[0,0])
                if min(counts)==0:
                    value,_ = action_values(weights,raw)
                    unsupported.append(dict(features=raw,observed_by_action=counts,
                                            learned_action=int(value[0].argmax()),q=value[0].tolist()))
    equal_credit_adv = [r["policies"]["value_head"]["dr"]-r["policies"]["release"]["dr"] for r in folds.values()]
    summary=summarize(rows)
    candidate=dict(schema=MODEL_SCHEMA,observation=OBSERVATION,anchor_sha256=HASHES[BASE],
                   native_label_frames=16,bomb_cost=COST,state_bins=STATE_BINS,coefficients=weights.tolist(),
                   decoder="argmax expected short-horizon utility, argmax returns first action on an exact tie")
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"bomb_value.json").write_text(json.dumps(candidate,indent=2,allow_nan=False),encoding="utf8")
    report=dict(training=dict(model="linear conditional means and centered treatment contrasts for two measured outcomes",
                targets=["native life losses","native Bombs consumed"],utility="minus losses minus 0.05 times consumption",
                regularization="ridge 1 for player-state means, 8 for state × held and state × centered stock; doubled ridge for treatment contrasts",
                hyperparameter_selection="fixed before training, no held-out selection",cross_validation="leave one complete credit out",
                enrollment="randomized actions conditional on enrolled, positive-stock, available intervention windows. Rare and common enrollment probabilities differ.",
                offpolicy="IPS and cross-fitted doubly robust values per enrolled window; each observed action has propensity 0.5. No inverse enrollment weighting.",
                runtime_limits="Inference scores every action including unsupported stock/held states; later stages and repeated policy decisions were not randomized by the study. Collector forced release on follow-up decisions."),
                sources=source,training_credits=list(credits),training_rows=len(rows),rejected_labels=rejected,support=support,
                extrapolated_contexts_in_opening_inventory_range=unsupported,
                cross_fitted=summary,per_credit=folds,per_state={str(s):summarize([r for r in rows if r["features"][0]==s]) for s in sorted({r["features"][0] for r in rows})},
                equal_credit_dr_advantage_over_release=float(np.mean(equal_credit_adv)),favorable_credits_vs_release=sum(d>0 for d in equal_credit_adv),
                rows=rows,model_sha256=digest(args.out/"bomb_value.json"),code_sha256=digest(Path(__file__)))
    (args.out/"evidence.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps({"n":len(rows),"favorable_credits":report["favorable_credits_vs_release"],"summary":summary,"support":support}))


class LearnedBomb:
    def __init__(self,anchor,model):
        if (model["schema"]!=MODEL_SCHEMA or tuple(model["observation"])!=OBSERVATION
                or model["anchor_sha256"]!=HASHES[BASE] or model["bomb_cost"]!=COST):
            raise ValueError("Invalid learned Bomb observation, objective or frozen actor.")
        self.anchor = anchor
        self.weights = np.asarray(model["coefficients"],dtype=float)
        if self.weights.shape!=(6*STATE_BINS,2) or not np.isfinite(self.weights).all():
            raise ValueError("Invalid action-value coefficients.")
        self.environment = None
        for k in ("stage_clock","motion_spans","retain_recent_motion"):
            setattr(self,k,getattr(anchor,k))

    def predict(self,observation,deterministic=False):
        # Use original sampling order and generator, preserving the three unaffected actions.
        action,_ = self.anchor.predict(observation,deterministic=deterministic)
        values,_ = action_values(self.weights,native_bomb_features(self.environment))
        action = np.array(action,copy=True)
        action[3] = int(values[0].argmax())
        return action,None


def evaluate(args):
    arms = load_arms([], [f"{BASE}={PPO397}"], TEMPLATE)
    orig = arms[BASE]
    if orig.provenance["weights_sha256"]!=HASHES[BASE]:
        raise ValueError("Unexpected PPO397 source.")
    model = json.loads(args.model.read_bytes())
    policy = LearnedBomb(orig.policy,model)
    arms[ARM] = Arm(policy,orig.history,dict(kind=MODEL_SCHEMA,anchor_sha256=HASHES[BASE],
                    deployment_actor=orig.provenance,model_sha256=digest(args.model),
                    extended_observation=OBSERVATION,training_stage=1))
    args.crossover = []
    args.timing_policy = []
    if args.only_candidate:
        del arms[BASE]
    if getattr(args, "transfer_ppo", None):
        if args.target == "stage1":
            raise ValueError("Use the stock story scorer with --target stage2 or later for transfer evaluation.")
        transferred = load_arms([], args.transfer_ppo, TEMPLATE)
        for name, source in transferred.items():
            if name in arms:
                raise ValueError(f"Duplicate transfer arm {name}.")
            if (source.history != orig.history
                    or source.policy.observation_space != orig.policy.observation_space
                    or source.policy.action_space != orig.policy.action_space
                    or any(getattr(source.policy, option) != getattr(orig.policy, option)
                           for option in ("motion_spans", "retain_recent_motion", "stage_clock"))):
                raise ValueError(f"Transfer actor {name} does not use the frozen anchor's input and action contract.")
            arms[name] = Arm(LearnedBomb(source.policy, model), source.history, dict(
                kind=MODEL_SCHEMA, training_anchor_sha256=HASHES[BASE],
                deployment_actor=source.provenance, model_sha256=digest(args.model),
                extended_observation=OBSERVATION, training_stage=1))
    run_arms = getattr(args, "run_arm", None)
    if run_arms:
        if set(run_arms) - arms.keys() or len(set(run_arms)) != len(run_arms):
            raise ValueError("Each requested evaluation arm must be loaded and named once.")
        arms = {name: arms[name] for name in run_arms}
    if args.target=="stage1":
        spec = credit.run_specification(args,arms)
    else:
        spec = story.run_specification(args,arms)
    spec["controller"]["kind"]=MODEL_SCHEMA
    spec["runtime"]["source_sha256"]["scripts.fit_randomized_bomb_value"]=digest(Path(__file__))
    spec["runtime"]["source_sha256"]["scripts.train_native_bomb_timing"]=digest(Path(sys.modules[native_bomb_features.__module__].__file__))
    spec=json.loads(json.dumps(spec))
    transition_archive = None
    if getattr(args, "record_transitions", False):
        from scripts.native_story_transitions import configure

        if args.target == "stage1":
            raise ValueError("Transition recording requires the stock story scorer. Use --target stage2 or later.")
        transition_archive = configure(args, spec)
    if not args.resume:
        args.out.mkdir(parents=True,exist_ok=False)
    with output_lock(args.out):
        if args.target=="stage1":
            @contextmanager
            def session(args,arms):
                with credit.native_stage(args,arms) as stage, story.bound_policy(policy,stage):
                    yield stage
            return credit.run_evaluation(args,arms,args.out/"report.json",spec,stage_session=session)
        refs=(args.game_over_default_reference,args.game_over_retry_reference,args.menu_crop)
        menu=story.GameOverMenu(*refs,args.menu_max_error) if all(refs) else None
        return story.evaluate(args,arms,spec,args.out/"report.json",menu,
                              story.hook(args.restart_hook) if args.restart_hook else None,
                              transition_archive=transition_archive)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    cmds=parser.add_subparsers(dest="mode",required=True)
    fitting=cmds.add_parser("fit")
    fitting.add_argument("--reports",type=Path,nargs="+",required=True)
    fitting.add_argument("--out",type=Path,required=True)
    native=cmds.add_parser("evaluate")
    native.add_argument("--out",type=Path,required=True)
    native.add_argument("--model",type=Path,required=True)
    native.add_argument("--target",choices=["stage1",*story.TARGETS],default="stage1")
    native.add_argument("--parakit",type=Path,default=REPO.parent/"parakit")
    native.add_argument("--only-candidate",action="store_true")
    native.add_argument("--transfer-ppo", action="append", metavar="NAME=CHECKPOINT",
                        help="Evaluate the same frozen Bomb value head with another native PPO actor.")
    native.add_argument("--run-arm", action="append",
                        help="Score this named arm while loading the same source policies.")
    native.add_argument("--record-transitions", action="store_true",
                        help="Archive exact actor inputs, actions and native combat state for each story credit.")
    native.add_argument("--transition-budget-mib", type=int, default=256)
    native.add_argument("--transition-reserve-mib", type=int, default=2048)
    native.add_argument("--seeds",type=int,nargs="+",default=[1901,1902,1903,1904])
    native.add_argument("--start-frame",type=int,default=30)
    native.add_argument("--limit",type=int)
    native.add_argument("--focus-wait",type=int,default=30)
    native.add_argument("--resume",action="store_true")
    native.add_argument("--restart-hook")
    native.add_argument("--game-over-default-reference",type=Path)
    native.add_argument("--game-over-retry-reference",type=Path)
    native.add_argument("--menu-crop",action="append",nargs=4,type=int)
    native.add_argument("--menu-max-error",type=float,default=.035)
    args=parser.parse_args()
    if args.mode=="fit":
        return train(args)
    args.limit=args.limit or (3300 if args.target=="stage1" else story.LIMITS[args.target])
    if min(args.start_frame,args.limit)<=0 or args.focus_wait<0 or any(not 0<=s<2**63 for s in args.seeds) or len(set(args.seeds))!=len(args.seeds):
        parser.error("Use distinct valid seeds and valid frame, decision and wait limits.")
    refs=(args.game_over_default_reference,args.game_over_retry_reference,args.menu_crop)
    if any(refs) and (not all(refs) or len(args.menu_crop)<2) or not 0<args.menu_max_error<.2:
        parser.error("Provide both verified menu references and at least two row crops.")
    if args.record_transitions and (args.target == "stage1" or args.transition_budget_mib < 1
                                    or args.transition_reserve_mib < 0):
        parser.error("Recording requires a story target and valid transition budget and reserve.")
    torch.set_num_threads(1)
    evaluate(args)


if __name__=="__main__":
    main()
