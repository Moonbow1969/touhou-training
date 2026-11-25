"""Fit one Bomb readout from two archived PPO policies using offline TH14 observations."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import gymnasium as gym
import torch
from torch.nn import functional as F

ROOT = next(p for p in Path(__file__).resolve().parents if (p/"work/th14-rl").is_dir())
REPO = ROOT/"work/th14-rl"
sys.path.insert(0, str(REPO))
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import DATA, HASHES, PPO211, PPO397, TEMPLATE
from scripts.evaluate_sft_native import load_arm, load_arms
from scripts.pretrain_policy import read_shard
from scripts.train_full_policy import warm_start

DATASET = DATA/"sft-native-data-261123/frozen-training-snapshot"
TRAIN = ("th14_ud0864","th14_ud05e0","th14_ud0613")
HOLD = ("th14_ud06d6","th14_ud068f")
RIDGE = 1e-5


def bomb_odds(logits):
    return logits[:, -1] - logits[:, -2]


@torch.inference_mode()
def read_credit(stem, source, teacher, candidate=None):
    rows = []
    for sidecar in sorted(DATASET.glob(stem+"-stage*.json")):
        meta = json.loads(sidecar.read_text(encoding="utf8"))
        assert meta["title"] == 14 and meta["difficulty"] == "Normal" and meta["shot"].replace(" ","") == "ReimuA"
        assert meta["replay"] == stem+".rpy" and (stem not in TRAIN or meta["split"] == "train")
        path = sidecar.with_suffix(".npz")
        bullets, context, _ = read_shard((path, dict(meta, samples=meta["pairs"])), motion_history=True,stage_clock=True)
        features, target, baseline, fitted = [], [], [], []
        for start in range(0,len(context),256):
            obs = {"bullets":torch.from_numpy(bullets[start:start+256]),"context":torch.from_numpy(context[start:start+256])}
            h = source.mlp_extractor.forward_actor(source.extract_features(obs))
            original = source.action_net(h)
            teacher_h = teacher.mlp_extractor.forward_actor(teacher.extract_features(obs))
            target.append(bomb_odds(teacher.action_net(teacher_h)))
            baseline.append(bomb_odds(original))
            if candidate is not None:
                cand_h = candidate.mlp_extractor.forward_actor(candidate.extract_features(obs))
                logits = candidate.action_net(cand_h)
                assert torch.equal(logits[:,:-2], original[:,:-2]), "A non-Bomb logit changed."
                fitted.append(bomb_odds(logits))
            else:
                features.append(h)
        rows.append(dict(stem=stem, stage=meta["stage"], player=meta["player"], split=meta["split"], pairs=meta["pairs"],
                         npz_sha256=digest(path), replay_sha256=meta["sha256"], x=torch.cat(features) if features else None,
                         target=torch.cat(target).double(), base=torch.cat(baseline).double(),
                         fitted=torch.cat(fitted).double() if fitted else None))
    if not rows:
        raise ValueError("Missing credit "+stem)
    return rows


def fit(rows):
    x = torch.cat([r["x"] for r in rows]).double()
    target = torch.cat([r["target"] for r in rows])
    base = torch.cat([r["base"] for r in rows])
    opening = torch.cat([torch.full((r["pairs"],),r["stage"] == 1) for r in rows])
    weight = torch.zeros_like(target)
    # Stage 1 and later stages receive equal mass. Within each, half is uniform
    # and half is weighted by Bernoulli Fisher information to emphasize teachable Bomb states.
    for group in (opening, ~opening):
        p = target[group].sigmoid()
        information = p*(1-p)
        weight[group] = .25/group.sum() + .25*information/information.sum()
    mu = weight @ x
    centered = x-mu
    y = target-base
    mean_y = weight@y
    gram = centered.T@(weight[:,None]*centered)
    gram.diagonal().add_(RIDGE)
    w = torch.linalg.solve(gram, centered.T@(weight*(y-mean_y)))
    b = mean_y - mu@w
    target_mean = target[opening].sigmoid().mean()
    values = (base+x@w+b)[opening]
    low, high = -30., 30.
    for _ in range(64):
        shift = (low+high)/2
        if (values+shift).sigmoid().mean() < target_mean:
            low = shift
        else:
            high = shift
    return w,b+(low+high)/2,dict(stage1_training_mean_teacher_p=float(target_mean),
                                stage1_intercept_adjustment=float((low+high)/2))


def metrics(t,b,f):
    q = t.sigmoid()
    oracle = F.binary_cross_entropy_with_logits(t,q)
    threshold = torch.quantile(q, .9)
    masks = {"all":torch.ones_like(q,dtype=torch.bool),"teacher_top_decile":q>=threshold,"teacher_p_ge_0.05":q>=.05}
    calibration = {}
    for name,mask in masks.items():
        calibration[name] = {"count":int(mask.sum())}
        if mask.any():
            calibration[name].update({key:float(values[mask].mean()) for key,values in
                                     (("ppo211",q),("ppo397",b.sigmoid()),("distilled",f.sigmoid()))})
    ce_parent = F.binary_cross_entropy_with_logits(b,q)
    ce_fit = F.binary_cross_entropy_with_logits(f,q)
    return {"samples":len(t),"teacher_entropy":float(oracle),"parent_cross_entropy":float(ce_parent),
            "distilled_cross_entropy":float(ce_fit),"parent_kl":float(ce_parent-oracle),"distilled_kl":float(ce_fit-oracle),
            "fraction_kl_removed":float(1-(ce_fit-oracle)/(ce_parent-oracle)),
            "distilled_mean_absolute_probability_error":float((f.sigmoid()-q).abs().mean()),"calibration":calibration}


def passed(m):
    c = m["calibration"]
    return (m["fraction_kl_removed"]>=.8 and .8<=c["all"]["distilled"]/c["all"]["ppo211"]<=1.2
            and .75<=c["teacher_top_decile"]["distilled"]/c["teacher_top_decile"]["ppo211"]<=1.25)


class SpacesOnly(gym.Env):
    """Provide spaces for a PPO initialization check without simulation or training."""
    def __init__(self, policy):
        self.observation_space, self.action_space = policy.observation_space, policy.action_space


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    out=parser.parse_args().out
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    arms=load_arms([], [f"ppo211={PPO211}",f"ppo397={PPO397}"], TEMPLATE)
    assert all(arms[n].provenance["weights_sha256"]==v for n,v in HASHES.items())
    source,teacher = arms["ppo397"].policy, arms["ppo211"].policy
    train=[r for stem in TRAIN for r in read_credit(stem,source,teacher)]
    w,b,intercept = fit(train)
    model=deepcopy(source)
    with torch.no_grad():
        model.action_net.weight[-2].sub_(w.float()/2)
        model.action_net.weight[-1].add_(w.float()/2)
        model.action_net.bias[-2].sub_(b.float()/2)
        model.action_net.bias[-1].add_(b.float()/2)
    for name,value in source.state_dict().items():
        other=model.state_dict()[name]
        if name in ("action_net.weight","action_net.bias"):
            assert torch.equal(value[:-2],other[:-2])
        else:
            assert torch.equal(value,other)
    out.mkdir(parents=True,exist_ok=False)
    torch.save(model.state_dict(),out/"policy.pt")
    side={k:v for k,v in arms["ppo397"].provenance.items() if k in ("hidden","encoder","motion_history","retain_recent_motion","stage_clock","motion_spans","factorized_movement")}
    (out/"metrics.json").write_text(json.dumps(side,indent=2),encoding="utf8")
    loaded=load_arm(out/"policy.pt").policy
    hold=[r for stem in HOLD for r in read_credit(stem,source,teacher,loaded)]
    assert not ({r["player"].casefold() for r in train}&{r["player"].casefold() for r in hold})
    trained_metrics={}
    for label,selected in (("stage1",[r for r in train if r["stage"]==1]),("later",[r for r in train if r["stage"]>1])):
        target,base=(torch.cat([r[k] for r in selected]) for k in ("target","base"))
        with torch.inference_mode():
            logits=torch.cat([bomb_odds(loaded.action_net(r["x"])).double() for r in selected])
        trained_metrics[label]=metrics(target,base,logits)
    evaluated={f'{r["stem"]}_stage{r["stage"]}':metrics(r["target"],r["base"],r["fitted"]) for r in hold}
    stage1=[s for name,s in evaluated.items() if name.endswith("_stage1")]
    new_ppo=warm_start(loaded,SpacesOnly(loaded),211)
    assert new_ppo.num_timesteps == 0 and len(new_ppo.policy.optimizer.state)==0
    report={"status":"experimental single-network actor warm start", "source_sha256":HASHES,
            "candidate_sha256":digest(out/"policy.pt"),"script_sha256":digest(Path(__file__)),
            "config":{"readout":"weighted ridge on PPO211 minus PPO397 Bernoulli Bomb log odds, centered two-row update",
            "ridge":RIDGE,"sample_weight":"50% stage1, 50% later; each half uniform and half teacher p*(1-p)",
            "calibration":"training Stage 1 mean probability intercept, after fixed regression; no held-out fitting or selection",
            "train_credit_ids":TRAIN,"held_out_credit_ids":HOLD,"sealed_test_used":False,"native_game_used":False,
            "held_out_note":"Toshi is historical development data; Aria is in an expanded SFT training pool but excluded here and from the original parent."},
            "calibration_fit":intercept,"training_fit":trained_metrics,"held_out":evaluated,
            "gates":"both held-out Stage 1 credits independently: remove >=80% parent-to-teacher KL, mean within ±20% of teacher, top teacher decile mean within ±25%. Native play is a separate gate.",
            "passes_offline_stage1_gate":all(passed(m) for m in stage1),
            "unchanged_non_bomb_state_tensors":True,"exact_non_bomb_logits_on_heldout_rows":sum(r["pairs"] for r in hold),
            "warm_start_check":{"fresh_ppo_steps":new_ppo.num_timesteps,"optimizer_states":len(new_ppo.policy.optimizer.state),
                                "actor_exact_after_critic_reset":True, "method":"scripts.train_full_policy.warm_start"},
            "data":[{k:v for k,v in r.items() if k not in ("x","target","base","fitted")} for r in train+hold]}
    (out/"distillation.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf8")
    if Path(__file__).resolve()!=(out/Path(__file__).name).resolve():
        shutil.copyfile(__file__,out/Path(__file__).name)
    print(json.dumps({"gate":report["passes_offline_stage1_gate"],"stage1":stage1,"output":str(out)}))


if __name__=="__main__":
    main()
