"""Fit matched TH08 advantage-weighted and uniform imitation policies."""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.th08_awr_human import load_human
from scripts.train_five_original_games import freeze
from scripts.train_shared_full_ppo import calculate_gae
from scripts.train_th08_camera_hazard import ENCODER, admit, read
from scripts.compare_th08_native_resolution import select
from scripts.train_th08_normal_current_ppo import CurrentActorCritic, WIDTHS
from scripts.train_th08_progress_lambda_one import PARENT
from scripts.train_th08_progress_pair import EXPECTED_ACTOR, GAMMA, trajectory


SEEDS = (808401, 808402, 808403)
UPDATES = 2000
HALF = 64
ACTOR_RATE = 3e-5
VALUE_STEPS = 800
LAMBDA = .985


@dataclass
class Original:
    seed: int
    cohort: str
    features: torch.Tensor
    actions: torch.Tensor
    behavior: torch.Tensor
    reward: np.ndarray


class Readout(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.prior = deepcopy(actor.prior)
        self.residual = deepcopy(actor.readout)

    def forward(self, x):
        prior = self.prior(x).split(WIDTHS, 1)
        return self.residual(x.unsqueeze(1).expand(-1,4,-1), prior)


def suffix_sum(values, discount):
    result = np.empty(len(values), dtype=np.float32)
    current = 0.
    for i in range(len(values)-1,-1,-1):
        current = float(values[i]) + discount * current
        result[i] = current
    return result


def ordered(records, namespace):
    return sorted(records, key=lambda r: hashlib.sha256(f"{namespace}:{r[1]['seed']}".encode()).digest())


def sources(data):
    manifest = data / "th08-highres-native/qualified-train.json"
    final, _ = select(manifest)
    original = admit(data) + [(manifest, row) for row in final]
    holdout = []
    groups = [[] for _ in range(5)]
    for offset, policy in enumerate((EXPECTED_ACTOR, PARENT, ENCODER)):
        entries = [row for row in original if row[1]["policy_sha256"] == policy]
        order = ordered(entries, "th08-awr-v1")
        holdout.extend(order[:3])
        for i,r in enumerate(order[3:]):
            groups[(i+offset) % 5].append(r)
    return groups, holdout


def model_from(path, sha):
    if digest(path) != sha:
        raise ValueError(f"Original behavior artifact changed: {path.name}")
    model = CurrentActorCritic().eval()
    model.load_state_dict(torch.jit.load(str(path), map_location="cpu").state_dict())
    return model


def load_original(entry, behavior, encoder, device):
    manifest, row = entry
    sha = row["policy_sha256"]
    common, rewards, source = trajectory(manifest, row, behavior[sha], "cpu", actor=sha,
        min_exact=.99 if sha == EXPECTED_ACTOR else .995, require_zero_value=sha == EXPECTED_ACTOR)
    frames, action, _, heads, _ = common
    with torch.no_grad():
        features = torch.cat([encoder(batch.to(device).float().div_(255)).cpu() for batch in frames.split(128)])
    print(f"Verified original {row['seed']}: {len(frames)} decisions, recomputed behavior logp max error {source['max_logp_error']:.6g}", flush=True)
    return Original(row["seed"],sha,features,action,heads,rewards[1].reward.astype(np.float32))


def crossfit(groups, device):
    features, actions, population = [], [], []
    critic_stats = []
    for fold, test in enumerate(groups):
        old = [credit for i,group in enumerate(groups) if i != fold for credit in group]
        torch.manual_seed(SEEDS[0]+fold)
        x = torch.cat([r.features for r in old]).to(device)
        y = torch.from_numpy(np.concatenate([suffix_sum(r.reward,GAMMA) for r in old])).to(device)
        value = nn.Sequential(nn.Linear(x.shape[1],64), nn.SiLU(), nn.Linear(64,1)).to(device)
        opt = torch.optim.AdamW(value.parameters(),lr=.001,weight_decay=.003)
        for step in range(1, VALUE_STEPS+1):
            ids = torch.randint(len(y),(256,),device=device)
            loss = F.mse_loss(value(x[ids]).flatten(),y[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        square = count = 0
        for credit in test:
            with torch.no_grad():
                prediction = value(credit.features.to(device)).flatten().cpu().numpy()
            returns = suffix_sum(credit.reward,GAMMA)
            square += float(np.square(prediction-returns).sum())
            count += len(returns)
            advantages = calculate_gae(credit.reward,prediction,GAMMA,LAMBDA)
            if not np.isfinite(advantages).all():
                raise ValueError(f"Nonfinite out-of-fold advantages for seed {credit.seed}")
            features.append(credit.features)
            actions.append(credit.actions)
            population.append(advantages)
        stat = dict(fold=fold,fit_credits=len(old),test_credits=len(test),final_minibatch_mse=float(loss.detach()),
                    out_of_fold_mc_rmse=(square/count)**.5,optimizer_updates=VALUE_STEPS)
        critic_stats.append(stat)
        print("Crossfit critic",stat,flush=True)
    return torch.cat(features),torch.cat(actions),np.concatenate(population),critic_stats


def weight_advantage(advantage):
    scale = max(float(np.std(advantage)),1e-5)
    best = None
    for beta in np.geomspace(scale/50.,scale*10.,240):
        raw = np.exp(np.minimum(advantage/beta,np.log(20.)))
        ess = float(raw.sum()**2/(len(raw)*np.square(raw).sum()))
        if best is None or abs(ess-.30)<abs(best[0]-.30):
            best = ess,float(beta),raw
    ess,beta,raw = best
    return torch.from_numpy((raw/raw.mean()).astype(np.float32)),ess,beta


def loss_rows(heads, action):
    return sum(F.cross_entropy(logits,action[:,i],reduction="none") for i,logits in enumerate(heads))


def human_metrics(head, data, device):
    features,target,switch = data[:3]
    with torch.no_grad():
        heads = [torch.cat(parts) for parts in zip(*(head(x.to(device)) for x in features.split(256)))]
        target,switch = target.to(device),switch.to(device)
        correct = heads[0].argmax(1) == target[:,0]
        return dict(rows=len(target),moves=float(correct.float().mean()), switches=int(switch.sum()),
            switch_accuracy=float(correct[switch].float().mean()),joint_nll=float(loss_rows(heads,target).mean()),
            four_accuracy=[float((p.argmax(1)==target[:,i]).float().mean()) for i,p in enumerate(heads)])


def behavior_metrics(head, records, device):
    results = []
    with torch.no_grad():
        for record in records:
            per_step = []
            for x, previous in zip(record.features.split(256),record.behavior.split(256)):
                prior = previous.to(device).split(WIDTHS,1)
                proposed = head(x.to(device))
                per_step.append(sum((p.softmax(1)*(p.log_softmax(1)-q.log_softmax(1))).sum(1)
                                    for p,q in zip(prior,proposed)).cpu())
            values = torch.cat(per_step)
            results.append(dict(seed=record.seed,decisions=len(values),mean_kl=float(values.mean()),cohort=record.cohort))
    return dict(credits=len(records), mean_credit_kl=float(np.mean([r["mean_kl"] for r in results])),rows=results)


def optimize_pair(seed, candidate, human, own, original_holdout, native_weights, parent, target, device):
    torch.manual_seed(seed)
    head = Readout(candidate.actor)
    models = {name: deepcopy(head).to(device) for name in ("uniform","awr")}
    optimizers = {k: torch.optim.AdamW(m.parameters(),lr=ACTOR_RATE,weight_decay=.003) for k,m in models.items()}
    generator = torch.Generator(device=device).manual_seed(seed)
    human_x,human_y = (x.to(device) for x in human["train"][:2])
    native_x,native_y = (x.to(device) for x in own)
    weights = native_weights.to(device)
    log = []
    for step in range(1,UPDATES+1):
        h = torch.randint(len(human_y),(HALF,),generator=generator,device=device)
        r = torch.randint(len(native_y),(HALF,),generator=generator,device=device)
        features = torch.cat((human_x[h],native_x[r]))
        labels = torch.cat((human_y[h],native_y[r]))
        row = dict(step=step)
        for name,model in models.items():
            losses = loss_rows(model(features),labels)
            own_loss = losses[HALF:] if name == "uniform" else losses[HALF:]*weights[r]
            objective = .5*(losses[:HALF].mean()+own_loss.mean())
            opt = optimizers[name]
            opt.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            if step == 1 or step % 200 == 0:
                row[name] = dict(loss=float(objective.detach()),human=float(losses[:HALF].mean().detach()),
                                 original=float(own_loss.mean().detach()))
        if len(row) > 1:
            log.append(row)
            print(f"Seed {seed}: {row}",flush=True)
    result = dict(seed=seed,optimizer_log=log,arms={})
    for name,head in models.items():
        metrics = human_metrics(head,human["held"],device)
        behavior = behavior_metrics(head,original_holdout,device)
        eligible = (metrics["moves"] >= parent["moves"]-.005
                    and metrics["switch_accuracy"] >= parent["switch_accuracy"]-.005
                    and behavior["mean_credit_kl"] <= .05)
        head = head.cpu().eval()
        deployed = deepcopy(candidate).cpu().eval()
        deployed.actor.prior.load_state_dict(head.prior.state_dict())
        deployed.actor.readout.load_state_dict(head.residual.state_dict())
        example = torch.rand(2,2,112,96,generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            frozen = torch.jit.trace(deployed,example)
            for actual,expected in zip(frozen(example)[:4],head(deployed.actor.features(example))):
                torch.testing.assert_close(actual,expected)
        path = target/f"seed{seed}-{name}.ts"
        frozen.save(str(path))
        result["arms"][name] = dict(human_held=metrics,original_held_behavior=behavior,gate=bool(eligible),policy_sha256=digest(path))
    return result


def prepare(data, target, device):
    groups, test = sources(data)
    if sum(map(len,groups)) != 63 or len(test) != 9:
        raise ValueError("Expected 63 whole train credits and nine native behavior-diagnostic credits")
    target.mkdir(parents=True,exist_ok=False)
    all_sources = [row for group in groups for row in group] + test
    freeze(target/"protocol.json",dict(seed_order=list(SEEDS),parent=ENCODER,optimizer_steps=UPDATES,batch=[HALF,HALF],actor_lr=ACTOR_RATE,
        native_manifests={p.name:digest(p) for p,_ in all_sources},
        native_fit_folds=[[r["seed"] for _,r in group]for group in groups],
        native_behavior_diagnostic=[r["seed"]for _,r in test],
        native_holdout_caveat="Nine credits excluded from this fit. The parent encoder has past exposure to some of this population. Use entirely new original credits for any live comparison.",
        human="Five signed original Ghost Normal players with distinct train and held roles. Final held development check has known prior experiment exposure.",
        objective="50% human plain four-head CE and 50% original own-policy four-head CE. Matched initialization and batches. The only arm difference is exp(A/beta) capped at 20 and normalized to mean one.",
        critic="Fresh readout per fold fits terminal Monte Carlo progress-plus-spare returns on the other four whole-credit folds. Held predictions supply GAE(lambda) advantages.",
        critic_steps=VALUE_STEPS,gamma=GAMMA,lambda_return=LAMBDA,ess_target=.30,min_ess=.20,
        deployment="Camera-only grayscale with previous camera difference. Frozen shared encoder; both learned action modules are optimized. All four categorical heads sampled. No privileged input.",
        evaluation="Fixed first paired seed. Both arms held human movement and switch accuracy within .5 percentage points of parent, mean held original behavior KL <= .05, ESS >=20%. Report all three final seeds.",
        native_release="After gate and review, three arms each six fresh original credits in balanced order: weighted, uniform and identical parent. Keep unchanged original controls and verified complete credits."))
    behavior = {
        EXPECTED_ACTOR:model_from(data/"th08-progress-reward/common-parent/policy.ts",EXPECTED_ACTOR),
        PARENT:model_from(data/"th08-progress-reward/first-paired-update/progress-plus-spares/policy.ts",PARENT),
        ENCODER:model_from(data/"th08-progress-lambda1-round2/policy/progress-plus-spares/policy.ts",ENCODER),
    }
    candidate = behavior[ENCODER]
    encoder = deepcopy(candidate.actor.features).eval().to(device)
    groups = [[load_original(entry,behavior,encoder,device) for entry in group] for group in groups]
    test = [load_original(entry,behavior,encoder,device) for entry in test]
    original_count = sum(len(r.actions)for group in groups for r in group)+sum(len(r.actions)for r in test)
    if original_count != 66428:
        raise ValueError(f"Original decision corpus count changed: {original_count}")
    return candidate, encoder, groups, test, original_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = args.out
    candidate,encoder,groups,test,original_count = prepare(args.data,target,device)
    x,y,adv,critic_stats = crossfit(groups,device)
    weights,ess,beta = weight_advantage(adv)
    print(f"Native advantage: {len(y)} fit decisions, beta={beta:.6f}, relative ESS={ess:.5f}",flush=True)
    human = load_human(args.data,encoder,device)
    parent = human_metrics(Readout(candidate.actor).to(device),human["held"],device)
    stats = dict(total_original_decisions=original_count, fit_original_decisions=len(y),
        held_original_decisions=sum(len(r.actions)for r in test),human_train_decisions=len(human["train"][1]),
        human_held_decisions=len(human["held"][1]),human_sources=human.get("_sources",{}),parent_held=parent,
        critics=critic_stats,beta=beta,effective_sample_fraction=ess,weight_quantiles=np.quantile(weights.numpy(),[0,.1,.5,.9,1]).tolist())
    freeze(target/"fit-inputs.json",stats)
    results = []
    for seed in SEEDS:
        row = optimize_pair(seed,candidate,human,(x,y),test,weights,parent,target,device)
        results.append(row)
        freeze(target/f"seed{seed}-results.json",row)
        print(f"Completed paired seed {seed}: {[(k,r['gate']) for k,r in row['arms'].items()]}",flush=True)
    first = results[0]["arms"]
    release = ess >= .20 and all(row["gate"]for row in first.values())
    freeze(target/"report.json",dict(protocol_sha256=digest(target/"protocol.json"),**stats,device=str(device),
                 fixed_seed=SEEDS[0],offline_release_gate=release,results=results))
    print("AWR offline comparison complete; gate:",release,flush=True)


if __name__ == "__main__":
    main()
