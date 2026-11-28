"""Score independent restored-state held continuations using native elapsed ticks."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PARENT = "lambda1-parent"
MIXED = "mixed-ppo"


def checked_rows(outcomes, study):
    rows = outcomes["attempts"]
    reg = json.loads((study / "registration.json").read_bytes())
    if not outcomes["complete"]:
        gate = json.loads((study / "intake-progress.json").read_bytes())
        if gate["continue"] or len(rows) != gate["physical"]:
            raise ValueError("Scoring requires completion or a registered terminal technical stop")
    elif len(rows) != len(reg["rows"]):
        raise ValueError("A completed cohort must include every registered physical disposition")
    assert outcomes["registration_sha256"] == hashlib.sha256((study / "registration.json").read_bytes()).hexdigest()
    assert [r["seed"] for r in rows] == [r["seed"] for r in reg["rows"][:len(rows)]]
    for row, assignment in zip(rows, reg["rows"]):
        assert all(row[k] == assignment[k] for k in ("seed","arm","role","policy_sha256"))
        if "receipt" not in row:
            assert not row["physical_endpoint_verified"]
            continue
        data = (study / row["receipt"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["receipt_sha256"]
        signed = json.loads(data)
        assert signed["outcome"] == row["endpoint"] and signed["elapsed_native_ticks"] == row["ticks"]
        assert signed["boundary_tick"] - signed["anchor_tick"] == row["ticks"] > 0
        if row["right_censored"]:
            assert row["endpoint"] == "native_tick_cutoff" and row["ticks"] >= 3000
        else:
            assert row["first_observed_spare_loss"] is not None or row["endpoint"] == "game_over"
    return rows


def survival(times, event):
    survival_probability = 1.0
    points = []
    median = None
    for t in np.unique(times):
        at_risk = int((times >= t).sum())
        deaths = int(((times == t) & event).sum())
        censored = int(((times == t) & ~event).sum())
        survival_probability *= 1 - deaths / at_risk
        if median is None and survival_probability <= .5 + 8*np.finfo(float).eps:
            median = int(t)
        points.append(dict(tick=int(t), at_risk=at_risk, first_losses=deaths, right_censored=censored,
                           survival=survival_probability))
    return median, points


def logrank(t0,d0,t1,d1):
    observed_minus_expected = 0.
    variance = 0.
    for t in np.unique(np.r_[t0[d0], t1[d1]]):
        y0,y1 = int((t0>=t).sum()),int((t1>=t).sum())
        deaths0,deaths1 = int(((t0==t)&d0).sum()),int(((t1==t)&d1).sum())
        total,events = y0+y1,deaths0+deaths1
        observed_minus_expected += deaths1 - y1*events/total
        if total>1:
            variance += y0*y1*events*(total-events)/(total*total*(total-1))
    if not variance:
        return dict(chi_squared=None,p_two_sided=None)
    chi_squared = observed_minus_expected**2/variance
    return dict(chi_squared=chi_squared,df=1,p_two_sided=math.erfc(math.sqrt(chi_squared/2)))


def restricted_mean(points, horizon):
    previous=0
    probability=1.
    area=0.
    for row in points:
        stop=min(row["tick"],horizon)
        area+=(stop-previous)*probability
        previous=stop
        if stop==horizon:
            return area
        probability=row["survival"]
    return area+(horizon-previous)*probability if probability==0 or previous>=horizon else None


def compare(rows, rng_seed, iterations):
    groups = {}
    vectors = {}
    for arm in (PARENT,MIXED):
        subset=[r for r in rows if r["arm"]==arm]
        if not subset:
            return {"detail":"One arm has zero technically interpretable endpoints"}
        t=np.array([r["ticks"] for r in subset],dtype=np.int64)
        d=np.array([not r["right_censored"] for r in subset],dtype=bool)
        median,points=survival(t,d)
        vectors[arm]=(t,d)
        loss_bombs=[r["first_observed_spare_loss"]["bombs_before"] for r in subset if r["first_observed_spare_loss"]]
        groups[arm]=dict(n=len(subset),first_spare_losses=int(d.sum()),right_censored=int((~d).sum()),native_ticks_at_risk=int(t.sum()),
                         first_losses_per_1000_native_ticks=1000*int(d.sum())/int(t.sum()),median_km_ticks=median,
                         restricted_mean_to_3000_ticks=restricted_mean(points,3000),
                         actual_game_overs=sum(r["endpoint"]=="game_over" for r in subset),kaplan_meier=points,
                         pre_loss_bomb_positive=sum(b>0 for b in loss_bombs),pre_loss_bomb_observed=len(loss_bombs),
                         n_measured_native_bomb_spends=sum(r["native_observed_bomb_spend_events"] for r in subset),
                         same_life_bomb_units_spent=sum(r["same_life_observed_bomb_units_spent"] for r in subset),
                         raw_x_requested_decisions=sum(r["policy_x_requested_decisions"] for r in subset),
                         boundary_stock_counts=dict(Counter(str(r["end_stock"])for r in subset)),
                         fully_consumed_commands=sum(r["consumed"] for r in subset),decisions=sum(r["decisions"] for r in subset),
                         exact_four_tick_windows=sum(r["exact_windows"] for r in subset),comparable_windows=sum(r["comparable_windows"] for r in subset))
    t0,d0=vectors[PARENT];t1,d1=vectors[MIXED]
    ratio=float((d1.sum()*t0.sum())/(d0.sum()*t1.sum())) if d0.sum() else None
    rng=np.random.default_rng(rng_seed)
    parts=[]
    for t,d in (vectors[PARENT],vectors[MIXED]):
        draw=rng.integers(0,len(t),size=(iterations,len(t)))
        parts.append(d[draw].sum(1)/t[draw].sum(1))
    mask=parts[0]>0
    draws=parts[1][mask]/parts[0][mask]
    return dict(arms=groups,candidate_parent_loss_rate_ratio=ratio,
                bootstrap_percentile_95=np.quantile(draws,[.025,.975]).tolist() if draws.size else None,
                bootstrap_valid_draws=int(mask.sum()),bootstrap_requested_draws=iterations,bootstrap_rng_seed=rng_seed,
                logrank=logrank(t0,d0,t1,d1))


def run(path,dest):
    outcome=json.loads(path.read_bytes())
    rows=checked_rows(outcome,path.parent)
    physical=[r for r in rows if r["physical_endpoint_verified"]]
    strict=[r for r in physical if r["qualified"]]
    primary=compare(strict,80965500,20000)
    ratio=primary.get("candidate_parent_loss_rate_ratio")
    if ratio is None:
        disposition="insufficient_qualified_comparison"
    elif ratio>=1:
        disposition="stop_at_current_scale"
    elif ratio<=.75:
        disposition="eligible_for_separate_normal_game_start_evaluation"
    else:
        disposition="below_preregistered_promotion_threshold"
    if len(rows)-len(strict)>4 or not outcome["complete"]:
        disposition="technical_control_gate_failed"
    summary=dict(physical_attempts=len(rows),physical_known_endpoints=len(physical),qualified_for_comparison=len(strict),
                 excluded_seeds=[r["seed"] for r in rows if not r["qualified"]],
                 unknown_native_endpoint_seeds=[r["seed"] for r in rows if not r["physical_endpoint_verified"]],
                 primary_qualified=primary,all_observed_physical_sensitivity=compare(physical,80965500,20000),
                 clock="Original elapsed Stage 1 ticks from the common frozen state to first observed spare loss or right censor.",
                 exposure="Each independent controller release contributes one at-risk interval from tick 2201 and at most one first loss.",
                 uncertainty="95% independent whole-fragment bootstrap within each arm. Log-rank compares survival, with ties using the finite-risk hypergeometric variance.",
                 disposition=disposition,outcomes_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    dest.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("outcomes",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    run(args.outcomes,args.output)
