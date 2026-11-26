"""Compare actual and exact-reference player positions against the same pre-hit stock bullets."""

import argparse
import json
from pathlib import Path
import numpy as np

from scripts.checkpoint_storage import digest
from scripts.train_full_policy import publish_json


def margins(world, velocity, radius, positions):
    # First update moves projectiles before the next player movement.
    dt = np.arange(1, 5)[:, None, None]
    if len(world) == 0:
        return None
    return float(np.min(np.linalg.norm(world + dt*velocity - positions[:, None, :], axis=-1) - radius - 1.))


def analyze(credit, audit_path, route_path):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if digest(credit) != audit["credit_sha256"]:
        raise ValueError("The phase reconstruction belongs to another native credit.")
    rows = []
    with np.load(credit, allow_pickle=False) as z, np.load(route_path, allow_pickle=False) as route:
        col = {v:i for i,v in enumerate(json.loads(str(z["metadata_json"]))["native_field_order"])}
        before, after = z["native_before"], z["native_after"]
        fields, context, actions = z["bullets"], z["context"], z["action"]
        for hit in audit["lethal_hits"]:
            stage, frame = hit["stage"], hit["native_hit_frame"]
            matching = np.flatnonzero((before[:,col["stage"]] == stage) & (before[:,col["stage_frame"]] == frame))
            if len(matching) != 1:
                raise ValueError("Ambiguous native decision.")
            i = matching[0]
            verified = (before[i,col["player_state"]] == 1 and after[i,col["player_state"]] == 4
                        and after[i,col["global_tick"]]-before[i,col["global_tick"]] == 4)
            position = context[i,:2] * [192,448]
            bullet = fields[i]
            bullet = bullet[(bullet[:,5] == 1) & (bullet[:,6] == 1)]
            world = bullet[:,:2] * [384,448] + position
            velocity, radius = bullet[:,2:4]*16, bullet[:,4]*32
            direction = int(actions[i,0]) + int(actions[i,1])*9
            deltas = route["action_deltas"][direction]/4
            actual_plan = np.clip(position + np.arange(4)[:,None]*deltas, *route["bounds"])
            reference_frame = hit["reference_frame"]
            coords = route[f"positions_{stage}"]
            clock = route[f"frame_{stage}"]
            reference = np.stack([np.interp(reference_frame + np.arange(4), clock, axis) for axis in coords.T], 1)
            values = dict(native_plan=margins(world,velocity,radius,actual_plan),
                          exact_reference=margins(world,velocity,radius,reference),
                          native_hold=margins(world,velocity,radius,np.tile(position,(4,1))))
            rows.append(dict(stage=stage, frame=frame, reference_frame=reference_frame, verified=bool(verified),
                             position_error=hit["track_error"], active_retained=len(bullet), margins=values))
    return dict(trial=audit["trial"], credit_sha256=audit["credit_sha256"], rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, nargs=2, action="append", required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    publish_json(args.out, dict(measurement="Signed surface clearance in game units for visible active circular bullets. "
        "Four counterfactual ticks use fixed pre-hit bullet velocities, fitted action deltas and a 1-unit player radius. "
        "No enemy bodies, curved laser paths, spawned bullets, or response of the game to alternative actions.",
        route_sha256=digest(args.route), credits=[analyze(*pair,args.route) for pair in args.pair]))
