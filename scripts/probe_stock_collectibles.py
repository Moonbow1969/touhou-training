"""Check item observations and fragment inventories without sending any game inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter,sleep

import numpy as np
import psutil
from environment.structured import StateReader,load_parakit_offsets
from environment.th14_items import ItemReader
from scripts.checkpoint_storage import digest
from scripts.th14_inventory_reward import read_inventory,potential
from scripts.train_full_policy import publish_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=60)
    parser.add_argument("--interval",type=float,default=.5)
    args=parser.parse_args()
    if not 1<=args.samples<=1200 or not .05<=args.interval<=5:
        parser.error("Choose a bounded passive sampling interval and count.")
    from environment import interface as game
    original=digest(Path(psutil.Process(game._game_pid).exe()))
    if original!="4609e701b677b8317e43be92041f0c7075a292182848a7c11215c3a226034f90":
        raise ValueError("Unexpected original game version.")
    parakit=Path(__file__).resolve().parents[2]/"parakit"
    reader=StateReader(game,load_parakit_offsets(parakit),96)
    items=ItemReader(reader)
    counts=Counter();duration=[];examples=[];valid=nonempty=unknown=0
    seen_frames=set()
    for _ in range(args.samples):
        tick=game.read_game_val("global_timer")
        start=perf_counter()
        if game.read_game_val("game_state")==2:
            stock=read_inventory(game)
            found,unmapped=items.visible()
            xy=np.array([game.read_game_val("f_player_pos_x"),game.read_game_val("f_player_pos_y")],dtype=np.float32)
            end_tick=game.read_game_val("global_timer")
            duration.append(1000*(perf_counter()-start))
            if tick==end_tick and tick not in seen_frames and stock[0]>=0:
                seen_frames.add(tick);valid+=1;nonempty+=int(len(found)>0);unknown+=unmapped
                potential(stock)
                tokens=items.tokens(found,xy)
                if int(tokens[:,-1].sum())!=min(24,len(found)):
                    raise ValueError("Item observation mask disagrees with native entity count.")
                counts.update([items.types[int(kind)] for kind in found[:,1]])
                if len(examples)<12 and len(found):
                    examples.append(dict(tick=tick,inventory=stock,player=xy.tolist(),item=found[0].tolist()))
        sleep(args.interval)
    if not duration:
        raise RuntimeError("No gameplay item scan completed.")
    result=dict(game_sha256=original,samples=args.samples,unique_single_tick_scans=valid,
        nonempty_scans=nonempty,unmapped_active_slot_observations=unknown,observed_type_occurrences=counts,
        pool_bytes=items.layout.zItem_len*items.layout.zItemManager_array_len,mean_ms=float(np.mean(duration)),
        p95_ms=float(np.quantile(duration,.95)),tokens=24,examples=examples,
        item_columns=["slot","type","state","x","y","vx","vy","age"],
        scope="Read-only snapshots. Repeated entity sightings are not unique drops or successful pickups.")
    publish_json(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k!="examples"}))


if __name__=="__main__":
    main()
