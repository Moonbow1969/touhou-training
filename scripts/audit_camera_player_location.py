"""Measure frozen player readout drift on the same chronologically aligned retail replay frames."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scripts.train_human_camera_policy import load_replay
from scripts.train_replay_camera_encoder import CameraPolicy,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',nargs='+',required=True,type=Path)
    p.add_argument('--held',nargs='+',required=True,type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
    rows=[load_replay(path,include_player_position=True) for path in a.held];report={}
    with torch.inference_mode():
        for file in a.weights:
            mod=CameraPolicy().to(device).eval();mod.load_state_dict(torch.load(file,weights_only=True,map_location=device))
            results={}
            for row in rows:
                target=row['player_xy'];valid=(target[:,0]>=0)&(target[:,0]<96)&(target[:,1]>=0)&(target[:,1]<112)
                raw=torch.from_numpy(row['raw'][valid]);coords=target[valid];indices=[]
                for chunk in raw.split(64):
                    z=mod.player(mod.spatial(chunk.to(device).float()/255)).flatten(1).argmax(1).cpu().numpy()
                    indices.append(z)
                z=np.concatenate(indices);xy=np.column_stack((z%48,z//48))*2
                error=np.linalg.norm((coords-xy)*4,axis=1)
                results[row['name']]=dict(group=row['group'],n=len(error),within_16_original_pixels=float(np.mean(error<=16)),
                    median_error_original_pixels=float(np.median(error)),mean_error_original_pixels=float(error.mean()),camera_sha256=row['sha256'])
            report[file.parent.name]=dict(model_sha256=sha(file),held=results)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
