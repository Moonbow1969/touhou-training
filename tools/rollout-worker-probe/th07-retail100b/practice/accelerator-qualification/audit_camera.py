"""Compare independent X root images, native camera packets and fixed Gamer HUD glyphs."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from PIL import Image

def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()

def reduce_camera(rgb,x=32,y=16):
    frame = rgb[y:y+448,x:x+384].astype(np.int32)
    weights = np.array([19595,38470,7471], dtype=np.int32)
    gray = (frame @ weights + 32768) >> 16
    rows = (gray.reshape(112,4,96,4).sum(axis=3)+2)//4
    return ((rows.sum(axis=1)+2)//4).astype(np.uint8)

def hud_iou(baseline,photo,x,y,w,h,dx,dy):
    a=baseline[y:y+h,x:x+w].min(axis=2)>180
    b=photo[y+dy:y+dy+h,x+dx:x+dx+w].min(axis=2)>180
    return int((a&b).sum())/max(1,int((a|b).sum()))

def main():
    cli=argparse.ArgumentParser()
    for key in ("case","gamer","fragment","cgroup","out"):
        cli.add_argument("--"+key,required=True,type=Path)
    args=cli.parse_args();case=args.case
    reference=np.array(Image.open(args.gamer).convert("RGB"));receipt=json.loads(args.fragment.read_bytes())
    assert receipt["qualified"] and receipt["role"]=="engineering" and receipt["native_elapsed_ticks"]==256
    assert receipt["exact_native_windows"]==receipt["original_windows"]==receipt["action_masks_with_native_consumption"]==64
    anchors={"maximum_power":(495,144,46,14),"two_spares":(497,98,26,12),"graze_zero":(495,159,13,15),"point_display":(497,175,54,15)}
    shots=[]
    with np.load(case/"policy-trace.npz",allow_pickle=False) as z:
        for i in (0,16,48):
            path=case/f"readback-root-{i:02d}.png"
            image=np.array(Image.open(path).convert("RGB"))
            bounds=dict(line.split("=") for line in (case/f"readback-window-{i:02d}.txt").read_text().splitlines())
            assert [int(bounds[k]) for k in ("X","Y","WIDTH","HEIGHT")]==[0,0,640,480]
            actual=z["frames"][i]
            delta=reduce_camera(image).astype(int)-actual.astype(int)
            assert np.max(np.abs(delta))==0
            background={}
            for name,rect in anchors.items():
                choices=[(hud_iou(reference,image,*rect,dx,dy),dx,dy) for dx in range(-4,5) for dy in range(-2,3)]
                top=max(x[0] for x in choices);coords=[(dx,dy) for value,dx,dy in choices if value==top]
                assert coords==[(0,0)] and top==1
                background[name]={"best_offset_xy":list(coords[0]),"iou":top}
            shots.append(dict(index=i,tick=int(z["native_tick"][i]),original_png_sha256=sha(path),
                root_xywh=[int(bounds[k]) for k in ("X","Y","WIDTH","HEIGHT")],x_server_size=[image.shape[1],image.shape[0]],
                mismatched_pixels=int(np.count_nonzero(delta)),
                wrong_horizontal_offsets_mismatches={str(dx):int(np.count_nonzero(reduce_camera(image,32+dx)!=actual)) for dx in (-4,4)},
                hud=background))
    c=[json.loads(x) for x in args.cgroup.read_text().splitlines()]
    a,b=c[0],c[-1]
    def stat(r):return {k:int(v) for k,v in (s.split() for s in r["cpu"].splitlines())}
    before,after=stat(a),stat(b);duration=(b["ns"]-a["ns"])/1e9
    metadata=json.loads((case/"qualification.json").read_bytes())
    d=next(csv.DictReader((case/"observer-distribution.csv").open()))
    titles=[json.loads(x) for x in (case/"engineering-title-scores.jsonl").read_text().splitlines()]
    win=max(titles[-1]["scores"],key=titles[-1]["scores"].get)
    h={"seed":2176503,"qualified":True,"real_original_attempts":1,"engineering_only":True,"eligible_for_training_or_frozen_held":False,
        "purpose":"Test the title-only alignment allowance on one independent Accelerator practice start with direct pixel/HUD/input proof.",
        "reference_gamer_sha256":sha(args.gamer),"initial_title_iou_at_signed_x":titles[-1]["scores"]["0"],"title_best_dx":int(win),"title_best_iou":titles[-1]["scores"][win],
        "independent_root_samples":shots,"gameplay_hud_rects_xywh":anchors,"stage_menu_stock_ious":metadata["stage_menu_glyph_iou"],
        "strict_receipt":receipt,"native_observer_p999_ms":int(d["p999_upper_ns"])/1e6,"native_observer_max_ms":int(d["max_ns"])/1e6,
        "native_original_first_last_ticks":[int(receipt["native_stage_brackets"]["1"][k]) for k in ("first_tick","last_tick")],
        "whole_reset_validation_s":metadata["total_setup_validation_seconds"],"cgroup_samples":len(c),"cgroup_observed_seconds":duration,"cgroup_mean_cores":(after["usage_usec"]-before["usage_usec"])/1e6/duration,
        "cgroup_throttled_delta_us":after["throttled_usec"]-before["throttled_usec"],
        "host_io_psi_some_avg10_max":max(float(q["io"].split("avg10=")[1].split()[0]) for q in c),
        "limitations":"Short 64-action transport calibration only. This host's ordinary/qualified checks and staged return to Gamer are independent. Three held-open screenshot readbacks add wall time only to this engineering case. One clean run cannot establish a population qualification rate."}
    args.out.write_text(json.dumps(h,indent=2)+"\n")

if __name__=="__main__":
    main()
