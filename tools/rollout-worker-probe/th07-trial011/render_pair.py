"""Place both prespecified full-credit 1x camera films on a shared wall-time timeline."""
import argparse,json,subprocess
from pathlib import Path

def duration(path):
    return float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",str(path)],text=True))
def run(root,seed):
    arms=["action","action-xy"];files=[root/f"held-{seed}-{arm}/native-credit.mp4"for arm in arms]
    times=list(map(duration,files));out=root/f"matched-native-{seed}.mp4"
    text=["Four-head SFT","Same SFT + player XY loss"];graph=[]
    for i in range(2):
        graph.append(f"[{i}:v]scale=448:336,pad=448:366:0:30,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{text[i]}':fontcolor=white:fontsize=16:x=10:y=6,tpad=stop_mode=clone:stop_duration={max(times)-times[i]+.1:.4f}[v{i}]")
    graph.append("[v0][v1]hstack=inputs=2[v]")
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(files[0]),"-i",str(files[1]),"-filter_complex_threads","1",
                    "-filter_complex",";".join(graph),"-map","[v]","-t",str(max(times)),"-c:v","libx264","-threads","1","-preset","fast","-crf","22",
                    "-pix_fmt","yuv420p","-movflags","+faststart",str(out)],check=True)
    subprocess.run(["ffmpeg","-v","error","-i",str(out),"-f","null","-"],check=True)
    (root/"matched-film.json").write_text(json.dumps(dict(seed=seed,arms=arms,source_duration_seconds=times,playback_rate=1,
        alignment="Each original stock opening then wall-timed deployed grayscale policy views, genuine full-color Continue and scorecard. Shorter credit holds its final scorecard.",
        merged_duration=duration(out)),indent=2)+"\n")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("seed",type=int);a=p.parse_args();run(a.root,a.seed)
