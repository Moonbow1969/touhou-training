"""Stage an original TH08 trial on a Linux machine running local Wine and CPU inference.

Use the repository's Wine worker and CPU learner Docker images tagged
toho-trial:lab-5b67741 and toho-ppo-pilot:cpu-v1. The game's locally supplied
assets directory contains game/, archived-official-trial.json and the native
stream executable built from tools/rollout-worker-probe/th08-trial/stream.c.

    python -m scripts.prepare_th08_local_linux --assets "$ASSETS" --policy "$MODEL" --destination "$STUDY"
    docker run --rm -v "$STUDY:/stage" --entrypoint python3 toho-ppo-pilot:cpu-v1 \
        -m pip install --no-deps --target /stage/deps opencv-python-headless==4.10.0.84
    sh "$STUDY/app/credit.sh" 10001 Hard Magic engineering-zero-shot-original

Use an absolute study directory. Each unique seed writes its first native Continue
frame, physical keys and timestamps, camera movie and action distributions under
games/<seed>. Original team selection is frozen in team.png and team-default.png.
The game and inference containers communicate through two private FIFOs in the
same directory. The supplied trial assets, model and data are never published
by this command. Fit admission requires independently checking both actual menus,
the original rank/stock, the terminal image and every logged model output.
"""
import argparse
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parents[1]
T=R/'tools/rollout-worker-probe/th08-trial'

def title_check_files(pre,old,ref):
    marker='snap title\ntap z; sleep 1; snap default-rank'
    insert='''for n in $(seq 1 190); do
    scrot -o /capture/title-sample.next.png
    mv /capture/title-sample.next.png /capture/title-sample.png
    test ! -s /capture/title-gate.json || break
    sleep .27
done
test -s /capture/title-gate.json
cp /capture/title-accepted.png /capture/title.png
tap z; sleep 1; snap default-rank'''
    assert pre.count(marker)==1
    entry=pre.replace(marker,insert)
    start='--entrypoint python3 toho-ppo-pilot:cpu-v1 -u -m scripts.run_th08_trial_camera'
    end='--role "$role" --verified-menu'
    assert old.count(start)==1 and old.count(end)==1
    driver=old.replace('$app/game-entry.sh:','$app/game-entry-v2.sh:').replace(start,'''--entrypoint sh toho-ppo-pilot:cpu-v1 -c 'python3 /app/wait_title.py & watch=$!; python3 -u -m scripts.run_th08_trial_camera "$@"; status=$?; wait "$watch" || exit 21; exit "$status"' sh''')
    driver=driver.replace('date -Ins > "$out/setup-start.txt"','sha256sum "$app/game-entry-v2.sh" "$app/credit-v2.sh" "$app/wait_title.py" > "$out/menu-driver.sha256"\ndate -Ins > "$out/setup-start.txt"')
    return {'game-entry-v2.sh':entry.encode(),'credit-v2.sh':driver.encode(),'wait_title.py':(T/'wait_title.py').read_bytes(),'title-reference.png':ref.read_bytes()}


def prepare(A,policy,destination,title_reference=None):
    """Preserve the registered camera/keyboard loop; replace just its WSL launch."""
    if not destination.is_absolute():
        destination=destination.resolve()
    origin=json.loads((A/'archived-official-trial.json').read_bytes())
    for name,expected in origin['trial_files'].items():
        if hashlib.sha256((A/'game'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Original trial asset changed: {name}')
    source=(R/'scripts/run_th08_trial_camera.py').read_text(encoding='utf8')
    start=source.index("    name=f'toho-th08-")
    end=source.index('    logs=[]',start)
    # Replace only the original Windows/WSL Docker invocation. The two processes
    # below connect through native named pipes in the *same* Linux filesystem.
    source=source[:start]+'''    cmd=['sh','-c','cat /capture/frames.fifo & child=$!; cat > /capture/actions.fifo; wait "$child"']
'''+source[end:]
    entry=(T/'launch.sh').read_text(encoding='utf8').replace('for n in $(seq 1 50);','for n in $(seq 1 150);').replace('for n in $(seq 1 200);','for n in $(seq 1 150);')
    rank_switch='tap Down\ncase "$1" in Hard) : ;; Lunatic) tap Down ;; *) exit 3 ;; esac'
    assert entry.count(rank_switch)==1
    entry=entry.replace(rank_switch,'case "$1" in Normal) : ;; Hard) tap Down ;; Lunatic) tap Down; tap Down ;; *) exit 3 ;; esac')
    entry=entry.replace('tap z; sleep 1.8; snap team\ntap z; sleep 1\nexec /input/stream', '''tap z; sleep 1.8; snap team-default
case "$2" in Border) steps=0 ;; Magic) steps=1 ;; Scarlet) steps=2 ;; Ghost) steps=3 ;; *) exit 4 ;; esac
i=0
while [ "$i" -lt "$steps" ]; do tap Right; sleep .4; i=$((i+1)); done
snap team
tap z; sleep 1
exec sh /input/postcredit-ranking.sh''')
    assert 'team-default' in entry and '/input/postcredit-ranking.sh' in entry
    files={f'assets/game/{p.name}':p.read_bytes() for p in (A/'game').iterdir() if p.is_file()}
    files['assets/stream']=(A/'stream').read_bytes()
    files['assets/archived-official-trial.json']=(A/'archived-official-trial.json').read_bytes()
    files['policy.ts']=policy.read_bytes()
    for n in ['checkpoint_storage.py','train_five_original_games.py','original_camera_heads.py','th08_camera_history.py']:
        files['scripts/'+n]=(R/'scripts'/n).read_bytes()
    files['scripts/__init__.py']=b''
    files['scripts/run_th08_trial_camera.py']=source.encode()
    for p in T.iterdir():
        if p.suffix in {'.npy','.npz'}: files['tools/rollout-worker-probe/th08-trial/'+p.name]=p.read_bytes()
    files['game-entry.sh']=entry.encode()
    files['postcredit-ranking.sh']=(T/'postcredit-ranking.sh').read_text().replace('/capture/original-stream','/input/stream').encode()
    files['credit.sh']=(T/'linux-credit.sh').read_text().encode()
    if title_reference is not None:
        files.update(title_check_files(entry,files['credit.sh'].decode(),title_reference))
    for name,data in files.items():
        path=destination/'app'/name
        if path.exists() and path.read_bytes()!=data:
            raise ValueError(f'Existing staging directory contains a different source: {name}')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        if path.suffix=='.sh' or name=='assets/stream':path.chmod(0o755)
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
    (destination/'staging-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    return hashes

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','policy','destination'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--verified-start',type=Path,help='Native screenshot with the Start entry selected; enables credit-v2.sh')
    a=p.parse_args();prepare(a.assets,a.policy,a.destination,a.verified_start)
