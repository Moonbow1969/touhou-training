"""Original TH14 0.01b held-title evaluation in local isolated WSL games."""
import argparse,concurrent.futures,json,subprocess,time
from pathlib import Path
import numpy as np
import torch

from scripts.benchmark_trial_workers import TrialWorker
from scripts.train_replay_camera_encoder import sha,save_json
from scripts.train_distributed_native_ppo import action_mask
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True,help="Experiment root with frozen checkpoints")
parser.add_argument('--linux-game-dir',required=True,help="Absolute WSL path to the owned original trial directory")
parser.add_argument('--camera-clock',type=Path,required=True,help="Windows observer built with TRIAL_CAMERA and TRIAL_CONTACT_TELEMETRY")
parser.add_argument('--wsl-distro',default='Ubuntu-26.04')
parser.add_argument('--image',default='toho-gamer-trial:v6')
parser.add_argument('--transfer-ablation',action='store_true')
parser.add_argument('--ppo-generation',type=int,choices=(2,3,4,5),help="Evaluate one later frozen generation on the registered six held-title seeds")
parser.add_argument('--transfer-arm',choices=('matched-th13','cross-sft','cross-ppo'),help="Evaluate one arm of the preregistered 12-seed transfer study")
parser.add_argument('--anchor-arm',choices=('plain-cross-ppo','anchored-cross-ppo'),help="Evaluate one replay-regularization arm on 12 fresh controllers")
parser.add_argument('--supervised-budget-arm',choices=('old-joint-sft','joint-v2-3000','joint-v2-9000'),help="Evaluate the registered replay corpus and training budget comparison")
parser.add_argument('--workers',type=int,choices=range(1,9),default=3)
opts=parser.parse_args()
def linux(path):
    return subprocess.check_output(['wsl','-d',opts.wsl_distro,'--exec','wslpath','-a','-u',path.resolve().as_posix()],text=True).strip()
d=opts.root.resolve();root=d/'zero-shot-th14'
launcher=linux(Path(__file__).resolve().parents[1]/'tools/rollout-worker-probe/run-th14-camera-worker.sh')
clock=linux(opts.camera_clock)
root.mkdir(exist_ok=True)
models={'sft':d/'camera-sft/policy.ts','ppo1':d/'camera-ppo/policy.ts'}
extra=opts.transfer_ablation
if extra:models={'expanded':d/'camera-sft-expanded/policy.ts','frequency':d/'camera-sft-frequency/policy.ts',
                 'transition':d/'camera-transition/conditional-policy.ts'}
if opts.ppo_generation:
    if extra:parser.error("Choose one experiment arm group")
    name=f'ppo{opts.ppo_generation}'
    models={name:d/f'camera-ppo{opts.ppo_generation}'/'policy.ts'}
if opts.transfer_arm:
    if extra or opts.ppo_generation:parser.error("Choose one experiment arm group")
    folder={'matched-th13':'camera-sft-th13-matched','cross-sft':'camera-cross-sft','cross-ppo':'camera-cross-ppo'}[opts.transfer_arm]
    models={opts.transfer_arm:d/folder/'policy.ts'}
if opts.anchor_arm:
    if extra or opts.ppo_generation or opts.transfer_arm:parser.error("Choose one experiment arm group")
    folder='camera-cross-ppo' if opts.anchor_arm=='plain-cross-ppo' else 'replay-anchor/policy'
    models={opts.anchor_arm:d/folder/'policy.ts'}
if opts.supervised_budget_arm:
    if extra or opts.ppo_generation or opts.transfer_arm or opts.anchor_arm:parser.error("Choose one experiment arm group")
    folder=d/'camera-cross-sft' if opts.supervised_budget_arm=='old-joint-sft' else d/'v2-supervised'/opts.supervised_budget_arm
    models={opts.supervised_budget_arm:folder/'policy.ts'}
plan={'title':14,'edition':'Official original trial 0.01b Normal Reimu A on Gamer WSL2, each process has private Xvfb, Wine state and input',
      'arms':{k:sha(v) for k,v in models.items()},'seeds':list(range(787001,787007)),
      'zero_shot':'Both direct checkpoints trained only on TH10/12/13. No TH14 optimizer input. Exclude technical failures from results.',
      'credit':'Original stock, four advancing native ticks per control, no continues. Zero keys during final hit animation, verify final negative stock and game-over menu with original capture. Controller randomness matched within seed.'}
if extra:
    plan['per_arm_seeds']={'expanded':list(range(787001,787007)),
                           'frequency':list(range(788001,788007)),'transition':list(range(788001,788007))}
    plan['condition']='Expanded SFT shares original SFT seeds. Frequency and learned transition use a new common seed block; all frozen before unblinding TH14.'
    del plan['seeds']
protocol_name=f'ppo{opts.ppo_generation}-protocol.json' if opts.ppo_generation else ('transfer-ablation-protocol.json' if extra else 'protocol.json')
if opts.transfer_arm:
    plan['seeds']=list(range(789001,789013));plan['source_mixture']='Matched supervised step budget and loss; single-title vs three-title human inputs, then multi-title on-policy PPO.'
    protocol_name=opts.transfer_arm+'-protocol.json'
if opts.anchor_arm:
    plan['seeds']=list(range(791001,791013))
    plan['training']='Identical 24 original credits and supervised parent, optional joint KL on its training replay images.'
    protocol_name=opts.anchor_arm+'-protocol.json'
if opts.supervised_budget_arm:
    plan['seeds']=list(range(792001,792013))
    plan['training']='Compare original SFT with enlarged human corpus at 3000 and 9000 updates on identical held controller seeds.'
    protocol_name=opts.supervised_budget_arm+'-protocol.json'
save_json(root/protocol_name,plan);torch.set_num_threads(1)

def play(arm,seed):
    model=torch.jit.load(str(models[arm]),map_location='cpu').eval();rng=np.random.default_rng(seed)
    name=f'v26th14-{arm}-{seed}';dest=root/name;dest.mkdir(exist_ok=True)
    if (dest/'report.json').exists():return
    command=['wsl','-d',opts.wsl_distro,'--','sh',launcher,name,linux(dest),opts.linux_game_dir,clock,opts.image]
    frames=[];actions=[];likelihood=[];state_rows=[];start=time.monotonic();prev=None;stage2=None
    with TrialWorker(command,shutdown_timeout=15) as worker,torch.inference_mode():
        s=worker.read(timeout=120)
        if (s['stage'],s['mode'],s['lives'])!=(1,2,2):raise ValueError(f'Unexpected initial stock: {str(s)[:100]}')
        start_stock=s['bombs']; first_tick=s['ticks']
        flags=0
        while True:
            if s['stage']==2 and stage2 is None:stage2=dict(spares=s['lives'],bombs=s['bombs'],tick=s['ticks'])
            if s['lives']<0 and s['mode']!=2:
                if s['mode']!=1:raise ValueError('Unverified terminal menu')
                break
            if s['ticks']-first_tick>160000 or time.monotonic()-start>3600:raise TimeoutError('Unverified long credit')
            if s['lives']<0:
                s=worker.request('S 0');flags+=1
                if flags>1000:raise ValueError('Unsettled hit animation')
                continue
            flags=0
            frame=np.frombuffer(bytes.fromhex(s['camera']),np.uint8).reshape(112,96)
            pixels=torch.tensor(frame,dtype=torch.float32)/255
            diff=torch.zeros_like(pixels) if prev is None else (pixels-prev).abs()
            view=torch.stack((pixels,diff))[None]
            heads=(model(view,torch.tensor([actions[-1][0] if actions else 4])) if arm=='transition' else model(view))[:4];prev=pixels
            probabilities=[head[0].softmax(0).numpy().astype(float) for head in heads]
            a=[int(rng.choice(len(p),p=p/p.sum())) for p in probabilities]
            frames.append(frame);actions.append(a);likelihood.append([np.log(q[k]) for q,k in zip(probabilities,a)])
            state_rows.append([s[k] for k in ('ticks','frame','stage','lives','bombs','mode')]+s['player'])
            if len(actions)==1:print('ACTIVE',arm,seed,s['lives'],s['bombs'],flush=True)
            s=worker.request(f'S {action_mask(a)}')
        subprocess.run(['wsl','-d',opts.wsl_distro,'--','docker','exec','-e','DISPLAY=:66',name,'scrot','-o','/capture/terminal.png'],check=True)
        trace=dest/'native.npz'
        np.savez_compressed(trace,frames=np.stack(frames),action=np.asarray(actions,np.int8),logp=np.asarray(likelihood,np.float32),
                            native=np.asarray(state_rows,np.float64))
        y=np.asarray(actions)
        save_json(dest/'report.json',dict(arm=arm,seed=seed,policy=sha(models[arm]),source='clock-locked official TH14 Normal trial',
            title=14,host='Gamer WSL2',continued=False,terminal='game_over',stage2=stage2,steps=len(y),elapsed_native_ticks=s['ticks']-first_tick,
            last_observed_stage=s['stage'],terminal_spares=s['lives'],terminal_bombs=s['bombs'],initial_spares=2,initial_bombs=start_stock,
            action_counts=[np.bincount(y[:,k],minlength=n).tolist() for k,n in enumerate((9,2,2,2))],trace_sha256=sha(trace),
            terminal_image_sha256=sha(dest/'terminal.png'),wall_seconds=time.monotonic()-start))
        print('COMPLETE',arm,seed,len(y),stage2,s['bombs'],flush=True)

if extra:
    jobs=[]
    for i in range(6):
        pair=['frequency','transition'] if i%2==0 else ['transition','frequency']
        jobs.append(('expanded',787001+i))
        jobs.extend((arm,788001+i) for arm in pair)
elif opts.ppo_generation or opts.transfer_arm or opts.anchor_arm or opts.supervised_budget_arm:
    jobs=[(next(iter(models)),seed) for seed in plan['seeds']]
else:jobs=[(arm,seed) for i,seed in enumerate(plan['seeds']) for arm in (['sft','ppo1'] if i%2==0 else ['ppo1','sft'])]
def retry(arm,seed):
    for attempt in range(3):
        try:
            play(arm,seed)
            return
        except Exception as exc:
            log=root/f'v26th14-{arm}-{seed}'/'technical-attempts.jsonl'
            with log.open('a') as f:f.write(json.dumps({'attempt':attempt+1,'error':str(exc),'epoch_seconds':time.time()})+'\n')
            if attempt==2:raise
            time.sleep(3)
with concurrent.futures.ThreadPoolExecutor(opts.workers) as pool:
    futures=[pool.submit(retry,arm,seed) for arm,seed in jobs]
    for future in concurrent.futures.as_completed(futures):future.result()
