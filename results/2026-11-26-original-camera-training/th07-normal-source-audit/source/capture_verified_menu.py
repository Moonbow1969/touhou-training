"""Collect an exact-version, pre-staged TH07 replay on a private original Wine/Xvfb lane.
The original build, replay and title image are external prerequisites, not bundled.
"""
import argparse
import hashlib
import subprocess
import time
from pathlib import Path
from PIL import Image
import numpy as np

def measure(path,reference,box):
    def vec(p):
        a=np.asarray(Image.open(p).crop(box).convert('L').resize((90,35)),dtype=float).ravel()
        return a-a.mean()
    a,b=vec(path),vec(reference)
    return float(a.dot(b)/max(1,np.linalg.norm(a)*np.linalg.norm(b)))

def collect(container,folder,title,expected):
    source=folder/'th7_01.rpy'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    def cmd(s):
        subprocess.run(['docker','exec',container,'sh','-ec',s],check=True)
    def key(k):
        cmd('xdotool keydown '+k+'; sleep .1; xdotool keyup '+k+'; sleep .72')
    def snap(filename):
        cmd('scrot -o /capture/'+filename)
        return folder/filename
    deadline=time.monotonic()+50
    while time.monotonic()<deadline:
        if (folder/'window').exists():
            shot=snap('capture-title.png')
            if measure(shot,title,(25,45,410,172))>.85:break
        time.sleep(.3)
    else:raise RuntimeError('Original title unavailable')
    # A newly launched pristine trial begins with Start highlighted. Wait until
    # menu text has faded in before selecting Replay (two disabled rows skipped).
    time.sleep(1.6)
    key('Down');snap('replay-highlight.png')
    key('z');snap('matched-author.png')
    key('z');snap('matched-stage-selector.png')
    key('z');snap('matched-speed.png')
    subprocess.run(['docker','exec','-d',container,'sh','-c',
        '/capture/final-sensor 11c9ae8 235 > /capture/final.log 2>&1 && touch /capture/final.done'],check=True)
    key('z');snap('matched-opening.png')
    # pack() independently rejects a wrong selection using both current and previous keys.
    deadline=time.monotonic()+245
    while time.monotonic()<deadline:
        if (folder/'final.done').exists():break
        time.sleep(1)
    else:raise RuntimeError('Original source capture failed')
    snap('after-replay.png')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('container','folder','title','source_sha256'):p.add_argument(arg)
    a=p.parse_args()
    collect(a.container,Path(a.folder),Path(a.title),a.source_sha256)
