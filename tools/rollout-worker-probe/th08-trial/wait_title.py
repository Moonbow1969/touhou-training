"""Require the real, selected TH08 Start menu before the first menu keystroke."""
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

def feature(data):
    with Image.open(io.BytesIO(data)) as img:
        values=np.asarray(img.convert('L').crop((95,173,171,210)),dtype=np.float32).ravel()
    values-=values.mean()
    return values/max(np.linalg.norm(values),1e-5)

def wait(folder,template,deadline=95):
    ref=feature(template.read_bytes());sample=folder/'title-sample.png';start=time.monotonic();rows=[];last=None
    while time.monotonic()-start<deadline:
        if sample.exists():
            stamp=sample.stat().st_mtime_ns
            if stamp!=last:
                data=sample.read_bytes()
                corr=float(feature(data)@ref)
                rows.append([round(time.monotonic()-start,3),corr])
                if corr>=.94:
                    (folder/'title-accepted.png').write_bytes(data)
                    (folder/'title-gate.json').write_text(json.dumps(dict(template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
                        photographed_sha256=hashlib.sha256(data).hexdigest(),roi=[95,173,171,210],required=.94,
                        original_samples=rows),indent=2)+'\n')
                    return
                last=stamp
        time.sleep(.12)
    (folder/'title-gate-timeout.json').write_text(json.dumps(rows)+'\n')
    raise RuntimeError('Original Start highlight did not appear before any policy input')

if __name__=='__main__':
    wait(Path('/capture'),Path('/app/title-reference.png'))
