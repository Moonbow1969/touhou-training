"""Compare exact, reversibly compressed camera archives from eight real whole credits."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import subprocess
import time

import imageio_ffmpeg
import numpy as np
import zstandard as zstd

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,required=True);args=p.parse_args()
data=args.data
ff=imageio_ffmpeg.get_ffmpeg_exe()
rows=[]
for path in sorted(data.glob('*/original-pixels.npz')):
    with np.load(path) as z:frames=z['obs']
    raw=frames.tobytes();shape=frames.shape;measurements={}
    for name in ('gzip6','zstd1','zstd3','horizontal-zstd3','temporal-zstd3','ffv1','x264-lossless'):
        start=time.perf_counter();codec=None
        if name=='gzip6':compressed=gzip.compress(raw,compresslevel=6,mtime=0)
        elif 'zstd' in name:
            adjusted=frames.copy()
            if name.startswith('horizontal'):np.subtract(frames[:,:,1:],frames[:,:,:-1],out=adjusted[:,:,1:])
            if name.startswith('temporal'):np.subtract(frames[1:],frames[:-1],out=adjusted[1:])
            compressed=zstd.ZstdCompressor(level=int(name[-1])).compress(adjusted.tobytes())
        else:
            codec=['-c:v','ffv1','-level','3','-context','1'] if name=='ffv1' else ['-c:v','libx264','-crf','0','-preset','veryfast','-color_range','pc']
            cmd=[ff,'-loglevel','error','-nostdin','-f','rawvideo','-pix_fmt','gray','-s','96x112','-r','8','-i','pipe:0','-threads','1',
                 *codec,'-pix_fmt','gray','-f','matroska','pipe:1']
            compressed=subprocess.run(cmd,input=raw,capture_output=True,check=True).stdout
        encoded=time.perf_counter()-start;start=time.perf_counter()
        if name=='gzip6':decoded=gzip.decompress(compressed)
        elif 'zstd' in name:
            raw_decoded=zstd.ZstdDecompressor().decompress(compressed)
            delta=np.frombuffer(raw_decoded,dtype=np.uint8).reshape(shape)
            if name.startswith('horizontal'):decoded=np.cumsum(delta,axis=2,dtype=np.uint8).tobytes()
            elif name.startswith('temporal'):decoded=np.cumsum(delta,axis=0,dtype=np.uint8).tobytes()
            else:decoded=raw_decoded
        else:
            cmd=[ff,'-loglevel','error','-nostdin','-threads','1','-i','pipe:0','-threads','1','-f','rawvideo','-pix_fmt','gray','pipe:1']
            decoded=subprocess.run(cmd,input=compressed,capture_output=True,check=True).stdout
        measurements[name]=dict(bytes=len(compressed),encode_s=encoded,decode_s=time.perf_counter()-start,exact=decoded==raw)
    rows.append(dict(seed=int(path.parent.name.split('-')[1]),frames=len(frames),uncompressed_bytes=len(raw),
        decoded_pixels_sha256=hashlib.sha256(raw).hexdigest(),measurements=measurements))
if not rows:raise ValueError('No complete original camera archives in the supplied directory')
total=sum(r['uncompressed_bytes'] for r in rows)
summary={}
for k in rows[0]['measurements']:
    b=sum(r['measurements'][k]['bytes'] for r in rows)
    summary[k]=dict(bytes=b,percent_of_raw=100*b/total,exact_credits=sum(r['measurements'][k]['exact'] for r in rows),
                    encode_s=sum(r['measurements'][k]['encode_s'] for r in rows),decode_s=sum(r['measurements'][k]['decode_s'] for r in rows))
result=dict(camera='96x112 uint8 before each action plus one final successor',credits=len(rows),camera_frames=sum(r['frames'] for r in rows),
            pixel_bytes=total,codec_threads=1,scope='Offline Windows CPU encode/decode. Same real source camera arrays. Timing includes ffmpeg process startup. Original game control and transport unchanged.',summary=summary,receipts=rows)
(data/'lossless-camera-codecs.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
