"""Measure exact-pixel alternatives for transporting original game evidence after each credit."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import zipfile
from PIL import Image, features
from scripts.checkpoint_storage import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if not features.check('webp'):raise RuntimeError('Missing lossless WebP support')
    warm=io.BytesIO();Image.new('RGB',(4,4)).save(warm,'WEBP',lossless=True)
    rows=[];exact_files=Counter()
    with zipfile.ZipFile(a.bundle) as archive:
        screenshots=[p for p in archive.namelist() if p.endswith('.png')]
        if len(screenshots)!=128:raise ValueError('Expected four original screenshots for each of 32 completed TH17 games')
        for name in screenshots:
            data=archive.read(name);sha=hashlib.sha256(data).hexdigest();exact_files[sha]+=1
            image=Image.open(io.BytesIO(data)).convert('RGBA');pixels=hashlib.sha256(image.tobytes()).hexdigest()
            row=dict(file=name,png_bytes=len(data),png_sha256=sha,rgba_sha256=pixels)
            for method in (0,2):
                buffer=io.BytesIO();start=time.perf_counter()
                image.save(buffer,'WEBP',lossless=True,exact=True,method=method)
                elapsed=time.perf_counter()-start
                encoded=buffer.getvalue()
                decoded=Image.open(io.BytesIO(encoded)).convert('RGBA')
                if decoded.size!=image.size or hashlib.sha256(decoded.tobytes()).hexdigest()!=pixels:raise ValueError('Changed original pixels')
                row[f'webp{method}']=dict(bytes=len(encoded),encode_seconds=elapsed)
            rows.append(row)
    png=sum(r['png_bytes'] for r in rows);options={}
    for m in ('webp0','webp2'):
        size=sum(r[m]['bytes'] for r in rows);secs=sum(r[m]['encode_seconds'] for r in rows)
        options[m]=dict(bytes=size,saved_bytes=png-size,saved_fraction=(png-size)/png,encode_seconds=secs,
                        break_even_link_Mbps=8*(png-size)/secs/1e6)
    result=dict(source=a.bundle.name,source_sha256=digest(a.bundle),host=platform.node(),pillow=Image.__version__,libwebp=features.version('webp'),
                credits=32,screenshots=len(rows),unique_png_digests=len(exact_files),png_bytes=png,options=options,images=rows)
    (a.out/'lossless-evidence-encoding.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    lines=['# Lossless transport of original TH17 evidence images', '',
           f'Across 32 completed Hard and Lunatic test credits, the four per-game PNGs occupied {png/2**20:.2f} MiB. There were {len(exact_files)} distinct full-file checksums among {len(rows)} PNGs. Each tested encoding preserved every decoded RGBA byte and image dimension. Existing source archives retain the original PNG file bytes.', '',
           '| WebP method | Output MiB | Saved MiB | Reduction | Additional encode time | Serial break-even link |','|---|---:|---:|---:|---:|---:|']
    for m,v in options.items():
        lines.append(f"| {m} | {v['bytes']/2**20:.2f} | {v['saved_bytes']/2**20:.2f} | {v['saved_fraction']:.1%} | {v['encode_seconds']:.2f} s | {v['break_even_link_Mbps']:.1f} Mbit/s |")
    lines+=['','Break-even divides saved wire bits by measured additional encoding seconds, assuming serial re-encoding of a PNG already produced by the worker. The timer covers encoding on the benchmark host. PNG decoding, output verification, network latency, protocol overhead and network transfer are outside it. Native direct encoding, work overlapped with gameplay, or a filesystem archive may have different costs. No step-time game traffic was rerouted in this measurement.', '',
            '[Per-image byte counts, timings and checksums](lossless-evidence-encoding.json) · [Original source bundle](../th17-held-native-records.release.json)']
    (a.out/'lossless-evidence-encoding.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('Verified',len(rows),'exact-pixel pairs for both encoders',options,flush=True)


if __name__=='__main__':main()
