"""Time byte-verified SSH transfers of one sealed original-game archive without remote storage."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
from scripts.checkpoint_storage import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--hosts',nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();sha=digest(a.archive);size=a.archive.stat().st_size;times={}
    for host in a.hosts:
        samples=[]
        for _ in range(3):
            with a.archive.open('rb') as data:
                start=time.perf_counter()
                returned=subprocess.check_output(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','Compression=no',host,'sha256sum'],stdin=data,timeout=45)
                elapsed=time.perf_counter()-start
            if returned.decode().split()[0]!=sha:raise ValueError('Remote byte verification failed')
            samples.append(elapsed)
        times[host]=dict(seconds=samples,median_seconds=statistics.median(samples),median_Mbps=8*size/statistics.median(samples)/1e6)
    result=dict(archive=a.archive.name,archive_sha256=sha,bytes_per_transfer=size,hosts=times,
                scope='Gamer to each named worker, one fresh SSH CLI per transfer, SSH compression off, remote standard input hashed and discarded. Includes process setup, authentication, transfer and hash confirmation. Existing workloads continued.')
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(times,flush=True)


if __name__=='__main__':main()
