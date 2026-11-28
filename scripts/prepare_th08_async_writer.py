"""Install the TH08 0.02 acknowledged archive writer into a new isolated original-game study."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess


FILES=("bracket.c","archive_seal.h","reader_async.c","archive_writer.h")
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(source,destination,wine_image):
    deploy=source/"app/tick4-deployment.sha256"
    subprocess.run(["sha256sum","-c",deploy.name],cwd=deploy.parent,stdout=subprocess.DEVNULL,check=True)
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"games").mkdir()
    (destination/"deps").symlink_to((source/"deps").resolve(),target_is_directory=True)
    app=destination/"app"
    subprocess.run(["cp","-a","--reflink=auto",str(source/"app"),str(app)],check=True)
    native=app/"tick4"
    for name in FILES:
        shutil.copyfile(ROOT/"tools/rollout-worker-probe/th08-trial002/tick4"/name,native/name)
    subprocess.run(["gcc","-std=c11","-O2","-static","-DTH08_EXPLICIT_ARCHIVE_SEAL","bracket.c","-o","bracket"],cwd=native,check=True)
    subprocess.run(["docker","run","--rm","--network","none","--pull","never","--entrypoint","sh","-v",f"{native}:/clock",wine_image,
        "-ec","cd /clock && i686-w64-mingw32-gcc-posix -std=c11 -O2 -Wl,--no-insert-timestamp reader_async.c -o clock.exe"],check=True)
    fin=app/"postcredit-ranking.sh"
    original=fin.read_text()
    marker="# Policy has stopped and all seven physical keys were released by its worker."
    if original.count(marker)!=1:raise ValueError("Source is not the qualified whole-credit ranking wrapper")
    new=original.replace(marker,'''# Preserve the first native endpoint before any ranking key is issued.
touch /capture/native-archive-request
waits=0
while ! test -s /capture/native-archive-bracket-status.tsv && test "$waits" -lt 500; do
  sleep .05
  waits=$((waits+1))
done
if ! test -s /capture/native-archive-bracket-status.tsv; then
  printf 'no acknowledged native archival close before ranking\n' > /capture/native-archive-unsealed.txt
fi
'''+marker)
    fin.write_text(new)
    names={line.split("  ",1)[1] for line in deploy.read_text().splitlines()}
    names.update(f"tick4/{name}" for name in FILES)
    (app/"tick4-deployment.sha256").write_text("".join(f"{sha(app/name)}  {name}\n" for name in sorted(names)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualified-study",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--wine-image",required=True)
    args=parser.parse_args()
    install(args.qualified_study.resolve(),args.destination.resolve(),args.wine_image)
