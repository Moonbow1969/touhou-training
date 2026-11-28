"""Clone an existing qualified compact-camera TH08 worker with a 224×192 evidence camera."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def install(source,destination,expected_policy):
    old=source/"app";deploy=old/"tick4-deployment.sha256"
    subprocess.run(["sha256sum","-c",deploy.name],cwd=old,stdout=subprocess.DEVNULL,check=True)
    if digest(old/"policy.ts")!=expected_policy:raise ValueError("Qualified source does not contain the requested frozen policy")
    destination.mkdir(parents=True,exist_ok=False)
    app=destination/"app";(destination/"games").mkdir()
    (destination/"deps").symlink_to((source/"deps").resolve(),target_is_directory=True)
    subprocess.run(["cp","-a","--reflink=auto",str(old),str(app)],check=True)
    runner=app/"scripts/run_th08_trial_camera.py"
    text=runner.read_text()
    for before,after in {
        "from scripts.th08_compact_rgb import CompactRGBQueue":"from scripts.th08_highres_camera import HighResolutionQueue",
        "CompactRGBQueue(root/'compact-rgb.npz',async_core)":"HighResolutionQueue(root/'highres-gray.npz',async_core)",
        "compact_rgb_sha256=digest(root/'compact-rgb.npz')":"highres_gray_sha256=digest(root/'highres-gray.npz')",
        "One compact 56x48 RGB crop per source observation":"One 224x192 grayscale crop per source observation",
    }.items():
        if text.count(before)!=1:raise ValueError(f"Expected one site in the qualified source runner: {before}")
        text=text.replace(before,after)
    runner.write_text(text)
    module=Path(__file__).with_name("th08_highres_camera.py")
    name=f"scripts/{module.name}";shutil.copyfile(module,app/name)
    files={line.split("  ",1)[1] for line in deploy.read_text().splitlines()}|{name}
    (app/deploy.name).write_text("".join(f"{digest(app/f)}  {f}\n" for f in sorted(files)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualified-study",type=Path,required=True);parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--policy-sha256",required=True)
    args=parser.parse_args();install(args.qualified_study.resolve(),args.destination.resolve(),args.policy_sha256)
