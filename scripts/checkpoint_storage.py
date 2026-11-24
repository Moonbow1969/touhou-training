"""Bounded, recoverable storage for native full-policy PPO checkpoints."""

from hashlib import file_digest
import io
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile


MIB = 1024 * 1024
DEFAULT_BUDGET_MIB = 256
DEFAULT_RESERVE_MIB = 2048
MANIFEST = "toho_checkpoint.json"


class CheckpointStorageError(RuntimeError):
    """The checkpoint cannot be safely saved or resumed."""


class CheckpointBuffer(io.BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise CheckpointStorageError(f"The new checkpoint exceeds the available {self.limit:,}-byte budget.")
        return super().write(data)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def checkpoint_manifest(path):
    try:
        with ZipFile(path) as archive:
            if MANIFEST not in archive.namelist():
                return None
            if not {"data", "policy.pth", "policy.optimizer.pth"}.issubset(archive.namelist()) or archive.testzip():
                raise CheckpointStorageError("The checkpoint archive failed integrity validation.")
            manifest = json.loads(archive.read(MANIFEST))
            steps = json.loads(archive.read("data")).get("num_timesteps")
        if (not isinstance(manifest, dict) or manifest.get("version") != 1
                or type(steps) is not int or steps < 0 or manifest.get("steps") != steps
                or not isinstance(manifest.get("compatibility"), dict)
                or type(manifest.get("warm_start_actor_exact")) is not bool):
            raise CheckpointStorageError("The checkpoint manifest is inconsistent with the PPO state.")
        return manifest
    except (BadZipFile, OSError, UnicodeError, ValueError, AttributeError, TypeError) as error:
        raise CheckpointStorageError(f"Could not validate checkpoint {path.name}: {error}") from error


def compatibility(report):
    settings = {
        "base_sha256": report["checkpoints"]["base_sha256"],
        "train_limit": report["task"]["train_limit"],
        "seed": report["seed"],
        "ppo_hyperparameters": report["ppo_hyperparameters"],
        "reward": report["reward"],
    }
    if report.get("initialization", "sft") != "sft":
        settings["initialization"] = report["initialization"]
    return settings


def read_resume(path, expected):
    manifest = checkpoint_manifest(path)
    fingerprint = digest(path)
    report_path = path.with_name("report.json")
    try:
        prior = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        if not isinstance(prior, dict):
            raise ValueError("A checkpoint report must be a JSON object.")
    except (OSError, ValueError) as error:
        if manifest is None:
            raise CheckpointStorageError("The legacy checkpoint report is unreadable.") from error
        prior = {}
    if manifest is None:
        if not prior or fingerprint != prior["checkpoints"].get("ppo_sha256") or compatibility(prior) != expected:
            raise CheckpointStorageError("The legacy checkpoint needs a matching report and training configuration.")
        return fingerprint, prior["warm_start_actor_exact"], "legacy_report"
    if manifest["compatibility"] != expected:
        raise CheckpointStorageError("The checkpoint training configuration differs from this run.")
    records = prior.get("checkpoint_storage", {})
    if manifest["steps"] == prior.get("checkpoint_steps") and fingerprint != prior["checkpoints"]["ppo_sha256"]:
        raise CheckpointStorageError("The checkpoint digest differs from its saved report.")
    previous = records.get("previous")
    if previous and manifest["steps"] == previous["steps"] and fingerprint != previous["sha256"]:
        raise CheckpointStorageError("The previous checkpoint digest differs from its saved report.")
    evidence = "embedded_manifest" if prior else "embedded_manifest_without_report"
    return fingerprint, manifest["warm_start_actor_exact"], evidence


class CheckpointStore:
    def __init__(self, out, budget_bytes, reserve_bytes):
        if budget_bytes < 1 or reserve_bytes < 0:
            raise ValueError("Checkpoint budget must be positive and disk reserve cannot be negative.")
        self.out = Path(out)
        self.budget_bytes = budget_bytes
        self.reserve_bytes = reserve_bytes
        self.current = self.out / "policy.zip"
        self.previous = self.out / "policy.previous.zip"

    def remaining(self, staged_bytes=0):
        volume = self.out.resolve()
        while not volume.exists():
            volume = volume.parent
        free_after_reserve = shutil.disk_usage(volume).free - self.reserve_bytes
        if free_after_reserve <= 0:
            raise CheckpointStorageError(f"Output disk has reached its {self.reserve_bytes:,}-byte free-space reserve.")
        occupied = staged_bytes + sum(path.stat().st_size for path in (self.current, self.previous) if path.exists())
        if occupied >= self.budget_bytes:
            raise CheckpointStorageError(f"Checkpoint archives have reached their {self.budget_bytes:,}-byte budget.")
        return min(self.budget_bytes - occupied, free_after_reserve)

    def save(self, writer, manifest):
        last = None
        if self.current.exists():
            existing = checkpoint_manifest(self.current)
            if existing is None or existing["steps"] >= manifest["steps"]:
                raise CheckpointStorageError("The current checkpoint has missing or non-increasing lineage.")
            last = {"steps": existing["steps"], "sha256": digest(self.current),
                    "bytes": self.current.stat().st_size}
        manifest = dict(manifest, version=1, prior_checkpoint_sha256=last["sha256"] if last else None)
        with CheckpointBuffer(self.remaining()) as buffer:
            writer(buffer)
            with ZipFile(buffer, "a") as archive:
                archive.writestr(MANIFEST, json.dumps(manifest, sort_keys=True, allow_nan=False))
            with buffer.getbuffer() as data:
                if len(data) > self.remaining():
                    raise CheckpointStorageError("Free disk space decreased before checkpoint publication.")
                self._publish(data, manifest, last)
        return {"current": {"steps": manifest["steps"], "sha256": digest(self.current),
                            "bytes": self.current.stat().st_size},
                "previous": last}

    def _publish(self, data, manifest, last):
        with NamedTemporaryFile(dir=self.out, prefix=".checkpoint-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        backup = temporary.with_suffix(".previous.zip")
        keep_backup = False
        try:
            with temporary.open("wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if checkpoint_manifest(temporary) != manifest:
                raise CheckpointStorageError("The staged checkpoint manifest did not verify.")
            if last:
                try:
                    os.link(self.current, backup)
                except OSError:
                    # Some volumes do not support hard links. Admit the extra copy
                    # before writing it while the original checkpoint stays in place.
                    if last["bytes"] > self.remaining(staged_bytes=len(data)):
                        raise CheckpointStorageError("Insufficient disk budget for atomic previous-checkpoint rotation.")
                    shutil.copyfile(self.current, backup)
                    with backup.open("rb+") as stream:
                        os.fsync(stream.fileno())
                    if digest(backup) != last["sha256"]:
                        raise CheckpointStorageError("The staged previous checkpoint failed digest verification.")
            temporary.replace(self.current)
            if last:
                try:
                    backup.replace(self.previous)
                except OSError as publish_error:
                    try:
                        backup.replace(self.current)
                    except OSError as rollback_error:
                        keep_backup = True
                        raise CheckpointStorageError(
                            f"Could not rotate the checkpoint. The last saved policy remains at {backup.name}."
                        ) from rollback_error
                    raise CheckpointStorageError(
                        "Could not rotate the previous checkpoint. The prior current checkpoint was restored."
                    ) from publish_error
        finally:
            temporary.unlink(missing_ok=True)
            if not keep_backup:
                backup.unlink(missing_ok=True)
