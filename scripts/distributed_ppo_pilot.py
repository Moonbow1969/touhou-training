"""Compare synchronous PPO and weight averaging on a three-host CartPole control."""

import argparse
import base64
import hashlib
import io
import json
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


TASK = "CartPole-v1"
ROLLOUT = 128
PPO_ARGS = {
    "n_steps": ROLLOUT,
    "batch_size": 64,
    "n_epochs": 4,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "policy_kwargs": {"net_arch": {"pi": [32, 32], "vf": [32, 32]}},
}


def checkpoint_bytes(model):
    stream = io.BytesIO()
    model.save(stream)
    return stream.getvalue()


def record(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def worker_environment():
    """Serve a single environment over standard input and output."""
    with gym.make(TASK) as env:
        print(json.dumps({"task": TASK}), flush=True)
        for line in sys.stdin:
            operation, argument = json.loads(line)
            if operation == "reset":
                observation, info = env.reset(seed=argument)
                result = [observation.tolist(), info]
            elif operation == "step":
                observation, reward, terminated, truncated, info = env.step(argument)
                done = terminated or truncated
                reset_info = None
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = observation.tolist()
                    observation, reset_info = env.reset()
                result = [observation.tolist(), reward, done, info, reset_info]
            else:
                raise ValueError(f"Unknown environment operation {operation!r}.")
            print(json.dumps(result, separators=(",", ":")), flush=True)


class EnvironmentProcess:
    def __init__(self, command):
        self.process = subprocess.Popen(
            [*command, "--worker", "environment"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        self.output = queue.Queue()
        self.reader = threading.Thread(target=self.read_output, daemon=True)
        self.reader.start()

    def read_output(self):
        try:
            for line in self.process.stdout:
                self.output.put(json.loads(line))
        except (OSError, ValueError) as error:
            self.output.put(error)
        finally:
            self.output.put(EOFError("Environment worker stdout closed."))

    def send(self, operation, argument):
        print(json.dumps([operation, argument]), file=self.process.stdin, flush=True)

    def receive(self):
        try:
            response = self.output.get(timeout=45)
        except queue.Empty as error:
            raise TimeoutError("Environment worker did not reply within 45 seconds.") from error
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.reader.join(timeout=1)
        self.process.stdout.close()


class RemoteCartPole(VecEnv):
    """Collect one same-policy transition from every host per vector step."""

    def __init__(self, commands):
        self.workers = []
        try:
            for command in commands:
                self.workers.append(EnvironmentProcess(command))
            for worker in self.workers:
                if worker.receive() != {"task": TASK}:
                    raise RuntimeError("Environment worker handshake did not match the requested task.")
            with gym.make(TASK) as env:
                super().__init__(len(commands), env.observation_space, env.action_space)
        except BaseException:
            self.close()
            raise

    def reset(self):
        for worker, seed in zip(self.workers, self._seeds):
            worker.send("reset", seed)
        replies = [worker.receive() for worker in self.workers]
        observations, reset_infos = zip(*replies)
        self.reset_infos = list(reset_infos)
        self._reset_seeds()
        self._reset_options()
        return np.asarray(observations, dtype=np.float32)

    def step_async(self, actions):
        for worker, action in zip(self.workers, actions):
            worker.send("step", int(action))

    def step_wait(self):
        observations, rewards, dones, infos, reset_infos = zip(*(worker.receive() for worker in self.workers))
        for index, info in enumerate(infos):
            if dones[index]:
                info["terminal_observation"] = np.asarray(info["terminal_observation"], dtype=np.float32)
                self.reset_infos[index] = reset_infos[index]
        return (np.asarray(observations, dtype=np.float32), np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=bool), infos)

    def close(self):
        for worker in self.workers:
            worker.close()
        self.workers.clear()

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None for _ in self._get_indices(indices)]
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(attr_name)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise AttributeError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


def learn_from_parent(parent, env, budget, seed):
    model = PPO.load(io.BytesIO(parent), env=env, seed=seed, device="cpu")
    start = time.perf_counter()
    model.learn(total_timesteps=budget)
    elapsed = time.perf_counter() - start
    if model.num_timesteps != budget:
        raise RuntimeError(f"Expected {budget} interactions, got {model.num_timesteps}.")
    timing = {"steps": budget, "seconds": elapsed, "steps_per_second": budget / elapsed, "seed": seed}
    return model, timing


def worker_train(budget, seed):
    parent = sys.stdin.buffer.read()
    with gym.make(TASK) as env:
        model, timing = learn_from_parent(parent, env, budget, seed)
        stream = io.BytesIO()
        torch.save(model.policy.state_dict(), stream)
        data = stream.getvalue()
    response = {
        "parent_sha256": hashlib.sha256(parent).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(data).hexdigest(),
        "checkpoint": base64.b64encode(data).decode("ascii"),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        **timing,
    }
    print(json.dumps(response, separators=(",", ":")), flush=True)


def train_branch(command, parent, budget, seed):
    process = subprocess.run(
        [*command, "--worker", "train", "--branch-steps", str(budget), "--seed", str(seed)],
        input=parent, stdout=subprocess.PIPE, check=True, timeout=240,
    )
    response = json.loads(process.stdout)
    data = base64.b64decode(response.pop("checkpoint"), validate=True)
    if (response["parent_sha256"] != hashlib.sha256(parent).hexdigest()
            or response["checkpoint_sha256"] != hashlib.sha256(data).hexdigest()
            or response["steps"] != budget):
        raise RuntimeError("Branch provenance does not match the requested experiment.")
    return data, response


def load_branch(parent, weights):
    # Preserve a local SB3 envelope. Remote checkpoints can use a different
    # NumPy module layout even when their PyTorch weights are compatible.
    model = PPO.load(io.BytesIO(parent), device="cpu")
    state = torch.load(io.BytesIO(weights), weights_only=True, map_location="cpu")
    model.policy.load_state_dict(state, strict=True)
    model.policy.optimizer.state.clear()
    model.num_timesteps = 0
    return model


def average_policies(parent, models):
    states = [model.policy.state_dict() for model in models]
    combined = {}
    for name, parameter in states[0].items():
        values = [state[name] for state in states]
        if parameter.is_floating_point():
            combined[name] = torch.stack(values).mean(dim=0)
        else:
            if any(not torch.equal(parameter, value) for value in values):
                raise ValueError(f"Non-floating model buffer differs across branches: {name}")
            combined[name] = parameter
    model = PPO.load(io.BytesIO(parent), device="cpu")
    model.policy.load_state_dict(combined, strict=True)
    model.policy.optimizer.state.clear()
    model.num_timesteps = 0
    return model


def evaluate(model, seeds):
    returns = []
    with gym.make(TASK) as env:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            total = 0.0
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(int(action))
                total += reward
                done = terminated or truncated
            returns.append(total)
    return {
        "mean_return": float(np.mean(returns)), "median_return": float(np.median(returns)),
        "returns": returns,
    }


def run(args, commands):
    output = args.out
    output.mkdir(parents=True, exist_ok=False)
    report = {"environment": TASK, "host_labels": list(commands), "local_demo": args.local_demo,
              "shared_seed": args.seed, "warm_steps": args.warm_steps, "branch_steps": args.branch_steps,
              "learner_numpy_version": np.__version__, "learner_torch_version": torch.__version__,
              "hyperparameters": PPO_ARGS, "status": "running"}
    report_path = output / "report.json"
    record(report_path, report)
    try:
        with gym.make(TASK) as env:
            model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", **PPO_ARGS)
            model.learn(total_timesteps=args.warm_steps)
            parent = checkpoint_bytes(model)
        report["parent_sha256"] = hashlib.sha256(parent).hexdigest()
        (output / "parent.zip").write_bytes(parent)
        record(report_path, report)

        started = time.perf_counter()
        branch_models = {}
        report["branches"] = {}
        with ThreadPoolExecutor(max_workers=len(commands)) as executor:
            futures = {
                name: executor.submit(train_branch, command, parent, args.branch_steps, args.seed + index + 1)
                for index, (name, command) in enumerate(commands.items())
            }
            for name, future in futures.items():
                data, metrics = future.result()
                (output / f"branch-{name}.pt").write_bytes(data)
                branch_models[name] = load_branch(parent, data)
                report["branches"][name] = metrics
                record(report_path, report)
        report["branch_wall_seconds"] = time.perf_counter() - started
        budget = len(commands) * args.branch_steps
        report["total_post_parent_steps"] = budget
        merged = average_policies(parent, list(branch_models.values()))
        merged.save(output / "merged.zip")
        record(report_path, report)
        print(f"Completed {len(commands)} independent learners.", flush=True)

        started = time.perf_counter()
        vector = RemoteCartPole(list(commands.values()))
        try:
            startup = time.perf_counter() - started
            synchronous, report["synchronous"] = learn_from_parent(parent, vector, budget, args.seed)
        finally:
            vector.close()
        report["synchronous"]["startup_seconds"] = startup
        synchronous.save(output / "synchronous.zip")
        record(report_path, report)
        print(f"Completed {budget} synchronous remote interactions.", flush=True)

        with gym.make(TASK) as env:
            serial, report["serial"] = learn_from_parent(parent, env, budget, args.seed)
        serial.save(output / "serial.zip")
        seeds = list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
        report["evaluation_seeds"] = seeds
        report["models"] = {}
        candidates = {"parent": model, "merged": merged, "synchronous": synchronous, "serial": serial,
                      **{f"branch-{name}": branch for name, branch in branch_models.items()}}
        for name, candidate in candidates.items():
            report["models"][name] = evaluate(candidate, seeds)
            record(report_path, report)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        record(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    launch = parser.add_mutually_exclusive_group()
    launch.add_argument("--workers", type=Path, help="JSON map of three worker labels to command arrays.")
    launch.add_argument("--local-demo", action="store_true", help="Use three local subprocess workers.")
    parser.add_argument("--worker", choices=("environment", "train"), help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--warm-steps", type=int, default=1024)
    parser.add_argument("--branch-steps", type=int, default=4096)
    parser.add_argument("--eval-seed", type=int, default=31_001)
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.worker == "environment":
        worker_environment()
        return
    if args.worker == "train":
        worker_train(args.branch_steps, args.seed)
        return

    if not args.out or not (args.workers or args.local_demo):
        parser.error("Provide --out and either --workers or --local-demo.")
    if any(steps <= 0 or steps % ROLLOUT for steps in (args.warm_steps, args.branch_steps)):
        parser.error(f"Training budgets must be positive multiples of {ROLLOUT}.")
    if args.eval_episodes < 1:
        parser.error("Provide at least one evaluation episode.")
    if args.local_demo:
        commands = {f"local-{index}": [sys.executable, "-u", "-m", "scripts.distributed_ppo_pilot"]
                    for index in range(1, 4)}
    else:
        commands = json.loads(args.workers.read_text(encoding="utf-8"))
    if (not isinstance(commands, dict) or len(commands) != 3 or any(
            not re.fullmatch(r"[a-z0-9-]+", label) or not isinstance(command, list) or not command
            or not all(isinstance(part, str) and part for part in command)
            for label, command in commands.items()
    )):
        parser.error("Provide exactly three safe worker labels and nonempty command arrays.")
    run(args, commands)


if __name__ == "__main__":
    main()
