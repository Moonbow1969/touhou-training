"""Synchronous PPO with policy sampling beside each native TH14 trial process."""

import argparse
from collections import Counter
from hashlib import sha256
import io
import json
from pathlib import Path
from statistics import median
import tempfile
from time import monotonic, perf_counter
from uuid import uuid4

import gymnasium as gym
import msgpack
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import zmq

from scripts.benchmark_trial_workers import TrialWorker, require_stage
from scripts.checkpoint_storage import CheckpointStore, MIB, checkpoint_manifest, digest
from scripts.rollout_transport import (
    LABEL, MAX_BODY, MAX_STEPS, FragmentKey, Receipt, decode_header,
    pack_fragment, rollout_socket, schema, unpack_fragment,
)
from scripts.train_structured import BulletSetEncoder


CONTEXT_WIDTH = 10
ACTION_WIDTHS = (9, 2, 2, 2)
PPO_SETTINGS = dict(learning_rate=8e-5, gamma=.997, gae_lambda=.95, clip_range=.15,
                    ent_coef=.004, n_epochs=4, target_kl=.03)
TASK = {
    "executable_sha256": "6c62bff1e6be644086c5811c36eb938818f14012f94ad5fce784ca7889c66d48",
    "setting": "TH14 trial 0.01b Normal Reimu A Stage 1",
    "observation_schema": "trial-bullets-96x7-context-10-v1",
    "action_schema": "movement9-focus2-shoot2-bomb2",
    "reward": "one divided by horizon per surviving decision, stage advance earns one, first miss earns zero",
    "native_ticks_per_decision": 4,
}


class PolicySpaces(gym.Env):
    """Space declaration for SB3. Its rollout collector is never called."""
    observation_space = gym.spaces.Dict({
        "bullets": gym.spaces.Box(-1, 1, (96, 7), dtype=np.float32),
        "context": gym.spaces.Box(-1, 1, (CONTEXT_WIDTH,), dtype=np.float32),
    })
    action_space = gym.spaces.MultiDiscrete(ACTION_WIDTHS)


def policy_kwargs(hidden):
    return dict(features_extractor_class=BulletSetEncoder,
                net_arch={"pi": [hidden, hidden], "vf": [hidden, hidden]})


def task_config(horizon, synthetic):
    task = dict(TASK, horizon=horizon, synthetic=synthetic)
    return task, sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def action_mask(action):
    move, focus, shoot, bomb = map(int, action)
    dy, dx = divmod(move, 3)
    arrows = (1, 0, 2)[dx] | (4, 0, 8)[dy]
    return arrows | (focus << 4) | (shoot << 5) | (bomb << 6)


def trial_observation(state, action, previous):
    count = state["bullet_count"]
    if type(count) is not int or count < 0 or len(state["bullets"]) != min(count, 96):
        raise ValueError("Native projectile count disagrees with the observation.")
    bullets = np.zeros((96, 7), dtype=np.float32)
    if state["bullets"]:
        entities = np.asarray(state["bullets"], dtype=np.float32)
        bullets[:len(entities), :6] = entities / (384, 448, 16, 16, 32, 1)
        bullets[:len(entities), 6] = 1
    x, y = state["player"]
    px, py = previous["player"]
    _, focus, shoot, bomb = action
    context = np.asarray((x / 192, y / 448, state["lives"] / 8, count / 512,
                          state["frame"] / 8192, focus, shoot, bomb,
                          (x - px) / 32, (y - py) / 32), dtype=np.float32)
    if not np.isfinite(bullets).all() or not np.isfinite(context).all():
        raise ValueError("Native observation contains non-finite data.")
    return {"bullets": bullets.clip(-1, 1), "context": context.clip(-1, 1)}


class NativeClockMismatch(RuntimeError):
    """The game became unsynchronized before the fragment could be sent."""


class NativeTrial:
    def __init__(self, command, horizon):
        self.command = command
        self.horizon = horizon
        self.worker = None
        self.state = None

    def close(self):
        if self.worker is not None:
            worker, self.worker = self.worker, None
            worker.__exit__(None, None, None)

    def reset(self):
        if self.worker is None or self.state["mode"] != 2 or self.state["stage"] != 1:
            self.close()
            self.worker = TrialWorker(self.command).__enter__()
            self.state = self.worker.read(timeout=100)
        else:
            prior_frame = self.state["frame"]
            self.state = self.worker.request("R")
            if self.state["frame"] >= prior_frame:
                raise RuntimeError("Native Retry did not rewind the stage clock.")
        require_stage(self.state)
        if not 0 <= self.state["frame"] < 90 or self.state["delta"] != 0:
            raise RuntimeError("Native reset did not return the opening stage state.")
        self.decisions = 0
        return trial_observation(self.state, (4, 0, 0, 0), self.state)

    def step(self, action):
        before = self.state
        after = self.worker.request(f"S {action_mask(action)}")
        self.state = after
        self.decisions += 1
        delta = (after["ticks"] - before["ticks"]) % 2**32
        if delta != TASK["native_ticks_per_decision"] or after["delta"] != delta:
            raise NativeClockMismatch(f"Expected four native ticks per action, got {delta}/{after['delta']}.")
        cleared = after["stage"] > 1
        missed = after["lives"] < before["lives"]
        left_stage = after["mode"] != 2 or after["stage"] != 1
        if not left_stage and not missed and after["frame"] < before["frame"]:
            raise RuntimeError("Native stage clock moved backwards during an action.")
        outcome = "clear" if cleared else "miss" if missed else "native_end" if left_stage else None
        if outcome is None and self.decisions == self.horizon:
            outcome = "horizon"
        reward = 1. if cleared else 0. if missed or left_stage else 1. / self.horizon
        direction = int(action[0]) % 3 - 1
        x = before["player"][0]
        eligible = bool(direction and before["frame"] >= 100 and -173 < x < 173
                        and not left_stage and not missed and not action[3])
        responds = eligible and direction * (after["player"][0] - x) > .05
        info = dict(outcome=outcome, frame=after["frame"], delta=delta, eligible=eligible,
                    responds=responds, bullets=after["bullet_count"], decisions=self.decisions)
        return trial_observation(after, action, before), reward, outcome is not None, info


def message(kind, **values):
    return msgpack.packb(dict(type=kind, **values), use_bin_type=True)


def actor_route(run, worker):
    return f"{run}/{worker}".encode("ascii")


def receive(socket, deadline):
    remaining = deadline - monotonic()
    if remaining <= 0 or not socket.poll(int(remaining * 1000), zmq.POLLIN):
        raise TimeoutError("Timed out waiting for distributed PPO peers.")
    return socket.recv_multipart()


def snapshot(policy):
    buffer = io.BytesIO()
    torch.save(policy.state_dict(), buffer)
    blob = buffer.getvalue()
    if len(blob) > 16 * MAX_BODY:
        raise ValueError("Policy snapshot exceeds the wire limit.")
    return sha256(blob).hexdigest(), [blob[i:i + MAX_BODY] for i in range(0, len(blob), MAX_BODY)]


class Actor:
    def __init__(self, env):
        self.env = env
        self.observation = None
        self.episode_start = True
        self.steps = 0
        self.sequence = 0

    @torch.inference_mode()
    def collect(self, policy, steps):
        columns = {key: np.zeros(shape, dtype=dtype) for key, (dtype, shape) in schema(steps, CONTEXT_WIDTH).items()}
        episodes, frames = [], Counter()
        eligible = responses = max_projectiles = 0
        started = perf_counter()
        self.attempt_decisions = 0
        for index in range(steps):
            if self.observation is None:
                self.observation = self.env.reset()
            for name, observation in self.observation.items():
                columns[name][index] = observation
            action, value, log_prob = policy(policy.obs_to_tensor(self.observation)[0])
            action = action[0].cpu().numpy()
            self.attempt_decisions += 1
            following, reward, terminated, info = self.env.step(action)
            columns["action"][index] = action
            columns["value"][index] = value.item()
            columns["log_prob"][index] = log_prob.item()
            columns["reward"][index] = reward
            columns["terminated"][index] = terminated
            columns["episode_start"][index] = self.episode_start
            columns["native_delta"][index] = info["delta"]
            self.episode_start = terminated
            self.observation = None if terminated else following
            if terminated:
                episodes.append({key: info[key] for key in ("outcome", "frame", "decisions")})
            frames[info["delta"]] += 1
            eligible += info["eligible"]
            responses += info["responds"]
            max_projectiles = max(max_projectiles, info["bullets"])
        # Reuse same-snapshot values for live, contiguous steps. A terminal transition's bootstrap stays zero.
        live = columns["terminated"][:-1] == 0
        columns["next_value"][:-1][live] = columns["value"][1:][live]
        if self.observation is not None:
            columns["next_value"][-1] = policy.predict_values(policy.obs_to_tensor(self.observation)[0]).item()
        if eligible >= 12 and responses == 0:
            raise RuntimeError("Native movement guard observed no horizontal response.")
        return columns, dict(seconds=perf_counter() - started, native_deltas=dict(frames),
                             horizontal_commands=eligible, horizontal_responses=responses,
                             max_projectiles=max_projectiles, episodes=episodes)


def collect_with_retry(actor, policy, steps, max_retries):
    started = perf_counter()
    failures = []
    for attempt in range(max_retries + 1):
        try:
            columns, metrics = actor.collect(policy, steps)
        except (RuntimeError, OSError, ValueError) as error:
            if attempt == max_retries:
                raise
            failures.append(dict(discarded_decisions=actor.attempt_decisions,
                                 reason=f"{type(error).__name__}: {error}"[:280]))
            # A failed fragment has not shipped any transitions. Start a new episode before replacement.
            actor.observation = None
            actor.episode_start = True
            if not isinstance(error, NativeClockMismatch):
                actor.env.close()
            continue
        metrics.update(seconds=perf_counter() - started, retries=failures)
        return columns, metrics


def run_actor(args, socket):
    socket.setsockopt(zmq.IDENTITY, actor_route(args.run, args.worker))
    socket.connect(args.connect)
    _, environment = task_config(args.horizon, args.synthetic)
    socket.send(message("hello", run=args.run, worker=args.worker, environment=environment))
    env = NativeTrial(args.native, args.horizon)
    actor = Actor(env)
    policy = None
    try:
        while True:
            header, *parts = receive(socket, monotonic() + args.timeout)
            control = decode_header(header)
            if control["type"] == "stop":
                return
            if control["type"] != "collect" or control["environment"] != environment:
                raise ValueError("Actor received the wrong control message or task signature.")
            if not 1 <= len(parts) <= 16:
                raise ValueError("Invalid policy snapshot frame count.")
            payload = b"".join(parts)
            if sha256(payload).hexdigest() != control["policy"]:
                raise ValueError("Policy snapshot does not match the learner fingerprint.")
            if policy is None:
                policy = MultiInputActorCriticPolicy(PolicySpaces.observation_space, PolicySpaces.action_space,
                                                     lambda _: 0., **policy_kwargs(control["hidden"]))
                policy.set_training_mode(False)
            policy.load_state_dict(torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True))
            for _ in range(control["fragments"]):
                key = FragmentKey(args.run, args.worker, control["lease"], control["policy"],
                                  environment, actor.sequence, actor.steps)
                columns, metrics = collect_with_retry(actor, policy, control["steps"], args.fragment_retries)
                packet, receipt = pack_fragment(key, columns)
                # Episode counts can be large on short horizons; send metrics outside the bounded rollout header.
                socket.send_multipart([msgpack.packb(metrics, use_bin_type=True), *packet])
                (ack,) = receive(socket, monotonic() + args.timeout)
                if ack == message("stop"):
                    return
                if Receipt.unpack(ack) != receipt:
                    raise ValueError("Learner acknowledged a different rollout fragment.")
                actor.sequence += 1
                actor.steps += control["steps"]
                print(json.dumps(dict(worker=args.worker, sequence=actor.sequence, **metrics)), flush=True)
    except (RuntimeError, OSError, ValueError) as error:
        try:
            socket.send(message("actor_error", worker=args.worker, reason=f"{type(error).__name__}: {error}"[:280]))
            socket.setsockopt(zmq.LINGER, 1000)
        except zmq.ZMQError:
            pass
        raise
    finally:
        env.close()


def check_view(arrays, steps):
    if arrays["context"].shape != (steps, CONTEXT_WIDTH):
        raise ValueError("Unexpected rollout observation shape.")
    terminated = arrays["terminated"].astype(bool)
    truncated = arrays["truncated"].astype(bool)
    done = terminated | truncated
    if (terminated & truncated).any() or (arrays["next_value"][terminated] != 0).any():
        raise ValueError("Unexpected terminal bootstrap or ambiguous episode flags.")
    if not np.array_equal(arrays["episode_start"][1:], done[:-1]):
        raise ValueError("Episode-start flags do not match transitions.")
    if (arrays["native_delta"] != TASK["native_ticks_per_decision"]).any():
        raise ValueError("Actor did not advance the expected number of native ticks.")
    np.testing.assert_allclose(arrays["next_value"][:-1][~done[:-1]], arrays["value"][1:][~done[:-1]],
                               rtol=1e-5, atol=1e-6)


def fill_buffer(model, fragments):
    """Treat fragment lanes independently; bootstrap at cuts and time limits."""
    buffer = model.rollout_buffer
    buffer.reset()
    # Time-limit bootstrapping follows SB3; a finite task horizon is marked truly terminal.
    column = {name: np.stack([data[name] for data in fragments], axis=1) for name in fragments[0]}
    obs = {name: column[name] for name in ("bullets", "context")}
    rewards = column["reward"] + model.gamma * column["next_value"] * column["truncated"]
    for step in range(model.n_steps):
        buffer.add({name: value[step] for name, value in obs.items()}, column["action"][step],
                   rewards[step], column["episode_start"][step], torch.from_numpy(column["value"][step]),
                   torch.from_numpy(column["log_prob"][step]))
    buffer.compute_returns_and_advantage(torch.from_numpy(column["next_value"][-1]),
                                         column["terminated"][-1] | column["truncated"][-1])
    return column


@torch.inference_mode()
def verify_snapshot(model, fragments):
    model.policy.set_training_mode(False)
    maximum = {"log_prob": 0., "value": 0.}
    for data in fragments:
        indices = [0, len(data["value"]) // 2, len(data["value"]) - 1]
        observations = model.policy.obs_to_tensor({name: data[name][indices] for name in ("bullets", "context")})[0]
        actions = torch.as_tensor(data["action"][indices], device=model.device)
        values, log_prob, _ = model.policy.evaluate_actions(observations, actions)
        for name, actual in (("log_prob", log_prob), ("value", values.flatten())):
            expected = data[name][indices]
            computed = actual.cpu().numpy()
            np.testing.assert_allclose(computed, expected, rtol=1e-3, atol=2e-4,
                                       err_msg=f"Actor and learner snapshot disagree on {name}.")
            maximum[name] = max(maximum[name], float(np.abs(computed - expected).max()))
    return maximum


def write_report(path, report):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".report-", delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False)
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def worker_quotas(values):
    quotas = {}
    for value in values:
        worker, equals, quota = value.partition("=")
        if not LABEL.fullmatch(worker) or worker in quotas:
            raise ValueError(f"Invalid or repeated worker label: {worker!r}.")
        quotas[worker] = int(quota) if equals else 1
    if not 1 <= sum(quotas.values()) <= 64 or min(quotas.values()) < 1:
        raise ValueError("Require 1–64 total fragments per round with positive per-worker quotas.")
    return quotas


def collect_round(socket, args, quotas, expected, model, environment):
    policy_hash, parts = snapshot(model.policy)
    lease = uuid4().hex
    for worker, count in quotas.items():
        control = message("collect", lease=lease, policy=policy_hash, environment=environment,
                          fragments=count, hidden=args.hidden, steps=args.fragment_steps)
        socket.send_multipart([actor_route(args.run, worker), control, *parts])
    received = Counter()
    fragments, metrics = [], []
    deadline = monotonic() + args.timeout
    while len(fragments) < sum(quotas.values()):
        try:
            identity, summary, *packet = receive(socket, deadline)
        except TimeoutError as error:
            missing = {worker: count - received[worker] for worker, count in quotas.items() if received[worker] < count}
            raise TimeoutError(f"Timed out collecting on-policy fragments from {missing}.") from error
        run, _, worker = identity.decode("ascii").partition("/")
        if run != args.run:
            continue
        if not packet:
            control = decode_header(summary)
            if control.get("type") == "actor_error" and control.get("worker") == worker and worker in quotas:
                raise RuntimeError(f"Actor {worker} failed: {control['reason']}")
            raise ValueError("Unexpected actor control message during collection.")
        key, data, receipt = unpack_fragment(packet, policy=policy_hash, environment=environment)
        if (worker not in quotas or received[worker] >= quotas[worker] or key.worker != worker
                or key.run != args.run or key.lease != lease or key.sequence != expected[worker]
                or key.first_step != expected[worker] * args.fragment_steps):
            raise ValueError("Unexpected, repeated, out-of-order, or stale rollout fragment.")
        check_view(data, args.fragment_steps)
        stats = msgpack.unpackb(summary, raw=False, strict_map_key=False)
        metrics.append(dict(worker=worker, sequence=key.sequence, **stats))
        expected[worker] += 1
        received[worker] += 1
        fragments.append(data)
        socket.send_multipart([identity, receipt.pack()])
    return policy_hash, fragments, metrics


def setup_model(args, quotas, environment):
    env = DummyVecEnv([PolicySpaces for _ in range(sum(quotas.values()))])
    settings = dict(PPO_SETTINGS, n_steps=args.fragment_steps,
                    batch_size=min(64, args.fragment_steps * env.num_envs))
    config = dict(environment=environment, hidden=args.hidden, seed=args.seed, settings=settings)
    if args.resume:
        manifest = checkpoint_manifest(args.resume)
        if not manifest or manifest["compatibility"] != config:
            raise ValueError("Resume checkpoint does not match this task, model, or PPO configuration.")
        model = PPO.load(args.resume, env=env, device=args.device)
        initial = dict(kind="ppo-resume", sha256=digest(args.resume),
                       warm_start_actor_exact=manifest["warm_start_actor_exact"])
    else:
        model = PPO(MultiInputActorCriticPolicy, env, policy_kwargs=policy_kwargs(args.hidden),
                    device=args.device, seed=args.seed, **settings)
        initial = {"kind": "random"}
    if args.initial_policy:
        companion = args.initial_policy.with_suffix(".json")
        if not companion.is_file() or json.loads(companion.read_text())["environment"] != environment:
            raise ValueError("SFT weights require an exact trial observation/action signature; full-game SFT is incompatible.")
        fresh_critic = {name: value.clone() for name, value in model.policy.state_dict().items()
                        if name.startswith(("value_net.", "mlp_extractor.value_net."))}
        weights = torch.load(args.initial_policy, map_location="cpu", weights_only=True)
        model.policy.load_state_dict(weights, strict=True)
        model.policy.load_state_dict(fresh_critic, strict=False)
        initial = dict(kind="sft", sha256=digest(args.initial_policy))
    model.set_logger(configure(None, []))
    return model, config, initial


def run_learner(args, socket):
    quotas = worker_quotas(args.workers)
    task, environment = task_config(args.horizon, args.synthetic)
    model, config, initial = setup_model(args, quotas, environment)
    args.out.mkdir(parents=True, exist_ok=True)
    storage = CheckpointStore(args.out, args.checkpoint_budget_mib * MIB, args.disk_reserve_mib * MIB)
    if storage.current.exists() or (args.out / "report.json").exists():
        raise ValueError("Use a fresh output directory, including when resuming.")
    storage.remaining()
    socket.bind(args.bind)
    print(json.dumps(dict(status="ready", run=args.run, workers=quotas, task=task)), flush=True)
    seen = set()
    deadline = monotonic() + args.timeout
    while len(seen) < len(quotas):
        try:
            identity, hello = receive(socket, deadline)
        except TimeoutError as error:
            raise TimeoutError(f"Missing actor handshakes: {', '.join(sorted(quotas.keys() - seen))}.") from error
        run, _, worker = identity.decode("ascii").partition("/")
        if run != args.run:
            continue
        if worker not in quotas or worker in seen or decode_header(hello) != dict(
                type="hello", run=args.run, worker=worker, environment=environment):
            raise ValueError("Unexpected or incompatible actor handshake.")
        seen.add(worker)
        print(json.dumps(dict(status="actor_ready", worker=worker, ready=len(seen), expected=len(quotas))), flush=True)
    expected = Counter()
    report = dict(run=args.run, task=task, environment=environment, initial_policy=initial,
                  learner_device=str(model.device), config=config, quotas=quotas, updates=[], status="training")
    report_path = args.out / "report.json"
    write_report(report_path, report)
    try:
        for index in range(args.rounds):
            started = perf_counter()
            policy_hash, fragments, metrics = collect_round(socket, args, quotas, expected, model, environment)
            collection = perf_counter() - started
            discrepancy = verify_snapshot(model, fragments)
            columns = fill_buffer(model, fragments)
            model.num_timesteps += model.n_steps * model.n_envs
            training_start = perf_counter()
            model.train()
            optimize = perf_counter() - training_start
            action_counts = [np.bincount(values.ravel(), minlength=width).tolist()
                             for values, width in zip(columns["action"].transpose(2, 0, 1), ACTION_WIDTHS)]
            row = dict(round=index + 1, steps=model.num_timesteps, rollout_policy_sha256=policy_hash,
                       collection_seconds=collection, optimize_seconds=optimize,
                       rollout_decisions_per_second=model.n_steps * model.n_envs / collection,
                       untrained_snapshot_max_error=discrepancy, action_counts=action_counts, actor_fragments=metrics,
                       losses={name: float(metric) for name, metric in model.logger.name_to_value.items()
                               if np.isfinite(metric)})
            report["checkpoint_storage"] = storage.save(model.save, dict(steps=model.num_timesteps,
                compatibility=config, warm_start_actor_exact=initial.get("warm_start_actor_exact", initial["kind"] == "sft")))
            report["updates"].append(row)
            write_report(report_path, report)
            print(json.dumps({key: value for key, value in row.items() if key != "actor_fragments"}), flush=True)
        report["status"] = "complete"
        write_report(report_path, report)
    except BaseException as error:
        report["status"] = "interrupted"
        report["error"] = str(error)
        write_report(report_path, report)
        raise
    finally:
        for worker in seen:
            try:
                socket.send_multipart([actor_route(args.run, worker), message("stop")])
            except zmq.ZMQError:
                pass
        socket.setsockopt(zmq.LINGER, 1000)


@torch.inference_mode()
def evaluate_episode(env, policy, seed, deterministic):
    torch.manual_seed(seed)
    observation = env.reset()
    actions = [np.zeros(width, dtype=np.int64) for width in ACTION_WIDTHS]
    eligible = responses = max_projectiles = 0
    total_reward = 0.
    done = False
    while not done:
        action = policy(policy.obs_to_tensor(observation)[0], deterministic=deterministic)[0][0].cpu().numpy()
        for counts, value in zip(actions, action):
            counts[value] += 1
        observation, reward, done, info = env.step(action)
        total_reward += reward
        eligible += info["eligible"]
        responses += info["responds"]
        max_projectiles = max(max_projectiles, info["bullets"])
    if eligible >= 12 and responses == 0:
        raise RuntimeError("Native movement guard observed no horizontal response during evaluation.")
    return dict(policy_seed=seed, outcome=info["outcome"], native_stage_frame=info["frame"],
                decisions=info["decisions"], reward=total_reward, horizontal_commands=eligible,
                horizontal_responses=responses, max_projectiles=max_projectiles,
                action_counts=[counts.tolist() for counts in actions])


def evaluate_with_retry(env, policy, seed, deterministic, max_retries):
    failures = []
    for attempt in range(max_retries + 1):
        try:
            row = evaluate_episode(env, policy, seed, deterministic)
        except (RuntimeError, OSError, ValueError) as error:
            if attempt == max_retries:
                raise
            failures.append(dict(discarded_decisions=getattr(env, "decisions", 0),
                                 reason=f"{type(error).__name__}: {error}"[:280]))
            if not isinstance(error, NativeClockMismatch):
                env.close()
            continue
        row["retries"] = failures
        return row


def run_evaluation(args):
    task, environment = task_config(args.horizon, args.synthetic)
    policies, sources = {}, {}
    for specification in args.policy:
        name, sep, source = specification.partition("=")
        if not LABEL.fullmatch(name) or not sep or not source or name in policies:
            raise ValueError("Use unique --policy LABEL=CHECKPOINT arguments.")
        if source == "random":
            torch.manual_seed(args.seed)
            policy = MultiInputActorCriticPolicy(PolicySpaces.observation_space, PolicySpaces.action_space,
                                                 lambda _: 0., **policy_kwargs(args.hidden))
            sources[name] = dict(kind="random", seed=args.seed, hidden=args.hidden, sha256=snapshot(policy)[0])
        else:
            path = Path(source)
            manifest = checkpoint_manifest(path)
            if not manifest or manifest["compatibility"]["environment"] != environment:
                raise ValueError("Evaluation checkpoint and native task signature differ.")
            policy = PPO.load(path, device="cpu").policy
            sources[name] = dict(kind="ppo", steps=manifest["steps"], sha256=digest(path))
        policy.set_training_mode(False)
        policies[name] = policy

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "evaluation.json"
    if path.exists():
        raise ValueError("Use a fresh evaluation output directory.")
    rows = {name: [] for name in policies}
    report = dict(run=args.run, task=task, environment=environment, sources=sources, episodes=rows,
                  deterministic=args.deterministic, native_game_rng_seed_controlled=False)
    names = list(policies)
    env = NativeTrial(args.native, args.horizon)
    try:
        for block, seed in enumerate(args.policy_seeds):
            for offset in range(len(names)):
                name = names[(block + offset) % len(names)]
                result = evaluate_with_retry(env, policies[name], seed, args.deterministic, args.episode_retries)
                rows[name].append(result)
                write_report(path, report)
                print(json.dumps(dict(policy=name, **result)), flush=True)
    finally:
        env.close()
    summary = {}
    for name, episodes in rows.items():
        misses = [row["native_stage_frame"] for row in episodes if row["outcome"] == "miss"]
        summary[name] = dict(episodes=len(episodes), outcomes=dict(Counter(row["outcome"] for row in episodes)),
                             median_furthest_frame=median(row["native_stage_frame"] for row in episodes),
                             median_observed_first_miss_frame=median(misses) if misses else None,
                             missing_first_miss_observations=len(episodes)-len(misses))
    report["summary"] = summary
    write_report(path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="role", required=True)
    learner = sub.add_parser("learner")
    actor = sub.add_parser("actor")
    evaluator = sub.add_parser("evaluate")
    for role in (learner, actor, evaluator):
        role.add_argument("--run", required=True)
        role.add_argument("--horizon", type=int, default=1024)
        role.add_argument("--timeout", type=int, default=360)
        role.add_argument("--seed", type=int, default=14)
        role.add_argument("--threads", type=int, default=1)
        role.add_argument("--synthetic", action="store_true", help="Label a synthetic protocol test separately.")
    learner.add_argument("--workers", nargs="+", required=True, metavar="WORKER[=FRAGMENTS]")
    learner.add_argument("--bind", default="tcp://127.0.0.1:5578", help="Use a private forwarding endpoint.")
    learner.add_argument("--out", type=Path, required=True)
    learner.add_argument("--device", default="cpu")
    learner.add_argument("--rounds", type=int, default=8)
    learner.add_argument("--hidden", type=int, default=128)
    learner.add_argument("--fragment-steps", type=int, default=MAX_STEPS)
    learner.add_argument("--checkpoint-budget-mib", type=int, default=256)
    learner.add_argument("--disk-reserve-mib", type=int, default=2048)
    initialize = learner.add_mutually_exclusive_group()
    initialize.add_argument("--initial-policy", type=Path)
    initialize.add_argument("--resume", type=Path)
    actor.add_argument("--worker", required=True)
    actor.add_argument("--connect", default="tcp://127.0.0.1:5578")
    actor.add_argument("--fragment-retries", type=int, default=3)
    actor.add_argument("--native", nargs=argparse.REMAINDER, required=True, help="Native worker command. Keep this flag last.")
    evaluator.add_argument("--out", type=Path, required=True)
    evaluator.add_argument("--policy", action="append", required=True, metavar="LABEL=CHECKPOINT")
    evaluator.add_argument("--policy-seeds", type=int, nargs="+", default=list(range(100, 108)))
    evaluator.add_argument("--hidden", type=int, default=128)
    evaluator.add_argument("--deterministic", action="store_true")
    evaluator.add_argument("--episode-retries", type=int, default=3)
    evaluator.add_argument("--native", nargs=argparse.REMAINDER, required=True, help="Native worker command. Keep this flag last.")
    args = parser.parse_args()
    if not LABEL.fullmatch(args.run) or min(args.horizon, args.timeout, args.threads) < 1:
        parser.error("Invalid run label, horizon, timeout, or thread count.")
    if args.role == "learner" and (not 2 <= args.fragment_steps <= MAX_STEPS or not 1 <= args.rounds
                                   or not 16 <= args.hidden <= 512):
        parser.error("Invalid fragment size, policy width, or number of rounds.")
    if args.role == "actor" and (not LABEL.fullmatch(args.worker) or not args.native or not 0 <= args.fragment_retries <= 8):
        parser.error("Require a valid worker label, native command, and 0–8 fragment retries.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.role == "evaluate":
        if (not args.native or not 16 <= args.hidden <= 512 or not 0 <= args.episode_retries <= 8
                or len(set(args.policy_seeds)) != len(args.policy_seeds)):
            parser.error("Require native command, valid model width, and distinct policy seeds.")
        run_evaluation(args)
        return
    context = zmq.Context()
    try:
        role = zmq.ROUTER if args.role == "learner" else zmq.DEALER
        with rollout_socket(context, role, timeout_ms=10000) as socket:
            if args.role == "learner":
                run_learner(args, socket)
            else:
                run_actor(args, socket)
    finally:
        context.term()


if __name__ == "__main__":
    main()
