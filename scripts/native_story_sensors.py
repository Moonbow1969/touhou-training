"""Opt-in passive TH14 sensor fields in bounded stock story transition archives."""

from dataclasses import asdict, dataclass
from pathlib import Path
from struct import Struct, error as StructError
from time import perf_counter_ns

import numpy as np

from environment import structured
from scripts import native_story_transitions as story
from scripts.checkpoint_storage import MIB, digest


SCHEMA = "th14-stock-near-player-sensors-v1"
PROJECTILE_FIELDS = ("relative_x", "relative_y", "velocity_x", "velocity_y", "hitbox_radius", "actor_rank")
BODY_FIELDS = ("x", "y", "velocity_x", "velocity_y", "half_width", "half_height", "rotation", "rectangular")
LASER_FIELDS = ("x", "y", "velocity_x", "velocity_y", "angle", "angular_velocity", "length", "half_width",
                "activation_ticks")
STOCK_FIELDS = ("power", "life_fragments", "bomb_fragments")
STATUSES = dict(not_sampled=0, valid=1, read_failed=2)
SUMMARY_FIELDS = ("projectiles", "bodies", "straight_lasers", "stock")
I32 = Struct("<i")
READ_ERRORS = (RuntimeError, OSError, ValueError, StructError)


@dataclass(frozen=True)
class SensorSettings:
    heavy_stride: int = 2
    projectile_limit: int = 8
    body_limit: int = 6
    laser_limit: int = 6
    near_radius: float = 96.
    urgent_radius: float = 32.
    horizon_ticks: int = 8
    memory_mib: int = 32

    def validate(self):
        counts = (self.heavy_stride, self.projectile_limit, self.body_limit, self.laser_limit,
                  self.horizon_ticks, self.memory_mib)
        if any(type(x) is not int or x < 1 for x in counts) or not 0 < self.urgent_radius <= self.near_radius:
            raise ValueError("Use finite positive sensor limits, with an urgent radius within the near radius.")
        if not np.isfinite(self.near_radius):
            raise ValueError("The near-player sensor radius must be finite.")


class EntityRows:
    def __init__(self, rows, width, limit):
        self.values = np.empty((rows * limit, width), dtype=np.float32)
        self.indptr = np.zeros(rows + 1, dtype=np.int32)
        self.source_count = np.full(rows, -1, dtype=np.int32)
        self.near_count = np.full(rows, -1, dtype=np.int32)
        self.limit = limit

    def add(self, row, source, near_count, selected):
        start, end = self.indptr[row], self.indptr[row] + len(selected)
        self.values[start:end] = selected
        self.indptr[row + 1] = end
        self.source_count[row], self.near_count[row] = source, near_count

    def skip(self, row):
        self.indptr[row + 1] = self.indptr[row]

    def arrays(self, name, rows):
        return {f"sensor_{name}": self.values[:self.indptr[rows]],
                f"sensor_{name}_indptr": self.indptr[:rows + 1],
                f"sensor_{name}_source_count": self.source_count[:rows],
                f"sensor_{name}_near_count": self.near_count[:rows]}


def closest(distance, near, limit):
    candidates = np.flatnonzero(np.isfinite(distance) & (distance <= near))
    order = np.argsort(distance[candidates], kind="stable")[:limit]
    return candidates[order], len(candidates)


def approach(relative, velocity, horizon):
    speed_squared = np.einsum("ij,ij->i", velocity, velocity)
    dot = np.einsum("ij,ij->i", relative, velocity)
    time = np.clip(-dot / np.maximum(speed_squared, 1e-12), 0, horizon)
    return np.linalg.norm(relative + time[:, None] * velocity, axis=1)


def projectiles_near(raw, settings):
    if raw is None or raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError("Missing raw projectile field.")
    active = np.flatnonzero((raw[:, 5] > 0) & (raw[:, 4] > 0))
    objects = raw[active]
    distance = approach(objects[:, :2], objects[:, 2:4], settings.horizon_ticks) - objects[:, 4]
    indices, count = closest(distance, settings.near_radius, settings.projectile_limit)
    # The snapshot's raw list is already sorted exactly as the actor's nearest-96 list.
    values = np.column_stack((objects[indices, :5], active[indices])).astype(np.float32, copy=False)
    return len(raw), count, values, bool(len(indices) and distance[indices[0]] <= settings.urgent_radius)


def bodies_near(raw, position, settings):
    distance = approach(raw[:, :2] - position, raw[:, 2:4], settings.horizon_ticks)
    radius = np.where(raw[:, 7] > 0, np.linalg.norm(raw[:, 4:6], axis=1), raw[:, 4])
    distance -= radius
    indices, count = closest(distance, settings.near_radius, settings.body_limit)
    return len(raw), count, raw[indices], bool(len(indices) and distance[indices[0]] <= settings.urgent_radius)


def lasers_near(raw, position, settings):
    at = np.array((0, settings.horizon_ticks), dtype=np.float32)[:, None]
    angle = raw[None, :, 4] + at * raw[None, :, 5]
    direction = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    origin = raw[None, :, :2] + at[..., None] * raw[None, :, 2:4]
    relative = position - origin
    along = np.clip((direction * relative).sum(axis=-1), 0, raw[None, :, 6])
    distance = np.linalg.norm(relative - along[..., None] * direction, axis=-1).min(axis=0) - raw[:, 7]
    # Retain beams whose interior sweep can enter the desired neighborhood.
    angular_motion = np.minimum(np.abs(raw[:, 5]) * settings.horizon_ticks, 2.) * raw[:, 6]
    distance -= angular_motion
    indices, count = closest(distance, settings.near_radius, settings.laser_limit)
    return len(raw), count, raw[indices], bool(len(indices) and distance[indices[0]] <= settings.urgent_radius)


def stock_offsets(game):
    offsets = {name: game._OFFSETS[name] for name in STOCK_FIELDS}
    if any(type(value) is not int or value < 0 for value in offsets.values()):
        raise ValueError("Native stock coalescing requires direct signed-int32 offsets.")
    if max(offsets.values()) + I32.size - min(offsets.values()) > 64:
        raise ValueError("Native stock fields exceed the bounded coalesced read.")
    return offsets


def read_stock(game, offsets):
    start, end = min(offsets.values()), max(offsets.values()) + I32.size
    payload = game._read_game_memory(start, end - start, rel=True)
    return [I32.unpack_from(payload, offsets[name] - start)[0] for name in STOCK_FIELDS]


class SensorCapture:
    def __init__(self, env, settings, decision_limit):
        self.settings = settings
        self.offsets = stock_offsets(env.game)
        self.total_rows = self.rows = self.elapsed_ns = 0
        self.reads = np.zeros(4, dtype=np.int64)
        self.read_failures = np.zeros(4, dtype=np.int64)
        self.read_ns = np.zeros(4, dtype=np.int64)
        self.previous_heavy_danger = False
        self.previous_stage = None
        fields = (PROJECTILE_FIELDS, BODY_FIELDS, LASER_FIELDS)
        limits = (settings.projectile_limit, settings.body_limit, settings.laser_limit)
        index_bytes = 4 * (1 + len(STOCK_FIELDS) + 3 * len(fields)) + len(SUMMARY_FIELDS)
        size = index_bytes + sum(4 * len(columns) * limit for columns, limit in zip(fields, limits))
        self.capacity = min(decision_limit, (settings.memory_mib * MIB - 4 * len(fields)) // size)
        if self.capacity < 1:
            raise ValueError("The native sensor memory cap cannot hold one decision.")
        self.entities = tuple(EntityRows(self.capacity, len(columns), limit) for columns, limit in zip(fields, limits))
        self.decision = np.empty(self.capacity, dtype=np.int32)
        self.status = np.zeros((self.capacity, len(SUMMARY_FIELDS)), dtype=np.uint8)
        self.stock = np.full((self.capacity, len(STOCK_FIELDS)), -1, dtype=np.int32)

    def sample_entity(self, row, channel, get):
        started = perf_counter_ns()
        self.reads[channel] += 1
        try:
            source, count, selected, danger = get()
            self.entities[channel].add(row, source, count, selected)
            self.status[row, channel] = STATUSES["valid"]
            return danger
        except READ_ERRORS:
            self.read_failures[channel] += 1
            self.entities[channel].skip(row)
            self.status[row, channel] = STATUSES["read_failed"]
            return False
        finally:
            self.read_ns[channel] += perf_counter_ns() - started

    def sample_stock(self, env, row):
        started = perf_counter_ns()
        self.reads[3] += 1
        try:
            self.stock[row] = read_stock(env.game, self.offsets)
            self.status[row, 3] = STATUSES["valid"]
        except READ_ERRORS:
            self.read_failures[3] += 1
            self.status[row, 3] = STATUSES["read_failed"]
        finally:
            self.read_ns[3] += perf_counter_ns() - started

    def capture(self, env, native, position):
        index = self.total_rows
        self.total_rows += 1
        if self.rows == self.capacity:
            return
        started = perf_counter_ns()
        row, self.rows = self.rows, self.rows + 1
        self.decision[row] = index
        if not native[-1]:
            for entity in self.entities:
                entity.skip(row)
            self.elapsed_ns += perf_counter_ns() - started
            return

        settings = self.settings
        projectile_danger = self.sample_entity(row, 0, lambda: projectiles_near(env.snapshot.projectiles, settings))
        stage = env.snapshot.stage
        heavy = (index % settings.heavy_stride == 0 or self.previous_stage != stage or self.previous_heavy_danger
                 or projectile_danger or env.snapshot.player_state == 4)
        self.previous_stage = stage
        if heavy:
            player = np.asarray(position[:2], dtype=np.float32)
            body_danger = self.sample_entity(row, 1, lambda: bodies_near(env.reader.read_enemy_hitboxes(), player, settings))
            laser_danger = self.sample_entity(row, 2, lambda: lasers_near(
                env.reader.read_straight_lasers(horizon=settings.horizon_ticks), player, settings,
            ))
            self.previous_heavy_danger = body_danger or laser_danger
        else:
            self.entities[1].skip(row)
            self.entities[2].skip(row)
        self.sample_stock(env, row)
        self.elapsed_ns += perf_counter_ns() - started

    def arrays(self):
        result = dict(sensor_decision_index=self.decision[:self.rows], sensor_status=self.status[:self.rows],
                      sensor_stock=self.stock[:self.rows])
        for name, rows in zip(SUMMARY_FIELDS, self.entities):
            result.update(rows.arrays(name, self.rows))
        return result

    def summary(self):
        return dict(schema=SCHEMA, observed_rows=self.rows, omitted_for_memory_limit=self.total_rows - self.rows,
                    raw_bytes=sum(array.nbytes for array in self.arrays().values()),
                    sensor_wall_ms=self.elapsed_ns / 1e6,
                    sensor_us_per_decision=self.elapsed_ns / (1000 * max(self.total_rows, 1)),
                    by_source={name: dict(attempted=int(attempted), failed=int(failed), wall_ms=ns / 1e6)
                               for name, attempted, failed, ns in zip(SUMMARY_FIELDS, self.reads,
                                                                    self.read_failures, self.read_ns)})


class SensorStoryTransitions(story.StoryTransitions):
    def __init__(self, env, settings, decision_limit):
        super().__init__(env)
        self.sensors = SensorCapture(env, settings, decision_limit)
        offsets = env.reader.offsets
        self.offsets["sensor_stock_direct_offsets"] = self.sensors.offsets
        self.offsets["sensor_enemy_offsets"] = asdict(offsets.enemies)
        self.offsets["sensor_laser_offsets"] = {name: asdict(getattr(offsets, name))
                                                for name in ("laser_base", "laser_line", "laser_infinite")}

    def before_step(self, env, observation):
        native, position = super().before_step(env, observation)
        self.sensors.capture(env, native, position)
        return native, position

    def arrays(self):
        return dict(super().arrays(), **self.sensors.arrays())


class SensorStoryArchive(story.StoryArchive):
    def __init__(self, args, specification, settings):
        super().__init__(args, specification)
        self.settings = settings

    def metadata(self, trial, offsets, rows):
        metadata = super().metadata(trial, offsets, rows)
        return dict(metadata, native_sensors=self.specification["native_sensors"])

    def start(self, env):
        story.archive.storage_remaining(self.out, self.args)
        return SensorStoryTransitions(env, self.settings, self.args.limit)

    def publish(self, trial, result, trace):
        result["native_sensors"] = trace.sensors.summary()
        return super().publish(trial, result, trace)


def configure(args, specification, settings=SensorSettings()):
    """Opt into sensors on a new run instead of calling native_story_transitions.configure."""
    settings.validate()
    story.configure(args, specification)
    specification["native_sensors"] = dict(
        schema=SCHEMA, settings=asdict(settings), statuses=STATUSES, status_columns=list(SUMMARY_FIELDS),
        entity_fields=dict(projectiles=list(PROJECTILE_FIELDS), bodies=list(BODY_FIELDS),
                           straight_lasers=list(LASER_FIELDS)),
        stock_fields=list(STOCK_FIELDS), decision_index="Zero-based into the parent credit's native_before array.",
        entity_layout="Indptr offsets into ragged float32 arrays. Source count precedes filtering; near count precedes cap.",
        sample_timing="Frozen before-policy snapshot. Extra body and beam reads on the configured stride or near-hazard/recovery bursts.",
        geometry="Proximity ranks retained observations for storage only. The additional sensors never enter the actor.",
        coverage="Active raw projectiles, nearby enemy bodies and straight laser segments. Curved lasers and item entities are absent.",
    )
    sources = specification["runtime"]["source_sha256"]
    sources[__name__] = digest(Path(__file__))
    sources[structured.__name__] = digest(Path(structured.__file__))
    return SensorStoryArchive(args, specification, settings)
