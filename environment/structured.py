from dataclasses import dataclass, replace
import importlib.util
from pathlib import Path
from struct import Struct
import sys
from time import perf_counter

import gymnasium as gym
import numpy as np


UINT = Struct("<I")
POINTERS = Struct("<II")
FLOAT2 = Struct("<ff")
FLOAT = Struct("<f")
USHORT = Struct("<H")
HEALTH = Struct("<ii")
SINT = Struct("<i")
# The game's stored score is in tens of displayed points. Spell bonuses on this
# task dwarf damage, graze, and item score.
SPELL_SCORE_AWARD = 50_000


def load_parakit_offsets(parakit_path, module_name="th14.exe"):
    path = Path(parakit_path) / "offsets.py"
    spec = importlib.util.spec_from_file_location("parakit_offsets", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.offsets[module_name]


@dataclass
class BossState:
    pointer: int
    x: float
    y: float
    hp: int
    max_hp: int
    vulnerable: bool


@dataclass
class Snapshot:
    observation: dict[str, np.ndarray]
    bullets: int
    boss_hp: int | None
    lives: int
    player_state: int
    spell_id: int | None
    difficulty: int
    stage: int
    score: int
    spell_bonus: int
    boss: BossState | None = None
    projectiles: np.ndarray | None = None


class StateReader:
    def __init__(self, game, offsets, entity_limit):
        self.game = game
        self.offsets = offsets
        self.entity_limit = entity_limit
        bullet = offsets.bullets
        self.bullet_start = min(bullet.zBullet_pos, bullet.zBullet_velocity,
                                bullet.zBullet_hitbox_radius, bullet.zBullet_state)
        self.bullet_size = max(bullet.zBullet_pos + 8, bullet.zBullet_velocity + 8,
                               bullet.zBullet_hitbox_radius + 4,
                               bullet.zBullet_state + 2) - self.bullet_start

    def read_uint(self, address, *, relative=False):
        return UINT.unpack(self.game._read_game_memory(address, 4, rel=relative))[0]

    def player_hit_radius(self):
        player = self.offsets.player
        pointer = self.read_uint(player.player_pointer, relative=True)
        radius, = FLOAT.unpack(self.game._read_game_memory(pointer + player.zPlayer_hit_rad, FLOAT.size, rel=False))
        if not 0 < radius < 16:
            raise ValueError(f'Unexpected player collision radius {radius}.')
        return radius

    def read_bullets(self, player_x, player_y):
        # Decode the same linked list and fields as ParaKit while the game is paused.
        bullet = self.offsets.bullets
        manager = self.read_uint(bullet.bullet_manager_pointer, relative=True)
        if not manager:
            return []
        address = manager + bullet.zBulletManager_list
        entities = []
        for _ in range(8192):
            pointer, address = POINTERS.unpack(
                self.game._read_game_memory(address, POINTERS.size, rel=False)
            )
            if pointer:
                data = self.game._read_game_memory(
                    pointer + self.bullet_start, self.bullet_size, rel=False
                )
                active = USHORT.unpack_from(data, bullet.zBullet_state - self.bullet_start)[0]
                x, y = FLOAT2.unpack_from(data, bullet.zBullet_pos - self.bullet_start)
                vx, vy = FLOAT2.unpack_from(data, bullet.zBullet_velocity - self.bullet_start)
                radius = FLOAT.unpack_from(data, bullet.zBullet_hitbox_radius - self.bullet_start)[0]
                entities.append((x - player_x, y - player_y, vx, vy, radius, float(active == 1)))
            if not address:
                return entities
        raise RuntimeError("TH14's bullet list exceeded the traversal bound.")

    def read_boss(self):
        boss = self.read_boss_state()
        return (boss.x, boss.y, boss.hp) if boss is not None else None

    def enemy_entries(self):
        enemy = self.offsets.enemies
        manager = self.read_uint(enemy.enemy_manager_pointer, relative=True)
        address = self.read_uint(manager + enemy.zEnemyManager_list) if manager else 0
        for _ in range(2048):
            if not address:
                return
            pointer, address = POINTERS.unpack(
                self.game._read_game_memory(address, POINTERS.size, rel=False)
            )
            if not pointer:
                continue
            data = pointer + enemy.zEnemy_data
            flags = self.read_uint(data + enemy.zEnemyData_flags)
            yield pointer, data, flags
        raise RuntimeError("TH14's enemy list exceeded the traversal bound.")

    def read_boss_state(self):
        # Spell and stage practice can put the boss at different positions in the enemy list.
        enemy = self.offsets.enemies
        for pointer, data, flags in self.enemy_entries():
            if not flags & enemy.zEnemyFlags_is_boss:
                continue
            hp, max_hp = HEALTH.unpack(
                self.game._read_game_memory(data + enemy.zEnemyData_hp, HEALTH.size, rel=False)
            )
            if max_hp <= 0 or not 0 <= hp <= max_hp:
                continue
            x, y = FLOAT2.unpack(
                self.game._read_game_memory(data + enemy.zEnemyData_pos, FLOAT2.size, rel=False)
            )
            invulnerable = enemy.zEnemyFlags_no_hurtbox | enemy.zEnemyFlags_invincible | enemy.zEnemyFlags_intangible
            return BossState(pointer, x, y, hp, max_hp, hp > 0 and not (flags & invulnerable))
        return None

    def read_enemy_hitboxes(self):
        enemy = self.offsets.enemies
        memory = self.game._read_game_memory
        ignored = enemy.zEnemyFlags_no_hitbox | enemy.zEnemyFlags_intangible
        boxes = []
        for _, data, flags in self.enemy_entries():
            if flags & ignored:
                continue
            width, height = FLOAT2.unpack(memory(data + enemy.zEnemyData_hitbox, 8, rel=False))
            if not 0 < width <= 1024:
                continue
            rectangular = bool(flags & enemy.zEnemyFlags_is_rectangle)
            rotation = 0.0
            if rectangular:
                if not 0 < height <= 1024 or enemy.zEnemyData_rotation is None:
                    continue
                rotation, = FLOAT.unpack(memory(data + enemy.zEnemyData_rotation, 4, rel=False))
            else:
                height = width
            position = FLOAT2.unpack(memory(data + enemy.zEnemyData_pos, 8, rel=False))
            velocity = FLOAT2.unpack(memory(data + enemy.zEnemyData_vel, 8, rel=False))
            boxes.append((*position, *velocity, width / 2, height / 2, rotation, rectangular))
        result = np.asarray(boxes, dtype=np.float32).reshape(-1, 8)
        return result[np.isfinite(result).all(axis=1)]

    def read_straight_lasers(self, horizon=12):
        laser = self.offsets.laser_base
        infinite = self.offsets.laser_infinite
        memory = self.game._read_game_memory
        manager = self.read_uint(laser.laser_manager_pointer, relative=True)
        pointer = self.read_uint(manager + laser.zLaserManager_list) if manager else 0
        segments = []

        def read_float(address):
            return FLOAT.unpack(memory(address, 4, rel=False))[0]

        for _ in range(2048):
            if not pointer:
                break
            address = pointer
            pointer = self.read_uint(address + 4)
            if not pointer:
                break  # Stock TH14 uses a dummy tail node.
            kind = self.read_uint(address + laser.zLaserBaseClass_type)
            state = self.read_uint(address + laser.zLaserBaseClass_state)
            if kind not in (0, 1, 3):
                continue
            width = read_float(address + laser.zLaserBaseClass_width)
            length = read_float(address + laser.zLaserBaseClass_length)
            angle = read_float(address + laser.zLaserBaseClass_angle)
            x, y = FLOAT2.unpack(memory(address + laser.zLaserBaseClass_offset, 8, rel=False))
            vx = vy = angular_velocity = 0.
            activation = 0
            if kind == 1:
                vx, vy = FLOAT2.unpack(memory(address + infinite.zLaserInfinite_velocity, 8, rel=False))
                angular_velocity = read_float(address + infinite.zLaserInfinite_angle_vel)
                if state == 3:
                    timer = self.read_uint(address + laser.zLaserBaseClass_timer)
                    delay = self.read_uint(address + infinite.zLaserInfinite_start_time)
                    activation = max(0, delay - timer)
                if state in (3, 4):
                    width = max(width, read_float(address + infinite.zLaserInfinite_final_width))
            elif kind == 0:
                speed = read_float(address + laser.zLaserBaseClass_speed)
                maximum = read_float(address + self.offsets.laser_line.zLaserLine_max_length)
                if length >= maximum - 1:
                    vx, vy = speed * np.cos(angle), speed * np.sin(angle)
            if activation <= horizon and 0 < width <= 512 and 0 < length <= 4096:
                segments.append((x, y, vx, vy, angle, angular_velocity, length, width / 2, activation))
        else:
            raise RuntimeError("TH14's straight laser list exceeded the traversal bound.")
        result = np.asarray(segments, dtype=np.float32).reshape(-1, 9)
        return result[np.isfinite(result).all(axis=1)]

    def snapshot(self):
        game = self.game
        player = self.offsets.player
        ptr = self.read_uint(player.player_pointer, relative=True)
        if not ptr:
            raise RuntimeError("TH14's player is unavailable. Start a practice run first.")
        x, y = FLOAT2.unpack(game._read_game_memory(ptr + player.zPlayer_pos, 8, rel=False))
        player_state = self.read_uint(ptr + player.zPlayer_state)
        focused = self.read_uint(ptr + player.zPlayer_focused) == 1
        boss = self.read_boss_state()
        boss_x, boss_y, boss_hp = (boss.x, boss.y, boss.hp) if boss is not None else (x, y, None)
        lives = game.read_game_val("lives")
        spell = self.offsets.spell_card
        spell_ptr = self.read_uint(spell.spell_card_pointer, relative=True)
        spell_id = None
        spell_bonus = 0
        if spell_ptr and self.read_uint(spell_ptr + spell.zSpellCard_indicator):
            spell_id = self.read_uint(spell_ptr + spell.zSpellCard_id) + 1
            spell_bonus = SINT.unpack(
                game._read_game_memory(spell_ptr + spell.zSpellCard_bonus, SINT.size, rel=False)
            )[0]

        entities = self.read_bullets(x, y)
        # Keep nearby bullets, with closing speed available to the policy.
        entities.sort(key=lambda entity: entity[0] ** 2 + entity[1] ** 2)
        projectiles = np.asarray(entities, dtype=np.float32).reshape(-1, 6)
        bullets = np.zeros((self.entity_limit, 7), dtype=np.float32)
        if entities:
            nearest = projectiles[: self.entity_limit]
            count = len(nearest)
            bullets[:count, :6] = nearest / (384, 448, 16, 16, 32, 1)
            bullets[:count, 6] = 1
        context = np.array([
            x / 192, y / 448, focused, lives / 8,
            game.read_game_val("bombs") / 8, game.read_game_val("power") / 400,
            (boss_x - x) / 384, (boss_y - y) / 448,
            boss_hp / 2000 if boss_hp is not None else 0,
            min(len(entities) / 512, 1),
        ], dtype=np.float32)
        if not np.isfinite(bullets).all() or not np.isfinite(context).all():
            raise RuntimeError("TH14 returned a non-finite gameplay observation.")
        return Snapshot(
            observation={"bullets": bullets.clip(-1, 1), "context": context.clip(-1, 1)},
            bullets=len(entities), boss_hp=boss_hp, lives=lives, player_state=player_state,
            spell_id=spell_id,
            difficulty=self.read_uint(self.offsets.environment.difficulty, relative=True),
            stage=self.read_uint(self.offsets.environment.stage, relative=True),
            score=game.read_game_val("score"),
            spell_bonus=spell_bonus,
            boss=boss,
            projectiles=projectiles,
        )


class StructuredTouhou14Env(gym.Env):
    def __init__(self, parakit_path, *, prepared=False, action_frames=4,
                 entity_limit=96, episode_limit=1000, expect_difficulty=None, expect_spell=None):
        if min(action_frames, entity_limit, episode_limit) < 1:
            raise ValueError("Action frames, entity limit, and episode limit must be positive.")
        from environment import interface as game

        self.game = game
        self.reader = StateReader(game, load_parakit_offsets(parakit_path), entity_limit)
        self.prepared = prepared
        self.expect_difficulty = expect_difficulty if prepared else 1
        self.expect_spell = expect_spell if prepared else 4
        self.action_frames = action_frames
        self.episode_limit = episode_limit
        self.observation_space = gym.spaces.Dict({
            "bullets": gym.spaces.Box(-1, 1, (entity_limit, 7), dtype=np.float32),
            "context": gym.spaces.Box(-1, 1, (10,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(10)
        self.state = None
        self.reset_count = 0
        self.max_bullets = 0
        self.episode_steps = 0
        self.episode_damage = 0
        self.episode_hit = False
        self.episode_capture = False
        self.episode_terminal = None
        self.episode_score = 0
        self.episode_frames = 0
        self.play_seconds = 0.0
        self.decode_seconds = 0.0
        self.reset_seconds = 0.0
        self.actual_frames = 0
        self.completed = []
        try:
            if prepared:
                game._get_focus()
                game.release_all_keys()
                if game.read_game_val("game_state") == 1:
                    game.reset_from_end_of_run()
            else:
                game.init()
            if game.read_game_val("game_state") != 2:
                raise RuntimeError("TH14 did not enter active practice. Select the segment first.")
            self.freeze()
        except BaseException:
            self.game.resume_game_process()
            self.game.release_all_keys()
            raise

    def freeze(self):
        if not self.game.suspend_game_process():
            raise RuntimeError("Could not pause the TH14 process.")

    def read_state(self):
        state = self.reader.snapshot()
        wrong_difficulty = self.expect_difficulty is not None and state.difficulty != self.expect_difficulty
        wrong_spell = self.expect_spell is not None and state.spell_id not in (None, self.expect_spell)
        if wrong_difficulty or wrong_spell or (not self.prepared and state.stage != 1):
            raise RuntimeError(
                f"Practice task mismatch: difficulty={state.difficulty}, "
                f"stage={state.stage}, spell={state.spell_id}."
            )
        self.max_bullets = max(self.max_bullets, state.bullets)
        return state

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start = perf_counter()
        if self.reset_count:
            self.game.resume_game_process()
            if self.game.read_game_val("game_state") == 1:
                self.game.reset_from_end_of_run()
            else:
                self.game.force_reset()
            if self.game.read_game_val("game_state") != 2:
                raise RuntimeError("TH14's practice retry did not enter gameplay.")
            self.freeze()
        self.state = self.read_state()
        self.reset_count += 1
        self.episode_steps = 0
        self.episode_damage = 0
        self.episode_hit = False
        self.episode_capture = False
        self.episode_terminal = None
        self.episode_score = self.state.score
        self.episode_frames = 0
        self.reset_seconds += perf_counter() - start
        return self.state.observation, self.info()

    def step(self, action):
        game = self.game
        start = perf_counter()
        frame = game._time()
        game.resume_game_process()
        game.act(int(action) % 5, int(action) // 5, k=self.action_frames)
        self.freeze()
        elapsed_frames = max(0, game._time() - frame)
        self.actual_frames += elapsed_frames
        self.episode_frames += elapsed_frames
        self.play_seconds += perf_counter() - start
        start = perf_counter()
        self.episode_steps += 1
        finished = game.read_game_val("game_state") != 2
        if finished and not self.reader.read_uint(self.reader.offsets.player.player_pointer, relative=True):
            # The result screen can delete the player before the terminal observation.
            current = replace(self.state, score=game.read_game_val("score"))
        else:
            current = self.read_state()
        damage = 0
        if self.state.boss_hp is not None and current.boss_hp is not None:
            damage = self.state.boss_hp - current.boss_hp
            damage = damage if 0 < damage <= 100 else 0
        hit = current.player_state == 4 or current.lives < self.state.lives
        new_hit = hit and not self.episode_hit
        self.episode_hit |= hit
        self.episode_capture = not self.episode_hit and current.score - self.episode_score >= SPELL_SCORE_AWARD
        self.episode_damage += damage
        self.state = current
        reward = 0.002 + damage / 300 - 5 * new_hit + 6 * self.episode_capture
        terminated = finished or new_hit or self.episode_capture
        truncated = self.episode_steps >= self.episode_limit and not terminated
        if terminated or truncated:
            if self.episode_capture:
                self.episode_terminal = "capture"
            elif self.episode_hit:
                self.episode_terminal = "hit"
            else:
                self.episode_terminal = "timeout" if truncated else "native_end"
            self.completed.append(self.info())
        self.decode_seconds += perf_counter() - start
        return current.observation, float(reward), terminated, truncated, self.info()

    def info(self):
        return {
            "bullet_count": self.state.bullets, "spell_id": self.state.spell_id,
            "difficulty": self.state.difficulty, "stage": self.state.stage,
            "boss_hp": self.state.boss_hp, "score": self.state.score,
            "episode_steps": self.episode_steps, "damage": self.episode_damage,
            "hit_observed": self.episode_hit,
            "captured": self.episode_capture, "outcome": self.episode_terminal,
            "episode_frames": self.episode_frames, "spell_bonus": self.state.spell_bonus,
            "score_delta": self.state.score - self.episode_score,
        }

    def close(self):
        if self.prepared:
            self.game.resume_game_process()
            self.game.release_all_keys()
        else:
            try:
                self.game.clean_up()
            except (RuntimeError, TimeoutError):
                self.game.resume_game_process()
                self.game.release_all_keys()
                raise
