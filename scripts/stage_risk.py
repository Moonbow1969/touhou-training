import numpy as np


_DIRECTIONS = np.array([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)], dtype=np.float32)
_DIRECTIONS /= np.linalg.norm(_DIRECTIONS, axis=1, keepdims=True).clip(min=1)
_SPEEDS = np.array([4.5, 2.0], dtype=np.float32)
_LOWER_BOUND = np.array([-184, 32], dtype=np.float32)
_UPPER_BOUND = np.array([184, 432], dtype=np.float32)
_PROJECTILE_POSITION_SCALE = np.array([384, 448], dtype=np.float32)


def player_paths(observation, horizon):
    position = observation['context'][:2] * np.array([192, 448], dtype=np.float32)
    time = np.arange(1, horizon + 1, dtype=np.float32)
    movement = _SPEEDS[:, None, None] * _DIRECTIONS[None]
    return np.clip(position + movement[:, :, None] * time[None, None, :, None], _LOWER_BOUND, _UPPER_BOUND)


def action_clearances(observation, player_radius, horizon=12):
    """Min predicted projectile surface distance for each focus/direction pair."""
    objects = observation['bullets']
    hazard = objects[(objects[:, 6] > 0) & (objects[:, 5] > 0) & (objects[:, 4] > 0)]
    return _projectile_clearances(observation, player_radius, horizon, hazard[:, :2] * _PROJECTILE_POSITION_SCALE,
                                  hazard[:, 2:4] * 16, hazard[:, 4] * 32)


def raw_projectile_clearances(observation, player_radius, projectiles, horizon=12):
    """Check native positions, velocities and radii without the policy's feature limits."""
    hazard = projectiles[(projectiles[:, 5] > 0) & (projectiles[:, 4] > 0)]
    hazard = hazard[np.isfinite(hazard).all(axis=1)]
    return _projectile_clearances(observation, player_radius, horizon, hazard[:, :2], hazard[:, 2:4], hazard[:, 4])


def _projectile_clearances(observation, player_radius, horizon, centers, velocity, radius):
    if not len(centers):
        return np.full((2, 9), 512, dtype=np.float32)
    position = observation['context'][:2] * np.array([192, 448], dtype=np.float32)
    time = np.arange(1, horizon + 1, dtype=np.float32)
    player = player_paths(observation, horizon)
    bullets = centers[:, None, :] + velocity[:, None, :] * time[None, :, None]
    relative = bullets[None, None] - (player - position)[:, :, None]
    distance = np.linalg.norm(relative, axis=-1)
    clearance = distance - radius[None, None, :, None] - player_radius
    return clearance.min(axis=(2, 3))


def body_clearances(observation, player_radius, hitboxes, horizon=12):
    if not len(hitboxes):
        return np.full((2, 9), 512, dtype=np.float32)
    time = np.arange(1, horizon + 1, dtype=np.float32)
    centers = hitboxes[:, None, :2] + hitboxes[:, None, 2:4] * time[None, :, None]
    relative = player_paths(observation, horizon)[:, :, None] - centers
    cosine, sine = np.cos(hitboxes[:, 6]), np.sin(hitboxes[:, 6])
    rx, ry = relative[..., 0], relative[..., 1]
    local_x = rx * cosine[:, None] + ry * sine[:, None]
    local_y = -rx * sine[:, None] + ry * cosine[:, None]
    edges = np.abs(np.stack((local_x, local_y), axis=-1)) - hitboxes[:, None, 4:6]
    rectangle = np.linalg.norm(np.maximum(edges, 0), axis=-1) + np.minimum(edges.max(-1), 0)
    circle = np.linalg.norm(relative, axis=-1) - hitboxes[:, None, 4]
    clearance = np.where(hitboxes[:, None, 7] > 0, rectangle, circle) - player_radius
    return clearance.min(axis=(2, 3))


def laser_clearances(observation, player_radius, segments, horizon=12):
    if not len(segments):
        return np.full((2, 9), 512, dtype=np.float32)
    time = np.arange(1, horizon + 1, dtype=np.float32)
    tail = segments[:, None, :2] + segments[:, None, 2:4] * time[None, :, None]
    angle = segments[:, None, 4] + segments[:, None, 5] * time
    direction = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    relative = player_paths(observation, horizon)[:, :, None] - tail
    projected = (relative * direction).sum(axis=-1)
    along = np.clip(projected, 0, segments[:, None, 6])
    distance = np.linalg.norm(relative - along[..., None] * direction, axis=-1)
    clearance = distance - segments[:, None, 7] - player_radius
    clearance = np.where(time >= segments[:, None, 8], clearance, 512)
    return clearance.min(axis=(2, 3))


def suggest_dodge(action, clearance, safety_margin=4.0):
    move, focus, _, _ = map(int, action)
    prior_margin = float(clearance[focus, move])
    if prior_margin >= safety_margin:
        return np.asarray(action).copy(), prior_margin, prior_margin
    # Preserve focus when equally safe. Only change the policy if the margin improves.
    priority = clearance.copy()
    priority[focus] += .5
    safer_focus, safer_move = np.unravel_index(priority.argmax(), priority.shape)
    safe_margin = float(clearance[safer_focus, safer_move])
    proposed = np.asarray(action).copy()
    if safe_margin > prior_margin + .75:
        proposed[:2] = (safer_move, safer_focus)
        return proposed, prior_margin, safe_margin
    return proposed, prior_margin, prior_margin


def dodge_regret(action, clearance):
    move, focus = map(int, action[:2])
    risk = np.clip((7 - clearance) / 12, 0, 2)
    return float(risk[focus, move] - risk.min())
