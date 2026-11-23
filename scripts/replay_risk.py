"""Four-tick collision supervision for paired replay states."""

import torch

from scripts.stage_risk import _DIRECTIONS, _LOWER_BOUND, _PROJECTILE_POSITION_SCALE, _UPPER_BOUND

MARISA_FLAG = 12  # Ten raw fields, difficulty, Reimu, then Marisa.


def collision_mask(bullets, context):
    position = context[:, :2] * context.new_tensor((192, 448))
    base_speeds = torch.where(context[:, MARISA_FLAG] > 0.5, 5., 4.5)
    speeds = torch.stack((base_speeds, torch.full_like(base_speeds, 2.)), dim=1)
    directions = context.new_tensor(_DIRECTIONS)
    times = torch.arange(1, 5, dtype=context.dtype, device=context.device)
    velocities = speeds[:, :, None, None] * directions[None, None]
    player = position[:, None, None, None] + velocities[..., None, :] * times[None, None, None, :, None]
    player = torch.clamp(player, context.new_tensor(_LOWER_BOUND), context.new_tensor(_UPPER_BOUND))
    bullet_position = bullets[..., :2] * bullets.new_tensor(_PROJECTILE_POSITION_SCALE)
    projected = bullet_position[:, :, None] + bullets[..., 2:4][:, :, None] * 16 * times[None, None, :, None]
    relative = projected[:, None, None] - (player - position[:, None, None, None])[:, :, :, None]
    radius = bullets[..., 4] * 32 + 1.
    hazard = (bullets[..., 6] > 0) & (bullets[..., 5] > 0) & (bullets[..., 4] > 0)
    overlaps = relative.square().sum(dim=-1) < radius[:, None, None, :, None].square()
    return (overlaps & hazard[:, None, None, :, None]).any(dim=(3, 4))


def replay_risk_terms(bullets, context, labels, movement_logits, focus_logits):
    collisions = collision_mask(bullets, context)
    items = torch.arange(len(bullets), device=bullets.device)
    demo_collision = collisions[items, labels[:, 1], labels[:, 0]]
    eligible = collisions.any(dim=(1, 2)) & ~demo_collision
    probabilities = focus_logits.softmax(1)[:, :, None] * movement_logits.softmax(1)[:, None, :]
    mass = (probabilities * collisions).sum(dim=(1, 2))
    return (mass * eligible).sum(), eligible.sum(), (collisions, eligible)
