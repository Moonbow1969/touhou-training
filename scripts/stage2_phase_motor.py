"""Learned Stage 2 movement option conditioned on the observed boss phase and its age."""
import numpy as np
import torch
from torch import nn

from scripts.relative_phase_student import StockRelativeActor
from scripts.train_demo_route import RouteTracker


class PhaseMotor(nn.Module):
    def __init__(self, phase_count):
        super().__init__()
        self.register_buffer("periods", torch.logspace(1.1, 3.8, 64))
        self.embedding = nn.Embedding(phase_count, 24)
        self.temporal = nn.Sequential(nn.Linear(153, 192), nn.SiLU(), nn.Linear(192, 192), nn.SiLU(),
                                      nn.Linear(192, 4))
        self.motor = RouteTracker()
        self.register_buffer("bounds", torch.zeros(2, 2))
        self.register_buffer("scale", torch.tensor([192., 448.]))

    def goals(self, phase, age):
        t = age.float().clamp(0, 8192)
        angle = t[:, None] * (2 * torch.pi / self.periods)
        x = torch.cat((self.embedding(phase), t[:, None] / 4096, angle.sin(), angle.cos()), 1)
        return self.temporal(x)

    def forward(self, phase, age, position):
        goals = self.goals(phase, age)
        displacement = ((goals[:, :2] * self.scale - position) / 16).clamp(-6, 6)
        room = (torch.cat((position - self.bounds[0], self.bounds[1] - position), 1) / 32).clamp(0, 1)
        motor_input = torch.cat((displacement, room, goals[:, 2:].sigmoid()), 1)
        return self.motor(motor_input)[0]


class PhaseAge:
    def __init__(self):
        self.key = None
        self.frame = None

    def read(self, key, frame):
        if key != self.key or self.frame is None or frame < self.frame:
            self.key, self.frame = key, frame
        return frame - self.frame


class Stage2Student(StockRelativeActor):
    def __init__(self, parent_path, phase_path):
        super().__init__(parent_path)
        p = torch.load(phase_path, map_location="cpu", weights_only=True)
        if p["schema"] != "th14-stage2-phase-motor-v1":
            raise ValueError("Unknown phase student.")
        self.index = {tuple(fields): i for i, fields in enumerate(p["phase_keys"])}
        self.option = PhaseMotor(len(self.index)).eval()
        self.option.load_state_dict(p["state_dict"])
        self.phase_clock = PhaseAge()
        self.chosen = 0
        self.changed = 0

    @torch.inference_mode()
    def predict(self, observation, deterministic=False):
        action, state = super().predict(observation, deterministic)
        env = self.environment
        s = env.snapshot
        b = s.boss
        pair = (int(b.max_hp), int(s.spell_id) if s.spell_id is not None else -1) if b and b.hp > 0 else None
        idx = self.index.get(pair) if s.stage == 2 else None
        age = self.phase_clock.read((env.reset_count, s.stage, idx), env.last_frame)
        if idx is not None:
            position = torch.as_tensor(observation["context"][None, :2]) * self.option.scale
            joint = self.option(torch.tensor([idx]), torch.tensor([age]), position).argmax(1).item()
            self.chosen += 1
            self.changed += joint != int(action[0] + 9 * action[1])
            action[:2] = np.array([joint % 9, joint // 9])
        return action, state
