import argparse
import json

import numpy as np
import torch

from . import __version__
from .core import Agent, Navigation, collect, optimize, update_penalty
from .io import seed_all, start_run, write_json


def train(a):
    seed_all(a.seed)
    config = vars(a).copy()
    output = start_run(a.output, config)
    agent = Agent()
    optimizer = torch.optim.Adam(agent.parameters(), lr=a.learning_rate)
    env = Navigation(a.seed, a.noise, a.horizon)
    optimizer_options = dict(radius=a.radius, samples=a.samples, gamma=a.gamma)
    penalty = a.initial_penalty
    with (output / "metrics.jsonl").open("w") as stream:
        for episode in range(a.episodes):
            data = collect(agent, env)
            metrics = optimize(agent, optimizer, data, penalty, **optimizer_options)
            cost = float(data["costs"].sum())
            penalty = update_penalty(penalty, cost, a.budget, a.penalty_rate)
            row = dict(
                episode=episode,
                reward=float(data["rewards"].sum()),
                cost=cost,
                penalty=penalty,
                **metrics,
            )
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    torch.save(
        {"agent": agent.state_dict(), "config": config, "penalty": penalty}, output / "policy.pt"
    )
    print(json.dumps(row))


def evaluate(a):
    seed_all(a.seed)
    checkpoint = torch.load(a.checkpoint, weights_only=True, map_location="cpu")
    agent = Agent()
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    records = []
    for noise in a.noise_grid:
        for index in range(a.episodes):
            env = Navigation(a.seed + index, noise, a.horizon)
            observation = env.reset()
            reward = cost = 0.0
            for _ in range(a.horizon):
                with torch.no_grad():
                    action = agent.policy(torch.tensor(observation)).numpy()
                observation, r, c, done = env.step(action)
                reward += r
                cost += c
                if done:
                    break
            records.append(
                dict(
                    noise=noise,
                    seed=a.seed + index,
                    reward=reward,
                    cost=cost,
                    violation=cost > a.budget,
                )
            )
    write_json(
        a.output,
        dict(
            episodes=records,
            budget=a.budget,
            mean_cost=float(np.mean([r["cost"] for r in records])),
        ),
    )
    print(f"Saved {len(records)} evaluation episodes to {a.output}")


def main():
    p = argparse.ArgumentParser(
        description="RiskWeave: fuzzy-capacity constrained policy experiments"
    )
    p.add_argument("--version", action="version", version="riskweave " + __version__)
    p.add_argument("command", choices=["train", "evaluate"])
    p.add_argument("--output", default=None)
    p.add_argument("--checkpoint")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--penalty-rate", type=float, default=0.05)
    p.add_argument("--initial-penalty", type=float, default=1.0)
    p.add_argument("--budget", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--radius", type=float, default=0.08)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--noise-grid", type=float, nargs="+", default=[0.0, 0.03, 0.08])
    a = p.parse_args()
    if any(not np.isfinite(v) or v < 0 for v in (a.penalty_rate, a.initial_penalty)):
        p.error("penalty-rate and initial-penalty must be finite and nonnegative")
    if not 0 <= a.gamma <= 1:
        p.error("gamma must be in [0,1]")
    if not np.isfinite(a.radius) or a.radius < 0 or a.samples < 1:
        p.error("radius must be finite and nonnegative; samples must be positive")
    if not np.isfinite(a.learning_rate) or a.learning_rate <= 0:
        p.error("learning-rate must be finite and positive")
    a.output = a.output or (
        "runs/report.json"
        if a.command == "evaluate"
        else "runs/data.npz"
        if a.command in ("generate", "bank")
        else "runs/train"
    )
    if a.episodes < 1 or a.horizon < 1 or a.budget < 0 or not np.isfinite(a.budget):
        p.error("episodes/horizon must be positive and budget finite and nonnegative")
    if a.command == "evaluate" and not a.checkpoint:
        p.error("evaluate requires --checkpoint; --output is a JSON filename")
    (train if a.command == "train" else evaluate)(a)
