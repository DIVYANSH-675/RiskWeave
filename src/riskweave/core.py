"""Constrained policy learning with state-conditioned cardinality capacities."""

import numpy as np
import torch
from torch import nn


def upper_choquet(values, power):
    """Choquet integral for the normalized capacity mu(A)=(|A|/K)**power."""
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("values must be a nonempty BxK tensor")
    if not torch.isfinite(values).all() or not torch.isfinite(power).all() or (power <= 0).any():
        raise ValueError("Finite values and positive powers are required")
    k = values.shape[1]
    capacity = (torch.arange(1, k + 1, device=values.device, dtype=values.dtype) / k)[
        None, :
    ] ** power.reshape(-1, 1)
    weights = capacity - torch.cat([torch.zeros_like(capacity[:, :1]), capacity[:, :-1]], dim=1)
    return (values.sort(dim=1, descending=True).values * weights).sum(1)


def discounted(values, gamma):
    result, tail = [], 0.0
    for value in reversed(values):
        tail = float(value) + gamma * tail
        result.append(tail)
    return torch.tensor(result[::-1], dtype=torch.float32)


def update_penalty(value, observed_cost, budget, rate):
    return max(0.0, value + rate * (observed_cost - budget))


class Navigation:
    """Finite-horizon point navigation around a circular hazard; not a robot simulator."""

    def __init__(self, seed=0, noise=0.0, horizon=48):
        if noise < 0 or not np.isfinite(noise) or horizon < 1:
            raise ValueError("Invalid noise or horizon")
        self.rng = np.random.default_rng(seed)
        self.noise, self.horizon = noise, horizon
        self.goal = np.array([0.8, 0.8], dtype=np.float32)
        self.reset()

    def observation(self):
        return np.concatenate(
            [self.position, self.goal - self.position, [1 - self.steps / self.horizon]]
        ).astype(np.float32)

    def reset(self):
        self.position = np.array([-0.8, -0.8], dtype=np.float32) + self.rng.normal(0, 0.03, 2)
        self.steps = 0
        return self.observation()

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite two-dimensional action")
        previous = np.linalg.norm(self.goal - self.position)
        self.position = np.clip(
            self.position + 0.12 * np.clip(action, -1, 1) + self.rng.normal(0, self.noise, 2),
            -1.2,
            1.2,
        )
        distance = np.linalg.norm(self.goal - self.position)
        cost = float(np.linalg.norm(self.position) < 0.32)
        success = distance < 0.12
        self.steps += 1
        return (
            self.observation(),
            float(previous - distance - 0.01 + success),
            cost,
            bool(success or self.steps >= self.horizon),
        )


class Agent(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, 2))
        self.value = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, 2))
        self.capacity = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, 1))
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def distribution(self, observations):
        return torch.distributions.Normal(
            self.policy(observations), self.log_std.clamp(-4, 1).exp()
        )

    def power(self, observations):
        return 0.2 + 0.8 * torch.sigmoid(self.capacity(observations)).squeeze(-1)


def collect(agent, env):
    observations, actions, rewards, costs, successors, endings, logprobs = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    observation = env.reset()
    for _ in range(env.horizon):
        state = torch.as_tensor(observation)
        with torch.no_grad():
            distribution = agent.distribution(state)
            action = distribution.sample()
            logprob = distribution.log_prob(action).sum()
        following, reward, cost, done = env.step(action.numpy())
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        costs.append(cost)
        successors.append(following)
        endings.append(done)
        logprobs.append(logprob)
        observation = following
        if done:
            break
    return dict(
        obs=torch.tensor(np.asarray(observations)),
        actions=torch.stack(actions),
        next=torch.tensor(np.asarray(successors)),
        done=torch.tensor(endings),
        rewards=torch.tensor(rewards),
        costs=torch.tensor(costs),
        logp=torch.stack(logprobs),
    )


def transition_neighbors(observations, samples, radius):
    """Perturb positions while preserving goal offsets and remaining episode time."""
    if samples < 1 or radius < 0 or not np.isfinite(radius):
        raise ValueError("Positive sample count and finite nonnegative radius required")
    neighbors = observations[:, None, :].expand(-1, samples, -1).clone()
    delta = radius * torch.randn(len(observations), samples, 2, device=observations.device)
    position = (neighbors[..., :2] + delta).clamp(-1.2, 1.2)
    displacement = position - neighbors[..., :2]
    neighbors[..., :2] = position
    neighbors[..., 2:4] -= displacement
    return neighbors


def optimize(agent, optimizer, data, penalty, *, gamma=0.97, radius=0.08, samples=8, updates=4):
    if updates < 1 or not 0 <= gamma <= 1 or penalty < 0 or not np.isfinite(penalty):
        raise ValueError("Invalid update count, discount, or penalty")
    returns = torch.stack(
        [discounted(data["rewards"], gamma), discounted(data["costs"], gamma)], dim=1
    )
    with torch.no_grad():
        advantages = returns - agent.value(data["obs"])
        advantages = advantages[:, 0] - penalty * advantages[:, 1]
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(
            1e-6
        )
        neighbors = transition_neighbors(data["next"], samples, radius)
        neighbor_values = agent.value(neighbors).detach()
    final = None
    for _ in range(updates):
        power = agent.power(data["next"])
        lower_reward = -upper_choquet(-neighbor_values[:, :, 0], power)
        upper_cost = upper_choquet(neighbor_values[:, :, 1], power)
        continuation = (~data["done"]).float()[:, None]
        targets = torch.stack(
            [data["rewards"], data["costs"]], dim=1
        ) + gamma * continuation * torch.stack([lower_reward, upper_cost], dim=1)
        value_loss = (agent.value(data["obs"]) - targets.detach()).square().mean()
        capacity_loss = (targets - returns).square().mean()
        distribution = agent.distribution(data["obs"])
        ratio = (distribution.log_prob(data["actions"]).sum(-1) - data["logp"]).exp()
        objective = torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages).mean()
        loss = (
            -objective
            + 0.5 * value_loss
            + 0.1 * capacity_loss
            - 0.005 * distribution.entropy().mean()
        )
        if not torch.isfinite(loss):
            raise ValueError("Training loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        final = dict(
            loss=float(loss.detach()),
            value_loss=float(value_loss.detach()),
            capacity_loss=float(capacity_loss.detach()),
        )
    return final
