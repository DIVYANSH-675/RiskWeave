import numpy as np
import torch

from riskweave.core import (
    Agent,
    Navigation,
    collect,
    discounted,
    optimize,
    update_penalty,
    upper_choquet,
)


def test_capacity_limits_translation_and_gradient():
    values = torch.tensor([[-3.0, 0.0, 6.0]])
    power = torch.tensor([0.5], requires_grad=True)
    risk = upper_choquet(values, power)
    assert -3 <= risk.item() <= 6
    torch.testing.assert_close(upper_choquet(values, torch.ones(1)), values.mean(1))
    torch.testing.assert_close(upper_choquet(values + 4, power), risk + 4)
    risk.sum().backward()
    assert torch.isfinite(power.grad).all()


def test_discount_and_penalty_direction():
    torch.testing.assert_close(discounted([1, 2, 3], 0.5), torch.tensor([2.75, 3.5, 3.0]))
    assert update_penalty(1, 5, 2, 0.1) > 1
    assert update_penalty(0, 0, 2, 0.1) == 0


def test_seeded_simulation_and_cost():
    first, second = Navigation(seed=3, noise=0.1), Navigation(seed=3, noise=0.1)
    np.testing.assert_array_equal(first.step([0, 0])[0], second.step([0, 0])[0])
    env = Navigation()
    env.position[:] = 0
    assert env.step([0, 0])[2] == 1


def test_actual_policy_update_is_finite():
    torch.manual_seed(4)
    torch.set_num_threads(1)
    agent = Agent()
    before = [p.detach().clone() for p in agent.parameters()]
    data = collect(agent, Navigation(horizon=8))
    metrics = optimize(agent, torch.optim.Adam(agent.parameters(), lr=1e-3), data, 1, updates=2)
    assert all(np.isfinite(x) for x in metrics.values())
    assert any(not torch.equal(old, new) for old, new in zip(before, agent.parameters()))


def test_transition_perturbations_preserve_environment_geometry():
    from riskweave.core import transition_neighbors

    observation = torch.tensor(Navigation(horizon=8).observation())[None]
    neighbors = transition_neighbors(observation, 16, 0.2)
    torch.testing.assert_close(
        neighbors[..., :2] + neighbors[..., 2:4], torch.full((1, 16, 2), 0.8)
    )
    torch.testing.assert_close(neighbors[..., 4], torch.ones(1, 16))


def test_capacity_rejects_accidental_batch_broadcasting():
    import pytest

    with pytest.raises(ValueError, match="one value per batch row"):
        upper_choquet(torch.ones(1, 3), torch.ones(2))
    with pytest.raises(ValueError, match="nonempty"):
        upper_choquet(torch.empty(0, 3), torch.ones(1))
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    torch.testing.assert_close(upper_choquet(values, torch.tensor(1.0)), values.mean(1))
    torch.testing.assert_close(upper_choquet(values, torch.ones(2)), values.mean(1))


def test_noise_conditions_are_summarized_separately():
    from riskweave.core import summarize_episodes

    report = summarize_episodes(
        [
            dict(noise=0.0, reward=1.0, cost=0.0, violation=False),
            dict(noise=0.1, reward=3.0, cost=2.0, violation=True),
            dict(noise=0.1, reward=1.0, cost=0.0, violation=False),
        ]
    )
    assert report[0]["episodes"] == 1
    assert report[1]["mean_reward"] == 2 and report[1]["violation_rate"] == 0.5
