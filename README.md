# RiskWeave

Constrained policy experiments with learned fuzzy-capacity risk aggregation.

RiskWeave studies how a state-conditioned Choquet aggregator can shape reward and cost value estimates under perturbed transitions. It includes a small navigation environment, a clipped policy-gradient update, two value heads, and a projected constraint multiplier.

## Status and provenance

Version 0.1.0 is an original, AI-assisted implementation developed for **Divyansh Gupta**. This repository was written independently: no upstream source files or Git history were imported. Its scientific inspiration is [Fuz-RL](https://arxiv.org/abs/2602.20729); the research ideas remain credited to that paper's authors. The implementation choices and limitations below describe this repository's scope. It is an experimental prototype, not a claimed reproduction of published results.

## Setup

Python 3.10 or newer:

```bash
python -m venv .venv
# Linux/macOS; on Windows use .venv\Scripts\activate
source .venv/bin/activate
# Install CPU-only PyTorch first (omit this line to choose another build).
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

## Run

```bash
riskweave train --episodes 100 --horizon 48 --budget 2 --seed 7 --output runs/train
riskweave evaluate --checkpoint runs/train/policy.pt --episodes 20 --horizon 48 --budget 2 --seed 101 --noise-grid 0 0.03 0.08 --output runs/evaluation.json
```

Use fresh output paths for new experiments. Training artifacts and generated data are ignored by Git. For independent evaluation, use a different generation seed or a separate dataset; evaluating on the training data is not an unbiased test.

## Implementation

The observation includes position, offset to the goal, and remaining episode time. Transition samples perturb position while preserving the goal relationship and time. For sorted values `v[1] >= ... >= v[K]`, the aggregator is

```text
mu(k) = (k / K) ** p(state),     0.2 < p(state) < 1
C(v) = sum_k (mu(k) - mu(k-1)) * v[k]
reward continuation = -C(-reward_values)
cost continuation = C(cost_values)
```

The capacity network fits the resulting Bellman targets toward sampled discounted returns. The critic fits detached targets; the actor uses normalized reward-minus-penalized-cost advantages. The nonnegative penalty increases when episode cost exceeds the configured budget. The environment's finite horizon is a true episode end, and its remaining time is part of the state.

## Data and artifacts

Training writes `config.json`, per-episode `metrics.jsonl`, and `policy.pt`. Evaluation records reward, cost, budget violations, noise, and seed for every episode. Keep horizon and budget consistent when comparing policies. All commands run on CPU.

## Validation

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

The test suite covers: Capacity normalization/translation behavior and gradients; discount and multiplier direction; seeded environment dynamics; geometry-preserving uncertainty samples; finite policy updates; command-line checkpoint/evaluation round trip.

CI runs the suite on Python 3.10 and 3.12. Small synthetic runs validate software behavior; they do not establish a research result. See [VALIDATION.md](VALIDATION.md) for the release checks actually performed.

## Limitations

This is a small constrained-learning experiment, not a reproduction of Fuz-RL, a certified safe controller, or a result on Safety-Gymnasium. It uses a cardinality power capacity and a learned Bellman regularizer. No theoretical robustness guarantee is established. Short runs are workflow checks; compare multiple seeds and budgets before interpreting outcomes. The toy navigation task can avoid costs simply by failing to move, so reward and violations must be assessed together.

## License

The original code in this repository is MIT licensed, copyright 2026 Divyansh Gupta. See [LICENSE](LICENSE). The cited paper is a separate work; this code's license does not relicense the paper or external datasets.
