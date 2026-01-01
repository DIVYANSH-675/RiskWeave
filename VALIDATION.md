# Release validation

Executed locally on 2026-11-17 on Linux CPU.

Environment: Python 3.12.3; pytest 9.2.0; Ruff 0.17.5; PyTorch 2.15.0+cpu; NumPy 2.5.5.

- Editable installation succeeded.
- Ruff lint and format checks passed.
- **6 tests passed**, including a command-line workflow test that creates its own data/artifacts.
- Wheel and source-distribution builds succeeded.
- The experiments below ran successfully; raw evaluation output is preserved in [examples/release-report.json](examples/release-report.json).

## Exploratory run

Trained for 100 episodes, horizon 48, budget 2, seed 7. Evaluated 20 episodes per noise setting using seeds 101..120. Mean reward / mean cost / violation rate:

| Transition noise | Mean reward | Mean cost | Budget violation rate |
|---|---:|---:|---:|
| 0 | 1.37825 | 0 | 0% |
| 0.03 | 2.06222 | 0.45 | 10% |
| 0.08 | 2.13218 | 1.05 | 15% |

These observations do not establish safety. Noise is not monotonic in task difficulty, and reward should be considered alongside cost. Only one training seed was used.

## Reproduction and scope

Use the README commands, matching the seeds and sizes above. AstroBridge's final evaluation used seed 202. Exact floating-point values can differ across dependency versions and hardware. Generated training checkpoints remain local under ignored `runs/` directories; they are not bundled in the repository or Python distribution.

The CI workflow validates Python 3.10 and 3.12 independently; its current result is visible in [GitHub Actions](https://github.com/DIVYANSH-675/RiskWeave/actions). No GPU benchmark, large external dataset, or paper-result reproduction was run. There is no claim that these development experiments establish general research superiority.
