import json
import math
import subprocess
import sys


def test_command_line_round_trip(tmp_path):
    def run(*args):
        result = subprocess.run(
            [sys.executable, "-m", "riskweave", *map(str, args)],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr

    run("train", "--episodes", "3", "--horizon", "8", "--output", tmp_path / "train")
    run(
        "evaluate",
        "--checkpoint",
        tmp_path / "train/policy.pt",
        "--episodes",
        "2",
        "--horizon",
        "8",
        "--seed",
        "101",
        "--output",
        tmp_path / "report.json",
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert len(report["episodes"]) == 6
    assert all(math.isfinite(row["reward"]) for row in report["episodes"])


def test_evaluation_records_task_completion(tmp_path):
    from argparse import Namespace

    import torch

    from riskweave.cli import evaluate
    from riskweave.core import Agent

    checkpoint = tmp_path / "policy.pt"
    torch.save({"agent": Agent().state_dict()}, checkpoint)
    output = tmp_path / "report.json"
    evaluate(
        Namespace(
            seed=7,
            checkpoint=checkpoint,
            episodes=1,
            horizon=2,
            noise_grid=[0.0],
            budget=2,
            output=output,
        )
    )
    row = json.loads(output.read_text())["episodes"][0]
    assert row["steps"] == 2 and isinstance(row["success"], bool)
