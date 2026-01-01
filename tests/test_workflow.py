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
