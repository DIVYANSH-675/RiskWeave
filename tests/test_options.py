import sys

import pytest

from riskweave import cli


def parse_training(monkeypatch, **options):
    captured = {}
    arguments = [
        item
        for key, value in options.items()
        for item in ("--" + key.replace("_", "-"), str(value))
    ]
    monkeypatch.setattr(sys, "argv", ["riskweave", "train", *arguments])
    monkeypatch.setattr(cli, "train", lambda args: captured.update(vars(args)))
    cli.main()
    return captured


def test_learning_rate_validation(monkeypatch):
    assert parse_training(monkeypatch, learning_rate=0.002)["learning_rate"] == 0.002
    with pytest.raises(SystemExit):
        parse_training(monkeypatch, learning_rate="nan")
