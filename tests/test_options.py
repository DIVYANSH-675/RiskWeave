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


def test_uncertainty_options(monkeypatch):
    args = parse_training(monkeypatch, radius=0, samples=3)
    assert args["radius"] == 0 and args["samples"] == 3
    with pytest.raises(SystemExit):
        parse_training(monkeypatch, samples=0)
    with pytest.raises(SystemExit):
        parse_training(monkeypatch, radius=-0.1)


def test_discount_option(monkeypatch):
    assert parse_training(monkeypatch, gamma=0)["gamma"] == 0
    with pytest.raises(SystemExit):
        parse_training(monkeypatch, gamma=1.01)


def test_constraint_multiplier_options(monkeypatch):
    args = parse_training(monkeypatch, penalty_rate=0, initial_penalty=2)
    assert args["penalty_rate"] == 0 and args["initial_penalty"] == 2
    with pytest.raises(SystemExit):
        parse_training(monkeypatch, initial_penalty=-1)
