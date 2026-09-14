from __future__ import annotations

import numpy as np
import pytest
import torch

from dummy_mlwp.errors import DeviceError
from dummy_mlwp.model import (
    DummyNet,
    channel_stats,
    feedback_index,
    output_stats,
    predict,
    select_device,
)
from dummy_mlwp.varspec import channel_layout, parse_level_coords, parse_var_specs

CPU = torch.device("cpu")


@pytest.fixture
def fields():
    rng = np.random.default_rng(0)
    surface = 280.0 + 5.0 * rng.standard_normal((4, 1, 8, 10))
    winds = 3.0 * rng.standard_normal((4, 2, 8, 10))
    return np.concatenate([surface, winds], axis=1).astype("float32")


def run(fields, n_steps, in_ch=3, out_ch=2, seed=0, feedback=None):
    net = DummyNet(in_ch, out_ch, hidden_channels=8, n_layers=3, seed=seed)
    mean, std = channel_stats(fields)
    out_mean = np.zeros(out_ch, dtype="float32")
    out_std = np.ones(out_ch, dtype="float32")
    if feedback is None:
        feedback = np.full(in_ch, -1, dtype="int64")
    return predict(fields, net, CPU, n_steps, mean, std, out_mean, out_std, feedback)


def test_minus_one_predicts_one_step_per_input_timestep(fields):
    result = run(fields, -1)
    assert result.shape == (4, 2, 8, 10)


def test_positive_steps_produce_that_many_timesteps(fields):
    assert run(fields, 5).shape == (5, 2, 8, 10)


def test_output_is_finite(fields):
    assert np.isfinite(run(fields, 6)).all()


def test_same_seed_gives_identical_output(fields):
    assert np.array_equal(run(fields, 3, seed=7), run(fields, 3, seed=7))


def test_different_seeds_give_different_output(fields):
    assert not np.allclose(run(fields, 3, seed=1), run(fields, 3, seed=2))


def test_rollout_depends_on_the_feedback_wiring(fields):
    """Feeding predictions back changes the trajectory — the rollout is real."""
    without = run(fields, 4, feedback=np.array([-1, -1, -1]))
    with_feedback = run(fields, 4, feedback=np.array([0, 1, -1]))
    assert not np.allclose(without, with_feedback)
    # Without feedback the state never changes, so every step is identical.
    assert np.allclose(without[0], without[-1])
    assert not np.allclose(with_feedback[0], with_feedback[-1])


def test_channel_stats_handle_a_constant_channel():
    constant = np.full((2, 1, 4, 4), 5.0, dtype="float32")
    mean, std = channel_stats(constant)
    assert mean[0] == pytest.approx(5.0)
    assert std[0] == 1.0  # not zero: normalisation must stay a plain shift


def test_output_stats_inherit_matching_input_variables():
    levels = parse_level_coords("isobaricInhPa:850/500")
    in_specs = parse_var_specs("t2m,t@isobaricInhPa", levels, "INPUT_VARIABLES")
    out_specs = parse_var_specs("t2m,tp", levels, "OUTPUT_VARIABLES")
    in_layout = channel_layout(in_specs, levels)
    out_layout = channel_layout(out_specs, levels)

    in_mean = np.array([280.0, 250.0, 230.0], dtype="float32")
    in_std = np.array([5.0, 8.0, 9.0], dtype="float32")
    mean, std = output_stats(in_layout, out_layout, in_mean, in_std)

    assert mean[0] == pytest.approx(280.0)  # t2m inherits t2m
    assert std[0] == pytest.approx(5.0)
    # tp is new, so it falls back to the average input scale.
    assert mean[1] == pytest.approx(in_mean.mean(), rel=1e-5)


def test_output_stats_match_per_level():
    levels = parse_level_coords("isobaricInhPa:850/500")
    specs = parse_var_specs("t@isobaricInhPa", levels, "INPUT_VARIABLES")
    layout = channel_layout(specs, levels)

    in_mean = np.array([250.0, 230.0], dtype="float32")
    in_std = np.array([8.0, 9.0], dtype="float32")
    mean, _ = output_stats(layout, layout, in_mean, in_std)
    assert list(mean) == [250.0, 230.0]


def test_feedback_index_matches_on_name_and_level():
    levels = parse_level_coords("isobaricInhPa:850/500")
    in_specs = parse_var_specs("t2m,u10,t@isobaricInhPa", levels, "INPUT_VARIABLES")
    out_specs = parse_var_specs("t@isobaricInhPa,t2m", levels, "OUTPUT_VARIABLES")
    index = feedback_index(channel_layout(in_specs, levels), channel_layout(out_specs, levels))
    # t2m -> output channel 2, u10 has no prediction, t levels -> channels 0 and 1.
    assert list(index) == [2, -1, 0, 1]


def test_cpu_is_always_selectable():
    assert select_device("cpu").type == "cpu"


def test_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"


def test_requesting_cuda_without_a_device_fails_loudly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceError, match="no CUDA device is visible"):
        select_device("cuda")
