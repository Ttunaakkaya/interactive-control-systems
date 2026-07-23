"""Tests for presentation-only sampling helpers."""

import pytest

from visualization import build_animation_sampling


def test_sampling_reduces_50_hz_animation_to_25_fps():
    sampling = build_animation_sampling(sample_count=500, dt=0.02)

    assert len(sampling.indices) == 250
    assert sampling.indices[0] == 0
    assert sampling.indices[-1] == 499
    assert all(a < b for a, b in zip(sampling.indices, sampling.indices[1:]))
    assert sampling.frame_duration_ms == 40
    assert sampling.source_fps == pytest.approx(50.0)
    assert sampling.rendered_fps == pytest.approx(25.0)


def test_sampling_keeps_low_rate_data_unchanged():
    sampling = build_animation_sampling(
        sample_count=100,
        dt=0.05,
        target_fps=25.0,
    )

    assert sampling.indices == tuple(range(100))
    assert sampling.frame_duration_ms == 50
    assert sampling.rendered_fps == pytest.approx(20.0)


def test_sampling_preserves_playback_duration_with_rounding_tolerance():
    sampling = build_animation_sampling(
        sample_count=403,
        dt=0.02,
        target_fps=25.0,
    )

    original_duration_ms = 403 * 0.02 * 1000
    rendered_duration_ms = (
        len(sampling.indices) * sampling.frame_duration_ms
    )
    assert abs(rendered_duration_ms - original_duration_ms) <= len(
        sampling.indices
    ) / 2


@pytest.mark.parametrize(
    ("sample_count", "dt", "target_fps"),
    [
        (0, 0.02, 25.0),
        (10, 0.0, 25.0),
        (10, float("nan"), 25.0),
        (10, 0.02, 0.0),
        (10, 0.02, float("inf")),
    ],
)
def test_sampling_rejects_invalid_inputs(sample_count, dt, target_fps):
    with pytest.raises(ValueError):
        build_animation_sampling(sample_count, dt, target_fps)
