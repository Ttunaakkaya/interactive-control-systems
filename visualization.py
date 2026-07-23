"""Pure helpers for keeping the presentation layer responsive."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AnimationSampling:
    """A duration-preserving subset of simulation samples for animation."""

    indices: tuple[int, ...]
    frame_duration_ms: int
    source_fps: float
    rendered_fps: float


def build_animation_sampling(
    sample_count: int,
    dt: float,
    target_fps: float = 25.0,
) -> AnimationSampling:
    """Choose animation frames without changing the simulation data.

    The browser does not benefit from receiving more frames than it can display
    smoothly. Samples are selected uniformly (including both endpoints), while
    the per-frame duration is increased so total playback time stays equal to
    the original ``sample_count * dt`` behavior.
    """

    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite number")
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a positive finite number")

    source_fps = 1.0 / dt
    desired_count = min(
        sample_count,
        max(1, math.ceil(sample_count * dt * target_fps)),
    )

    if desired_count == 1:
        indices = (0,)
    else:
        last_index = sample_count - 1
        indices = tuple(
            round(i * last_index / (desired_count - 1))
            for i in range(desired_count)
        )

    frame_duration_ms = max(
        1,
        round(1000.0 * sample_count * dt / len(indices)),
    )
    rendered_fps = len(indices) / (sample_count * dt)

    return AnimationSampling(
        indices=indices,
        frame_duration_ms=frame_duration_ms,
        source_fps=source_fps,
        rendered_fps=rendered_fps,
    )
