"""RoboMME benchmark that perturbs the conditioning video before the policy sees it.

Only the frames sent to the model as ``video_history`` change. The scene, seed and task are
the stock benchmark's, so every condition runs the same test episodes and outcomes can be
compared episode by episode. The model, its memory module and the harness are untouched.

Perturbations (the ``perturbation`` benchmark param):
    none          original video
    blank         every frame black, same length: removes content, keeps the buffer size
    shuffle       all frames randomly permuted: changes order and, because FrameSamp samples
                  frames by buffer position, also which frames it keeps
    shuffle_kept  only the frames FrameSamp keeps are permuted among themselves: identical
                  content in scrambled order, a pure test of temporal order
    reverse       frames in reverse order: tests causal direction; FrameSamp keeps the same
                  frames up to a one-frame rounding offset

Permutations are seeded from (task, episode), so they are identical across models and reruns.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
from vla_eval.benchmarks.robomme.benchmark import RoboMMEBenchmark

logger = logging.getLogger(__name__)

PERTURBATIONS = ("none", "blank", "shuffle", "shuffle_kept", "reverse")

# FrameSamp keeps budget // token_per_image = 512 // 16 frames, evenly spaced over the buffer:
# mme_vla_suite.shared.data_utils.even_sampling_indices, at step_idx = n_frames - 1.
KEPT_FRAMES = 32


def kept_indices(n_frames: int, budget: int = KEPT_FRAMES) -> list[int]:
    """Buffer positions FrameSamp loads from an n-frame video."""
    if n_frames <= budget:
        return list(range(n_frames))
    return np.linspace(0, n_frames - 1, budget, dtype=np.int32).tolist()


def frame_order(n_frames: int, perturbation: str, rng: np.random.Generator) -> list[int]:
    """Source index for each output position (``blank`` is handled separately)."""
    order = list(range(n_frames))
    if perturbation == "shuffle":
        order = rng.permutation(n_frames).tolist()
    elif perturbation == "reverse":
        order.reverse()
    elif perturbation == "shuffle_kept":
        kept = kept_indices(n_frames)
        for position, source in zip(kept, rng.permutation(kept).tolist()):
            order[position] = source
    return order


class PerturbedRoboMMEBenchmark(RoboMMEBenchmark):
    def __init__(self, *args: Any, perturbation: str = "none", **kwargs: Any) -> None:
        if perturbation not in PERTURBATIONS:
            raise ValueError(f"perturbation must be one of {PERTURBATIONS}, got {perturbation!r}")
        super().__init__(*args, **kwargs)
        self.perturbation = perturbation

    def reset(self, task: Any) -> Any:
        raw_obs = super().reset(task)
        n_frames = len(self._video_frames)
        if self.perturbation == "none" or n_frames == 0:
            return raw_obs

        key = f"{task['env_id']}:{task.get('episode_idx', 0)}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
        wrist = getattr(self, "_wrist_video_frames", [])

        if self.perturbation == "blank":
            self._video_frames = [np.zeros_like(f) for f in self._video_frames]
            self._wrist_video_frames = [np.zeros_like(f) for f in wrist]
        else:
            order = frame_order(n_frames, self.perturbation, rng)
            self._video_frames = [self._video_frames[i] for i in order]
            if len(wrist) == n_frames:
                self._wrist_video_frames = [wrist[i] for i in order]

        logger.info("video perturbation %s: %s episode %s, %d frames",
                    self.perturbation, task["env_id"], task.get("episode_idx"), n_frames)
        return raw_obs
