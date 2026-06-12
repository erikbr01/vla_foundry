#! /usr/bin/env python3
"""LBM policy server with per-inference-call embedding capture.

Drop-in replacement for ``inference_policy.py`` that additionally records one
policy embedding per inference call (one ``generate_actions`` invocation) and
dumps them to ``--embedding_dump_dir`` segmented by episode.

The captured embedding is the vision-language backbone's action-conditioning
output (``VisionLanguageBackboneOutput.embeddings``, shape ``[B, N, D]``),
mean-pooled over the token axis. It is computed exactly once per
``generate_actions`` call and conditions every denoising step of the action
chunk, making it the per-inference-call policy representation.

Episode segmentation relies on the gRPC reset protocol: the simulator resets
each client with a per-episode seed (``reset_batch(clients: {uuid: seed})``).
A new segment is opened per (client, reset) and closed by the next reset of
that client or by server shutdown. Completed segments are flushed to disk on
every reset; remaining open segments are flushed on SIGTERM/SIGINT/atexit.

Dump layout (under ``--embedding_dump_dir``)::

    segments.json                   [{file, seed, client, order, num_rows}]
    segment_0000.npy                (num_rows, D) float32

Embedding capture must never crash the rollout: capture errors are counted
and reported in the manifest, never raised into the serving path.
"""

import argparse
import atexit
import json
import logging
import os
import signal
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from grpc_workspace.lbm_policy_server import LbmPolicyServerConfig, run_policy_server
from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers

import vla_foundry.visualizers.visualizer as vz
from vla_foundry.inference.robotics.inference_policy import InferenceDiffusionPolicy
from vla_foundry.logger import setup_logging


@dataclass
class _Segment:
    """Embedding rows for one (client, reset) episode session."""

    seed: int
    client: str
    order: int
    rows: list = field(default_factory=list)


class EmbeddingCaptureDiffusionPolicy(InferenceDiffusionPolicy):
    """InferenceDiffusionPolicy that records one embedding per inference call."""

    def __init__(self, *args, embedding_dump_dir: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._dump_dir = Path(embedding_dump_dir)
        self._dump_dir.mkdir(parents=True, exist_ok=True)
        self._capture_lock = threading.Lock()
        self._open_segments: dict[uuid.UUID, _Segment] = {}
        self._closed_segments: list[_Segment] = []
        self._segment_counter = 0
        self._capture_errors = 0
        self._last_embedding = None
        self._wrap_backbone()

    def _wrap_backbone(self):
        backbone = self.model.vision_language_backbone
        original = backbone.get_action_conditioning

        def capturing(*args, **kwargs):
            output = original(*args, **kwargs)
            try:
                emb = output.embeddings.detach().float().mean(dim=(0, 1)).cpu().numpy()
                self._last_embedding = np.ascontiguousarray(emb, dtype=np.float32)
            except Exception:
                logging.exception("Embedding capture failed (rollout continues)")
                self._capture_errors += 1
            return output

        backbone.get_action_conditioning = capturing

    def step(self, observation: MultiarmObservation, client_id: uuid.UUID) -> PosesAndGrippers:
        self._last_embedding = None
        actions = super().step(observation, client_id)
        if self._last_embedding is not None:
            with self._capture_lock:
                segment = self._open_segments.get(client_id)
                if segment is None:
                    # Step before any reset — should not happen with lbm_eval,
                    # but never lose rows: open an unseeded segment.
                    logging.warning("step() for %s before reset; opening seedless segment", client_id)
                    segment = _Segment(seed=-1, client=str(client_id), order=self._segment_counter)
                    self._segment_counter += 1
                    self._open_segments[client_id] = segment
                segment.rows.append(self._last_embedding)
            self._last_embedding = None
        return actions

    def reset_batch(self, clients: dict[uuid.UUID, int]):
        with self._capture_lock:
            for client_id, seed in clients.items():
                previous = self._open_segments.pop(client_id, None)
                if previous is not None:
                    self._closed_segments.append(previous)
                self._open_segments[client_id] = _Segment(
                    seed=int(seed), client=str(client_id), order=self._segment_counter
                )
                self._segment_counter += 1
                logging.info("Opened embedding segment %d (seed=%s)", self._segment_counter - 1, seed)
        super().reset_batch(clients)
        self.flush(include_open=False)

    def flush(self, include_open: bool):
        """Write all captured segments + manifest to the dump dir."""
        with self._capture_lock:
            segments = list(self._closed_segments)
            if include_open:
                segments += list(self._open_segments.values())
            segments = sorted(segments, key=lambda s: s.order)
            manifest = []
            for segment in segments:
                name = f"segment_{segment.order:04d}.npy"
                rows = np.stack(segment.rows) if segment.rows else np.zeros((0, 0), dtype=np.float32)
                np.save(self._dump_dir / name, rows)
                manifest.append(
                    {
                        "file": name,
                        "seed": segment.seed,
                        "client": segment.client,
                        "order": segment.order,
                        "num_rows": len(segment.rows),
                    }
                )
            payload = {
                "segments": manifest,
                "capture_errors": self._capture_errors,
                "complete": include_open,
            }
            tmp = self._dump_dir / "segments.json.tmp"
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self._dump_dir / "segments.json")


def main():
    parser = argparse.ArgumentParser(description="LBM policy server with embedding capture")
    LbmPolicyServerConfig.add_argparse_arguments(parser)
    parser.add_argument("--checkpoint_directory", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_flow_steps", type=int, default=8)
    parser.add_argument("--open_loop_steps", type=int, default=8)
    parser.add_argument("--gripper_debounce_open_threshold", type=float, default=0.6)
    parser.add_argument("--gripper_debounce_close_threshold", type=float, default=0.4)
    parser.add_argument(
        "--embedding_dump_dir", type=str, required=True, help="Directory for embedding segments + manifest"
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if os.environ.get("DEBUG") == "1" else logging.INFO
    setup_logging(log_file=None, level=log_level)

    policy = EmbeddingCaptureDiffusionPolicy(
        checkpoint_directory=args.checkpoint_directory,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
        num_flow_steps=args.num_flow_steps,
        open_loop_steps=args.open_loop_steps,
        gripper_debounce_open_threshold=args.gripper_debounce_open_threshold,
        gripper_debounce_close_threshold=args.gripper_debounce_close_threshold,
        embedding_dump_dir=args.embedding_dump_dir,
    )

    def flush_and_exit(signum, frame):
        logging.info("Signal %s received — flushing embedding segments", signum)
        policy.flush(include_open=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, flush_and_exit)
    signal.signal(signal.SIGINT, flush_and_exit)
    atexit.register(policy.flush, include_open=True)

    vz.init(run_name=f"PolicyDataAdapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}", add_rank_to_run=True)
    run_policy_server(policy, args)


if __name__ == "__main__":
    main()
