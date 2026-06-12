#!/usr/bin/env python3
"""Run one lbm_eval task and save rollouts in the canonical trajectory layout.

Single-entry orchestrator for policy_graphs: launches the embedding-capturing
policy server (``inference_policy_capture.py``) on this machine, runs the
lbm_eval Drake simulator in Docker against it, then assembles the canonical
layout from the simulator's ``results.json``/videos and the server's
embedding segments::

    <output_dir>/episodes/metadata.yaml        episodes: [{index, length, success}]
    <output_dir>/policy_embeddings/action_conditioning.npz   key "embeddings"
    <output_dir>/media/ep{idx:04d}_{success|failure}.mp4

``index`` is the lbm_eval scenario index, ``length`` counts inference calls
(one embedding row each; the policy re-plans every ``--open_loop_steps`` env
steps). Episodes are matched to embedding segments via the per-episode seed
the simulator sends in the gRPC reset — the run fails loudly if seeds and
scenario indices cannot be reconciled, never writing misaligned arrays.

Episodes may run in parallel (``--num-processes``): segments are keyed by
gRPC client+seed, so interleaving is safe. Retries produce several segments
with the same seed and only the last one (the attempt that produced the
recorded result) is kept.

Usage::

    uv run --no-sync --group inference python vla_foundry/eval/run_eval_save_rollouts.py \
        --checkpoint TRI-ML/Foundry-VLA-1.7B-sim --task PutMugOnSaucer \
        --num-episodes 10 --output-dir /tmp/rollouts --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

LAYER_NAME = "action_conditioning"
DEFAULT_DOCKER_IMAGE = "toyotaresearch/lbm-eval-oss:vla-foundry"


def wait_for_port(port: int, proc: subprocess.Popen, log_path: Path, timeout: int = 1800) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text()[-4000:] if log_path.exists() else "<no log>"
            raise RuntimeError(f"Policy server exited with code {proc.returncode}. Log tail:\n{tail}")
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise RuntimeError(f"Policy server did not open port {port} within {timeout}s")


def get_render_gid() -> str | None:
    try:
        return str(os.stat("/dev/dri/renderD128").st_gid)
    except OSError:
        return None


def launch_policy_server(args, dump_dir: Path, log_path: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "inference" / "robotics" / "inference_policy_capture.py"),
        "--checkpoint_directory",
        args.checkpoint,
        "--num_flow_steps",
        str(args.num_flow_steps),
        "--open_loop_steps",
        str(args.open_loop_steps),
        "--device",
        args.device,
        "--server-uri",
        f"0.0.0.0:{args.port}",
        "--embedding_dump_dir",
        str(dump_dir),
    ]
    if args.checkpoint_name:
        cmd += ["--checkpoint_name", args.checkpoint_name]
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Survive co-tenant GPU pressure: a fragmented small alloc OOM kills the
    # server task and every simulator then hangs on gRPC forever.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_file = open(log_path, "w")  # noqa: SIM115 — must outlive Popen
    return subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env, start_new_session=True)


def run_simulator(args, raw_dir: Path, log_path: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(raw_dir, 0o777)
    group_add = ["--group-add", "video"]
    render_gid = get_render_gid()
    if render_gid:
        group_add += ["--group-add", render_gid]
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--label",
        "vla-foundry-eval",
        "--runtime=nvidia",
        "--gpus",
        f"device={args.sim_gpu if args.sim_gpu is not None else args.gpu}"
        if (args.sim_gpu is not None or args.gpu is not None)
        else "all",
        "--device",
        "/dev/dri",
        *group_add,
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=all",
        # The image pins cuda>=12.6; CUDA minor-version compatibility makes it
        # run fine on 12.5-era drivers, so skip the hard requirement check.
        "-e",
        "NVIDIA_DISABLE_REQUIRE=1",
        "-e",
        f"LAUNCH_DEMONSTRATION_INDICES={args.num_episodes}",
        "-e",
        f"NUM_PROCESSES={args.num_processes}",
        "-e",
        "POLICY_HOST=localhost",
        "-e",
        f"POLICY_PORT={args.port}",
        "-e",
        f"USE_EVAL_SEED={args.use_eval_seed}",
        "-e",
        f"MAX_RETRIES={args.max_retries}",
        "-e",
        "RECORD_VIDEO=1",
        "-e",
        f"VIDEO_FPS={args.video_fps}",
        "-e",
        f"MAX_SAMPLE_SIZE_PER_MODEL={args.max_sample_size}",
        "-v",
        f"{raw_dir.resolve()}:/tmp/lbm/rollouts",
        args.docker_image,
        "bash",
        "/opt/lbm_eval/launch_sim.sh",
        args.task,
    ]
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=log_file)
    if result.returncode != 0:
        tail = log_path.read_text()[-4000:]
        raise RuntimeError(f"Simulator exited with code {result.returncode}. Log tail:\n{tail}")


def load_results(raw_dir: Path) -> tuple[Path, list[dict]]:
    """Return (timestamp_dir, completed evaluations) from the single results.json."""
    result_files = sorted(raw_dir.rglob("results.json"))
    if len(result_files) != 1:
        raise RuntimeError(f"Expected exactly one results.json under {raw_dir}, found {len(result_files)}")
    data = json.loads(result_files[0].read_text())
    evaluations = [ep for ep in data.get("evaluations", []) if ep.get("is_success") is not None]
    skipped = len(data.get("evaluations", [])) - len(evaluations)
    if skipped:
        print(f"WARNING: {skipped} episode(s) pending/crashed (no is_success) — excluded", flush=True)
    if not evaluations:
        raise RuntimeError(f"No completed evaluations in {result_files[0]}")
    return result_files[0].parent, evaluations


def load_segments(dump_dir: Path) -> list[dict]:
    manifest_path = dump_dir / "segments.json"
    if not manifest_path.exists():
        raise RuntimeError(f"No embedding manifest at {manifest_path} — server flush failed?")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise RuntimeError("Embedding manifest is missing the final flush (complete=false)")
    if manifest.get("capture_errors"):
        raise RuntimeError(f"{manifest['capture_errors']} embedding capture error(s) during the run")
    segments = manifest["segments"]
    for segment in segments:
        segment["rows"] = np.load(dump_dir / segment["file"])
    return segments


def match_segments(evaluations: list[dict], segments: list[dict]) -> list[tuple[dict, dict]]:
    """Pair each completed evaluation with its embedding segment via the reset seed.

    The simulator derives the reset seed from the scenario index by a constant
    offset (observed: ``seed = 100000100 + scenario_index`` with
    ``USE_EVAL_SEED=1``; ``0`` would mean seeds are the raw indices). The
    offset is recovered from the data: sorted seeds paired with sorted
    scenario indices must differ by one constant, otherwise the run aborts
    rather than writing misaligned arrays.

    Retries produce several segments with the same seed; the last one (highest
    ``order``) corresponds to the attempt whose outcome was recorded.
    """
    by_seed: dict[int, dict] = {}
    for segment in sorted(segments, key=lambda s: s["order"]):
        if segment["num_rows"] > 0:
            by_seed[segment["seed"]] = segment
    seeds = sorted(by_seed)
    indices = sorted(ep["scenario_index"] for ep in evaluations)
    if len(seeds) != len(indices):
        raise RuntimeError(
            f"Cannot align embeddings to episodes: {len(indices)} completed evaluation(s) "
            f"(scenario indices {indices}) vs {len(seeds)} embedding segment(s) "
            f"(seeds {seeds}). Refusing to write misaligned arrays."
        )
    offsets = {seed - idx for seed, idx in zip(seeds, indices, strict=True)}
    if len(offsets) != 1:
        raise RuntimeError(
            f"Cannot align embeddings to episodes: reset seeds {seeds} are not scenario "
            f"indices {indices} shifted by one constant (offsets {sorted(offsets)}). "
            f"Refusing to write misaligned arrays."
        )
    offset = offsets.pop()
    print(f"Seed/scenario-index offset: {offset}", flush=True)
    return [
        (ep, by_seed[ep["scenario_index"] + offset]) for ep in sorted(evaluations, key=lambda e: e["scenario_index"])
    ]


def find_video(timestamp_dir: Path, skill_type: str, scenario_index: int) -> Path | None:
    demo_dir = timestamp_dir / skill_type / f"demonstration_{scenario_index}"
    if demo_dir.is_dir():
        for video in sorted(demo_dir.glob("*.mp4")):
            return video
    return None


def write_canonical_layout(output_dir: Path, pairs: list[tuple[dict, dict]], timestamp_dir: Path, args) -> dict:
    episodes_meta = []
    rows = []
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    dims = {p[1]["rows"].shape[1] for p in pairs}
    if len(dims) != 1:
        raise RuntimeError(f"Inconsistent embedding dims across episodes: {dims}")
    for ep, segment in pairs:
        index = int(ep["scenario_index"])
        success = bool(ep["is_success"])
        episodes_meta.append({"index": index, "length": int(segment["num_rows"]), "success": success})
        rows.append(segment["rows"])
        video = find_video(timestamp_dir, ep["skill_type"], index)
        if video is not None:
            shutil.copy2(video, media_dir / f"ep{index:04d}_{'success' if success else 'failure'}.mp4")
    embeddings = np.concatenate(rows, axis=0).astype(np.float32)
    total = sum(e["length"] for e in episodes_meta)
    if embeddings.shape[0] != total:
        raise RuntimeError(f"Row count mismatch: embeddings {embeddings.shape[0]} vs metadata {total}")

    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    (episodes_dir / "metadata.yaml").write_text(yaml.safe_dump({"episodes": episodes_meta}, sort_keys=False))
    emb_dir = output_dir / "policy_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(emb_dir / f"{LAYER_NAME}.npz", embeddings=embeddings)

    return {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "num_episodes": len(episodes_meta),
        "success_rate": sum(e["success"] for e in episodes_meta) / len(episodes_meta),
        "total_inference_calls": total,
        "embedding_dim": int(embeddings.shape[1]),
        "layer": LAYER_NAME,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir, hf:// URI, or HF repo id")
    parser.add_argument("--checkpoint_name", default=None)
    parser.add_argument("--task", required=True, help="lbm_eval task name (PascalCase)")
    parser.add_argument("--num-episodes", default="0:10", help="Scenario index range start:end (or a count N = 0:N)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU for the policy server (and simulator unless --sim-gpu)"
    )
    parser.add_argument("--sim-gpu", type=int, default=None, help="Separate GPU for the simulator's rendering")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--num_flow_steps", type=int, default=8)
    parser.add_argument("--open_loop_steps", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--docker_image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Parallel simulator processes against the one policy server. Embedding "
        "segments are keyed by gRPC client+seed, so interleaved episodes stay "
        "correctly attributed; the seed/scenario alignment checks are order-independent.",
    )
    parser.add_argument("--use_eval_seed", type=int, default=1)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument("--max_sample_size", type=int, default=200)
    parser.add_argument("--keep-raw", action="store_true", help="Keep the raw simulator output dir")
    args = parser.parse_args()

    if ":" not in args.num_episodes:
        args.num_episodes = f"0:{int(args.num_episodes)}"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    dump_dir = output_dir / "raw" / "embedding_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    server_log = output_dir / "policy_server.log"
    sim_log = output_dir / "simulator.log"

    server = launch_policy_server(args, dump_dir, server_log)
    try:
        print(f"Waiting for policy server on port {args.port} (log: {server_log})", flush=True)
        wait_for_port(args.port, server, server_log)
        print(f"Running simulator: task={args.task} episodes={args.num_episodes} (log: {sim_log})", flush=True)
        run_simulator(args, raw_dir, sim_log)
    finally:
        if server.poll() is None:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired as err:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
                raise RuntimeError("Policy server did not flush embeddings within 120s of SIGTERM") from err

    timestamp_dir, evaluations = load_results(raw_dir)
    segments = load_segments(dump_dir)
    print(f"{len(evaluations)} completed episode(s); reset seeds: {sorted({s['seed'] for s in segments})}", flush=True)
    pairs = match_segments(evaluations, segments)
    summary = write_canonical_layout(output_dir, pairs, timestamp_dir, args)

    if not args.keep_raw:
        cleanup_raw(raw_dir, args.docker_image)

    print(json.dumps(summary, indent=2), flush=True)


def cleanup_raw(raw_dir: Path, docker_image: str) -> None:
    """Best-effort removal of the raw simulator output.

    The simulator container writes its result tree as the image's own user,
    so a plain rmtree from the host hits PermissionError. Fall back to
    deleting through a container with the same user, and never fail the
    eval over cleanup.
    """
    try:
        shutil.rmtree(raw_dir)
        return
    except PermissionError:
        pass
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{raw_dir.resolve()}:/cleanup",
            docker_image,
            "bash",
            "-c",
            "rm -rf /cleanup/* /cleanup/.[!.]*",
        ],
        capture_output=True,
    )
    try:
        if result.returncode == 0:
            shutil.rmtree(raw_dir)
            return
    except PermissionError:
        pass
    print(f"WARNING: could not remove raw simulator output at {raw_dir}", flush=True)


if __name__ == "__main__":
    main()
