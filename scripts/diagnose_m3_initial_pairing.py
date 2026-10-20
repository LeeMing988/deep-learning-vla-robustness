#!/usr/bin/env python3
"""Bounded N10 reset diagnostic. Does not patch code, train models, or write evaluation results.

Reuses the installed N10 rollout and its original strict InitialStateAudit. Captures
raw starting observations BEFORE the guard, so a mismatch is inspectable rather
than reduced to an opaque hash. At most replicas 0..2 are rolled out for each model;
replica 3 is reset only. Diagnostic outcomes are NOT research performance results.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
from importlib import metadata
import inspect
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import traceback
import zipfile

import numpy as np

TASK = 2
SEED = 42
CONDITION = "E0_ID"
MAX_REPLICA = 3


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def leaves(value, path="root") -> dict[str, np.ndarray]:
    """Copies leaf values; never holds views into mutable simulator observations."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, dict):
        result = {}
        for key in sorted(value):
            result.update(leaves(value[key], path + "/" + str(key)))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, item in enumerate(value):
            result.update(leaves(item, path + "/" + str(index)))
        return result
    arr = np.asarray("<None>" if value is None else value)
    if arr.dtype.kind == "O":
        raise TypeError(f"Unsupported object-valued observation at {path}")
    return {path: np.array(arr, copy=True)}


def compare(left: dict, right: dict) -> list[dict]:
    details = []
    for key in sorted(set(left) | set(right)):
        row = {"field": key}
        if key not in left or key not in right:
            row["difference"] = "missing field"
            details.append(row)
            continue
        a, b = left[key], right[key]
        row.update(left_shape=list(a.shape), right_shape=list(b.shape),
                   left_dtype=str(a.dtype), right_dtype=str(b.dtype))
        same_layout = a.shape == b.shape and a.dtype == b.dtype
        row["byte_equal"] = bool(same_layout and a.tobytes() == b.tobytes())
        if row["byte_equal"]:
            continue
        if a.shape == b.shape:
            if a.dtype.kind in "buif" and b.dtype.kind in "buif":
                av, bv = a.astype(np.float64), b.astype(np.float64)
                row["finite"] = bool(np.isfinite(av).all() and np.isfinite(bv).all())
                row["numerically_changed_elements"] = int(np.count_nonzero(av != bv))
                row["elements"] = int(a.size)
                if row["finite"] and a.size:
                    delta = np.abs(av - bv)
                    row["max_abs_difference"] = float(delta.max())
                    row["mean_abs_difference"] = float(delta.mean())
                # Differences are reported, never accepted using a new tolerance.
            else:
                row["difference"] = "non-numeric field or dtype difference"
        else:
            row["difference"] = "shape differs"
        details.append(row)
    return details


class SourceCollector:
    """Read sources for actual environment/wrapper classes encountered at runtime."""
    def __init__(self, directory: Path):
        self.directory = directory
        self.seen_files = set()
        self.records = []

    def capture(self, env) -> None:
        seen = set()
        pending = [("vector_env", env, 0)]
        while pending:
            label, obj, depth = pending.pop(0)
            if obj is None or id(obj) in seen or depth > 8:
                continue
            seen.add(id(obj))
            cls = type(obj)
            if cls.__module__.startswith(("numpy", "torch")):
                continue
            record = {"object": label, "class": cls.__module__ + "." + cls.__qualname__}
            self.records.append(record)
            for base in cls.__mro__:
                if base is object:
                    continue
                try:
                    filename = inspect.getsourcefile(base)
                    if not filename:
                        continue
                    path = Path(filename).resolve()
                    if path in self.seen_files or not path.is_file():
                        continue
                    if path.stat().st_size > 2 * 1024**2:
                        continue
                    content = path.read_text(encoding="utf-8")
                    self.seen_files.add(path)
                    name = f"{len(self.seen_files):02d}_{path.name}.txt"
                    self.directory.mkdir(parents=True, exist_ok=True)
                    (self.directory / name).write_text(
                        f"SOURCE: {path}\nSHA256: {digest(path)}\n\n" + content,
                        encoding="utf-8")
                except (OSError, TypeError, UnicodeError):
                    continue
            # Read already-stored attributes, not arbitrary descriptors/getters.
            attrs = vars(obj) if hasattr(obj, "__dict__") else {}
            for key in ("env", "_env", "wrapped_env", "_wrapped_env", "environment", "_environment"):
                child = attrs.get(key)
                if child is not None:
                    pending.append((label + "." + key, child, depth + 1))
            children = attrs.get("envs")
            if isinstance(children, (list, tuple)):
                pending.extend((label + f".envs[{i}]", child, depth + 1)
                               for i, child in enumerate(children[:1]))


class CaptureReset:
    def __init__(self, env, callback):
        self.env = env
        self.callback = callback

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        self.callback(observation)
        return observation, info

    def call(self, *args, **kwargs):
        return self.env.call(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    if Path(sys.prefix).name != "vla-libero":
        raise SystemExit("Use /home/leeming/miniforge3/envs/vla-libero/bin/python.")
    evaluator_path = root / "src/m3_rva_evaluation.py"
    seed_dir = root / "results/metrics/m3_rva_evaluation/v1/seed42"
    if not evaluator_path.is_file() or not (seed_dir / "inputs.json").is_file():
        raise SystemExit("Expected the existing N10 evaluator and seed42 inputs.json; no files changed.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "results/metrics/m3_initial_pair_diagnostics" / stamp
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "reset_diagnostic.json"
    report = {"purpose": "RESET_DIAGNOSTIC_ONLY; not an evaluation result",
              "task": TASK, "training_seed": SEED, "condition": CONDITION,
              "max_full_rollouts": 6, "final_replica_reset_only": MAX_REPLICA,
              "status": "STARTED", "snapshots": [], "exceptions": []}
    collector = SourceCollector(output / "environment_source")
    (output / "reviewed_local_evaluator.py.txt").write_text(
        evaluator_path.read_text(encoding="utf-8"), encoding="utf-8")
    # Inventory completed episode files without modifying them.
    initial_episode_hashes = {str(p.relative_to(root)): digest(p)
                              for p in sorted((seed_dir / "episodes").glob("*.json"))}
    report["existing_episode_count"] = len(initial_episode_hashes)
    report["existing_episode_counts_by_variant"] = {}
    old_rows = []
    for rel in initial_episode_hashes:
        row = json.loads((root / rel).read_text(encoding="utf-8"))
        variant = str(row.get("variant", "UNKNOWN"))
        counts = report["existing_episode_counts_by_variant"]
        counts[variant] = counts.get(variant, 0) + 1
        if row.get("suite_task_id") == TASK:
            old_rows.append(row)
    write_json(output / "existing_task2_episode_records.json", old_rows)
    for name in ("runtime.json", "evaluation_protocol.json", "last_error.json"):
        path = seed_dir.parent / name
        if path.is_file():
            (output / ("original_" + name)).write_bytes(path.read_bytes())
    spec = importlib.util.spec_from_file_location("m3_pair_diag_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise SystemExit("Could not inspect the evaluator module.")
    ev = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ev
    before = {}
    try:
        spec.loader.exec_module(ev)  # Does not invoke main().
        with ev.exclusive_rig(root):
            context = ev.Evaluation(root, [SEED])
            context.prerequisites()
            # Do not call freeze(), seed_inputs(), execute(), or summarize(): no formal writes.
            before = {p: digest(root / p) for p in context.small_inputs}
            ev.EVALUATION_PROTOCOL_SHA256 = context.evaluation_hash
            ev.QUALITATIVE_DIR = output / "unused_qualitative"
            ev.load_components(root)
            original = json.loads((seed_dir / "inputs.json").read_text(encoding="utf-8"))
            report["evaluator_sha256"] = digest(evaluator_path)
            report["runtime"] = {name: metadata.version(name) for name in
                                 ("gymnasium", "lerobot", "hf-libero", "mujoco", "numpy", "torch")}
            reference = {}
            for variant in ("M2-LE", "M3-RVA"):
                checkpoint = ev.local_path(root, original["checkpoint_paths"][variant])
                expected_files = original["checkpoint_file_hashes"][variant]
                for name, sha in expected_files.items():
                    ev.require(digest(checkpoint / name) == sha, f"Checkpoint changed: {variant}/{name}")
                print(f"DIAGNOSTIC | loading {variant}; no training", flush=True)
                policy, pre, post = ev.load_model(variant, checkpoint)
                envs = None
                env = None
                try:
                    cfg = ev.build_env_config(TASK)
                    env_pre, env_post = ev.make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config)
                    envs = ev.make_env(cfg, n_envs=1, use_async_envs=False)
                    env = envs[ev.LIBERO_SUITE][TASK]
                    collector.capture(env)
                    for replica in range(MAX_REPLICA + 1):
                        tag = ("m2le" if variant == "M2-LE" else "m3rva") + f"_rep{replica}"

                        def capture(observation, replica=replica, tag=tag, variant=variant):
                            flat = leaves(observation)
                            obs_hash = ev.observation_sha(observation)
                            np.savez_compressed(output / (tag + "_initial.npz"), **flat)
                            snapshot = {"variant": variant, "replica": replica,
                                        "initial_observation_sha256": obs_hash,
                                        "arrays": tag + "_initial.npz"}
                            prior = next((r for r in old_rows if r.get("variant") == variant
                                          and r.get("condition_id") == CONDITION
                                          and r.get("replica") == replica), None)
                            if prior:
                                snapshot["existing_hash"] = prior["initial_observation_sha256"]
                                snapshot["matches_existing_record"] = obs_hash == prior["initial_observation_sha256"]
                            if variant == "M2-LE":
                                reference[replica] = (obs_hash, flat)
                            else:
                                baseline_hash, baseline_flat = reference[replica]
                                snapshot["matches_diagnostic_baseline"] = obs_hash == baseline_hash
                                snapshot["field_differences"] = compare(baseline_flat, flat)
                            report["snapshots"].append(snapshot)
                            write_json(report_path, report)
                            print(f"CAPTURED | {tag} | hash={obs_hash}", flush=True)
                            if snapshot.get("field_differences"):
                                print(json.dumps(snapshot["field_differences"], indent=2), flush=True)

                        raw_proxy = CaptureReset(env, capture)
                        expected = reference[replica][0] if variant == "M3-RVA" else None
                        audited = ev.InitialStateAudit(raw_proxy, expected)
                        if replica == MAX_REPLICA:
                            audited.reset(seed=100000 + TASK * 100 + replica)
                            print(f"DIAGNOSTIC | {variant} rep{replica}: reset only; no action", flush=True)
                        else:
                            ev.evaluate_one_episode(
                                env=audited, policy=policy, preprocessor=pre, postprocessor=post,
                                env_preprocessor=env_pre, env_postprocessor=env_post,
                                run_id="RESET_DIAGNOSTIC_ONLY", variant=variant, model_seed=SEED,
                                checkpoint_signature=original["checkpoint_signatures"][variant],
                                condition_id=CONDITION, suite_task_id=TASK, replica=replica,
                                capture_qualitative=False)
                            print(f"DIAGNOSTIC | {variant} rep{replica}: prior history replayed; not a scored result", flush=True)
                finally:
                    if env is not None:
                        env.close()
                    del policy, pre, post, env, envs
                    gc.collect()
                    ev.torch.cuda.empty_cache()
            report["status"] = "NO_PAIR_MISMATCH_IN_THIS_BOUNDED_REPLAY"
            report["interpretation"] = "Does not rule out drift after other prior histories or in saved results."
    except Exception as exc:
        report["exceptions"].append({"type": type(exc).__name__, "message": str(exc),
                                      "traceback": traceback.format_exc()})
        report["status"] = ("PAIR_MISMATCH_CAPTURED" if "Paired initial observations differ" in str(exc)
                            else "DIAGNOSTIC_ERROR")
        print(traceback.format_exc(), flush=True)
    finally:
        report["environment_classes"] = collector.records
        report["original_small_inputs_unchanged"] = all((root / p).is_file() and digest(root / p) == sha
                                                       for p, sha in before.items()) if before else None
        report["existing_episode_files_unchanged"] = all((root / p).is_file() and digest(root / p) == sha
                                                        for p, sha in initial_episode_hashes.items())
        report["formal_episode_count_after"] = len(list((seed_dir / "episodes").glob("*.json")))
        write_json(report_path, report)
        bundle = output / "reset_diagnostic_bundle.zip"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(output.rglob("*")):
                if path.is_file() and path != bundle:
                    zf.write(path, path.relative_to(output))
        print("\nDIAGNOSTIC STATUS:", report["status"], flush=True)
        print("Existing episode files unchanged:", report["existing_episode_files_unchanged"], flush=True)
        print("Upload this diagnostic ZIP:", bundle, flush=True)
        print("No training, patching, tolerance change, or formal evaluation result was performed.", flush=True)
    return 0 if report["status"] != "DIAGNOSTIC_ERROR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
