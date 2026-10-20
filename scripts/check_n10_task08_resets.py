#!/usr/bin/env python3
"""Three reset-only checks of the captured N10 V2 mismatch.

No policy/model loading, no learned actions, no scored episodes, no threshold
changes, no edits to N09/N10 or existing results. LIBERO's normal reset still
includes its unchanged built-in settling steps. All three observations are
retained, regardless of whether they satisfy the existing V2 image rule.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

EVALUATOR_SHA = '60e5757f7c08c5207268ea3631e9e5f22949877a4f68de82c9148d72c721e9ef'
AUDIT_REL = 'results/metrics/m3_rva_evaluation/v2/seed42/pairing_audits/E1_L3__task08__rep2.json'
TASK, REPLICA, CONDITION, RESET_SEED, REPEATS = 8, 2, 'E1_L3', 100802, 3


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def reset_checks(ev, make_env, seed_dir, audit, output: Path, report: dict) -> None:
    """Uses the exact V2 reset procedure, but only reads comparisons; no acceptance gate."""
    np = ev.np
    references = {}
    for label, key in [('M2-LE', 'baseline_capture'), ('M3-RVA', 'modified_capture')]:
        path = ev.local_path(seed_dir, audit[key])
        metadata = ev.read_json(path.with_suffix('.json'))
        ev.require(metadata['variant'] == label, 'Wrong captured model label.')
        for field in ('task_id', 'condition_id', 'replica', 'initial_state_index',
                      'initial_state_vector_sha256', 'evaluation_protocol_sha256'):
            ev.require(metadata[field] == audit[field], f'Capture identity changed: {field}')
        references[label] = ev.read_initial_capture(seed_dir, metadata)
    report['saved_pair_recomputed'] = ev.compare_initial_arrays(references['M2-LE'], references['M3-RVA'])
    for index in range(REPEATS):
        env = None
        try:
            ev.random.seed(RESET_SEED); np.random.seed(RESET_SEED); ev.torch.manual_seed(RESET_SEED)
            config = ev.build_env_config(TASK)
            envs = make_env(config, n_envs=1, use_async_envs=False)
            env = envs[ev.LIBERO_SUITE][TASK]
            selected = ev.prepare_fixed_state(env, TASK, CONDITION, REPLICA)
            ev.require(selected['initial_state_index'] == audit['initial_state_index'], 'State index changed.')
            ev.require(selected['initial_state_vector_sha256'] == audit['initial_state_vector_sha256'], 'Benchmark state vector changed.')
            ev.random.seed(RESET_SEED); np.random.seed(RESET_SEED); ev.torch.manual_seed(RESET_SEED)
            if ev.torch.cuda.is_available():
                ev.torch.cuda.manual_seed_all(RESET_SEED)
            raw, _ = env.reset(seed=RESET_SEED)
            ev.require(env.envs[0].init_state_id == selected['initial_state_index'] + 1, 'Wrapper selected/advanced an unexpected state.')
            arrays = ev.flatten_observation(raw)
            ev.validate_observation_arrays(arrays)
            filename = f'reset_{index+1}_observations.npz'
            np.savez_compressed(output / filename, **arrays)
            item = {
                'reset_number': index+1, 'selection': selected,
                'raw_observation_sha256': ev.observation_sha(raw),
                'capture': filename, 'capture_sha256': digest(output / filename),
                'comparisons': {label: ev.compare_initial_arrays(reference, arrays)
                                for label, reference in references.items()},
                'diagnostic_only': True,
            }
            # Optional read-only physics snapshot. Absence does not change any result.
            # This snapshot is not claimed to include every hidden renderer/controller variable.
            try:
                wrapped = getattr(env.envs[0], '_env', None)
                inner = getattr(wrapped, 'env', wrapped)
                sim = getattr(inner, 'sim', None)
                if sim is not None and callable(getattr(sim, 'get_state', None)):
                    state = np.array(sim.get_state().flatten(), copy=True)
                    np.save(output / f'reset_{index+1}_sim_state.npy', state, allow_pickle=False)
                    item['sim_get_state_sha256'] = ev.observation_sha(state)
                else:
                    item['sim_get_state_note'] = 'Not exposed; observation comparison still recorded.'
            except Exception as exc:
                item['sim_get_state_note'] = f'Optional capture unavailable: {type(exc).__name__}: {exc}'
            report['resets'].append(item)
            save_json(output / 'reset_repeatability.json', report)
            print(f'RESET {index+1}/{REPEATS} | no policy actions | hash={item["raw_observation_sha256"]}', flush=True)
            for label, comparison in item['comparisons'].items():
                print(f'  Compared with saved {label}: state_exact={comparison["observed_robot_state_exact"]}, V2_rule_pass={comparison["pass"]}', flush=True)
                for camera in comparison['camera_comparisons']:
                    print(f'    {camera["field"]}: changed_channels={camera["changed_channels"]}, max_difference={camera["max_abs_difference"]}', flush=True)
        finally:
            if env is not None:
                env.close()
    report['diagnostic_status'] = 'THREE_RESETS_RECORDED'
    report['all_resets_pass_original_baseline_rule'] = all(r['comparisons']['M2-LE']['pass'] for r in report['resets'])
    report['distinct_observation_hashes'] = len({r['raw_observation_sha256'] for r in report['resets']})
    report['interpretation'] = 'Repeatability evidence only, not a fix or authorization to accept a failed V2 pair.'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    if Path(sys.prefix).name != 'vla-libero':
        raise SystemExit('Use /home/leeming/miniforge3/envs/vla-libero/bin/python; do not install packages into system Python.')
    root = args.root.resolve()
    evaluator = root / 'src/m3_rva_evaluation.py'
    if not evaluator.is_file() or digest(evaluator) != EVALUATOR_SHA:
        raise SystemExit('The installed evaluator differs from the reviewed V2 file. No diagnostic run or patch performed.')
    audit_path = root / AUDIT_REL
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if (audit['task_id'], audit['replica'], audit['condition_id'], audit['initial_state_index']) != (TASK, REPLICA, CONDITION, 7):
        raise SystemExit('Unexpected captured episode identity; no diagnostic run performed.')
    seed_dir = audit_path.parent.parent
    spec = importlib.util.spec_from_file_location('_reviewed_n10_reset_probe', evaluator)
    if spec is None or spec.loader is None:
        raise SystemExit('Unable to read the evaluator module.')
    ev = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ev
    spec.loader.exec_module(ev)  # Import definitions only; main(), execute(), and load_model() are not called.
    with ev.exclusive_rig(root):
        source = root.parent / 'lerobot/src/lerobot/envs/libero.py'
        ev.require(digest(source) == ev.ENV_SOURCE_SHA, 'LIBERO reset-wrapper source changed.')
        ev.require(ev.git(root.parent/'lerobot', 'rev-parse', 'HEAD') == ev.PIN, 'LeRobot revision changed.')
        ev.require(not ev.git(root.parent/'lerobot', 'status', '--porcelain', '--untracked-files=no'), 'Tracked LeRobot files changed.')
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        try:
            from lerobot.envs.factory import make_env
        except ImportError:
            from lerobot.envs import make_env
        ev.LiberoEnvConfig = LiberoEnvConfig
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        output = root / 'results/metrics/m3_task08_reset_repeatability' / stamp
        output.mkdir(parents=True, exist_ok=False)
        formal = root / 'results/metrics/m3_rva_evaluation/v2'
        before = {p: digest(p) for p in formal.rglob('*') if p.is_file()}
        report = {'purpose': 'RESET_REPEATABILITY_ONLY', 'resets_planned': REPEATS, 'resets': [],
                  'evaluator_sha256': digest(evaluator), 'input_archive_episode': AUDIT_REL,
                  'diagnostic_status': 'STARTED', 'no_training': True, 'formal_episodes_added': 0,
                  'no_image_threshold_change': True, 'policy_models_loaded': 0}
        status = 0
        try:
            reset_checks(ev, make_env, seed_dir, audit, output, report)
        except Exception as exc:
            report['diagnostic_status'] = 'DIAGNOSTIC_ERROR'
            report['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
            print(report['error']['traceback'], flush=True)
            status = 1
        finally:
            after_paths = {p for p in formal.rglob('*') if p.is_file()}
            unchanged = set(before) == after_paths and all(p.is_file() and digest(p) == h for p, h in before.items())
            report['formal_v2_files_unchanged'] = unchanged
            report['evaluator_unchanged'] = digest(evaluator) == EVALUATOR_SHA
            if not unchanged or not report['evaluator_unchanged']:
                report['diagnostic_status'] = 'INPUT_INTEGRITY_ERROR'
                status = 1
            save_json(output / 'reset_repeatability.json', report)
            bundle = output / 'task08_reset_repeatability.zip'
            with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED) as z:
                for path in sorted(output.iterdir()):
                    if path.is_file() and path != bundle:
                        z.write(path, path.name)
            print('\nDIAGNOSTIC STATUS:', report['diagnostic_status'], flush=True)
            print('Formal V2 files unchanged:', unchanged, flush=True)
            print('Upload:', bundle, flush=True)
            print('No thresholds or notebook files changed; do not restart the full evaluation on this result alone.', flush=True)
        return status


if __name__ == '__main__':
    raise SystemExit(main())
