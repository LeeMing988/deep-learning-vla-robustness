"""N10: bounded M3-RVA closed-loop evaluation; no training or changes to N00--N09.

The eight N07 rollout/helper functions below are retained structurally (AST-equivalent).
New evaluation harness: explicit per-episode RNG reset, initial-observation pairing audit,
fresh M2-LE comparator evaluation, atomic per-episode cache, and isolated report outputs.
The four conditions were already declared by N09. All M3 comparisons are exploratory.
"""
from __future__ import annotations
import os
for _key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):
    os.environ[_key]='1'
os.environ.setdefault('GALLIUM_DRIVER','d3d12')
os.environ.setdefault('MESA_D3D12_DEFAULT_ADAPTER_NAME','NVIDIA')
os.environ.setdefault('MUJOCO_GL','glfw')
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib
from importlib import metadata
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Any
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

PIN='7e241bd630a3719a56157a497ce5d08f244784f1'
RELEASE='m3_rva_evaluation_v2'
PLANNED_SEEDS=[42,123,2026]
CONDITION_IDS=['E0_ID','E1_L3','E2_SEEN_PHOTO','E3_HELDOUT_NOISE']
VARIANTS=['M2-LE','M3-RVA']
EPISODES_PER_TASK=5
TASK_IDS=list(range(10))
BOOTSTRAP_SAMPLES=5000
BOOTSTRAP_SEED=20261018
DEVICE='cuda'
LIBERO_SUITE='libero_spatial'
CONDITIONS={
 'E0_ID':{'label':'ID / L0','language_level':'L0','image_shift':'NONE','shift_role':'ID'},
 'E1_L3':{'label':'Language L3 held-out','language_level':'L3','image_shift':'NONE','shift_role':'heldout_language'},
 'E2_SEEN_PHOTO':{'label':'Seen photometric shift','language_level':'L0','image_shift':'SEEN_PHOTO','shift_role':'seen_visual'},
 'E3_HELDOUT_NOISE':{'label':'Held-out sensor noise','language_level':'L0','image_shift':'HELDOUT_NOISE','shift_role':'heldout_visual'},
}
EXPECTED_TRAINER_SHA='b6dae4fe7bb2ebd573dc6b4d7e6b73a9161b7c88c890002a28f7810f4ada3a38'
N07_FUNCTION_HASHES={'apply_seen_photometric_shift': 'b0d701f73fd652628fd82c25e0859708d44a8f076ca13365b3cfda3358d7ae5d', 'apply_heldout_sensor_noise': 'ed1cb8c71f12d309b51ac3de04cbc7f0e5fe16c106504d57da9bc1c32d602c87', 'perturb_observation_images': 'a5ea9dde975dc668a9cf2529d3da04af665a057de6053d1908dcc0102bfb2e6c', 'extract_success': '7355bf98a45714635cce430fcf918a2cfbef009200d92435e6b65b146bd524a1', 'get_condition_instruction': 'c5966aa4baa57bd412758375190adf9f13762e90a0eb5e4754b1bfe7bcc2ab2c', 'build_env_config': '24f0c7983789a4f9cf9db8a9a2a0a94b10c6db3803031c576678538e6bf87b42', 'save_tensor_image': '3e8dc9af407d1cd68084f2c80321305edfd84c19a786378fa7fee15a9eba8e3e', 'evaluate_one_episode': '6e40981f121e65af37724d48db4704b208a9a78c6be657b67aa290dde10251e8'}

class NotReady(RuntimeError):
    """Expected missing/in-progress prerequisite; never a model-performance result."""


def require(value, message):
    if not bool(value): raise RuntimeError(message)


def read_json(path):
    with Path(path).open(encoding='utf-8') as f: return json.load(f)


def jsonable(x):
    if isinstance(x,Path): return str(x)
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [jsonable(v) for v in x]
    return x


def obj_sha(x):
    return hashlib.sha256(json.dumps(jsonable(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(jsonable(payload),f,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def save_csv(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');frame.to_csv(tmp,index=False);tmp.replace(path)


def local_path(root,text):
    p=Path(text).expanduser();p=(root/p if not p.is_absolute() else p).resolve()
    require(p.is_relative_to(root),f'Artifact must be inside the project: {p}')
    return p


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def normalize_seeds(values):
    values=[int(s) for s in values]
    require(values and len(set(values))==len(values),'Provide nonempty, unique seed IDs.')
    require(set(values).issubset(PLANNED_SEEDS),f'Use only planned seeds: {PLANNED_SEEDS}')
    return [s for s in PLANNED_SEEDS if s in values]


def canonical_ast_dump(node):
    """Audited Python 3.12/3.13 representation with empty AST fields retained.

    Python 3.13's default dump omits empty fields; Python 3.12's retains them.
    Output cells, execution counters, Markdown and ordinary comments are not hashed.
    Executable structure and literal values remain checked against audited constants.
    """
    options = {'include_attributes': False}
    if sys.version_info >= (3, 13):
        options['show_empty'] = True
    return ast.dump(node, **options)


def verify_n07_functions(path):
    found={}
    for cell in read_json(path)['cells']:
        if cell['cell_type']!='code':continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.FunctionDef) and node.name in N07_FUNCTION_HASHES:
                require(node.name not in found,f'Duplicate N07 function: {node.name}')
                found[node.name]=hashlib.sha256(canonical_ast_dump(node).encode()).hexdigest()
    require(found==N07_FUNCTION_HASHES,
            'Local N07 rollout helpers differ from the reviewed version. Preserve N09 and review the evaluator change.')


def observation_sha(observation):
    """Hash raw initial observations, including images and state, without changing them."""
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
        if isinstance(x,np.ndarray):
            require(x.dtype.kind!='O','Object arrays cannot be a stable observation signature.')
            arr=np.ascontiguousarray(x)
            h.update(b'array');h.update(str(arr.dtype).encode());h.update(repr(arr.shape).encode());h.update(arr.tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for key in sorted(x):h.update(str(key).encode());visit(x[key])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode());h.update(str(len(x)).encode())
            for value in x:visit(value)
        elif isinstance(x,(str,int,float,bool,np.generic)) or x is None:
            h.update(json.dumps(jsonable(x),sort_keys=True,allow_nan=False).encode())
        else:raise TypeError(f'Unsupported raw-observation type: {type(x).__name__}')
    visit(observation);return h.hexdigest()


def episode_path(seed_dir,variant,condition,task,replica):
    tag='m2le' if variant=='M2-LE' else 'm3rva'
    return seed_dir/'episodes'/f'{tag}__{condition}__task{task:02d}__rep{replica}.json'


def expected_episode_keys(seeds):
    return {(int(s),v,c,t,r) for s in seeds for v in VARIANTS for c in CONDITION_IDS for t in TASK_IDS for r in range(5)}


def strict_bool(value):
    if isinstance(value,(bool,np.bool_)):return bool(value)
    if isinstance(value,(int,np.integer)) and value in (0,1):return bool(value)
    if isinstance(value,str) and value.lower() in ('true','false'):return value.lower()=='true'
    raise ValueError(f'Invalid boolean: {value!r}')


# ---- V2: explicit initial-state selection and inspectable pairing evidence ----
PAIRING_RULE = {
    'version': 'observed_state_exact_sparse_uint8_v2',
    'state_rule': 'All seven exposed robot-state arrays must be byte-identical.',
    'fixed_state_rule': 'Same task, fixed-state index and selected benchmark vector SHA256.',
    'initial_state_index_formula': 'condition_index_in_N09_order * 5 + replica',
    'reset_rule': 'Fresh vector/environment wrapper for each episode; set fixed-state index before reset.',
    'image_rule': 'Both uint8 camera arrays keep exact schema; compare signed differences without altering inputs.',
    'max_channel_difference': 1,
    'max_changed_channel_fraction': 0.0001,
    'fraction_comparison': 'changed_channels * 10000 <= channels_per_camera',
    'calibration': 'Technical amendment after reset diagnostic, not chosen using success scores.',
    'scope': 'Initial observations only; not a claim of full hidden-state or behavioral equivalence.',
}
ENV_SOURCE_SHA = '97c984f12331527626812ec19967ef399e545535b3571becf044db2417ae9d71'
IMAGE_FIELDS = ('root/pixels/image', 'root/pixels/image2')
STATE_SHAPES = {
    'root/robot_state/eef/pos': (1, 3),
    'root/robot_state/eef/quat': (1, 4),
    'root/robot_state/eef/mat': (1, 3, 3),
    'root/robot_state/gripper/qpos': (1, 2),
    'root/robot_state/gripper/qvel': (1, 2),
    'root/robot_state/joints/pos': (1, 7),
    'root/robot_state/joints/vel': (1, 7),
}


def flatten_observation(observation, prefix='root'):
    if isinstance(observation, dict):
        result = {}
        for key in sorted(observation):
            require(isinstance(key, str) and '/' not in key, 'Unexpected observation key.')
            result.update(flatten_observation(observation[key], prefix + '/' + key))
        return result
    if isinstance(observation, torch.Tensor):
        observation = observation.detach().cpu().numpy()
    array = np.asarray(observation)
    require(array.dtype.kind != 'O', f'Object-valued observation: {prefix}')
    return {prefix: np.array(array, copy=True)}


def validate_observation_arrays(arrays):
    require(set(arrays) == set(IMAGE_FIELDS) | set(STATE_SHAPES), 'Observation fields differ from reviewed wrapper.')
    for name in IMAGE_FIELDS:
        array = arrays[name]
        require(array.dtype == np.dtype('uint8') and array.shape == (1, 256, 256, 3),
                f'Unexpected raw camera shape/dtype: {name}')
    for name, shape in STATE_SHAPES.items():
        array = arrays[name]
        require(array.dtype == np.dtype('float64') and array.shape == shape,
                f'Unexpected robot-state shape/dtype: {name}')
        require(np.isfinite(array).all(), f'Non-finite robot state: {name}')


def unflatten_observation(arrays):
    root = {}
    for path, value in arrays.items():
        parts = path.split('/')
        require(parts[0] == 'root' and len(parts) > 1, 'Malformed observation field.')
        node = root
        for key in parts[1:-1]:
            node = node.setdefault(key, {})
        node[parts[-1]] = value
    return root


def compare_initial_arrays(left, right):
    """No rounding, image replacement, relative tolerance, or adaptive thresholds."""
    validate_observation_arrays(left)
    validate_observation_arrays(right)
    bad_state = [name for name in STATE_SHAPES if left[name].tobytes() != right[name].tobytes()]
    camera_reports = []
    for name in IMAGE_FIELDS:
        # Signed subtraction is essential: uint8 subtraction would wrap at 0/255.
        delta = np.abs(left[name].astype(np.int16) - right[name].astype(np.int16))
        changed = int(np.count_nonzero(delta))
        maximum = int(delta.max())
        count = int(delta.size)
        camera_reports.append({
            'field': name, 'changed_channels': changed, 'channels': count,
            'changed_pixels': int(np.count_nonzero(np.any(delta != 0, axis=-1))),
            'max_abs_difference': maximum, 'mean_abs_difference': float(delta.mean()),
            'changed_fraction': changed / count,
            'pass': bool(maximum <= 1 and changed * 10000 <= count),
        })
    return {
        'rule': PAIRING_RULE['version'],
        'observed_robot_state_exact': not bad_state,
        'different_robot_state_fields': bad_state,
        'camera_comparisons': camera_reports,
        'all_images_byte_identical': all(row['changed_channels'] == 0 for row in camera_reports),
        'pass': bool(not bad_state and all(row['pass'] for row in camera_reports)),
    }


def fixed_initial_state_index(condition, replica):
    require(condition in CONDITION_IDS and replica in range(EPISODES_PER_TASK), 'Invalid episode selector.')
    return CONDITION_IDS.index(condition) * EPISODES_PER_TASK + int(replica)


def prepare_fixed_state(env, task, condition, replica):
    """Uses the exact LiberoEnv attributes in the uploaded source; no LeRobot edits."""
    require(hasattr(env, 'envs') and len(env.envs) == 1, 'Expected one synchronous LIBERO sub-environment.')
    leaf = env.envs[0]
    require(type(leaf).__module__ == 'lerobot.envs.libero' and type(leaf).__name__ == 'LiberoEnv',
            'Unexpected LIBERO wrapper; review reset integration.')
    require(leaf.task_id == task and leaf.init_states is True, 'Wrong task or fixed states disabled.')
    require(leaf._reset_stride == 1 and leaf.num_steps_wait == 10 and leaf.hard_reset is True,
            'Unexpected reset stride, settling count, or reset mode.')
    index = fixed_initial_state_index(condition, replica)
    require(leaf._init_states is not None and index < len(leaf._init_states), 'Fixed initial-state index unavailable.')
    selected = leaf._init_states[index]
    if isinstance(selected, torch.Tensor): selected = selected.detach().cpu().numpy()
    vector = np.asarray(selected)
    require(vector.ndim == 1 and vector.size > 0 and np.isfinite(vector).all(), 'Invalid benchmark initial state.')
    leaf.init_state_id = index
    return {'task_id': int(task), 'initial_state_index': index,
            'initial_state_vector_sha256': observation_sha(vector)}


def save_initial_capture(seed_dir, variant, condition, task, replica, raw_obs, state_info):
    arrays = flatten_observation(raw_obs)
    validate_observation_arrays(arrays)
    directory = seed_dir / 'initial_observations'
    directory.mkdir(parents=True, exist_ok=True)
    stem = episode_path(seed_dir, variant, condition, task, replica).stem
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path = directory / f'{stem}__{stamp}.npz'
    require(not path.exists(), 'Initial-capture name collision.')
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('xb') as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    temporary.replace(path)
    record = dict(state_info,
        file=str(path.relative_to(seed_dir)), file_sha256=sha(path),
        initial_observation_sha256=observation_sha(raw_obs),
        nonimage_observation_sha256=observation_sha({k: arrays[k] for k in STATE_SHAPES}),
        variant=variant, condition_id=condition, replica=int(replica),
        evaluation_protocol_sha256=EVALUATION_PROTOCOL_SHA256)
    write_json(path.with_suffix('.json'), record)
    return record


def read_initial_capture(seed_dir, record):
    path = local_path(seed_dir, record['file'])
    require(path.is_file() and sha(path) == record['file_sha256'], 'Initial-observation capture missing or changed.')
    require(read_json(path.with_suffix('.json')) == record, 'Initial-capture metadata changed.')
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), 'Duplicate initial-capture arrays.')
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    validate_observation_arrays(arrays)
    require(observation_sha(unflatten_observation(arrays)) == record['initial_observation_sha256'],
            'Initial-capture observation signature differs.')
    require(observation_sha({k: arrays[k] for k in STATE_SHAPES}) == record['nonimage_observation_sha256'],
            'Initial-capture robot-state signature differs.')
    return arrays


def audit_initial_pair(seed_dir, baseline, modified):
    identity_fields = ('task_id', 'condition_id', 'replica', 'initial_state_index',
                       'initial_state_vector_sha256', 'evaluation_protocol_sha256')
    require(all(baseline[k] == modified[k] for k in identity_fields), 'Initial-state selection/pair identity differs.')
    require(baseline['variant'] == 'M2-LE' and modified['variant'] == 'M3-RVA', 'Incorrect comparison order.')
    result = compare_initial_arrays(read_initial_capture(seed_dir, baseline), read_initial_capture(seed_dir, modified))
    result.update({k: baseline[k] for k in identity_fields})
    result.update(baseline_capture=baseline['file'], modified_capture=modified['file'],
        baseline_raw_sha256=baseline['initial_observation_sha256'],
        modified_raw_sha256=modified['initial_observation_sha256'])
    destination = seed_dir / 'pairing_audits' / (
        f"{baseline['condition_id']}__task{baseline['task_id']:02d}__rep{baseline['replica']}.json")
    write_json(destination, result)  # Retain measurable failure evidence, not just a hash.
    require(result['pass'], f'Initial pairing outside the fixed V2 rule. Inspect {destination}; do not widen thresholds.')
    return result


class InitialStateAudit:
    """Fixed benchmark state, raw capture, exact robot state and narrow image audit."""
    def __init__(self, env, *, seed_dir, variant, condition, task, replica, counterpart=None):
        self.env = env; self.seed_dir = seed_dir; self.variant = variant
        self.condition = condition; self.task = task; self.replica = replica
        self.counterpart = counterpart; self.initial_hash = None; self.policy_seed = None
        self.capture = None; self.pair_audit = None

    def reset(self, *, seed):
        require(seed == 100000 + self.task * 100 + self.replica, 'Unexpected episode seed.')
        state_info = prepare_fixed_state(self.env, self.task, self.condition, self.replica)
        # Seed before deferred environment creation as well as the wrapper's seeded reset.
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        obs, info = self.env.reset(seed=seed)
        require(self.env.envs[0].init_state_id == state_info['initial_state_index'] + 1,
                'The wrapper did not use/advance the explicit benchmark state as expected.')
        self.capture = save_initial_capture(self.seed_dir, self.variant, self.condition,
                                           self.task, self.replica, obs, state_info)
        self.initial_hash = self.capture['initial_observation_sha256']
        if self.counterpart is not None:
            baseline = self.capture if self.variant == 'M2-LE' else self.counterpart
            modified = self.capture if self.variant == 'M3-RVA' else self.counterpart
            self.pair_audit = audit_initial_pair(self.seed_dir, baseline, modified)
        self.policy_seed = 700000 + int(seed)
        random.seed(self.policy_seed); np.random.seed(self.policy_seed); torch.manual_seed(self.policy_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.policy_seed)
        # The measured raw observation is passed through unchanged; it is never rounded/replaced.
        return obs, info

    def call(self, *args, **kwargs): return self.env.call(*args, **kwargs)
    def step(self, *args, **kwargs): return self.env.step(*args, **kwargs)


@contextmanager
def exclusive_rig(root):
    """Shares N09's live job lock. It never stops a training worker or deletes its lock."""
    import fcntl
    path=root/'results/metrics/m3_rva_extension/v1/rig_job.lock'
    if not path.parent.exists():raise NotReady('N09 has not created its training workspace yet.')
    with path.open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise NotReady('N09 or another M3 job is running. Leave it running; evaluate afterwards.')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


# ---- Original N07 functions (formatting compacted; AST-equivalent) ----
def apply_seen_photometric_shift(image: torch.Tensor):
    x = image.to(torch.float32)
    x = x * 0.85
    mean = x.mean(dim=(-2, -1), keepdim=True)
    x = (x - mean) * 1.15 + mean
    if x.shape[-3] >= 3:
        rgb = x[..., :3, :, :]
        gray = 0.2989 * rgb[..., 0:1, :, :] + 0.587 * rgb[..., 1:2, :, :] + 0.114 * rgb[..., 2:3, :, :]
        rgb = gray + (rgb - gray) * 0.85
        x = x.clone()
        x[..., :3, :, :] = rgb
    return x.clamp(0.0, 1.0)

def apply_heldout_sensor_noise(image: torch.Tensor, seed: int):
    x = image.to(torch.float32)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * 0.05
    return (x + noise).clamp(0.0, 1.0)

def perturb_observation_images(observation: dict, image_shift: str, episode_seed: int, step_index: int):
    output = dict(observation)
    camera_keys = sorted([key for key in output if key.startswith('observation.images.')])
    for camera_index, camera_key in enumerate(camera_keys):
        image = output[camera_key]
        if not isinstance(image, torch.Tensor):
            continue
        if image_shift == 'NONE':
            shifted = image
        elif image_shift == 'SEEN_PHOTO':
            shifted = apply_seen_photometric_shift(image)
        elif image_shift == 'HELDOUT_NOISE':
            deterministic_noise_seed = int(episode_seed) * 10000 + int(step_index) * 100 + camera_index
            shifted = apply_heldout_sensor_noise(image, seed=deterministic_noise_seed)
        else:
            raise ValueError(f'Unknown image_shift: {image_shift}')
        output[camera_key] = shifted
    return output

def extract_success(info, reward_value):
    success = bool(reward_value > 0)
    if isinstance(info, dict):
        for key in ['is_success', 'success']:
            if key in info:
                try:
                    success = success or bool(np.asarray(info[key]).reshape(-1)[0])
                except Exception:
                    pass
    return success

def get_condition_instruction(suite_task_id: int, language_level: str, original_instruction: str):
    if language_level == 'L0':
        return original_instruction
    dataset_task_id = suite_to_dataset_task[int(suite_task_id)]
    return language_lookup[int(dataset_task_id), str(language_level)]

def build_env_config(suite_task_id: int):
    return LiberoEnvConfig(task=LIBERO_SUITE, task_ids=[int(suite_task_id)], obs_type='pixels_agent_pos', camera_name='agentview_image,robot0_eye_in_hand_image', init_states=True, observation_height=256, observation_width=256, control_mode='relative')

def save_tensor_image(image_tensor, path: Path):
    x = image_tensor.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3, 4):
        x = x.permute(1, 2, 0)
    array = x.numpy()
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(array, 0.0, 1.0))

def evaluate_one_episode(*, env, policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor, run_id: str, variant: str, model_seed: int, checkpoint_signature: str, condition_id: str, suite_task_id: int, replica: int, capture_qualitative: bool):
    condition = CONDITIONS[condition_id]
    episode_seed = 100000 + int(suite_task_id) * 100 + int(replica)
    policy.reset()
    raw_obs, _ = env.reset(seed=episode_seed)
    try:
        original_instruction = str(list(env.call('task_description'))[0])
    except Exception:
        original_instruction = str(env.call('task')[0])
    evaluation_instruction = get_condition_instruction(suite_task_id=suite_task_id, language_level=condition['language_level'], original_instruction=original_instruction)
    max_steps = int(env.call('_max_episode_steps')[0])
    success = False
    terminated_value = False
    truncated_value = False
    inference_times = []
    simulator_step_times = []
    first_image = None
    final_image = None
    rollout_start = time.perf_counter()
    for step_index in range(max_steps):
        obs = preprocess_observation(raw_obs)
        obs = perturb_observation_images(obs, image_shift=condition['image_shift'], episode_seed=episode_seed, step_index=step_index)
        obs['task'] = [evaluation_instruction]
        if first_image is None:
            camera_keys = sorted([key for key in obs if key.startswith('observation.images.')])
            if camera_keys:
                first_image = obs[camera_keys[0]].detach().cpu().clone()
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_start = time.perf_counter()
        with torch.inference_mode():
            action_normalized = policy.select_action(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_times.append(time.perf_counter() - inference_start)
        action = postprocessor(action_normalized)
        action_transition = env_postprocessor({ACTION: action})
        action_for_env = action_transition[ACTION]
        if isinstance(action_for_env, torch.Tensor):
            action_for_env = action_for_env.detach().cpu().numpy()
        action_for_env = np.asarray(action_for_env, dtype=np.float32)
        if not np.isfinite(action_for_env).all():
            raise RuntimeError(f'Non-finite action: {run_id} {condition_id} task={suite_task_id} replica={replica}')
        simulator_start = time.perf_counter()
        raw_obs, reward, terminated, truncated, info = env.step(action_for_env)
        simulator_step_times.append(time.perf_counter() - simulator_start)
        reward_value = float(np.asarray(reward).reshape(-1)[0])
        terminated_value = bool(np.asarray(terminated).reshape(-1)[0])
        truncated_value = bool(np.asarray(truncated).reshape(-1)[0])
        success = success or extract_success(info, reward_value)
        if success or terminated_value or truncated_value:
            break
    rollout_seconds = time.perf_counter() - rollout_start
    steps_executed = step_index + 1
    try:
        final_obs = preprocess_observation(raw_obs)
        final_obs = perturb_observation_images(final_obs, image_shift=condition['image_shift'], episode_seed=episode_seed, step_index=steps_executed)
        final_camera_keys = sorted([key for key in final_obs if key.startswith('observation.images.')])
        if final_camera_keys:
            final_image = final_obs[final_camera_keys[0]].detach().cpu().clone()
    except Exception:
        final_image = None
    if success:
        failure_proxy = 'SUCCESS'
    elif truncated_value or steps_executed >= max_steps:
        failure_proxy = 'TIMEOUT'
    elif terminated_value:
        failure_proxy = 'TERMINATED_NO_SUCCESS'
    else:
        failure_proxy = 'OTHER_NO_SUCCESS'
    if capture_qualitative and first_image is not None:
        stem = f'{run_id}__{condition_id}__task{suite_task_id}__rep{replica}'
        save_tensor_image(first_image, QUALITATIVE_DIR / f'{stem}__initial.png')
        if final_image is not None:
            save_tensor_image(final_image, QUALITATIVE_DIR / f'{stem}__final.png')
    return {'system_status': 'PASS', 'evaluation_protocol_sha256': EVALUATION_PROTOCOL_SHA256, 'checkpoint_signature': str(checkpoint_signature), 'run_id': str(run_id), 'variant': str(variant), 'model_seed': int(model_seed), 'condition_id': str(condition_id), 'condition_label': condition['label'], 'shift_role': condition['shift_role'], 'language_level': condition['language_level'], 'image_shift': condition['image_shift'], 'suite_task_id': int(suite_task_id), 'dataset_task_index': int(suite_to_dataset_task[suite_task_id]), 'replica': int(replica), 'episode_seed': int(episode_seed), 'success': bool(success), 'steps_executed': int(steps_executed), 'max_steps': int(max_steps), 'terminated': bool(terminated_value), 'truncated': bool(truncated_value), 'failure_proxy': failure_proxy, 'wall_seconds': float(rollout_seconds), 'mean_select_action_seconds': float(np.mean(inference_times)), 'median_select_action_seconds': float(np.median(inference_times)), 'mean_simulator_step_seconds': float(np.mean(simulator_step_times)), 'instruction': evaluation_instruction}

# ---- New bounded evaluation harness ----

class Evaluation:
    def __init__(self,root,seeds):
        self.root=Path(root).resolve();self.seeds=normalize_seeds(seeds)
        self.output=self.root/'results/metrics/m3_rva_evaluation/v2'
        self.figures=self.root/'results/figures/m3_rva_evaluation/v2'
        self.train_root=self.root/'results/metrics/m3_rva_extension/v1'
        self.protocol_path=self.root/'configs/experiments/m3_rva_training_protocol_v1.json'
        self.trainer_path=self.root/'src/m3_rva_training.py'
        self.mapping_path=self.root/'results/metrics/dataset_eda/libero_spatial_dataset_task_mapping.csv'
        self.language_path=self.root/'configs/experiments/libero_spatial_language_augmentations_v1.csv'
        self.n07_path=self.root/'notebooks/07_definitive_robustness_evaluation.ipynb'
        self.n07_protocol_path=self.root/'configs/experiments/definitive_evaluation_protocol_v1.json'
        self.registry_path=self.root/'configs/experiments/definitive_final_model_registry_v1.csv'
        self.definition_path=self.root/'configs/experiments/m3_rva_evaluation_protocol_v2.json'
        self.expected_count=len(expected_episode_keys(self.seeds))

    def prerequisites(self):
        required=[self.protocol_path,self.trainer_path,self.mapping_path,self.language_path,
                  self.n07_path,self.n07_protocol_path,self.registry_path,
                  self.train_root/'training_summary.json',self.train_root/'qualification.json']
        missing=[str(p.relative_to(self.root)) for p in required if not p.is_file()]
        if missing:raise NotReady('Waiting for prerequisites: '+', '.join(missing))
        self.training=read_json(self.protocol_path)
        require(self.training['source_pin']==PIN and int(self.training['steps'])==25000,'Unexpected M3 training pin/budget.')
        require(self.training['planned_evaluation_conditions']==CONDITION_IDS,'N09 evaluation conditions differ; do not silently change the experiment.')
        require(self.training['planned_seeds']==PLANNED_SEEDS,'N09 planned seed matrix differs.')
        self.training_hash=obj_sha(self.training)
        require(sha(self.trainer_path)==EXPECTED_TRAINER_SHA==self.training['runner_sha256'],
                'N09 runner differs from this evaluation release. Finish/review the training fix before adapting N10.')
        summary=read_json(self.train_root/'training_summary.json')
        require(summary['protocol_hash']==self.training_hash,'N09 summary/protocol disagree.')
        unfinished=[s for s in self.seeds if s not in summary.get('completed_seeds',[])]
        if unfinished:raise NotReady(f'N09 has not completed training and reload verification for selected seeds: {unfinished}')
        q=read_json(self.train_root/'qualification.json')
        require(q.get('status')=='PASS' and q.get('protocol_hash')==self.training_hash,'Real-data qualification has not passed for this protocol.')
        verify_n07_functions(self.n07_path)
        env_source = self.root.parent/'lerobot/src/lerobot/envs/libero.py'
        require(env_source.is_file() and sha(env_source) == ENV_SOURCE_SHA,
                'LIBERO reset wrapper differs from the supplied diagnostic. Do not bypass this check.')
        n07=read_json(self.n07_protocol_path)
        require(all(n07['conditions'].get(k)==CONDITIONS[k] for k in CONDITION_IDS),'N07 condition definitions changed.')
        require(n07['episode_seed_formula']=='100000 + suite_task_id*100 + replica','N07 episode seed formula changed.')
        require(n07['heldout_sensor_noise_sigma']==0.05 and n07['seen_photometric_shift']==
                {'brightness':0.85,'contrast':1.15,'saturation':0.85},'N07 corruption settings changed.')
        mapping=pd.read_csv(self.mapping_path);language=pd.read_csv(self.language_path)
        require({'suite_task_id','dataset_task_index','benchmark_instruction'}.issubset(mapping.columns),'Task mapping schema differs.')
        require(len(mapping)==10 and set(mapping.suite_task_id.astype(int))==set(TASK_IDS),'Expected ten LIBERO-Spatial tasks.')
        require(mapping.dataset_task_index.nunique()==10,'Dataset task map is not one-to-one.')
        require({'task_id','level','augmented_instruction'}.issubset(language.columns),'Language schema differs.')
        require(len(language)==40 and not language[['task_id','level']].duplicated().any(),'Expected forty unique language rows.')
        require(language.augmented_instruction.notna().all(),'Missing reviewed language text.')
        if 'semantic_review' in language:require((language.semantic_review=='PASS').all(),'Unreviewed language row.')
        global suite_to_dataset_task,language_lookup,QUALITATIVE_DIR
        suite_to_dataset_task={int(r.suite_task_id):int(r.dataset_task_index) for r in mapping.itertuples()}
        language_lookup={(int(r.task_id),str(r.level)):str(r.augmented_instruction) for r in language.itertuples()}
        require(all((d,l) in language_lookup for d in suite_to_dataset_task.values() for l in ('L0','L3')),'Missing task language pairing.')
        QUALITATIVE_DIR=self.root/'results/qualitative/m3_rva_evaluation/v2'
        self.small_inputs={str(p.relative_to(self.root)):sha(p) for p in
                           [self.protocol_path,self.trainer_path,self.mapping_path,self.language_path,self.n07_protocol_path,self.registry_path]}
        require(sha(self.language_path)==self.training['artifact_hashes']['language'],'Evaluation language differs from N09 frozen language.')
        self.registry=pd.read_csv(self.registry_path)
        self.definition={
            'release':RELEASE,'scientific_role':'EXPLORATORY_POST_HOC_ARCHITECTURAL_COMPARISON',
            'training_protocol_hash':self.training_hash,'planned_training_seeds':PLANNED_SEEDS,
            'variants':VARIANTS,'conditions':CONDITIONS,'suite':LIBERO_SUITE,
            'task_ids':TASK_IDS,'replica_ids':list(range(5)),
            'environment':{'obs_type':'pixels_agent_pos','camera_name':'agentview_image,robot0_eye_in_hand_image',
                'init_states':True,'observation_height':256,'observation_width':256,'control_mode':'relative',
                'n_envs':1,'use_async_envs':False},
            'horizon':'unchanged environment _max_episode_steps; no horizon tuning',
            'episode_seed_formula':'100000 + suite_task_id*100 + replica',
            'policy_rng_seed_formula':'700000 + episode_seed; reset after env.reset for both models',
            'pairing_audit':PAIRING_RULE,
            'environment_wrapper_sha256':ENV_SOURCE_SHA,
            'protocol_amendment':'V2: explicit indexed fresh resets; exact exposed robot state and sparse one-level image tolerance.',
            'amendment_basis':'Task 2 replica 3 reset diagnostic: 12/196608 channels changed by one; same-model baseline replay also had a different historical hash.',
            'legacy_evaluation_policy':'Preserve V1 files unmodified, but do not import or pool V1 outcomes; captures were not stored.',
            'scope_of_equivalence':'Not bitwise equivalence of all pixels or every hidden simulator variable.',
            'noise_sigma':0.05,'photometric':{'brightness':0.85,'contrast':1.15,'saturation':0.85},
            'checkpoint_selection':'fixed completed step 25000; no intermediate or best-result selection',
            'control_evaluation':'fresh M2-LE rollouts in the same harness; original N07/N07b results not pooled or overwritten',
            'no_additional_training':True,'no_consistency_training':True,'no_new_E6':True,
            'bootstrap':{'samples':BOOTSTRAP_SAMPLES,'seed':BOOTSTRAP_SEED,'method':'paired resampling of seed, task and replicas; single-seed intervals condition on checkpoint pair'},
            'tests':'nominal exact episode-level McNemar + Holm across these four exploratory comparisons; clustering caveat applies',
            'success_is_not_an_engineering_gate':True,'source_inputs':self.small_inputs,
            'n07_function_hashes':N07_FUNCTION_HASHES,'evaluator_sha256':sha(Path(__file__)),
        }
        self.evaluation_hash=obj_sha(self.definition)
        return summary

    def freeze(self):
        global EVALUATION_PROTOCOL_SHA256
        EVALUATION_PROTOCOL_SHA256=self.evaluation_hash
        if self.definition_path.exists():
            require(read_json(self.definition_path)==self.definition,'Existing N10 protocol differs. Do not overwrite an evaluation.')
        else:write_json(self.definition_path,self.definition)
        self.output.mkdir(parents=True,exist_ok=True)
        self.figures.mkdir(parents=True,exist_ok=True)

    def seed_inputs(self,seed):
        run=self.root/f'experiments/m3_rva_extension/v1/m3rva_seed{seed}_25000'
        result_path=run/'result.json'
        if not result_path.exists():raise NotReady(f'Missing completed N09 result for seed {seed}.')
        result=read_json(result_path)
        require(result['status']=='PASS' and result['seed']==seed and result['steps']==25000
                and result['protocol_hash']==self.training_hash,'Invalid N09 completed result.')
        require(np.isfinite(float(result['validation_loss'])),'N09 validation loss is not finite.')
        require(result['restoration']['status']=='PASS' and result['restoration']['fresh_process'] is True,
                'N09 fresh-process model/processor reconstruction not verified.')
        m3=local_path(self.root,result['checkpoint'])
        manifest=read_json(m3/'m3_manifest.json')
        require(manifest['seed']==seed and manifest['step']==25000 and manifest['protocol_hash']==self.training_hash,
                'Final M3 export does not match result.json.')
        require(manifest['runner_sha256']==EXPECTED_TRAINER_SHA and manifest['loader']=='src.m3_rva_training.load_m3_bundle',
                'Explicit M3 loader contract differs.')
        for name,digest in manifest['files'].items():
            p=local_path(m3,name);require(p.is_file() and sha(p)==digest,f'M3 bundle file differs: {name}')
        old=self.registry[(self.registry.variant=='M2-LE') & (self.registry.seed.astype(int)==seed)]
        require(len(old)==1,f'Expected one frozen M2-LE checkpoint for seed {seed}.')
        baseline=local_path(self.root,str(old.iloc[0]['checkpoint']))
        require(baseline==local_path(self.root,self.training['comparator_checkpoints'][str(seed)]),'M2-LE comparator path differs from N09.')
        require(int(old.iloc[0]['training_steps'])==25000,'M2-LE comparator has a different training budget.')
        paths={'M2-LE':baseline,'M3-RVA':m3}
        inventory={}
        for variant,path in paths.items():
            require((path/'config.json').is_file() and any(path.glob('*.safetensors')),f'Missing final checkpoint files: {path}')
            inventory[variant]={str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
        signatures={v:obj_sha(inv) for v,inv in inventory.items()}
        seed_dir=self.output/f'seed{seed}'
        record={'evaluation_protocol_sha256':self.evaluation_hash,'seed':seed,
                'checkpoint_paths':{v:str(p.relative_to(self.root)) for v,p in paths.items()},
                'checkpoint_file_hashes':inventory,'checkpoint_signatures':signatures,
                'm3_result_sha256':sha(result_path),'steps':25000}
        input_file=seed_dir/'inputs.json'
        if input_file.exists():require(read_json(input_file)==record,'Seed evaluation inputs changed. Refuse to mix old and new episodes.')
        else:write_json(input_file,record)
        return seed_dir,paths,record

    def check_original_inputs(self):
        for name,digest in self.small_inputs.items():require(sha(self.root/name)==digest,f'Read-only original/protocol file changed: {name}')
        require(git(self.root.parent/'lerobot','rev-parse','HEAD')==PIN,'LeRobot revision changed during evaluation.')
        require(not git(self.root.parent/'lerobot','status','--porcelain','--untracked-files=no'),'LeRobot tracked source changed during evaluation.')


def load_components(root):
    # No models are loaded until the N09 job lock and completed export checks pass.
    global LiberoEnvConfig,make_env,make_env_pre_post_processors,preprocess_observation,ACTION
    global SmolVLAPolicy,make_pre_post_processors,load_m3_bundle
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    try:
        from lerobot.envs.factory import make_env,make_env_pre_post_processors
    except ImportError:
        from lerobot.envs import make_env,make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION
    sys.path.insert(0,str(root))
    loader=importlib.import_module('src.m3_rva_training')
    require(Path(loader.__file__).resolve()==(root/'src/m3_rva_training.py').resolve(),'Wrong M3 runner import.')
    require(sha(Path(loader.__file__))==EXPECTED_TRAINER_SHA,'M3 runner changed on import.')
    load_m3_bundle=loader.load_m3_bundle


def load_model(variant,path):
    gc.collect();torch.cuda.empty_cache()
    if variant=='M3-RVA':
        policy,pre,post=load_m3_bundle(path,device=DEVICE)
        require(hasattr(policy.model,'m3_visual_adapter'),'M3 adapter missing after load.')
        require(hasattr(policy.model,'_m3_visual_hook'),'M3 connector integration missing.')
    else:
        policy=SmolVLAPolicy.from_pretrained(str(path),local_files_only=True).to(DEVICE)
        policy.eval()
        pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=str(path),
                    preprocessor_overrides={'device_processor':{'device':DEVICE}})
        require(not hasattr(policy.model,'m3_visual_adapter'),'Unexpected adapter in original M2-LE comparator.')
    policy.eval();return policy,pre,post


def check_saved_row(row,inputs,variant,condition,task,replica):
    require(row.get('system_status')=='PASS','Cached system error is not a completed episode.')
    require(row.get('evaluation_protocol_sha256')==inputs['evaluation_protocol_sha256'],'Episode protocol mismatch.')
    require(row.get('checkpoint_signature')==inputs['checkpoint_signatures'][variant],'Episode checkpoint mismatch.')
    require((row['model_seed'],row['variant'],row['condition_id'],row['suite_task_id'],row['replica'])
            ==(inputs['seed'],variant,condition,task,replica),'Episode identity mismatch.')
    require(row['episode_seed']==100000+task*100+replica and row['policy_rng_seed']==800000+task*100+replica,
            'Episode random seed mismatch.')
    require(len(row['initial_observation_sha256'])==64,'Missing initial-observation signature.')
    require(row.get('pairing_rule') == PAIRING_RULE['version'], 'Not a V2 pairing-audited episode.')
    require(row.get('initial_state_index') == fixed_initial_state_index(condition, replica), 'Wrong fixed-state index.')
    capture = row.get('initial_observation_record', {})
    require(capture.get('initial_observation_sha256') == row['initial_observation_sha256'], 'Initial capture/episode differ.')
    require(capture.get('nonimage_observation_sha256') == row.get('nonimage_observation_sha256'), 'Robot-state signature differs.')
    require(capture.get('initial_state_vector_sha256') == row.get('initial_state_vector_sha256'), 'Initial-vector signature differs.')
    require(capture.get('evaluation_protocol_sha256') == inputs['evaluation_protocol_sha256'], 'Capture protocol differs.')
    require((capture.get('task_id'), capture.get('condition_id'), capture.get('replica'), capture.get('variant'))
            == (task, condition, replica, variant), 'Capture identity differs.')
    strict_bool(row['success'])
    require(0<int(row['steps_executed'])<=int(row['max_steps']),'Invalid rollout length.')
    require(np.isfinite(float(row['wall_seconds'])) and row['wall_seconds']>=0,'Invalid episode runtime.')


def evaluate_seed(ctx, seed, smoke=False):
    seed_dir, paths, inputs = ctx.seed_inputs(seed)
    # Exercise the previously failing reset FIRST; both outcomes count in the matrix.
    tasks = [2] if smoke else TASK_IDS
    conditions = ['E0_ID'] if smoke else CONDITION_IDS
    replicas = [3] if smoke else list(range(5))
    expected_per_model = len(tasks) * len(conditions) * len(replicas)
    model_records = []
    for variant in VARIANTS:
        missing = [(c, t, r) for t in tasks for c in conditions for r in replicas
                   if not episode_path(seed_dir, variant, c, t, r).exists()]
        if not missing:
            print(f'{variant} seed {seed}: validating/reusing {expected_per_model} selected V2 episodes.', flush=True)
            for task in tasks:
                for condition in conditions:
                    for replica in replicas:
                        saved = read_json(episode_path(seed_dir, variant, condition, task, replica))
                        check_saved_row(saved, inputs, variant, condition, task, replica)
                        read_initial_capture(seed_dir, saved['initial_observation_record'])
            continue
        policy, pre, post = load_model(variant, paths[variant])
        cfg = policy.config
        current = {'input_features': {k: {'type': str(v.type), 'shape': list(v.shape)} for k, v in cfg.input_features.items()},
                   'action_shape': list(cfg.action_feature.shape), 'n_action_steps': int(cfg.n_action_steps),
                   'chunk_size': int(cfg.chunk_size), 'num_steps': int(cfg.num_steps), 'use_cache': bool(cfg.use_cache),
                   'resize_imgs_with_padding': list(cfg.resize_imgs_with_padding) if cfg.resize_imgs_with_padding else None,
                   'adapt_to_pi_aloha': bool(cfg.adapt_to_pi_aloha), 'tokenizer_max_length': int(cfg.tokenizer_max_length)}
        cfg_file = seed_dir / 'inference_interface.json'
        if cfg_file.exists(): require(read_json(cfg_file) == current, 'Model inference interfaces differ.')
        else: write_json(cfg_file, current)
        require(not current['adapt_to_pi_aloha'], 'Unexpected Aloha conversion in LIBERO evaluation.')
        model_records.append({'variant': variant, 'seed': seed, 'total_parameters': sum(p.numel() for p in policy.parameters()),
                              'trainable_flags_count': sum(p.numel() for p in policy.parameters() if p.requires_grad)})
        try:
            for condition, task, replica in missing:
                destination = episode_path(seed_dir, variant, condition, task, replica)
                counterpart_variant = 'M2-LE' if variant == 'M3-RVA' else 'M3-RVA'
                counterpart_path = episode_path(seed_dir, counterpart_variant, condition, task, replica)
                counterpart = None
                if counterpart_path.exists():
                    other = read_json(counterpart_path)
                    check_saved_row(other, inputs, counterpart_variant, condition, task, replica)
                    counterpart = other['initial_observation_record']
                # Fresh environment per episode removes previous-policy/reset-history dependence.
                episode_seed = 100000 + task * 100 + replica
                random.seed(episode_seed); np.random.seed(episode_seed); torch.manual_seed(episode_seed)
                env_cfg = build_env_config(task)
                env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
                envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
                env = envs[LIBERO_SUITE][task]
                try:
                    proxy = InitialStateAudit(env, seed_dir=seed_dir, variant=variant, condition=condition,
                                              task=task, replica=replica, counterpart=counterpart)
                    row = evaluate_one_episode(env=proxy, policy=policy, preprocessor=pre, postprocessor=post,
                        env_preprocessor=env_pre, env_postprocessor=env_post,
                        run_id=f'{variant.replace("-", "")}seed{seed}', variant=variant, model_seed=seed,
                        checkpoint_signature=inputs['checkpoint_signatures'][variant], condition_id=condition,
                        suite_task_id=task, replica=replica,
                        capture_qualitative=(seed == 42 and task == 0 and replica == 0))
                    row.update(initial_observation_sha256=proxy.initial_hash, policy_rng_seed=proxy.policy_seed,
                               initial_observation_record=proxy.capture,
                               initial_state_index=proxy.capture['initial_state_index'],
                               initial_state_vector_sha256=proxy.capture['initial_state_vector_sha256'],
                               nonimage_observation_sha256=proxy.capture['nonimage_observation_sha256'],
                               pairing_rule=PAIRING_RULE['version'], evidence_role='EXPLORATORY',
                               control_source='FRESH_N10_V2_PAIRED_EVALUATION',
                               recorded_at_utc=datetime.now(timezone.utc).isoformat())
                    check_saved_row(row, inputs, variant, condition, task, replica)
                    write_json(destination, row)
                    print(f'{variant} seed={seed} | {condition} | task={task} rep={replica} | '
                          f'{"SUCCESS" if row["success"] else row["failure_proxy"]} | '
                          f'steps={row["steps_executed"]} | {row["wall_seconds"]:.1f}s', flush=True)
                    if proxy.pair_audit is not None and not proxy.pair_audit['all_images_byte_identical']:
                        print('PAIR AUDIT: exact exposed robot state; sparse <=1-level image differences logged.', flush=True)
                finally:
                    try: env.close()
                    finally: del env, envs, env_pre, env_post; gc.collect()
        finally:
            del policy, pre, post; gc.collect(); torch.cuda.empty_cache()
    for variant, path in paths.items():
        after = {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
        require(after == inputs['checkpoint_file_hashes'][variant], f'{variant} checkpoint bytes changed.')
    ctx.check_original_inputs()
    if model_records:
        params_path = seed_dir / 'model_parameters.csv'
        combined = pd.concat([pd.read_csv(params_path), pd.DataFrame(model_records)], ignore_index=True) if params_path.exists() else pd.DataFrame(model_records)
        save_csv(combined.drop_duplicates(['variant', 'seed'], keep='last'), params_path)
    name = 'smoke_pair.json' if smoke else 'completion.json'
    write_json(seed_dir / name, {'status': 'PASS', 'evaluation_protocol_sha256': ctx.evaluation_hash,
        'seed': seed, 'episodes': 2 * expected_per_model, 'checkpoints_unchanged': True,
        'pairing_rule': PAIRING_RULE['version'], 'success_threshold_required': False})


def validate_all_initial_pairs(ctx, df):
    records = []
    for seed in ctx.seeds:
        seed_dir = ctx.output / f'seed{seed}'
        selected = df[df.model_seed == seed]
        for condition in CONDITION_IDS:
            for task in TASK_IDS:
                for replica in range(5):
                    pair = selected[(selected.condition_id == condition) & (selected.suite_task_id == task) & (selected.replica == replica)]
                    require(len(pair) == 2 and set(pair.variant) == set(VARIANTS), 'Missing initial-observation pair.')
                    baseline = pair[pair.variant == 'M2-LE'].iloc[0]['initial_observation_record']
                    modified = pair[pair.variant == 'M3-RVA'].iloc[0]['initial_observation_record']
                    report = audit_initial_pair(seed_dir, baseline, modified)
                    records.append({'model_seed': seed, 'condition_id': condition, 'suite_task_id': task, 'replica': replica,
                        'initial_state_index': report['initial_state_index'], 'observed_robot_state_exact': report['observed_robot_state_exact'],
                        'all_images_byte_identical': report['all_images_byte_identical'], 'pairing_pass': report['pass'],
                        'image_changed_channels': report['camera_comparisons'][0]['changed_channels'],
                        'image2_changed_channels': report['camera_comparisons'][1]['changed_channels'],
                        'maximum_channel_difference': max(c['max_abs_difference'] for c in report['camera_comparisons'])})
    result = pd.DataFrame(records)
    save_csv(result, ctx.output / 'initial_pairing_audit.csv')
    return result


def exact_mcnemar(b,c):
    """Nominal episode-level test; does not itself model training/task clustering."""
    n=int(b)+int(c)
    if n==0:return 1.0
    terms=[math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2) for k in range(min(int(b),int(c))+1)]
    m=max(terms)
    return min(1.0,2.0*math.exp(m)*sum(math.exp(x-m) for x in terms))


def holm_adjust(values):
    values=np.asarray(values,dtype=float);order=np.argsort(values,kind='stable');out=np.empty(len(values));prior=0.0
    for rank,index in enumerate(order):
        prior=max(prior,(len(values)-rank)*float(values[index]));out[index]=min(1.0,prior)
    return out


def paired_interval(delta,seed,draws=BOOTSTRAP_SAMPLES):
    """delta shape = training seed x task x replica, each observation is paired."""
    delta=np.asarray(delta,dtype=float)
    require(delta.ndim==3 and all(n>0 for n in delta.shape),'Expected nonempty seed/task/replica array.')
    rng=np.random.default_rng(seed);ns,nt,nr=delta.shape;values=np.empty(draws)
    for b in range(draws):
        si=rng.integers(ns,size=ns);ti=rng.integers(nt,size=nt)
        sample=delta[si[:,None],ti[None,:],:]
        ri=rng.integers(nr,size=(ns,nt,nr))
        values[b]=np.take_along_axis(sample,ri,axis=2).mean()
    return [float(v) for v in np.quantile(values,[.025,.975])]


def collect_results(ctx):
    rows=[]
    for seed in ctx.seeds:
        seed_dir=ctx.output/f'seed{seed}';input_path=seed_dir/'inputs.json'
        if not input_path.exists():continue
        inputs=read_json(input_path)
        require(inputs['evaluation_protocol_sha256']==ctx.evaluation_hash,'Cached seed belongs to another evaluation protocol.')
        for variant in VARIANTS:
            for condition in CONDITION_IDS:
                for task in TASK_IDS:
                    for replica in range(5):
                        p=episode_path(seed_dir,variant,condition,task,replica)
                        if p.exists():
                            row=read_json(p);check_saved_row(row,inputs,variant,condition,task,replica);rows.append(row)
    df=pd.DataFrame(rows)
    if not df.empty:
        df['success']=df['success'].map(strict_bool)
        require(not df[['model_seed','variant','condition_id','suite_task_id','replica']].duplicated().any(),'Duplicate result keys.')
    return df


def pair_frame(df):
    keys=['model_seed','condition_id','suite_task_id','replica']
    fields=keys+['success','initial_observation_sha256','episode_seed','policy_rng_seed','max_steps','instruction',
                'initial_state_index','initial_state_vector_sha256','nonimage_observation_sha256']
    a=df.loc[df.variant=='M2-LE',fields].copy();b=df.loc[df.variant=='M3-RVA',fields].copy()
    paired=a.merge(b,on=keys,suffixes=('_baseline','_m3'),validate='one_to_one',how='outer',indicator=True)
    require((paired['_merge']=='both').all(),'Incomplete model pair; missing outcomes must not be treated as failures.')
    for field in ['episode_seed','policy_rng_seed','max_steps','instruction','initial_state_index',
                  'initial_state_vector_sha256','nonimage_observation_sha256']:
        require((paired[field+'_baseline']==paired[field+'_m3']).all(),f'Pairing mismatch: {field}')
    paired['delta']=paired.success_m3.astype(int)-paired.success_baseline.astype(int)
    return paired.drop(columns=['_merge'])


def summarize(ctx):
    df=collect_results(ctx)
    if not df.empty:save_csv(df,ctx.output/'closed_loop_episode_results.csv')
    complete=(len(df)==ctx.expected_count)
    if not complete:
        result={'status':'INCOMPLETE','selected_seeds':ctx.seeds,'completed_episodes':len(df),
                'planned_selected_episodes':ctx.expected_count,'missing_episodes':ctx.expected_count-len(df),
                'scientific_conclusion':'NOT_READY; no imputation or success-rate claim for the missing matrix'}
        write_json(ctx.output/'notebook10_summary.json',result);print(json.dumps(result,indent=2));return result
    observed=set(zip(df.model_seed,df.variant,df.condition_id,df.suite_task_id,df.replica))
    require(observed==expected_episode_keys(ctx.seeds),'Evaluation matrix keys differ despite matching count.')
    pairing_audit=validate_all_initial_pairs(ctx,df)
    paired=pair_frame(df);save_csv(paired,ctx.output/'paired_episode_results.csv')
    rates=df.groupby(['model_seed','variant','condition_id'],sort=False)['success'].agg(episodes='size',successes='sum',success_rate='mean').reset_index()
    save_csv(rates,ctx.output/'per_seed_success.csv')
    per_task=df.groupby(['model_seed','variant','condition_id','suite_task_id'])['success'].agg(episodes='size',successes='sum',success_rate='mean').reset_index()
    save_csv(per_task,ctx.output/'per_task_success.csv')
    effects=[]
    for ix,condition in enumerate(CONDITION_IDS):
        p=paired[paired.condition_id==condition].sort_values(['model_seed','suite_task_id','replica'])
        delta=p.delta.to_numpy().reshape(len(ctx.seeds),10,5)
        low,high=paired_interval(delta,BOOTSTRAP_SEED+ix)
        changed_to_success=int(((~p.success_baseline)&p.success_m3).sum())
        changed_to_failure=int((p.success_baseline&(~p.success_m3)).sum())
        base_rates=rates[(rates.variant=='M2-LE')&(rates.condition_id==condition)].success_rate.to_numpy()
        new_rates=rates[(rates.variant=='M3-RVA')&(rates.condition_id==condition)].success_rate.to_numpy()
        effects.append({'condition_id':condition,'training_seed_count':len(ctx.seeds),'paired_episodes':len(p),
            'M2_LE_success_pct':100*float(p.success_baseline.mean()),'M3_RVA_success_pct':100*float(p.success_m3.mean()),
            'delta_pp':100*float(delta.mean()),'paired_bootstrap_low_pp':100*low,'paired_bootstrap_high_pp':100*high,
            'baseline_sd_across_seeds_pp':100*float(base_rates.std(ddof=1)) if len(ctx.seeds)>1 else None,
            'm3_sd_across_seeds_pp':100*float(new_rates.std(ddof=1)) if len(ctx.seeds)>1 else None,
            'seeds_with_positive_delta':int((delta.mean(axis=(1,2))>0).sum()),
            'baseline_fail_m3_success':changed_to_success,'baseline_success_m3_fail':changed_to_failure,
            'nominal_mcnemar_p':exact_mcnemar(changed_to_success,changed_to_failure),
            'uncertainty_scope':'conditional_on_single_checkpoint_pair' if len(ctx.seeds)==1 else 'limited_three_seed_hierarchical_estimate',
            'evidence_role':'EXPLORATORY; not the original N07 confirmatory family'})
    effect=pd.DataFrame(effects);effect['exploratory_holm_p']=holm_adjust(effect.nominal_mcnemar_p)
    save_csv(effect,ctx.output/'m3_vs_m2le_effects.csv')
    gaps=[]
    for seed in ctx.seeds:
        for variant in VARIANTS:
            r=rates[(rates.model_seed==seed)&(rates.variant==variant)].set_index('condition_id')
            for condition in CONDITION_IDS[1:]:
                gaps.append({'model_seed':seed,'variant':variant,'condition_id':condition,
                    'ID_success_pct':100*float(r.loc['E0_ID','success_rate']),
                    'OOD_success_pct':100*float(r.loc[condition,'success_rate']),
                    'ID_minus_OOD_gap_pp':100*float(r.loc['E0_ID','success_rate']-r.loc[condition,'success_rate'])})
    save_csv(pd.DataFrame(gaps),ctx.output/'generalization_gaps.csv')
    efficiency=df.groupby(['variant','condition_id']).agg(episodes=('success','size'),success_rate=('success','mean'),
            mean_steps_all_episodes=('steps_executed','mean'),mean_wall_seconds=('wall_seconds','mean'),
            mean_per_control_step_select_action_seconds=('mean_select_action_seconds','mean'),
            mean_simulator_step_seconds=('mean_simulator_step_seconds','mean')).reset_index()
    save_csv(efficiency,ctx.output/'descriptive_efficiency.csv')
    # Separate figures, using matplotlib's default color cycle. No results are invented.
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(4);width=.36
    ax.bar(x-width/2,effect.M2_LE_success_pct,width,label='M2-LE: fresh paired evaluation')
    ax.bar(x+width/2,effect.M3_RVA_success_pct,width,label='M3-RVA')
    ax.set_xticks(x,['ID / L0','L3 language','Photometric','Gaussian noise'])
    ax.set_ylim(0,100);ax.set_ylabel('Task success (%)')
    ax.set_title(f'M3 architecture extension | {len(ctx.seeds)} training seed(s) | exploratory')
    ax.legend();fig.tight_layout();fig.savefig(ctx.figures/'m3_success_comparison.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for i,row in effect.iterrows():
        ax.plot([row.paired_bootstrap_low_pp,row.paired_bootstrap_high_pp],[i,i],linewidth=2)
        ax.plot(row.delta_pp,i,marker='o',linestyle='none')
    ax.axvline(0,linestyle='--',linewidth=1);ax.set_yticks(range(4),['ID / L0','L3 language','Photometric','Gaussian noise'])
    ax.set_xlabel('M3-RVA minus M2-LE success (percentage points)')
    ax.set_title('Paired 95% bootstrap intervals — exploratory')
    fig.tight_layout();fig.savefig(ctx.figures/'m3_paired_effects.png',dpi=200);plt.close(fig)
    ctx.check_original_inputs()
    for seed in ctx.seeds:
        inp=read_json(ctx.output/f'seed{seed}/inputs.json')
        for variant,text in inp['checkpoint_paths'].items():
            path=local_path(ctx.root,text)
            after={str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
            require(after==inp['checkpoint_file_hashes'][variant],f'Read-only {variant} checkpoint integrity failure.')
    result={'status':'PASS','evaluation_protocol_sha256':ctx.evaluation_hash,'scientific_role':'EXPLORATORY',
         'planned_training_seeds':PLANNED_SEEDS,'evaluated_training_seeds':ctx.seeds,
         'all_planned_seeds_evaluated':ctx.seeds==PLANNED_SEEDS,'completed_episodes':len(df),
         'paired_episode_count':len(paired),'conditions':CONDITION_IDS,'models':VARIANTS,
         'training_performed':False,'new_corruptions_introduced':False,'original_results_overwritten':False,
         'checkpoint_bytes_unchanged':True,'initial_pairing_rule_verified':True,
         'initial_pairing_rule':PAIRING_RULE,
         'pairs_with_bitwise_identical_images':int(pairing_audit.all_images_byte_identical.sum()),
         'pairs_with_tolerated_image_differences':int((~pairing_audit.all_images_byte_identical).sum()),
         'legacy_v1_results_included':False,
         'full_hidden_simulator_state_equivalence_claimed':False,
         'robot_performance':'MEASURED; see effect table, PASS is not superiority',
         'effects':effect.where(pd.notna(effect),None).to_dict(orient='records'),
         'limitations':['Post-hoc architectural extension; original outcomes were already observed.',
             'Single-seed results, if selected, cannot establish training-seed reproducibility.',
             'Three seeds still provide limited information about training variability.',
             'McNemar p-values are nominal episode-level diagnostics and ignore within-task/training-seed dependence.',
             'Bootstrap intervals depend on this small task/replica/seed design; no universal robustness claim.',
             'Control-step select_action latency includes queued actions and is not a pure network-forward benchmark.',
             'Fresh paired baseline rollouts are separate from the original N07/N07b results.',
             'V2 reset/pairing protocol was amended after an image-hash mismatch; all V1 outcomes are excluded.',
             'Equal fixed benchmark vectors and exposed robot states do not verify every hidden simulator variable.',
             'Images may differ within a fixed sparse one-level tolerance; no behavioral equivalence is implied.',
             'Fresh environment allocation per episode changes overhead; timing is descriptive, not a pure model benchmark.'],
         'next':'Write the architectural-comparison report; consistency training is not required by this notebook.'}
    # pandas may keep float NaNs despite .where(None); replace explicitly before strict JSON.
    for row in result['effects']:
        for key,value in list(row.items()):
            if isinstance(value,float) and not math.isfinite(value):row[key]=None
    write_json(ctx.output/'notebook10_summary.json',result)
    print('\nM3 CLOSED-LOOP COMPARISON: COMPLETE',flush=True)
    print(effect[['condition_id','M2_LE_success_pct','M3_RVA_success_pct','delta_pp',
                  'paired_bootstrap_low_pp','paired_bootstrap_high_pp']].to_string(index=False),flush=True)
    print('PASS means the evaluation completed correctly, not that M3 won.',flush=True)
    return result


def execute(ctx):
    with exclusive_rig(ctx.root):
        ctx.prerequisites()
        require(Path(sys.prefix).name=='vla-libero',f'Use vla-libero, not {sys.executable}')
        require(torch.cuda.is_available(),'CUDA unavailable; do not run the simulation comparison on CPU by accident.')
        ctx.check_original_inputs();ctx.freeze();load_components(ctx.root)
        legacy_root=ctx.root/'results/metrics/m3_rva_evaluation/v1'
        legacy_files={str(p.relative_to(ctx.root)):sha(p) for p in legacy_root.rglob('*') if p.is_file()}
        legacy_check=ctx.output/'legacy_v1_preservation.json'
        legacy_record={'files':legacy_files,'note':'V1 retained unchanged, excluded from V2 statistics.'}
        if legacy_check.exists():
            require(read_json(legacy_check)==legacy_record,'Legacy V1 evidence changed since this V2 run began.')
        else:write_json(legacy_check,legacy_record)
        runtime={'python':sys.version.split()[0],'torch':str(torch.__version__),'cuda':torch.version.cuda,
                 'gpu':torch.cuda.get_device_name(0),'lerobot_head':PIN,
                 'packages':{n:metadata.version(n) for n in ('lerobot','transformers','numpy','mujoco','hf-libero')},
                 'GALLIUM_DRIVER':os.environ.get('GALLIUM_DRIVER'),'MUJOCO_GL':os.environ.get('MUJOCO_GL')}
        runtime_path=ctx.output/'runtime.json'
        if runtime_path.exists():require(read_json(runtime_path)==runtime,'Evaluation runtime changed; do not mix results.')
        else:write_json(runtime_path,runtime)
        for seed in ctx.seeds:
            print(f'Checking task 2 / E0 / replica 3 first for seed {seed} (included in the 400-episode matrix)...',flush=True)
            evaluate_seed(ctx,seed,smoke=True)
            evaluate_seed(ctx,seed)
        require(all((ctx.root/name).is_file() and sha(ctx.root/name)==digest for name,digest in legacy_files.items()),
                'Legacy V1 evidence changed during V2; preserve and review.')
        return summarize(ctx)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['plan','run','summarize'])
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--seeds',nargs='+',type=int,default=[42])
    args=parser.parse_args();ctx=Evaluation(args.root,args.seeds)
    try:
        if args.mode=='plan':
            ctx.prerequisites()
            print(json.dumps({'status':'PREREQUISITES_PRESENT; execute only after N09 is idle',
                  'selected_seeds':ctx.seeds,'planned_seeds':PLANNED_SEEDS,'conditions':CONDITION_IDS,
                  'M3_episodes_per_seed':200,'fresh_M2LE_control_episodes_per_seed':200,
                  'total_selected_episodes':ctx.expected_count,'training_performed':False,
                  'next':'Set RUN_M3_CLOSED_LOOP=True only after N09 has finished.'},indent=2))
        elif args.mode=='run':execute(ctx)
        else:
            with exclusive_rig(ctx.root):
                ctx.prerequisites();ctx.freeze();summarize(ctx)
    except NotReady as exc:
        print('WAITING_FOR_N09: '+str(exc),flush=True)
        print('No model was evaluated. Do not stop or edit the active N09 training job.',flush=True)
        return
    except BaseException as exc:
        write_json(ctx.output/'last_error.json',{'type':type(exc).__name__,'message':str(exc),
                'traceback':traceback.format_exc(),'at_utc':datetime.now(timezone.utc).isoformat(),
                'note':'Completed episode JSON files retained. A system error is not a task failure.'})
        raise


if __name__=='__main__':main()
