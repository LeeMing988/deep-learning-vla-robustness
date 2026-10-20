"""Project-local M3-RVA training extension; does not edit LeRobot or the N00--N08 artifacts.

Adapted data/augmentation/optimizer logic: user's executed Notebook 06.
Adapter definition: user's executed Notebook 09 Stage A.
New work: real-data qualification, full model packaging, recovery, and bounded M3 training.
This module is embedded in Notebook 09; it is not a change to third-party SmolVLA code.
"""
from __future__ import annotations
import os
for _name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[_name] = '1'
import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from importlib import metadata
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler

RELEASE = 'm3_rva_training_v1'
PIN = '7e241bd630a3719a56157a497ce5d08f244784f1'
N06_CODE_SHA = '4bc5ce22031e4b97ffbe06d1a5a7166c1d390daa6fddb5925b34b4f827867604'
PLANNED_SEEDS = [42, 123, 2026]
RANK = 64
SCALE = 0.1
TOTAL_STEPS = 25000
LOG_EVERY = 100
SAVE_EVERY = 1000
QUAL_WARMUP = 10
QUAL_MEASURED = 100
DEVICE = 'cuda'


def require(value: Any, message: str) -> None:
    if not bool(value):
        raise RuntimeError(message)


def read_json(path: Path) -> dict:
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def jsonable(x: Any) -> Any:
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    return x


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(jsonable(data), indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()


def obj_sha(data: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(data), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


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


def notebook_code_sha(path: Path) -> str:
    parts = [canonical_ast_dump(ast.parse(''.join(c['source'])))
             for c in read_json(path)['cells'] if c['cell_type'] == 'code']
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()


def project_path(root: Path, text: str) -> Path:
    p = Path(text).expanduser()
    p = (root / p if not p.is_absolute() else p).resolve()
    require(p.is_relative_to(root), f'Expected project-local artifact: {p}')
    return p


def clean_memory() -> None:
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(s: dict) -> None:
    random.setstate(s['python'])
    n = s['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), int(n[2]), int(n[3]), float(n[4])))
    torch.set_rng_state(s['torch'].cpu())
    if s['cuda']: torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])


def tree_cpu(x: Any) -> Any:
    if isinstance(x, torch.Tensor): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k: tree_cpu(v) for k, v in x.items()}
    if isinstance(x, list): return [tree_cpu(v) for v in x]
    if isinstance(x, tuple): return tuple(tree_cpu(v) for v in x)
    return deepcopy(x)


def tree_equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def atomic_torch_save(data: dict, path: Path) -> None:
    temp = path.with_name(path.name + '.tmp')
    torch.save(data, temp)
    temp.replace(path)


class RobustVisualAdapter(nn.Module):
    """Same token-wise residual adapter that passed the user's N09 Stage A."""
    def __init__(self, hidden_size: int, rank: int = RANK, initial_scale: float = SCALE):
        super().__init__()
        if not (0 < rank <= hidden_size): raise ValueError('Require 0 < rank <= hidden_size.')
        if not math.isfinite(initial_scale) or initial_scale == 0: raise ValueError('Use a nonzero finite scale.')
        self.hidden_size, self.rank = int(hidden_size), int(rank)
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank)
        self.activation = nn.GELU()
        self.up = nn.Linear(rank, hidden_size)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError(f'Expected [B,N,{self.hidden_size}], received {tuple(tokens.shape)}')
        if not tokens.is_floating_point(): raise TypeError('Tokens must be floating point.')
        if tokens.device != self.down.weight.device: raise RuntimeError('Adapter/token device mismatch.')
        x = tokens.to(self.down.weight.dtype)
        delta = self.up(self.activation(self.down(self.norm(x))))
        return (x + self.scale * delta).to(tokens.dtype)


def install_adapter(policy, seed: int):
    core = policy.model
    require(not hasattr(core, 'm3_visual_adapter'), 'Adapter already installed; start a fresh model.')
    dim = int(core.vlm_with_expert.config.text_config.hidden_size)
    # Isolate new initialization: it must not advance the inherited training RNG stream.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed + 9009)
        adapter = RobustVisualAdapter(dim).to(device=next(core.parameters()).device, dtype=torch.float32)
    core.add_module('m3_visual_adapter', adapter)
    connector = core.vlm_with_expert.get_vlm_model().connector
    def apply(module, inputs, output):
        require(isinstance(output, torch.Tensor), 'Unexpected connector output type.')
        return core.m3_visual_adapter(output)
    core._m3_visual_hook = connector.register_forward_hook(apply)
    return adapter


class LanguageAugmentedDataset(Dataset):
    """Episode-level L0/L1/L2 assignment copied in substance from N06."""
    def __init__(self, base, mapping):
        self.base_dataset, self.episode_language_map = base, mapping
        self.meta, self.camera_keys = base.meta, base.meta.camera_keys
    def __len__(self): return len(self.base_dataset)
    def __getitem__(self, index):
        item = dict(self.base_dataset[index])
        item['task'] = self.episode_language_map[int(item['episode_index'])]['instruction']
        return item


class RecoverableBatchSampler:
    """Materialize an epoch's N06 RandomSampler batches; checkpoint the cursor.

    num_workers MUST stay 0. No future batch is consumed before its optimizer step.
    Epoch orders use the same RandomSampler/BatchSampler distributions as N06.
    The DataLoader and RandomSampler share a dedicated generator, as in N06.
    """
    def __init__(self, dataset, batch_size: int, generator):
        self.dataset, self.batch_size, self.generator = dataset, batch_size, generator
        self.batches, self.cursor, self.epoch = None, 0, 0
    def __len__(self): return len(self.dataset) // self.batch_size
    def __iter__(self):
        if self.batches is None:
            sampler = RandomSampler(self.dataset, generator=self.generator)
            self.batches = list(BatchSampler(sampler, self.batch_size, drop_last=True))
            require(len(self.batches) > 0, 'Training dataset smaller than one complete batch.')
        while self.cursor < len(self.batches):
            b = self.batches[self.cursor]
            self.cursor += 1
            yield b
    def new_epoch(self):
        require(self.batches is not None and self.cursor == len(self.batches), 'Cannot skip an incomplete epoch.')
        self.batches, self.cursor = None, 0
        self.epoch += 1
    def state_dict(self):
        return {'batches': deepcopy(self.batches), 'cursor': self.cursor, 'epoch': self.epoch,
                'generator': self.generator.get_state(), 'dataset_length': len(self.dataset),
                'batch_size': self.batch_size}
    def load_state_dict(self, s):
        require(s['dataset_length'] == len(self.dataset) and s['batch_size'] == self.batch_size,
                'Recovery sampler configuration differs.')
        self.batches, self.cursor, self.epoch = deepcopy(s['batches']), int(s['cursor']), int(s['epoch'])
        self.generator.set_state(s['generator'].cpu())


class TrainingStream:
    def __init__(self, dataset, batch_size, seed, collate_fn, saved=None):
        self.generator = torch.Generator().manual_seed(seed)
        self.sampler = RecoverableBatchSampler(dataset, batch_size, self.generator)
        self.loader = DataLoader(dataset, batch_sampler=self.sampler, generator=self.generator,
                                 num_workers=0, pin_memory=(DEVICE == 'cuda'), collate_fn=collate_fn)
        if saved is not None: self.sampler.load_state_dict(saved)
        self.iterator = None
        self.compensate_resume_seed = saved is not None
    def __next__(self):
        if self.iterator is None:
            before = self.generator.get_state()
            self.iterator = iter(self.loader)
            # Mid-epoch resume should not add an extra DataLoader base-seed draw.
            if self.compensate_resume_seed:
                self.generator.set_state(before)
                self.compensate_resume_seed = False
        try:
            return next(self.iterator)
        except StopIteration:
            self.sampler.new_epoch()
            self.iterator = iter(self.loader)
            return next(self.iterator)
    def state_dict(self): return self.sampler.state_dict()


def schedule_multiplier(step, warmup, end, minimum):
    if step < warmup: return max(1e-8, float(step + 1) / float(warmup))
    if step < end:
        progress = min(max((step - warmup) / float(end - warmup), 0.0), 1.0)
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum


class Study:
    """Read the frozen N06 artifacts. This class never rewrites those artifacts."""
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.lerobot = self.root.parent / 'lerobot'
        require(torch.cuda.is_available(), 'CUDA unavailable. Select the vla-libero WSL kernel.')
        require(Path(sys.prefix).name == 'vla-libero', f'Wrong Python environment: {sys.executable}')
        require(git(self.lerobot, 'rev-parse', 'HEAD') == PIN, 'LeRobot source pin changed; do not upgrade.')
        require(not git(self.lerobot, 'status', '--porcelain', '--untracked-files=no'), 'Tracked LeRobot files changed.')
        n06 = self.root / 'notebooks/06_validated_final_retraining.ipynb'
        actual_n06_sha = notebook_code_sha(n06)
        require(actual_n06_sha == N06_CODE_SHA,
                'N06 executable code does not match the audited source after Python-compatible normalization. '
                f'Expected {N06_CODE_SHA}; got {actual_n06_sha}; Python {sys.version.split()[0]}. '
                'Save and upload the local N06 for review; do not replace its code or bypass this check.')
        print('PASS | N06 code matches audited source (Python-compatible AST fingerprint).', flush=True)
        base = self.root / 'configs/experiments'
        self.paths = {
            'handoff': base / 'notebook06_final_budget_handoff_v2.json',
            'n02': self.root / 'results/metrics/robustness_training/notebook02_summary.json',
            'n03': self.root / 'results/metrics/tuning_final_training/notebook03_summary.json',
            'registry': base / 'definitive_final_model_registry_v1.csv',
            'augmentation': base / 'final_robustness_augmentation_manifest_v1.json',
        }
        self.handoff = read_json(self.paths['handoff'])
        self.n02, self.n03 = read_json(self.paths['n02']), read_json(self.paths['n03'])
        h = self.handoff
        require(h['final_budget_frozen'] is True and int(h['final_training_steps']) == TOTAL_STEPS,
                'Expected the frozen 25k handoff.')
        require(h['training_scope'] == 'standard', 'Expected N06 standard trainability scope.')
        require(self.n02['status'] == 'PASS' and self.n03['status'] == 'PASS', 'Original prerequisite status differs.')
        self.paths['split'] = project_path(self.root, h['split_file'])
        self.paths['language'] = project_path(self.root, self.n02['language_table'])
        self.paths['validation'] = project_path(self.root, self.n03['validation_subset'])
        require(sha(self.paths['split']) == h['split_sha256'], 'Frozen split hash mismatch.')
        self.split = pd.read_csv(self.paths['split'])
        self.split['split'] = self.split['split'].astype(str).str.lower().str.strip().replace({'val':'validation','valid':'validation'})
        require(not self.split['episode_index'].duplicated().any(), 'Duplicate split episode IDs.')
        self.train_ids = sorted(self.split.loc[self.split['split'] == 'train', 'episode_index'].astype(int).tolist())
        self.val_ids = sorted(self.split.loc[self.split['split'] == 'validation', 'episode_index'].astype(int).tolist())
        require(len(self.train_ids) == 346 and len(self.val_ids) == 86 and set(self.train_ids).isdisjoint(self.val_ids),
                'Expected the original 346/86 episode split.')
        self.language = pd.read_csv(self.paths['language'])
        require(len(self.language) == 40 and self.language['task_id'].nunique() == 10, 'Language table differs.')
        require(not self.language[['task_id', 'level']].duplicated().any(), 'Duplicate language table keys.')
        require(set(self.language['level']) == {'L0','L1','L2','L3'}, 'Language levels differ.')
        if 'semantic_review' in self.language:
            require((self.language['semantic_review'] == 'PASS').all(), 'Unreviewed language rows.')
        require(self.language['augmented_instruction'].notna().all(), 'Missing instruction text.')
        self.language_lookup = {(int(r.task_id), str(r.level)): str(r.augmented_instruction)
                                for r in self.language.itertuples()}
        self.augmentation = read_json(self.paths['augmentation'])
        expected = {k: {'weight':1.0, 'type':'ColorJitter', 'kwargs':{k:[0.85,1.15]}}
                    for k in ('brightness','contrast','saturation')}
        require(self.augmentation['environment_augmentation'] == expected, 'Photometric settings differ from N06.')
        require(self.augmentation['max_environment_transforms_per_image'] == 2 and
                self.augmentation['random_environment_transform_order'] is True and
                self.augmentation['geometric_augmentation'] is False, 'Augmentation protocol differs.')
        self.lr, self.wd, self.bs = float(h['optimizer']['learning_rate']), float(h['optimizer']['weight_decay']), int(h['optimizer']['batch_size'])
        self.warmup = int(h['scheduler']['warmup_steps'])
        self.end = int(h['scheduler']['cosine_decay_end_step'])
        self.minimum = float(h['scheduler']['min_lr_ratio'])
        require((self.lr, self.wd, self.bs, self.warmup, self.end, self.minimum) ==
                (1e-4, 0.01, 4, 2500, 25000, 0.025), 'Frozen optimizer/schedule differ from the executed N06.')
        self.registry = pd.read_csv(self.paths['registry'])
        require({'variant','seed','checkpoint'}.issubset(self.registry.columns), 'Registry schema mismatch.')
        self.comparators = {}
        for seed in PLANNED_SEEDS:
            rows = self.registry[(self.registry['variant'] == 'M2-LE') & (self.registry['seed'].astype(int) == seed)]
            require(len(rows) == 1, f'Expected one M2-LE comparator at seed {seed}.')
            checkpoint = project_path(self.root, str(rows.iloc[0]['checkpoint']))
            require((checkpoint/'config.json').is_file(), f'Missing original comparator: {checkpoint}')
            self.comparators[str(seed)] = str(checkpoint.relative_to(self.root))
        self.original_hashes = {key: sha(path) for key,path in self.paths.items()}
        self.run_root = self.root / 'experiments/m3_rva_extension/v1'
        self.metrics_root = self.root / 'results/metrics/m3_rva_extension/v1'
        self.protocol_path = base / 'm3_rva_training_protocol_v1.json'
        self.protocol = {
            'release': RELEASE, 'scientific_role': 'POST_HOC_ARCHITECTURAL_EXTENSION',
            'model': 'M3-RVA', 'comparator': 'M2-LE', 'planned_seeds': PLANNED_SEEDS,
            'steps': TOTAL_STEPS, 'batch_size': self.bs, 'learning_rate': self.lr, 'weight_decay': self.wd,
            'warmup_steps': self.warmup, 'decay_end_step': self.end, 'minimum_lr_ratio': self.minimum,
            'gradient_clip': 10.0, 'adapter_rank': RANK, 'initial_residual_scale': SCALE,
            'adapter_initialization': 'zero up-projection; isolated RNG seed=training_seed+9009',
            'trainability': 'N06 standard trainable parameters PLUS adapter; not adapter-only',
            'initialization_repo': 'lerobot/smolvla_base', 'initialization_revision': h['base_model_revision'],
            'dataset_repo': 'lerobot/libero', 'dataset_revision': h['dataset_revision'],
            'source_pin': PIN, 'n06_code_sha256': N06_CODE_SHA, 'runner_sha256': sha(Path(__file__)),
            'runtime': {'python':sys.version.split()[0], 'torch':str(torch.__version__),
                        'cuda':torch.version.cuda, 'lerobot':metadata.version('lerobot'),
                        'transformers':metadata.version('transformers')},
            'artifact_hashes': self.original_hashes,
            'language_levels': ['L0','L1','L2'], 'environment': self.augmentation,
            'num_workers': 0, 'sampler': 'checkpointable N06 RandomSampler/BatchSampler epoch order',
            'comparator_checkpoints': self.comparators,
            'checkpoint_selection': 'fixed final step 25000; no best-OOD or best-validation replacement',
            'validation_subset_frames': 200, 'offline_validation_is_diagnostic': True,
            'inference_and_ood_evaluation': 'NOT_EXECUTED_BY_THIS_TRAINING_RELEASE',
            'planned_evaluation_conditions': ['E0_ID','E1_L3','E2_SEEN_PHOTO','E3_HELDOUT_NOISE'],
            'planned_evaluation_design': 'same N07 tasks, initial states, replica IDs and evaluator; 10 tasks x 5 replicas per seed/condition',
            'prior_outcomes_seen': True, 'extension_evaluation_label': 'EXPLORATORY',
            'new_E6_condition': 'not introduced in this bounded M3-RVA run',
            'consistency_objective': 'not included in this release',
        }
        self.protocol_hash = obj_sha(self.protocol)
        if self.protocol_path.exists():
            require(read_json(self.protocol_path) == self.protocol,
                    'Existing M3 protocol differs. Do not overwrite completed/in-progress experiment settings.')
        else: write_json(self.protocol_path, self.protocol)
        self.metrics_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.ready = False

    def unchanged(self):
        require(all(sha(path) == self.original_hashes[k] for k,path in self.paths.items()), 'An original input artifact changed.')
        require(git(self.lerobot, 'rev-parse','HEAD') == PIN and not git(self.lerobot,'status','--porcelain','--untracked-files=no'), 'LeRobot changed.')

    def setup_data(self):
        if self.ready: return
        from huggingface_hub import snapshot_download
        from lerobot.configs import FeatureType
        from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.utils.feature_utils import dataset_to_policy_features
        from lerobot.utils.collate import lerobot_collate_fn
        self.collate = lerobot_collate_fn
        self.dataset_class = LeRobotDataset
        self.base_snapshot = Path(snapshot_download(repo_id='lerobot/smolvla_base',
                           revision=self.handoff['base_model_revision'], local_files_only=True))
        self.meta = LeRobotDatasetMetadata('lerobot/libero', revision=self.handoff['dataset_revision'])
        features = dataset_to_policy_features(self.meta.features)
        self.base_config = SmolVLAConfig.from_pretrained(str(self.base_snapshot))
        self.base_config.output_features = {k:v for k,v in features.items() if v.type is FeatureType.ACTION}
        self.base_config.input_features = {k:v for k,v in features.items() if k not in self.base_config.output_features}
        self.base_config.device = DEVICE
        self.base_config.push_to_hub = False
        for k,v in self.handoff['training_scope_config'].items():
            require(bool(getattr(self.base_config,k)) == bool(v), f'Scope differs: {k}')
        self.deltas = resolve_delta_timestamps(self.base_config, self.meta)
        base_val = LeRobotDataset('lerobot/libero', revision=self.handoff['dataset_revision'],
                    episodes=self.val_ids, delta_timestamps=self.deltas, video_backend='pyav', return_uint8=True)
        positions = pd.read_csv(self.paths['validation'])['dataset_position'].astype(int).tolist()
        require(len(positions) == 200 and len(set(positions)) == 200 and all(0 <= i < len(base_val) for i in positions),
                'Invalid frozen 200-frame validation subset.')
        self.val_subset = torch.utils.data.Subset(base_val, positions)
        self.ready = True

    def make_training_dataset(self, seed):
        self.setup_data()
        from lerobot.transforms import ImageTransformConfig, ImageTransformsConfig, ImageTransforms
        transform = ImageTransforms(ImageTransformsConfig(enable=True, max_num_transforms=2, random_order=True,
            tfs={k:ImageTransformConfig(**v) for k,v in self.augmentation['environment_augmentation'].items()}))
        base = self.dataset_class('lerobot/libero', revision=self.handoff['dataset_revision'], episodes=self.train_ids,
                     delta_timestamps=self.deltas, image_transforms=transform, video_backend='pyav', return_uint8=True)
        table = self.split.loc[self.split['split']=='train',['episode_index','task_index']].drop_duplicates().sort_values(['task_index','episode_index'])
        levels, mapping, audit = ['L0','L1','L2'], {}, []
        for task_id, group in table.groupby('task_index'):
            for rank, ep in enumerate(group['episode_index'].astype(int).tolist()):
                level = levels[(rank + int(seed) % 3) % 3]
                mapping[ep] = {'task_index':int(task_id), 'level':level,
                               'instruction':self.language_lookup[(int(task_id),level)]}
                audit.append({'seed':seed,'episode_index':ep,'task_index':int(task_id),'language_level':level})
        require(set(mapping)==set(self.train_ids), 'Incomplete language assignment.')
        return LanguageAugmentedDataset(base,mapping), pd.DataFrame(audit)

    def new_policy(self, seed, with_adapter=True):
        self.setup_data()
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained(str(self.base_snapshot), config=deepcopy(self.base_config),
                                              local_files_only=True).to(DEVICE)
        pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=None, dataset_stats=self.meta.stats)
        if with_adapter: install_adapter(policy, seed)
        return policy,pre,post

    def prepare(self, raw, pre):
        raw = dict(raw)
        for k in self.meta.camera_keys:
            if k in raw and isinstance(raw[k],torch.Tensor) and raw[k].dtype == torch.uint8:
                raw[k] = raw[k].to(torch.float32)/255.0
        return pre(raw)

    def optimizer(self, policy):
        params = [p for p in policy.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd,
                               betas=tuple(self.base_config.optimizer_betas), eps=float(self.base_config.optimizer_eps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt,
                 lambda step:schedule_multiplier(step,self.warmup,self.end,self.minimum))
        return params,opt,sched

    def validation(self, policy, pre):
        loader = DataLoader(self.val_subset, batch_size=self.bs, shuffle=False,
             generator=torch.Generator().manual_seed(0), num_workers=0, pin_memory=True, collate_fn=self.collate)
        saved = rng_state()
        seed_all(17090)  # common stochastic validation draws; not used to select a checkpoint
        policy.eval()
        weighted, count = 0.0, 0
        try:
            with torch.no_grad():
                for raw in loader:
                    batch = self.prepare(raw, pre)
                    loss,_ = policy(batch)
                    require(torch.isfinite(loss), 'Nonfinite validation loss.')
                    n = int(batch['action'].shape[0]); weighted += float(loss)*n; count += n
        finally:
            restore_rng(saved)
        require(count == 200, 'Validation frame count changed.')
        return {'validation_loss': weighted/count, 'validation_samples':count,
                'validation_role':'OFFLINE_DIAGNOSTIC_ONLY_NOT_ROBOT_SUCCESS'}


def full_bundle(policy, pre, post, directory: Path, study: Study, seed: int, step: int):
    """Explicit M3 export: all tensors + config + BOTH processors + loader manifest.

    The standard SmolVLAPolicy loader alone must not be used for this bundle.
    """
    from safetensors.torch import save_file
    require(not directory.exists(), f'Bundle already exists: {directory}')
    directory.mkdir(parents=True)
    # Use the same proven LeRobot export entry point as N06, then store all state keys explicitly.
    policy.save_pretrained(directory)
    pre.save_pretrained(directory)
    post.save_pretrained(directory)
    # Clone breaks tied storage for a self-contained strict full-state dictionary.
    weights = {k:v.detach().cpu().contiguous().clone() for k,v in policy.state_dict().items()}
    save_file(weights,str(directory/'model.safetensors'))
    del weights
    manifest = {'format':RELEASE,'source_pin':PIN,'protocol_hash':study.protocol_hash,
                'runner_sha256':sha(Path(__file__)), 'seed':seed,'step':step,
                'adapter':{'rank':RANK,'initial_scale':SCALE},
                'trainable_names':[n for n,p in policy.named_parameters() if p.requires_grad],
                'loader':'src.m3_rva_training.load_m3_bundle',
                'files':{p.name:sha(p) for p in directory.iterdir() if p.is_file()}}
    write_json(directory/'m3_manifest.json',manifest)
    return manifest


def load_m3_bundle(directory: str | Path, device='cuda'):
    """Explicit full M3 reconstruction for validation and the later robot evaluator."""
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies import make_pre_post_processors
    directory = Path(directory)
    m = read_json(directory/'m3_manifest.json')
    require(m['format']==RELEASE and m['source_pin']==PIN, 'M3 bundle format/pin differs.')
    require(m['runner_sha256']==sha(Path(__file__)), 'M3 loader code differs from saved bundle.')
    require(m['adapter']=={'rank':RANK,'initial_scale':SCALE}, 'M3 adapter settings differ.')
    for name,digest in m['files'].items(): require(sha(directory/name)==digest,f'Bundle hash mismatch: {name}')
    cfg = SmolVLAConfig.from_pretrained(str(directory),local_files_only=True)
    cfg.device = device
    cfg.push_to_hub = False
    policy = SmolVLAPolicy(cfg)
    install_adapter(policy, int(m['seed']))
    state = load_file(str(directory/'model.safetensors'),device='cpu')
    # assign preserves saved dtypes. Remember/reconstruct any tied parameter aliases.
    # No optimizer exists at this point (optimizers must be created after assignment).
    alias_groups = {}
    for name, parameter in policy.named_parameters(remove_duplicate=False):
        alias_groups.setdefault(id(parameter), []).append(name)
    policy.load_state_dict(state,strict=True,assign=True)
    for names in alias_groups.values():
        if len(names) > 1:
            primary = policy.get_parameter(names[0])
            for name in names[1:]:
                require(torch.equal(state[names[0]], state[name]), 'Saved tied parameter values disagree.')
                parent_name, leaf = name.rsplit('.', 1)
                setattr(policy.get_submodule(parent_name), leaf, primary)
    del state
    policy.to(device)
    required = set(m['trainable_names'])
    require(required == {n for n,p in policy.named_parameters() if p.requires_grad}, 'Trainability did not reconstruct.')
    pre,post = make_pre_post_processors(policy_cfg=policy.config,pretrained_path=str(directory),
                    preprocessor_overrides={'device_processor':{'device':device}})
    policy.eval()
    return policy,pre,post


def fixed_probe(study, policy, pre, post, raw, directory):
    batch = study.prepare(deepcopy(raw),pre)
    # Read-only deterministic probe, separate from training draws.
    generator = torch.Generator(device=DEVICE).manual_seed(77109)
    shape = (batch['action'].shape[0],policy.config.chunk_size,policy.config.max_action_dim)
    noise = torch.randn(shape,generator=generator,device=DEVICE)
    time_input = torch.full((shape[0],),0.5,device=DEVICE)
    policy.eval()
    with torch.no_grad():
        loss,_ = policy(deepcopy(batch),noise=noise,time=time_input)
        policy.reset()
        chunk = policy.predict_action_chunk(deepcopy(batch),noise=noise.clone())
        action = post(chunk.clone())
    require(torch.isfinite(loss) and torch.isfinite(chunk).all(), 'Invalid fixed probe output.')
    payload = {'raw_batch':tree_cpu(raw),'processed_batch':tree_cpu(batch),
               'noise':noise.cpu(),'time':time_input.cpu(),'loss':loss.cpu(),
               'chunk':chunk.cpu(),'unnormalized_action':tree_cpu(action)}
    atomic_torch_save(payload,directory/'probe.pt')
    return payload


def verify_bundle(directory: Path, probe_path: Path, output: Path):
    policy,pre,post = load_m3_bundle(directory, device=DEVICE)
    probe = torch.load(probe_path,map_location='cpu',weights_only=True)
    raw = deepcopy(probe['raw_batch'])
    # Camera keys from reconstructed config; conversion is the same N06 conversion.
    for k in policy.config.image_features:
        if k in raw and isinstance(raw[k],torch.Tensor) and raw[k].dtype==torch.uint8:
            raw[k]=raw[k].float()/255.0
    batch = pre(raw)
    require(tree_equal(tree_cpu(batch),probe['processed_batch']), 'Reloaded processor changed the real input batch.')
    noise,time_input=probe['noise'].to(DEVICE),probe['time'].to(DEVICE)
    with torch.no_grad():
        loss,_=policy(deepcopy(batch),noise=noise,time=time_input)
        policy.reset()
        chunk=policy.predict_action_chunk(deepcopy(batch),noise=noise.clone()).cpu()
        action=tree_cpu(post(chunk.to(DEVICE)))
    require(torch.allclose(loss.cpu(),probe['loss'],rtol=1e-5,atol=1e-6),'Reloaded loss mismatch.')
    require(torch.allclose(chunk,probe['chunk'],rtol=1e-5,atol=1e-6),'Reloaded action chunk mismatch.')
    # Postprocessor outputs are tensors in the existing LeRobot pipeline.
    if isinstance(action,torch.Tensor):
        require(torch.allclose(action,probe['unnormalized_action'],rtol=1e-5,atol=1e-6),'Unnormalized action mismatch.')
    else:
        require(tree_equal(action,probe['unnormalized_action']),'Postprocessor output mismatch.')
    result={'status':'PASS','fresh_process':True,'input_processor_exact':True,'loss_error':float(abs(loss.cpu()-probe['loss'])),
            'action_max_abs_error':float((chunk-probe['chunk']).abs().max()),'postprocessor_checked':True}
    write_json(output,result)
    print('FRESH-PROCESS FULL MODEL + BOTH PROCESSORS: PASS',flush=True)


def spawn_verification(directory, probe, output):
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'verify','--model',str(directory),
                    '--probe',str(probe),'--output',str(output)],check=True)
    return read_json(output)


def check_real_batch(study, policy, pre, raw):
    require('episode_index' in raw and 'action_is_pad' in raw,'Real batch missing episode ID or action padding mask.')
    eps=torch.as_tensor(raw['episode_index']).reshape(-1).tolist()
    require(all(int(e) in set(study.train_ids) for e in eps),'A validation episode entered the training batch.')
    batch=study.prepare(deepcopy(raw),pre)
    action=batch['action']; mask=batch['action_is_pad'].bool()
    require(action.shape==(study.bs,policy.config.chunk_size,policy.config.action_feature.shape[0]),
            f'Unexpected action shape: {tuple(action.shape)}')
    require(mask.shape==action.shape[:2] and (~mask).any(),'Invalid action mask.')
    require(torch.isfinite(action).all() and torch.isfinite(batch['observation.state']).all(),'Nonfinite action/state.')
    camera_info={}
    for k in policy.config.image_features:
        require(k in batch,f'Missing expected training camera: {k}')
        x=batch[k]
        require(x.is_floating_point() and torch.isfinite(x).all() and float(x.min())>=0 and float(x.max())<=1,
                f'Camera is not finite floating [0,1]: {k}')
        camera_info[k]={'shape':list(x.shape),'dtype':str(x.dtype),'min':float(x.min()),'max':float(x.max())}
    return {'episode_ids':[int(e) for e in eps],'action_shape':list(action.shape),
            'valid_action_timesteps':int((~mask).sum()),'cameras':camera_info}


@contextmanager
def job_lock(study):
    import fcntl
    path=study.metrics_root/'rig_job.lock'
    with path.open('a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('An M3 job is already running. Do not launch a duplicate.')
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)


def qualify(study):
    report_path=study.metrics_root/'qualification.json'
    if report_path.exists():
        previous=read_json(report_path)
        if previous.get('status')=='PASS' and previous.get('protocol_hash')==study.protocol_hash:
            study.unchanged(); print('Compatible real-data qualification already passed; reusing.',flush=True); return previous
    out=study.metrics_root/'qualification_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    study.setup_data()
    seed_all(42)
    dataset,audit=study.make_training_dataset(42)
    stream=TrainingStream(dataset,study.bs,42,study.collate)
    policy,pre,post=study.new_policy(42,with_adapter=False)
    raw=next(stream)
    batch_report=check_real_batch(study,policy,pre,raw)
    batch=study.prepare(deepcopy(raw),pre)
    noise=torch.zeros((study.bs,policy.config.chunk_size,policy.config.max_action_dim),device=DEVICE)
    flow_time=torch.full((study.bs,),0.5,device=DEVICE)
    policy.eval()
    with torch.no_grad(): reference,_=policy(deepcopy(batch),noise=noise,time=flow_time)
    inherited={n:bool(p.requires_grad) for n,p in policy.named_parameters()}
    adapter=install_adapter(policy,42)
    with torch.no_grad(): identity,_=policy(deepcopy(batch),noise=noise,time=flow_time)
    require(torch.allclose(reference,identity,rtol=1e-5,atol=1e-6),'Adapter initial identity failed on real batch.')
    require(all(p.requires_grad==inherited[n] for n,p in policy.named_parameters() if n in inherited),'Inherited trainability changed.')
    params,opt,scheduler=study.optimizer(policy)
    require({id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in params},'Optimizer membership differs.')
    init_adapter=adapter.up.weight.detach().clone()
    tracked_expert=next(p for n,p in policy.named_parameters() if n in inherited and p.requires_grad)
    expert_version=tracked_expert._version
    del reference,identity,batch,noise,flow_time
    clean_memory()
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    policy.train()
    for step in range(1,QUAL_WARMUP+QUAL_MEASURED+1):
        torch.cuda.synchronize(); started=time.perf_counter()
        raw=next(stream)
        batch=study.prepare(raw,pre)
        opt.zero_grad(set_to_none=True)
        loss,_=policy(batch)
        require(torch.isfinite(loss),f'Nonfinite real loss at diagnostic step {step}.')
        loss.backward()
        require(adapter.up.weight.grad is not None and torch.isfinite(adapter.up.weight.grad).all(),'Adapter gradient missing/nonfinite.')
        grad=torch.nn.utils.clip_grad_norm_(params,10.0,error_if_nonfinite=True)
        opt.step();scheduler.step()
        if hasattr(policy,'update'): policy.update()
        torch.cuda.synchronize()
        rows.append({'step':step,'warmup':step<=QUAL_WARMUP,'seconds':time.perf_counter()-started,
                     'loss':float(loss.detach()),'grad_norm':float(grad),
                     'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                     'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3})
        if step in (1,QUAL_WARMUP) or step%25==0 or step==QUAL_WARMUP+QUAL_MEASURED:
            print(f'REAL-DATA QUALIFICATION | {step}/{QUAL_WARMUP+QUAL_MEASURED} | loss={float(loss):.5f}',flush=True)
    require(not torch.equal(init_adapter,adapter.up.weight.detach()),'Real-data updates did not change the adapter.')
    require(tracked_expert._version>expert_version,'Planned expert parameters were not updated.')
    df=pd.DataFrame(rows); df.to_csv(out/'real_training_speed_probe.csv',index=False)
    seconds=float(df.loc[~df.warmup,'seconds'].mean())
    print(f'MEASURED | {seconds:.4f} s/step | 25k compute estimate {seconds*25000/60:.1f} min, excluding validation/save/load.',flush=True)
    opt.zero_grad(set_to_none=True)
    manifest=full_bundle(policy,pre,post,out/'model',study,42,QUAL_WARMUP+QUAL_MEASURED)
    raw_probe = next(stream)
    fixed_probe(study,policy,pre,post,raw_probe,out)
    counts={'total_parameters':sum(p.numel() for p in policy.parameters()),
            'trainable_parameters':sum(p.numel() for p in policy.parameters() if p.requires_grad),
            'adapter_parameters':sum(p.numel() for p in adapter.parameters())}
    # Release every model/gradient reference BEFORE the fresh-process verification.
    del policy,pre,post,opt,scheduler,params,adapter,tracked_expert,init_adapter,loss,batch,grad,stream,dataset
    clean_memory()
    verification=spawn_verification(out/'model',out/'probe.pt',out/'reload_verification.json')
    study.unchanged()
    result={'status':'PASS','protocol_hash':study.protocol_hash,'batch_size':study.bs,
            'steps':QUAL_WARMUP+QUAL_MEASURED,'diagnostic_updates_discarded':True,
            'real_batch':batch_report,'parameter_counts':counts,'mean_seconds_per_step':seconds,
            'estimate_25k_minutes_excluding_overheads':seconds*25000/60,
            'peak_allocated_gib':float(df.peak_allocated_gib.max()),'peak_reserved_gib':float(df.peak_reserved_gib.max()),
            'full_model_and_processors_reload':verification,'output_directory':str(out.relative_to(study.root)),
            'scientific_status':'NOT_EVALUATED','created_at':datetime.now(timezone.utc).isoformat()}
    write_json(out/'qualification.json',result);write_json(report_path,result)
    print('M3 REAL-DATA / BATCH-4 / FULL-RELOAD QUALIFICATION: PASS',flush=True)
    return result


def recover_checkpoint(path, policy, opt, scheduler, stream, study):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    require(saved['protocol_hash']==study.protocol_hash,'Recovery belongs to a different protocol.')
    policy.load_state_dict(saved['model'],strict=True)  # model dtypes match the same base initialization
    opt.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    return saved


def train_seed(study,seed):
    require(seed in PLANNED_SEEDS,'Seed outside frozen matrix.')
    q=read_json(study.metrics_root/'qualification.json')
    require(q.get('status')=='PASS' and q.get('protocol_hash')==study.protocol_hash,'Real-data qualification must pass first.')
    run=study.run_root/f'm3rva_seed{seed}_{TOTAL_STEPS}'
    run.mkdir(parents=True,exist_ok=True)
    result_path=run/'result.json'; recovery_path=run/'recovery.pt'
    if result_path.exists():
        old=read_json(result_path)
        require(old.get('protocol_hash')==study.protocol_hash,'Existing result uses a different protocol.')
        if old.get('status')=='PASS':
            manifest=read_json(run/'model/m3_manifest.json')
            require(manifest['protocol_hash']==study.protocol_hash and manifest['step']==TOTAL_STEPS,'Final model manifest differs.')
            for n,h in manifest['files'].items():require(sha(run/'model'/n)==h,f'Final bundle changed: {n}')
            print(f'M3 seed {seed}: completed compatible run reused.',flush=True);return old
    manifest_path=run/'run_manifest.json'
    config={'seed':seed,'protocol_hash':study.protocol_hash,'protocol':study.protocol}
    if manifest_path.exists(): require(read_json(manifest_path)==config,'Existing run configuration mismatch.')
    else: write_json(manifest_path,config)
    # Do not archive or overwrite unknown model output. Our recovery remains the authority.
    if (run/'model').exists():
        existing = read_json(run/'model/m3_manifest.json')
        require(existing['protocol_hash'] == study.protocol_hash and existing['step'] == TOTAL_STEPS
                and recovery_path.is_file(), 'Final bundle/recovery mismatch; preserve the files for review.')
    require(shutil.disk_usage(run).free >= 8 * 1024**3, 'Allow at least 8 GiB free for model, optimizer recovery and atomic saves.')
    study.setup_data();seed_all(seed)
    dataset,audit=study.make_training_dataset(seed)
    audit.to_csv(run/'language_assignment.csv',index=False)
    stream=TrainingStream(dataset,study.bs,seed,study.collate)
    policy,pre,post=study.new_policy(seed)
    params,opt,scheduler=study.optimizer(policy)
    done=0; train_seconds=0.0; rows=[]; peak_allocated=0.0;peak_reserved=0.0
    if recovery_path.exists():
        saved=recover_checkpoint(recovery_path,policy,opt,scheduler,stream,study)
        done=int(saved['step']); require(0<=done<=TOTAL_STEPS,'Invalid recovery step.')
        stream=TrainingStream(dataset,study.bs,seed,study.collate,saved=saved['sampler'])
        # Materialize iterator lazily; generator compensation does not consume the global RNG.
        restore_rng(saved['rng'])
        train_seconds=float(saved['train_seconds']);rows=saved['rows']
        peak_allocated=float(saved['peak_allocated_gib']);peak_reserved=float(saved['peak_reserved_gib'])
        del saved
        print(f'RESUMING M3 seed {seed} at completed optimizer step {done}.',flush=True)
    else:
        print(f'STARTING M3 seed {seed} from the pinned BASE, not from M2-LE.',flush=True)
    policy.train();torch.cuda.reset_peak_memory_stats()
    opt.zero_grad(set_to_none=True)
    for step in range(done+1,TOTAL_STEPS+1):
        torch.cuda.synchronize();started=time.perf_counter()
        raw=next(stream);batch=study.prepare(raw,pre)
        opt.zero_grad(set_to_none=True)
        used_lr=float(opt.param_groups[0]['lr'])
        loss,_=policy(batch)
        require(torch.isfinite(loss),f'Nonfinite training loss at seed {seed}, step {step}. Last recovery retained.')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,10.0,error_if_nonfinite=True)
        opt.step();scheduler.step()
        if hasattr(policy,'update'):policy.update()
        torch.cuda.synchronize()
        step_seconds=time.perf_counter()-started
        train_seconds+=step_seconds
        peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated()/1024**3)
        peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved()/1024**3)
        if step==1 or step%LOG_EVERY==0 or step==TOTAL_STEPS:
            row={'step':step,'loss':float(loss.detach()),'grad_norm':float(grad),
                 'learning_rate_used':used_lr,'learning_rate_next':float(opt.param_groups[0]['lr']),
                 'step_seconds':step_seconds,'training_seconds':train_seconds,
                 'peak_allocated_gib':peak_allocated,'peak_reserved_gib':peak_reserved}
            rows.append(row)
            print(f'M3-RVA seed={seed} | {step:5d}/{TOTAL_STEPS} | loss={row["loss"]:.5f} | lr={used_lr:.3e} | '
                  f'compute={train_seconds/60:.1f} min',flush=True)
        if step%SAVE_EVERY==0 or step==TOTAL_STEPS:
            opt.zero_grad(set_to_none=True)
            checkpoint={'protocol_hash':study.protocol_hash,'seed':seed,'step':step,
                        'model':policy.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                        'sampler':stream.state_dict(),'rng':rng_state(),'rows':rows,'train_seconds':train_seconds,
                        'peak_allocated_gib':peak_allocated,'peak_reserved_gib':peak_reserved}
            atomic_torch_save(checkpoint,recovery_path)
            pd.DataFrame(rows).to_csv(run/'train_metrics.csv',index=False)
            del checkpoint
            print(f'  Recovery saved at step {step}.',flush=True)
    # Export only the fixed final checkpoint; no best-validation or OOD selection.
    opt.zero_grad(set_to_none=True)
    model=run/'model'
    if model.exists():
        require(done == TOTAL_STEPS, 'Existing final bundle cannot be paired with an earlier recovery.')
        candidate=model
    else:
        candidate=run/('model_candidate_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
        full_bundle(policy,pre,post,candidate,study,seed,TOTAL_STEPS)
    raw_probe=next(iter(DataLoader(study.val_subset,batch_size=study.bs,shuffle=False,
                           generator=torch.Generator().manual_seed(0),num_workers=0,collate_fn=study.collate)))
    fixed_probe(study,policy,pre,post,raw_probe,run)
    val=study.validation(policy,pre)
    counts={'total_parameters':sum(p.numel() for p in policy.parameters()),
            'trainable_parameters':sum(p.numel() for p in policy.parameters() if p.requires_grad)}
    del policy,pre,post,opt,scheduler,params,stream,dataset
    # If this was a completed-step recovery, these loop variables may not exist.
    for name in ('loss','grad','batch','raw'):
        if name in locals():
            if name=='loss':loss=None
            elif name=='grad':grad=None
            elif name=='batch':batch=None
            elif name=='raw':raw=None
    clean_memory()
    verification=spawn_verification(candidate,run/'probe.pt',run/'reload_verification.json')
    study.unchanged()
    if candidate != model: candidate.rename(model)
    result={'status':'PASS','scientific_status':'CLOSED_LOOP_NOT_EVALUATED','variant':'M3-RVA',
            'seed':seed,'steps':TOTAL_STEPS,'protocol_hash':study.protocol_hash,
            'checkpoint':str(model.relative_to(study.root)),'loader':'src.m3_rva_training.load_m3_bundle',
            'training_seconds_excluding_checkpoint_io':train_seconds,'training_minutes':train_seconds/60,
            'peak_allocated_gib':peak_allocated,'peak_reserved_gib':peak_reserved,
            'restoration':verification,**val,**counts}
    write_json(result_path,result)
    print(f'M3 seed {seed}: TRAINING + OFFLINE VALIDATION + FULL RELOAD PASS. Robot success not yet evaluated.',flush=True)
    return result


def update_registry(study):
    records=[]
    for seed in PLANNED_SEEDS:
        p=study.run_root/f'm3rva_seed{seed}_{TOTAL_STEPS}'/'result.json'
        if p.exists():
            r=read_json(p)
            if r.get('status')=='PASS' and r.get('protocol_hash')==study.protocol_hash: records.append(r)
    if records:
        pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in records]).to_csv(
                 study.metrics_root/'m3_model_registry.csv',index=False)
    write_json(study.metrics_root/'training_summary.json',{
        'protocol_hash':study.protocol_hash,'planned_seeds':PLANNED_SEEDS,
        'completed_seeds':[r['seed'] for r in records],
        'training_status':'ALL_THREE_SEEDS_COMPLETE' if len(records)==3 else 'PARTIAL_PLANNED_MATRIX',
        'robot_success':'NOT_EVALUATED','original_models_retrained':False,'consistency_arm_started':False,
        'results':records})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['plan','qualify','train','verify'])
    parser.add_argument('--root',type=Path)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42])
    parser.add_argument('--model',type=Path)
    parser.add_argument('--probe',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='verify':
        require(args.model is not None and args.probe is not None and args.output is not None,'Missing verification paths.')
        verify_bundle(args.model,args.probe,args.output);return
    require(args.root is not None,'Provide --root research repository.')
    study=Study(args.root)
    if args.mode=='plan':
        stage_a=read_json(study.root/'results/metrics/m3_design_audit/m3_preflight_latest.json')
        require(stage_a.get('engineering_status')=='PASS','N09 Stage A PASS pointer missing.')
        previous=read_json(project_path(study.root,stage_a['summary']))
        require(previous.get('engineering_status')=='PASS' and previous['observations']['environment']['lerobot_head']==PIN,
                'Stage A did not pass on the expected source.')
        print(json.dumps({'protocol_path':str(study.protocol_path),'protocol_hash':study.protocol_hash,
                          'planned_seeds':PLANNED_SEEDS,'train_now':args.seeds,'steps_per_seed':TOTAL_STEPS,
                          'batch_size':study.bs,'adapter_rank':RANK,'original_artifacts_read_only':True},indent=2),flush=True)
        return
    with job_lock(study):
        try:
            if args.mode=='qualify': qualify(study)
            else:
                for seed in args.seeds:
                    train_seed(study,seed)
                    update_registry(study)
        except BaseException as exc:
            write_json(study.metrics_root/'last_error.json',{'mode':args.mode,'type':type(exc).__name__,
                'message':str(exc),'traceback':traceback.format_exc(),'at':datetime.now(timezone.utc).isoformat(),
                'note':'Existing core checkpoints untouched; valid recovery checkpoint retained.'})
            raise

if __name__=='__main__': main()
