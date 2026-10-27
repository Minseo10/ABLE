"""Record the exact source and configuration used for the validation sweeps."""
import dataclasses
import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess

from kinder_pddlstream_planning.limbrepositioning3d.stream import MPCConfig

ROOT = pathlib.Path('/home/minseo/research/learning-forceful-manipulation')
OUT = ROOT / 'data/archive/limb_pipeline_fix'
repos = {
    'kinder-baselines': ROOT.parent / 'kinder-baselines',
    'kindergarden': ROOT.parent / 'kindergarden',
    'project': ROOT,
}
files = {
    'kinder-baselines': [
        *('kinder-pddlstream-planning/src/kinder_pddlstream_planning/'
          f'limbrepositioning3d/{name}.py' for name in ('utils', 'stream', 'run')),
        'kinder-pddlstream-planning/tests/test_mpc_replay.py',
    ],
    'kindergarden': ['src/kinder/envs/dynamic3d/limbrepositioning3d.py'],
    'project': ['src/limb_baseline/utils.py', 'scripts/variant_sweep.py',
                'scripts/check_limb_pipeline.py'],
}

def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True)

manifest = {
    'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'gravity': [0, 0, -9.81], 'muscle_tone': 'none',
    'mpc': dataclasses.asdict(MPCConfig()),
    'seeds': [0, 1, 2], 'variants': 16, 'max_planning_time_s': 600,
    'seed0_workers': 16,
    'validation': {
        'final_seed0': 'final_seed0/results.jsonl',
        'additional_seeds': 'seeds1_2/results.jsonl',
        'local_pipeline': 'local_pipeline/*.json',
        'initial_sweep_source': 'sweep_source',
        'initial_sweep_manifest': 'sweep_manifest.json',
        'note': ('Initial 48-run sweep preceded the custom-belief context-copy fix; '
                 'that fix preserves instance constraint hooks, with unchanged '
                 'default variant constraints. Final seed 0 and local pipeline '
                 'checks use the final source snapshot.'),
    },
    'repositories': {}, 'source_sha256': {},
}
for name, repo in repos.items():
    manifest['repositories'][name] = {
        'branch': git(repo, 'branch', '--show-current').strip(),
        'head': git(repo, 'rev-parse', 'HEAD').strip(),
        'status': git(repo, 'status', '--short'),
    }
    if name != 'project':
        manifest['repositories'][name]['main_head'] = git(
            repo, 'rev-parse', 'main').strip()
    selected = files[name]
    snapshot = OUT / 'source' / name
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / 'changes.patch').write_text(git(repo, 'diff', '--', *selected))
    for relative in selected:
        source = repo / relative
        if not source.exists():
            continue
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest['source_sha256'][f'{name}/{relative}'] = hashlib.sha256(
            source.read_bytes()).hexdigest()
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
