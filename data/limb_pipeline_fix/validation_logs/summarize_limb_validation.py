"""Aggregate sweep checks and verify that every saved recording is readable."""
from collections import Counter
import json
from pathlib import Path
import statistics

from PIL import Image

ROOT = Path('/home/minseo/research/learning-forceful-manipulation')
OUT = ROOT / 'data/limb_pipeline_fix'
rows = []
for directory in ('final_seed0', 'seeds1_2'):
    rows.extend(json.loads(line) for line in
                (OUT / directory / 'results.jsonl').read_text().splitlines())
assert len(rows) == len({(r['variant'], r['seed']) for r in rows}) == 48
records = []
for row in rows:
    with Image.open(row['video']) as image:
        assert image.n_frames > 1, row['video']
        duration = 0
        for frame in range(image.n_frames):
            image.seek(frame)
            image.load()
            duration += image.info.get('duration', 0)
        records.append({'variant': row['variant'], 'seed': row['seed'],
                        'frames': image.n_frames, 'duration_ms': duration,
                        'path': row['video']})
summary = {'runs': len(rows), 'goal_reached': sum(r['goal_reached'] for r in rows),
           'by_seed': {}, 'by_limb': {},
           'max_replay_joint_error_rad': max(r['replay_max_joint_error'] for r in rows),
           'peak_robot_joint_velocity_rad_s': max(r['robot_peak_velocity'] for r in rows),
           'planning_time_s': {
               'min': min(r['plan_time'] for r in rows),
               'median': statistics.median(r['plan_time'] for r in rows),
               'max': max(r['plan_time'] for r in rows)},
           'videos': len(records), 'motion_outcomes': dict(Counter(
               {key: sum(r['motion_outcomes'].get(key, 0) for r in rows)
                for key in {k for r in rows for k in r['motion_outcomes']}}))}
for seed in (0, 1, 2):
    selected = [r for r in rows if r['seed'] == seed]
    summary['by_seed'][seed] = {
        'runs': len(selected), 'goal_reached': sum(r['goal_reached'] for r in selected)}
for limb in ('arm', 'leg'):
    selected = [r for r in rows if r['variant'].endswith(limb)]
    summary['by_limb'][limb] = {
        'runs': len(selected), 'goal_reached': sum(r['goal_reached'] for r in selected)}
local = [json.loads(path.read_text()) for path in
         sorted((OUT / 'local_pipeline').glob('*.json'))]
summary['local_pipeline'] = {'runs': len(local),
    'goal_reached': sum(r['goal_reached'] for r in local),
    'rom_violation_steps': sum(r['rom_violation_steps'] for r in local),
    'variants': local}
summary['tests'] = {'project': 83, 'planner_package': 56,
                    'mpc_replay_regressions_in_package': 21,
                    'mypy': 'passed', 'black': 'passed', 'isort': 'passed',
                    'pylint': 'passed with obsolete config options disabled',
                    'ruff_local_changes': 'passed'}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(OUT / 'video_metadata.json').write_text(json.dumps(records, indent=2) + '\n')
(OUT / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
print(json.dumps({k: v for k, v in summary.items() if k != 'local_pipeline'}, indent=2))
print('Local pipeline:', summary['local_pipeline']['runs'],
      summary['local_pipeline']['goal_reached'])
