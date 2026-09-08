"""Bounded BC diagnosis; preserves all existing checkpoints and populations."""
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import torch

from controllers.policy_network import flatten_parameters
from training.behavior_clone import BehaviorCloningConfig, save_behavior_clone, train_behavior_clone
from training.dataset import load_demonstrations
from training.evaluate import FitnessConfig, evaluate_policy

ROOT = Path(__file__).resolve().parent
PATHS = {int(p.stem.split('-seed-')[1]): p for p in Path('artifacts/human-driving-corrections').glob('**/trial-*.jsonl')}
FIT = FitnessConfig(**json.loads(Path('artifacts/evolution/phase-009.json').read_text())['config']['fitness'])
SEEDS = (110, 111, 112, 271, 997)
torch.set_num_threads(1)
trials = [
    ('trial117-50', [117], 50),
    ('trial117-150', [117], 150),
    ('trial117-300', [117], 300),
    ('trials115-117-50', [115, 117], 50),
    ('trials115-117-150', [115, 117], 150),
    ('trials115-117-119-150', [115, 117, 119], 150),
]
summaries = []
for name, source_seeds, epochs in trials:
    target = ROOT / name
    target.mkdir(exist_ok=False)
    data = load_demonstrations([PATHS[s] for s in source_seeds])
    cfg = BehaviorCloningConfig(epochs=epochs, validation_fraction=0)
    policy, metadata = train_behavior_clone(data, cfg)
    metadata['study'] = {'selection_seeds': SEEDS, 'source_seeds': source_seeds, 'purpose': 'development challenger; no held-out action validation'}
    save_behavior_clone(target / 'policy.pt', policy, metadata)
    results = [evaluate_policy(flatten_parameters(policy), seed, round_seconds=30, fitness_config=FIT).to_dict() for seed in SEEDS]
    summary = {'name': name, 'training': metadata['training'], 'fitness_config': asdict(FIT), 'results': results,
               'laps_seeds': sum(r['lap_count'] > 0 for r in results),
               'mean_distance': mean(r['scored_distance_m'] for r in results),
               'fitness': .75 * mean(r['fitness'] for r in results) + .25 * min(r['fitness'] for r in results)}
    (target / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    summaries.append(summary)
    print(json.dumps({k: summary[k] for k in ('name', 'laps_seeds', 'mean_distance', 'fitness')}), flush=True)
(ROOT / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
