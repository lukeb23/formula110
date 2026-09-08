import json
from pathlib import Path
from statistics import mean

import torch

from controllers.policy_network import flatten_parameters
from training.behavior_clone import BehaviorCloningConfig, save_behavior_clone, train_behavior_clone
from training.dataset import load_demonstrations
from training.evaluate import FitnessConfig, evaluate_policy

root = Path(__file__).resolve().parent
target = root / 'trial117-smooth-l1-150'
target.mkdir(exist_ok=False)
path = next(Path('artifacts/human-driving-corrections').glob('**/trial-*-seed-117.jsonl'))
torch.set_num_threads(1)
policy, metadata = train_behavior_clone(load_demonstrations([path]), BehaviorCloningConfig(epochs=150, validation_fraction=0, loss='smooth_l1'))
save_behavior_clone(target / 'policy.pt', policy, metadata)
fit = FitnessConfig(**json.loads(Path('artifacts/evolution/phase-009.json').read_text())['config']['fitness'])
results = [evaluate_policy(flatten_parameters(policy), s, round_seconds=30, fitness_config=fit).to_dict() for s in (110,111,112,271,997)]
summary = {'name': target.name, 'training': metadata['training'], 'results': results,
           'laps_seeds': sum(r['lap_count'] > 0 for r in results),
           'mean_distance': mean(r['scored_distance_m'] for r in results),
           'fitness': .75 * mean(r['fitness'] for r in results) + .25 * min(r['fitness'] for r in results)}
(target / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ('name','laps_seeds','mean_distance','fitness')}))
