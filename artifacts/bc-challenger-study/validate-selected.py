import json
from pathlib import Path
from statistics import mean

import torch

from controllers.policy_network import load_policy_checkpoint, flatten_parameters
from training.evaluate import evaluate_policy, FitnessConfig

root = Path(__file__).resolve().parent
summaries = [json.loads(p.read_text()) for p in root.glob('*/results.json')]
winner = max(summaries, key=lambda r: (r['laps_seeds'], r['fitness']))
path = root / winner['name'] / 'policy.pt'
policy, _ = load_policy_checkpoint(path)
torch.set_num_threads(1)
fit = FitnessConfig(**json.loads(Path('artifacts/evolution/phase-009.json').read_text())['config']['fitness'])
seeds = [2021, 2022, 2023, 2024, 2025]
results = [evaluate_policy(flatten_parameters(policy), s, round_seconds=30, fitness_config=fit).to_dict() for s in seeds]
report = {'checkpoint': str(path), 'selection_rule': 'most selection seeds completing a lap, then aggregate fitness',
          'selection_result': winner, 'heldout_seeds': seeds, 'heldout_results': results,
          'heldout_lap_seeds': sum(r['lap_count'] > 0 for r in results),
          'heldout_mean_distance': mean(r['scored_distance_m'] for r in results)}
(root / 'selected.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('selection_result','heldout_results')}), flush=True)
