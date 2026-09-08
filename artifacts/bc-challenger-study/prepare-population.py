"""Create a separate development population with explicit BC provenance."""
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from controllers.observation import observation_metadata
from controllers.policy_network import (
    PolicyNetwork, checkpoint_payload, flatten_parameters, load_parameter_vector,
    load_policy_checkpoint, parameter_mutation_mask, policy_metadata,
)
from training.evaluate import FitnessConfig, FitnessResult
from training.evolution import CandidateEvaluation, EvolutionConfig, create_next_generation

study = Path(__file__).resolve().parent
selection = json.loads((study / 'selected.json').read_text())
assert selection['heldout_lap_seeds'] >= 4
source = Path('artifacts/evolution/generation-124')
destination = Path('artifacts/evolution-bc-challenger-20260908')
destination.mkdir(exist_ok=False)
initial = destination / 'generation-000'
fit = FitnessConfig(**json.loads(Path('artifacts/evolution/phase-009.json').read_text())['config']['fitness'])
config = EvolutionConfig(generations=25, elite_count=5, mutation_std=.002,
    mutation_stds=(.01,.01,.005,.002,.015,.04),
    mutation_scopes=('steer_output','throttle_output','output','all','all','all'),
    mutation_weights=(.20,.25,.25,.25,.05,0), simulator_seeds=(110,111,112,271,997),
    round_seconds=30, worst_seed_weight=.25, fitness=fit)
raw = json.loads((source / 'results.json').read_text())['ranking']
ranked = [CandidateEvaluation(index=r['index'], checkpoint=r['checkpoint'], origin=r['origin'],
    parent_index=r['parent_index'], fitness=r['fitness'],
    seed_results=tuple(FitnessResult(**s) for s in r['seed_results'])) for r in raw]
create_next_generation(ranked, source_dir=source, output_dir=initial,
    generation_index=0, population_size=45, config=config)
manifest = json.loads((initial / 'manifest.json').read_text())
bc_path = Path(selection['checkpoint'])
bc, _ = load_policy_checkpoint(bc_path)
base = flatten_parameters(bc)
for offset, (scope, std) in enumerate((('all',0),('output',.005),('output',.005),('all',.002),('all',.002))):
    index = 45 + offset
    seed = 91000 + index
    policy = PolicyNetwork()
    vector = base.clone()
    if std:
        vector += torch.randn(vector.shape, generator=torch.Generator().manual_seed(seed)) * std * parameter_mutation_mask(policy,scope)
    load_parameter_vector(policy,vector)
    name = f'policy-{index:03d}-bc-challenger.pt'
    metadata = {**policy_metadata(), 'observation':observation_metadata(),
        'generation':0, 'population_index':index, 'origin':'bc_challenger' if not std else 'bc_mutated',
        'parent_index':None, 'source_checkpoint':str(bc_path), 'mutation_seed':seed if std else None,
        'mutation_std':std, 'mutation_scope':scope if std else 'none'}
    torch.save(checkpoint_payload(policy,metadata=metadata),initial/name)
    manifest['candidates'].append({'index':index,'path':name,'origin':metadata['origin'],
        'parent_index':None,'source_checkpoint':str(bc_path),'mutation_seed':metadata['mutation_seed'],
        'mutation_std':std,'mutation_scope':metadata['mutation_scope'],
        'sha256':hashlib.sha256((initial/name).read_bytes()).hexdigest()})
manifest.update(population_size=50, source_generation=124, source_population=str(source.resolve()),
    purpose='Development continuation with one validated BC challenger and four independent perturbations')
(initial/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for i in range(5):
    assert torch.equal(flatten_parameters(load_policy_checkpoint(initial/manifest['candidates'][i]['path'])[0]),
                       flatten_parameters(load_policy_checkpoint(source/ranked[i].checkpoint)[0]))
bc_vectors = [flatten_parameters(load_policy_checkpoint(initial/c['path'])[0]).numpy().tobytes() for c in manifest['candidates'][45:]]
assert len(set(bc_vectors)) == 5
for entry in manifest['candidates']:
    assert hashlib.sha256((initial/entry['path']).read_bytes()).hexdigest()==entry['sha256']
plan={'status':'prepared_not_started','config':asdict(config),'initial_population':str(initial.resolve()),
    'source_generation':124,'local_generation_numbering':'0–24; development continuation of generation 124',
    'composition':{'evolved_elites':5,'evolved_mutations':40,'exact_bc':1,'bc_mutations':4},
    'bc_selection':selection,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'note':'Fitness unchanged. BC survival is not guaranteed. This is not the controlled BC-vs-random MVP experiment.'}
(destination/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Verified 50 checkpoints, five unchanged elites and five distinct BC vectors:',initial)
