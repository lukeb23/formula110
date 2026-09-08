# Formula 110 Evolution Handoff

Read `AGENTS.md` first. This file records transient experiment state and must be
verified against the artifacts because evolution may still be running.

## Current objective

Continue improving the evolved policy toward fitness 1600 under the current
fitness definition while keeping the car fast, precise, healthy, and consistent
across simulator seeds. A higher score caused only by changing fitness weights
does not count as behavioral improvement.

## Repository state at handoff

- Date checked: 2026-09-08 (America/New_York).
- Branch: `neurogenerative-policies`.
- Commit: `0ae69fc` (`first 80 neurogenerative policies`).
- The worktree was clean before this handoff file and the `AGENTS.md` pointer
  were added.
- Evolution artifacts and model checkpoints under `artifacts/` are ignored by
  Git and must not be deleted or force-added.
- The complete test suite passed after scoped mutation was added: 146 tests.
- Focused Ruff and strict Pyright checks passed for the changed policy and
  evolution files.

## Implemented pipeline

- Shared 12-value observation encoder: `src/controllers/observation.py`.
- Shared `12 -> 32 -> 32 -> 2` tanh policy: `src/controllers/policy_network.py`.
- Action order is `[steer, throttle]`.
- Behavior cloning, mixed BC/random initialization, graphical policy viewing,
  headless multi-seed evaluation, elitist evolution, phased continuation, and
  fitness-component recording are implemented.
- Fitness accounts for distance, laps, best-lap time, damage/health, wall
  contact, low-progress time, and worst-seed performance.
- Mutation profiles support different standard deviations and these scopes:
  `steer_output`, `throttle_output`, `output`, and `all`.
- Elites are copied exactly and cached between unchanged evaluations.
- Each checkpoint and generation manifest records its mutation standard
  deviation, scope, seed, and parent lineage.

## Active overnight experiment

The intended active run is phase 9, continuing generation 79 through generation
179 with population 50 and five elites. Phase 8 is an abandoned/incomplete
attempt from the same source and is not authoritative.

At the last inspection:

- Latest completed generation: 124 of 179.
- Phase file: `artifacts/evolution/phase-009.json`.
- Phase status: `running` with 45 completed generations.
- Best fitness: `1081.3837113307295`.
- Generation-124 checkpoint: `artifacts/evolution/generation-124/policy-000-elite.pt`.
- The actual improvement originated at generation 112 as
  `policy-015-mutated.pt`.
- That improvement used `mutation_std=0.01`, scope `throttle_output`, and parent
  index 0.

Aggregate generation-124 winner metrics across seeds 110, 111, 112, 271, 997:

- Average scored distance: 485.150 m.
- Total laps across five seeds: 10 (two per seed).
- Best lap: 10.517 s.
- Average damage: 0.00321.
- Average wall contact: 0.147 s.
- Average low-progress time: 0.290 s.
- Maximum observed speed: 28.742 m/s.

The active fitness/configuration is:

```text
population_size=50
elite_count=5
generations=100
simulator_seeds=110,111,112,271,997
round_seconds=30
worst_seed_weight=0.25
distance_weight=1.5
lap_completion_bonus=100
fast_lap_bonus=300
health_exponent=1.0
damage_penalty=0
wall_contact_penalty_per_s=20
low_progress_penalty_per_s=10
mutation_stds=0.01,0.01,0.005,0.002,0.015,0.04
mutation_scopes=steer_output,throttle_output,output,all,all,all
mutation_weights=0.23,0.23,0.20,0.19,0.10,0.05
```

The run was launched in tmux session `formula110-evolution` under
`caffeinate -is`. Do not start another evolution process against the same
`artifacts/evolution` directory while it is live.

## First checks in a fresh chat

Find the newest completed generation:

```bash
for d in artifacts/evolution/generation-*; do
  test -f "$d/results.json" && basename "$d"
done | sort -V | tail -1
```

Inspect the active phase and process:

```bash
jq '{status, completed:(.generations|length), last:.generations[-1], final_generation}' \
  artifacts/evolution/phase-009.json
tmux ls
ps -ax -o pid=,etime=,command= | rg 'racing-evolve|training.evolution'
```

Attach to the run with:

```bash
tmux attach -t formula110-evolution
```

Detach with `Ctrl-B`, then `D`. Stop deliberately with `Ctrl-C`. Closing a
MacBook lid suspends computation; reopening normally resumes it. Completed
generation results remain durable even after interruption.

## Reviewing or watching the winner

For a completed generation `NNN`, inspect `results.json`; its
`best_checkpoint` is relative to that generation directory. Watch it with:

```bash
uv run racing-watch-policy \
  artifacts/evolution/generation-NNN/policy-000-elite.pt \
  --seed 110
```

Use every training seed when reviewing metrics. Also evaluate on unseen seeds
before selecting/exporting a final deployable model.

## Fitness target context

- The current score baseline after the last fitness change was about 1050.70;
  generation 124 improved it genuinely to 1081.38 without another rescore.
- The track centerline is about 183.07 m and the current best lap is around
  10.5 s.
- A curvature-based estimate suggested roughly 8.5–9.5 s as a plausible
  near-optimal lap range; the straight-line mathematical lower bound is 6.28 s
  and is not achievable.
- Fitness 1600 likely requires close to four consistent laps in 30 seconds or
  approximately 7.5–8.0 s laps, so it may exceed the realistic physical ceiling.
- Always compare distance, lap time, health, contact, and worst-seed behavior in
  addition to the scalar fitness.

## Additional human demonstrations

Do not add generic human laps to the active evolved lineage. The present policy
is already faster and cleaner than the BC starting point; retraining toward
ordinary human actions would probably make it more cautious and would
contaminate the controlled BC-versus-random experiment.

Additional demonstrations are worthwhile only after visually identifying a
repeatable failure state (for example, a particular corner or wall-recovery
sequence). Collect targeted corrective demonstrations, train/evaluate them as a
separate challenger, and preserve the current evolved elites. Injecting such a
challenger into the active population is not implemented and should be treated
as a new experimental phase, not as an invisible recentering operation.

## Next decision point

Let phase 9 finish unless it is intentionally stopped. Then:

1. Identify every generation that established a new best and its mutation
   scope.
2. Compare generation 179 with generation 79 using physical metrics.
3. Watch the final elite on all five training seeds and several unseen seeds.
4. If fitness plateaued far below 1600, determine whether the limit is physical,
   fitness-related, or representational before changing weights again.
5. Export a final policy only through an explicit, non-overwriting export step.

