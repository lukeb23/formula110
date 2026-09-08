# Formula 110 Evolution Handoff

Read `AGENTS.md` first and verify this snapshot against live artifacts.

## Latest focused push — finished; user controls further runs

User requested another real improvement attempt and reduction to 20 vehicles.
Then explicitly requested no further automatic iterations after this attempt;
provide the uv command for them to run. Respect this instruction.

Outcome: `artifacts/focused-push/outcome.json`. No improvement over
1081.3837113307295; the 1600 goal was NOT reached. Source controller and ZIP
were not replaced. Do not present changed mutation settings as achieved gains.

First 24-policy attempt was interrupted to honor population 20. Then two
20-policy small-mutation generations completed (36 new candidates), followed by
six 20-policy coherent Gaussian mutation generations (109 new candidates).
All 145 completed new candidates were evaluated on the same five seeds and
fitness. No new winner, so no new held-out comparison/export was warranted.
Completed results: `artifacts/focused-push/population-20` and `directed-20`.
Incomplete folders from stopped diagnostics are preserved, not completed results.

The coherent mutation script explores output gains, throttle bias and first-layer
speed/yaw sensitivity within the shared architecture. Profile determinism,
finite/bounded outputs and unchanged source parameters were checked. It has
CLI options for output, generation count, mutation seed and source checkpoint.
It always uses 20 policies and refuses to overwrite its output directory.

Prepared command for the USER to launch; not launched by the agent:

```bash
uv run python artifacts/focused-push/directed.py \
  --output artifacts/evolution-next-20 \
  --generations 25 \
  --training-seed 91408
```

This starts from the same verified champion with new mutation samples. It is
not a proven plateau fix. Finished diagnostic process exited normally; no
additional evolution run was started after it.

## Deployment ready — latest update

User needs the controller soon. Verified challenger generations 000–012 retain
the same 1081.3837113307295 champion. Exported the byte-identical generation-124
champion to `src/controllers/model/champion-1081.pt` and added runtime factory
`controllers.evolved_controller`, using shared encoder/network on CPU.
Submission archive: `artifacts/formula110-evolved-champion-1081.zip`.
Watch: `uv run racing --student-module controllers.evolved_controller --seed 110`.
Two laps on each of seeds 110,2021,2022,2023,2024,2025 in 30 seconds. Seed 2022
damage was .4654 and seed 2024 .2350: robustness remains imperfect.
Rounded validation and provenance: `src/controllers/model/champion-1081.json`.
1000 finite/bounded ticks, independent factories, no per-tick reload, CPU/eval,
and extracted ZIP loading from an unrelated working directory verified.
Local peak RSS before simulator 215.31 MiB; with simulator 289.80 MiB. This is
not an official isolated-worker memory certification. 148 tests passed;
focused Ruff and Pyright passed. Evolution process was not stopped in this turn.
This deployment update supersedes earlier statements below about no export.

## Current launch decision — supersedes older preparation below

Launch recovery: user attempted the challenger launch; `experiment.json` exists
with zero completed generations. No evolution process was live at inspection.
Fixed `run_evolution --resume` to reuse the initial population when zero
generations completed, and updated the launch script to pass `--resume` when
its experiment manifest exists. No artifacts were deleted or restarted.
The fix does not address interruptions after later populations are created;
inspect such cases before assuming the existing resume path handles them.

The user authorized the BC challenger study and preparation of the next run.
Study scripts, checkpoints and measured results are in
`artifacts/bc-challenger-study/`. Seven candidates compared training duration,
trial combinations and MSE versus Smooth L1 loss. Selection used number of
seeds completing laps, then original aggregate fitness. All use the shared
12/32/32/2 model and encoder. Training used seed 110 and no action holdout;
simulator validation is separate. Do not interpret training loss as validation.

Selected checkpoint: `artifacts/bc-challenger-study/trial117-150/policy.pt`.
Trial seed 117 alone, 150 epochs, MSE; 4/5 selection seeds complete laps,
mean scored distance 220.497 m, aggregate fitness 378.865.
Fresh held-out seeds 2021–2025: 4/5 complete laps, mean distance 215.666 m.
Full selection and held-out metrics: `artifacts/bc-challenger-study/selected.json`.
The checkpoint is still much slower than evolved generation 124. Its role is a
validated development challenger, not a replacement champion.

Prepared, NOT launched: `artifacts/evolution-bc-challenger-20260908/generation-000`.
50 policies: five exact generation-124 elites, 40 evolved mutations, one exact
selected BC model and four independent BC mutations. Parameter equality of
elites, uniqueness of BC vectors and all checkpoint hashes were verified.
This isolated experiment uses local generation numbers 0–24, with source
generation 124 recorded in its manifest. Original artifacts remain preserved.
Standard selection applies after its first evaluation; BC survival is not
guaranteed. No recurring BC injection or fitness reweighting was added.

Use this command instead of the older development launch command below:

```bash
zsh artifacts/bc-challenger-study/run-next-generation.sh
```

The launch runs under caffeinate, with one numerical-library thread, five
evaluation seeds and the same fitness as phase 9. `plan.json` records full
configuration/provenance. It is a development continuation, not the controlled
BC-versus-random experiment. No competitive win-rate evaluation is claimed.

Training now has an optional `--loss smooth_l1` (beta .5); default remains MSE.
MSE metrics retain their meaning under either loss. The alternative loss did
not outperform the selected MSE candidate. All 147 tests passed; focused Ruff
and strict Pyright passed. No runtime controller or simulator source changed.

Earlier claims that temporal context or conflicting actions were proven causes
were too strong: attenuated outputs and rollout failures are observed, but
nearest-neighbor comparisons used different distance distributions and training
budgets differed across dataset sizes. Those causal explanations remain hypotheses.

## Latest decision (2026-09-08)

The user intends to stop phase 9 without waiting for generation 125 results.
Use generation 124 as the measured baseline. Never extrapolate missing results.
The agent stopped the run at the user’s request and closed tmux session
`formula110-evolution`. Worker 54862 and launcher 54860 have exited.
Phase 9 is interrupted; generation 125 was not completed.

## Verified baseline

- Original results: `artifacts/evolution/generation-124/results.json`.
- Champion: `policy-000-elite.pt`, fitness 1081.3837113307295.
- Best originated at generation 112; no further improvement through 124.
- Five seeds: 110,111,112,271,997; 30 seconds; two laps per seed.
- Mean distance 485.150 m; best lap across seeds 10.517 s, not a typical lap.
- Original generation 125 had no results at inspection and was not used.
- Original checkpoints and runtime export are unchanged; phase 9 metadata was marked interrupted.

## Updated demonstrations and BC validation

Latest update: ten accepted sessions on seeds 110–119, 23,461 rows.
New run `run-20260908T142607Z-35b39e86` contains five completed trials on
seeds 115–119 (13,248 new samples). Original trials 2 and 4 remain excluded
outside the training root in `artifacts/excluded-human-driving`.

Latest artifacts: `artifacts/bc-corrections-20260908-v2/`:
- `dataset-manifest.json`: accepted files, sessions, seeds and SHA-256 hashes.
- `imitation_model.pt`: new, separately trained ten-trial BC checkpoint.
- `training.json`: shared architecture/encoder and full training provenance.
- `evaluation.json`: five-seed, 30-second direct simulator validation.
- `fit-diagnostics.json`: per-action fitting errors and split caveat.
- `summary.json`: validation outcome and no-injection decision.

Training uses existing action-MSE path, seed 110, 50 epochs, batch 256,
learning rate .001, session validation fraction .2. 20,192 training rows and
3,269 validation rows; train MSE .12449, validation MSE .19203.
The validation sessions changed when the dataset grew, so v1/v2 validation
MSEs are not a fixed-holdout comparison.

Measured outcome: zero laps on all five seeds 110,111,112,271,997;
mean scored distance 55.661 m versus 99.101 m for the five-trial BC model.
No BC injection or runtime export. More data did not resolve the BC failure;
debug fitting and first rollout errors before collecting more or evolving it.
This is not evidence that the human recordings themselves are bad.
Previous five-trial artifacts remain at `artifacts/bc-corrections-20260908`.

## Prepared development continuation (NOT launched)

Isolated root: `artifacts/evolution-development-20260908`.
A full byte-verified copy of generation 124 is the source population.
Run after the old evolution process stops:

```bash
zsh artifacts/evolution-development-20260908/run-next-phase.sh
```

25 generations (125 through 149), population 50, five exact elites.
Architecture, seeds, duration, fitness, mutation scopes and scales are unchanged.
Mutation allocation changes from .23,.23,.20,.19,.10,.05 to
.20,.25,.25,.25,.05,0 for scopes/scales:
steer_output/.01, throttle_output/.01, output/.005, all/.002, all/.015, all/.04.
This is an unvalidated development experiment, not a promise of bigger gains.

Evidence and configuration: `development-plan.json` in that root.
Across generations 80–124, full-network std .015 and .04 yielded no elite-beating
candidates in 315 evaluations. Retain a small .015 exploration allocation;
remove .04 for this short pilot. Doubling/tripling fast-lap reward on saved
candidate metrics retains the same champion, so fitness weights are unchanged.

Fitness: distance 1.5, lap bonus 100, fast-lap bonus 300, health exponent 1,
damage penalty 0, wall contact penalty 20/s, low-progress penalty 10/s;
aggregate .75 mean + .25 worst seed. Scores are comparable to phase 9.

The user does not want to build competitive controllers. Current evaluation is
against a stationary benchmark and does not measure competitive win rate.
Do not describe the continuation as optimizing measured winningness.

## Boundaries and next checks

No simulator, runtime controller, policy architecture, or training source changed.
No BC was injected and no final controller exported. Preserve original lineage.
Assess the pilot using physical metrics and unseen seeds before claiming progress.
This continuation is not the controlled BC-versus-random MVP comparison.
Earlier handoff is archived at
`artifacts/bc-corrections-20260908/previous-handoff.md`.
