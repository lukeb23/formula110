# Formula 110 Agent Instructions

For the current long-running experiment and cross-chat working state, read
`EVOLUTION_HANDOFF.md` after this file and verify its snapshot against the live
artifacts before acting.

## Authority and scope

- Follow the user's latest explicit instructions first, then the existing codebase and public Formula 110 interfaces, then this file.
- If documentation conflicts with working repository behavior, inspect the relevant code and tests before deciding. State any material discrepancy.
- Do not modify the simulator or depend on private simulator internals. Build against the public controller, sensor, recording, and headless-evaluation interfaces.
- Keep the MVP focused on proving whether behavior-cloning initialization accelerates neuroevolution. Do not add unrelated algorithms or infrastructure.

## Project objective

Implement and evaluate this pipeline:

```text
human demonstrations
  -> behavior cloning
  -> one imitation policy
  -> initial population (50% independently mutated BC policies, 50% random policies)
  -> neuroevolution
  -> best policy
  -> deployable Formula 110 controller
```

The primary experiment compares:

1. `10 random policies -> evolution`
2. `5 BC-derived policies + 5 random policies -> evolution`

Hold architecture, observations, preprocessing, population size, mutation settings, generation count, simulator seeds, evaluation budget, and fitness definition constant between experiments.

## Non-negotiable invariants

There must be exactly one authoritative definition of each of the following:

- observation fields and ordering
- preprocessing, clipping, and normalization constants
- policy-network architecture
- action ordering

Behavior-cloning training, evolutionary evaluation, and the deployed controller must import and use those shared definitions. Never copy or reimplement them in separate paths.

Use one explicit action convention everywhere. Prefer `[steer, throttle]` internally because that is the training-plan convention, then construct `RobotCommand` with named arguments. If the repository already establishes another convention, follow it consistently and document it next to the shared definition.

Every controller output must be finite and clamped to `[-1, 1]`.

## MVP observation and preprocessing

Start with a fixed 12-scalar observation vector:

1. speed in m/s
2. wall-LiDAR left
3. wall-LiDAR front-left
4. wall-LiDAR front
5. wall-LiDAR front-right
6. wall-LiDAR right
7. camera center offset in meters
8. camera heading error in degrees
9. camera lookahead offset 0
10. camera lookahead offset 1
11. camera lookahead offset 2
12. IMU yaw rate in degrees/s

Implement it in one shared function such as `encode_observation(sensors) -> np.ndarray`.

- Replace LiDAR `inf`/no-hit values with a configured cap (initially 100 m), then scale by that cap.
- Clip physically extreme values before normalization.
- Scale heading error by 180.
- Clip and scale yaw rate using one named configuration constant.
- Reject or safely handle missing, NaN, or infinite values so the returned vector is always fixed-size, finite, normalized, and `float32`.
- Do not initially use world heading, raw camera images, recurrent state, or additional sensors.

## Shared policy

Use one small MLP for behavior cloning, evolution, and deployment:

```text
12 -> 32 -> ReLU -> 32 -> ReLU -> 2 -> tanh
```

Do not create separate model definitions for different stages. Keep parameter flattening/unflattening deterministic and verify a round trip preserves every parameter value and shape.

## Training and runtime boundaries

- Put training, datasets, experiment orchestration, evolutionary logic, and training-only dependencies outside the deployable controller code (normally under `training/`).
- Put shared runtime policy/observation code and final model artifacts under `src/controllers/` or the repository's established equivalent.
- The final controller must expose the repository-required `create_controller()` factory so each car/race gets independent controller state.
- Load the model once per controller creation, never once per 60 Hz tick.
- Load weights on CPU, call `.eval()`, and use `torch.inference_mode()` during inference.
- Resolve model artifacts relative to the controller module, not the process working directory.
- Respect CPU-only inference and the 512 MiB resident-memory limit. Do not package demonstrations, experiment logs, or other training-only data with the runtime controller.

## Human demonstrations and behavior cloning

- Preserve append-only JSONL recording semantics and separate trajectories by `session_id`.
- Treat each command as corresponding to the sensor snapshot immediately before the action was applied.
- Collect roughly 5–10 competent laps across multiple starting seeds for the MVP.
- Parse recordings through the same shared observation encoder used at runtime.
- Train the shared policy with simple action MSE initially. Avoid premature imitation-learning optimization.
- Save the imitation checkpoint and enough metadata/configuration to reproduce its observation, preprocessing, architecture, and action conventions.

Before implementing or running evolution, validate the BC policy directly in Formula 110. It must respond to track geometry, remain on track for a meaningful interval (ideally a lap), and emit only finite, bounded commands. If it cannot drive, debug encoding, normalization, action order, data parsing, checkpoint loading, and network outputs first. Do not compensate for a broken BC path with evolution or reward shaping.

## Neuroevolution

Default MVP configuration:

- population: 10
- initial mixed population: 5 independently perturbed BC policies and 5 independently initialized random policies
- elites: configurable, initially 2 or 3
- generations: configurable, initially 25–100
- Gaussian mutation standard deviation: configurable, initially 0.05
- early fitness: scored track distance

BC-derived members must use independent Gaussian perturbations; do not insert five identical BC copies. Random policies must use the identical architecture and encoder but independent initialization.

Use a simple loop: evaluate, rank, retain elites, mutate survivors to refill the population, repeat. Do not add crossover, PPO, SAC, TD3, NEAT, CMA-ES, MAP-Elites, novelty search, topology evolution, multi-agent evolution, or elaborate reward shaping for the MVP.

Prefer deterministic solo evaluation. Expose a narrow interface such as `evaluate_policy(weights, seed) -> FitnessResult` that instantiates the shared policy, loads candidate weights, runs public headless simulation facilities, and returns metrics. Add head-to-head evaluation only after solo driving works.

Fresh random-policy injection is optional and should be added only if results show premature convergence.

## Fitness and experiments

Use scored track distance as the initial fitness because it provides selection pressure before lap completion. Only after policies reliably complete laps should lap completion, lap time, collision, or damage terms be considered.

Track at least:

- initial best fitness
- best and aggregate fitness by generation
- final best fitness
- time/generation to first completed lap
- final lap time
- scored/raw distance
- damage/contact metrics

Do not use human lap time as the behavior-cloning fitness baseline. Demonstrations are supervised state/action data.

Evaluate final claims across multiple simulator seeds. One seed is useful for debugging but insufficient evidence.

## Reproducibility and artifacts

Explicitly seed every independent source of randomness, including Python where applicable, NumPy, PyTorch, population initialization/mutation, and the simulator. A simulator seed does not seed the training stack.

Record with every experiment:

- code/config version
- training random seed and simulator seeds
- population size, elite count, generations, and mutation standard deviation
- architecture and parameter count
- observation ordering and normalization constants
- fitness definition
- checkpoint paths and evaluation results

Never silently overwrite the best checkpoint. Save generation/experiment metadata alongside it and export the selected final weights to the runtime model location only through an explicit export step.

## Suggested structure

Follow existing repository conventions where present; otherwise prefer:

```text
src/controllers/
  evolved_controller.py
  policy_network.py
  observation.py
  model/best_policy.pt
training/
  __init__.py
  dataset.py
  behavior_clone.py
  evaluate.py
  evolution.py
artifacts/
  human-driving.jsonl
  imitation_model.pt
  evolution/
```

Generated recordings, checkpoints, and experiment outputs should not be committed unless the repository explicitly tracks them. Commit small configuration/metadata examples when useful.

## Implementation order

Work in this dependency order and do not skip validation gates:

1. Shared observation encoder and finite/shape tests.
2. Shared policy network and bounded-output tests.
3. Model-backed controller and simulator smoke test.
4. Human recording and JSONL dataset loader.
5. Behavior-cloning training.
6. Direct BC-policy simulator validation.
7. Parameter flatten/unflatten utilities and round-trip tests.
8. Random initialization and independent BC perturbation.
9. Automated solo policy evaluation.
10. Selection, elitism, and Gaussian mutation loop.
11. Random-only and mixed-population experiments under identical conditions.
12. Best-policy export and multi-seed evaluation.
13. Optimization or optional extensions only after the complete MVP works.

## Testing and completion discipline

- Inspect the repository README, `src/racing/student/api.py`, relevant sensor definitions, recording format, and public evaluation APIs before implementing integrations.
- Add focused unit tests for observation shape/order/finite values, LiDAR infinity handling, normalization boundaries, action mapping, output clamping, checkpoint loading, and parameter-vector round trips.
- Add a smoke test that loads the final controller on CPU and executes repeated ticks without recreating the model.
- Use deterministic tiny configurations for evolutionary tests; do not require long full experiments in the normal test suite.
- Run the narrowest relevant tests while iterating, then the repository's broader validation suite before handoff.
- Preserve unrelated user changes in a dirty worktree. Do not perform destructive git operations.
- Do not claim the research hypothesis is supported merely because the pipeline runs. Report the controlled comparison and uncertainty honestly.

## MVP definition of done

The MVP is complete only when human JSONL data trains a BC policy; that BC policy drives meaningfully; BC and random policies seed comparable populations; candidates are evaluated automatically; evolution improves recorded fitness; the controlled random-only versus mixed experiment runs; and the exported best policy loads as a normal CPU Formula 110 controller within runtime limits.
