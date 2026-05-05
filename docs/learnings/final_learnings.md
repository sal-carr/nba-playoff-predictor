# Final Learnings

## What I learned about the environment

- Python 3.13.9 with all required packages pre-installed in `.venv`
- Metaflow 2.19.23 works smoothly for local execution
- Local datastore created automatically in `.metaflow/` directory
- No Outerbounds/cloud configuration present - purely local execution

## What I learned about Metaflow

### Core Concepts That Mattered Most

1. **Artifacts (`self.x`)**: Automatic persistence between steps is powerful. The model trained in `tune_model` was seamlessly available in `join_tuning` and later retrievable via Client API.

2. **`foreach` for parallel work**: The hyperparameter tuning pattern (`self.next(step, foreach="param_grid")`) elegantly handles fan-out/fan-in without explicit threading or multiprocessing code.

3. **Client API**: Loading models from past runs (`Flow("TrainNBAModelFlow").latest_successful_run`) enables clean separation between training and inference flows.

4. **Cards**: The `@card(type="blank")` decorator with `current.card.append()` creates portable evaluation reports tied to specific runs.

### Key Patterns

- **Checkpointing**: Each step is a checkpoint. If feature engineering fails, the raw data load doesn't need to repeat.
- **Artifact inheritance in joins**: After `foreach`, the join step receives `inputs` list with all branch artifacts accessible via `branch.artifact_name`.
- **Separation of concerns**: Keep orchestration in flows, transformation logic in `src/` modules.

## What I learned about Outerbounds

- Not configured in this environment
- Would require: `OUTERBOUNDS_PROFILE`, `METAFLOW_PROFILE`, etc.
- Deployment commands: `argo-workflows create` or `step-functions create`
- See `documents/outerbounds_handoff.md` for setup instructions

## Bugs and root causes

| Bug | Root Cause | Fix |
|-----|------------|-----|
| `ModuleNotFoundError: No module named 'src'` | Flows run from different working directory | Added `sys.path.insert(0, str(Path(__file__).parent.parent))` |
| Cards failed to render | Matplotlib tried to open display | Added `matplotlib.use('Agg')` before importing pyplot |
| XGBoost verbose output cluttered logs | Default verbosity | Set `verbosity=0` in XGBClassifier |
| Duplicate rows in predictions | Feature merge created duplicates for repeat matchups | Not a bug - multiple games between same teams |

## What the tutorial got right

1. **Progressive complexity**: Starting with HelloNBAFlow, then parameters, then full training flow builds understanding incrementally.

2. **Anti-leakage emphasis**: The `shift(1)` pattern for rolling features is critical and well-explained.

3. **Local-first approach**: Running locally before cloud deployment catches most issues early.

4. **Artifact-centric design**: Treating the model as a versioned artifact (not a file) is the right mental model.

5. **Card recommendation**: Evaluation cards create auditable, shareable reports without manual export.

## What the tutorial should change

1. **Data acquisition**: Tutorial assumes data exists but doesn't provide it. Should include a data acquisition section or sample data.

2. **Import paths**: Tutorial code uses `from src.xxx import` but doesn't explain how to make that work from `flows/` directory.

3. **Matplotlib backend**: Should mention `matplotlib.use('Agg')` for headless/flow execution.

4. **Validation split**: Tutorial uses random split. Should mention chronological split is better for time series sports data.

5. **Synthetic data caveat**: If using synthetic betting lines, should clearly warn that `home_cover` and `total_over` targets are artificial.

## What should be automated next

1. **Scheduled retraining**: Daily/weekly flow runs with `@schedule` decorator

2. **Model comparison**: Automated comparison of new model vs production model before promotion

3. **Data freshness checks**: Pre-flow validation that data is recent enough

4. **Prediction output routing**: Push predictions to database/API instead of CSV

5. **Alerting**: Notify when model AUC drops below threshold

6. **Feature drift detection**: Monitor feature distributions for concept drift

## Performance notes

- Training flow: ~10 seconds total (3 parallel tune_model tasks)
- Prediction flow: ~4 seconds total
- Data acquisition: ~15 seconds (rate-limited API calls)
- Model AUC ~0.59 is modest but expected for game outcome prediction with limited features
