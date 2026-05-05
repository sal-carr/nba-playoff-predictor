# Outerbounds Deployment Handoff

## Overview

This document describes how to deploy the NBA prediction pipeline to Outerbounds after local development is complete.

## Prerequisites

1. Outerbounds account (sign up at https://outerbounds.com)
2. AWS account (if using AWS-backed infrastructure)
3. Docker installed locally (for building images)

## Environment Variables

Configure these in your shell or `.env` file:

```bash
# Outerbounds configuration
export OUTERBOUNDS_PROFILE=default
export METAFLOW_PROFILE=outerbounds

# Metaflow service URLs (provided by Outerbounds)
export METAFLOW_SERVICE_URL=https://your-outerbounds-instance.outerbounds.dev
export METAFLOW_METADATA_SERVICE_URL=https://your-outerbounds-instance.outerbounds.dev

# AWS configuration (if using S3-backed storage)
export AWS_PROFILE=your-aws-profile
export AWS_DEFAULT_REGION=us-west-2
```

## Setup Steps

### 1. Install Outerbounds CLI

```bash
pip install outerbounds
```

### 2. Configure Outerbounds

```bash
outerbounds configure
```

Follow the prompts to authenticate and set up your workspace.

### 3. Verify Configuration

```bash
# Check that Metaflow can connect to Outerbounds
python -c "from metaflow import Flow; print('Connected!')"

# List existing flows (should work if connected)
python -c "from metaflow import Metaflow; print(list(Metaflow()))"
```

## Deployment Commands

### Deploy Training Flow to Argo Workflows

```bash
python flows/train_nba_model.py --with retry argo-workflows create
```

### Deploy with Scheduling

Add the `@schedule` decorator to the flow class:

```python
from metaflow import schedule

@schedule(daily=True)
class TrainNBAModelFlow(FlowSpec):
    ...
```

Then deploy:

```bash
python flows/train_nba_model.py --with retry argo-workflows create
```

### Trigger a Manual Run

```bash
python flows/train_nba_model.py argo-workflows trigger --target home_win
```

### Deploy Prediction Flow

```bash
python flows/predict_nba_model.py --with retry argo-workflows create
```

## Cloud Compute Configuration

### Add Remote Execution Decorators

For steps that should run on cloud compute, add decorators:

```python
from metaflow import kubernetes, resources, pypi

@kubernetes
@resources(cpu=4, memory=16000)
@pypi(packages={
    "xgboost": "2.1.0",
    "pandas": "2.2.3",
    "scikit-learn": "1.5.2"
})
@step
def tune_model(self):
    ...
```

### Run with Cloud Compute

```bash
python flows/train_nba_model.py run --with kubernetes
```

## Alternative: AWS Step Functions

If using self-hosted Metaflow with AWS:

```bash
# Deploy
python flows/train_nba_model.py --with retry step-functions create

# Trigger
python flows/train_nba_model.py step-functions trigger --target home_win
```

## Monitoring

### Outerbounds UI

Access the Outerbounds dashboard to:
- View deployment status
- Monitor run history
- Inspect artifacts
- View cards
- Compare runs

### CLI Commands

```bash
# List recent runs
python flows/train_nba_model.py show

# View logs for a specific run
python flows/train_nba_model.py logs RUN_ID/step_name

# View card
python flows/train_nba_model.py card view join_tuning
```

## Production Checklist

- [ ] Configure environment variables
- [ ] Verify Outerbounds connection
- [ ] Test run with `--with kubernetes` locally first
- [ ] Deploy training flow
- [ ] Set up scheduling
- [ ] Deploy prediction flow
- [ ] Configure flow chaining with `@trigger_on_finish` if needed
- [ ] Set up alerting for failed runs
- [ ] Tag production runs for model retrieval

## Code Changes for Cloud

No code changes are required for basic deployment. The same flows that run locally will run on Outerbounds.

For production optimization:
1. Add `@pypi` decorators to pin dependencies
2. Add `@resources` decorators for appropriate compute allocation
3. Consider adding `@retry` decorators for fault tolerance
4. Use `@trigger_on_finish` to chain training → prediction flows

## Troubleshooting

### "No module found" errors in remote execution
Add dependencies to `@pypi` decorator or ensure they're in your Docker image.

### Artifact not found
Check that you're using `latest_successful_run` not `latest_run`.

### Connection refused
Verify `METAFLOW_SERVICE_URL` is correct and accessible.

### Permission denied
Check AWS credentials and Outerbounds workspace permissions.
