# Modeling-improvement agent report

Generated: 2026-08-31T23:40:55.228825+00:00
Project: `a02e5cc03982`
Phase: `PipelinePhase.EXTERNAL_EVALUATED`
Stopping reason: `completed`

## Outcome

Search stopped with reason `unknown`. Unsatisfied requirements: none recorded. Attempted 0 adaptive experiments over 0 iterations.

## Ranking (deterministic policy)

Hard-requirement passers rank above failers, then outer-CV R², lower error,
lower variability, smaller train–CV gap, fewer features, simpler estimator.

_No adaptive experiments were recorded._

## Diagnosis trail

- Last diagnosis: n/a
- Last hypothesis: n/a
- Stagnation count: 0

## Integrity

- dataset_hash: `8b547ba5d2a0032165ca1bfc37aad3924cde60de3ef2a0014f33a8566e8041d4`
- development_split_hash: `8bb901fedd7f8ac65e1803ae7bb4c23fe5b11a6af8c51c10adc041756b7ca1db`
- sealed_test_hash: `ec15b9508692afe5af37d20f3c1c31652eb0ce99da68e2991143bf3c1a076865`
- handoff validation passed: True


## Sealed external-test evaluation

Evaluated once after freeze. These confirmatory metrics were not used for model improvement.

```json
{'experiment_id': '7ed734693ad815e3', 'tool_name': 'evaluate_sealed_test', 'arguments': {'selected_features': ['Mordred_AATSC1d', 'Mordred_ATSC8Z', 'Mordred_GATS5c', 'Mordred_GATS7i', 'Mordred_NdsssP', 'Mordred_RNCS', 'Mordred_VSA_EState3', 'RDKit_SMR_VSA10']}, 'status': 'completed', 'metrics': {'test_r2': 0.32977316695522096, 'test_rmse': 0.8300898485496737, 'test_mae': 0.6598771709925724, 'train_r2': 0.9219840892823041, 'val_r2': 0.4131055889931107}, 'selected_features': ['Mordred_AATSC1d', 'Mordred_ATSC8Z', 'Mordred_GATS5c', 'Mordred_GATS7i', 'Mordred_NdsssP', 'Mordred_RNCS', 'Mordred_VSA_EState3', 'RDKit_SMR_VSA10'], 'artifact_paths': {'sealed_test': 'outputs/a02e5cc03982/agent_results/sealed_test'}, 'warnings': [], 'errors': [], 'runtime_seconds': None, 'parent_experiment_id': None, 'extra': {'confirmatory': True}}
```

