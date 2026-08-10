COMMON_CONSTRAINTS = """
Scientific constraints (mandatory):
- Use only the provided AgentArtifactView / agent_visible_summary evidence.
- Cite evidence with source_artifact and source_field names from the payload.
- Do not invent metrics, predictions, or statistics.
- External-test metrics, predictions, residuals, AD classifications, and scatter plots
  are unavailable and must not be requested or assumed.
- Do not recommend experiments already present in the experiment ledger digest.
- Prefer scientifically conservative QSAR reasoning.
- Admit insufficient evidence when the summary lacks required fields.
- Return structured JSON only matching the requested schema.
- Respect budget and stopping constraints in the payload.
""".strip()
