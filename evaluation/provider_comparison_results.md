# Day 10 Provider Comparison — REAL measured results

PROJECT_SPEC.md Section 35: "Do not fabricate benchmark numbers. Run the benchmark and report actual results." A provider with no credentials configured in this environment is reported as **not run**, never given fabricated numbers.

| Provider | Latency (mean) | Schema Validity | Groundedness | Output Quality | Cost |
|---|---|---|---|---|---|
| ollama | 11.99s | 75% | 75% | see notes below | Local |
| groq | not run | not run | not run | not run | API |
| claude | not run | not run | not run | not run | API |

## Notes

### ollama
4 scenarios run for real against this provider (config: `configs`/`.env`, real network call each time).
- Scenario 0: 21.77s — "A chest X-ray image was analyzed by a vision model, which flagged pneumonia with a probability of 0.87. The model's output is a pattern-matching result, not a clinical diagnosis, and should not be used for treatment guidance."
- Scenario 1: 9.14s — "The X-ray image shows an area of increased opacity in the lungs, which may indicate pneumonia. A pulmonary nodule is also present, but its classification as benign or malignant cannot be determined from this model."
- Scenario 2: 12.07s — "A chest X-ray image was analyzed by a vision model, which flagged a possible pneumonia with a probability of 0.12. The model's output is a pattern-matching result, not a clinical diagnosis, and should not be used for treatment guidance."
- Scenario 3: FAILED — Failed to produce a valid, grounded, safe response after 2 attempts: 1 validation error for MedicalReport
limitations
  Input should be a valid array [type=list_type, input_value='The vision model did not...alities for this image.', input_type=str]
    For further information visit https://errors.pydantic.dev/2.13/v/list_type

### groq
**Not run.** Not configured in this environment: LLM_PROVIDER=groq requires GROQ_API_KEY to be set.

### claude
**Not run.** Not configured in this environment: LLM_PROVIDER=claude requires ANTHROPIC_API_KEY to be set.