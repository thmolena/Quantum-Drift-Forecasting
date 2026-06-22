# qdriftforecast

Standalone, CPU-only artifact for regenerating the figures, table bodies, and
machine-readable source data used by `submission/main.tex`.

```bash
pip install -e .
qdrift-reproduce
```

The command uses fixed seeds and writes the four manuscript figures into
`submission/figures/`, the three manuscript table bodies into
`submission/tables/`, and JSON source data into `submission/code/generated_data/`.
The synthetic telemetry generator is included inside this package, so
reproduction does not depend on files outside the submission folder. A
from-basics explanation of the computation and learning objects is in
`THEORY.txt`.
