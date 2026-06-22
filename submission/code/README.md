# qdriftforecast

Standalone, CPU-only artifact for regenerating the figures and numeric table
values used by `submission/main.tex`.

```bash
pip install -e .
qdrift-reproduce
```

The command uses fixed seeds and writes the four manuscript figures into
`submission/figures/`. The synthetic telemetry generator is included inside this
package, so reproduction does not depend on files outside the submission folder.
