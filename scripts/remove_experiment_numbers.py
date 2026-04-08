"""Remove Experiment N numbering from all three notebooks, replacing with natural descriptive references."""
import json, pathlib

NOTEBOOKS_DIR = pathlib.Path(__file__).resolve().parent.parent / "notebooks"

def rep(src: str, old: str, new: str) -> str:
    if old not in src:
        print(f"    WARN: pattern not found: {old[:80]!r}")
        return src
    return src.replace(old, new)

# ──────────────────────────────────────────────────────────────
# rnn_drift_forecast.ipynb
# ──────────────────────────────────────────────────────────────
RNN_PATCHES = {
    # H1 title
    "# Experiment 1 — Recurrent Architectures for Drift Forecasting and Anomaly Detection in Quantum Hardware Telemetry":
        "# Recurrent Architectures for Drift Forecasting and Anomaly Detection in Quantum Hardware Telemetry",

    # Abstract
    "This notebook constitutes the first of three experiments evaluating recurrent":
        "This notebook is the first study in a three-part benchmark evaluating recurrent",

    # Role within benchmark
    "Experiments 2 and 3 extend this evaluation to long-context attention on periodic calibration signals and to a cross-domain comparison across three heterogeneous regimes.":
        "The transformer calibration study and the cross-domain benchmark extend this evaluation to long-context attention on periodic calibration signals and to a cross-domain comparison across three heterogeneous regimes.",

    # QC contribution cell header
    "## Quantum Computing Contribution — Experiment 1":
        "## Quantum Computing Contribution",

    # Statistical comparison cell header
    "## Statistical Comparison — Experiment 1 Results":
        "## Statistical Comparison — Recurrent Architecture Results",

    # Technical interpretation forward-reference
    "Experiment 3 will test whether this advantage survives regime change across multiple datasets.":
        "The cross-domain benchmark will test whether this advantage survives regime change across multiple datasets.",

    # Key Takeaways header
    "## 9. Key Takeaways — Experiment 1":
        "## 9. Key Takeaways",

    # Key Takeaways forward-references
    "Experiment 2 asks whether self-attention provides additional gains on a different and more periodic regime. Experiment 3 determines whether either advantage survives cross-domain generalization.":
        "The transformer calibration study asks whether self-attention provides additional gains on a different and more periodic regime. The cross-domain benchmark determines whether either advantage survives cross-domain generalization.",
}

# ──────────────────────────────────────────────────────────────
# transformer_calibration.ipynb
# ──────────────────────────────────────────────────────────────
TRANSFORMER_PATCHES = {
    # H1 title
    "# Experiment 2 — Transformer-Based Calibration and Anomaly Ranking for Quantum Hardware Operational Signals":
        "# Transformer-Based Calibration and Anomaly Ranking for Quantum Hardware Operational Signals",

    # Abstract
    "This notebook constitutes the second of three experiments evaluating Transformer-based":
        "This notebook is the second study in a three-part benchmark evaluating Transformer-based",

    # Role within benchmark
    "Experiment 1 establishes the recurrent baseline for incident-aware thermal drift detection. This experiment determines whether replacing recurrence with self-attention changes the preferred evaluation objective when the signal has periodic structure: it does — the Transformer's strongest result is calibration and ranking quality, not single-threshold F1. Experiment 3 then tests whether this architectural preference survives across diverse hardware signal regimes.":
        "The recurrent architecture study establishes the baseline for incident-aware thermal drift detection. This notebook determines whether replacing recurrence with self-attention changes the preferred evaluation objective when the signal has periodic structure: it does — the Transformer's strongest result is calibration and ranking quality, not single-threshold F1. The cross-domain benchmark then tests whether this architectural preference survives across diverse hardware signal regimes.",

    # QC contribution cell header
    "## Quantum Computing Contribution — Experiment 2":
        "## Quantum Computing Contribution",

    # Statistical comparison cell header
    "## Statistical Comparison — Experiment 2 Results":
        "## Statistical Comparison — Transformer Calibration Results",

    # Technical interpretation forward-reference
    "Experiment 3 tests whether this advantage persists in a cross-domain setting.":
        "The cross-domain benchmark tests whether this advantage persists across heterogeneous signal regimes.",

    # Key Takeaways header
    "## 9. Key Takeaways — Experiment 2":
        "## 9. Key Takeaways",

    # Key Takeaways forward-references
    "Combined with Experiment 1's recurrent result, the two experiments now set up the cross-domain question: does either advantage survive regime change?":
        "Combined with the recurrent study's result, these two notebooks now set up the cross-domain question: does either advantage survive regime change?",

    # Benchmark discipline cross-reference
    "The same chronological split, weighted objective, and MC-Dropout uncertainty protocol from Experiment 1 are reused here, enabling a fair cross-architecture comparison of the experiments when reading the full benchmark in Experiment 3.":
        "The same chronological split, weighted objective, and MC-Dropout uncertainty protocol from the recurrent study are reused here, enabling a fair cross-architecture comparison across all three notebooks.",
}

# ──────────────────────────────────────────────────────────────
# quantum_drift_combined.ipynb
# ──────────────────────────────────────────────────────────────
COMBINED_PATCHES = {
    # H1 title
    "# Experiment 3 — Cross-Domain Benchmarking for Objective-Aware Architecture Selection in Quantum Hardware Monitoring":
        "# Cross-Domain Benchmarking for Objective-Aware Architecture Selection in Quantum Hardware Monitoring",

    # Abstract
    "This notebook constitutes the third and final experiment, conducting a cross-domain model selection study":
        "This notebook is the third and final study in the benchmark, conducting a cross-domain model selection study",

    # Distinguishing question cross-refs
    "Experiments 1 and 2 are model-centric, evaluating specific architectural advantages on targeted signal types.":
        "The recurrent architecture study and the transformer calibration notebook are model-centric, evaluating specific architectural advantages on targeted signal types.",

    "This experiment is benchmark-centric: it asks whether the architectural advantages identified earlier — GRU's forecast accuracy gain on thermal drift signals, the Transformer's calibration superiority on periodic calibration-like data — survive when signal periodicity, anomaly density, and background volatility all change simultaneously, as they do across different quantum hardware subsystems.":
        "This notebook is benchmark-centric: it asks whether the architectural advantages identified in the earlier studies — GRU's forecast accuracy gain on thermal drift signals, the Transformer's calibration superiority on periodic calibration-like data — survive when signal periodicity, anomaly density, and background volatility all change simultaneously, as they do across different quantum hardware subsystems.",

    # Role within benchmark
    "This experiment synthesizes evidence from Experiments 1 and 2 into a unified cross-domain model-selection argument directly applicable to quantum hardware monitoring pipeline design.":
        "This notebook synthesizes evidence from the recurrent architecture study and the transformer calibration study into a unified cross-domain model-selection argument directly applicable to quantum hardware monitoring pipeline design.",

    # QC contribution cell header
    "## Quantum Computing Contribution — Experiment 3":
        "## Quantum Computing Contribution",

    # Statistical comparison cell header
    "## Statistical Comparison — Experiment 3 Cross-Domain Results":
        "## Statistical Comparison — Cross-Domain Benchmark Results",

    # Key Takeaways header
    "## 9. Key Takeaways — Experiment 3":
        "## 9. Key Takeaways",

    # Key Takeaways role-within-paper paragraph
    "**Role within the paper.** Experiment 3 is the synthesis chapter of the benchmark. Experiment 1 ([rnn_drift_forecast.ipynb](rnn_drift_forecast.ipynb)) established the gated-recurrent baseline on a single thermal dataset. Experiment 2 ([transformer_calibration.ipynb](transformer_calibration.ipynb)) demonstrated that attention-based calibration improves anomaly-ranking quality on a cloud-infrastructure series. Experiment 3 asks whether either advantage is preserved when the evaluation is extended to three datasets with heterogeneous structure. The answer — that it depends on the objective — is the benchmark's headline result and the motivation for the paper's title.":
        "**Role within the paper.** This notebook is the synthesis chapter of the benchmark. The recurrent architecture study ([rnn_drift_forecast.ipynb](rnn_drift_forecast.ipynb)) established the gated-recurrent baseline on a single thermal dataset. The transformer calibration study ([transformer_calibration.ipynb](transformer_calibration.ipynb)) demonstrated that attention-based calibration improves anomaly-ranking quality on a cloud-infrastructure series. This cross-domain benchmark asks whether either advantage is preserved when the evaluation is extended to three datasets with heterogeneous structure. The answer — that it depends on the objective — is the benchmark's headline result and the motivation for the paper's title.",

    # Benchmark discipline
    "Any architecture advantage observed in Experiment 1 or 2 that does not replicate in Experiment 3 is evidence of overfitting to a specific dataset or operating regime, not of genuine architectural superiority.":
        "Any architecture advantage observed in the earlier two studies that does not replicate in the cross-domain evaluation is evidence of overfitting to a specific dataset or operating regime, not of genuine architectural superiority.",
}

JOBS = [
    ("rnn_drift_forecast.ipynb", RNN_PATCHES),
    ("transformer_calibration.ipynb", TRANSFORMER_PATCHES),
    ("quantum_drift_combined.ipynb", COMBINED_PATCHES),
]

for fname, patches in JOBS:
    path = NOTEBOOKS_DIR / fname
    nb = json.loads(path.read_text())
    changed = 0
    for i, cell in enumerate(nb["cells"]):
        original = "".join(cell.get("source", []))
        patched = original
        for old, new in patches.items():
            if old in patched:
                patched = patched.replace(old, new)
                changed += 1
        if patched != original:
            # Preserve source as a list of lines matching original format
            nb["cells"][i]["source"] = [line + ("\n" if j < len(patched.splitlines()) - 1 else "")
                                         for j, line in enumerate(patched.splitlines())]
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"  OK {fname} — {changed} replacement(s) applied")
