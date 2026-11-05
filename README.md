# Explainable Knowledge Distillation with PCE/HDMR and EBM

**Research implementation | Datasets: UCI Phishing Websites and UCI Human Activity Recognition (HAR)**

This repository investigates whether a high-performing black-box classifier can be distilled into analytically interpretable student models, and whether **predictive fidelity** also implies **explanation fidelity**.

The pipeline trains an **XGBoost teacher** and two students: a **sparse polynomial-chaos / HDMR-style surrogate** and a **class-wise Explainable Boosting Machine (EBM) surrogate**. It evaluates predictive performance, agreement with the teacher, probability-distribution fidelity, and feature-attribution agreement using a shared SHAP evaluation space.

> **Research status:** The HAR results below were read from supplied experiment files. The Phishing results were reported from an earlier run and should be regenerated under the current repository for a strictly matched comparison. This is a research prototype, not a validated production package.

## 1. Research objectives

1. Train and freeze an accurate teacher before fitting students.
2. Distill teacher class probabilities into interpretable surrogate models.
3. Quantify the trade-off between accuracy, probability fidelity, and model complexity.
4. Compare SHAP feature attributions for the teacher and students in the **same original feature space**.
5. Evaluate whether strong predictive agreement guarantees similar explanations.

## 2. Datasets and evaluation protocols

| Dataset | Samples | Input features | Classes | Split |
|---|---:|---:|---:|---|
| [UCI Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites) | 11,055 | 30 | 2 | Stratified train/validation/test (60/20/20) |
| [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | 10,299 | 561 | 6 | Official subject-disjoint test set; group-based validation within official training set |

For HAR, the recorded split sizes are **5,551 training**, **1,801 validation**, and **2,947 test** observations. The audit reports zero missing values and zero duplicate feature rows. It identifies the split as `official_subject_disjoint_test_group_validation`. This audit summary does **not**, by itself, independently prove subject-ID non-overlap or that every preprocessing operation was train-only; inspect and test the splitting and preprocessing code before making that claim.

The repository contains **only Phishing and HAR** code. The previous Gas Sensor Drift workflow has been removed.

## 3. Methods

| Model | Role | Implementation |
|---|---|---|
| XGBoost | Teacher | Multiclass/binary gradient-boosted trees; hyperparameters selected by validation log loss |
| Sparse PCE/HDMR | Student 1 | Degree-2 Hermite polynomial expansion with sparse regression of teacher log-probabilities |
| EBM | Student 2 | One ExplainableBoostingRegressor per class, trained to approximate teacher log-probabilities |
| Permutation SHAP | Post-hoc explanation | Shared evaluation samples, training-derived background, original input features, and `log P(class | x)` target |

Students use teacher-selected feature subsets, typically **8, 12, or 16** features. Student hyperparameters are selected by validation **KL(teacher || student)**, not by test accuracy. The current tuning script implements a **fixed grid**, not an Optuna optimization study, even though Optuna appears in the dependency list.

### Key evaluation metrics

- **Accuracy / macro-F1:** predictive performance against ground-truth labels.
- **Log loss:** probabilistic quality against ground-truth labels.
- **Teacher agreement:** fraction of samples where student and teacher predict the same class.
- **KL divergence:** difference between teacher and student probability distributions; lower is better.
- **Probability MAE:** average absolute difference between predicted probabilities; lower is better.
- **SHAP comparison:** class-specific global attribution rankings and local explanations; explanation similarity is **not** evidence of explanation correctness.

## 4. Installation

Recommended: **Python 3.11 or 3.12** on Windows, Linux, or macOS. The uploaded HAR run records Python 3.14.6; for a new reproducible environment prefer 3.11/3.12 and record exact installed package versions.

Open a terminal in the **extracted `kd_research_code` folder**, where `requirements.txt` is located.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

On Linux/macOS, activate with `source .venv/bin/activate`. If PowerShell blocks activation, use `.venv\Scripts\python.exe` instead of `python` for subsequent commands.

The first dataset download requires internet access. Do not commit raw datasets or caches containing downloaded source files.

## 5. Run the experiments

All commands below run from the project root.

**A. Inspect the datasets and splits**

```powershell
python diagnostics.py --dataset all
```

**B. Quick pilot**

```powershell
python scripts/run_all.py --smoke-test --dataset har
python scripts/run_all.py --smoke-test --dataset phishing
```

**C. Validation-based model tuning**

```powershell
python tune_hyperparameters.py --dataset har
python tune_hyperparameters.py --dataset phishing
```

Or run both datasets:

```powershell
python tune_hyperparameters.py --dataset all
```

**D. Compare SHAP explanations**

```powershell
python compare_shap.py --dataset har --samples 8 --background 8 --class-index 0
python compare_shap.py --dataset phishing --samples 20 --background 15
```

`--class-index` is **zero-based**. For HAR, repeat it for `0` through `5` to inspect all six classes. Increasing sample/background counts may be computationally expensive for 561 features. Do not interpret an 8-sample SHAP run as a stable population-level explanation.

**E. Generate summary tables and plots**

```powershell
python plot_results.py
```

**F. Convenience runner**

```powershell
python scripts/run_all.py --dataset all --skip-shap
```

Omit `--skip-shap` only if you want the potentially expensive default SHAP computation.

## 6. Recorded experimental results

### Test-set metrics

| Dataset | Model | Accuracy | Macro-F1 | Log loss | Teacher agreement | KL(Teacher || Student) |
|---|---|---:|---:|---:|---:|---:|
| Phishing | XGBoost Teacher | **96.38%** | **96.33%** | **0.0896** | — | — |
| Phishing | PCE/HDMR | 93.85% | 93.75% | 0.1482 | 96.29% | 0.0391 |
| Phishing | EBM | 93.58% | 93.49% | 0.1496 | 95.84% | 0.0400 |
| HAR | XGBoost Teacher | **94.91%** | **94.77%** | **0.1372** | — | — |
| HAR | PCE/HDMR | 84.42% | 84.23% | 0.5156 | 86.39% | 0.3364 |
| HAR | EBM | 83.37% | 83.12% | 0.5737 | 85.92% | 0.3223 |

**Source of measurements:** HAR: user-provided `har_tuned.json` (5 November 2026); Phishing: earlier experiment output reported in the project conversation. Do not claim these are fully harmonized reruns. In particular, the test results have been inspected during project development; further test-driven model changes would make the evaluation exploratory.

### HAR selected configurations

- **Teacher:** `n_estimators=500`, `max_depth=3`, `learning_rate=0.04`.
- **PCE/HDMR:** `n_features=16`, `alpha=0.01`.
- **EBM:** `n_features=16`, `learning_rate=0.025`, `max_rounds=400`, `interactions=6`.
- HAR validation selection: Teacher log loss **0.12466**; PCE KL **0.26637**; EBM KL **0.19398**.

The HAR teacher is highly accurate, but the 16-feature students lose roughly 10–12 percentage points of accuracy. EBM yields slightly lower test KL and probability MAE than PCE, while PCE yields slightly better test accuracy and macro-F1.

## 7. SHAP explanation analysis

The supplied `har_shap_class0.json` evaluates **class label 1** (`class_index=0`) using **8 evaluation samples** and **8 background samples**. Its accompanying figure, `har_shap_class0_comparison.png`, displays mean absolute permutation SHAP values on the `log P(class 1 | x)` scale. In this small diagnostic sample, `har_feature_053` and `har_feature_182` are prominent for multiple models, but their relative importances differ.

The students operate on a restricted 16-feature subset, while the teacher can use all 561 features. Consequently, differences in attribution rankings partly reflect **feature-budget constraints**. Before drawing strong explanation-fidelity conclusions, repeat SHAP with larger, pre-specified samples, all classes, uncertainty intervals, and a consistent background protocol.

## 8. Data-leakage prevention and reporting safeguards

1. Keep the official HAR test participants outside model fitting, preprocessing-statistic estimation, feature selection, and hyperparameter search.
2. Split HAR training and validation by **subject ID**, not by individual windows; assert that subject sets are disjoint.
3. Fit imputation/scaling/feature-selection transforms **on training data only**; apply fitted transforms to validation/test.
4. Freeze the teacher before student distillation. Fit students on **training-row teacher predictions** only.
5. Select teacher settings by validation log loss; select student settings by validation KL; never select using test outcomes.
6. Use SHAP background examples from training data. Keep SHAP evaluation and model selection conceptually separate.
7. Preserve dataset version, seed, split IDs, dependency versions, and execution commands.
8. Disclose that HAR replaced Gas Sensor Drift after exploratory inspection of the earlier dataset. A genuinely untouched confirmation set is needed for final confirmatory claims.

## 9. Files and outputs

```text
kd_research_code/
├── README.md
├── requirements.txt
├── pyproject.toml
├── run_experiments.py          # Data loading, models, pilot experiments
├── tune_hyperparameters.py     # Validation-based teacher/student search
├── compare_shap.py             # Shared-feature-space SHAP comparisons
├── diagnostics.py              # Dataset and split audit
├── plot_results.py             # Result tables and figures
├── metrics_extended.py         # Supplementary metrics
├── scripts/
│   └── run_all.py
├── tests/
│   ├── test_har_split.py
│   ├── test_metrics.py
│   └── test_pce_math.py
└── results/                     # Generated locally by experiment scripts
```

Typical outputs:

| File | Purpose |
|---|---|
| `results/har_audit.json` | HAR data and split audit |
| `results/har_pilot.json` | Pilot results |
| `results/har_tuned.json` | HAR tuning trials, selected parameters, test metrics |
| `results/phishing_tuned.json` | Phishing tuning and metrics, after running |
| `results/har_shap_class0.json` | Class-specific SHAP summary and comparisons |
| `results/har_shap_class0_local.npz` | Per-sample SHAP arrays |
| `results/har_shap_class0_comparison.png` | Feature-importance comparison plot |
| `results/metrics_summary.csv` | Aggregated metrics from available runs |

This cleaned archive includes the **actual user-supplied HAR result files**, including JSON and SHAP artifacts. The Phishing figures in the README are historical reported measurements; no Phishing JSON was provided, so **no Phishing results file has been fabricated**. Run `python tune_hyperparameters.py --dataset phishing` to produce `results/phishing_tuned.json`. The provided HAR results have not been independently re-executed here.

## 10. Limitations and future work

- Sparse PCE uses a Gaussian-reference Hermite basis. Correlated observed HAR features mean coefficient-energy scores **must not be presented as empirical Sobol indices for correlated inputs** without further justification.
- The implemented students approximate teacher **log-probabilities via regression**; they do not directly optimize KL during fitting. KL is used to select configurations.
- Current PCE is degree 2; degree-3, interaction-order ablations, repeated seeds, bootstrap confidence intervals, and full Pareto analyses are not yet implemented.
- SHAP with an independent masker can generate unrealistic combinations of correlated sensor features. Treat attributions as model-behavior diagnostics, not causal effects.
- Model complexity counts for PCE and EBM are different constructs and should not be directly interpreted as equivalent units of interpretability.
- Further improvement should be based on a pre-specified validation protocol, not repeated selection on the already-inspected test set.

## 11. Reproducibility checklist

```powershell
python --version
python -m pip freeze > environment_versions.txt
python -m pytest -q
python diagnostics.py --dataset all
python tune_hyperparameters.py --dataset all
python plot_results.py
```

Record the exact source data files/checksums, operating system, CPU, random seed (recorded as **42** in the supplied HAR run), run time, and any errors or skipped experiments. Do not fabricate missing results.

## 12. Academic disclosure

This repository is intended for a PhD research exercise. Document AI assistance used to generate, modify, or debug code according to the assignment and institutional rules. The researcher remains responsible for reviewing the implementation, verifying dataset provenance, running experiments, and interpreting the findings.

## Dataset citations

- UCI Machine Learning Repository. *Phishing Websites*. https://archive.ics.uci.edu/dataset/327/phishing+websites
- UCI Machine Learning Repository. *Human Activity Recognition Using Smartphones*. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

---

**Main research takeaway:** Strong teacher accuracy and high student agreement do not, by themselves, demonstrate faithful explanations. The study evaluates both predictive and explanatory fidelity, with explicit attention to feature restrictions and leakage-controlled evaluation.


## Cleaned release and provenance

- Removed obsolete Gas Sensor Drift code and all `gas` CLI choices.
- Bundled user-supplied HAR audit, pilot, tuning, and SHAP outputs under `results/`.
- Phishing measurements in the table are historical; the cleaned archive does not include a Phishing measured JSON artifact.
- HAR SHAP output is class-specific with a small evaluation/background sample; it is exploratory.
- The HAR test set was already inspected during model development. Treat any subsequent optimization against its results as exploratory, not confirmatory.
- This cleanup does not constitute an independent end-to-end rerun or proof of absence of leakage.
