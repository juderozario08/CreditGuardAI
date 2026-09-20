# CreditGuard AI — Technical Overview

This document provides an exhaustive, in-depth walkthrough of everything that was built for CreditGuard AI: the full project architecture, every library used and why it was chosen, all engineering decisions and the reasoning behind them, and the complete end-to-end pipeline from raw data ingestion to real-time credit risk scoring.

---

## Table of Contents

1. [Project Motivation](#project-motivation)
2. [Dataset](#dataset)
3. [Project Architecture](#project-architecture)
4. [Libraries and Dependencies](#libraries-and-dependencies)
5. [Data Ingestion Layer](#data-ingestion-layer)
6. [Feature Engineering Pipeline](#feature-engineering-pipeline)
7. [Exploratory Data Analysis](#exploratory-data-analysis)
8. [Classical Machine Learning Models](#classical-machine-learning-models)
9. [Deep Learning Model (PyTorch MLP)](#deep-learning-model-pytorch-mlp)
10. [GPU Acceleration Strategy](#gpu-acceleration-strategy)
11. [Hyperparameter Tuning and Threshold Optimization](#hyperparameter-tuning-and-threshold-optimization)
12. [Evaluation Dashboard](#evaluation-dashboard)
13. [Real-Time Inference Demo](#real-time-inference-demo)
14. [CLI Design](#cli-design)
15. [Code Quality and Tooling](#code-quality-and-tooling)
16. [Generated Artifacts](#generated-artifacts)
17. [Benchmark Results](#benchmark-results)

---

## Project Motivation

CreditGuard AI is a portfolio machine learning project built to demonstrate end-to-end applied ML engineering: data ingestion, feature engineering, model training, hyperparameter tuning, evaluation dashboarding, GPU-accelerated deep learning, and real-time inference — all orchestrated through a clean CLI interface.

The problem domain — credit card default prediction — was chosen because it is a high-impact, real-world financial risk problem where model interpretability, class imbalance handling, and threshold tuning all matter. Unlike toy classification tasks, credit default prediction requires domain-aware feature engineering (utilization ratios, delinquency trends, payment-to-bill ratios) and careful treatment of the precision-recall tradeoff, since the cost of missing a defaulter far outweighs the cost of a false alarm.

---

## Dataset

**Source:** UCI Machine Learning Repository — Dataset #350, "Default of Credit Card Clients."

**Contents:** 30,000 customer records from a Taiwanese bank, each containing:
- A credit limit (`LIMIT_BAL`)
- Demographic fields: `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- 6 months of repayment status codes (`PAY_0` through `PAY_6`), where negative values indicate on-time payment and positive values indicate months of delinquency
- 6 months of billing amounts (`BILL_AMT1` through `BILL_AMT6`)
- 6 months of payment amounts (`PAY_AMT1` through `PAY_AMT6`)
- A binary target: `default payment next month` (1 = defaulted, 0 = settled)

**Class Distribution:** 77.9% non-default (settled) vs. 22.1% default — a moderately imbalanced dataset, which directly influenced every modeling decision from loss weighting to evaluation metric selection.

**Format Quirk:** The UCI download provides the data with a double-header row. Row 0 contains generic labels (`X1`, `X2`, `X3`...) and row 1 contains the real column names (`LIMIT_BAL`, `SEX`, etc.). The data loader handles this by reading with `header=1` and falling back to `header=0` if the target column is not found. Additionally, some versions of the dataset name the first repayment column `PAY_1` instead of `PAY_0` — the loader normalizes this automatically.

---

## Project Architecture

```
CreditGuardAI/
├── main.py                         # Unified CLI entry point
├── data/                           # Dataset storage directory (.csv, .xls, .xlsx)
├── src/
│   ├── __init__.py                 # Package marker
│   ├── data_loader.py              # Dual-format data ingestion (CSV/XLS/XLSX)
│   ├── preprocess.py               # Feature engineering + StandardScaler
│   ├── eda_visualizations.py       # 4 EDA Matplotlib/Seaborn charts
│   ├── train_classical.py          # 3 classical ML models with cross-validation
│   ├── train_pytorch.py            # PyTorch MLP with GPU support
│   ├── evaluate.py                 # 4-panel benchmark dashboard
│   ├── tune_hyperparameters.py     # Parallel RandomizedSearchCV + threshold optimization
│   └── device.py                   # GPU/CPU device selector (NVIDIA/AMD/CPU)
├── models/                         # Saved model artifacts (generated at runtime)
├── figures/                        # Saved chart PNGs (generated at runtime)
├── requirements.txt                # Python dependencies
├── README.md                       # Simple user-facing README
└── .gitignore                      # Python/venv exclusions
```

The codebase follows a modular, single-responsibility design. Each `src/` module handles exactly one stage of the pipeline, and `main.py` orchestrates them through CLI flags. There is no shared mutable state between modules — data flows explicitly through function arguments and return values.

---

## Libraries and Dependencies

### Pandas

**Role:** Tabular data loading, manipulation, and feature engineering.

**Why:** Pandas is the standard for structured data in Python. It was used to read both CSV and Excel files, perform column renaming/normalization, compute grouped aggregations (e.g., delinquency risk by `PAY_0`), generate one-hot encodings for categorical variables, and build the final feature matrix. Its DataFrame API made the feature engineering pipeline readable and maintainable.

### NumPy

**Role:** Numerical array operations and metric computations.

**Why:** NumPy underpins every numerical operation in the pipeline — from clipping utilization ratios and computing means to building threshold sweep arrays (`np.linspace`) and converting between Pandas DataFrames and the raw arrays that Scikit-Learn and PyTorch expect. It is a required dependency of both Pandas and Scikit-Learn.

### Scikit-Learn

**Role:** Classical ML model training, cross-validation, hyperparameter tuning, preprocessing, and evaluation metrics.

**Why:** Scikit-Learn provides the complete classical ML toolkit in a consistent API. Specific components used:

- **`StandardScaler`**: Z-score normalization of all features before model training. Chosen over MinMaxScaler because StandardScaler is more robust to outliers (which exist heavily in billing amounts and utilization ratios) and is the expected input distribution for logistic regression and neural networks.
- **`train_test_split`**: 80/20 stratified split to preserve the 22.1% default class ratio in both train and test sets.
- **`StratifiedKFold`**: 5-fold stratified cross-validation. Stratification ensures each fold maintains the same class ratio, preventing optimistic or pessimistic bias from random splits.
- **`LogisticRegression`**: Baseline linear classifier with `class_weight="balanced"` and L-BFGS solver.
- **`RandomForestClassifier`**: Ensemble of 150 decision trees with `max_depth=12`, `min_samples_split=10`, and `class_weight="balanced"`. Also used as the source of feature importance scores in the evaluation dashboard.
- **`HistGradientBoostingClassifier`**: Histogram-based gradient boosting (Scikit-Learn's native implementation, similar to LightGBM). Chosen over `GradientBoostingClassifier` because it is significantly faster on datasets of this size and natively supports `class_weight="balanced"`.
- **`RandomizedSearchCV`**: Randomized hyperparameter search across 25 parameter combinations with 5-fold CV, parallelized across all CPU cores (`n_jobs=-1`).
- **Metrics suite**: `roc_auc_score`, `average_precision_score`, `precision_score`, `recall_score`, `f1_score`, `accuracy_score`, `balanced_accuracy_score`, `confusion_matrix`, `roc_curve`, `precision_recall_curve`.

### PyTorch

**Role:** Deep learning model definition, training loop, GPU tensor operations.

**Why:** PyTorch was chosen over TensorFlow/Keras for its explicit, imperative training loop — which gave full control over early stopping, checkpoint saving with CPU tensor serialization (required for DirectML compatibility), custom `pos_weight` computation, and learning rate scheduling. The explicit `model.train()` / `model.eval()` pattern also ensures BatchNorm and Dropout behave correctly without framework-level "magic."

### torch-directml

**Role:** AMD GPU acceleration on Windows via Microsoft's DirectML backend.

**Why:** The development machine has an AMD Radeon RX 9070 XT. AMD GPUs are not supported by CUDA, and ROCm is Linux-only. DirectML is the only option for GPU-accelerating PyTorch on Windows with AMD hardware. The `torch-directml` package registers a `privateuseone` device type that acts as a drop-in replacement for CUDA in most operations. A conditional install marker (`sys_platform == 'win32'`) in `requirements.txt` ensures this dependency is only installed on Windows.

### Matplotlib

**Role:** All chart and figure generation.

**Why:** Matplotlib is the foundational plotting library in Python. It was used to generate all 6 output charts (4 EDA + 1 threshold curve + 1 4-panel dashboard). The `seaborn-v0_8-whitegrid` style was applied globally for clean, professional-looking charts. All figures are saved at 300 DPI for high-resolution output.

### Seaborn

**Role:** Statistical visualization layer on top of Matplotlib.

**Why:** Seaborn provides higher-level plot types that are tedious to build with raw Matplotlib — specifically `sns.barplot` for the class distribution and delinquency analysis, `sns.boxplot` for utilization distributions, and `sns.heatmap` for the correlation matrix and confusion matrix. It integrates natively with Pandas DataFrames and automatically handles grouping, color palettes, and annotations.

### xlrd

**Role:** Reading `.xls` (legacy Excel 97-2003) files.

**Why:** The UCI dataset is distributed as a `.xls` file (not `.xlsx`). Pandas delegates `.xls` reading to `xlrd`, which is the only Python library that can parse this legacy binary format. Without it, `pd.read_excel()` raises an error on `.xls` files.

### openpyxl

**Role:** Reading `.xlsx` (modern Excel 2007+) files.

**Why:** If the dataset is converted to or provided as `.xlsx`, Pandas delegates reading to `openpyxl`. Including both `xlrd` and `openpyxl` ensures the loader handles every Excel variant the user might have.

### joblib

**Role:** Serializing trained model objects and preprocessor state.

**Why:** `joblib` is the recommended serialization format for Scikit-Learn models. It is faster than `pickle` for objects containing large NumPy arrays (which fitted models internally store) and produces smaller files. All classical models, the fitted `StandardScaler`, and the feature name list are saved as `.joblib` files.

### Ruff

**Role:** Linting and formatting.

**Why:** Ruff is an extremely fast Python linter and formatter (written in Rust) that combines the functionality of Flake8, isort, pyupgrade, and Black into a single tool. All source files pass `ruff check .` with zero errors and `ruff format .` with zero changes.

---

## Data Ingestion Layer

**Module:** `src/data_loader.py`

The data loader (`load_raw_data`) was designed to be format-agnostic. The user can provide a path to a `.csv`, `.xls`, or `.xlsx` file via the `--file` CLI flag, or the loader will auto-discover the dataset by searching a predefined list of candidate filenames in the project root and `data/` directory.

**Key decisions:**

1. **Dual-header detection**: The UCI `.xls` file has a double header. The loader first reads with `header=1` and checks if `default payment next month` exists in the columns. If not, it falls back to `header=0`. This handles both the official UCI format and cleaned versions of the dataset.

2. **Target column normalization**: The target column can appear as `default payment next month`, `Y`, `DEFAULT_PAYMENT_NEXT_MONTH`, or `default` depending on the dataset version. The loader checks all four and renames the matched column to `DEFAULT`.

3. **`PAY_1` → `PAY_0` renaming**: Some dataset versions use `PAY_1` for the most recent repayment status. The loader normalizes this to `PAY_0` so the rest of the pipeline can use consistent column names.

4. **Category cleanup**: The `EDUCATION` column contains undocumented category codes (0, 5, 6) that are not in the official codebook. These are collapsed into category 4 ("Other"). Similarly, `MARRIAGE` code 0 is collapsed into category 3 ("Other"). This prevents the one-hot encoder from creating spurious features for meaningless categories.

5. **Numeric coercion and NaN removal**: All columns are coerced to numeric types and any rows with NaN values are dropped. This is a safety net for datasets that might have string artifacts or missing values.

---

## Feature Engineering Pipeline

**Module:** `src/preprocess.py`

The raw dataset has 23 features after dropping `ID`. The feature engineering pipeline expands this to 47 features by computing domain-specific financial indicators. The goal was to surface credit risk signals that the raw numbers obscure.

### Engineered Features

| Feature Group | Features Created | Rationale |
|---|---|---|
| **Credit Utilization** | `UTILIZATION_1` through `UTILIZATION_6` (6 features) | `BILL_AMT / LIMIT_BAL` — measures how close the customer is to maxing out their credit line each month. Utilization above 1.0 means the customer owes more than their limit (overlimit). Clipped to [-1.0, 5.0] to contain extreme outliers. |
| **Utilization Summary** | `AVG_UTILIZATION` (1 feature) | Mean of the 6 monthly utilization ratios. A single number summarizing overall credit usage intensity. |
| **Payment-to-Bill Ratios** | `PAY_RATIO_1` through `PAY_RATIO_5` (5 features) | `PAY_AMT{i} / BILL_AMT{i+1}` — how much of the previous month's bill did the customer actually pay? A ratio near 0 means minimum payment or less; near 1 means full payment. Clipped to [0.0, 10.0]. |
| **Payment Ratio Summary** | `AVG_PAY_RATIO` (1 feature) | Mean of the 5 payment-to-bill ratios. Summarizes overall repayment discipline. |
| **Aggregate Amounts** | `AVG_BILL_AMT`, `AVG_PAY_AMT` (2 features) | Mean billing and payment amounts across 6 months. |
| **Delinquency Indicators** | `MAX_DELINQUENCY`, `MIN_DELINQUENCY` (2 features) | Maximum and minimum repayment status codes across 6 months. A `MAX_DELINQUENCY` of 8 means the customer was 8+ months late at some point. |
| **Delinquency Velocity** | `DELINQUENCY_TREND` (1 feature) | `PAY_0 - PAY_6` — is the customer's delinquency getting worse (positive) or improving (negative) over the 6-month window? |
| **Late Month Count** | `LATE_MONTHS_COUNT` (1 feature) | Number of months (out of 6) where the repayment status was positive (i.e., the customer was late). |
| **Balance Trajectory** | `NET_BALANCE_CHANGE` (1 feature) | `BILL_AMT1 - BILL_AMT6` — is the customer's outstanding balance growing or shrinking? |
| **Total Repayment Effort** | `TOTAL_PAY_VS_TOTAL_BILL` (1 feature) | Sum of all 6 payments divided by sum of all 6 bills. Measures overall repayment coverage across the entire observation window. |
| **One-Hot Encoding** | ~3-4 features from `SEX`, `EDUCATION`, `MARRIAGE` | Categorical variables are one-hot encoded with `drop_first=True` to avoid multicollinearity. |

### Scaling

All 47 features are z-score normalized using `StandardScaler` (mean=0, std=1). The scaler is fitted on the training set only and applied to the test set to prevent data leakage. The fitted scaler is serialized to `models/scaler.joblib` so the real-time inference demo can transform new applicants using the exact same normalization parameters.

### Train/Test Split

An 80/20 stratified split (`train_test_split` with `stratify=y`) ensures both sets maintain the original 77.9%/22.1% class distribution. The random state is fixed at 42 for full reproducibility.

---

## Exploratory Data Analysis

**Module:** `src/eda_visualizations.py`

Four publication-quality charts are generated to understand the data before modeling:

### 1. Class Distribution Bar Chart (`eda_class_distribution.png`)
A bar chart showing the count and percentage of defaulted vs. settled accounts. This immediately highlights the ~3.5:1 class imbalance, which justifies all subsequent decisions around `class_weight="balanced"`, `pos_weight`, and evaluation with balanced accuracy / PR-AUC.

### 2. Delinquency vs. Default Rate (`eda_delinquency_vs_default.png`)
Groups customers by their most recent repayment status (`PAY_0`) and plots the observed default rate for each group. Demonstrates that customers with `PAY_0 >= 2` (2+ months late) have default rates exceeding 60%, confirming that `PAY_0` is by far the strongest single predictor — a finding that the feature importance analysis later validates.

### 3. Credit Utilization Distribution (`eda_utilization_distribution.png`)
A box plot comparing the distribution of `AVG_UTILIZATION` between defaulted and settled customers. Shows that defaulters tend to have higher and more variable utilization, validating the engineering of utilization ratio features.

### 4. Correlation Heatmap (`eda_correlation_matrix.png`)
A 10×10 heatmap of Pearson correlations between the target variable and the most important engineered features. Reveals that `PAY_0`, `MAX_DELINQUENCY`, `LATE_MONTHS_COUNT`, and `DELINQUENCY_TREND` have the strongest positive correlations with default, while `LIMIT_BAL` and `TOTAL_PAY_VS_TOTAL_BILL` have negative correlations (higher credit limits and more total repayment correspond to lower default risk).

---

## Classical Machine Learning Models

**Module:** `src/train_classical.py`

Three classical models are trained, cross-validated, and benchmarked:

### Logistic Regression
- **Role:** Baseline linear classifier. Establishes a performance floor that more complex models must beat.
- **Config:** `class_weight="balanced"`, `max_iter=1000`, `solver="lbfgs"`.
- **Why `class_weight="balanced"`:** Automatically adjusts class weights inversely proportional to class frequency (`n_samples / (n_classes * n_class_samples)`). This increases the penalty for misclassifying the minority (default) class, which is critical for a 22.1% minority rate.

### Random Forest
- **Role:** Non-linear ensemble that also provides feature importance scores.
- **Config:** 150 trees, `max_depth=12`, `min_samples_split=10`, `class_weight="balanced"`, `n_jobs=-1`.
- **Why 150 trees, depth 12:** Enough trees to stabilize predictions and deep enough to capture non-linear interactions (like utilization × delinquency), but not so deep that individual trees overfit. `min_samples_split=10` acts as additional regularization.
- **Why `n_jobs=-1`:** Trains all 150 trees in parallel across all CPU cores.

### Histogram-Based Gradient Boosting
- **Role:** Best-performing classical model. Gradient boosting builds trees sequentially, with each tree correcting the errors of the previous ensemble.
- **Config:** `max_iter=150`, `max_depth=6`, `learning_rate=0.08`, `class_weight="balanced"`.
- **Why `HistGradientBoostingClassifier` over `GradientBoostingClassifier`:** The histogram-based variant bins continuous features into 256 discrete levels, reducing the computational cost of finding optimal splits from O(n·features) to O(bins·features). On 30,000 rows this translates to roughly 5-10x faster training. It also natively supports `class_weight="balanced"`, which the older `GradientBoostingClassifier` does not.

### Cross-Validation
All three models undergo 5-fold stratified cross-validation on the training set (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) using ROC-AUC as the scoring metric. This provides a more reliable estimate of generalization performance than a single train/test split and checks for overfitting (large gap between CV score and test score would indicate overfitting).

### Model Persistence
Each trained model is serialized to `models/` as a `.joblib` file. The evaluation module and real-time inference demo load these saved models rather than retraining, enabling independent execution of pipeline stages.

---

## Deep Learning Model (PyTorch MLP)

**Module:** `src/train_pytorch.py`

### Architecture: `CreditDefaultMLP`

A 3-hidden-layer Multi-Layer Perceptron:

```
Input (47 features)
  → Linear(47, 128) → BatchNorm(128) → LeakyReLU(0.1) → Dropout(0.3)
  → Linear(128, 64) → BatchNorm(64)  → LeakyReLU(0.1) → Dropout(0.2)
  → Linear(64, 32)  → BatchNorm(32)  → LeakyReLU(0.1) → Dropout(0.1)
  → Linear(32, 1)   → [raw logit output]
```

**Design decisions:**

1. **BatchNorm before activation**: Normalizes layer outputs to stabilize training and allow higher learning rates. Placed before activation (pre-activation normalization) following common practice for MLPs.

2. **LeakyReLU(0.1) over ReLU**: Standard ReLU can cause "dead neurons" (neurons that output 0 for all inputs and stop learning). LeakyReLU with a negative slope of 0.1 prevents this by allowing a small gradient for negative inputs.

3. **Decreasing Dropout (0.3 → 0.2 → 0.1)**: Higher dropout in early layers (which have more parameters and are more prone to overfitting) and lower dropout in later layers (which learn more abstract, less redundant representations). This is a common heuristic in MLP design.

4. **Raw logit output (no sigmoid)**: The final layer outputs a single raw logit. Sigmoid is applied externally — by `BCEWithLogitsLoss` during training (which is numerically more stable than applying sigmoid then BCELoss) and by `torch.sigmoid()` during inference.

### Training Configuration

- **Loss Function:** `BCEWithLogitsLoss` with `pos_weight=3.52` (computed as `neg_count / pos_count = 23364 / 6636 ≈ 3.52`). This makes the loss for a missed defaulter 3.52× more costly than a false alarm, directly addressing the 22.1% class imbalance at the loss level. This is the neural network equivalent of `class_weight="balanced"` in Scikit-Learn.

- **Optimizer:** AdamW with `lr=0.001` and `weight_decay=1e-4`. AdamW was chosen over vanilla Adam because it implements decoupled weight decay regularization (the L2 penalty is applied directly to the weights, not mixed into the adaptive gradient estimates). This provides more effective regularization for tabular data where feature scales vary widely even after normalization.

- **Learning Rate Scheduler:** `ReduceLROnPlateau(mode="max", factor=0.5, patience=2)`. Monitors validation ROC-AUC. If the AUC doesn't improve for 2 consecutive epochs, the learning rate is halved. This allows the model to make large parameter updates early in training and fine-tune with smaller steps as it approaches convergence.

- **Early Stopping:** Patience of 6 epochs. Training halts if validation ROC-AUC doesn't improve for 6 consecutive epochs. This prevents overfitting and saves compute time.

- **Batch Size:** 256. Small enough to provide stochastic gradient noise (which acts as regularization) but large enough to efficiently utilize GPU parallelism.

- **Checkpoint Strategy:** The model with the best validation ROC-AUC is saved to disk. After training completes (or early stopping triggers), the best checkpoint is reloaded for final evaluation. Importantly, tensors are moved to CPU before saving (`{k: v.cpu() for k, v in model.state_dict().items()}`). This is required because DirectML tensors cannot be serialized directly — attempting to `torch.save()` a model with DirectML tensors raises a serialization error. Similarly, `torch.load()` uses `map_location="cpu"` and `weights_only=False` because DirectML checkpoints contain a custom `_rebuild_device_tensor_from_numpy` function that `weights_only=True` rejects.

---

## GPU Acceleration Strategy

**Module:** `src/device.py`

The `get_compute_device` function implements a cascading device selection strategy:

```
User passes --gpu?
├── No  → CPU (all threads via torch.set_num_threads)
└── Yes → Check NVIDIA CUDA available?
    ├── Yes → torch.device("cuda") → NVIDIA GPU
    └── No  → Check torch-directml available and has GPU?
        ├── Yes → torch_directml.device() → AMD GPU via DirectML
        └── No  → Check ROCm (torch.version.hip)?
            ├── Yes → torch.device("cuda") → AMD ROCm GPU
            └── No  → CPU fallback (with warning)
```

**Why this order:**
1. **CUDA first**: If available, CUDA has the most mature PyTorch support with the fewest compatibility issues.
2. **DirectML second**: On Windows with AMD GPUs, this is the only option. It covers the RX 9070 XT on the development machine.
3. **ROCm third**: For Linux users with AMD GPUs. ROCm uses the CUDA device type under the hood.
4. **CPU fallback**: If no GPU is available or the user doesn't pass `--gpu`, CPU is used with all threads enabled.

**DirectML compatibility notes (discovered during development):**
- DirectML registers as `privateuseone:0` device type
- The `aten::log_sigmoid_forward` operator falls back to CPU on DirectML (harmless performance warning, not an error)
- `torch-directml` pinned PyTorch to version 2.4.1 at the time of development (auto-handled by `uv` during dependency resolution)

---

## Hyperparameter Tuning and Threshold Optimization

**Module:** `src/tune_hyperparameters.py`

### Parallel Randomized Search

A `RandomizedSearchCV` explores 25 random parameter combinations for `HistGradientBoostingClassifier` across a 5-dimensional search space:

| Parameter | Search Values | Purpose |
|---|---|---|
| `learning_rate` | [0.03, 0.05, 0.08, 0.10, 0.14] | Controls how much each tree corrects previous errors. Lower = more trees needed but less overfitting. |
| `max_depth` | [4, 6, 8, 10, None] | Maximum tree depth. `None` = no limit. |
| `min_samples_leaf` | [15, 25, 50, 100] | Minimum samples required in a leaf node. Higher = more regularization. |
| `l2_regularization` | [0.0, 0.1, 0.5, 1.0, 3.0] | L2 penalty on leaf weights. Prevents individual trees from making extreme predictions. |
| `max_iter` | [150, 200, 250] | Number of boosting iterations (trees). |

All 25 iterations run with `n_jobs=-1`, meaning every cross-validation fold is distributed across all available CPU cores. On a 12-core/24-thread Ryzen 9 9900X, this provides massive parallelism.

### Decision Threshold Optimization

After finding the best model, the module sweeps 51 thresholds from 0.20 to 0.70 and computes recall (coverage), precision, balanced accuracy, and accuracy at each threshold. This produces three key operating points:

1. **Standard (0.50)**: The default classification threshold. Balanced but may miss a significant portion of defaulters.
2. **75% Coverage**: The threshold that achieves at least 75% recall, meaning 75% of all actual defaulters are flagged.
3. **80% Coverage (0.37)**: A more aggressive threshold where 80.3% of defaulters are caught. The tradeoff is lower precision — more false alarms — but in credit risk, catching defaulters is typically more valuable than avoiding false flags.

A threshold-vs-coverage chart (`threshold_coverage_tradeoff.png`) visualizes the precision-recall-accuracy tradeoff across the entire threshold range, with vertical reference lines at the 0.50 standard and 0.37 aggressive thresholds.

---

## Evaluation Dashboard

**Module:** `src/evaluate.py`

Generates a single 4-panel figure (`model_benchmark_dashboard.png`) that provides a complete visual comparison of all 4 models:

### Panel A: ROC Curves
Overlays the Receiver Operating Characteristic curves of all models on one plot, with a diagonal random-guess baseline. The ROC curve plots True Positive Rate (recall) against False Positive Rate at every possible threshold. Higher AUC = better discrimination between defaulters and non-defaulters regardless of threshold choice.

### Panel B: Precision-Recall Curves
More informative than ROC for imbalanced datasets. Plots precision vs. recall at every threshold, with a horizontal baseline at the class prior (22.1%). Models with higher PR-AUC maintain better precision as recall increases, meaning they can catch more defaulters without flooding underwriters with false alarms.

### Panel C: Confusion Matrix
Shows the best model's (Gradient Boosting) confusion matrix as a heatmap. The four cells show True Negatives (correctly approved), False Positives (false alarms), False Negatives (missed defaults), and True Positives (caught defaults).

### Panel D: Top 10 Feature Importances
A horizontal bar chart of the 10 most predictive features according to the Random Forest's Gini importance scores. This provides model interpretability — showing underwriters which factors most influence the risk assessment.

---

## Real-Time Inference Demo

**Function:** `predict_sample` in `main.py`

Simulates a production credit underwriting system. Given a dictionary of applicant features and a decision threshold, it:

1. Loads the saved scaler, feature names, and best Gradient Boosting model from disk
2. Creates a full feature vector (filling missing features with 0.0)
3. Scales the input using the same StandardScaler from training
4. Computes the default probability
5. Maps the probability to a risk tier and recommended action:

| Default Probability | Risk Tier | Action |
|---|---|---|
| < threshold - 0.10 | LOW RISK | Approve Credit Line |
| threshold - 0.10 to threshold | MODERATE RISK | Review Pay Stubs & Assets |
| threshold to 0.60 | ELEVATED RISK | Reject Application |
| ≥ 0.60 | HIGH RISK | Reject Application |

The demo uses a sample applicant with characteristics that suggest high risk: low credit limit ($20K), young age (25), two consecutive months of delinquency, 95% utilization, and 4 out of 6 months with late payments.

---

## CLI Design

**Module:** `main.py`

The CLI is built with Python's `argparse` and exposes these flags:

| Flag | Purpose |
|---|---|
| `--all` | Execute the complete pipeline end-to-end (default if no flags specified) |
| `--eda` | Generate EDA figures only |
| `--train-classical` | Train and benchmark Scikit-Learn models only |
| `--train-pytorch` | Train PyTorch MLP only |
| `--evaluate` | Run evaluation and generate the 4-panel dashboard |
| `--tune` | Execute parallel hyperparameter tuning |
| `--demo` | Run the real-time risk assessment demo |
| `--gpu` | Enable GPU acceleration for PyTorch operations |
| `--threshold FLOAT` | Custom decision threshold for the demo (default 0.50) |
| `--file PATH` | Explicit path to the dataset file |

If no pipeline flags are provided, `--all` is automatically set, so running `python main.py` executes the entire pipeline.

---

## Code Quality and Tooling

- **Linter:** Ruff (`ruff check .`) — zero errors across all source files.
- **Formatter:** Ruff (`ruff format .`) — zero formatting changes needed.
- **Type Annotations:** All functions have full type annotations for parameters and return types, including complex generics like `tuple[dict[str, Any], dict[str, dict[str, Any]], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]`.
- **No unnecessary comments:** Following project conventions, comments are only used where the intent is genuinely non-obvious (e.g., the numbered chart sections in `eda_visualizations.py`).

---

## Generated Artifacts

### Model Files (`models/`)

| File | Contents |
|---|---|
| `scaler.joblib` | Fitted `StandardScaler` with learned mean/std from training data |
| `feature_names.joblib` | Ordered list of 47 feature names (needed to reconstruct feature vectors for inference) |
| `logistic_regression.joblib` | Trained Logistic Regression model |
| `random_forest.joblib` | Trained Random Forest model (150 trees) |
| `gradient_boosting.joblib` | Trained HistGradientBoosting model (best classical performer) |
| `pytorch_credit_model.pth` | PyTorch MLP state dict (CPU tensors) |

### Figures (`figures/`)

| File | Contents |
|---|---|
| `eda_class_distribution.png` | Class balance bar chart |
| `eda_delinquency_vs_default.png` | PAY_0 vs. default rate analysis |
| `eda_utilization_distribution.png` | Utilization box plot by outcome |
| `eda_correlation_matrix.png` | 10×10 risk factor correlation heatmap |
| `model_benchmark_dashboard.png` | 4-panel evaluation dashboard |
| `threshold_coverage_tradeoff.png` | Threshold optimization curve |

---

## Benchmark Results

Final test-set performance (80/20 stratified split, 6,000 test samples):

| Model | ROC-AUC | PR-AUC | Recall | Precision | Balanced Acc | F1 |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.7816** | — | **63.98%** | — | — | — |
| PyTorch MLP | 0.7783 | — | 61.72% | — | — | — |
| Random Forest | 0.7763 | — | 59.68% | — | — | — |
| Logistic Regression | 0.7473 | — | 60.06% | — | — | — |

Cross-validation (5-fold stratified on training set, 24,000 samples):

| Model | CV ROC-AUC |
|---|---|
| **Gradient Boosting** | **0.7872** |

Threshold tuning (Gradient Boosting, tuned model):
- **Standard (0.50):** ~64% default coverage
- **Aggressive (0.37):** **80.3% default coverage** — catches 4 out of every 5 defaulters
