# CreditGuard AI

CreditGuard AI is an automated financial risk assessment and credit default prediction engine built with **Python**, **Pandas**, **NumPy**, **Scikit-Learn**, **PyTorch**, and **Matplotlib**.

The system analyzes customer payment behavior, billing statements, and credit limits to forecast whether an account holder will default on their next payment cycle.

---

## What the Project Does

### 1. The Core Financial Problem
When financial institutions issue revolving credit lines, unsecured credit cards present significant loss exposure. Customers who fail to settle their balances (default) trigger expensive charge-offs and collections procedures. 

CreditGuard AI acts as an automated underwriting and credit monitoring engine that evaluates applicant solvency and transaction history in real time, assigning a calibrated default probability score and decision recommendation (`Approved`, `Manual Review`, or `Declined`).

---

### 2. The Dataset (UCI Machine Learning Repository #350)
The engine is trained and evaluated on 30,000 consumer credit accounts from the **UCI Machine Learning Repository**:
* **Demographics:** Age, Education Level, Marital Status, Sex.
* **Credit Line:** Total credit limit (`LIMIT_BAL`).
* **6-Month Repayment Trajectory (April–September):**
  * `PAY_0` to `PAY_6`: Repayment status (-2 = no consumption, -1 = paid in full, 0 = revolving credit, 1 to 8 = months of payment delay).
  * `BILL_AMT1` to `BILL_AMT6`: Monthly statement balance amounts.
  * `PAY_AMT1` to `PAY_AMT6`: Amount paid towards statements each month.
* **Target:** `DEFAULT` (0 = Settled / Non-default, 1 = Defaulted on next payment).

---

### 3. Solving the 78/22 Class Imbalance Problem
In consumer credit portfolios, non-defaulters heavily outnumber defaulters (~77.9% settled vs. 22.1% defaulted).

* **The Accuracy Trap:** A naive model that blindly predicts "Settled" for every account will record **77.9% accuracy**, yet catches **0% of defaults**, resulting in severe financial loss.
* **The Solution:** CreditGuard AI implements **Cost-Sensitive Loss Weighting**:
  * In **Scikit-Learn**: Models use `class_weight='balanced'`.
  * In **PyTorch**: The neural network uses `BCEWithLogitsLoss(pos_weight=3.52)`.
  * This penalizes missed defaults 3.5× more than false alarms, forcing the algorithms to optimize for **ROC-AUC**, **Balanced Accuracy**, and **Recall (Coverage)** rather than raw accuracy.

---

### 4. Domain-Specific Financial Feature Engineering
The raw 23 attributes are transformed into **47 predictive indicators**:
* **Credit Utilization Ratios ($1\dots6$):** $\text{BILL\_AMT}_i / \text{LIMIT\_BAL}$ — identifies borrowers maxing out their revolving credit limits.
* **Payment-to-Bill Solvency Ratios ($1\dots5$):** $\text{PAY\_AMT}_i / \text{BILL\_AMT}_{i+1}$ — detects whether an applicant pays balances in full or only makes minimum payments.
* **Delinquency Velocity & Trend:** $\text{PAY\_0} - \text{PAY\_6}$ — measures whether repayment behavior is accelerating into severe delinquency.
* **Maximum Delinquency Level:** Highest delay level reached over the 6-month observation window.
* **Late Month Counter:** Total number of billing cycles where payments were overdue.
* **Net Balance Change:** Overall balance growth over 6 months ($\text{BILL\_AMT1} - \text{BILL\_AMT6}$).

---

### 5. Multi-Model Benchmarking Suite
CreditGuard AI trains and benchmarks four distinct architectures on a stratified 80/20 train/test split (24,000 training accounts, 6,000 held-out test accounts):

1. **Logistic Regression:** Linear, interpretable baseline with balanced class weights.
2. **Random Forest:** Bagging ensemble of 150 decision trees averaging variance across feature subsets.
3. **HistGradientBoosting (Top Performer):** Boosting ensemble building trees sequentially to minimize residual loss.
4. **PyTorch Deep Learning MLP:** 3-layer feedforward neural network with Batch Normalization (`BatchNorm1d`), LeakyReLU activations, and Dropout (0.3, 0.2, 0.1) for regularization.

---

### 6. Benchmark Performance Scoreboard

| Model Architecture | Accuracy | Balanced Acc | Precision | Recall (Coverage) | F1-Score | ROC-AUC | PR-AUC | 5-Fold CV AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | 75.53% | **71.13%** | 46.12% | **63.23%** | **0.5334** | **0.7793** | 0.5571 | **0.7872 (±0.006)** |
| **PyTorch Neural Net (MLP)** | 75.42% | 70.89% | 45.92% | 62.77% | 0.5304 | **0.7787** | **0.5641** | — |
| **Random Forest** | **77.40%** | 71.06% | **49.10%** | 59.68% | 0.5388 | 0.7763 | 0.5550 | 0.7842 (±0.004) |
| **Logistic Regression** | 74.23% | 69.16% | 43.96% | 60.06% | 0.5076 | 0.7473 | 0.5005 | 0.7595 (±0.007) |

---

### 7. Decision Threshold Optimization (Maximizing Coverage)
In credit risk underwriting, decision thresholds are tuned based on a bank's risk appetite:

| Decision Threshold | Coverage (Recall) | Precision | Balanced Acc | Operating Stance |
| :---: | :---: | :---: | :---: | :--- |
| **0.50** | **64.0%** | 45.0% | 70.9% | Standard default cutoff |
| **0.40** | **75.4%** | 37.0% | 69.4% | Balanced underwriter stance (catches 3 in 4 defaults) |
| **0.37** | **80.3%** | 35.3% | 69.2% | **Aggressive risk shield (catches >80% of defaults)** |

---

### 8. Visualization Dashboard (`figures/`)
The system outputs high-resolution (300 DPI) publication-ready charts:
* `model_benchmark_dashboard.png`: 4-panel dashboard containing combined ROC curves, Precision-Recall curves, confusion matrices, and top 10 feature importances.
* `threshold_coverage_tradeoff.png`: Trade-off curves mapping default coverage vs. precision across decision thresholds.
* `eda_delinquency_vs_default.png`: Observed default probability spikes by repayment delay status.
* `eda_utilization_distribution.png`: Credit utilization distribution comparing settled vs. defaulted accounts.
* `eda_correlation_matrix.png`: Heatmap of primary solvency metrics.
* `eda_class_distribution.png`: Target class balance visualization.

---

## Cross-Platform Setup & Execution

CreditGuard AI is fully cross-platform and runs on **Windows**, **macOS**, and **Linux**.

### Prerequisites
* **Python 3.10, 3.11, or 3.12** installed on your system.
* Git (optional, for cloning).

---

### Installation

#### Windows (PowerShell / Command Prompt)
```powershell
# 1. Navigate to the project directory
cd C:\Users\juder\CreditGuardAI

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# PowerShell:
.venv\Scripts\Activate.ps1
# Or Command Prompt:
.venv\Scripts\activate.bat

# 4. Install dependencies
pip install -r requirements.txt
```

#### macOS / Linux (bash / zsh)
```bash
# 1. Navigate to the project directory
cd /path/to/CreditGuardAI

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

#### Using `uv` (Fastest, All Platforms)
If you have `uv` installed:
```bash
uv venv .venv --python 3.12
uv pip install -p .venv -r requirements.txt
```

---

### Running the System

#### 1. Complete End-to-End Pipeline
Executes data ingestion, exploratory charts, classical ML training, PyTorch deep learning, benchmark scoreboard, and applicant inference:

* **Windows:**
  ```powershell
  python main.py --all
  ```
* **macOS / Linux:**
  ```bash
  python3 main.py --all
  ```

---

#### 2. Hardware Acceleration (Parallel Multi-Core Tuning)
Uses all available CPU cores (`n_jobs=-1`, e.g., 24 threads on Ryzen 9 9900X) to run parallel hyperparameter grid searches across 125 fits and compute optimal decision thresholds:

* **Windows:**
  ```powershell
  python main.py --tune
  ```
* **macOS / Linux:**
  ```bash
  python3 main.py --tune
  ```

---

#### 3. Real-Time Applicant Risk Scoring (Demo)
Tests real-time inference on an applicant profile with adjustable coverage thresholds:

```bash
# Standard 0.50 threshold:
python main.py --demo

# Tuned 80% coverage threshold:
python main.py --demo --threshold 0.37
```

Output:
```text
Applicant Profile: {'LIMIT_BAL': 20000.0, 'AGE': 25.0, 'PAY_0': 2.0, 'UTILIZATION_1': 0.95, ...}
 -> Default Probability: 88.4%
 -> Operating Threshold: 0.37
 -> Prediction: DEFAULT
 -> Risk Assessment: HIGH RISK (Declined)
 -> Recommended Action: Reject Application
```

---

#### 4. Switching Ingestion Formats (CSV vs. Excel)
CreditGuard AI natively parses both `.csv` and `.xls` (or `.xlsx`) files with double-header offsets:

```bash
# Explicitly ingest CSV:
python main.py --file "default of credit card clients.csv"

# Explicitly ingest Excel:
python main.py --file "default of credit card clients.xls"
```

---

#### 5. Running Individual Modules
```bash
python main.py --eda               # Generate Exploratory Data Analysis figures
python main.py --train-classical   # Train and cross-validate Scikit-Learn models
python main.py --train-pytorch     # Train PyTorch Deep Learning MLP
python main.py --evaluate          # Generate 4-panel Matplotlib dashboard
```

---

### Viewing Generated Visualizations

After running `--eda`, `--evaluate`, or `--all`, open the `figures/` folder:

* **Windows:**
  ```powershell
  explorer figures
  ```
* **macOS:**
  ```bash
  open figures
  ```
* **Linux:**
  ```bash
  xdg-open figures
  ```

---

## Codebase Quality & Linting

All Python files are statically typed and formatted according to PEP 8 standards. You can verify code quality using `ruff`:

```bash
# Run static analysis and linter:
ruff check .

# Check formatting:
ruff format --check .
```
