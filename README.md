# CreditGuard AI

CreditGuard AI is a machine learning tool that predicts whether a credit card customer is likely to default on their next payment. 

Built with **Python**, **Pandas**, **NumPy**, **Scikit-Learn**, **PyTorch**, and **Matplotlib**, it analyzes customer spending patterns, billing statements, and past payment history to evaluate credit risk.

---

## Overview

* **The Problem:** Credit card default occurs when a customer fails to make required payments, resulting in financial loss for the lender. CreditGuard AI helps lenders identify high-risk accounts early so they can make informed underwriting decisions.
* **The Data:** Uses 30,000 customer records from the UCI Machine Learning Repository, tracking credit limits, demographic info, and 6 months of billing and payment history.
* **Smart Ratios:** Automatically calculates key financial indicators, such as credit card utilization (how close a customer is to their limit) and repayment trends (whether payments are improving or falling behind).
* **Models Compared:** Trains and benchmarks 4 different models—**Logistic Regression**, **Random Forest**, **Gradient Boosting**, and a **PyTorch Neural Network**—and outputs a comparative performance scoreboard and visual charts.
* **Real-Time Scoring:** Includes a live underwriter demo that takes an applicant's numbers and returns an instant credit decision (`Approved`, `Manual Review`, or `Declined`).

---

## Quick Start (Cross-Platform)

CreditGuard AI works on **Windows**, **macOS**, and **Linux**.

### 1. Setup Environment

#### Windows (PowerShell)
```powershell
cd C:\Users\juder\CreditGuardAI
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### macOS / Linux (Terminal)
```bash
cd /path/to/CreditGuardAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. How to Run

#### Run the Entire Pipeline
Loads the data, trains all models, prints the benchmark scoreboard, and saves all visual charts:
```bash
python main.py --all
```

#### Enable GPU Acceleration (NVIDIA or AMD Radeon)
Automatically uses your GPU (NVIDIA CUDA or AMD Radeon via DirectML) when passed, or defaults to multi-core CPU:
```bash
python main.py --all --gpu
```

#### Test a Real-Time Applicant (Demo)
Evaluates a sample applicant and outputs an instant risk score and decision:
```bash
python main.py --demo
```

#### Use All CPU Hardware (Multi-Core Tuning)
Runs parallel hyperparameter tuning across all your CPU cores to optimize the model and decision threshold:
```bash
python main.py --tune
```

#### Ingest CSV or Excel Files Explicitly
CreditGuard AI automatically supports both formats:
```bash
# Using CSV:
python main.py --file "data/default of credit card clients.csv"

# Using Excel:
python main.py --file "data/default of credit card clients.xls"
```

#### Run Individual Steps
```bash
python main.py --eda               # Generate exploratory data charts
python main.py --train-classical   # Train Scikit-Learn models only
python main.py --train-pytorch     # Train PyTorch Neural Network only
python main.py --evaluate          # Generate 4-panel comparison dashboard
```

---

### 3. Viewing Visual Charts

All charts are saved directly into the **`figures/`** folder:

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

Key charts included:
* `model_benchmark_dashboard.png`: 4-panel dashboard comparing all models (ROC curves, confusion matrix, feature importance).
* `threshold_coverage_tradeoff.png`: Visualizes how decision thresholds affect default detection rates.
* `eda_class_distribution.png`: Breakdown of settled vs. defaulted accounts.
* `eda_delinquency_vs_default.png`: Shows how missed payments increase default risk.
