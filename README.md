# 🎓 Dual-Stage Campus Placement Predictive Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7+-orange.svg)
![Status](https://img.shields.io/badge/Stage-1--Tabular%20Portfolio-success.svg)

An **end-to-end machine learning engineering framework** designed to audit, analyze, and predict university recruitment dynamics using a **dual-stage modeling approach**.

---

# 📑 Table of Contents
- [📌 Project Architecture](#-project-architecture)
- [📊 Core Statistical Insights (EDA Summary)](#-core-statistical-insights-eda-summary)
- [🤖 Model Engineering & Iterative Benchmarks](#-model-engineering--iterative-benchmarks)
- [🔍 Technical Takeaway](#-technical-takeaway)
- [💻 Local Environment Setup & Installation](#-local-environment-setup--installation)
- [🛠️ Built With](#️-built-with)

---

# 📌 Project Architecture

This system addresses **campus placement forecasting** by splitting the problem into **two distinct mathematical pipelines**:

## Stage 1 — Classification
Evaluates student profiles to predict **whether** a candidate will successfully clear recruitment thresholds and secure a placement offer.

**Binary Outcome**
- `1` → Placed
- `0` → Not Placed

## Stage 2 — Regression
Predicts the **specific starting salary package** offered to successfully placed candidates.

**Continuous Value Forecasting**

### Pipeline Flow

```text
[ Raw Tabular CSV ]
        │
        ▼
[ Preprocessing ]
(One-Hot Encoding + Stratified Train/Test Split)
        │
        ├──► [ Stage 1: Logistic Regression ]
        │          └──► Classifies: Placed vs Not Placed
        │
        └──► [ Data Masking (Placed Only) ]
                    └──► [ Stage 2: Log-Linear Regression ]
                                └──► Predicts Salary
```

---

# 📊 Core Statistical Insights (EDA Summary)

### 1. Missing Data Anomalies
- Found exactly **67 null values** isolated entirely in the `salary` column.
- This verified that **unplaced students** had empty salary fields.
- Required **Data Masking** during Stage 2 to prevent zero-value distortions.

### 2. Feature Drivers
Spearman Rank Correlation mapping identified:

| Feature | Correlation Strength |
|---|---:|
| `ssc_p` (Senior Secondary Percentage) | **0.60** |
| `degree_p` (Undergraduate Percentage) | **0.50** |

These emerged as the strongest monotonic indicators of **overall employability**.

### 3. Categorical Expansion
- Implemented **One-Hot Encoding** using `drop_first=True`.
- Expanded feature space by **2 columns**.
- Features such as `hsc_s` and `degree_t` contained **3 unique categories**.

Mathematically:

```text
N categories → N - 1 binary variables
```

This prevented the **Dummy Variable Trap** (multicollinearity).

---

# 🤖 Model Engineering & Iterative Benchmarks

## 🗂️ Stage 1 — Placement Classification

### Model
- **Algorithm:** Logistic Regression
- **Optimization:** Binary Cross-Entropy Loss

### Data Strategy
- **80/20 Train-Test Split**
- Used:

```python
stratify=y
```

This preserved identical class distributions in both subsets.

### Performance
- **Test Accuracy:** **81.40%**

### Confusion Matrix Audit
- Total errors on unseen validation set: **8**
  - **3 False Positives**
  - **5 False Negatives**

This demonstrated a **balanced and reliable decision boundary**.

---

## 📉 Stage 2 — Salary Prediction (Continuous Regression)

Predicting exact wage outcomes exposed a **classic real-world data challenge**.

### Iteration 1 — Standard OLS Regression

| Metric | Value |
|---|---:|
| R² | **-0.1587** |
| RMSE | **98,973.50** |

**Interpretation:**
- Negative R² showed the model performed **worse than predicting the average salary**.
- Strong influence of **right-skewed salary outliers**.

---

### Iteration 2 — Random Forest Regression

| Metric | Value |
|---|---:|
| R² | **-0.3229** |
| RMSE | **105,757.05** |

**Interpretation:**
- Performance degraded further.
- Tree ensemble **overfitted heavily** on extreme salary outliers.
- Limited masked dataset reduced generalization.

---

### Iteration 3 — Logarithmic Target Transformation

Applied:

```python
np.log1p()
```

to compress exponential salary skew.

### Final Results

| Metric | Value |
|---|---:|
| Final RMSE | **95,293.56** |

### Outcome
The logarithmic transformation successfully:
- Stabilized the regression pipeline
- Reduced variance caused by outliers
- Produced the **best-performing salary model**

---

# 🔍 Technical Takeaway

This project demonstrated that:

- Academic scores are strong indicators of **baseline employability threshold clearance**.
- Academic variables lose predictive power over **compensation value**.

Final salary packages are strongly affected by **non-academic variables**, including:
- Technical portfolios
- Interview performance
- Negotiation ability
- Employer-specific hiring dynamics

---

# 💻 Local Environment Setup & Installation

## 1️⃣ Clone Repository & Initialize Environment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Campus-Placement-Prediction.git

# Move into project directory
cd Campus-Placement-Prediction

# Create virtual environment
python -m venv venv
```

### Activate Environment

**Windows (PowerShell)**

```powershell
.\venv\Scripts\Activate.ps1
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

## 2️⃣ Install Core Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn ipykernel
```

---

## 3️⃣ Notebook Kernel Setup (VS Code)

1. Launch **VS Code** inside project directory.
2. Open:

```text
placement_exploration.ipynb
```

3. Click:

```text
Select Kernel → Python Environments → local venv
```

4. Drag `Placement_Data.csv` into the root folder.

This supplies the required **relative dataset path**.

---

# 🛠️ Built With

| Technology | Purpose |
|---|---|
| **Python 3** | Foundational programming language |
| **Pandas & NumPy** | Vectorized matrix operations & data auditing |
| **Scikit-Learn** | Machine learning algorithms & data splitting |
| **Seaborn & Matplotlib** | Statistical visualization & EDA |

---

# ⭐ Project Summary

A **dual-stage predictive ML pipeline** combining:

✅ Placement Classification  
✅ Salary Prediction  
✅ Statistical EDA  
✅ Feature Engineering  
✅ Regression Stabilization via Log Transform  

to model **real-world campus recruitment dynamics** with interpretable machine learning workflows.
