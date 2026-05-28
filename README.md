# 🎓 Dual-Stage Campus Placement Predictive Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7+-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Stage-1--Tabular%20Portfolio-success.svg)]()

An end-to-end machine learning engineering framework designed to audit, analyze, and predict university recruitment dynamics using a dual-stage modeling approach. 

---

## 📌 Project Architecture
This system addresses campus placement forecasting by splitting the problem into two distinct mathematical pipelines:

1. **Stage 1 (Classification):** Evaluates student profiles to predict *whether* a candidate will successfully clear recruitment thresholds and secure a placement offer (Binary Outcome: `1` for Placed, `0` for Not Placed).
2. **Stage 2 (Regression):** Predicts the *specific starting salary package* offered to successfully placed candidates (Continuous Value Forecasting).

[ Raw Tabular CSV ]│▼[ Preprocessing ] ──► (One-Hot Encoding, Stratified Train/Test Split)│├───► [ Stage 1: Logistic Regression ] ──► Classifies: Placed vs Not Placed│└───► [ Data Masking (Placed Only) ] ──► [ Stage 2: Log-Linear Regression ] ──► Predicts Salary
---

## 📊 Core Statistical Insights (EDA Summary)
* **Missing Data Anomalies:** Found exactly 67 null values isolated entirely in the `salary` column. This perfectly verified that unplaced students had empty fields, requiring rigorous **Data Masking** during Stage 2 to prevent zero-value distortions.
* **Feature Drivers:** Spearman Rank Correlation mapping identified Senior Secondary Percentage (`ssc_p`) at **0.60** and Undergrad Percentage (`degree_p`) at **0.50** as the primary monotonic drivers of overall employability.
* **Categorical Expansion:** Implementing One-Hot Encoding with `drop_first=True` expanded the feature space by 2 columns. Features like high school stream (`hsc_s`) and undergrad degree type (`degree_t`) contained 3 unique options, which mathematically resolve into $N-1$ (2) binary channels to prevent the *Dummy Variable Trap* (multicollinearity).

---

## 🤖 Model Engineering & Iterative Benchmarks

### 🗂️ Stage 1: Placement Classification
* **Model:** Logistic Regression optimized via Binary Cross-Entropy Loss.
* **Data Strategy:** Implemented an 80/20 train-test split stratified against the target label (`stratify=y`) to maintain identical class ratios in both subsets.
* **Metric:** Achieved a definitive test prediction accuracy of **81.40%**. 
* **Confusion Matrix Audit:** Out of the unseen validation subset, the boundary made only 8 total errors (3 False Positives / 5 False Negatives), proving a highly balanced decision boundary.

### 📉 Stage 2: Salary Prediction (Continuous Regression)
Predicting exact wage outcomes highlighted a classic real-world data challenge:
* **Iteration 1 (Standard OLS):** Yielded an $R^2$ of **-0.1587** with an RMSE of **98,973.50**. The negative $R^2$ proved the straight line performed worse than predicting a simple flat average, heavily distorted by right-skewed wage outliers.
* **Iteration 2 (Random Forest):** Degraded performance further ($R^2$: **-0.3229**, RMSE: **105,757.05**). The tree ensemble overfitted drastically on extreme salary outliers within a constrained, masked data volume.
* **Iteration 3 (Logarithmic Target Transformation):** By applying a natural log transform (`np.log1p`) to the target, the exponential skew was completely compressed. This successfully stabilized the regression pipeline, dropping the final salary prediction error (RMSE) down to **95,293.56**.

### 🔍 Technical Takeaway
This project successfully demonstrated that academic scores are phenomenal indicators of baseline **employability threshold clearance** (Stage 1), but lose their linear predictive power over **compensation value** (Stage 2). Final starting packages are heavily driven by variables omitted from standard academic records (e.g., technical portfolios, negotiation dynamics, and interview panel performance).

---

## 💻 Local Environment Setup & Installation

To run this pipeline locally on your machine using VS Code, execute the following workflow:

### 1. Clone the Workspace & Initialize Environment
```bash
# Clone this repository
git clone [https://github.com/YOUR_USERNAME/Campus-Placement-Prediction.git](https://github.com/YOUR_USERNAME/Campus-Placement-Prediction.git)
cd Campus-Placement-Prediction

# Create an isolated Python virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\Activate.ps1
# On Mac/Linux:
source venv/bin/activate
2. Download Core Tabular DependenciesBashpip install pandas numpy matplotlib seaborn scipy scikit-learn ipykernel
3. Notebook Connection SetupLaunch VS Code inside this project directory.Open placement_exploration.ipynb.In the top-right corner, click Select Kernel $\rightarrow$ Python Environments... $\rightarrow$ select the local venv path.Drag your Placement_Data.csv directly into the root folder to supply the relative data path runner.🛠️ Built WithPython 3 - Foundational Programming LanguagePandas & NumPy - Vectorized Matrix Transformations & Data AuditingScikit-Learn - Machine Learning Algorithm Implementations & Data SplittingSeaborn & Matplotlib - Statistical EDA Data Visualizations