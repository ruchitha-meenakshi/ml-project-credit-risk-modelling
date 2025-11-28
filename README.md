# 🏦 **Lauki Finance: Credit Risk Modelling (Classification Model)**

*A portfolio project built as part of the CodeBasics Gen AI & Data Science Bootcamp*

---

## 🚀 **Project Overview**

This project is a complete **end-to-end Credit Risk Modelling solution** that predicts the **default probability** of loan applicants and generates a **credit score (300–900)** along with actionable risk ratings (Excellent → Poor).

The project simulates a real engagement with **Lauki Finance**, an NBFC that provides loans to customers underserved by traditional banks. Their rapid growth demands a modern, AI-powered system to evaluate borrower risk efficiently and consistently.

The final solution includes:

- A full data cleaning & preprocessing pipeline
- Exploratory data analysis & business-rule validation
- Advanced feature engineering
- Model training with imbalance handling (RUS + SMOTETomek)
- Hyperparameter tuning using Optuna
- Model packaging & versioning
- Deployment-ready Streamlit UI
- Automated PDF credit report generator

---

## 📌 **Live App:**

🔗 **[https://ml-project-credit-risk-modelling-codebasics.streamlit.app/](https://ml-project-credit-risk-modelling-codebasics.streamlit.app/)**

## 📌 **GitHub Repository:**

🔗 [https://github.com/ruchitha-meenakshi/ml-project-credit-risk-modelling](https://github.com/ruchitha-meenakshi/ml-project-credit-risk-modelling)

---

# 🧩 **Business Story: Lauki Finance’s Transformation**

After the success of the S.H.I.E.L.D. Insurance project, Bruce Harley and his AI startup **AtliQ.ai** quickly gained industry attention.
One of the first calls came from **Steve Singh**, the Head of **Lauki Finance**, an NBFC serving borrowers who are unable to obtain loans from traditional banks.

### **The problem?**

Lauki Finance relied heavily on:

* Manual credit evaluation
* Inconsistent decisions
* Slow processing times
* High operational overhead

These bottlenecks restricted business growth and limited their ability to scale.

Seeing potential in Steve’s vision, Bruce reached out to Tony at AtliQ.ai to build a system that could:

⚡ Automate credit decisioning  
⚡ Reduce dependency on manual reviews  
⚡ Improve risk prediction accuracy  
⚡ Support faster loan approvals  
⚡ Maintain transparency & explainability for regulatory needs

Tony assigned **Peter** as the project lead. What began as a “small assignment” turned into a foundational AI initiative for Lauki Finance.

---

# 🎯 **Project Objectives**

### 🎯 Primary Goal

Build a machine learning model that predicts **loan default probability** and generates an **interpretable credit score**.

### 📌 Success Criteria (Defined by Lauki Finance)

| Metric                        | Target                                       |
| ----------------------------- | -------------------------------------------- |
| **Recall (Default Class)**    | > 90%                                        |
| **Precision (Default Class)** | > 50%                                        |
| **Explainability**            | Must support rule-based interpretation       |
| **Rank Ordering**             | Must exhibit monotonic decile-level ordering |
| **Real-time Deployment**      | Streamlit web app for decision automation    |

---

# 🧱 **Project Structure**

```
ml-project-credit-risk-modelling
│
├── app/                              # Streamlit web application
│   ├── main.py                       # UI + feature collection
│   ├── prediction_helper.py          # Feature engineering + model scoring
│   └── report_generator.py           # PDF credit report generator
│
├── artifacts/                        # Final model, scaler & metadata
│   └── model_data.joblib
│
├── data/
│   ├── raw/                          # NOT uploaded (proprietary)
│   └── processed/                    # Cleaned datasets NOT uploaded (proprietary)
│
├── outputs/                          # EDA, model evaluation artifacts
│   ├── figures/                      # ROC, KS, Rank Ordering plots
│   ├── models/                       # EDA/testing pickles
│   └── tables/                       # Rank Ordering tables
│
├── scripts/                          # Jupyter notebooks
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── imports.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 🧠 **Technical Stack**

### 💻 Languages & Libraries

* Python 3.10
* Pandas, NumPy
* Scikit-Learn
* XGBoost
* Imbalanced-learn (SMOTE-Tomek)
* Optuna (Hyperparameter tuning)
* Matplotlib, Seaborn
* Joblib
* Streamlit
* FPDF (PDF report generation)

---

# 🔍 **Model Overview**

This project addresses a **binary classification** problem where the objective is to estimate the likelihood of a borrower defaulting on a loan. The target variable is defined as follows:

| Target Class | Value      | Business Interpretation                                 |
| ------------ | ---------- | ------------------------------------------------------- |
| **0**        | No Default | Customer is expected to repay (Good / Non-Event)        |
| **1**        | Default    | Customer is likely to default (Bad / Event of Interest) |

Lauki Finance’s business requirement strongly emphasizes **high recall on the defaulter class**, ensuring fewer risky applicants are incorrectly approved.

### 📌 **Models Evaluated**

A range of models were explored to balance predictive performance and explainability:

| Model                   | Rationale                                                                                                               |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | Highly interpretable; aligns with financial regulatory expectations; easy to translate coefficients into business rules |
| **Random Forest**       | Strong non-linear modelling capability; good baseline for complex interactions                                          |
| **XGBoost**             | High predictive power; excellent handling of tabular data, though lower interpretability                                |

Logistic Regression was selected as the **final model** due to its strong performance, interpretability, and alignment with business constraints.

---

## **Data Cleaning & Business Rule Enforcement**

Prior to modelling, several domain-driven validation rules were applied to ensure data quality and regulatory readiness:

### Business Rule Validations

| Rule                                   | Description                                | Outcome                    |
| -------------------------------------- | ------------------------------------------ | -------------------------- |
| **Processing Fee ≤ 3% of Loan Amount** | Ensures fees are realistic                 | 5 records removed          |
| **GST ≤ 20%**                          | Tax amount validation                      | No violations detected     |
| **Net Disbursement ≤ Loan Amount**     | Prevents negative or inflated disbursement | No violations detected     |
| **Missing Values Handling**            | Categorical → Mode; Numeric → Domain logic | All missing values imputed |

These checks ensured a **clean and credible modelling dataset** aligned with financial industry standards.

---

## **Feature Engineering**

A combination of domain knowledge and statistical insights informed the creation of high-impact features:

| Feature                                  | Contribution                                                    |
| ---------------------------------------- | --------------------------------------------------------------- |
| **Loan-to-Income Ratio**                 | Primary indicator of repayment capacity                         |
| **Delinquency Ratio (%)**                | Measures historical repayment discipline                        |
| **Avg DPD per Delinquency**              | Captures severity of past delays                                |
| **Credit Utilization Ratio (%)**         | Reflects overall credit stress                                  |
| **One-hot encoded categorical features** | Enables behavioural segmentation                                |
| **MinMax Scaling**                       | Stabilizes model training and ensures coefficient comparability |

All transformations were persisted to ensure **consistent preprocessing during deployment**.

---

## **Class Imbalance Strategy**

The dataset exhibited a **significant imbalance** toward non-defaulters. Two approaches were tested:

### 1️⃣ **Random Undersampling**

* Balanced the classes but removed valuable majority-class information.

### 2️⃣ **SMOTE + Tomek Links (Final Approach)**

* Generated synthetic minority samples while removing borderline/noisy observations.
* Provided the most stable performance with strong recall on the minority (default) class.

This combination produced a balanced training dataset without distorting real-world distributions.

---

## **Hyperparameter Optimization (Optuna)**

A targeted Optuna search optimized Logistic Regression parameters, exploring:

* Regularization strength (**C**)
* Optimization algorithms (**solver**)
* Convergence threshold (**tol**)
* Class weighting strategies

**Best Parameters Identified:**

```text
C = 4.46
solver = "saga"
tol = 6.3e-06
class_weight = "balanced"
```

These settings maximized recall while maintaining model stability and interpretability.

---

## 📈 **Model Performance Evaluation**

### **Classification Metrics (Default Class)**

| Metric        | Value    | Status                           |
| ------------- | -------- | -------------------------------- |
| **Recall**    | **94%**  | ✔ Meets business target          |
| **Precision** | **56%**  | ✔ Acceptable (with human review) |
| **F1-Score**  | **0.70** | Balanced performance             |

---

### **Robustness Metrics**

| Metric                   | Value     | Interpretation                                |
| ------------------------ | --------- | --------------------------------------------- |
| **ROC–AUC**              | **0.98**  | Exceptional discriminatory power              |
| **Gini Coefficient**     | **0.96**  | Strong rank ordering capability               |
| **KS Statistic**         | **85.9%** | Excellent separation of good vs bad customers |
| **Decile Rank Ordering** | Achieved  | Higher deciles capture higher-risk borrowers  |

Collectively, these metrics confirm that the model is **deployment-ready**, exhibits strong discriminatory power, and satisfies Lauki Finance’s operational and regulatory needs.

---

# 🌐 **Streamlit App**

The deployed Streamlit application enables:

✔ Real-time data entry  
✔ Automated feature engineering  
✔ Probability of default calculation  
✔ Credit score generation (300–900)  
✔ Risk category: Excellent / Good / Average / Poor  
✔ Downloadable PDF credit report  
✔ Clean, modern UI with custom CSS

📌 **Live App:** [https://ml-project-credit-risk-modelling-codebasics.streamlit.app/](https://ml-project-credit-risk-modelling-codebasics.streamlit.app/)

## 🧾 **Automated PDF Report Generation**

After generating a prediction, the app allows users to **download a complete credit decision report** as a PDF.

**The report includes:**

* Default Probability
* Calculated Credit Score
* Final Rating
* Full list of model input parameters

This feature simulates a **real NBFC workflow**, enabling easy auditing and documentation of credit decisions.

📌 **Sample PDF Output:**  
<img width="772" height="647" alt="Screenshot 2025-11-28 at 11 03 15" src="https://github.com/user-attachments/assets/4e1a0be8-f0c3-45b1-a949-83deb897376e" />

---

# 🎨 **App Preview**

### 🎥 Demo Video

https://github.com/user-attachments/assets/4adfe1b8-4013-4069-88e9-e27b2c472db0

---

# 🔒 **Data Privacy Notice**

The dataset used for this project is provided exclusively as part of the CodeBasics Bootcamp and is **NOT publicly distributable**.

To comply with licensing:

* `data/raw/` and `data/processed/` are added to `.gitignore`
* Only `.gitkeep` placeholder files are included
* No proprietary data is uploaded

---

# 🛠 **How to Run Locally**

### Step 1 — Clone the repository

```bash
git clone https://github.com/ruchitha-meenakshi/ml-project-credit-risk-modelling.git
cd ml-project-credit-risk-modelling
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Launch the application

```bash
streamlit run app/main.py
```

---

# 🌱 **Learnings from the Project**

This project helped me strengthen:

✔ Business-first problem framing
✔ Data cleaning using domain rules, not just statistics
✔ Feature engineering for credit datasets
✔ Model training on imbalanced classification problems
✔ Hyperparameter tuning with Optuna
✔ Understanding of regulatory requirements for ML models
✔ Building a production-grade Streamlit app
✔ PDF generation & real-time scoring workflow
✔ Managing model artifacts & reproducibility

---

# 🙌 **Acknowledgements**

Special thanks to:

* **CodeBasics Bootcamp** – for industry-grade project design
* **Dhaval Patel, Hemanand Vadivel & Team** – for guidance
* **AtliQ.ai & Lauki Finance fictional team** – for driving the narrative

---

# 👩‍💻 **Author**

**Ruchitha Uppuluri**
Aspiring Data Scientist | CodeBasics ML Bootcamp

🔗 LinkedIn: [https://www.linkedin.com/in/ruchithauppuluri](https://www.linkedin.com/in/ruchithauppuluri)

---
