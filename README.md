# 🏦 **Lauki Finance: Credit Risk Modelling (Classification Model)**

*A portfolio project built as part of the CodeBasics Gen AI & Data Science Bootcamp*

---

## 🚀 **Project Overview**

This project is a complete **end-to-end Credit Risk Modelling solution** that predicts the **default probability** of loan applicants and generates a **credit score (300–900)** along with actionable risk ratings (Excellent → Poor).

The project simulates a real engagement with **Lauki Finance**, an NBFC that provides loans to customers underserved by traditional banks. Their rapid growth demands a modern, AI-powered system to evaluate borrower risk efficiently and consistently.

The final solution includes:

✔ A full data cleaning & preprocessing pipeline
✔ Exploratory data analysis & business-rule validation
✔ Advanced feature engineering
✔ Model training with imbalance handling (RUS + SMOTETomek)
✔ Hyperparameter tuning using Optuna
✔ Model packaging & versioning
✔ Deployment-ready Streamlit UI
✔ Automated PDF credit report generator

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
│   └── models/                       # EDA/testing pickles
│   └── tables/                       # Rank Ordering tables
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

This is a **binary classification** problem where:

| Target Variable State | Numeric Value | Business Meaning |           Risk Category            | 
| --------------------- | ------------- | ---------------- | ---------------------------------- |
| Default = False       |        0      | Non-Defaulter    | "Good" Customer (Desired outcome)  |
| Default = True        |        1      | Defaulter        | "Bad" Customer (Event of Interest) |


### 📌 Models Explored:

| Model               | Notes                                   |
| ------------------- | --------------------------------------- |
| Logistic Regression | Interpretable, regulation-friendly      |
| Random Forest       | High performance, less explainable      |
| XGBoost             | Strong predictive power, black-box risk |

---

# 🧹 **Data Cleaning & Business Rule Validation**

Before modeling, several **business rules** were implemented:

### ✔ Processing Fee Validation

* Must NOT exceed **3%** of loan amount
* 5 records violated the rule → removed

### ✔ GST Validation

* Must NOT exceed **20%** of loan amount

### ✔ Net Disbursement

* Must be ≤ loan amount

### ✔ Imputation

* Missing categorical: mode
* Missing numeric: domain-informed logic

This ensured a **clean, regulation-ready dataset**.

---

# 🏗️ **Feature Engineering**

### Engineered Key Features

| Feature                             | Why It’s Important                          |
| ----------------------------------- | ------------------------------------------- |
| `loan_to_income`                    | Strongest predictor of repayment capability |
| `delinquency_ratio`                 | Indicates past payment behavior             |
| `avg_dpd_per_delinquency`           | Stability of repayments                     |
| One-hot encoded residence/loan type | Behavioral segmentation                     |
| MinMax scaling                      | Ensures model stability                     |

All transformations were saved with the model to ensure deployability.

---

# ⚖️ **Handling Class Imbalance**

Default class was **highly imbalanced**.
Techniques attempted:

### 1️⃣ Random Undersampling

* Pros: Balanced data
* Cons: Loss of information

### 2️⃣ SMOTE + Tomek Links (Final Choice)

* Pros: Synthetic minority samples + noise removal
* Stable decision boundary
* Best recall score

---

# 🔧 **Hyperparameter Optimization (Optuna)**

A search space was designed for:

* C
* Solver
* Tolerance
* Class weights

Final params:

```
C = 4.46
solver = 'saga'
tol = 6.3e-06
class_weight = 'balanced'
```

---

# 📈 **Model Evaluation**

### ✔ Classification Report (Final Model)

| Metric        | Default Class Result            |
| ------------- | ------------------------------- |
| **Recall**    | **94%** ✔ Meets business target |
| **Precision** | **56%** ✔ Meets business target |
| **F1 Score**  | 70%                             |

---

### ✔ ROC-AUC = **0.98**

Outstanding discrimination capability.

### ✔ Gini Coefficient = **0.96**

Strong rank ordering.

### ✔ KS Statistic = **85.9%**

Excellent separation between “Good” and “Bad” customers.

### ✔ Decile Ordering

Monotonic ordering achieved → highly deployment-ready.

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
![PDF Report Demo](<img width="772" height="647" alt="Screenshot 2025-11-28 at 11 03 15" src="https://github.com/user-attachments/assets/4e8624f1-5685-43c4-bd73-9c3995ffa895" />)

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

* **CodeBasics Team** – for industry-grade project design
* **Dhaval Patel & Hemanand Vadivel** – for guidance
* **AtliQ.ai fictional team** – for driving the narrative

---

# 👩‍💻 **Author**

**Ruchitha Uppuluri**
Aspiring Data Scientist | CodeBasics ML Bootcamp

🔗 LinkedIn: [https://www.linkedin.com/in/ruchithauppuluri](https://www.linkedin.com/in/ruchithauppuluri)

---
