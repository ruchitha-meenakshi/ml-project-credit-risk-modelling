# scripts/imports.py

# Common scientific libraries
import pandas as pd
import numpy as np
import math

# formatting 
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
np.set_printoptions(suppress=True)

# Visualization
from matplotlib import pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Model training
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import uniform, randint

# Model tuning
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import optuna

# Evaluation
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.metrics import make_scorer, f1_score

# File helpers
from joblib import dump, load

# Parquet helper
def load_parquet(path):
    """Load a parquet file safely."""
    return pd.read_parquet(path)

def save_parquet(df, path):
    """Save parquet with a consistent pattern."""
    df.to_parquet(path, index=False)

# Joblib helper
def load_model(path):
    """Uniform loader for joblib models."""
    return load(path)

def save_model(obj, path):
    """Uniform saver for joblib models."""
    dump(obj, path)