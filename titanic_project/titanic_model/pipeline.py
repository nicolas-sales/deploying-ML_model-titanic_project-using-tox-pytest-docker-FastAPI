#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Répertoire parent au PYTHONPATH, garantit que les modules sont accessibles peu importe d'où le script est exécuté
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from titanic_model.data_processing import (
    ExtractLetterTransformer,
    NUMERICAL_VARIABLES,
    CATEGORICAL_VARIABLES,
    CABIN
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)
from feature_engine.encoding import (
    RareLabelEncoder,
    OneHotEncoder)

def create_pipeline():
    titanic_pipeline = Pipeline([
        ('categorical_imputation', CategoricalImputer(
            imputation_method='missing', variables=CATEGORICAL_VARIABLES)),
        ('missing_indicator', AddMissingIndicator(variables=NUMERICAL_VARIABLES)),
        ('median_imputation', MeanMedianImputer(
            imputation_method='median', variables=NUMERICAL_VARIABLES)),
        ('extract_letter', ExtractLetterTransformer(variables=CABIN)),
        ('rare_label_encoder', RareLabelEncoder(
            tol=0.05, n_categories=1, variables=CATEGORICAL_VARIABLES)),
        ('categorical_encoder', OneHotEncoder(
            drop_last=True, variables=CATEGORICAL_VARIABLES)),
        ('scaler', StandardScaler()),
        ('Logit', LogisticRegression()),
    ])
    return titanic_pipeline