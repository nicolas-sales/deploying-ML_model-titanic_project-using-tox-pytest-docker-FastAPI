#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import pytest
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from titanic_model.load_data import load_data
from titanic_model.pipeline import create_pipeline
from titanic_model.predict import make_prediction

@pytest.fixture
def sample_input_data():
    
    test_file_path = r'C:\Users\nico_\Desktop\repos\titanic_project\titanic_model\dataset\titanic.csv'
    data = load_data(test_file_path)
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_test, y_test

def test_make_prediction(sample_input_data):
    X_test, y_test = sample_input_data

    model_path = 'titanic_pipeline.pkl'
    pipeline = joblib.load(model_path)

    class_preds, prob_preds = make_prediction(pipeline, X_test)

    # Vérification du type de prédictions
    assert isinstance(class_preds, np.ndarray)
    assert isinstance(prob_preds, np.ndarray)
    assert isinstance(class_preds[0], np.int64)
    assert isinstance(prob_preds[0], np.float64)
    
    # Vérification du nombre de prédictions
    assert len(class_preds) == len(X_test)
    assert len(prob_preds) == len(X_test)
    
    # Vérification des scores:
    accuracy = accuracy_score(y_test, class_preds)
    roc_auc = roc_auc_score(y_test, prob_preds)
    
    assert accuracy > 0.7
    assert roc_auc > 0.7

