#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib
from titanic_model.train import train_model
from titanic_model.evaluation import evaluate_model

def make_prediction(pipeline, X_test):
    class_preds = pipeline.predict(X_test)
    prob_preds = pipeline.predict_proba(X_test)[:, -1]
    return class_preds, prob_preds


if __name__ == "__main__":

    file_path = r"C:\Users\nico_\Desktop\repos\titanic_project\titanic_model\dataset\titanic.csv"
    _, X_test, y_test = train_model(file_path)

    model_path = 'titanic_pipeline.pkl'
    pipeline = joblib.load(model_path)

    class_preds, prob_preds = make_prediction(pipeline, X_test)

    evaluate_model(y_test, class_preds, prob_preds)