#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
from titanic_model.load_data import load_data
from titanic_model.pipeline import create_pipeline
import joblib

def split_data(data, test_size=0.2, random_state=0):
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_model(file_path):
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(data)

    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, 'titanic_pipeline.pkl')

    return pipeline, X_test, y_test


if __name__ == "__main__":
    file_path = r"C:\Users\nico_\Desktop\repos\titanic_project\titanic_model\dataset\titanic.csv"
    pipeline, X_test, y_test = train_model(file_path)