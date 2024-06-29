#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

NUMERICAL_VARIABLES = ["Age", "Fare"]
CATEGORICAL_VARIABLES = ['Sex', 'Cabin', 'Embarked']
CABIN = ['Cabin']

class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables

        
    def fit(self, X, y=None):
        return self

    
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].str[0]
        return X