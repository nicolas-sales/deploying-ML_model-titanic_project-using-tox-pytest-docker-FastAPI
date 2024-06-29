#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data