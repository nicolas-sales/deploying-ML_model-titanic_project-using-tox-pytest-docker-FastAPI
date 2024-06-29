#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(y_true, class_preds, prob_preds):
    # ROC-AUC score
    roc_auc = roc_auc_score(y_true, prob_preds)
    print('ROC-AUC score: {:.4f}'.format(roc_auc))
    
    # Accuracy score
    accuracy = accuracy_score(y_true, class_preds)
    print('Accuracy score: {:.4f}'.format(accuracy))