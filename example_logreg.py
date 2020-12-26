#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:48:30 2020

@author: anton
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             roc_auc_score,
                             f1_score,
                             precision_score,
                             recall_score,
                             )

from explain_ml_model_results import ExplainResults
# =============================================================================
# Data preparation
# =============================================================================
breast_cancer = load_breast_cancer()
breast_cancer_df = pd.DataFrame(breast_cancer.data,
                                columns=breast_cancer.feature_names,
                                )
breast_cancer_df['target'] = pd.Series(breast_cancer.target)

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_df.drop('target', axis=1).values,
                                                    breast_cancer_df['target'].values,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42,
                                                    )
# =============================================================================
# Data scaling
# =============================================================================
scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
# =============================================================================
# LogReg classifier create and fit
# =============================================================================
logreg = LogisticRegression(C=0.1)

logreg.fit(X_train_sc, y_train)
# =============================================================================
# Predict targets and proba
# =============================================================================
y_train_pred = logreg.predict(X_train_sc)
y_test_pred = logreg.predict(X_test_sc)
y_train_pred_scores = logreg.predict_proba(X_train_sc)[:, 1]
y_test_pred_scores = logreg.predict_proba(X_test_sc)[:, 1]
# =============================================================================
# Classifier qualite check
# =============================================================================
print("ROC-AUC metric: train: {}, test: {}".format(roc_auc_score(y_train,
                                                                 y_train_pred_scores),
                                                   roc_auc_score(y_test,
                                                                 y_test_pred_scores),
                                                   ),
      )
print("Accuracy metric: train: {}, test: {}".format(accuracy_score(y_train,
                                                                   y_train_pred),
                                                    accuracy_score(y_test,
                                                                   y_test_pred),
                                                    ),
      )
print("F1-score metric: train: {}, test: {}".format(f1_score(y_train,
                                                             y_train_pred),
                                                    f1_score(y_test,
                                                             y_test_pred),
                                                    ),
      )
print("Precision metric: train: {}, test: {}".format(precision_score(y_train,
                                                                     y_train_pred),
                                                     precision_score(y_test,
                                                                     y_test_pred),
                                                     ),
      )
print("Recall metric: train: {}, test: {}".format(recall_score(y_train,
                                                               y_train_pred),
                                                  recall_score(y_test,
                                                               y_test_pred),
                                                  ),
      )
# =============================================================================
# Example id
# =============================================================================
id = 20
# =============================================================================
# Predicted class and probability by classes
# =============================================================================
print('\nPredicted class: {}, proba: {}'.format(logreg.predict(X_test_sc[id, :].reshape(1, -1)),
                                                logreg.predict_proba(X_test_sc[id, :].reshape(1, -1)),
                                                ),
      )
# =============================================================================
# Plot individual SHAP value for observation and save as .html
# =============================================================================
expl = ExplainResults(model=logreg,
                      X_train=X_train_sc,
                      model_type='linear',
                      is_proba=True,
                      scaler=scaler
                      )

expl.single_plot(breast_cancer_df.drop('target', axis=1).columns,
                 X_test_sc[id, :],
                 )
# =============================================================================
# Summary plot features importance and save as .jpeg
# =============================================================================
n_max=5

expl.summary_plot(breast_cancer_df.drop('target', axis=1).columns,
                  X_test_sc,
                  is_bar=False,
                  max_display=len(breast_cancer_df.columns)
                 )
# =============================================================================
# Get impact of each top n_max features
# =============================================================================
pos_imp = expl.get_impact_of_n_max_shap_values(X_test_sc[id, :],
                                               features_list=breast_cancer_df.drop('target', axis=1).columns,
                                               n_max=n_max,
                                               is_pos=True,
                                               )

neg_imp = expl.get_impact_of_n_max_shap_values(X_test_sc[id, :],
                                               features_list=breast_cancer_df.drop('target', axis=1).columns,
                                               n_max=n_max,
                                               is_pos=False,
                                               )
print('\nPositive class features impacts:')
print(pos_imp)

print('\nNegative class features impacts:')
print(neg_imp)
# =============================================================================
# Pie plot for each class
# =============================================================================
expl.pie_plot_impacts_by_classes(pos_imp, neg_imp)
# =============================================================================
# Summary pie plot 
# =============================================================================
expl.pie_plot_summary_impacts(X_test_sc,
                              features_list=breast_cancer_df.drop('target', axis=1).columns,
                              n_max=n_max,
                              )
