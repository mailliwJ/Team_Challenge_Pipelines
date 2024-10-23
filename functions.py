import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

def compare_cross_val(estimators, names, X_train, y_train, *, n_cv = 5, scoring):
    val_score = []
    for model in estimators:
        metric = np.mean(cross_val_score(model, X_train, y_train, cv = n_cv, scoring = scoring, n_jobs = -1))
        val_score.append(metric)
    
    df = pd.DataFrame({'Model': names, 'Scoring': scoring, 'Validation Score': val_score})
    return df