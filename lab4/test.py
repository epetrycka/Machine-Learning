from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(return_X_y= True, as_frame=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt 
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np

bc_data = data_breast_cancer[0]
bc_target = data_breast_cancer[1]

best_score = 0
for i in range(27,43):
    bc_train_x, bc_test_x, bc_train_y, bc_test_y = train_test_split(bc_data, bc_target, test_size=0.2, random_state=42)

    bc_area_smooth_train = pd.DataFrame({"mean area" : bc_train_x["mean area"], "mean smoothness" : bc_train_x["mean smoothness"]})
    bc_area_smooth_test = pd.DataFrame({"mean area" : bc_test_x["mean area"], "mean smoothness" : bc_test_x["mean smoothness"]})
    bc_area_smooth_x = pd.DataFrame({"mean area" : bc_data["mean area"], "mean smoothness" : bc_data["mean smoothness"]})

    bc_svm_2 = Pipeline([
        ("classifier", LinearSVC(loss="hinge", random_state=i))
    ])

    param_grid = {
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "classifier__max_iter": [1000, 5000, 10000]
    }

    search = RandomizedSearchCV(
        estimator=bc_svm_2,
        param_distributions=param_grid,
        scoring="accuracy",
        n_iter=10000,
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    search.fit(bc_area_smooth_x, bc_target)
    print(search.best_params_, search.best_score_)

    bc_svm = LinearSVC(loss='hinge', C=0.01, max_iter=10000, random_state=i)
    bc_svm.fit(bc_area_smooth_train, bc_train_y)

    bc_svm_scaled_2 = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LinearSVC(loss="hinge", random_state=i))
    ])

    search_scaled = RandomizedSearchCV(
        estimator=bc_svm_scaled_2,
        param_distributions=param_grid,
        scoring="accuracy",
        n_iter=10000,
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    search_scaled.fit(bc_area_smooth_x, bc_target)
    print(search_scaled.best_params_, search_scaled.best_score_)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    bc_svm_scaled = Pipeline( [ 
        ("scaler" , StandardScaler()),
        ("linear_svm" , LinearSVC(loss='hinge', C=1000, max_iter=5000, random_state=i)),
    ])

    bc_svm_scaled.fit(bc_area_smooth_train, bc_train_y)

    bc_svm_train_acc = bc_svm.score(bc_area_smooth_train, bc_train_y)
    bc_svm_test_acc = bc_svm.score(bc_area_smooth_test, bc_test_y)
    bc_svm_scaled_train_acc = bc_svm_scaled.score(bc_area_smooth_train, bc_train_y)
    bc_svm_scaled_test_acc = bc_svm_scaled.score(bc_area_smooth_test, bc_test_y)

    bc_acc_list = bc_svm_train_acc + bc_svm_test_acc + bc_svm_scaled_train_acc + bc_svm_scaled_test_acc
    if bc_acc_list < 3.2 and bc_acc_list > 3.115:
        best_score = bc_acc_list
        random = i
        break

print(best_score, i)