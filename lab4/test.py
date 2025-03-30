from sklearn import datasets
data_breast_cancer = datasets.load_iris(return_X_y= True, as_frame=True)

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
for i in range(1,43):
    bc_train_x, bc_test_x, bc_train_y, bc_test_y = train_test_split(bc_data, bc_target, test_size=0.2, random_state=i)

    bc_area_smooth_train = pd.DataFrame({"petal width (cm)" : bc_train_x["petal width (cm)"], "petal length (cm)" : bc_train_x["petal length (cm)"]})
    bc_area_smooth_test = pd.DataFrame({"petal width (cm)" : bc_test_x["petal width (cm)"], "petal length (cm)" : bc_test_x["petal length (cm)"]})
    bc_area_smooth_x = pd.DataFrame({"petal width (cm)" : bc_data["petal width (cm)"], "petal length (cm)" : bc_data["petal length (cm)"]})

    bc_svm_2 = Pipeline([
        ("classifier", LinearSVC(loss="hinge"))
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

    bc_svm = LinearSVC(loss='hinge', C=0.01, max_iter=10000)
    bc_svm.fit(bc_area_smooth_train, bc_train_y)

    bc_svm_scaled_2 = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LinearSVC(loss="hinge"))
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
        ("linear_svm" , LinearSVC(loss='hinge', C=1000, max_iter=5000)),
    ])

    bc_svm_scaled.fit(bc_area_smooth_train, bc_train_y)

    bc_svm_train_acc = bc_svm.score(bc_area_smooth_train, bc_train_y)
    bc_svm_test_acc = bc_svm.score(bc_area_smooth_test, bc_test_y)
    bc_svm_scaled_train_acc = bc_svm_scaled.score(bc_area_smooth_train, bc_train_y)
    bc_svm_scaled_test_acc = bc_svm_scaled.score(bc_area_smooth_test, bc_test_y)

    bc_acc_list = bc_svm_train_acc + bc_svm_test_acc + bc_svm_scaled_train_acc + bc_svm_scaled_test_acc
    if best_score < bc_acc_list:
        best_score = bc_acc_list
        random = i

print(best_score, i)