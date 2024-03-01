from sklearn.datasets import make_regression
import numpy as np
from plsr_ext.jit_plsr import JIT_PLSR
from sklearn.metrics import mean_absolute_percentage_error

X_train, y_train = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
X_test, y_test = make_regression(n_features=4, n_informative=2, random_state=42, shuffle=False)

model = JIT_PLSR(max_n_components=3, k_nearest=10, k_fold=3, scoring='neg_mean_absolute_percentage_error', sim_metric='euclidean')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAPE score: ", mean_absolute_percentage_error(y_test, y_pred))


