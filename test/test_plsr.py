from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import numpy as np
from plsr_ext.plsr import PLSR
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

#X, y = load_diabetes(return_X_y=True, scaled=False)
diabetes = load_diabetes(scaled=False)
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_test)
# Scaling
model1 = PLSR(n_components=4, max_iter=500, tol=1e-06, scale=True)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)

print("MAPE score (Scaling): ", mean_absolute_percentage_error(y_test, y_pred))
print("R2 score (Scaling): ", r2_score(y_test, y_pred))
print(y_pred)
# Normalising
model2 = PLSR(n_components=4, max_iter=500, tol=1e-06, scale=False)
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)

print("MAPE score (No scaling): ", mean_absolute_percentage_error(y_test, y_pred))
print("R2 score (No Scaling): ", r2_score(y_test, y_pred))
print(y_pred)

from sklearn.cross_decomposition import PLSRegression

sk_plsr = PLSRegression(n_components=4, max_iter=500, tol=1e-06)
sk_plsr.fit(X_train, y_train)
y_pred_sk = sk_plsr.predict(X_test).ravel()
print("MAPE score (sklearn): ", mean_absolute_percentage_error(y_test, y_pred_sk))
print("R2 score (sklearn): ", r2_score(y_test, y_pred_sk))
print(y_pred_sk)