from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import numpy as np
from plsr_ext.rplsr import RPLSR
from sklearn.metrics import mean_absolute_percentage_error

X, y = load_diabetes(return_X_y=True, scaled=False)
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_update, y_train, y_update = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42)
print(y_test)
# Scaling
model1 = RPLSR(n_components=4, forgetting_lambda=1, max_iter=5000, tol=1e-6, scale=True)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)

print("MAPE score (Scaling): ", mean_absolute_percentage_error(y_test, y_pred))
print(y_pred)
model1.update(X_update, y_update, True)
y_pred = model1.predict(X_test)

print("MAPE score Updating and Mean-Std update (Scaling): ", mean_absolute_percentage_error(y_test, y_pred))
print(y_pred)

print("Incremental learning")
model2 = RPLSR(n_components=4, forgetting_lambda=1, max_iter=5000, tol=1e-6, scale=True)
model2.fit(X_train, y_train)
for i in range(X_update.shape[0]):
    model2.update(X_update[i], y_update[i], True)

    y_pred_2 = model2.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_2)
    print("(Scaling) MAPE score Incremental learning and updating mean + std: ", mape)

model3 = RPLSR(n_components=4, forgetting_lambda=1, max_iter=5000, tol=1e-6, scale=False)
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)
print("MAPE score (No scaling): ", mean_absolute_percentage_error(y_test, y_pred))
print(y_pred)

model3.update(X_update, y_update, False)
y_pred = model3.predict(X_test)
print("MAPE score Updating (No scaling): ", mean_absolute_percentage_error(y_test, y_pred))
print(y_pred)

print("Incremental learning")
model4 = RPLSR(n_components=4, forgetting_lambda=1, max_iter=5000, tol=1e-6, scale=False)
model4.fit(X_train, y_train)
for i in range(X_update.shape[0]):
    model4.update(X_update[i], y_update[i], False)

    y_pred_4 = model4.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_4)
    print("(No Scaling) MAPE score Incremental learning: ", mape)






