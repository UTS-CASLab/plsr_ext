from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import numpy as np
from plsr_ext.rplsr import RPLSR
from plsr_ext.mplsr import MPLSR
from sklearn.metrics import mean_absolute_percentage_error

X, y = load_diabetes(return_X_y=True, scaled=False)
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_update, y_train, y_update = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42)

# Scaling
model1 = RPLSR(n_components=4, forgetting_lambda=1, max_iter=5000, tol=1e-6, scale=True)
model1.fit(X_train, y_train)
model1.update(X_update, y_update, False)
y_pred = model1.predict(X_test)

print("MAPE score Updating (Scaling): ", mean_absolute_percentage_error(y_test, y_pred))
print(model1.C)


merge_X = np.vstack((X_train, X_update))
merge_Y = np.hstack((y_train, y_update))
model2 = MPLSR(n_components=4, max_iter=5000, tol=1e-6, scale=True)
model2.fit(X_train, y_train)
y_pred_2 = model2.predict(X_test)

print("MAPE score Updating and Mean-Std update (Scaling): ", mean_absolute_percentage_error(y_test, y_pred_2))
print(model2.C)






