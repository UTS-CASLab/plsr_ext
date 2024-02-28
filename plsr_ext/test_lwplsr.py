from sklearn.datasets import make_regression
import numpy as np
from plsr_ext.lwplsr import LWPLSR

X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
model = LWPLSR(n_components=2, lambda_in_similarity=0.01, sim_metric='euclidean')

model.fit(X, y)

# Make prediction
print(model.predict(np.array([1.76405235,  0.40015721,  0.97873798,  2.2408932], dtype=float)))

print(model.predict(np.array([0, 0, 0, 0], dtype=float)))
