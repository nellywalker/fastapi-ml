# model_training.py
import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load data
iris = load_iris()
X, y = iris.data, iris.target
# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)
# Save model
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)