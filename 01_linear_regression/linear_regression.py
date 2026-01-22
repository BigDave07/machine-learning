
import numpy as np
from numpy import random as rn
from pathlib import Path
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate Dataset
n = 100
X = np.random.uniform(0, 10, n)    #Hours
noise = np.random.normal(0, 2, n)  #Noise
y = 8 * X + 20 + noise              #Scores


# -----------------------------
# STEP 2: Model + Loss
# y_hat = w*x + b
# -----------------------------

w = rn.randn()  # weight
b = rn.randn()  # bias

def predict(X, w, b):
    return w * X + b

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# -----------------------------
# STEP 3: Gradient Descent
# dw = (-2/n) * sum(x*(y - y_hat))
# db = (-2/n) * sum(y - y_hat)
# -----------------------------

lr = 0.01
iters = 1500
losses = []

for i in range(iters):
    y_hat = predict(X, w, b)

    dw = (-2/n) * np.sum(X * (y - y_hat))
    db = (-2/n) * np.sum(y - y_hat)

    w -= lr * dw
    b -= lr * db

    loss = mse(y, y_hat)
    losses.append(loss)

    if (i + 1) % 200 == 0:
        print(f"Iteration {i+1}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")


# -----------------------------
# STEP 4: Plots (save to /plots)
# -----------------------------
plots_path = Path("01_linear_regression/plots")
plots_path.mkdir(parents=True, exist_ok=True)

#Plot regresion Line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data Points')
x_line = np.linspace(0, 10, 100)
y_line = predict(x_line, w, b)
plt.title("Linear Regression From Scratch: Exam Score vs Hours Studied")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.savefig(plots_path / "regression_line.png", dpi=200, bbox_inches="tight")
plt.close()

# Plot Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title("Loss (MSE) over Training Iterations")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")    
plt.savefig(plots_path / "loss_curve.png", dpi=200, bbox_inches="tight")
plt.close()
