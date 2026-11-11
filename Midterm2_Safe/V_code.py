"""
Fit a V-shaped dataset with two ReLU neurons using gradient descent,
using the same math as the Excel sheet:

    u      = x - mean(x)                      (centered input)
    z1     = m1*u + b1,   z2 = m2*u + b2
    guess  = ReLU(z1) + ReLU(z2)
    error  = (y - guess) * (-1)
    db_i   = error * slope_i                  slope_i = 1 if z_i >= 0 else 0
    dm_i   = error * slope_i * u
    param  = param - learning_rate * average(gradient)

Usage:
    python relu_fit.py data.tsv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- settings ----------------
LEARNING_RATE = 0.001
EPOCHS = 500                    # match the Excel sheet; try 500+ to see it converge
M1, B1 = 0.1, 0.0              # neuron 1 (right arm) starting values
M2, B2 = -0.1, 0.0             # neuron 2 (left arm) starting values
# ------------------------------------------

path = sys.argv[1] if len(sys.argv) > 1 else "data.tsv"
df = pd.read_csv(path, sep="\t")
x = df.iloc[:, 1].to_numpy(dtype=float)    # column B
y = df.iloc[:, 2].to_numpy(dtype=float)    # column C

x_center = x.mean()


def forward(m1, b1, m2, b2, x):
    u = x - x_center
    z1 = m1 * u + b1
    z2 = m2 * u + b2
    guess = np.maximum(0, z1) + np.maximum(0, z2)
    return u, z1, z2, guess


m1, b1, m2, b2 = M1, B1, M2, B2
losses = []

print(f"x center = {x_center:.4f}")
print(f"{'epoch':>5} {'m1':>10} {'b1':>10} {'m2':>10} {'b2':>10} {'MSE':>12}")

for epoch in range(EPOCHS + 1):
    u, z1, z2, guess = forward(m1, b1, m2, b2, x)
    error = (y - guess) * (-1)
    mse = np.mean(error ** 2)
    losses.append(mse)
    print(f"{epoch:>5} {m1:>10.5f} {b1:>10.5f} {m2:>10.5f} {b2:>10.5f} {mse:>12.4f}")

    if epoch == EPOCHS:
        break  # last row: report loss of the final parameters, no more updates

    slope1 = (z1 >= 0).astype(float)
    slope2 = (z2 >= 0).astype(float)

    db1 = np.mean(error * slope1)
    dm1 = np.mean(error * slope1 * u)
    db2 = np.mean(error * slope2)
    dm2 = np.mean(error * slope2 * u)

    m1 -= LEARNING_RATE * dm1
    b1 -= LEARNING_RATE * db1
    m2 -= LEARNING_RATE * dm2
    b2 -= LEARNING_RATE * db2

# Parameters in terms of the original (uncentered) x
print("\nIn original x:  y = ReLU(m1*x + b1') + ReLU(m2*x + b2')")
print(f"  m1 = {m1:.5f}, b1' = {b1 - m1 * x_center:.5f}")
print(f"  m2 = {m2:.5f}, b2' = {b2 - m2 * x_center:.5f}")

# ---------------- plots ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

x_line = np.linspace(x.min(), x.max(), 400)
_, _, _, y_line = forward(m1, b1, m2, b2, x_line)

ax1.scatter(x, y, s=20, label="data")
ax1.plot(x_line, y_line, color="red", linewidth=2, label=f"fit after {EPOCHS} epochs")
ax1.set_xlabel("x")
ax1.set_ylabel("output")
ax1.set_title("Data and two-ReLU fit")
ax1.legend()

ax2.plot(range(EPOCHS + 1), losses, marker="o")
ax2.set_xlabel("epoch")
ax2.set_ylabel("MSE loss")
ax2.set_title("Loss vs. epoch")
if EPOCHS > 20:
    ax2.set_yscale("log")

fig.tight_layout()
fig.savefig("relu_fit.png", dpi=150)
plt.show()