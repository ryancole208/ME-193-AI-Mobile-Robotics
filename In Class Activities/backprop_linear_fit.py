# Compare three tiny networks fit to the annacode-v dataset with PyTorch backprop:
#   A) 1 neuron, no activation:   y = m*x + b
#   B) 1 neuron + ReLU:           y = ReLU(m*x + b)
#   C) 2 ReLU neurons, summed:    y = ReLU(w1*x + b1) + ReLU(w2*x + b2)
# Each step: loss per point L_i = (y_pred_i - y_i)^2, then backprop gives dL/dw
# and dL/db for every weight and bias averaged over all points, which are used
# to update them.

import torch
import matplotlib.pyplot as plt

# (time, x, output 1) from annacode-v.tsv
data = [
    (0, 20.3133, 81.6953), (0.1, 62.5876, 8.75714), (0.2, 91.1726, 57.6214), (0.3, 88.3446, 53.403),
    (0.4, 60.8373, 1.15292), (0.5, 81.6655, 40.6197), (0.6, 55.7309, 7.80748), (0.7, 76.0945, 33.6766),
    (0.8, 71.4818, 22.6546), (0.9, 85.7171, 50.8484), (1, 27.2607, 68.2794), (1.1, 84.6289, 49.6776),
    (1.2, 46.6619, 21.6335), (1.3, 57.6035, 8.17689), (1.4, 26.1983, 67.4404), (1.5, 95.588, 73.0875),
    (1.6, 65.5718, 9.05043), (1.7, 87.2195, 53.5723), (1.8, 43.4202, 33.0679), (1.9, 52.5449, 14.7814),
    (2, 29.3457, 61.4134), (2.1, 29.5907, 62.7043), (2.2, 40.6375, 39.4781), (2.3, 69.9907, 19.544),
    (2.4, 96.3404, 74.5163), (2.5, 49.5358, 21.7348), (2.6, 26.4729, 68.5835), (2.7, 86.4375, 49.7005),
    (2.8, 33.6029, 51.7668), (2.9, 61.1689, 3.10899), (3, 76.2596, 29.7569), (3.1, 74.1758, 24.9412),
    (3.2, 59.139, 1.3724), (3.3, 77.2642, 34.7115), (3.4, 66.5489, 12.9046), (3.5, 93.5584, 69.7599),
    (3.6, 82.9515, 43.6148), (3.7, 68.9877, 16.5984), (3.8, 66.0651, 10.0632), (3.9, 37.9508, 43.0572),
    (4, 94.9028, 65.5368), (4.1, 39.3284, 42.2743), (4.2, 37.8013, 43.9316), (4.3, 72.7456, 25.2983),
    (4.4, 33.0974, 57.247), (4.5, 93.8979, 65.4468), (4.6, 82.2054, 47.7221), (4.7, 57.6779, 2.75164),
    (4.8, 73.3453, 31.55), (4.9, 74.0013, 26.9262), (5, 69.609, 18.0014), (5.1, 72.0596, 22.2081),
    (5.2, 45.0007, 29.7118), (5.3, 30.4107, 60.648), (5.4, 55.5891, 9.44857), (5.5, 80.2599, 42.481),
    (5.6, 75.9702, 29.7854), (5.7, 20.5282, 81.1835), (5.8, 26.7124, 65.8122), (5.9, 63.9064, 8.53925),
    (6, 95.5653, 67.6596), (6.1, 85.8856, 51.8063), (6.2, 57.3467, 6.55358), (6.3, 46.9618, 23.7419),
    (6.4, 70.7135, 22.4529), (6.5, 56.8589, 10.8099), (6.6, 29.5684, 62.3405), (6.7, 54.9862, 10.4224),
    (6.8, 89.373, 56.7306), (6.9, 37.4571, 47.3537), (7, 97.7704, 74.356), (7.1, 86.6914, 55.3573),
    (7.2, 83.3527, 47.0614), (7.3, 84.5069, 48.3598), (7.4, 79.6139, 38.7279), (7.5, 89.5275, 59.8986),
    (7.6, 70.9326, 23.0528), (7.7, 75.3537, 30.6967), (7.8, 91.5748, 65.1166), (7.9, 72.0701, 21.7729),
    (8, 63.0269, 8.11677), (8.1, 56.7294, 9.82535), (8.2, 31.1787, 57.8927), (8.3, 43.3324, 32.0524),
    (8.4, 45.3355, 27.6051), (8.5, 25.4005, 65.9093), (8.6, 30.2323, 61.0216), (8.7, 76.4723, 31.6873),
    (8.8, 50.7804, 19.3109), (8.9, 62.262, 7.93199), (9, 29.4421, 61.8416), (9.1, 37.2388, 41.2693),
    (9.2, 79.6889, 36.9762), (9.3, 35.9608, 45.6398), (9.4, 84.9881, 48.0213), (9.5, 67.1519, 12.4159),
    (9.6, 22.1735, 74.3275), (9.7, 79.4951, 37.6872), (9.8, 99.4262, 77.0124), (9.9, 56.8495, 5.93906),
    (10, 88.7065, 55.2767), (10.1, 49.95, 19.7881), (10.2, 30.1099, 62.0057), (10.3, 89.6647, 59.6387),
    (10.4, 44.5146, 34.5763), (10.5, 38.1282, 45.945), (10.6, 36.7469, 46.7345), (10.7, 20.2702, 81.3299),
    (10.8, 85.5427, 49.759), (10.9, 49.7573, 19.4243), (11, 52.3625, 15.1427), (11.1, 24.4403, 69.0096),
    (11.2, 35.09, 48.4491), (11.3, 98.2069, 76.1067), (11.4, 72.5763, 25.805), (11.5, 48.987, 23.1349),
    (11.6, 64.4477, 9.82167), (11.7, 97.3807, 73.4114), (11.8, 83.6164, 49.4164), (11.9, 66.9108, 14.8685),
]

x = torch.tensor([row[1] for row in data])
y = torch.tensor([row[2] for row in data])


def P(value):
    return torch.tensor(value, requires_grad=True)


# Initial values put every neuron's m*x + b above zero for all x in the data
# (x is 20..100). A ReLU whose input is below zero passes back zero gradient,
# so a neuron that started "off" everywhere would never learn.
#
# Model C's neurons start as one rising (w1 > 0) and one falling (w2 < 0) line.
# If they started identical they would get identical gradients and stay
# identical forever, acting like a single neuron.
m_a, b_a = P(0.0), P(1.0)
m_b, b_b = P(0.0), P(1.0)
w1, b1 = P(0.5), P(0.0)
w2, b2 = P(-0.5), P(60.0)

models = [
    {
        "name": "1 neuron (no activation)",
        "neurons": 1, "activation_layers": 0,
        "params": {"m": m_a, "b": b_a},
        "forward": lambda x: m_a * x + b_a,
        "formula": lambda: f"y = {m_a.item():.4f}*x + {b_a.item():.4f}",
    },
    {
        "name": "1 neuron + ReLU",
        "neurons": 1, "activation_layers": 1,
        "params": {"m": m_b, "b": b_b},
        "forward": lambda x: torch.relu(m_b * x + b_b),
        "formula": lambda: f"y = ReLU({m_b.item():.4f}*x + {b_b.item():.4f})",
    },
    {
        "name": "2 ReLU neurons",
        "neurons": 2, "activation_layers": 1,
        "params": {"w1": w1, "b1": b1, "w2": w2, "b2": b2},
        "forward": lambda x: torch.relu(w1 * x + b1) + torch.relu(w2 * x + b2),
        "formula": lambda: (f"y = ReLU({w1.item():.4f}*x + {b1.item():.4f})"
                            f" + ReLU({w2.item():.4f}*x + {b2.item():.4f})"),
    },
]

# x goes up to ~100, so dL/dweight is ~100x bigger than dL/dbias. Separate
# learning rates keep the weights from blowing up while still letting biases move.
lr_weight = 1e-4
lr_bias = 1e-2
epochs = 20000


def lr_for(name):
    return lr_bias if name.startswith("b") else lr_weight


def zero_grads(params):
    for p in params.values():
        p.grad.zero_()


for model in models:
    params, forward = model["params"], model["forward"]
    print("=" * 70)
    print(model["name"])
    print(f"Number of neurons:           {model['neurons']}")
    print(f"Number of activation layers: {model['activation_layers']}\n")

    # --- Sanity check: averaging per-point gradients == gradient of the mean loss ---
    sums = {name: 0.0 for name in params}
    for xi, yi in zip(x, y):
        Li = (forward(xi) - yi) ** 2
        Li.backward()
        for name, p in params.items():
            sums[name] += p.grad.item()
        zero_grads(params)
    print("per-point averaged:  " + "   ".join(f"dL/d{n} = {s / len(x):.4f}" for n, s in sums.items()))

    L = ((forward(x) - y) ** 2).mean()
    L.backward()
    print("mean-loss backward:  " + "   ".join(f"dL/d{n} = {p.grad.item():.4f}" for n, p in params.items()) + "\n")
    zero_grads(params)

    # --- Gradient descent ---
    losses = []
    for epoch in range(epochs):
        L = ((forward(x) - y) ** 2).mean()  # MSE: average of per-point losses
        L.backward()                         # p.grad = avg dL/dp for every parameter

        with torch.no_grad():
            for name, p in params.items():
                p -= lr_for(name) * p.grad
        grads = {name: p.grad.item() for name, p in params.items()}
        zero_grads(params)

        losses.append(L.item())
        if epoch % 4000 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:5d}  loss {L.item():9.3f}  "
                  + "  ".join(f"dL/d{n} {g:9.4f}" for n, g in grads.items()))

    model["losses"] = losses
    print(f"\nfinal fit: {model['formula']()}   (loss {losses[-1]:.3f})\n")

# --- Plot: all three fits on the data, and their loss curves ---
colors = ["tab:red", "tab:orange", "tab:green"]
# Models A and B end up identical (the ReLU never cuts anything off), so B is
# dashed to keep A visible underneath it.
styles = ["-", "--", "-"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.scatter(x, y, s=12, color="gray", label="data")
xs = torch.linspace(x.min(), x.max(), 200)
for model, c, ls in zip(models, colors, styles):
    label = f"{model['name']}  ({model['neurons']} neuron, {model['activation_layers']} act. layer)"
    ax1.plot(xs, model["forward"](xs).detach(), color=c, ls=ls, lw=2, label=label)
    ax2.plot(model["losses"], color=c, ls=ls, label=model["name"])
ax1.set_xlabel("x")
ax1.set_ylabel("output 1")
ax1.set_title("Fits")
ax1.legend(fontsize=8)
ax2.set_xlabel("epoch")
ax2.set_ylabel("MSE loss")
ax2.set_yscale("log")
ax2.set_title("Training loss")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()
