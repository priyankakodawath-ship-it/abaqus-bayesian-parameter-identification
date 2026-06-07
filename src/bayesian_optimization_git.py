# =========================================================
# Bayesian Optimization for Abaqus-Based Parameter Identification
# Public demo version with generic file names
# =========================================================

import os
import logging
import csv
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

nameExp = "example_experimental_curve.csv"
nameFE = "simulation_odbField.csv"
nameJob = "simulation_model"
file_odb = "ODB_Results_Nodal.py"

nCPU = 4

# Parameters: [E, k, Smax, dmax, mu]
bounds = np.array([
    [0.5, 18.0, 0.00, 0.00, 0.00],
    [5.0, 22.0, 1.00, 0.13, 0.02]
])

logging.basicConfig(
    filename="bayesopt.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)

csv_file = "bayes_history.csv"

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "E", "k", "Smax", "dmax", "mu", "RMSE"])


def run(cmd):
    rc = os.system(cmd)
    logging.info("Command: %s | Exit code: %d" % (cmd, rc))
    return rc


def replace_next_line(lines, keyword, newline):
    for i, line in enumerate(lines):
        if line.strip().startswith(keyword):
            lines[i + 1] = newline + "\n"
            return
    raise RuntimeError("Keyword not found: " + keyword)


def objective(x, pen_exp, force_exp):
    E, k, smax, dmax, mu = [float(v) for v in x]

    logging.info("Evaluating: %s" % [E, k, smax, dmax, mu])

    for ext in [".odb", ".dat", ".log", ".sta", ".msg", ".com", ".prt", ".lck"]:
        f = nameJob + ext
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

    with open(nameJob + ".inp", "r") as f:
        lines = f.readlines()

    replace_next_line(lines, "*Elastic", "%f, 0.45" % E)
    replace_next_line(lines, "*Elastic, type=traction", "%f, %f" % (k, k))
    replace_next_line(lines, "*Damage initiation", "%f, %f" % (smax, smax))
    replace_next_line(lines, "*Damage evolution", "%f" % dmax)
    replace_next_line(lines, "*Friction", "%f" % mu)

    with open(nameJob + ".inp", "w") as f:
        f.writelines(lines)

    if run("abaqus job=%s cpus=%d interactive ask_delete=no" % (nameJob, nCPU)) != 0:
        return -1e6

    if run("abaqus python %s" % file_odb) != 0:
        return -1e6

    dataFE = np.loadtxt(nameFE, delimiter=",", skiprows=1)

    FE_pen = dataFE[:, 0]
    FE_force = dataFE[:, 1]

    FE_interp = np.interp(pen_exp, FE_pen, FE_force)

    rmse = np.sqrt(np.mean((FE_interp - force_exp) ** 2))

    logging.info("RMSE = %.6f" % rmse)

    return -rmse


def expected_improvement(X, gp, f_best):
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)

    Z = (mu - f_best) / sigma
    EI = (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)

    EI[sigma == 0] = 0

    return EI


data = np.genfromtxt(nameExp, delimiter=",", names=True)

pen_exp = data["Displacement"]
force_exp = data["Force"]

N_INIT = 15
N_TOTAL = 100


def random_design(n):
    X = np.random.rand(n, bounds.shape[1])
    return bounds[0] + X * (bounds[1] - bounds[0])


X = random_design(N_INIT)
y = np.array([objective(x, pen_exp, force_exp) for x in X])

kernel = Matern(nu=2.5) + WhiteKernel(1e-6)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

best_rmse = np.min(-y)
iteration = len(X)
skip_count = 0

while len(X) < N_TOTAL:
    gp.fit(X, y)

    Xcand = random_design(2000)
    EI = expected_improvement(Xcand, gp, np.max(y))
    x_next = Xcand[np.argmax(EI)]

    mu, sigma = gp.predict(x_next.reshape(1, -1), return_std=True)
    predicted_rmse = -mu[0]

    if predicted_rmse > best_rmse + 5:
        print("Skipping Abaqus run: predicted poor point")
        skip_count += 1
        y_next = mu[0]
    else:
        y_next = objective(x_next, pen_exp, force_exp)
        best_rmse = min(best_rmse, -y_next)

    iteration += 1
    X = np.vstack([X, x_next])
    y = np.append(y, y_next)

    rmse = -y_next

    print("Iteration", iteration, "RMSE =", rmse, "Skipped =", skip_count)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([iteration, *x_next, rmse])

best = np.argmax(y)

print("\nBest parameters [E, k, Smax, dmax, mu]")
print(X[best])
print("Best RMSE =", -y[best])