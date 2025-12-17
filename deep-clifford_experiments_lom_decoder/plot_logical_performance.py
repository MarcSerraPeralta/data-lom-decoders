import os
import pathlib
import csv
import ast
import numpy as np
import matplotlib.pyplot as plt

from qec_util.samplers import read_failures_from_file
from qec_util.performance import confidence_interval_binomial

import matplotlib

matplotlib.rcParams.update(
    {
        "font.size": 11,
        "font.family": "cmr10",
        "font.weight": "normal",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "text.usetex": True,
        "axes.formatter.limits": (0, 0),
    }
)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


DISTANCES = [3, 5, 7, 9, 11]
MAX_BP_ITERS = [0, 10, 100]

FILE_NAME_FORMAT = "deep_logical_32_layers_4_qubits_{nr}_nr_0.001_perr_{distance}_d.txt"

GP_DATA = "deep_logical_circuits/deep_logical.csv"

#################

# load LOM decoder data
failures = np.zeros((len(MAX_BP_ITERS), len(DISTANCES)))
samples = np.zeros((len(MAX_BP_ITERS), len(DISTANCES)))
for i, max_iters in enumerate(MAX_BP_ITERS):
    if max_iters != 0:
        DIR_OUTPUT = pathlib.Path(f"output_BeliefMoMatching_max-bp-iters-{max_iters}")
    else:
        DIR_OUTPUT = pathlib.Path("output_MoMatching")

    for j, distance in enumerate(DISTANCES):
        file_name = FILE_NAME_FORMAT.format(nr=1, distance=distance)
        num_failures, num_shots, extra = read_failures_from_file(DIR_OUTPUT / file_name)
        failures[i, j] = num_failures
        samples[i, j] = num_shots

lower_bound, upper_bound = confidence_interval_binomial(failures, samples)

# load Ghost-Protocol data
NUM_ROUNDS = [1, 2, 3]
failures_gp = np.zeros((len(NUM_ROUNDS), len(DISTANCES)))
samples_gp = np.zeros((len(NUM_ROUNDS), len(DISTANCES)))
with open(GP_DATA) as csv_file:
    csv_read = csv.reader(csv_file, delimiter=",")
    for i, row in enumerate(csv_read):
        if i == 0:
            continue
        num_failures = int(row[1])
        num_shots = int(row[0])
        metadata = ast.literal_eval(row[-2])
        i = NUM_ROUNDS.index(int(metadata["nr"]))
        j = DISTANCES.index(int(metadata["distance"]))
        failures_gp[i, j] = num_failures
        samples_gp[i, j] = num_shots

lower_bound_gp, upper_bound_gp = confidence_interval_binomial(failures_gp, samples_gp)

# plot data
fig, ax = plt.subplots(figsize=cm2inch(10, 10))

colors = ["deepskyblue", "royalblue", "navy"]
for max_iters in MAX_BP_ITERS[::1]:
    i = MAX_BP_ITERS.index(max_iters)
    ax.plot(
        DISTANCES,
        (failures[i] / samples[i]) / 32,
        marker="o",
        linestyle="none",
        color=colors[i],
        label=f"BP+LOM, $n_{{\\rm BP}} \\leq {max_iters}$" if max_iters != 0 else "LOM",
        markersize=4,
    )
    ax.fill_between(
        DISTANCES, lower_bound[i] / 32, upper_bound[i] / 32, color=colors[i], alpha=0.25
    )

COLOR_GP = "red"
ax.plot(
    DISTANCES,
    (failures_gp[0] / samples_gp[0]) / 32,
    marker="s",
    color=COLOR_GP,
    linestyle="none",
    markersize=4,
)
ax.fill_between(
    DISTANCES,
    lower_bound_gp[0] / 32,
    upper_bound_gp[0] / 32,
    color=COLOR_GP,
    alpha=0.25,
)

ax.plot(
    [],
    [],
    marker="s",
    label="Ghost Protocol",
    color=COLOR_GP,
    linestyle="none",
    markersize=4,
    # alpha=0.25,
)

ax.set_xlabel("distance")
ax.set_yscale("log")
ax.set_ylim(1e-7, 5e-3)

ax.set_xticks(DISTANCES)
ax.set_xticklabels([f"{d:d}" for d in DISTANCES])

ax.set_ylabel("observable error probability per transversal layer")
ax.set_ylabel("Decoder failure probability per gate layer")
ax.legend(loc="upper right", frameon=False, fontsize=10)

fig.tight_layout()

fig.savefig("deep_logical_circuit.pdf", format="pdf")

plt.show()
