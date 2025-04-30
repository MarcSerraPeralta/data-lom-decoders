import pathlib

import numpy as np

from surface_sim.models import PhenomenologicalDepolNoiseModel
from qec_util.performance import read_failures_from_file

from lomatching import MoMatching


# INPUTS
FILE_NAME = "20250130_cliffords_modulo_paulis_and_swap.txt"
DISTANCES = [3, 5, 7]
PROBS = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.01165914,
    0.01359356,
    0.01584893,
    0.0184785,
    0.02,
    0.02154435,
    0.02352158,
    0.02511886,
    0.02766324,
    0.02928645,
    0.031,
    0.03253415,
    0.03414549,
    0.03528237,
    0.03826274,
    0.03981072,
    0.04149486,
    0.045,
    0.04641589,
    0.05,
    0.1,
]
NOISE_MODEL = PhenomenologicalDepolNoiseModel
BASES = ["Z", "X"]
FRAMES = ["pre-gate"]
DECODER = MoMatching

# DATA STORAGE
NAME_FORMAT = "{exp_name}_{noise_model}_{decoder}_d{distance}_b{basis}_f{frame}_s0_p{prob:0.10f}.txt"
DATA_DIR = pathlib.Path(f"data_{FILE_NAME.replace('.txt', '')}")

############################################################################

if not DATA_DIR.exists():
    DATA_DIR.mkdir(exist_ok=True, parents=True)

with open(FILE_NAME, "r") as file:
    data = file.read()
circuits = [
    block.split("TOTAL CIRCUIT:\n")[1]
    for block in data.split("\n----------\n")
    if block != ""
]
EXPERIMENTS = list(range(len(circuits)))

NUM_FAILURES = np.zeros(
    (len(EXPERIMENTS), len(BASES), len(FRAMES), len(DISTANCES), len(PROBS))
)
NUM_SAMPLES = np.zeros(
    (len(EXPERIMENTS), len(BASES), len(FRAMES), len(DISTANCES), len(PROBS))
)

for i, experiment_name in enumerate(EXPERIMENTS):
    for j, basis in enumerate(BASES):
        for k, frame in enumerate(FRAMES):
            for l, distance in enumerate(DISTANCES):
                for m, prob in enumerate(PROBS):
                    file_name = NAME_FORMAT.format(
                        exp_name=experiment_name,
                        noise_model=NOISE_MODEL.__name__,
                        distance=distance,
                        basis=basis,
                        frame=frame,
                        prob=prob,
                        decoder=DECODER.__name__,
                    )

                    if not (DATA_DIR / file_name).exists():
                        # print(DATA_DIR / file_name)
                        continue

                    try:
                        num_failures, num_samples, extra = read_failures_from_file(
                            DATA_DIR / file_name
                        )
                        NUM_FAILURES[i, j, k, l, m] = num_failures
                        NUM_SAMPLES[i, j, k, l, m] = num_samples
                    except:
                        print("BAD - ", DATA_DIR / file_name)

np.save("num-failures_phen-noise.npy", NUM_FAILURES)
np.save("num-samples_phen-noise.npy", NUM_SAMPLES)
