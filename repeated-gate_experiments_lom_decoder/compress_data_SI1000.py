import pathlib

from surface_sim.models import SI1000NoiseModel
from qec_util.performance import merge_batches_in_file

from lomatching import MoMatching


# INPUTS
EXPERIMENTS = ["I", "S", "H", "CNOT-alternating", "CNOT-no-alternating"]
DISTANCES = [3, 5, 7]
PROBS = [
    0.0002,
    0.0005,
    0.001,
    0.00116591,
    0.00135936,
    0.00158489,
    0.00184785,
    0.002,
    0.00215443,
    0.00251189,
    0.00292864,
    0.00341455,
    0.00398107,
    0.00464159,
    0.005,
    0.0054117,
    0.00630957,
    0.00735642,
    0.00857696,
    0.01,
]
NOISE_MODEL = SI1000NoiseModel
BASES = ["Z", "X"]
FRAMES = ["pre-gate"]
NUM_QEC_PER_GATE = 1
DECODER = MoMatching

# DATA STORAGE
NAME_FORMAT = "{exp_name}_{noise_model}_{decoder}_d{distance}_b{basis}_f{frame}_s0_ncycle-{ncycle}_p{prob:0.10f}.txt"
DATA_DIR = pathlib.Path("data")


############################################################################

if not DATA_DIR.exists():
    DATA_DIR.mkdir(exist_ok=True, parents=True)

for experiment_name in EXPERIMENTS:
    for basis in BASES:
        for frame in FRAMES:
            for distance in DISTANCES:
                for prob in PROBS:
                    file_name = NAME_FORMAT.format(
                        exp_name=experiment_name,
                        noise_model=NOISE_MODEL.__name__,
                        distance=distance,
                        basis=basis,
                        frame=frame,
                        prob=prob,
                        decoder=DECODER.__name__,
                        ncycle=NUM_QEC_PER_GATE,
                    )

                    merge_batches_in_file(DATA_DIR / file_name)
