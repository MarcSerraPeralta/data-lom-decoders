import pathlib

from surface_sim.models import SI1000NoiseModel
from qec_util.performance import merge_batches_in_file

from pymatching import Matching


# INPUTS
EXPERIMENTS = ["S", "CNOT-no-alternating"]
DISTANCES = [3, 5, 7]
PROBS  = [1e-05, 2e-05, 4e-05, 5e-05, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.05, 0.1]
NOISE_MODEL = SI1000NoiseModel
BASES = ["Z"]
FRAMES = ["pre-gate"]
DECODER = Matching

# DATA STORAGE
NAME_FORMAT = "{exp_name}_{noise_model}_{decoder}_d{distance}_b{basis}_f{frame}_s0_p{prob:0.10f}.txt"
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
                    )

                    merge_batches_in_file(DATA_DIR / file_name)
