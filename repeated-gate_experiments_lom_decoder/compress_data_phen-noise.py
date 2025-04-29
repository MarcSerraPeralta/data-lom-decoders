import pathlib

from surface_sim.models import PhenomenologicalDepolNoiseModel
from qec_util.performance import merge_batches_in_file

from lomatching import MoMatching


# INPUTS
EXPERIMENTS = ["I", "S", "H", "CNOT-alternating", "CNOT-no-alternating"]
DISTANCES = [3, 5, 7]
PROBS = [
    0.005,
    0.01,
    0.01174619,
    0.0137973,
    0.01620657,
    0.01903654,
    0.02236068,
    0.02626528,
    0.03085169,
    0.03623898,
    0.042567,
    0.05,
]
NOISE_MODEL = PhenomenologicalDepolNoiseModel
BASES = ["Z", "X"]
FRAMES = ["pre-gate", "post-gate"]
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
