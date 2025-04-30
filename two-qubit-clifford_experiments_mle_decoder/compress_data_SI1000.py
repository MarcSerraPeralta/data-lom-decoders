import pathlib

from surface_sim.models import SI1000NoiseModel
from qec_util.performance import merge_batches_in_file

from mle_decoder import MLEDecoder
from lomatching import MoMatching


# INPUTS
FILE_NAME = "20250130_cliffords_modulo_paulis_and_swap.txt"
EXPERIMENTS = [0, 248]
TOTAL_DEPTH = 14
DISTANCES = [3]
PROBS = [
    0.001,
    0.002,
    0.003,
    0.004,
    0.0045,
    0.005,
    0.0055,
    0.006,
    0.0065,
    0.007,
    0.008,
    0.01,
]
NOISE_MODEL = SI1000NoiseModel
BASES = ["Z", "X"]
FRAMES = ["pre-gate"]
DECODERS = [MLEDecoder, MoMatching]

# DATA STORAGE
NAME_FORMAT = "{exp_name}_{noise_model}_{decoder}_d{distance}_b{basis}_f{frame}_s0_p{prob:0.10f}.txt"
DATA_DIR = pathlib.Path(f"data_{FILE_NAME.replace('.txt', '')}")


############################################################################

if not DATA_DIR.exists():
    DATA_DIR.mkdir(exist_ok=True, parents=True)

for decoder in DECODERS:
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
                            decoder=decoder.__name__,
                        )

                        merge_batches_in_file(DATA_DIR / file_name)
