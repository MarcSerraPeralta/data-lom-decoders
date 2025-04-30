import pathlib
import time
import random
import stim

from surface_sim.setup import SI1000
from surface_sim.models import SI1000NoiseModel
from surface_sim import Detectors
from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.layouts import unrot_surface_codes
from qec_util.performance import sample_failures

from hyper_decom import decompose_dem

from pymatching import Matching


# INPUTS
EXPERIMENTS = ["CNOT-alternating", "CNOT-no-alternating", "S"]
DISTANCES = [3, 5, 7]
PROBS = [
    1e-5,
    2e-5,
    4e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    3e-3,
    5e-3,
    7e-3,
    1e-2,
    2e-2,
    5e-2,
    1e-1,
]
NOISE_MODEL = SI1000NoiseModel
BASES = ["Z"]  # , "Z"]
FRAMES = ["pre-gate"]  # ["gate-independent"]#, "pre-gate", "post-gate"]
DECODER = Matching

# DATA STORAGE
NAME_FORMAT = "{exp_name}_{noise_model}_{decoder}_d{distance}_b{basis}_f{frame}_s0_p{prob:0.10f}.txt"
DATA_DIR = pathlib.Path("data")


# EXTRA METRICS
def extra_metrics_2q(log_errors):
    e_0 = log_errors[:, 0]
    e_1 = log_errors[:, 1]
    e_2 = log_errors[:, 2]
    e_p = log_errors[:, 0] ^ log_errors[:, 1]
    return [e_0, e_1, e_2, e_p]


def no_extra_metrics(log_errors):
    return []


# GENERATE CIRCUIT
def get_num_qubits(experiment):
    return 1 if experiment in ["I", "S", "H", "SQRT_X"] else 2


def get_circuit(experiment, distance, basis):
    num_qubits = get_num_qubits(experiment)

    # reset
    circuit_str = ""
    if basis == "X":
        circuit_str = "RX " + " ".join([f"{i}" for i in range(num_qubits)]) + "\nTICK\n"
    else:
        circuit_str = "R " + " ".join([f"{i}" for i in range(num_qubits)]) + "\nTICK\n"

    # clifford gates + QEC cycles
    if num_qubits == 1:
        circuit_str += f"{experiment} 0\nTICK\n" * (distance + 1)
    elif experiment == "CNOT-no-alternating":
        circuit_str += "CNOT 0 1\nTICK\n" * (distance + 1)
    elif experiment == "CNOT-alternating":
        circuit_str += "CNOT 0 1\nTICK\nCNOT 1 0\nTICK\n" * ((distance + 1) // 2)
    elif experiment == "CZ":
        circuit_str += "CZ 0 1\nTICK\n" * (distance + 1)
    else:
        raise ValueError(f"{experiment} is not known.")

    # measurment
    if basis == "X":
        circuit_str += "MX " + " ".join([f"{i}" for i in range(num_qubits)])
    else:
        circuit_str += "M " + " ".join([f"{i}" for i in range(num_qubits)])

    circuit = stim.Circuit(circuit_str)

    return circuit


############################################################################

if not DATA_DIR.exists():
    DATA_DIR.mkdir(exist_ok=True, parents=True)

"""
# avoid having multiple threads writting on the same file
random.shuffle(EXPERIMENTS)
random.shuffle(BASES)
random.shuffle(FRAMES)
random.shuffle(DISTANCES)
random.shuffle(PROBS)
"""

for experiment_name in EXPERIMENTS:
    for basis in BASES:
        for frame in FRAMES:
            for distance in DISTANCES:
                layouts = unrot_surface_codes(
                    get_num_qubits(experiment_name), distance=distance
                )
                qubit_inds = {}
                anc_coords = {}
                anc_qubits = []
                stab_coords = {}
                for l, layout in enumerate(layouts):
                    qubit_inds.update(layout.qubit_inds())
                    anc_qubits += layout.get_qubits(role="anc")
                    coords = layout.anc_coords()
                    anc_coords.update(coords)
                    stab_coords[f"Z{l}"] = [v for k, v in coords.items() if k[0] == "Z"]
                    stab_coords[f"X{l}"] = [v for k, v in coords.items() if k[0] == "X"]

                circuit = get_circuit(experiment_name, distance, basis)

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

                    setup = SI1000()
                    setup.set_var_param("prob", prob)
                    model = NOISE_MODEL(setup=setup, qubit_inds=qubit_inds)
                    detectors = Detectors(
                        anc_qubits, frame=frame, anc_coords=anc_coords
                    )

                    schedule = schedule_from_circuit(circuit, layouts, gate_to_iterator)
                    experiment = experiment_from_schedule(
                        schedule, model, detectors, anc_reset=True, anc_detectors=None
                    )
                    if get_num_qubits(experiment_name) == 2:
                        # add product Z1*Z2
                        new_log = (
                            experiment[-1].targets_copy()
                            + experiment[-2].targets_copy()
                        )
                        experiment.append(
                            stim.CircuitInstruction(
                                "OBSERVABLE_INCLUDE", targets=new_log, gate_args=[2]
                            )
                        )

                    if frame != "gate-independent":
                        dem = experiment.detector_error_model(
                            allow_gauge_detectors=True,
                            decompose_errors=True,
                            ignore_decomposition_failures=True,
                        )
                    else:
                        dem = decompose_dem(
                            experiment.detector_error_model(allow_gauge_detectors=True),
                            ignore_logical_error=True,
                        )
                    decoder = DECODER(dem)

                    decoding_failure = lambda x: x.any(axis=1)
                    if "CNOT" in experiment_name:
                        decoding_failure = lambda x: x[:, :2].any(axis=1)

                    print(time.strftime("%D %H:%M:%S"), file_name)
                    sample_failures(
                        dem,
                        decoder,
                        max_failures=5_000 if prob < 1e-3 else 500_000,
                        max_time=60 * 10,  # s
                        max_samples=5_000_000_000,
                        max_batch_size=300_000,  # memory issues in the cluster
                        file_name=DATA_DIR / file_name,
                        extra_metrics=(
                            extra_metrics_2q
                            if get_num_qubits(experiment_name) == 2
                            else no_extra_metrics
                        ),
                        decoding_failure=decoding_failure,
                    )
