import pathlib
import time
import random
import stim

from surface_sim.setup import CircuitNoiseSetup
from surface_sim.models import PhenomenologicalDepolNoiseModel
from surface_sim import Detectors
from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.layouts import unrot_surface_codes
from qec_util.performance import sample_failures

from somatching import SoMatching


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
DECODER = SoMatching

# DATA STORAGE
NAME_FORMAT = "{exp_name}_{noise_model}_{decoder}_d{distance}_b{basis}_f{frame}_s0_ncycle-{ncycle}_p{prob:0.10f}.txt"
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


def get_circuit(experiment, distance, basis, num_qec_per_gate):
    num_qubits = get_num_qubits(experiment)

    # reset
    circuit_str = ""
    if basis == "X":
        circuit_str = "RX " + " ".join([f"{i}" for i in range(num_qubits)]) + "\nTICK\n"
    else:
        circuit_str = "R " + " ".join([f"{i}" for i in range(num_qubits)]) + "\nTICK\n"

    # clifford gates + QEC cycles
    qecs = "TICK\n" * num_qec_per_gate
    if num_qubits == 1:
        circuit_str += f"{experiment} 0\n{qecs}" * (distance + 1)
    elif experiment == "CNOT-no-alternating":
        circuit_str += f"CNOT 0 1\n{qecs}" * (distance + 1)
    elif experiment == "CNOT-alternating":
        circuit_str += f"CNOT 0 1\n{qecs}CNOT 1 0\n{qecs}" * ((distance + 1) // 2)
    else:
        raise ValueError(f"{experiment} is not known.")

    # measurment
    if basis == "X":
        circuit_str += "MX " + " ".join([f"{i}" for i in range(num_qubits)])
    else:
        circuit_str += "M " + " ".join([f"{i}" for i in range(num_qubits)])

    circuit = stim.Circuit(circuit_str)

    return circuit


def get_observables(experiment, basis):
    num_qubits = get_num_qubits(experiment)

    if basis == "X" and num_qubits == 1:
        return [["X0"]]
    if basis == "Z" and num_qubits == 1:
        return [["Z0"]]
    if basis == "X" and num_qubits == 2:
        return [["X0"], ["X1"], ["X0", "X1"]]
    if basis == "Z" and num_qubits == 2:
        return [["Z0"], ["Z1"], ["Z0", "Z1"]]

    raise ValueError


############################################################################

if not DATA_DIR.exists():
    DATA_DIR.mkdir(exist_ok=True, parents=True)

# avoid having multiple threads writting on the same file
random.shuffle(EXPERIMENTS)
random.shuffle(BASES)
random.shuffle(FRAMES)
random.shuffle(DISTANCES)
random.shuffle(PROBS)

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

                circuit = get_circuit(
                    experiment_name, distance, basis, NUM_QEC_PER_GATE
                )

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

                    setup = CircuitNoiseSetup()
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

                    dem = experiment.detector_error_model(allow_gauge_detectors=True)
                    decoder = DECODER(
                        dem,
                        circuit,
                        get_observables(experiment_name, basis),
                        stab_coords,
                        frame,
                    )

                    decoding_failure = lambda x: x.any(axis=1)
                    if "CNOT" in experiment_name:
                        decoding_failure = lambda x: x[:, :2].any(axis=1)

                    print(time.strftime("%D %H:%M:%S"), file_name)
                    sample_failures(
                        dem,
                        decoder,
                        max_failures=50_000 if prob < 1e-3 else 500_000,
                        max_time=60 * 15,  # s
                        max_samples=5_000_000_000,
                        max_batch_size=80_000,  # memory issues in the cluster
                        file_name=DATA_DIR / file_name,
                        extra_metrics=(
                            extra_metrics_2q
                            if get_num_qubits(experiment_name) == 2
                            else no_extra_metrics
                        ),
                        decoding_failure=decoding_failure,
                    )
