import pathlib
import time
import random
import stim

from surface_sim.setup import CircuitNoiseModel
from surface_sim.models import PhenomenologicalDepolNoiseModel
from surface_sim import Detectors
from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
from surface_sim.circuit_blocks.unrot_surface_code_css import gate_to_iterator
from surface_sim.layouts import unrot_surface_codes
from qec_util.performance import sample_failures

from lomatching import MoMatching

# INPUTS
FILE_NAME = "20250130_cliffords_modulo_paulis_and_swap.txt"
TOTAL_DEPTH = 14
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

FIRST_EXPS = []

# DATA STORAGE
NAME_FORMAT = "{exp_name}_{noise_model}_{decoder}_d{distance}_b{basis}_f{frame}_s0_p{prob:0.10f}.txt"
DATA_DIR = pathlib.Path(f"data_{FILE_NAME.replace('.txt', '')}")


# EXTRA METRICS
def extra_metrics_2q(log_errors):
    e_0 = log_errors[:, 0]
    e_1 = log_errors[:, 1]
    e_2 = log_errors[:, 2]
    e_p = log_errors[:, 0] ^ log_errors[:, 1]
    return [e_0, e_1, e_2, e_p]


def no_extra_metrics(log_errors):
    return []


def get_num_qubits(experiment):
    return 2


# GENERATE CIRCUIT
with open(FILE_NAME, "r") as file:
    data = file.read()
circuits = [
    block.split("TOTAL CIRCUIT:\n")[1]
    for block in data.split("\n----------\n")
    if block != ""
]
labelled_circuits_z = {}
labelled_circuits_x = {}
for k, circuit in enumerate(circuits):
    labelled_circuits_z[k] = circuit
    labelled_circuits_x[k] = circuit.replace("R 0 1", "RX 0 1").replace(
        "M 0 1", "MX 0 1"
    )

EXPERIMENTS = list(range(len(labelled_circuits_z)))
assert len(EXPERIMENTS) == 360


def get_circuit(experiment, distance, basis):
    num_qubits = 2

    circuits = labelled_circuits_z if basis == "Z" else labelled_circuits_x
    circuit_str = circuits[experiment]

    # add idling so that the total depth is fixed
    depth = len(circuit_str.split("TICK")) - 2
    if depth > TOTAL_DEPTH:
        raise ValueError(f"depth exceeds total depth: {depth} > {TOTAL_DEPTH}.")
    circuit_str = circuit_str.replace(
        "M 0 1", "TICK\n" * (TOTAL_DEPTH - depth) + "M 0 1"
    )
    circuit_str = circuit_str.replace(
        "MX 0 1", "TICK\n" * (TOTAL_DEPTH - depth) + "MX 0 1"
    )

    circuit = stim.Circuit(circuit_str)

    return circuit


def get_observables(experiment, basis):
    num_qubits = 2

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
random.shuffle(FIRST_EXPS)

# prioritize given experiments
EXPERIMENTS = FIRST_EXPS + EXPERIMENTS


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

                    setup = CircuitNoiseModel()
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

                    decoding_failure = lambda x: x[:, :2].any(axis=1)

                    print(time.strftime("%D %H:%M:%S"), file_name)
                    sample_failures(
                        dem,
                        decoder,
                        max_failures=50_000 if prob < 1e-3 else 500_000,
                        max_time=60 * 10,  # s
                        max_samples=5_000_000_000,
                        max_batch_size=70_000,  # memory issues in the cluster
                        file_name=DATA_DIR / file_name,
                        extra_metrics=(
                            extra_metrics_2q
                            if get_num_qubits(experiment_name) == 2
                            else no_extra_metrics
                        ),
                        decoding_failure=decoding_failure,
                    )
