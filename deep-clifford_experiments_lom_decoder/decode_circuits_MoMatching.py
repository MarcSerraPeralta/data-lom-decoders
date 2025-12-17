import pathlib

import stim
from lomatching import MoMatching
from qec_util.samplers import sample_failures

# The stim circuit inside the 'deep_logical_circuits' directory have been downloaded from:
# https://zenodo.org/records/15544576
DIR = pathlib.Path("deep_logical_circuits")
OUTPUT_DIR = pathlib.Path("output_MoMatching")

######################################3


def get_stab_coords(distance):
    stab_coords = []
    for k in range(4):
        x_coords = []
        for col in range(distance - 1):
            if col % 2 == 0:
                c = [2, 2]
            else:
                c = [4 - 2, 0]

            for row in range(distance // 2 + 1):
                coord = (c[0] + col * 2 + (2 * distance + 2) * k, c[1] + 4 * row)
                x_coords.append(coord)

        z_coords = []
        for row in range(distance - 1):
            if row % 2 == 0:
                c = [0, 2]
            else:
                c = [2, 4 - 2]

            for col in range(distance // 2 + 1):
                coord = (c[0] + col * 4 + (2 * distance + 2) * k, c[1] + 2 * row)
                z_coords.append(coord)

        assert len(x_coords) == len(z_coords)
        assert len(x_coords + z_coords) == distance**2 - 1

        stab_coords.append({"X": x_coords, "Z": z_coords})

    return stab_coords


#########################

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

for nr in [1]:
    for distance in [3, 5, 7, 9, 11]:
        print(f"nr={nr} distance={distance}")
        encoded_circuit = stim.Circuit.from_file(
            DIR
            / f"deep_logical_32_layers_4_qubits_{nr}_nr_0.001_perr_{distance}_d.stim"
        )
        stab_coords = get_stab_coords(distance)

        decoder = MoMatching(encoded_circuit, stab_coords)
        dem = encoded_circuit.detector_error_model()

        sample_failures(
            dem,
            decoder,
            min_failures=5_000,
            max_failures=100_000,
            max_samples=50_000_000,
            max_batch_size=80_000,
            extra_metrics=lambda x: [x[:, 0], x[:, 1], x[:, 2], x[:, 3]],
            file_name=OUTPUT_DIR
            / f"deep_logical_32_layers_4_qubits_{nr}_nr_0.001_perr_{distance}_d.txt",
        )
