# data-lom-decoders

This repository contains the 
- scripts to generate the data
- the data
- and the jupyter notebooks to plot the data
for the project about the Logical Observable Matching (LOM) decoders.

There are two sets of experiments:
- repeated-gate experiments
- two-qubit-clifford experiments
Each set of experiments has their own directories.

The experiments are simulated under two noise models:
- phenomenological depolarizing noise
- SI1000 circuit-level noise

The directories ending with "_decoder" contain the logical error probabilities for that decoder.
To obtain the data, run the scripts in the following order (for e.g. SI1000 noise):
```
decode_experiment_SI1000.py
compress_data_SI1000.py
```
> [!TIP]
> To sampler faster, one can run several instances of `decode_experiments_SI1000.py`.
> The samplers in `qec-util` use file locking to avoid different threads writing in the same file at the same time.

To plot the data, open the relevant jupyter notebook and run all cells. 

The file `20250130_cliffords_modulo_paulis_and_swap.txt` is generated in `two-qubit-clifford_experiments_generation/`.
