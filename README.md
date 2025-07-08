# exovelo_paper
This repository contains the code used for the research described in the
"Exovelo" paper (https://www.biorxiv.org/content/10.1101/2025.07.02.662720v1). All the code including the datasets are in the `code` folder.
The datasets are compressed in `xz` format.

The required packages are specified in the `requirements.txt` file. The file
`mamba.env.txt` contains the exact specification of the mamba environment used
in the research.

Exovelo is available as a python module. To use it just download the `exovelo`
subfolder and import it in your script e.g as`import exovelo as exo`.

The code folder contains a jupyer-lab file with example how to use Exovelo to
create joint embedding and velocity projection using Battich's RPE1 dataset.
The other jupyter-lab files document the production of the original paper's
results.

