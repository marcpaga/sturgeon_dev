# Sturgeon development

Code used in the development, training and validation, of Sturgeon.

For more information please check our [preprint](https://www.medrxiv.org/content/10.1101/2023.01.25.23284813v1).

If you just want to predict using our models then please refer to the following repository: https://github.com/marcpaga/sturgeon

## Installation

Requires python 3.7 and a linux system.

```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip

pip3 install -r requirements.txt
```

## Contents

`sturgeon_dev`: contains the source code, mostly functions and classes.

`scripts`: contains custom scripts for the processing of the data.

`static`: contains small definitions, such as the CNS classification system, the location of the microarray probes and the plotting colors of each CNS class.

`workflows`: contains snakemake workflows to run the training/validation pipelines.
