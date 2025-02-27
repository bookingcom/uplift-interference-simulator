## Clustered Network Interference Simulator (InfSim)

### About InfSim
This repository contains the code for InfSim, a simulation framework for cluster network interference published alongside the paper:
_Qini curve estimation under clustered network interference_ - **Rickard K.A. Karlsson**, **Bram van den Akker**, Felipe Moraes, Hugo Proença, Jesse H. Krijthe

The simulator targets researchers interested in causal inference or uplift modelling, and provides a playground for testing methodology for practitioners.

We recommend starting in the notebook section with `hello-world-compare-policies-with-ground-truth.ipynb`.

#### Repository

InfSim has three main components (found under `src/infsim`):
- `environments`, containing a base simulator that can be configured with various components (defined in `utils`).
- `policies`, containing various "optimal" and control policies that can be used for simulated data collection and benchmarking.
- `utils`, containing configurable components for the base environment and general utils such as ground truth evaluation.



### Setup

#### Requirements
- Python 3.9

#### Installation
After cloning the repository, navigate to the root directory and install the requirements in a virtual environment.

Start by creating a virtual environment, make sure [virtualenv is installed](https://pypi.org/project/virtualenv/).

```bash
python -m venv venv
```

Activate virtual environment
```bash
source venv/bin/activate
```

Install requirements
```bash
python -m pip install -r requirements.txt
```

If you are using this codebase outside an IDE, you need to do an interactive installation in pip.
For this, simply run the following in the root of the repo.

```bash
python -m pip install -e .
```

To validate everything is set up correctly, you can run the test suite from the root of the repo:

```
pytest tests
```

You know everything is ok when you see something like this.

```
==== 129 passed, 1 xfailed in 1.18s ====
```
