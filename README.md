# rlcoding

#### Code and examples for the Reinforcement Learning classes

###### Course Instructors: Prof. Nicolò Cesa-Bianchi and Prof. Alfio Ferrara

###### Course Assistants: Elisabetta Rocchetti (PhD student), Luigi Foscari (PhD student)

Data Science and Economics Master Degree, Università degli Studi di Milano

### Lecture notes
- Prof. Cesa-Bianchi lecture notes are available at [https://cesa-bianchi.di.unimi.it/RL/](https://cesa-bianchi.di.unimi.it/RL/)
- Prof. Ferrara lecture notes are available in this repository in the `2024-25` folder.

## Repository organization

Materials are organized by Academic Year.

<!-- Coding is implemented using the Python libraries [pyenv](https://github.com/pyenv/pyenv) for setting up the virtual environment. -->

AY 2024-25 **lecture notes** from Alfio Ferrara and Luigi Foscari are available in the `2024-25` folder, mainly as `Jupyter notebooks`. These documents are **frequently updated**, thus stay tuned with the last versions.

The suggested version of Python is 3.11, you can use [pyenv](https://github.com/pyenv/pyenv) to change it. It is possible to create a _virtual environment_ to manage the required packages like `gymnasium` and `pettingzoo`. Start by creating the environment and then activating it
```bash
python3 -m venv .venv
source .venv/bin/activate
```
then install the required packages (this might take a while)
```bash
pip install -r requirement.txt
```