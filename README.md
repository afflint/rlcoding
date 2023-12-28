### Proff. Nicolò Cesa-Bianchi and Alfio Ferrara

# rlcoding

#### Code and examples for the Reinforcement Learning classes

Data Science and Economics Master Degree, Università degli Studi di Milano



## Repository organization

The current Academic Year's materials are stored in the `code` folder. Other folders are storing examples and code snippets of previous years (e.g., the folder `2022-23` contains materials for the AA 2022-23).

Coding is implemented using the Python libraries [pyenv](https://github.com/pyenv/pyenv) for setting up the virtual environment and [poetry](https://python-poetry.org/) for managing dependencies. The main poetry project is called `rl`. 

The code follows two approaches:

1. Implementing environment, models and algorithms **from scratch** (mainly for teaching purposes). This is stored in the package `rl.scratch`
2. Code implemented using the [gymnasium](https://gymnasium.farama.org/) library, stored in the package `rl.gym`
