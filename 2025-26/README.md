# Reinforcement Learning — 2025-26

Course materials for the Reinforcement Learning module, academic year 2025-26.

## Overview

This module builds on the theoretical foundations of tabular RL methods covered in the introductory part of the course. The goal is twofold:

1. **Practical track.** Introduce [Gymnasium](https://gymnasium.farama.org/) as
   the standard interface for RL environments, and implement, train, and evaluate the classical tabular agents (Q-learning, SARSA, Monte Carlo) on both built-in and custom environments.
2. **Theoretical track.** Extend tabular methods to **value function approximation**, from linear approximators with hand-crafted features all the way to deep neural networks (DQN).

The module consists of 5 two-hour lectures combining theoretical exposition and live notebook walkthroughs.

## Prerequisites

- Solid understanding of MDPs, Bellman equations, and tabular RL methods (policy/value iteration, Q-learning, SARSA, Monte Carlo).
- Proficiency with Python, NumPy, and PyTorch.
- Familiarity with basic supervised learning (gradient descent, loss functions, training/validation protocols).

## Schedule

| # | Lecture | Topics |
|---|---------|--------|
| 1 | [Gymnasium, MDPs and Tabular Q-Learning](./lecture_01_gymnasium_tabular.ipynb) | Gymnasium API, MDP modeling, custom environments, Q-learning implementation |
| 2 | [SARSA, Monte Carlo and Experimental Protocol](./lecture_02_sarsa_montecarlo.ipynb) | On-policy vs off-policy control, MC methods, exploration schedules, evaluation protocol |
| 3 | [Linear Value Function Approximation](./lecture_03_linear_vfa.ipynb) | Feature construction (tile coding, RBF), MC and TD with linear VFA, semi-gradient methods |
| 4 | [Control with VFA and the Deadly Triad](./lecture_04_control_vfa.ipynb) | Action-value approximation, semi-gradient SARSA and Q-learning, divergence and the deadly triad |
| 5 | [Deep Reinforcement Learning with DQN](./lecture_05_deep_rl.ipynb) | Neural-network approximators, experience replay, target networks, DQN and variants |

## Repository structure

```
2025-26/
├── README.md                              # this file
├── lecture_01_gymnasium_tabular.ipynb
├── lecture_02_sarsa_montecarlo.ipynb
├── lecture_03_linear_vfa.ipynb
├── lecture_04_control_vfa.ipynb
├── lecture_05_deep_rl.ipynb
└── rlc/                                   # shared course package
    ├── envs/                              # custom environments
    ├── agents/                            # agent classes
    └── utils/                             # training and plotting utilities
```

The `rlc` package is built incrementally over the five lectures: each lecture adds new modules that are then reused in subsequent lectures and in the final project. Notebooks import from `rlc` and focus on orchestration, execution, and visualization; algorithmic logic and reusable classes live in the package.

## Final project

The final project asks you to apply the methods covered in the course to a Gymnasium environment of your choice (built-in or custom). The required steps are:

- Select an environment and motivate the choice with respect to its state/action structure and difficulty.
- Implement and train **at least an agent**.
- Run a **rigorous experimental comparison**: multiple random seeds, learning curves with variance bands, deterministic evaluation episodes, and a discussion of hyperparameter choices.
- Produce a written report (~10 pages) discussing methodology, results, and limitations.

The project complements a theoretical exam in the final evaluation.

## Setup

Lecture notebooks expect to be launched from the `2025-26/` directory so that the `rlc` package is importable. A minimal setup is:

```bash
cd rlcourse/2025-26
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install gymnasium[classic-control] numpy matplotlib torch jupyter
jupyter lab
```

Individual lectures may require additional dependencies, listed at the top of each notebook.

## References

The course follows the notation and structure of:

- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction* (2nd ed.), MIT Press, 2018.
  [Free online edition](http://incompleteideas.net/book/the-book-2nd.html).

Additional readings are cited within each lecture notebook.
