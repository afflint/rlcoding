###### Università degli Studi di Milano, Data Science and Economics Master Degree

# Code and examples for the Reinforcement Learning classes

## Course Instructors: Prof. Nicolò Cesa-Bianchi and Prof. Alfio Ferrara

### Course Assistant: Luigi Foscari (PhD student)

---

#### Lecture 0: Introduction
- [Introduction to Gymnasium](L0.0-gym-introduction.ipynb)
- [A basic duels game](L0.1-basic-duels.ipynb)

This lecture introduces the **[Gymnasium](https://gymnasium.farama.org/)** library as a versatile tool for creating and experimenting with reinforcement learning environments. We will explore the fundamentals of Markov Decision Processes (MDPs), modeling real-world problems as MDPs, and creating custom environments in Gymnasium to test and compare these methods. Hands-on examples and coding exercises help bridge theory and practice, offering a clear understanding of how these techniques apply to different scenarios. 

#### Lecture 1: Tabular Agents
- [Backward induction](L1.0-tabular-agents-dp-backward-induction.ipynb)
- [RL algorithms](L1.1-tabular-agents-rl-algorithms.ipynb)

In this lecture we will focus on modeling Agents that learn by interacting with an MDP environment.
In particular, we provide a quick recap of the main tabular reinforcement learning methods, including **dynamic programming**, **Monte Carlo**, **SARSA**, and **Q-learning**, by focusing on their implementation.

#### Lecture 2: Value Function Approximation
- [Introduction to VFA](L2.0-value-function-approximation.ipynb)
- [Linear Function Approximation](L2.1-linear-function-approximation.ipynb)

In this lecture, we discuss RL problems where the large number of states and the need to generalize the notion of state make the tabular solutions unfeasible. We will see how we can address such an issue by approximating the state value function $V(s)$ or the state-action function $Q(s, a)$.

#### Lecture 3: Linear Function Approximation
- [Implementing Linear Function approximation](./L3.0-linear-fa.ipynb)

In this lecture, we will explore a from scratch implementation of Linear FA and we will discuss some limits of this approach.

#### [Lecture 7: Introducing to multi-agent reinforcement learning](L7-multiagent-rl.ipynb)

Departing from the single-agent setting, we explore the complexities that arise when considering a system with multiple learning agents to understand what knowledge can be trasferred. We formalize the notion of MDP in the multi-agent setting and start to understand that to model multi-agent systems and learning we require a different theorical model, which leverages notions from the study of strategic interactions between agents, also called _game theory_. Examples are provided using the **[PettingZoo](https://pettingzoo.farama.org/)** library from the Farama foundation, which provides the same API as Gymansium.