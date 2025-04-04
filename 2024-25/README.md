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
- [Baird's counterexample](./L3.1-baird.ipynb)

In this lecture, we will explore a from scratch implementation of Linear FA and we will discuss some limits of this approach.

#### Lecture 4: Deep Q-Learning (DQN)

- [Deep Reinforcement Learning overview](./L4.0-drl.ipynb)

- [DQN overview](./L4.1-dqn.ipynb)

In this lecture we will explore how deep learning has changed RL, allowing agents to solve complex tasks that were previously intractable. We will touch upon key concepts and algorithms that underpin this approach and we will discuss a simple implementation of a DQN.

#### Lecture 5: Policy Gradient

- [REINFORCE](./L5.0-reinforce.ipynb)

In this lecture, we introduce the idea of modeling the policy as a probability distribution over the action space. In particular, we will touch the Policy Gradient Theorem and we will see a simple implementation of REINFORCE.

#### Lecture 6: Introducing Multi-Agent Reinforcement Learning

- [Introduction to MARL](./L6.0-multiagent-rl.ipynb)

Departing from the single-agent setting, we explore the complexities that arise when considering a system with multiple learning agents to understand what results can be trasferred over from classic RL and the associated limitations. We consider the case of a single global agent learning policies for all players simultaneously and the case where each agent learns independently, comparing the results and dicussing scalability. Examples are developed using the **[PettingZoo](https://pettingzoo.farama.org/)** library from the Farama foundation, which provides the same API as Gymansium.

#### Lecture 7: Game Theory Basics for MARL

- [Game Theory](L7.0-introduction-to-game-theory.ipynb)

We introduce the necessary theoretical background to understand how multiple agents can interact in a complex environment and analyse different solution concepts that may arise. Examples of theoretical games are proposed and discussed under the lens of equilibrium selection, with particular interest towards Nash equilibria, their existence and computability. We formalize the notion of MDP in the multi-agent setting and start to understand that to model multi-agent systems and learning we require the presented game-theoretical approach.

#### Projects

The ideas of the final project are available [here](./projects/reinforcement_learning_projects_2024-25.pdf)