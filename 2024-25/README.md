###### Università degli Studi di Milano, Data Science and Economics Master Degree

#### Code and examples for the Reinforcement Learning classes

###### Course Instructors: Prof. Nicolò Cesa-Bianchi and Prof. Alfio Ferrara

###### Course Assistant: Luigi Foscari (PhD student)

---

#### [Lecture 0: Introduction and recap](L0-gym-introduction.ipynb)

This lecture introduces the **[Gymnasium](https://gymnasium.farama.org/)** library as a versatile tool for creating and experimenting with reinforcement learning environments. We will explore the fundamentals of Markov Decision Processes (MDPs), modeling real-world problems as MDPs, and creating custom environments in Gymnasium to test and compare these methods. Hands-on examples and coding exercises help bridge theory and practice, offering a clear understanding of how these techniques apply to different scenarios. 

#### [Lecture 1: Tabular Agents](L1-tabular-agents.ipynb)
In this lecture we will focus on modeling Agents that learn by interacting with an MDP environment.
In particular, we provide a quick recap of the main tabular reinforcement learning methods, including **dynamic programming**, **Monte Carlo**, **SARSA**, and **Q-learning**, by focusing on their implementation.

#### 

#### [Lecture 7: Introducing to multi-agent reinforcement learning](L7-multiagent-rl.ipynb)

Departing from the single-agent setting, we explore the complexities that arise when considering a system with multiple learning agents to understand what knowledge can be trasferred. We formalize the notion of MDP in the multi-agent setting and start to understand that to model multi-agent systems and learning we require a different theorical model, which leverages notions from the study of strategic interactions between agents, also called _game theory_. Examples are provided using the **[PettingZoo](https://pettingzoo.farama.org/)** library from the Farama foundation, which provides the same API as Gymansium.