"""GATE script for Lecture 5, Part 2 -- does a standard DQN solve the
realistic microbe environment?

Per the build plan: train DQN on the realistic microbe, MULTI-SEED, and report
diagnostics. If the DQN does not reliably solve it, the environment must be
redesigned before any lecture markdown is written.

Setup
-----
Place ``realistic_microbe.py`` in ``rlc/envs/`` and ``dqn.py`` in
``rlc/agents/``, then run this script from the repository root:

    python gate_dqn_microbe.py

The three ``from rlc...`` imports below assume that layout; adjust them if your
tree differs. ``NormalizeObs`` is inlined here so the script does not depend on
the repo copy.

Cost: order of one to a few minutes per seed on CPU. Reduce N_SEEDS or
N_EPISODES below if needed -- partial output is still informative, since each
seed is printed as it finishes.

What to report back
-------------------
The full printout: the per-seed learning curves and final assessments, and the
across-seed summary.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rlc.envs.realistic_microbe import RealisticMicrobeEnv
from rlc.agents.dqn import DQNAgent
from rlc.utils.training import train

# --- configuration -----------------------------------------------------------
N_SEEDS = 5
N_EPISODES = 800           # per seed
MAX_STEPS = 300            # hard cap per episode (the env itself never truncates)
EVAL_EVERY = 100           # greedy-eval snapshot cadence, for the learning curve


class NormalizeObs(gym.ObservationWrapper):
    """Rescale Box observations to [0, 1] using the env's fixed, known bounds.

    The Q-network reads the raw state directly -- there is no hand-crafted
    feature map to put the coordinates on a common scale -- so the first linear
    layer and gradient descent are sensitive to input scale. Normalizing
    removes that sensitivity.
    """

    def __init__(self, env):
        super().__init__(env)
        low, high = env.observation_space.low, env.observation_space.high
        self._low = low
        self._span = high - low
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=low.shape, dtype=np.float32)

    def observation(self, obs):
        return ((obs - self._low) / self._span).astype(np.float32)


def make_env():
    return NormalizeObs(RealisticMicrobeEnv())


def greedy_rollout(agent, env, max_steps):
    """Run one greedy episode and return (return, length, success, mean_toxin).

    The environment is deterministic with a fixed start, so a single greedy
    rollout is exhaustive -- the spread of interest is across seeds, not across
    evaluation episodes.
    """
    obs, _ = env.reset()
    ep_return, steps, toxin_sum, success = 0.0, 0, 0.0, False
    while True:
        action = agent.select_action(obs, greedy=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        toxin_sum += info["toxin"]
        steps += 1
        if terminated:
            success = True
            break
        if truncated or steps >= max_steps:
            break
    return ep_return, steps, success, toxin_sum / steps


def run_seed(seed):
    """Train one DQN seed and print its diagnostics. Returns (return, success)."""
    env, eval_env = make_env(), make_env()
    agent = DQNAgent(n_state_features=4, n_actions=4, seed=seed)

    history = train(
        agent, env, N_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        eval_every=EVAL_EVERY, eval_episodes=1,   # deterministic env: 1 is enough
        eval_env=eval_env, eval_max_steps=MAX_STEPS,
        seed=seed,
    )

    ret, length, success, mean_toxin = greedy_rollout(agent, eval_env, MAX_STEPS)

    print(f"\n--- seed {seed} " + "-" * 52)
    curve = "  ".join(f"{e}:{m:+.0f}"
                       for e, m in zip(history.eval_episodes,
                                       history.eval_mean_returns))
    print(f"  greedy learning curve (episode:return)")
    print(f"    {curve}")
    print(f"  behaviour-policy return, last 100 episodes : "
          f"{np.mean(history.episode_returns[-100:]):+.1f}")
    print(f"  FINAL greedy rollout:")
    print(f"    reached goal   : {'YES' if success else 'NO'}")
    print(f"    return         : {ret:+.1f}")
    print(f"    episode length : {length} steps")
    print(f"    mean toxin/step: {mean_toxin:.3f}")
    wn, gn = agent.weight_norms, agent.grad_norms
    print(f"    network        : final |w| {wn[-1]:.2f}, "
          f"max |w| {max(wn):.2f}, max |grad| {max(gn):.2f}")
    return ret, success


def main():
    print("=" * 66)
    print("GATE -- DQN on the realistic microbe environment")
    print(f"  seeds={N_SEEDS}  episodes/seed={N_EPISODES}  max_steps={MAX_STEPS}")
    print("=" * 66)

    returns, successes = [], []
    for seed in range(N_SEEDS):
        ret, success = run_seed(seed)
        returns.append(ret)
        successes.append(success)

    returns = np.array(returns)
    successes = np.array(successes)
    print("\n" + "=" * 66)
    print("ACROSS-SEED SUMMARY")
    print(f"  greedy return : mean {returns.mean():+.1f}  std {returns.std():.1f}"
          f"  [min {returns.min():+.1f}, max {returns.max():+.1f}]")
    print(f"  per-seed      : {np.round(returns, 1)}")
    print(f"  reached goal  : {successes.sum()}/{N_SEEDS} seeds")
    print()
    print("  GATE intent: every seed reaches the goal, returns clustered and")
    print("  clearly positive. Wide spread, negative returns, a seed that")
    print("  fails, or an exploding |w| => tune or redesign before markdown.")
    print("=" * 66)


if __name__ == "__main__":
    main()