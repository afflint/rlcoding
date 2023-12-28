import numpy as np 
from tqdm import tqdm


def monte_carlo_first_visit(env, policy, shape: int, x: callable,
                            max_episode_len: int = 100,
                            iterations: int = 1000, alpha: float = 0.00001):
    w = np.zeros(shape)
    for _ in tqdm(range(iterations)):
        y = np.random.choice(list(range(env.start, env.N)))
        e = policy.generate_episode(y, env, max_iter=max_episode_len)
        visited = set()
        for i, (s, _, _) in enumerate(e):
            if s in visited:
                pass
            else:
                visited.add(s)
                G = sum(r for _, _, r in e[i:])
                w = w + (alpha * (G - x(s).dot(w))) * x(s)  
    return w 


def td_zero(env, policy, shape: int, x: callable, iterations: int = 1000, 
            alpha: float = 0.00001, gamma: float = 1):
    w = np.zeros(shape)
    for _ in tqdm(range(iterations)):
        s = np.random.choice(list(range(env.start, env.N)))
        a = policy[s]
        s_prime = env.outcome(a, s)
        r = env.reward(s_prime)
        w = alpha * (r + gamma * x(s_prime).T.dot(w) - x(s).T.dot(w)) * x(s)
    return w 
    