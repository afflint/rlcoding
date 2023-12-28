import networkx as nx

from models import DDP
from heapq import heappop, heappush


def backtracking_search(model: DDP):
    """
    Note: for the sake of the example, we are gonna keep the whole
    history of exploration. In a real application, we just need to keep
    the best history discovered so far
    Memory: O(N), Time: O(b^N)
    with b => branching factor, N => depth
    :param model:
    :return:
    """
    paths, costs = [], []

    def path(state, history, total_cost):
        if model.end(state):
            costs.append(total_cost)
            paths.append(history)
            return
        for action, new_state, cost in model.options(state):
            path(new_state, history+[(action, new_state, cost)], total_cost + cost)

    path(state=model.start_state, history=[], total_cost=0)
    return paths, costs


def dynamic_programming(model: DDP):
    """
    Works of acyclic graphs induced by model.actions(state) and model.successor(state, action)
    O(S), with S number of states
    :param model: model
    :return: cost
    """
    cache = {} # this stores the future cost for the visited states

    def future(state):
        if model.end(state):
            return 0
        if state in cache:
            return cache[state]
        future_cost = min(
            cost + future(next_state) for action, next_state, cost in model.options(state)
        )
        cache[state] = future_cost
        return future_cost

    return future(model.start_state)


def uniform_cost_search(model: DDP):
    """
    Requires positive costs but support cycles in graphs
    O(s log s) with s being the number of states closer to end
    We use a representation of states that includes a priority (cost) as first element
    :param model: model
    :return: cost
    """
    queue = [(0, model.start_state)]
    to_visit = {model.start_state}
    visited = set()
    path = []
    while len(to_visit) > 0:
        c, state = heappop(queue)
        if model.end(state):
            return path
        visited.add(state)
        for action, next_state, cost in model.options(state):
            if next_state not in visited:
                heappush(queue, (c + cost, next_state))
                path.append((state, next_state, action, c, cost))
                to_visit.add(next_state)
        to_visit = to_visit.difference(visited)
    return path


