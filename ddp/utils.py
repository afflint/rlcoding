import networkx as nx
import random
from typing import Callable, List, Tuple
import math


def fibonacci(n):
    a, b = 0, 1
    for _ in range(1, n):
        a, b = b, a + b
        yield b


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2,
                       vert_loc=0., xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def radial_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = hierarchy_pos(G, root, width=2 * math.pi, xcenter=0)
    return {u: (r * math.cos(theta), r * math.sin(theta)) for u, (theta, r) in pos.items()}


def to_graph(model,
             node: object = None,
             node_label: Callable = lambda x: "{}".format(x),
             G: nx.DiGraph = None,
             multi: bool = False
             ) -> nx.DiGraph:
    if G is None:
        if multi:
            G = nx.MultiDiGraph()
        else:
            G = nx.DiGraph()
    if node is None:
        node = model.start_state
    for action, next_node, cost in model.options(node):
        if (node, next_node, action) not in [(n1, n2, a['action']) for n1, n2, a in G.edges(data=True)]:
            G.add_edge(node, next_node, action=action, cost=cost)
        to_graph(model, node=next_node, G=G)
    attrs = dict([(n, {
        'label': node_label(n),
        'type': 'state'
    }) for n in G.nodes])
    nx.set_node_attributes(G, attrs)
    return G


def to_action_graph(model,
             node: object = None,
             node_label: Callable = lambda x: "{}".format(x),
             G: nx.DiGraph = None) -> nx.DiGraph:
    """
    Provides a graph where options are materialized as nodes
    :param model: a models.DDP model
    :param node: the current node
    :param node_label: callable functions
    :param G: graph used for recursion
    :return: the action graph
    """
    if G is None:
        G = nx.DiGraph()
    if node is None:
        node = model.start_state
        state = node
    else:
        state = node[2]
    for action, next_node, cost in model.options(state):
        step = (node, action, next_node)
        G.add_node(step,
                   label="{}: {}".format(action, node_label(next_node)),
                   type='action'
                   )
        G.add_edge(node, step, action=action, cost=cost)
        to_action_graph(model, node=step, G=G)
    attrs = dict([(n, {
        'label': node_label(n),
        'type': 'state'
    }) for n, a in G.nodes(data=True) if 'type' not in a.keys()])
    nx.set_node_attributes(G, attrs)
    return G


def search_graph(model, solutions: List[List[tuple]]):
    G = nx.DiGraph()
    G.add_node((-1,-1), action='start', state=model.start_state, label='start')
    for x, solution in enumerate(solutions):
        total_cost = 0
        for y, (action, state, cost) in enumerate(solution):
            total_cost += cost
            G.add_node((x, y),
                       action=action,
                       state=state, label="{} : {}".format(action, state))
            if y == 0:
                G.add_edge((-1,-1), (x, y), cost=total_cost, action=action)
            else:
                G.add_edge((x, y-1), (x, y), cost=total_cost, action=action)
    return G


def draw_graph(graph: nx.DiGraph,
               ax,
               pos: dict = None, node_size: int = 2500,
               font_size: int = 14,
               box: dict = None, box_node: dict = None,
               edge_font_color: str = 'white',
               node_font_color: str = 'black',
               node_line: int = 9
               ):
    if pos is None:
        pos = dict([(node, node) for node in graph.nodes])
    edge_labels = dict([((s, e), "{} : {}".format(a['action'], a['cost']))
                        for s, e, a in graph.edges(data=True)])
    node_labels = dict([(node, a['label']) for node, a in graph.nodes(data=True)])
    if box is not None:
        bb = box
    else:
        bb = dict(
            boxstyle="round, pad=0.3",
            fc="black", ec="green", alpha=0.6, mutation_scale=10)
    if box_node is not None:
        bbn = box_node
    else:
        bbn = dict(
            boxstyle="circle, pad=0.6",
            fc="w", ec="white", alpha=0.9, mutation_scale=10)
    nx.draw_networkx_edges(graph, pos,
                           width=1.5, alpha=0.5,
                           edge_color='green',
                           ax=ax, connectionstyle="arc3")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,
                                 label_pos=0.5, font_size=font_size, bbox=bb, ax=ax,
                                 font_color=edge_font_color,
                                 rotate=False)
    nx.draw_networkx_nodes(graph, pos, node_size=node_size,
                           node_color='white', alpha=0.6,
                           edgecolors='green',
                           linewidths=node_line,
                           ax=ax)
    nx.draw_networkx_labels(graph, pos,
                            labels=node_labels,
                            font_color=node_font_color, alpha=1.,
                            font_size=font_size, bbox=bbn,
                            font_family='sans-serif', ax=ax)


def draw_multi_graph(graph: nx.DiGraph,
               ax,
               pos: dict = None, node_size: int = 2500,
               font_size: int = 14,
               box: dict = None, box_node: dict = None,
               edge_font_color: str = 'white',
               node_font_color: str = 'black',
               node_line: int = 9
               ):
    if pos is None:
        pos = dict([(node, node) for node in graph.nodes])
    edge_labels = dict([((s, e), "{} : {}".format(a['action'], a['cost']))
                        for s, e, a in graph.edges(data=True)])
    node_labels = dict([(node, a['label']) for node, a in graph.nodes(data=True)])
    if box is not None:
        bb = box
    else:
        bb = dict(
            boxstyle="round, pad=0.3",
            fc="black", ec="green", alpha=0.6, mutation_scale=10)
    if box_node is not None:
        bbn = box_node
    else:
        bbn = dict(
            boxstyle="circle, pad=0.6",
            fc="w", ec="white", alpha=0.9, mutation_scale=10)
    nx.draw_networkx_nodes(graph, pos, node_size=node_size,
                           node_color='white', alpha=0.6,
                           edgecolors='green',
                           linewidths=node_line,
                           ax=ax)
    nx.draw_networkx_labels(graph, pos,
                            labels=node_labels,
                            font_color=node_font_color, alpha=1.,
                            font_size=font_size, bbox=bbn,
                            font_family='sans-serif', ax=ax)
    for e in graph.edges:
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
                                    ),
                                    ),
                    )


