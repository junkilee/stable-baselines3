import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import defaultdict
from .ansi_colors import *
#rc('font', **{'family': 'serif'})
#rc('text', usetex=True)


def plot_graph(g, pos=None, identifier=None):
    fig = plt.figure()
    ax = plt.subplot(111)
    nx.draw(g, pos=nx.nx_pydot.graphviz_layout(g, prog='dot') if not pos else pos, with_labels=True, node_size=600,
            node_color='w', edgecolors='k', linewidths=1)
    # ax.text(1.2, -1, 'collider')
    plt.tight_layout(pad=0.1)
    if identifier:
        plt.savefig(identifier + ".pdf")
    else:
        plt.show()


def check_true(truth_table, var_id):
    val = truth_table[var_id - 1]    
    assert abs(val) == var_id
    return True if val > 0 else False


def plot_tree(name, n, var_dict, var_list, truth_table, prev_edge_tables):
    """

    Args:
        var_dict: 
        var_list:
        truth_table:
    """
    edges = defaultdict(lambda: defaultdict(int))
    g = nx.DiGraph()
    for i in range(1, n+1):
        g.add_node(i)
    for j in range(2, n+1):
        for i in range(j // 2, min(j-1, n) + 1):
            var_name = "p_{}_{}".format(j, i)
            assert var_name in var_dict
            var_id = var_dict[var_name]            
            if check_true(truth_table, var_id):
                edges[j][i] = 1
                edges[i][j] = 1
                g.add_edge(i, j)
    if edges in prev_edge_tables:
        return None
    else:    
        plot_graph(g, identifier=name)
    return edges


def display_truth_table(var_dict, truth_table):
    for name in var_dict:
        _id = var_dict[name]
        if check_true(truth_table, _id):
            print(BRIGHT_BLUE + name + RESET, end=' ')
        else:
            print(BRIGHT_RED + name + RESET, end=' ')
    print()