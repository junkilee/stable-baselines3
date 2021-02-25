import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import defaultdict
from .ansi_colors import *
#rc('font', **{'family': 'serif'})
#rc('text', usetex=True)


def plot_graph(g, pos=None, identifier=None, labels=None):
    fig = plt.figure()
    ax = plt.subplot(111)
    nx.draw(g, pos=nx.nx_pydot.graphviz_layout(g, prog='dot') if not pos else pos, with_labels=True, node_size=600,
            labels=labels, node_color='w', edgecolors='k', linewidths=1)
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


def check_true_by_name(var_name, truth_table, var_dict):
    print(var_name)    
    assert var_name in var_dict
    var_id = var_dict[var_name]  
    return check_true(truth_table, var_id)


def generate_graph(n, var_dict, truth_table):
    edges = defaultdict(lambda: defaultdict(int))
    g = nx.DiGraph()
    for i in range(1, n+1):
        g.add_node(i)
    for j in range(2, n+1):
        for i in range(j // 2, min(j-1, n) + 1):
            if check_true_by_name("p_{}_{}".format(j, i), truth_table, var_dict):
                edges[j][i] = 1
                edges[i][j] = 1
                g.add_edge(i, j)
    return edges, g


def generate_labels(n, var_dict, truth_table, main_feat_size, sub_feat_size):
    labels = {}
    for i in range(1, n+1):
        # leaf or non-leaf
        if check_true_by_name("v_{}".format(i), truth_table, var_dict):
            # leaf node
            if check_true_by_name("c_{}".format(i), truth_table, var_dict):
                labels[i] = "1"
            else:
                labels[i] = "0"
        else:
            features = []
            for m in range(1, main_feat_size+1):
                for s in range(1, sub_feat_size+1):
                    r = (m-1) * sub_feat_size + s
                    if check_true_by_name("a_{}_{}".format(r, i), truth_table, var_dict):
                        features = [str(m) + ":" + str(s)]
            labels[i] = ",".join(features)
    return labels

def display_truth_table(var_dict, truth_table):
    for name in var_dict:
        _id = var_dict[name]
        if check_true(truth_table, _id):
            print(BRIGHT_BLUE + name + RESET, end=' ')
        else:
            print(BRIGHT_RED + name + RESET, end=' ')
    print()