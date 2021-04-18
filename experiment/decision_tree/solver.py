import hydra
import logging
import os
import glob
from threading import Timer
from pysat.solvers import Solver  # , Glucose4
from .cnf_generator import CNFGenDebugLevel, CNFGenerator 
from experiment.decision_tree.utils.tree_visualizer import check_true, check_true_by_name, generate_graph, \
        display_truth_table, generate_labels, plot_graph, output_truth_table, \
        check_true, check_true_by_name
from experiment.decision_tree.example_generator import simple_example
from .example_generator import load_data
from .finite_cartpole import CartPoleRangeSet, default_range_set
from experiment.decision_tree.utils.ansi_colors import *


def unpack_feature_list(feature_list):
    unpacked = []
    for features in feature_list:
        for feature in features:
            unpacked += [int(feature)]
    return unpacked


def retrieve_truth_table_without_dummies(var_dict, truth_table):
    new_truth_table = []
    for k, v in sorted(var_dict.items()):
        if not k.startswith("dummy"):
            new_truth_table += [truth_table[v - 1]]
    return new_truth_table


def check_features(features, other_features):
    if len(features) != len(other_features):
        return False
    check = True
    for feature, other_feature in zip(features, other_features):
        if feature != other_feature:            
            check = False
            break
    return check


class TreeNode(object):
    def __init__(self, is_leaf, node_id, features=None, _class=None, left=None, right=None):
        self._is_leaf = is_leaf
        self._node_id = node_id
        self._features = features
        self._class = _class
        self._left = left
        self._right = right

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def node_id(self):
        return self._node_id

    @property
    def features(self):
        return self._features

    @property
    def node_class(self):
        return self._class

    @property
    def left_child(self):
        return self._left

    @property
    def right_child(self):
        return self._right

    def print(self):
        if self.is_leaf:
            print("({}:{})".format(int(self.node_id), int(self.node_class)), end='')
        else:
            print("(" + str(self.node_id) + ':', end='')
            print(self.features, end='')
            print(" l: ", end='')
            self.left_child.print()
            print(" r: ", end='')
            self.right_child.print()
            print(")", end='')
    
    def compare(self, othernode):
        if self.is_leaf:
            if self.node_id == othernode.node_id and self.node_class == othernode.node_class:
                return True
            else:
                return False            
        else:
            if othernode.is_leaf:
                return False
            # if othernode.features is None or self.features is None:
            #     self.print()
            #     print("")
            #     othernode.print()
            #     print("")
            #     print(self.node_id, othernode.node_id)
            if self.node_id == othernode.node_id and check_features(self.features, othernode.features):
                return self.left_child.compare(othernode.left_child) and self.right_child.compare(othernode.right_child)
            else:
                return False


class DecisionTreeSATSolver(object):
    def __init__(self, cfg):   
        self._use_simple = cfg.solver.use_simple
        if self._use_simple:
            self._feature_action_list = simple_example
            self._n_example_steps = len(simple_example)
            self._total_example_steps = len(simple_example)  
            self._n_features = 4
            self._n_sub_features = 1
        else:            
            self._load_example(cfg.solver.example_file)        
            self._n_example_steps   = cfg.solver.n_example_steps                
            if not cfg.solver.use_custom_range:
                self._range = default_range_set            
            else:
                # self._custom_range = cfg.solver.custom_range
                raise NotImplementedError
            self._data_to_feature_list()
            self._n_features = cfg.solver.n_features
            self._n_sub_features = cfg.solver.n_sub_features
            self._test_env_params = cfg.solver.test_env_params
        self._starting_tree_size = cfg.solver.starting_tree_size
        self._max_tree_size = cfg.solver.max_tree_size
        self._model_output_file = cfg.solver.model_output_file
        self._max_output_models = cfg.solver.max_output_models
        self._solver_timeout = cfg.solver.solver_timeout

    def _load_example(self, filename):
        self._data = load_data(filename)
        self._total_example_steps = len(self._data)        
    
    def _data_to_feature_list(self):
        self._feature_action_list = []
        for data in self._data[:self._n_example_steps]:
            obs, action = data
            # print(obs, action)
            features = unpack_feature_list(self._range.convert_to_features(obs))
            self._feature_action_list += [(features, action)]
            # print(features, action)
    
    def build_decision_tree(self, cur, var_dict, truth_table, cnf):        
        if not check_true(truth_table, cur):
            # non leaf node
            features = []
            for m in range(1, self._n_features+1):
                for s in range(1, self._n_sub_features+1):
                    r = (m-1) * self._n_sub_features + s
                    if check_true_by_name("a_{}_{}".format(r, cur), truth_table, var_dict):
                        features += [(r, m, s)]
            left = None
            right = None
            for j in cnf.LR(cur):                
                if check_true_by_name("l_{}_{}".format(cur, j), truth_table, var_dict):
                    assert left is None
                    left = j
            for j in cnf.RR(cur):                
                if check_true_by_name("r_{}_{}".format(cur, j), truth_table, var_dict):
                    assert right is None
                    right = j
            return TreeNode(is_leaf=False, node_id=cur, features=features, 
                        left=self.build_decision_tree(left, var_dict, truth_table, cnf),
                        right=self.build_decision_tree(right, var_dict, truth_table, cnf)
                    )
        else:
            # leaf node
            return TreeNode(is_leaf=True, node_id=cur, _class=check_true_by_name("c_{}".format(cur), truth_table, var_dict))

    def check_class_from_tree(self, example, tree_node:TreeNode):
        if tree_node.is_leaf:
            return tree_node.node_class
        else:
            f = tree_node.features[0][0]
            if example[f - 1]:
                return self.check_class_from_tree(example, tree_node.left_child)
            else:
                return self.check_class_from_tree(example, tree_node.right_child)

    def test_example(self, root_node):
        correct = 0
        for features, action in self._feature_action_list:
            _result = int(self.check_class_from_tree(features, root_node))
            if action ==_result:
                correct += 1
        accuracy = correct / len(self._feature_action_list) * 100
        return accuracy

    def solve(self):
        # Output decision trees from the smallest to the larger ones
        # Save them in a file                

        counter = 1
        prev_edge_tables = []
        print(MAGENTA + "READ EXAMPLES - of size {}".format(self._total_example_steps))
        print(RED + "START - max tree size = {}".format(self._max_tree_size) + RESET)

        output_file = open(self._model_output_file, "w")

        for tree_size in range(self._starting_tree_size, self._max_tree_size + 1, 2):
            cnf_gen = CNFGenerator(
                tree_size,
                self._n_features * self._n_sub_features,
                self._n_example_steps,
                debug_level=CNFGenDebugLevel.SOLVER_LEVEL)
            
            s = Solver(name='g4')
            print(GREEN + "Generate - {}".format(len(self._feature_action_list)) + RESET)
            s.append_formula(cnf_gen.generate(self._feature_action_list))
            print(GREEN + "Formulas generated." + RESET)
            
            def interrupt(s):
                print(BRIGHT_RED + "Timed Out." + RESET)
                s.interrupt()

            timer = Timer(1000, interrupt, [s])
            timer.start()
            
            result = s.solve_limited(expect_interrupt=True)

            print(RED + "RESULT " + CYAN + str(tree_size) + RED + " : " + RESET)
            print(result)                    

            before_truth_table = []
            prev_nodes = []
                        
            for m in s.enum_models():                
                m_after = retrieve_truth_table_without_dummies(cnf_gen.var_map, m)
                if before_truth_table != m_after:                    
                    # print(m)
                    # display_truth_table(cnf_gen.var_map, m)
                    output_truth_table(output_file, cnf_gen.var_map, m, counter)
                    edges, g = generate_graph(tree_size, cnf_gen.var_map, m)
                    labels = generate_labels(tree_size, cnf_gen.var_map, m, self._n_features, self._n_sub_features)
                    # labels = None
                    # if edges not in prev_edge_tables:
                    #     plot_graph(g, identifier="tree_" + str(counter), labels=labels)
                    #     prev_edge_tables += [edges]
                    # plot_graph(g, identifier="tree_" + str(counter), labels=labels)
                    
                    root_node = self.build_decision_tree(1, cnf_gen.var_map, m, cnf_gen)
                    prev_node_check = True
                    for prev_node in prev_nodes:
                        if root_node.compare(prev_node):    
                            prev_node_check = False                            
                            break                    
                    if prev_node_check:
                        prev_nodes += [root_node]
                        print(RED + "MODELS {} - {}".format(counter, len(prev_nodes)) + RESET)
                        plot_graph(g, identifier="tree_" + str(counter), labels=labels)
                        root_node.print()
                        # print("-----")
                        # cnf_gen.check_tree_related_cnfs(m)
                        # cnf_gen.check_decision_tree_related_cnfs(m)
                        cnf_gen.check_example_constraints_cnfs(m, debug=False)
                        print(RED + "Accuracy: {}".format(self.test_example(root_node)) + RESET)
                        counter = counter + 1
                        before_truth_table = m_after
                        if self._max_output_models < counter:
                            print(RED + "Models all found." + RESET)
                            break
                    else:
                        print(".", end='')
            s.delete()
            print('')
            if self._max_output_models < counter:
                print(RED + "Terminating." + RESET)
                break
        print("Closing files.")
        output_file.close()


@hydra.main(config_path="config", config_name="solver")
def main(cfg):
    logging.info("Working directory : {}".format(os.getcwd()))
    solver = DecisionTreeSATSolver(cfg)    
    solver.solve()
    print("done.")


if __name__ == "__main__":
    main()
    print("done..")
    exit(0)
