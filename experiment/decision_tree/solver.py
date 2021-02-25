import hydra
import logging
import os
import glob
from pysat.solvers import Solver # , Glucose4
from pysat.formula import CNF # , WCNF
from .cnf_generator import CNFGenerator 
from experiment.decision_tree.utils.tree_visualizer import generate_graph, display_truth_table, generate_labels, plot_graph
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

class DecisionTreeSATSolver(object):
    def __init__(self, cfg):   
        self._use_simple = cfg.solver.use_simple
        if self._use_simple:
            
            self._feature_action_list = simple_example
            self._n_example_steps = len(simple_example)
            self._total_example_steps = len(simple_example)
            self._n_features        = 4
            self._n_sub_features    = 1
        else:            
            self._load_example(cfg.solver.example_file)        
            self._n_example_steps   = cfg.solver.n_example_steps                
            if not cfg.solver.use_custom_range:
                self._range = default_range_set            
            else:
                #self._custom_range      = cfg.solver.custom_range
                raise NotImplementedError
        
            self._data_to_feature_list()
            self._n_features        = cfg.solver.n_features
            self._n_sub_features    = cfg.solver.n_sub_features        
            self._test_env_params   = cfg.solver.test_env_params
        self._max_tree_size     = cfg.solver.max_tree_size


    def _load_example(self, filename):
        self._data = load_data(filename)
        self._total_example_steps = len(self._data)        
    
    def _data_to_feature_list(self):
        self._feature_action_list = []
        for data in self._data[:self._n_example_steps]:
            obs, action = data
            print(obs, action)
            features = unpack_feature_list(self._range.convert_to_features(obs))
            self._feature_action_list += [(features, action)]
            print(features, action)


    def solve(self):
        # Output decision trees from the smallest to the larger ones
        # Save them in a file                

        counter = 0
        prev_edge_tables = []
        print(RED + "START {}".format(self._max_tree_size) + RESET)
        for tree_size in range(3, self._max_tree_size + 1, 2):
            cnf_gen = CNFGenerator(
                tree_size,
                self._n_features * self._n_sub_features,
                self._n_example_steps)
            
            s = Solver(name='g4')
            print(GREEN + "Generate - {}".format(len(self._feature_action_list)) + RESET)
            s.append_formula(cnf_gen.generate(self._feature_action_list))
            

            result = s.solve()
            print(RED + "RESULT " + CYAN + str(tree_size) + RED + " : " + RESET)
            print(result)                    
                        
            for m in s.enum_models():
                print(RED + "MODELS {}".format(counter) + RESET)
                print(m)
                display_truth_table(cnf_gen.var_map, m)                
                edges, g = generate_graph(tree_size, cnf_gen.var_map, m)
                labels = generate_labels(tree_size, cnf_gen.var_map, m, self._n_features, self._n_sub_features)
                if edges not in prev_edge_tables:
                    plot_graph(g, identifier="tree_" + str(counter), labels=labels)
                    prev_edge_tables += [edges]
                counter += 1
            
            s.delete()


@hydra.main(config_path="config", config_name="solver")
def main(cfg):
    logging.info("Working directory : {}".format(os.getcwd()))
    solver = DecisionTreeSATSolver(cfg)    
    solver.solve()

if __name__ == "__main__":
    main()